# Roadmap

## Phase 0 — Scaffold (done/in progress)
- [x] Repo scaffold, ADR-0001 (backend strategy)
- [x] `tile-train-core` crate skeleton that builds
- [ ] **Spike**: tch-rs on CMP 50HX — tiny MLP loss-decrease test (ADR-0001 exit criterion)

## Phase 1 — Trainer on tch-rs (CMP 50HX)
- [ ] Training loop: dataloader → forward → loss → backward → optimizer step
- [ ] AdamW + gradient accumulation + grad clipping
- [ ] AMP: bf16 autocast (CC 7.5 supports bf16 storage; fp16 compute) + grad scaling
- [ ] Model: small GPT (embedding → transformer blocks → LM head), config-driven
- [ ] Activation checkpointing (torch's checkpoint via tch-rs or manual segment re-forward)
- [ ] Checkpoint save/load (safetensors-compatible format)
- [ ] Smoke train: tiny shakespeare or gsm8k-style task, loss curve recorded

## Phase 2 — Custom tiled CUDA ops (CubeCL, the TileLang spirit, in Rust)
- [ ] Op trait seam; benchmark harness (vs libtorch baseline)
- [ ] CubeCL spike: compile + run one kernel on CMP 50HX (CUDA target only — no Apple/wgpu targets)
- [ ] Fused RMSNorm forward+backward (CubeCL)
- [ ] FlashAttention-style tiled attention forward (sm_75) + backward
- [ ] Apex-equivalent fused utilities on Rust (no Python Apex dependency):
  fused optimizers (Phase-2 Fused AdamW, Muon NS kernel), fused norms
  (Phase-2 RMSNorm fwd+bwd), fused loss scaling / grad-clip kernels, fused
  ternary quantize for BitLinear — all expressed as CubeCL kernels with
  libtorch fallbacks, verified against the tch reference on sm_75 and sm_90
- [ ] Fused AdamW
- [ ] H100 port: sm_90 paths (wgmma/TMA where it matters), verify on rented H100

## Phase 3 — Scale-out & decentralization
- [x] **Decentralized DDP with own runtime (no NCCL):** `tile-dist` crate —
  TCP ring all-reduce (reduce-scatter + all-gather), coord/worker control
  protocol, data-parallel replicas; 2 worker processes verified on the 50HX
  (loss 6.5 → 0.00097, bit-identical replica checksums, 16/16 greedy).
  Evidence: `docs/phase3_ddp_50hx.log`
- [x] **ZeRO-1: sharded optimizer state via `ShardId` addressing** —
  `zero.rs` (sharded AdamW, axis-0 `ShardId` shards) + variable-length ring
  all-gather; unit tests prove size=1 ≡ size=3 **bitwise**; 2-worker run on
  the 50HX: replicas exactly synced, loss curve identical to full-state
  baseline, opt state 540672 / 1081344 bytes = 50.0% per rank.
  Evidence: `docs/phase3_zero1_50hx.log`
- [ ] ZeRO-2: reduce-scatter gradients instead of full all-reduce (grad
  memory ÷ N); reuses the same `ShardId` + ring primitives
- [ ] ZeRO-3: shard the *parameters* too (weight all-gather per layer before
  forward, freed after backward) — full ZeRO-3 memory profile; the restore
  item above is specified against this stage as the minimum coverage
- [ ] Muon optimizer (momentum + Newton–Schulz orthogonalization of 2D
  updates; AdamW for 1D params). Fits the ring: momentum is all-reduced in
  full, NS is deterministic → every rank computes the same orthogonalized
  update, param shards all-gather as in ZeRO-1
- [ ] BitLinear b1.58 (ternary {-1,0,+1} weights, absmean scale, STE QAT,
  fp32 master weights) — memory *and* ring-traffic win; follow-up: 1-bit
  gradient compression for the all-reduce (1-bit-SGD/PowerSGD style)
- [ ] Lorentz-metric experiment (hyperbolic distances via Minkowski form,
  e.g. for embeddings) — flagged A/B, keep only if the loss curve justifies it
- [ ] Model modernization flags (modded-nanogpt practices, each A/B-measured
  on the 50HX): RoPE, QK-norm, zero-init out projections, ReLU² MLP, logit
  softcapping, value embeddings/residual
- [ ] Multi-GPU on one node (ring over CUDA IPC/P2P instead of loopback TCP)
- [ ] Tensor parallel attention/MLP (tile-style sharding)
- [ ] Decentralized training track: K-trajectory gossip for trustless peers;
  DiLoCo-style local-step + periodic-sync candidate
- [ ] Per-peer gradient validation (norm bounds) + signed checkpoints —
  runtime policy, not docs (ADR-0002 §4)
- [ ] Robustness: read timeouts / run-level watchdog on control + ring planes
  (a hung worker currently hangs the run); listen-host as a parameter before
  leaving loopback
- [ ] Paged KV + inference interop with TileRT (shared checkpoint format)
- [ ] Multi-node launcher (optional): Kubernetes manifests for coord Service
  + workers Job with headless-DNS peer discovery — k8s may *launch* workers
  but must never be in the training topology (decentralization survives its
  disappearance; ADR-0002 §3). Interesting base: user's fork
  https://github.com/357357user357357/kubernetes
- [ ] Launcher shard/flow restore (ZeRO-3 coverage **at least**): workers may
  rejoin mid-run and recover their `ShardId`-addressed state from the latest
  signed checkpoint + a replayable step/flow journal — params, optimizer
  shards, and (at ZeRO-3) parameter shards re-gathered via the ring; a
  restarted worker must re-enter the ring without a full restart, and the
  run must survive both coord and worker restarts. Runtime policy, not docs
  (ADR-0002 §3/§4)

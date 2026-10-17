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
- [ ] Fused AdamW
- [ ] H100 port: sm_90 paths (wgmma/TMA where it matters), verify on rented H100

## Phase 3 — Scale-out & decentralization
- [ ] DDP via NCCL (tch-rs) — 2+ GPUs
- [ ] Tensor parallel attention/MLP (tile-style sharding)
- [ ] Decentralized training track: gradient/optimizer-state sync over CubeCL's multi-GPU runtime; DiLoCo-style local-step + periodic-sync candidate
- [ ] Paged KV + inference interop with TileRT (shared checkpoint format)

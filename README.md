# tile-train

Rust-native **training** runtime for LLMs — the training counterpart to
[TileRT](https://github.com/tile-ai/TileRT) (ultra-low-latency *inference*).

**Relationship to TileRT:** TileRT's public repo is Python orchestration +
*closed prebuilt* native backends (`libtilert_*.so`) — its compute engine is not
open source, so it cannot be forked. tile-train therefore re-implements the
TileRT *approach* (tile-based kernels, model–system co-design, extreme memory
efficiency) natively in Rust, and uses TileRT's MIT-licensed reference kernels
(`tilert/models/*/refs/kernel.py`) as porting specs. Same spirit, new engine,
training-first.

## Goal

Train LLMs in Rust, not only run inference. First-class support for:

- **Training loop**: forward / loss / backward / optimizer steps with autograd
- **Mixed precision**: bf16/fp16 AMP with gradient scaling
- **Memory efficiency**: activation checkpointing (gradient checkpointing), fused optimizers (AdamW), gradient accumulation
- **Ops with backward**: matmul, attention (FlashAttention-style tiling), layernorm/rmsnorm, GELU/SiLU, embeddings, cross-entropy
- **Multi-GPU**: data parallelism first, tensor/pipeline parallel later (NCCL)

## Hardware

| Target | GPU | Compute capability | Status |
|---|---|---|---|
| Now | NVIDIA CMP 50HX (Turing TU102) | 7.5 (SM 75) | primary dev target |
| Next | NVIDIA H100 | 9.0 (SM 90) | planned |

CUDA arch flags: `sm_75` now, `sm_90` later (CUDA 12.4 on the dev box supports both).

## Status

- **Phase 1 — trainer on tch-rs: DONE.** Tiny GPT (2-layer, 64-embd) trains on
  the CMP 50HX: loss 6.4 → 0.001 in 500 steps, greedy continuation reproduces
  the counting rule. Validated off-machine on vast.ai RTX 4060 Ti too.
  Run: `cargo run --release --example train_tiny_gpt`
- **Phase 2 — CubeCL spike: DONE.** One custom kernel compiled and run on the
  50HX (Turing) and cross-arch on an RTX 4060 Ti (Ada) — same kernel, PASS
  both. CubeCL stays scoped to the custom-kernel layer; tch/libtorch remains
  the training workhorse.
- **Phase 3 — decentralized training, first increment: DONE.** `tile-dist`
  crate: hand-rolled TCP ring all-reduce (reduce-scatter + all-gather), a
  coordinator/worker control protocol, and data-parallel training of identical
  replicas — 2 worker processes on the 50HX, loss 6.5 → 0.00097, replica
  checksums bit-identical, no NCCL, no parameter server (ADR-0002).
  Run: `cargo run --release --example ddp_tiny_gpt -p tile-dist -- --workers 2`
- ADRs: [0001 — backend strategy](docs/adr/0001-backend-strategy.md),
  [0002 — decentralized topology](docs/adr/0002-decentralized-training-topology.md)
- [Roadmap](docs/ROADMAP.md)

## Building

The Phase-0 spike reuses the conda `tilelang` env's PyTorch 2.5.1+cu121
(`tch 0.18` matches libtorch 2.5.x) — no 2 GB libtorch download:

```bash
export TORCH_DIR=~/miniforge3/envs/tilelang/lib/python3.12/site-packages/torch
export LIBTORCH=$TORCH_DIR LIBTORCH_BYPASS_VERSION_CHECK=1
export LIBTORCH_CXX11_ABI=0   # conda-forge torch uses the legacy C++ ABI
export LD_LIBRARY_PATH=$(find ~/miniforge3/envs/tilelang/lib/python3.12/site-packages/nvidia -name lib -type d | tr '\n' ':')$TORCH_DIR/lib:$LD_LIBRARY_PATH
cargo run --release --example spike_mlp
```

The example calls `tile_train_core::ensure_cuda_backend()` itself (dlopens
`libtorch_cuda.so`), so no `LD_PRELOAD` is needed.

## Naming

Rust crate: `tile-train-core` (library), binary targets under `crates/` as they appear.

## License

MIT (same as TileRT).

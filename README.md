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

🚧 Early scaffold. Design decisions are being recorded in [docs/adr/](docs/adr/).

- [ADR-0001 — Backend strategy: tch-rs vs burn vs raw CUDA](docs/adr/0001-backend-strategy.md) — proposed, spike in progress
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

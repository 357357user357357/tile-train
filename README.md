# tile-train

Rust-native **training** runtime for LLMs — the training counterpart to [TileRT](https://github.com/357357user357357/TileRT) (inference).

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

- [ADR-0001 — Backend strategy: tch-rs vs burn vs raw CUDA](docs/adr/0001-backend-strategy.md) — proposed, pending spike
- [Roadmap](docs/ROADMAP.md)

## Naming

Rust crate: `tile-train-core` (library), binary targets under `crates/` as they appear.

## License

TBD (will match TileRT's licensing choice).

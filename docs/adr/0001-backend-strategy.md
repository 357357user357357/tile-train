# ADR-0001: Backend strategy — tch-rs vs burn vs raw CUDA

- **Status:** proposed (pending spike)
- **Date:** 2026-10-17
- **Deciders:** nikolas, codewhale

## Context

We need a Rust repo that can **train** LLMs, not just run inference.

- Current dev GPU: **CMP 50HX** — Turing, compute capability **7.5**, 10 GB VRAM.
- Near-future GPU: **H100** — Hopper, compute capability **9.0**.
- CUDA toolchain on dev box: **12.4** (supports both sm_75 and sm_90).
- Existing ecosystem: [TileRT](https://github.com/357357user357357/TileRT) (fork of Tile-ai's inference runtime) is inference-oriented; TileOPs/TileLang are Python-side. Training must be native to this repo.

## Options

### A. tch-rs (libtorch bindings) — recommended starting point
- ✅ Full autograd, CUDA, cuDNN, NCCL out of the box — training works day one.
- ✅ Battle-tested kernels; CMP 50HX (CC 7.5) and H100 (SM 9.0) both supported by libtorch CUDA 12.x builds.
- ❌ Links C++ libtorch (~2 GB download), FFI overhead per op, not "pure Rust".

### B. burn (pure Rust deep-learning framework)
- ✅ Pure Rust, autograd included, multiple backends (wgpu, cudarc, libtorch).
- ❌ cudarc backend historically lags on tuning for older/odd GPUs; CMP 50HX (CC 7.5) is an unusual mining card — risk of missing/mis-tuned kernels.
- ❌ Large framework to fight against when we want custom tiled ops.

### C. Raw CUDA via cudarc + hand-written kernels (the TileLang spirit, in Rust)
- ✅ Maximum control for tiled kernels (flash attention, fused ops); the long-term direction.
- ❌ No autograd — we would hand-write backward for every op. Very slow to a working trainer.

## Decision

**Phase 1 (now):** `tch-rs` for tensors/autograd/training loop so real training runs on the CMP 50HX quickly. Structure the code so ops are isolated behind a trait.

**Phase 2 (later):** replace hot ops one-by-one with custom CUDA kernels (cudarc / raw PTX via `sm_75`, later `sm_90`) behind the same trait — the TileLang-style tiled ops. Start with attention.

**Rejected for now:** pure burn (GPU risk on CC 7.5), pure raw CUDA (no autograd, too slow to first training run).

## Spike (exit criterion for this ADR)

A minimal loss-decrease test on the CMP 50HX: train a tiny MLP on synthetic data for ~1000 steps with tch-rs, confirm (a) CUDA device is used, (b) loss decreases, (c) VRAM stays bounded. Record results here.

## Consequences

- Phase 1 gives us a working trainer on 50HX; H100 compatibility comes free via libtorch CUDA 12.x.
- The op-trait seam is the contract; Phase 2 must not change the training loop.

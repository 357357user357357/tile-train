# ADR-0001: Backend strategy — tch-rs vs burn vs raw CUDA

- **Status:** proposed (pending spike)
- **Date:** 2026-10-17
- **Deciders:** nikolas, codewhale

## Context

We need a Rust repo that can **train** LLMs, not just run inference.

- Current dev GPU: **CMP 50HX** — Turing, compute capability **7.5**, 10 GB VRAM.
- Near-future GPU: **H100** — Hopper, compute capability **9.0**.
- CUDA toolchain on dev box: **12.4** (supports both sm_75 and sm_90).
- Existing ecosystem: [TileRT](https://github.com/tile-ai/TileRT) (ultra-low-latency inference) — verified 2026-10-17: its public repo is Python orchestration + **closed prebuilt** `.so` backends (`libtilert_dsv32.so`, `libtilert_glm5.so`), so it cannot be forked as code. Its **MIT-licensed reference kernels** (`tilert/models/deepseek_v3_2/refs/kernel.py`) and architecture (tile-based, model co-design, MTP) serve as the design spec tile-train re-implements in Rust. Training must be native to this repo.

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

## Spike result (2026-10-17) — **ACCEPTED**

Spike run on the CMP 50HX (`examples/spike_mlp.rs`, tch 0.18.1 + conda libtorch 2.5.1+cu121):

- CUDA device used: yes (device 0 of 2)
- Loss: 0.1450 → 0.000001 over 1000 Adam steps (>10x, actually >10^4x)
- VRAM: bounded ~198 MB throughout
- **Verdict: PASS — Phase 1 backend is tch-rs.**

Build notes (conda libtorch, old C++ ABI):
- `LIBTORCH_CXX11_ABI=0` required (conda-forge torch builds use the legacy ABI; pypi wheels use the new one).
- `libtorch_cuda.so` must be loaded explicitly (`tile_train_core::ensure_cuda_backend()` dlopens it) — `--as-needed` drops it otherwise since the C API lives in `libtorch_cpu.so`.
- pip-style nvidia wheels (`site-packages/nvidia/*/lib`) must be on `LD_LIBRARY_PATH`.

## Phase-2 kernel language — CubeCL

**Decision:** use [CubeCL](https://github.com/tracel-ai/cubecl) as the kernel language for custom tiled ops (Phase 2), replacing the earlier idea of hand-written cudarc/PTX.

- Targets CUDA + HIP natively; WGPU/Metal targets exist but are **out of scope** (no Apple support needed).
- Rust-native kernel authoring with autotune — the TileLang-style "tile-first" workflow without leaving Rust.
- Multi-GPU runtime abstraction supports future decentralized/distributed training directions.
- Burn's tensor layer sits on CubeCL, so burn remains an optional higher-level layer, not a requirement.

## Consequences

- Phase 1 gives us a working trainer on 50HX; H100 compatibility comes free via libtorch CUDA 12.x.
- The op-trait seam is the contract; Phase 2 must not change the training loop.

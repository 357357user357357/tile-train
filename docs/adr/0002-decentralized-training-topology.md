# ADR-0002: Decentralized training topology (ColossalAI-inspired, Rust-native)

Status: draft — shapes Phase 2+; nothing here is implemented yet.
Date: 2026-10-17

## Context

User direction: take the training-structure ideas from
[ColossalAI](https://github.com/hpcaitech/ColossalAI) — parallelism strategy,
memory management, distributed optimizer — but build them on **tile-train's own
Rust runtime**, not on PyTorch, and with **no Apple/wgpu targets**. Decentralized
training (nodes that do not trust each other, no fixed parameter server) is a
goal; raw multi-GPU speed is secondary.

ColossalAI's useful abstractions, and our verdict:

| ColossalAI concept | Verdict for tile-train |
|---|---|
| Sharded Data Parallel (ZeRO-style param/grad/opt-state sharding) | **Adopt.** Core primitive for both multi-GPU and multi-node training. |
| Tensor parallelism (row/col-split matmuls) | Adopt later, tiled: our CubeCL kernels (Phase 2) split work tile-wise anyway; TP becomes a tiling decision, not a separate system. |
| Pipeline parallelism (1F1B schedule) | Adopt only when models stop fitting on one card. The runtime seam below keeps it possible. |
| Gemini memory manager (offload state to CPU/disk by hotness) | Adopt the *idea* (tiered state placement), implement in Rust with explicit placement policy per tensor class. |
| Booster/plugin API | Rejected as an API shape: too PyTorch-specific. tile-train uses a `ParallelPlan` + executor instead. |

## Decision

1. **Own runtime, single control plane.** A coordinator process (`tile-train
   coord`) holds the global plan (model config, parallelism strategy, peer
   list); workers (`tile-train worker`) execute shards. Plain Rust, no Python
   driver. Communication: gRPC for control, hand-rolled RDMA/TCP tensor
   transport for hot paths (Phase 3+, start with TCP).
2. **Everything is a shard.** A `ShardId` addresses (tensor, axis, range).
   The optimizer state, gradients, and parameters all use the same addressing,
   so ZeRO-style sharding and decentralized aggregation are the same mechanism.
3. **Decentralized aggregation, not a parameter server.** Each step: workers
   exchange gradient shards over a peer ring/tree (all-reduce can be
   approximated by K-trajectory gossip for the trustless case; start with
   ring all-reduce among *cooperative* peers, gossip later).
4. **Trust boundary is explicit.** Signed checkpoints and per-peer gradient
   validation (norm bounds, stochastic variance tests) are runtime policy,
   not documentation — the same principle as "guarantees live in mechanism".
5. **No Apple/wgpu.** Backends: CUDA (50HX today, H100-class via rental),
   CPU for tests. CubeCL kernels compile for these targets only.
6. **Phase ordering.** Phase 2 (CubeCL tiled kernels on one GPU) → Phase 3
   (single-node multi-GPU sharded DP with ring all-reduce) → Phase 4
   (multi-node decentralized: gossip aggregation, signed state sync).

## Consequences

- The Phase-1 `Op` seam and `train.rs` already isolate the parts that will
  become sharded: params live in a `VarStore` addressable by name; gradients
  are plain tensors.
- Adding sharding must not require rewriting the training loop: the loop
  consumes a `ShardSchedule` the executor builds from the `ParallelPlan`.
- A tiny-GPT H100 verification run on rented hardware (vast.ai) doubles as
  the Phase-3 dry run target.

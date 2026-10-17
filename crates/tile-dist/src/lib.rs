//! tile-dist — decentralized training topology for tile-train (ADR-0002).
//!
//! ColossalAI's *structure*, tile-train's *own Rust runtime*: no PyTorch
//! driver, no NCCL, no parameter server. This crate is the Phase-3 increment:
//!
//! - `ring` — hand-rolled TCP ring all-reduce (reduce-scatter + all-gather),
//!   the primitive every later sharding scheme (ZeRO-style state sharding,
//!   gossip aggregation) builds on. Everything exchanged is a `Vec<f32>`;
//!   tch tensors are packed/unpacked at the edges.
//! - `proto` / `frame` — length-prefixed framing and the control messages
//!   between a coordinator (holds the global plan) and workers (execute
//!   shards). Plain TCP; the transport seam stays replaceable.
//! - `coord` / `worker` — data-parallel training: each worker holds a full
//!   model replica (identical seed), sees a different minibatch, all-reduces
//!   gradients each step, then takes an identical optimizer step. Ring
//!   all-reduce makes the replicas stay bit-identical, which the coord
//!   verifies with a parameter checksum at the end.
//!
//! Later phases reuse the same addressing: grads, params and optimizer state
//! all become `ShardId`s (ADR-0002 §2), and the ring becomes a K-trajectory
//! gossip among untrusting peers (§3).

pub mod coord;
pub mod frame;
pub mod proto;
pub mod ring;
pub mod worker;

/// Errors surfaced to callers.
pub use anyhow::Result;

//! Control messages between coordinator and workers, and the tensor payload
//! header used on the (future) hot-path transport.

use serde::{Deserialize, Serialize};

/// Coordinator ↔ worker control messages. JSON in length-prefixed frames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlMsg {
    /// Worker → coord on connect; carries the worker's data-plane listener.
    Hello {
        rank_hint: usize,
        listen_addr: String,
    },
    /// Coord → worker: placement in the ring.
    Assign {
        rank: usize,
        size: usize,
        /// Data-plane addresses indexed by rank (the ring order).
        peer_addrs: Vec<String>,
    },
    /// Coord → worker: run training step `step`.
    Step { step: i64 },
    /// Worker → coord: local minibatch loss for step `step`.
    Loss { rank: usize, step: i64, loss: f64 },
    /// Coord → worker: training finished.
    Done,
    /// Worker → coord: checksum of all parameters (sum of squares) after the
    /// final step — coord asserts all ranks match, proving ring all-reduce
    /// kept replicas bit-identical.
    ParamCheck { rank: usize, checksum: f64 },
    /// Worker → coord: optimizer-state memory report (ZeRO-1 evidence).
    /// `opt_state_bytes` = bytes of AdamW state this rank actually holds,
    /// `opt_state_bytes_full` = bytes a full (unsharded) state would need.
    MemReport { rank: usize, opt_state_bytes: u64, opt_state_bytes_full: u64 },
}

/// Header of a raw tensor payload on the data plane (grads / params / opt
/// state). Phase 3 packs tensors into `Vec<f32>` inside control flows; this
/// header is the contract the dedicated tensor transport will carry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorHeader {
    pub name: String,
    pub dims: Vec<i64>,
    pub numel: usize,
}

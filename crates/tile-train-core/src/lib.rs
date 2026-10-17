//! tile-train-core — training-capable tensor ops and training loop for tile-train.
//!
//! Design contract (ADR-0001): ops live behind a trait seam so the Phase-1
//! tch-rs backend can later be swapped per-op for custom tiled CUDA kernels
//! (Phase 2) without touching the training loop.

/// Backend-agnostic op seam. Implementations: tch-rs (Phase 1), custom CUDA
/// kernels (Phase 2). Not yet implemented — scaffold placeholder.
pub trait Op {
    /// Stable name used in benchmarks and kernel-dispatch tables.
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Nop;
    impl Op for Nop {
        fn name(&self) -> &'static str {
            "nop"
        }
    }

    #[test]
    fn op_trait_seam_compiles() {
        assert_eq!(Nop.name(), "nop");
    }
}

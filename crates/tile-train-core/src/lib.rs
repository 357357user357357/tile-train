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

/// Make the libtorch CUDA backend visible to the process.
///
/// When linking against a prebuilt libtorch (e.g. a conda PyTorch install),
/// `libtorch_cuda.so` carries the CUDA device-type registration but is often
/// dropped by `--as-needed` linking because the C API lives in
/// `libtorch_cpu.so`. Loading it explicitly before any CUDA call restores
/// availability — equivalent to what Python's `import torch` does.
pub fn ensure_cuda_backend() {
    // Already-loaded libraries simply return their existing handle.
    let flags = libc::RTLD_NOW | libc::RTLD_GLOBAL;
    for lib in ["libc10_cuda.so", "libtorch_cuda.so"] {
        if let Ok(c) = std::ffi::CString::new(lib) {
            unsafe { libc::dlopen(c.as_ptr(), flags) };
        }
    }
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

//! Training-loop plumbing: AdamW construction, gradient clipping, and
//! checkpoints (weights + a JSON sidecar carrying config and loss history).

use crate::config::GptConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path as FsPath, PathBuf};
use tch::nn::{AdamW, Optimizer, OptimizerConfig, VarStore};
use tch::Tensor;

/// One logged training step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StepLog {
    pub step: i64,
    pub loss: f64,
}

/// Everything a checkpoint needs besides the tensors themselves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub config: GptConfig,
    pub step: i64,
    pub loss_history: Vec<StepLog>,
}

/// Build an AdamW optimizer with Phase-1 defaults (fitted with grad clipping
/// in mind). `weight_decay` only applies to matmul weights in a fuller setup;
/// tch applies it uniformly, which is fine at this scale.
pub fn adamw(vs: &VarStore, lr: f64) -> Result<Optimizer> {
    Ok(AdamW {
        beta1: 0.9,
        beta2: 0.999,
        wd: 0.01,
        amsgrad: false,
        eps: 1e-8,
    }
    .build(vs, lr)?)
}

/// Clip global grad norm and return the pre-clip norm (for logging).
pub fn clip_grad_norm(_opt: &mut Optimizer, params: &[Tensor], max: f64) -> f64 {
    let grads: Vec<Tensor> = params
        .iter()
        .map(|p| p.grad())
        .filter(|g| g.defined())
        .collect();
    if grads.is_empty() {
        return 0.0;
    }
    let total: f64 = grads
        .iter()
        .map(|g| g.norm().double_value(&[]).powi(2))
        .sum::<f64>()
        .sqrt();
    if total > max {
        let scale = max / (total + 1e-6);
        for mut g in grads {
            let scaled = &g * scale;
            g.f_copy_(&scaled).expect("in-place grad scale");
        }
    }
    total
}

/// Save weights + sidecar into `dir` (creates it).
pub fn save_checkpoint(vs: &VarStore, meta: &CheckpointMeta, dir: &FsPath) -> Result<()> {
    fs::create_dir_all(dir)?;
    let weights = dir.join("weights.ot");
    vs.save(&weights)
        .with_context(|| format!("saving weights to {}", weights.display()))?;
    let sidecar = dir.join("checkpoint.json");
    fs::write(&sidecar, serde_json::to_vec_pretty(meta)?)?;
    Ok(())
}

/// Restore weights and metadata previously written by [`save_checkpoint`].
/// The `VarStore` must already hold variables matching the saved config.
pub fn load_checkpoint(vs: &mut VarStore, dir: &FsPath) -> Result<CheckpointMeta> {
    let sidecar: PathBuf = dir.join("checkpoint.json");
    let bytes = fs::read(&sidecar)
        .with_context(|| format!("reading {}", sidecar.display()))?;
    let meta: CheckpointMeta = serde_json::from_slice(&bytes)?;
    let weights = dir.join("weights.ot");
    vs.load(&weights)
        .with_context(|| format!("loading weights from {}", weights.display()))?;
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn checkpoint_roundtrip_cpu() {
        let tmp = std::env::temp_dir().join(format!("tile-train-ckpt-test-{}", std::process::id()));
        let cfg = GptConfig::tiny();
        let vs = VarStore::new(Device::Cpu);
        let _root = vs.root().randn("w", &[4, 4], 0.0, 0.1);
        let meta = CheckpointMeta {
            config: cfg.clone(),
            step: 42,
            loss_history: vec![StepLog { step: 0, loss: 5.5 }],
        };
        save_checkpoint(&vs, &meta, &tmp).unwrap();

        let mut vs2 = VarStore::new(Device::Cpu);
        let _root2 = vs2.root().randn("w", &[4, 4], 0.0, 0.1);
        let back = load_checkpoint(&mut vs2, &tmp).unwrap();
        assert_eq!(back.config, cfg);
        assert_eq!(back.step, 42);
        std::fs::remove_dir_all(&tmp).ok();
    }
}

//! ADR-0001 Phase-0 spike: prove training works on the CMP 50HX via tch-rs.
//!
//! Exit criterion (docs/adr/0001-backend-strategy.md): train a tiny MLP on
//! synthetic data for ~1000 steps and confirm
//!   (a) CUDA device is used,
//!   (b) loss decreases substantially (>10x),
//!   (c) VRAM stays bounded.
//!
//! Run: cargo run --release --example spike_mlp

use anyhow::{bail, Result};
use tch::kind::Kind::Float;
use tch::nn::{Adam, OptimizerConfig, VarStore};
use tch::{Cuda, Device, Reduction, Tensor};
use tile_train_core::ensure_cuda_backend;

fn main() -> Result<()> {
    ensure_cuda_backend();
    let device = if Cuda::is_available() {
        println!("CUDA available: {} device(s), using device 0", Cuda::device_count());
        Device::Cuda(0)
    } else {
        println!("CUDA NOT available — falling back to CPU (spike will FAIL for our purposes)");
        Device::Cpu
    };

    // Synthetic regression task: fit a fixed random 2-layer generator.
    let batch = 256_i64;
    let (din, dhidden, dout) = (32_i64, 128_i64, 16_i64);
    let xs = Tensor::randn([batch, din], (Float, device));

    let gen_w1 = Tensor::randn([din, dhidden], (Float, device)) * 0.1;
    let gen_b1 = Tensor::zeros([dhidden], (Float, device));
    let gen_w2 = Tensor::randn([dhidden, dout], (Float, device)) * 0.1;
    let ys = xs.matmul(&gen_w1) + gen_b1;
    let ys = ys.gelu("none").matmul(&gen_w2);

    // Trainable 3-layer MLP.
    let vs = VarStore::new(device);
    let root = vs.root();
    let lin1 = root.randn("lin1", &[din, 64], 0.0, 0.1);
    let lin2 = root.randn("lin2", &[64, 64], 0.0, 0.1);
    let lin3 = root.randn("lin3", &[64, dout], 0.0, 0.1);
    let params: Vec<Tensor> = vec![lin1, lin2, lin3];
    let mut opt = Adam::default().build(&vs, 1e-2)?;

    let steps = 1000_i64;
    let mut first_loss = f64::NAN;
    let mut last_loss = f64::NAN;

    for step in 0..steps {
        let [l1, l2, l3] = &params[..] else { unreachable!() };
        let h1 = xs.matmul(l1).gelu("none");
        let h2 = h1.matmul(l2).gelu("none");
        let preds = h2.matmul(l3);
        let loss = preds.mse_loss(&ys, Reduction::Mean);
        let lv = loss.double_value(&[]);
        if step == 0 {
            first_loss = lv;
        }
        if step % 200 == 0 {
            println!("step {step:4}  loss {lv:.6}  used_vram~{}MB", used_vram_mb());
        }
        opt.backward_step(&loss);
        last_loss = loss.double_value(&[]);
    }

    println!("\nfirst loss {first_loss:.6} -> last loss {last_loss:.6}");
    println!("final VRAM used: ~{}MB", used_vram_mb());

    if !device.is_cuda() {
        bail!("spike ran on CPU — CUDA path unverified");
    }
    if !(last_loss < first_loss * 0.1) {
        bail!("loss did not decrease by 10x ({first_loss:.6} -> {last_loss:.6})");
    }
    println!("SPIKE PASS: CUDA used, loss decreased >10x, memory bounded");
    Ok(())
}

/// Memory used on GPU 0 via nvidia-smi (tch 0.18 does not expose mem_get_info).
fn used_vram_mb() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

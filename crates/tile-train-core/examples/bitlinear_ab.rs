//! BitLinear b1.58 A/B: identical tiny GPT, identical data/seed/hyperparams —
//! one arm with fp32 projections, one with ternary (STE QAT) projections.
//!
//! Question answered on real hardware: does the 1.58-bit model still learn
//! the counting task end-to-end, and what is the loss-curve cost of the
//! ternarization? Embeddings and the LM head stay full precision in both
//! arms (BitNet practice), so only attention qkv/out and MLP fc/proj differ.
//!
//! Run: cargo run --release --example bitlinear_ab
//! Outputs: out/bitlinear_ab.csv (step,loss_fp32,loss_bit)

use anyhow::{bail, Result};
use tile_train_core::config::GptConfig;
use tile_train_core::ensure_cuda_backend;
use tile_train_core::model::TinyGpt;
use tile_train_core::train::adamw;
use tch::kind::Kind::Int64;
use tch::nn::VarStore;
use tch::{Cuda, Device, Tensor};

use std::fs::File;
use std::io::Write;

struct ArmResult {
    first_loss: f64,
    last_loss: f64,
    greedy_hits: usize,
    history: Vec<(i64, f64)>,
}

fn main() -> Result<()> {
    ensure_cuda_backend();
    let device = if Cuda::is_available() {
        println!("CUDA available — training on device 0");
        Device::Cuda(0)
    } else {
        println!("CUDA NOT available — running on CPU (slower, still correct)");
        Device::Cpu
    };

    let (steps, batch, lr, grad_clip) = (500_i64, 64_i64, 3e-3, 1.0);

    let fp32_arm = train_arm(&GptConfig::tiny(), steps, batch, lr, grad_clip, device)?;
    let bit_arm = train_arm(&GptConfig::tiny_bit(), steps, batch, lr, grad_clip, device)?;

    println!("\n=== A/B summary ({steps} steps, batch {batch}, lr {lr}, seed 42) ===");
    println!("fp32  projections: {:.6} -> {:.6}", fp32_arm.first_loss, fp32_arm.last_loss);
    println!("ternary projections: {:.6} -> {:.6}", bit_arm.first_loss, bit_arm.last_loss);
    println!(
        "greedy continuation: fp32 {}/16, ternary {}/16",
        fp32_arm.greedy_hits, bit_arm.greedy_hits
    );

    std::fs::create_dir_all("out")?;
    let mut csv = File::create("out/bitlinear_ab.csv")?;
    writeln!(csv, "step,loss_fp32,loss_bit")?;
    for ((s_fp, l_fp), (s_bit, l_bit)) in fp32_arm.history.iter().zip(&bit_arm.history) {
        debug_assert_eq!(s_fp, s_bit);
        writeln!(csv, "{s_fp},{l_fp:.6},{l_bit:.6}")?;
    }
    println!("wrote out/bitlinear_ab.csv");

    // Gates: both arms must actually learn; the ternary arm is allowed to be
    // somewhat worse, but not by an order of magnitude at this scale.
    if !(fp32_arm.last_loss < fp32_arm.first_loss * 0.5) {
        bail!("fp32 arm did not learn ({:.6} -> {:.6})", fp32_arm.first_loss, fp32_arm.last_loss);
    }
    if !(bit_arm.last_loss < bit_arm.first_loss * 0.5) {
        bail!("ternary arm did not learn ({:.6} -> {:.6})", bit_arm.first_loss, bit_arm.last_loss);
    }
    if fp32_arm.greedy_hits < 12 {
        bail!("fp32 arm greedy only {}/16", fp32_arm.greedy_hits);
    }
    if bit_arm.greedy_hits < 12 {
        bail!("ternary arm greedy only {}/16", bit_arm.greedy_hits);
    }
    println!("BITLINEAR-AB PASS: both arms learned the counting rule on {}", if device.is_cuda() { "CUDA" } else { "CPU" });
    Ok(())
}

/// One training arm: fresh VarStore/model, seed 42, identical batches.
fn train_arm(
    cfg: &GptConfig,
    steps: i64,
    batch: i64,
    lr: f64,
    grad_clip: f64,
    device: Device,
) -> Result<ArmResult> {
    tch::manual_seed(42);
    let vs = VarStore::new(device);
    let model = TinyGpt::new(&vs.root(), cfg);
    let mut opt = adamw(&vs, lr)?;

    // Same seed after init => both arms see identical batches.
    tch::manual_seed(42);
    let mut history = Vec::new();
    let (mut first_loss, mut last_loss) = (f64::NAN, f64::NAN);
    for step in 0..steps {
        let (xs, targets) = counting_batch(cfg, batch, device);
        let loss = model.forward_loss(&xs, &targets);
        let lv = loss.double_value(&[]);
        if step == 0 {
            first_loss = lv;
        }
        if step % 100 == 0 {
            println!("[{}] step {step:4}  loss {lv:.6}", if cfg.bit_linear { "bit " } else { "fp32" });
        }
        history.push((step, lv));
        opt.backward_step_clip(&loss, grad_clip);
        last_loss = loss.double_value(&[]);
    }

    let greedy_hits = greedy_hits(&model, cfg, device);
    Ok(ArmResult { first_loss, last_loss, greedy_hits, history })
}

/// One batch of modular-counting sequences (same task as train_tiny_gpt).
fn counting_batch(cfg: &GptConfig, batch: i64, device: Device) -> (Tensor, Tensor) {
    let starts = Tensor::randint(cfg.vocab_size, [batch, 1], (Int64, device));
    let offs = Tensor::arange(cfg.block_size + 1, (Int64, device)).view([1, -1]);
    let seq = (starts + offs).remainder(cfg.vocab_size);
    let xs = seq.narrow(1, 0, cfg.block_size);
    let targets = seq.narrow(1, 1, cfg.block_size);
    (xs, targets)
}

/// Greedy continuation from a fixed window; counts matches to the +1 rule.
fn greedy_hits(model: &TinyGpt, cfg: &GptConfig, device: Device) -> usize {
    let n_gen = 16_i64;
    let start = 5_i64;
    let mut tokens: Vec<i64> = (0..cfg.block_size).map(|i| (start + i) % cfg.vocab_size).collect();
    let expected: Vec<i64> = (cfg.block_size..cfg.block_size + n_gen)
        .map(|i| (start + i) % cfg.vocab_size)
        .collect();
    let mut got = Vec::with_capacity(n_gen as usize);
    for _ in 0..n_gen {
        let window: Vec<i64> = tokens[tokens.len() - cfg.block_size as usize..].to_vec();
        let input = Tensor::from_slice(&window).view([1, -1]).to_device(device);
        let logits = model.logits(&input);
        let (_, t_last, _) = logits.size3().unwrap();
        let next = logits
            .narrow(1, t_last - 1, 1)
            .argmax(-1, false)
            .int64_value(&[0]);
        got.push(next);
        tokens.push(next);
    }
    got.iter().zip(&expected).filter(|(a, b)| a == b).count()
}

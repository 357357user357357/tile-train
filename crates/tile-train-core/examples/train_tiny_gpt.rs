//! Phase-1 smoke trainer: tiny GPT learns modular counting on the 50HX.
//!
//! Task: given a random start token s, the sequence is `s, s+1, ... (mod V)`.
//! A model that learns the `+1 mod V` transition reaches near-zero loss and
//! reproduces the sequence greedily — a real end-to-end training signal
//! through embeddings, attention and the LM head.
//!
//! Run: cargo run --release --example train_tiny_gpt
//! Outputs: out/loss.csv, out/tiny-gpt/ (checkpoint + JSON sidecar).

use anyhow::{bail, Result};
use tch::kind::Kind::Int64;
use tch::nn::VarStore;
use tch::{Cuda, Device, Tensor};
use tile_train_core::config::GptConfig;
use tile_train_core::model::TinyGpt;
use tile_train_core::train::{adamw, save_checkpoint, CheckpointMeta, StepLog};
use tile_train_core::ensure_cuda_backend;

use std::fs::File;
use std::io::Write;

fn main() -> Result<()> {
    ensure_cuda_backend();
    let device = if Cuda::is_available() {
        println!("CUDA available ({} device(s)) — training on device 0", Cuda::device_count());
        Device::Cuda(0)
    } else {
        println!("CUDA NOT available — running on CPU (slower, still correct)");
        Device::Cpu
    };

    let cfg = GptConfig::tiny();
    let (steps, batch, lr, grad_clip, log_every) = (500_i64, 64_i64, 3e-3, 1.0, 50_i64);

    let vs = VarStore::new(device);
    let model = TinyGpt::new(&vs.root(), &cfg);
    println!(
        "model: {} layers, {} embd, {} heads, block {} — {} tensors",
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.block_size,
        vs.variables().len()
    );

    let mut opt = adamw(&vs, lr)?;

    let mut history: Vec<StepLog> = Vec::new();
    let mut first_loss = f64::NAN;
    let mut last_loss = f64::NAN;

    for step in 0..steps {
        let (xs, targets) = counting_batch(&cfg, batch, device);
        let loss = model.forward_loss(&xs, &targets);
        let lv = loss.double_value(&[]);
        if step == 0 {
            first_loss = lv;
        }
        if step % log_every == 0 {
            println!(
                "step {step:4}  loss {lv:.6}  used_vram~{}MB",
                used_vram_mb()
            );
        }
        history.push(StepLog { step, loss: lv });
        // backward + grad-norm clip + AdamW step.
        opt.backward_step_clip(&loss, grad_clip);
        last_loss = loss.double_value(&[]);
    }

    // Persist the curve and a checkpoint.
    std::fs::create_dir_all("out")?;
    let mut csv = File::create("out/loss.csv")?;
    writeln!(csv, "step,loss")?;
    for h in &history {
        writeln!(csv, "{},{}", h.step, h.loss)?;
    }
    let meta = CheckpointMeta {
        config: cfg.clone(),
        step: steps,
        loss_history: history,
    };
    save_checkpoint(&vs, &meta, std::path::Path::new("out/tiny-gpt"))?;
    println!("\nwrote out/loss.csv and out/tiny-gpt/ (weights.ot + checkpoint.json)");

    // Greedy continuation check: start at 5, the truth is 6, 7, 8, ... (mod V).
    let n_gen = 16_i64;
    let mut tokens: Vec<i64> = (0..cfg.block_size).map(|i| (5 + i) % cfg.vocab_size).collect();
    let expected: Vec<i64> = (cfg.block_size..cfg.block_size + n_gen)
        .map(|i| (5 + i) % cfg.vocab_size)
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
    let hits = got.iter().zip(&expected).filter(|(a, b)| a == b).count();
    println!("greedy continuation: expected {expected:?}");
    println!("greedy continuation: got      {got:?}  ({hits}/{n_gen} correct)");

    println!("\nfirst loss {first_loss:.6} -> last loss {last_loss:.6}");

    if !device.is_cuda() {
        bail!("training ran on CPU — CUDA path unverified");
    }
    if !(last_loss < first_loss * 0.5) {
        bail!("loss did not decrease by 2x ({first_loss:.6} -> {last_loss:.6})");
    }
    if (hits as f64) < 0.75 * n_gen as f64 {
        bail!("greedy continuation only {hits}/{n_gen} correct");
    }
    println!("PHASE-1 PASS: CUDA used, loss decreased, model reproduces the counting rule");
    Ok(())
}

/// One batch of modular-counting sequences: input `[B, T]`, target `[B, T]`
/// shifted by one. `(start + arange) % V`.
fn counting_batch(cfg: &GptConfig, batch: i64, device: Device) -> (Tensor, Tensor) {
    let starts = Tensor::randint(cfg.vocab_size, [batch, 1], (Int64, device));
    let offs = Tensor::arange(cfg.block_size + 1, (Int64, device)).view([1, -1]);
    let seq = (starts + offs).remainder(cfg.vocab_size);
    let xs = seq.narrow(1, 0, cfg.block_size);
    let targets = seq.narrow(1, 1, cfg.block_size);
    (xs, targets)
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

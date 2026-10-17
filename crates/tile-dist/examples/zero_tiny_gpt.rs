//! ZeRO-1 smoke test: like `ddp_tiny_gpt`, but the AdamW optimizer state is
//! sharded across the ring (`ShardId` addressing, ADR-0002 §2) and updated
//! parameter shards are all-gathered each step.
//!
//! PASS = every rank's param checksum matches exactly (replicas stayed in
//! sync through the sharded path) and the loss curve matches the 1-worker
//! baseline bitwise-tight — proof that sharding did not change the math.
//!
//! Run (baseline first, then sharded):
//!   cargo run --release --example zero_tiny_gpt -- --workers 1 --steps 500 \
//!       --save-baseline /tmp/zero1_baseline.json
//!   cargo run --release --example zero_tiny_gpt -- --workers 2 --steps 500 \
//!       --baseline /tmp/zero1_baseline.json

use anyhow::{bail, Context, Result};
use std::io::Write;
use std::net::TcpListener;
use std::process::Command;
use tile_dist::coord::{run_coord, CoordPlan};
use tile_dist::worker::{run_worker, Mode, WorkerConfig};
use tile_train_core::config::GptConfig;

fn main() -> Result<()> {
    // Subprocess role?
    if let Ok(coord) = std::env::var("TILE_DIST_COORD") {
        let device = pick_device();
        let cfg = WorkerConfig {
            coord,
            model: GptConfig::tiny(),
            device,
            batch: 64,
            lr: 3e-3,
            grad_clip: 1.0,
            seed: 42,
            mode: Mode::Zero1,
        };
        let summary = run_worker(&cfg)?;
        println!("[worker {}] summary: {summary:?}", summary.rank);
        return Ok(());
    }

    // Coordinator role.
    let workers = arg_usize("--workers", 2)?;
    let steps = arg_i64("--steps", 500)?;
    let port = arg_u16("--port", 9124)?;
    let log_every = 50;
    let save_baseline = arg_str("--save-baseline");
    let baseline = arg_str("--baseline");
    if workers == 0 {
        bail!("--workers must be >= 1");
    }

    let listener = TcpListener::bind(("127.0.0.1", port))
        .with_context(|| format!("binding coord port {port}"))?;
    println!(
        "[coord] listening on 127.0.0.1:{port}, expecting {workers} workers, {steps} steps (ZeRO-1)"
    );

    // Spawn worker subprocesses of this binary.
    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for _ in 0..workers {
        let child = Command::new(&exe)
            .env("TILE_DIST_COORD", format!("127.0.0.1:{port}"))
            .spawn()
            .context("spawning worker subprocess")?;
        children.push(child);
    }

    let plan = CoordPlan { size: workers, steps, log_every };
    let summary = run_coord(plan, listener)?;
    let first = summary.mean_losses[0];
    let last = *summary.mean_losses.last().unwrap();

    println!("\n[coord] mean loss {first:.6} -> {last:.6} over {steps} steps");
    println!("[coord] checksums: {:?}", summary.checksums);
    for (rank, (owned, full)) in summary.mem.iter().enumerate() {
        println!("[coord] rank {rank} opt state: {owned} / {full} bytes");
    }

    // Reap workers.
    let mut worker_fails = 0;
    for (i, mut c) in children.into_iter().enumerate() {
        let status = c.wait().with_context(|| format!("waiting worker {i}"))?;
        if !status.success() {
            eprintln!("[coord] worker {i} exited with {status}");
            worker_fails += 1;
        }
    }
    if worker_fails > 0 {
        bail!("{worker_fails} worker(s) failed");
    }
    if !summary.replicas_synced {
        bail!("replica checksums diverged — sharded optimizer all-gather is broken");
    }
    if !(last < first * 0.5) {
        bail!("loss did not decrease 2x ({first:.6} -> {last:.6})");
    }

    // Baseline round-trip: loss curves are only comparable between runs
    // with the SAME worker count (the coord reports the *mean* over per-rank
    // minibatch losses, so the effective batch differs with worker count).
    if let Some(path) = &baseline {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading baseline {path}"))?;
        let saved: serde_json::Value = serde_json::from_str(&raw)?;
        let saved_workers = saved["workers"].as_u64().unwrap_or(0);
        anyhow::ensure!(
            saved_workers == workers as u64,
            "baseline was saved with {saved_workers} worker(s), this run has {workers} — \
             loss curves are not comparable across worker counts (mean over per-rank \
             minibatches); rerun the baseline with --workers {workers}"
        );
        let losses: Vec<f64> = saved["losses"]
            .as_array()
            .with_context(|| "baseline JSON has no 'losses' array")?
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        anyhow::ensure!(
            losses.len() == summary.mean_losses.len(),
            "baseline has {} losses, run has {}",
            losses.len(),
            summary.mean_losses.len()
        );
        for (i, (a, b)) in losses.iter().zip(&summary.mean_losses).enumerate() {
            let scale = a.abs().max(b.abs()).max(1e-12);
            anyhow::ensure!(
                (a - b).abs() / scale < 1e-4,
                "loss diverged from baseline at step {i}: baseline {a:.6}, got {b:.6} — \
                 sharding changed the math"
            );
        }
        println!("[coord] baseline match: all {} losses within 1e-4 relative", losses.len());
    }
    if let Some(path) = &save_baseline {
        let json = serde_json::json!({
            "workers": workers,
            "steps": steps,
            "losses": summary.mean_losses,
            "checksum": summary.checksums[0],
        });
        let mut f = std::fs::File::create(path)
            .with_context(|| format!("creating baseline file {path}"))?;
        f.write_all(json.to_string().as_bytes())?;
        println!("[coord] baseline saved to {path}");
    }

    println!(
        "PHASE-3.2 PASS (ZeRO-1): {workers} worker(s) trained with ShardId-addressed \
         optimizer state; replicas synced exactly{}",
        if baseline.is_some() { ", loss curve matches full-state baseline" } else { "" }
    );
    Ok(())
}

fn pick_device() -> tch::Device {
    // Load CUDA backend libs first — is_available() is false until
    // libtorch_cuda.so is visible (see ensure_cuda_backend docs).
    tile_train_core::ensure_cuda_backend();
    if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
}

fn arg_str(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let pos = args.iter().position(|a| a == name)?;
    args.get(pos + 1).cloned()
}

fn arg_u16(name: &str, default: u16) -> Result<u16> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == name) {
        let v = args
            .get(pos + 1)
            .context(format!("missing value for {name}"))?
            .parse()
            .context(format!("parsing {name} as port (0-65535)"))?;
        return Ok(v);
    }
    Ok(default)
}

fn arg_usize(name: &str, default: usize) -> Result<usize> {
    arg_i64(name, default as i64).map(|v| v as usize)
}

fn arg_i64(name: &str, default: i64) -> Result<i64> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == name) {
        let v = args
            .get(pos + 1)
            .context(format!("missing value for {name}"))?
            .parse()
            .context(format!("parsing {name}"))?;
        return Ok(v);
    }
    Ok(default)
}

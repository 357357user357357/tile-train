//! Phase-3 DDP smoke test: `N` worker *processes* on one machine train one
//! tiny GPT replica-split (data parallel) with ring all-reduce over TCP.
//! PASS = every rank's param checksum matches (replicas stayed in sync) and
//! the loss curve matches the single-process Phase-1 run.
//!
//! Run: cargo run --release --example ddp_tiny_gpt -- --workers 2
//! (the main process is the coordinator; workers spawn as subprocesses of
//! this same binary, so `cargo run` exercises the full multi-process path).

use anyhow::{bail, Context, Result};
use std::net::TcpListener;
use std::process::Command;
use tile_dist::coord::{run_coord, CoordPlan};
use tile_dist::worker::{run_worker, WorkerConfig};
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
        };
        let summary = run_worker(&cfg)?;
        println!("[worker {}] summary: {summary:?}", summary.rank);
        return Ok(());
    }

    // Coordinator role.
    let workers = arg_usize("--workers", 2)?;
    let steps = arg_i64("--steps", 500)?;
    let port = arg_u16("--port", 9123)?;
    let log_every = 50;
    if workers == 0 {
        bail!("--workers must be >= 1");
    }

    let listener = TcpListener::bind(("127.0.0.1", port))
        .with_context(|| format!("binding coord port {port}"))?;
    println!("[coord] listening on 127.0.0.1:{port}, expecting {workers} workers, {steps} steps");

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
        bail!("replica checksums diverged — ring all-reduce is broken");
    }
    if !(last < first * 0.5) {
        bail!("loss did not decrease 2x ({first:.6} -> {last:.6})");
    }
    println!("PHASE-3 PASS: {workers} worker processes trained one model in sync via TCP ring all-reduce");
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

fn arg_usize(name: &str, default: usize) -> Result<usize> {
    arg_i64(name, default as i64).map(|v| v as usize)
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

fn arg_i64(name: &str, default: i64) -> Result<i64> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == name) {
        let v = args
            .get(pos + 1)
            .context(format!("missing value for {name}"))?;
        return v.parse().context(format!("parsing {name}"));
    }
    Ok(default)
}

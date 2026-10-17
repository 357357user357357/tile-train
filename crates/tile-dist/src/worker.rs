//! Worker: holds a full model replica (identical seed → identical init),
//! trains on its own minibatch stream, all-reduces gradients over the ring
//! each step, then takes the identical optimizer step as every other rank.
//! Ring all-reduce guarantees every rank applies the same averaged gradient,
//! so replicas converge bit-for-bit — verified by the coord's checksum.

use crate::frame::{recv_msg, send_msg};
use crate::proto::ControlMsg;
use crate::ring::Ring;
use crate::zero::{AdamWCfg, ShardedAdamW};
use anyhow::{bail, Context, Result};
use std::net::TcpStream;
use tch::kind::Kind::Int64;
use tch::nn::VarStore;
use tch::{Device, Tensor};
use tile_train_core::config::GptConfig;
use tile_train_core::model::TinyGpt;
use tile_train_core::train::adamw;
use tile_train_core::ensure_cuda_backend;

/// Parallelism mode for this run (ADR-0002 §2 increments).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Plain DDP: full optimizer state on every rank (Phase 3).
    Ddp,
    /// ZeRO-1: AdamW state sharded across the ring, updated params
    /// all-gathered each step.
    Zero1,
}

/// Worker settings.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Coordinator control-plane address, e.g. "127.0.0.1:9123".
    pub coord: String,
    pub model: GptConfig,
    pub device: Device,
    pub batch: i64,
    pub lr: f64,
    pub grad_clip: f64,
    /// Same seed on every rank → identical replicas. Batch streams diverge
    /// per rank (reseeded rank-dependently after init) so every rank sees a
    /// different minibatch and the all-reduce average does real work.
    pub seed: i64,
    pub mode: Mode,
}

/// Worker-side result of a run.
#[derive(Debug, Clone, Copy)]
pub struct WorkerSummary {
    pub rank: usize,
    pub last_loss: f64,
    pub checksum: f64,
}

/// Run one worker to completion (connect → train → ParamCheck).
pub fn run_worker(cfg: &WorkerConfig) -> Result<WorkerSummary> {
    if cfg.device.is_cuda() {
        ensure_cuda_backend();
    }

    // Data-plane listener must exist before Hello so the coord can hand out
    // a reachable address to the ring neighbors.
    let (ring_listener, listen_addr) = Ring::bind().context("worker: binding ring listener")?;

    // ---- control-plane connect + assignment --------------------------------
    let mut coord_stream = TcpStream::connect(&cfg.coord)
        .with_context(|| format!("worker: connecting to coord at {}", cfg.coord))?;
    send_msg(&mut coord_stream, &ControlMsg::Hello { rank_hint: 0, listen_addr })?;
    let assign = match recv_msg(&mut coord_stream)? {
        ControlMsg::Assign { rank, size, peer_addrs } => (rank, size, peer_addrs),
        other => bail!("worker: expected Assign, got {other:?}"),
    };
    let (rank, size, peer_addrs) = assign;
    println!(
        "[worker] rank {rank}/{size} starting, device={:?}, mode={:?}",
        cfg.device, cfg.mode
    );

    // ---- model replica ------------------------------------------------------
    tch::manual_seed(cfg.seed);
    let vs = VarStore::new(cfg.device);
    let model = TinyGpt::new(&vs.root(), &cfg.model);

    // Deterministic parameter order (sorted by name) so every rank packs
    // gradients into byte-identical buffers.
    let mut params: Vec<(String, Tensor)> = vs.variables().into_iter().collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let sizes: Vec<usize> = params
        .iter()
        .map(|(_, t)| t.numel() as usize)
        .collect();
    let total: usize = sizes.iter().sum();
    let mut grad_buf: Vec<f32> = vec![0.0; total];
    let param_tensors: Vec<Tensor> = params.iter().map(|(_, t)| t.shallow_clone()).collect();

    // Optimizer: full AdamW (DDP) or ZeRO-1 sharded AdamW. Only one exists
    // per mode; the unused side stays None.
    let names_dims: Vec<(String, Vec<i64>)> =
        params.iter().map(|(n, t)| (n.clone(), t.size())).collect();
    let mut sharded_opt = match cfg.mode {
        Mode::Ddp => None,
        Mode::Zero1 => Some(ShardedAdamW::new(
            &names_dims,
            rank,
            size,
            AdamWCfg { lr: cfg.lr, ..AdamWCfg::default() },
            cfg.device,
        )?),
    };
    let mut opt = match cfg.mode {
        Mode::Ddp => Some(adamw(&vs, cfg.lr)?),
        Mode::Zero1 => None,
    };

    // ---- ring ---------------------------------------------------------------
    let mut ring = if size > 1 {
        Some(Ring::connect(rank, size, ring_listener, &peer_addrs)?)
    } else {
        None
    };

    // Diverge the batch RNG per rank *after* model init: ranks need identical
    // init (same seed → identical replicas) but *different* minibatches —
    // identical batches would make the gradient average a no-op (found by
    // Phase-3 review).
    tch::manual_seed(cfg.seed.wrapping_mul(1_000_003).wrapping_add(rank as i64));

    // ---- training loop ------------------------------------------------------
    let mut last_loss = f64::NAN;
    loop {
        match recv_msg(&mut coord_stream).context("worker: reading next control msg")? {
            ControlMsg::Step { step } => {
                let (xs, targets) = counting_batch(&cfg.model, cfg.batch, cfg.device);
                let loss = model.forward_loss(&xs, &targets);
                match cfg.mode {
                    Mode::Ddp => {
                        let opt = opt.as_mut().expect("DDP mode has a tch optimizer");
                        opt.zero_grad();
                        loss.backward();

                        // Pack local grads (missing grads stay zero), all-reduce avg,
                        // write back as the gradient the optimizer will apply.
                        pack_grads(&params, &sizes, &mut grad_buf)?;
                        if let Some(r) = ring.as_mut() {
                            r.all_reduce_avg(&mut grad_buf)?;
                        }
                        set_grads(&params, &sizes, &grad_buf, cfg.device)?;
                        tile_train_core::train::clip_grad_norm(opt, &param_tensors, cfg.grad_clip);
                        opt.step();
                    }
                    Mode::Zero1 => {
                        // No tch optimizer here: clear stale grads by hand so
                        // backward does not accumulate across steps.
                        for (_, p) in &params {
                            let mut g = p.grad();
                            if g.defined() {
                                let _ = g.zero_();
                            }
                        }
                        loss.backward();

                        // ZeRO-1 needs every gradient — a missing one would
                        // silently become a zero shard update and break the
                        // all-gather equivalence.
                        pack_grads_strict(&params, &sizes, &mut grad_buf)?;
                        if let Some(r) = ring.as_mut() {
                            r.all_reduce_avg(&mut grad_buf)?;
                            // Clip at the same point DDP does: after
                            // averaging, before the optimizer step.
                            clip_buffer(&mut grad_buf, cfg.grad_clip);
                            let so = sharded_opt.as_mut().expect("ZeRO-1 mode has a sharded optimizer");
                            so.step(&params, &grad_buf)?;
                            // All-gather updated parameter shards so every
                            // rank holds the full post-step params again.
                            let mine = so.pack_owned_params(&params)?;
                            let chunks = r.all_gather_var(&mine)?;
                            so.write_gathered(&params, &chunks)?;
                        } else {
                            // size == 1: sharded opt degenerates to a full
                            // AdamW; no ring traffic.
                            clip_buffer(&mut grad_buf, cfg.grad_clip);
                            sharded_opt
                                .as_mut()
                                .expect("ZeRO-1 mode has a sharded optimizer")
                                .step(&params, &grad_buf)?;
                        }
                    }
                }

                last_loss = loss.double_value(&[]);
                send_msg(
                    &mut coord_stream,
                    &ControlMsg::Loss { rank, step, loss: last_loss },
                )?;
            }
            ControlMsg::Done => break,
            other => bail!("worker: unexpected control msg {other:?}"),
        }
    }

    // Rank 0 shows the model actually learned the counting rule.
    if rank == 0 {
        let (hits, n) = greedy_check(&model, &cfg.model, cfg.device);
        println!("[worker 0] greedy continuation: {hits}/{n} correct");
    }

    // Replica-sync evidence: sum of squared parameters, computed locally.
    let mut flat = vec![0.0f32; total];
    pack_params(&params, &sizes, &mut flat)?;
    let checksum: f64 = flat.iter().map(|v| (*v as f64) * (*v as f64)).sum();

    // Memory report: ZeRO-1 evidence (owned vs full optimizer state bytes).
    let (owned_b, full_b) = match &sharded_opt {
        Some(so) => (so.owned_bytes(), so.full_bytes()),
        // DDP keeps the full state on every rank.
        None => ((total as u64) * 4 * 2, (total as u64) * 4 * 2),
    };
    send_msg(
        &mut coord_stream,
        &ControlMsg::MemReport {
            rank,
            opt_state_bytes: owned_b,
            opt_state_bytes_full: full_b,
        },
    )?;
    send_msg(&mut coord_stream, &ControlMsg::ParamCheck { rank, checksum })?;
    println!("[worker] rank {rank} done, last loss {last_loss:.6}");

    Ok(WorkerSummary { rank, last_loss, checksum })
}

/// Pack gradients of all params into `buf` (params sorted by name; missing
/// grads contribute zeros so every rank produces the same layout).
fn pack_grads(params: &[(String, Tensor)], sizes: &[usize], buf: &mut [f32]) -> Result<()> {
    let mut off = 0usize;
    for ((_, p), &n) in params.iter().zip(sizes) {
        let g = p.grad();
        if g.defined() {
            let g_cpu = g.to_device(Device::Cpu).reshape([-1]);
            g_cpu.copy_data(&mut buf[off..off + n], n);
        } else {
            buf[off..off + n].fill(0.0);
        }
        off += n;
    }
    Ok(())
}

/// Pack parameters themselves (used for the replica-sync checksum).
fn pack_params(params: &[(String, Tensor)], sizes: &[usize], buf: &mut [f32]) -> Result<()> {
    let mut off = 0usize;
    for ((_, p), &n) in params.iter().zip(sizes) {
        let p_cpu = p.to_device(Device::Cpu).reshape([-1]);
        p_cpu.copy_data(&mut buf[off..off + n], n);
        off += n;
    }
    Ok(())
}

/// Pack gradients of all params into `buf`, bailing if any param lacks a
/// gradient. ZeRO-1 reads updates straight from this buffer, so a missing
/// grad would silently become a zero shard update and break the exact
/// equivalence between sharded and full runs.
fn pack_grads_strict(params: &[(String, Tensor)], sizes: &[usize], buf: &mut [f32]) -> Result<()> {
    let mut off = 0usize;
    for ((name, p), &n) in params.iter().zip(sizes) {
        let g = p.grad();
        if !g.defined() {
            bail!("worker: param '{name}' has no gradient (ZeRO-1 needs every grad)");
        }
        let g_cpu = g.to_device(Device::Cpu).reshape([-1]);
        g_cpu.copy_data(&mut buf[off..off + n], n);
        off += n;
    }
    Ok(())
}

/// Scale `buf` in place so its global L2 norm is at most `max` (no-op if
/// already below). Same semantics as `core::train::clip_grad_norm`, but on
/// the packed buffer — the sharded optimizer reads grads from here, not
/// from the variables.
fn clip_buffer(buf: &mut [f32], max: f64) {
    let norm: f64 = buf.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
    if norm > max && norm > 0.0 {
        let scale = (max / norm) as f32;
        for v in buf.iter_mut() {
            *v *= scale;
        }
    }
}

/// Write averaged gradients back onto the variables for the optimizer step.
/// Grads are mutated in place (tch 0.18 has no `set_grad`). `pack_grads`
/// zero-fills undefined grads, so a param with no grad here would silently
/// diverge from ranks that did apply the averaged update — bail instead.
fn set_grads(params: &[(String, Tensor)], sizes: &[usize], buf: &[f32], device: Device) -> Result<()> {
    let mut off = 0usize;
    for ((name, p), &n) in params.iter().zip(sizes) {
        let mut g = p.grad();
        if g.defined() {
            let averaged = Tensor::from_slice(&buf[off..off + n])
                .reshape(p.size().as_slice())
                .to_device(device);
            g.copy_(&averaged);
        } else {
            bail!(
                "worker: param '{name}' has no gradient but packing assumed one — \
                 replicas would diverge"
            );
        }
        off += n;
    }
    Ok(())
}

/// One batch of modular-counting sequences (same task as Phase-1).
pub fn counting_batch(cfg: &GptConfig, batch: i64, device: Device) -> (Tensor, Tensor) {
    let starts = Tensor::randint(cfg.vocab_size, [batch, 1], (Int64, device));
    let offs = Tensor::arange(cfg.block_size + 1, (Int64, device)).view([1, -1]);
    let seq = (starts + offs).remainder(cfg.vocab_size);
    let xs = seq.narrow(1, 0, cfg.block_size);
    let targets = seq.narrow(1, 1, cfg.block_size);
    (xs, targets)
}

/// Greedy continuation check from a start token of 5; returns (hits, n).
pub fn greedy_check(model: &TinyGpt, cfg: &GptConfig, device: Device) -> (usize, i64) {
    tch::no_grad(|| {
        let n_gen = 16i64;
        let mut tokens: Vec<i64> =
            (0..cfg.block_size).map(|i| (5 + i) % cfg.vocab_size).collect();
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
        (hits, n_gen)
    })
}

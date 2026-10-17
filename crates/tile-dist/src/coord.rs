//! Coordinator: holds the global plan (ring membership, step schedule),
//! collects losses, verifies replica sync. Plain TCP control plane —
//! ColossalAI's "single control plane" idea without the Python driver
//! (ADR-0002 §1). The coord never touches tensors; gradients flow only
//! between workers over the ring (`crate::ring`), so this is data-parallel
//! training without a parameter server (§3).

use crate::frame::{recv_msg, send_msg};
use crate::proto::ControlMsg;
use anyhow::{bail, Context, Result};
use std::net::{TcpListener, TcpStream};

/// What the coord runs: `size` workers for `steps` steps.
#[derive(Debug, Clone)]
pub struct CoordPlan {
    pub size: usize,
    pub steps: i64,
    pub log_every: i64,
}

/// Outcome of a coordinated run.
#[derive(Debug, Clone)]
pub struct TrainSummary {
    /// Mean of per-rank losses, one entry per step.
    pub mean_losses: Vec<f64>,
    /// Post-training checksum (sum of squared params) per rank.
    pub checksums: Vec<f64>,
    /// True when every rank's checksum matches to f64 tolerance — proof the
    /// ring all-reduce + identical optimizer steps kept replicas bit-close.
    pub replicas_synced: bool,
    /// Optimizer-state memory per rank: (bytes owned, bytes a full state
    /// would need). With ZeRO-1 sharding owned < full; identical on DDP.
    pub mem: Vec<(u64, u64)>,
}

/// Run the coordinator on an already-bound listener until training completes.
pub fn run_coord(plan: CoordPlan, listener: TcpListener) -> Result<TrainSummary> {
    // ---- gather workers: Hello{listen_addr} from each ----------------------
    let mut streams: Vec<TcpStream> = Vec::new();
    let mut peer_addrs: Vec<String> = Vec::new();
    for i in 0..plan.size {
        let (mut stream, _addr) = listener
            .accept()
            .with_context(|| format!("coord: accepting worker {i}"))?;
        match recv_msg(&mut stream)
            .with_context(|| format!("coord: reading Hello from worker {i}"))?
        {
            ControlMsg::Hello { listen_addr, .. } => peer_addrs.push(listen_addr),
            other => bail!("coord: expected Hello from worker {i}, got {other:?}"),
        }
        streams.push(stream);
    }

    // ---- assign ring positions --------------------------------------------
    for (rank, stream) in streams.iter_mut().enumerate() {
        send_msg(
            stream,
            &ControlMsg::Assign {
                rank,
                size: plan.size,
                peer_addrs: peer_addrs.clone(),
            },
        )
        .with_context(|| format!("coord: sending Assign to rank {rank}"))?;
    }

    // ---- step loop: broadcast Step, collect Loss from every rank -----------
    let mut mean_losses = Vec::with_capacity(plan.steps as usize);
    // Data-parallel liveness check: with per-rank minibatches the losses must
    // differ across ranks at least once, or the gradient average was a no-op.
    let mut ever_differed = plan.size <= 1;
    for step in 0..plan.steps {
        for (rank, stream) in streams.iter_mut().enumerate() {
            send_msg(stream, &ControlMsg::Step { step })
                .with_context(|| format!("coord: sending Step {step} to rank {rank}"))?;
        }
        let mut losses = Vec::with_capacity(plan.size);
        for (rank, stream) in streams.iter_mut().enumerate() {
            match recv_msg(stream)
                .with_context(|| format!("coord: reading Loss from rank {rank}, step {step}"))?
            {
                ControlMsg::Loss { loss, step: s, .. } if s == step => losses.push(loss),
                other => bail!("coord: expected Loss(step {step}) from rank {rank}, got {other:?}"),
            }
        }
        if losses.iter().any(|l| *l != losses[0]) {
            ever_differed = true;
        }
        let mean = losses.iter().sum::<f64>() / plan.size as f64;
        if step % plan.log_every == 0 {
            println!("[coord] step {step:4}  mean loss {mean:.6}");
        }
        mean_losses.push(mean);
    }

    // ---- shutdown + replica-sync verification ------------------------------
    for (rank, stream) in streams.iter_mut().enumerate() {
        send_msg(stream, &ControlMsg::Done)
            .with_context(|| format!("coord: sending Done to rank {rank}"))?;
    }
    // Memory reports (ZeRO-1 evidence), then checksums, in that order.
    let mut mem = Vec::with_capacity(plan.size);
    for (rank, stream) in streams.iter_mut().enumerate() {
        match recv_msg(stream)
            .with_context(|| format!("coord: reading MemReport from rank {rank}"))?
        {
            ControlMsg::MemReport { opt_state_bytes, opt_state_bytes_full, .. } => {
                mem.push((opt_state_bytes, opt_state_bytes_full))
            }
            other => bail!("coord: expected MemReport from rank {rank}, got {other:?}"),
        }
    }
    let mut checksums = Vec::with_capacity(plan.size);
    for (rank, stream) in streams.iter_mut().enumerate() {
        match recv_msg(stream)
            .with_context(|| format!("coord: reading ParamCheck from rank {rank}"))?
        {
            ControlMsg::ParamCheck { checksum, .. } => checksums.push(checksum),
            other => bail!("coord: expected ParamCheck from rank {rank}, got {other:?}"),
        }
    }
    let first = checksums[0];
    // f32 params are copied bit-exact through the ring, so checksums must be
    // exactly equal; any slack here could mask low-bit divergence.
    let replicas_synced = checksums.iter().all(|c| *c == first);
    println!(
        "[coord] checksums: {:?}  replicas_synced={replicas_synced}",
        checksums
    );
    if !ever_differed {
        bail!(
            "coord: ranks reported identical losses at every step — \
             per-rank minibatches are not diverging, gradient averaging is a no-op"
        );
    }
    let (owned, full): (u64, u64) =
        mem.iter().copied().fold((0, 0), |(a, b), (o, f)| (a + o, b + f));
    if full > 0 {
        println!(
            "[coord] opt state: {} bytes owned vs {} full across {} rank(s) ({:.1}% of full)",
            owned,
            full,
            plan.size,
            100.0 * owned as f64 / full as f64
        );
    }
    Ok(TrainSummary { mean_losses, checksums, replicas_synced, mem })
}

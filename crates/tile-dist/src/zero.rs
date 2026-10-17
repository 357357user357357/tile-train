//! ZeRO-1: shard the AdamW optimizer state (momentum `m`, variance `v`)
//! across the ring and all-gather updated parameter shards after each step
//! (ADR-0002 §2). Each rank owns 1/size of the optimizer state; parameters
//! and gradients are still replicated (ZeRO-2/3 remove that later).
//!
//! Correctness argument: the AdamW update is elementwise, so a shard
//! boundary cannot change the arithmetic — a `size == 1` ShardedAdamW *is* a
//! full AdamW, and a size-N run reassembled must equal it bit-for-bit. The
//! unit tests pin both facts (vs. a scalar reference and vs. full state),
//! and the `zero_tiny_gpt` example pins it end-to-end against a 1-worker
//! baseline.
//!
//! Sharding axis: axis 0 of every tensor (rows for weights, elements for
//! biases), addressed with `crate::shard::ShardId` — the same layout the
//! ring uses, so shard c belongs to rank c.

use crate::shard::{contiguous_ranges, ShardId};
use anyhow::{ensure, Result};
use tch::{Device, Kind, Tensor};

/// AdamW hyperparameters (defaults match `tile_train_core::train::adamw`).
#[derive(Debug, Clone, Copy)]
pub struct AdamWCfg {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWCfg {
    fn default() -> Self {
        Self { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 }
    }
}

/// Per-tensor sharding plan + this rank's owned optimizer state.
struct TensorShards {
    name: String,
    /// Full tensor shape.
    dims: Vec<i64>,
    /// Shape of one axis-0 shard (dims with dim 0 replaced by the shard len).
    shard_dims: Vec<i64>,
    /// Elements per axis-0 row (product of dims[1..]).
    rest: usize,
    /// Offset of this tensor's flat block in the packed gradient buffer.
    flat_off: usize,
    /// This rank's owned row range [start, end) along axis 0.
    owned: (usize, usize),
    /// Address of the moment shard this rank owns (ADR-0002 §2).
    shard_id: ShardId,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
}

/// Sharded AdamW: optimizer state split across `size` ranks.
pub struct ShardedAdamW {
    pub rank: usize,
    pub size: usize,
    cfg: AdamWCfg,
    device: Device,
    step_t: u32,
    plan: Vec<TensorShards>,
}

impl ShardedAdamW {
    /// Build the sharding plan from the (sorted-name) parameter list.
    /// `names_dims[i]` = (param name, full shape); must be in the same order
    /// on every rank (the worker sorts by name) and match `grad_flat`'s
    /// packing order in `step`.
    pub fn new(
        names_dims: &[(String, Vec<i64>)],
        rank: usize,
        size: usize,
        cfg: AdamWCfg,
        device: Device,
    ) -> Result<Self> {
        ensure!(size >= 1, "size must be >= 1");
        ensure!(rank < size, "rank {rank} out of range for size {size}");
        let mut plan = Vec::with_capacity(names_dims.len());
        let mut flat_off = 0usize;
        for (name, dims) in names_dims {
            ensure!(!dims.is_empty(), "param '{name}' has rank 0 (scalar) — not shardable yet");
            ensure!(
                dims.iter().all(|d| *d >= 0),
                "param '{name}' has a negative dimension: {dims:?}"
            );
            let d0 = dims[0].max(0) as usize;
            let rest: usize = dims[1..].iter().product::<i64>().max(0) as usize;
            let ranges = contiguous_ranges(d0, size);
            let (s, e) = ranges[rank];
            let mut shard_dims = dims.clone();
            shard_dims[0] = (e - s) as i64;
            let exp_avg = Tensor::zeros(shard_dims.as_slice(), (Kind::Float, device));
            let exp_avg_sq = Tensor::zeros(shard_dims.as_slice(), (Kind::Float, device));
            plan.push(TensorShards {
                name: name.clone(),
                dims: dims.clone(),
                shard_dims,
                rest,
                flat_off,
                owned: (s, e),
                shard_id: ShardId {
                    tensor: name.clone(),
                    axis: 0,
                    start: s as i64,
                    end: e as i64,
                },
                exp_avg,
                exp_avg_sq,
            });
            flat_off += d0 * rest;
        }
        Ok(Self { rank, size, cfg, device, step_t: 0, plan })
    }

    /// One AdamW step on the owned shards.
    ///
    /// `params` must be the same list `new` was built from (same order/names);
    /// `grad_flat` is the *averaged* gradient buffer in the same flat packing
    /// order (the worker's `grad_buf` after `ring.all_reduce_avg`). Only the
    /// rank-owned slices of `grad_flat` are read.
    pub fn step(&mut self, params: &[(String, Tensor)], grad_flat: &[f32]) -> Result<()> {
        ensure!(
            params.len() == self.plan.len(),
            "sharded opt: {} params, plan built for {}",
            params.len(),
            self.plan.len()
        );
        for ((name, _), ts) in params.iter().zip(&self.plan) {
            ensure!(
                *name == ts.name,
                "sharded opt: param order mismatch: got '{name}', plan has '{}'",
                ts.name
            );
        }
        for ((_, p), ts) in params.iter().zip(&self.plan) {
            ensure!(
                p.size() == ts.dims,
                "sharded opt: shape mismatch for '{}': tensor {:?}, plan {:?}",
                ts.name,
                p.size(),
                ts.dims
            );
        }
        self.step_t += 1;
        let t = self.step_t as f64;
        let bc1 = 1.0 - self.cfg.beta1.powf(t);
        let bc2 = 1.0 - self.cfg.beta2.powf(t);
        let cfg = self.cfg;

        // Params are leaf variables with requires_grad; in-place updates on
        // their views must run under no_grad (exactly what torch optimizers
        // do internally).
        tch::no_grad(|| {
            for (ts, (_, p)) in self.plan.iter_mut().zip(params) {
                let (s, e) = ts.owned;
                let shard_numel = (e - s) * ts.rest;
                // With size > shardable rows a rank may own an empty shard.
                if shard_numel == 0 {
                    continue;
                }
                let begin = ts.flat_off + s * ts.rest;
                let end = ts.flat_off + e * ts.rest;
                ensure!(
                    end <= grad_flat.len(),
                    "sharded opt: grad buffer too short for '{}' (need {end}, have {})",
                    ts.name,
                    grad_flat.len()
                );
                let g = Tensor::from_slice(&grad_flat[begin..end])
                    .to_device(self.device)
                    .reshape(ts.shard_dims.as_slice());

                let m = &mut ts.exp_avg;
                let v = &mut ts.exp_avg_sq;
                // m = β1 m + (1-β1) g ; v = β2 v + (1-β2) g²
                let _ = m.f_mul_scalar_(cfg.beta1)?;
                let _ = m.f_add_(&g.f_mul_scalar(1.0 - cfg.beta1)?)?;
                let g2 = g.f_mul(&g)?;
                let _ = v.f_mul_scalar_(cfg.beta2)?;
                let _ = v.f_add_(&g2.f_mul_scalar(1.0 - cfg.beta2)?)?;
                // Bias-corrected update; decoupled weight decay (AdamW).
                let m_hat = m.f_div_scalar(bc1)?;
                let v_hat = v.f_div_scalar(bc2)?;
                let denom = v_hat.f_sqrt()?.f_add_scalar(cfg.eps)?;
                let upd = m_hat.f_div(&denom)?.f_mul_scalar(-cfg.lr)?;
                let mut p_shard = p.narrow(0, s as i64, (e - s) as i64);
                if cfg.weight_decay > 0.0 {
                    let _ = p_shard.f_mul_scalar_(1.0 - cfg.lr * cfg.weight_decay)?;
                }
                let _ = p_shard.f_add_(&upd)?;
            }
            Ok(())
        })
    }

    /// Bytes of optimizer state this rank owns (m + v, f32).
    pub fn owned_bytes(&self) -> u64 {
        self.plan
            .iter()
            .map(|ts| {
                let n: i64 = ts.shard_dims.iter().product::<i64>().max(0);
                (n as u64) * 4 * 2
            })
            .sum()
    }

    /// Bytes a full (unsharded) AdamW state would need for these params.
    pub fn full_bytes(&self) -> u64 {
        self.plan
            .iter()
            .map(|ts| {
                let n: i64 = ts.dims.iter().product::<i64>().max(0);
                (n as u64) * 4 * 2
            })
            .sum()
    }

    /// The shard addresses this rank owns (proof the state really is
    /// `ShardId`-addressed; also the unit of later gossip exchange).
    pub fn owned_shard_ids(&self) -> Vec<ShardId> {
        self.plan.iter().map(|ts| ts.shard_id.clone()).collect()
    }

    /// Pack this rank's *updated parameter shards* (all tensors, plan order)
    /// into one compact f32 buffer for `Ring::all_gather_var`.
    pub fn pack_owned_params(&self, params: &[(String, Tensor)]) -> Result<Vec<f32>> {
        ensure!(params.len() == self.plan.len(), "pack_owned_params: param count mismatch");
        let mut out = Vec::new();
        for (ts, (_, p)) in self.plan.iter().zip(params) {
            let (s, e) = ts.owned;
            let n_rows = e - s;
            if n_rows == 0 {
                continue;
            }
            let shard = p
                .narrow(0, s as i64, n_rows as i64)
                .reshape([-1])
                .to_device(Device::Cpu);
            let mut buf = vec![0.0f32; shard.numel() as usize];
            let n = buf.len();
            shard.copy_data(&mut buf, n);
            out.extend_from_slice(&buf);
        }
        Ok(out)
    }

    /// Write gathered parameter shards back into `params`. `chunks[src]` is
    /// the compact buffer `pack_owned_params` produced on rank `src` (the
    /// layout is identical everywhere, so one plan decodes every chunk).
    pub fn write_gathered(&self, params: &[(String, Tensor)], chunks: &[Vec<f32>]) -> Result<()> {
        ensure!(
            chunks.len() == self.size,
            "write_gathered: got {} chunks, size {}",
            chunks.len(),
            self.size
        );
        ensure!(params.len() == self.plan.len(), "write_gathered: param count mismatch");
        // Chunks are *compact* per-rank buffers: chunk[src] holds only rank
        // src's owned rows, concatenated in plan order. `consumed[src]`
        // tracks the read cursor per chunk.
        let mut consumed = vec![0usize; self.size];
        for (ts, (_, p)) in self.plan.iter().zip(params) {
            let d0 = ts.dims[0].max(0) as usize;
            let ranges = contiguous_ranges(d0, self.size);
            for (src, chunk) in chunks.iter().enumerate() {
                let (s, e) = ranges[src];
                let n_rows = e - s;
                if n_rows == 0 {
                    continue;
                }
                let want = n_rows * ts.rest;
                ensure!(
                    consumed[src] + want <= chunk.len(),
                    "write_gathered: chunk {src} too short for '{}' (need {}, len {})",
                    ts.name,
                    consumed[src] + want,
                    chunk.len()
                );
                let data = Tensor::from_slice(&chunk[consumed[src]..consumed[src] + want])
                    .to_device(self.device);
                let dst = p.narrow(0, s as i64, n_rows as i64);
                // Same leaf-variable rule as step: param writes under no_grad.
                tch::no_grad(|| dst.view([-1]).copy_(&data));
                consumed[src] += want;
            }
        }
        // Every chunk must be fully accounted for — guards a layout mismatch
        // that would otherwise pass silently, leaving stale tail bytes.
        for (src, chunk) in chunks.iter().enumerate() {
            ensure!(
                consumed[src] == chunk.len(),
                "write_gathered: chunk {src} has {} elements, layout consumes {}",
                chunk.len(),
                consumed[src]
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names_dims() -> Vec<(String, Vec<i64>)> {
        vec![("w".to_string(), vec![6, 3]), ("b".to_string(), vec![4])]
    }

    fn make_params(device: Device, seed: f32) -> (Vec<(String, Vec<i64>)>, Vec<Tensor>) {
        let nd = names_dims();
        let params: Vec<Tensor> = nd
            .iter()
            .enumerate()
            .map(|(i, (_, dims))| {
                let numel: i64 = dims.iter().product();
                let vals: Vec<f32> =
                    (0..numel as usize).map(|j| seed + (i * 100 + j) as f32 * 0.01).collect();
                Tensor::from_slice(&vals).reshape(dims.as_slice()).to_device(device)
            })
            .collect();
        (nd, params)
    }

    fn to_cpu_flat(t: &Tensor) -> Vec<f32> {
        let flat = t.reshape([-1]).to_device(Device::Cpu);
        let mut buf = vec![0.0f32; flat.numel() as usize];
        let n = buf.len();
        flat.copy_data(&mut buf, n);
        buf
    }

    /// Straightforward scalar AdamW reference (same math, f64 accumulate,
    /// decoupled weight decay) over a flat parameter vector.
    fn reference_adamw(p0: &[f32], g: &[f32], cfg: &AdamWCfg, steps: u32) -> Vec<f64> {
        let mut p: Vec<f64> = p0.iter().map(|v| *v as f64).collect();
        let mut m = vec![0.0f64; p.len()];
        let mut v = vec![0.0f64; p.len()];
        for t_i in 1..=steps {
            let t = t_i as f64;
            let bc1 = 1.0 - cfg.beta1.powf(t);
            let bc2 = 1.0 - cfg.beta2.powf(t);
            for i in 0..p.len() {
                let gi = g[i] as f64;
                m[i] = cfg.beta1 * m[i] + (1.0 - cfg.beta1) * gi;
                v[i] = cfg.beta2 * v[i] + (1.0 - cfg.beta2) * gi * gi;
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                p[i] *= 1.0 - cfg.lr * cfg.weight_decay;
                p[i] -= cfg.lr * m_hat / (v_hat.sqrt() + cfg.eps);
            }
        }
        p
    }

    #[test]
    fn narrow_view_copy_writes_through_to_param() {
        // Guards the in-place update path: narrow(0,..) must be a view.
        let p = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).reshape([2, 2]);
        p.narrow(0, 1, 1).fill_(9.0);
        // narrow(0,1,1) is row 1 = [3,4] → both elements become 9.
        assert_eq!(to_cpu_flat(&p), vec![1.0, 2.0, 9.0, 9.0]);
    }

    #[test]
    fn sharded_adamw_matches_scalar_reference() {
        let cfg = AdamWCfg { lr: 0.05, ..AdamWCfg::default() };
        let (nd, params) = make_params(Device::Cpu, 0.5);
        let mut opt = ShardedAdamW::new(&nd, 0, 1, cfg, Device::Cpu).unwrap();
        let w0 = to_cpu_flat(&params[0]);
        let b0 = to_cpu_flat(&params[1]);
        // One fixed "averaged gradient" buffer in flat packing order.
        let grad: Vec<f32> = (0..22).map(|i| ((i as f32) * 0.37 - 3.0).sin()).collect();
        for _ in 0..3 {
            let ps: Vec<(String, Tensor)> =
                nd.iter().zip(&params).map(|((n, _), t)| (n.clone(), t.shallow_clone())).collect();
            opt.step(&ps, &grad).unwrap();
        }
        let got_w = to_cpu_flat(&params[0]);
        let got_b = to_cpu_flat(&params[1]);
        let ref_w = reference_adamw(&w0, &grad[..18], &cfg, 3);
        let ref_b = reference_adamw(&b0, &grad[18..], &cfg, 3);
        for (i, (got, want)) in got_w.iter().zip(&ref_w).enumerate() {
            let (g, w) = (*got as f64, *want);
            assert!((g - w).abs() < 1e-5, "w[{i}]: got {g}, want {w}");
        }
        for (i, (got, want)) in got_b.iter().zip(&ref_b).enumerate() {
            let (g, w) = (*got as f64, *want);
            assert!((g - w).abs() < 1e-5, "b[{i}]: got {g}, want {w}");
        }
    }

    /// N in-process "ranks" with identical params and grads; run one step on
    /// each; reassemble rank 0's params from gathered chunks.
    fn run_sharded(size: usize, steps: u32) -> (Vec<f32>, Vec<f32>, ShardedAdamW) {
        let cfg = AdamWCfg { lr: 0.05, ..AdamWCfg::default() };
        let (nd, params0) = make_params(Device::Cpu, 0.5);
        let grad: Vec<f32> = (0..22).map(|i| ((i as f32) * 0.37 - 3.0).sin()).collect();
        let mut opts: Vec<ShardedAdamW> = Vec::new();
        let mut params_per_rank: Vec<Vec<(String, Tensor)>> = Vec::new();
        for rank in 0..size {
            let opts_p: Vec<Tensor> =
                params0.iter().map(|t| t.shallow_clone()).collect();
            let ps: Vec<(String, Tensor)> =
                nd.iter().zip(opts_p).map(|((n, _), t)| (n.clone(), t)).collect();
            params_per_rank.push(ps);
            opts.push(ShardedAdamW::new(&nd, rank, size, cfg, Device::Cpu).unwrap());
        }
        for _ in 0..steps {
            for (rank, opt) in opts.iter_mut().enumerate() {
                opt.step(&params_per_rank[rank], &grad).unwrap();
            }
        }
        // All-gather simulation: chunks[src] = rank src's packed owned params.
        let chunks: Vec<Vec<f32>> = opts
            .iter()
            .enumerate()
            .map(|(src, o)| o.pack_owned_params(&params_per_rank[src]).unwrap())
            .collect();
        opts[0].write_gathered(&params_per_rank[0], &chunks).unwrap();
        (
            to_cpu_flat(&params_per_rank[0][0].1),
            to_cpu_flat(&params_per_rank[0][1].1),
            opts.remove(0),
        )
    }

    #[test]
    fn size1_sharded_equals_full_and_size3_reassembles_identically() {
        // size 1: sharded opt with full state == the only meaningful "full".
        let (w1, b1, opt1) = run_sharded(1, 2);
        let (w3, b3, opt3) = run_sharded(3, 2);
        assert_eq!(w1, w3, "params diverged between size=1 and size=3");
        assert_eq!(b1, b3, "biases diverged between size=1 and size=3");
        // Memory story: with 2 tensors (18 + 4 elements, m+v, f32):
        // full = 22*4*2 = 176 bytes; size 3: rank 0 owns rows [0,2) of w (2*3=6)
        // and [0,2) of b (2) → (6+2)*4*2 = 64 bytes.
        assert_eq!(opt1.full_bytes(), 176);
        assert_eq!(opt3.owned_bytes(), 64);
        assert_eq!(opt3.full_bytes(), 176);
        // The owned state really is ShardId-addressed.
        let ids = opt3.owned_shard_ids();
        assert_eq!(ids[0], ShardId { tensor: "w".into(), axis: 0, start: 0, end: 2 });
        assert_eq!(ids[1], ShardId { tensor: "b".into(), axis: 0, start: 0, end: 2 });
    }

    #[test]
    fn reassembly_from_other_ranks_only() {
        // Rank 0 must not silently keep its own rows: overwrite rank 0's
        // params after its step with garbage, then write_gathered must
        // restore every row from the gathered chunks.
        let cfg = AdamWCfg { lr: 0.05, ..AdamWCfg::default() };
        let (nd, params0) = make_params(Device::Cpu, 0.5);
        let grad: Vec<f32> = (0..22).map(|i| ((i as f32) * 0.37 - 3.0).sin()).collect();
        let size = 3;
        let mut params_per_rank: Vec<Vec<(String, Tensor)>> = Vec::new();
        let mut opts: Vec<ShardedAdamW> = Vec::new();
        for rank in 0..size {
            let ps: Vec<(String, Tensor)> = nd
                .iter()
                .zip(params0.iter())
                .map(|((n, _), t)| (n.clone(), t.shallow_clone()))
                .collect();
            params_per_rank.push(ps);
            opts.push(ShardedAdamW::new(&nd, rank, size, cfg, Device::Cpu).unwrap());
        }
        for (rank, opt) in opts.iter_mut().enumerate() {
            opt.step(&params_per_rank[rank], &grad).unwrap();
        }
        let chunks: Vec<Vec<f32>> = opts
            .iter()
            .enumerate()
            .map(|(src, o)| o.pack_owned_params(&params_per_rank[src]).unwrap())
            .collect();
        // Sabotage rank 0's params entirely.
        for (_, p) in params_per_rank[0].iter_mut() {
            let _ = p.fill_(12345.0);
        }
        opts[0].write_gathered(&params_per_rank[0], &chunks).unwrap();
        // Compare against a clean full (size=1) run.
        let (_w1, _b1, opt1) = {
            let (nd, params) = make_params(Device::Cpu, 0.5);
            let ps: Vec<(String, Tensor)> = nd
                .iter()
                .zip(params)
                .map(|((n, _), t)| (n.clone(), t))
                .collect();
            let mut opt = ShardedAdamW::new(&nd, 0, 1, cfg, Device::Cpu).unwrap();
            opt.step(&ps, &grad).unwrap();
            let w = to_cpu_flat(&ps[0].1);
            let b = to_cpu_flat(&ps[1].1);
            (w, b, opt)
        };
        let _ = opt1;
        // (recomputed above for clarity; rank0's sabotaged params must equal
        // the gathered truth, not the garbage)
        let w0 = to_cpu_flat(&params_per_rank[0][0].1);
        assert!(
            w0.iter().zip(&_w1).all(|(a, b)| (a - b).abs() < 1e-6),
            "rank 0 params were not restored from gathered shards"
        );
        let b0 = to_cpu_flat(&params_per_rank[0][1].1);
        assert!(
            b0.iter().zip(&_b1).all(|(a, b)| (a - b).abs() < 1e-6),
            "rank 0 biases were not restored from gathered shards"
        );
    }

    #[test]
    fn mismatched_param_order_bails() {
        let cfg = AdamWCfg::default();
        let (nd, params) = make_params(Device::Cpu, 0.5);
        let mut opt = ShardedAdamW::new(&nd, 0, 1, cfg, Device::Cpu).unwrap();
        let mut shuffled: Vec<Tensor> = params.iter().map(|t| t.shallow_clone()).collect();
        shuffled.reverse();
        let ps: Vec<(String, Tensor)> = nd
            .iter()
            .zip(shuffled)
            .map(|((n, _), t)| (n.clone(), t))
            .collect();
        assert!(opt.step(&ps, &vec![0.0f32; 22]).is_err());
    }

    #[test]
    fn short_grad_buffer_bails() {
        let cfg = AdamWCfg::default();
        let (nd, params) = make_params(Device::Cpu, 0.5);
        let ps: Vec<(String, Tensor)> =
            nd.iter().zip(params).map(|((n, _), t)| (n.clone(), t)).collect();
        let mut opt = ShardedAdamW::new(&nd, 0, 1, cfg, Device::Cpu).unwrap();
        assert!(opt.step(&ps, &vec![0.0f32; 5]).is_err());
    }
}

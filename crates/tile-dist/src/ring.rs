//! TCP ring all-reduce: reduce-scatter + all-gather (Baidu ring, Horovod /
//! NCCL style). Hand-rolled — tile-train's own runtime, no NCCL (ADR-0002 §1).
//!
//! Ring layout: ranks 0..n, each rank dials its successor `next` and accepts
//! from its predecessor `prev`. On every ring step a rank sends one chunk to
//! `next` and receives one chunk from `prev`, so bandwidth is symmetric and
//! no rank ever acts as a server for the others — the decentralization
//! property we need before gossip replaces the ring (§3).

use anyhow::{Context, Result};
use std::net::{TcpListener, TcpStream};

/// Established ring neighborhood: connections to the two ring neighbors.
pub struct Ring {
    pub rank: usize,
    pub size: usize,
    next: TcpStream,
    prev: TcpStream,
}

impl Ring {
    /// Data-plane endpoints each worker must expose: one listener.
    pub fn bind() -> Result<(TcpListener, String)> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?.to_string();
        Ok((listener, addr))
    }

    /// Establish the ring: dial the successor, accept from the predecessor.
    /// The worker already holds its data-plane listener (bound before Hello),
    /// so accept happens on it.
    ///
    /// Every rank dials *before* accepting; dialing succeeds because the peer
    /// bound its listener before `Assign` went out, so no cross-rank deadlock.
    pub fn connect(
        rank: usize,
        size: usize,
        listener: TcpListener,
        peer_addrs: &[String],
    ) -> Result<Self> {
        if size == 1 {
            anyhow::bail!("size 1 has no ring; callers must skip all-reduce");
        }
        if peer_addrs.len() != size {
            anyhow::bail!(
                "ring: peer_addrs has {} entries, expected {size}",
                peer_addrs.len()
            );
        }
        let next_rank = (rank + 1) % size;
        let prev_rank = (rank + size - 1) % size;
        let next = TcpStream::connect(&peer_addrs[next_rank])
            .with_context(|| format!("ring: dialing next (rank {next_rank} at {})", peer_addrs[next_rank]))?;
        let (prev, _peer) = listener
            .accept()
            .with_context(|| format!("ring: accepting prev (rank {prev_rank})"))?;
        Ok(Self { rank, size, next, prev })
    }

    /// In-place all-reduce SUM over `buf` across the ring.
    ///
    /// Phase 1 (reduce-scatter): after n-1 steps chunk `c` is fully summed
    /// and lives on rank `c`. Phase 2 (all-gather): n-1 more steps circulate
    /// the fully summed chunks to everyone.
    pub fn all_reduce_sum(&mut self, buf: &mut [f32]) -> Result<()> {
        let n = self.size;
        if buf.len() % n != 0 && buf.len() < n {
            anyhow::bail!("ring: buffer of {} elements too small for {} ranks", buf.len(), n);
        }
        // Chunk c (possibly uneven for the last chunks), precomputed so the
        // exchange loop owns no borrow of `buf`.
        let ranges: Vec<(usize, usize)> = (0..n)
            .map(|c| {
                let base = buf.len() / n;
                let rem = buf.len() % n;
                let start = c * base + c.min(rem);
                (start, start + base + if c < rem { 1 } else { 0 })
            })
            .collect();

        // ---- reduce-scatter: send (rank - i) % n, recv-add (rank - i - 1) % n
        for i in 1..n {
            let send_c = (self.rank + n - i) % n;
            let recv_c = (self.rank + n - i - 1) % n;
            self.exchange_chunk(buf, &ranges, send_c, recv_c, true)?;
        }
        // ---- all-gather: send owned (rank + 1 - i) % n, recv (rank - i) % n
        for i in 1..n {
            let send_c = (self.rank + 1 + n - i) % n;
            let recv_c = (self.rank + n - i) % n;
            self.exchange_chunk(buf, &ranges, send_c, recv_c, false)?;
        }
        Ok(())
    }

    /// Convenience: all-reduce then divide by ring size (gradient averaging).
    pub fn all_reduce_avg(&mut self, buf: &mut [f32]) -> Result<()> {
        self.all_reduce_sum(buf)?;
        let n = self.size as f32;
        for v in buf.iter_mut() {
            *v /= n;
        }
        Ok(())
    }

    /// One ring step: send chunk `send_c`, receive chunk `recv_c`.
    /// `add = true` accumulates into the local chunk; `false` replaces it.
    /// Send runs on a helper thread so both neighbors make progress (a fully
    /// blocking send-then-recv can deadlock once chunks exceed socket buffers).
    fn exchange_chunk(
        &mut self,
        buf: &mut [f32],
        ranges: &[(usize, usize)],
        send_c: usize,
        recv_c: usize,
        add: bool,
    ) -> Result<()> {
        let (s, e) = ranges[send_c];
        let outgoing: Vec<f32> = buf[s..e].to_vec();
        let bytes = f32_slice_as_bytes(&outgoing);
        let mut next = self.next.try_clone().context("cloning ring socket")?;
        let sender = std::thread::spawn(move || crate::frame::send_frame(&mut next, &bytes));
        let incoming = crate::frame::recv_frame(&mut self.prev)?;
        let (rs, re) = ranges[recv_c];
        let got: Vec<f32> = incoming
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        anyhow::ensure!(
            got.len() == re - rs,
            "ring: chunk size mismatch: got {}, expected {}",
            got.len(),
            re - rs
        );
        if add {
            for (dst, src) in buf[rs..re].iter_mut().zip(got) {
                *dst += src;
            }
        } else {
            buf[rs..re].copy_from_slice(&got);
        }
        sender.join().map_err(|_| anyhow::anyhow!("ring sender panicked"))??;
        Ok(())
    }
}

fn f32_slice_as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spin up `size` interconnected fake ranks on localhost and run the ring
    /// on each with its own buffer. Returns each rank's buffer.
    fn run_ring(size: usize, bufs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let listeners: Vec<TcpListener> =
            (0..size).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
        let addrs: Vec<String> = listeners
            .iter()
            .map(|l| l.local_addr().unwrap().to_string())
            .collect();
        let handles: Vec<_> = (0..size)
            .map(|rank| {
                let listener = listeners[rank].try_clone().unwrap();
                let addrs = addrs.clone();
                let mut buf = bufs[rank].clone();
                std::thread::spawn(move || {
                    let mut ring = Ring::connect(rank, size, listener, &addrs).unwrap();
                    ring.all_reduce_sum(&mut buf).unwrap();
                    buf
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    #[test]
    fn ring_sum_three_ranks_uneven_chunks() {
        // buffers sized 7 (uneven chunks over 3 ranks) with distinct values.
        let bufs: Vec<Vec<f32>> = (0..3)
            .map(|r| (0..7).map(|i| (r * 100 + i) as f32).collect())
            .collect();
        let out = run_ring(3, bufs);
        for (rank, buf) in out.iter().enumerate() {
            let expected: Vec<f32> = (0..7)
                .map(|i| (0..3).map(|r| (r * 100 + i) as f32).sum())
                .collect();
            assert_eq!(buf, &expected, "rank {rank} buffer wrong");
        }
    }

    #[test]
    fn ring_sum_four_ranks_constant_buffers() {
        let bufs: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..8).map(|_i| (r + 1) as f32).collect())
            .collect();
        let out = run_ring(4, bufs);
        for buf in out {
            // sum of 1+2+3+4 = 10 everywhere
            assert!(buf.iter().all(|v| (*v - 10.0).abs() < 1e-5), "buf={buf:?}");
        }
    }

    #[test]
    fn ring_matches_single_process_reference() {
        let bufs: Vec<Vec<f32>> = (0..3)
            .map(|r| (0..16).map(|i| ((r * 7 + i * 3) % 11) as f32).collect())
            .collect();
        let reference: Vec<f32> = (0..16)
            .map(|i| (0..3).map(|r| ((r * 7 + i * 3) % 11) as f32).sum())
            .collect();
        let out = run_ring(3, bufs);
        for buf in out {
            assert_eq!(buf, reference);
        }
    }
}

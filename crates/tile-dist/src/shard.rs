//! ShardId addressing (ADR-0002 §2): "everything is a shard". A shard is
//! addressed as (tensor, axis, range) — parameters, gradients and optimizer
//! state all use the same addressing, so ZeRO-style state sharding and later
//! decentralized aggregation (gossip over shards) are one mechanism.
//!
//! The contiguous range layout here is the *same* the ring all-reduce uses
//! (chunk c belongs to rank c), so a shard id doubles as a ring destination.

use serde::{Deserialize, Serialize};

/// Address of one contiguous slice of a named tensor along `axis`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardId {
    pub tensor: String,
    pub axis: usize,
    /// Inclusive start along `axis`.
    pub start: i64,
    /// Exclusive end along `axis`.
    pub end: i64,
}

/// Split `len` elements into `parts` contiguous ranges; the first
/// `len % parts` ranges get one extra element. Range `c` belongs to rank `c`
/// on the ring. Requires `parts >= 1`; ranges may be empty when
/// `parts > len` (callers normally guard against that).
pub fn contiguous_ranges(len: usize, parts: usize) -> Vec<(usize, usize)> {
    let base = len / parts;
    let rem = len % parts;
    (0..parts)
        .map(|c| {
            let start = c * base + c.min(rem);
            (start, start + base + if c < rem { 1 } else { 0 })
        })
        .collect()
}

impl ShardId {
    /// Shard a tensor dimension into one ShardId per part, in ring order.
    pub fn split(tensor: impl Into<String>, axis: usize, len: i64, parts: usize) -> Vec<ShardId> {
        let name: String = tensor.into();
        contiguous_ranges(len.max(0) as usize, parts)
            .into_iter()
            .map(|(s, e)| ShardId {
                tensor: name.clone(),
                axis,
                start: s as i64,
                end: e as i64,
            })
            .collect()
    }

    /// Number of elements this shard covers along its axis.
    pub fn len(&self) -> i64 {
        (self.end - self.start).max(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uneven_split_covers_everything_exactly_once() {
        let ranges = contiguous_ranges(7, 3);
        assert_eq!(ranges, vec![(0, 3), (3, 5), (5, 7)]);
        // coverage: no gaps, no overlap, union == [0, 7)
        let mut covered = vec![false; 7];
        for (s, e) in &ranges {
            for i in *s..*e {
                assert!(!covered[i], "element {i} covered twice");
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|c| *c));
    }

    #[test]
    fn even_split_and_single_part() {
        assert_eq!(contiguous_ranges(8, 4), vec![(0, 2), (2, 4), (4, 6), (6, 8)]);
        assert_eq!(contiguous_ranges(5, 1), vec![(0, 5)]);
    }

    #[test]
    fn more_parts_than_elements_gives_empty_ranges() {
        let ranges = contiguous_ranges(2, 4);
        assert_eq!(ranges, vec![(0, 1), (1, 2), (2, 2), (2, 2)]);
    }

    #[test]
    fn shard_id_split_len_and_serde_roundtrip() {
        let ids = ShardId::split("model.w", 0, 6, 3);
        assert_eq!(
            ids,
            vec![
                ShardId { tensor: "model.w".into(), axis: 0, start: 0, end: 2 },
                ShardId { tensor: "model.w".into(), axis: 0, start: 2, end: 4 },
                ShardId { tensor: "model.w".into(), axis: 0, start: 4, end: 6 },
            ]
        );
        assert_eq!(ids[0].len(), 2);
        assert!(!ids[0].is_empty());
        let json = serde_json::to_string(&ids[1]).unwrap();
        let back: ShardId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ids[1]);
    }
}

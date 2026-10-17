//! Model hyperparameters, serializable so checkpoints carry their own shape.

use serde::{Deserialize, Serialize};

/// Hyperparameters for the Phase-1 tiny GPT. Every checkpoint sidecar embeds
/// one of these so `load_checkpoint` can rebuild a matching `VarStore`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GptConfig {
    /// Token vocabulary size (embeddings + LM head output).
    pub vocab_size: i64,
    /// Transformer blocks.
    pub n_layer: i64,
    /// Attention heads per block.
    pub n_head: i64,
    /// Residual-stream width.
    pub n_embd: i64,
    /// Maximum context length (learned positional embeddings).
    pub block_size: i64,
}

impl GptConfig {
    /// Head width: `n_embd` must divide evenly.
    pub fn head_dim(&self) -> i64 {
        assert!(
            self.n_embd % self.n_head == 0,
            "n_embd ({}) must be divisible by n_head ({})",
            self.n_embd,
            self.n_head
        );
        self.n_embd / self.n_head
    }

    /// Phase-1 default: deliberately tiny — trains in seconds on the 50HX.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            n_layer: 2,
            n_head: 2,
            n_embd: 64,
            block_size: 32,
        }
    }
}

impl Default for GptConfig {
    fn default() -> Self {
        Self::tiny()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_serde_roundtrip() {
        let cfg = GptConfig::tiny();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GptConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn head_dim_divides() {
        assert_eq!(GptConfig::tiny().head_dim(), 32);
    }
}

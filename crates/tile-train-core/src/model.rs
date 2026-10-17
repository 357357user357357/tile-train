//! Phase-1 tiny GPT implemented with tch-rs ops (ADR-0001 backend).
//!
//! Structure follows the classic decoder-only recipe: token + learned
//! positional embeddings, pre-LN causal self-attention blocks with GELU MLPs,
//! final LN, and a tied-free LM head. Ops stay plain tch calls so Phase-2 can
//! swap individual ops for tiled CubeCL kernels without touching the loop.

use crate::config::GptConfig;
use tch::nn::{self, embedding, linear, Linear, Module, Path};
use tch::{Kind, Tensor};

/// Multi-head causal self-attention.
#[derive(Debug)]
struct CausalSelfAttention {
    qkv_proj: Linear,
    out_proj: Linear,
    n_head: i64,
    head_dim: i64,
}

impl CausalSelfAttention {
    fn new(p: &Path, cfg: &GptConfig) -> Self {
        Self {
            qkv_proj: linear(p / "qkv_proj", cfg.n_embd, 3 * cfg.n_embd, Default::default()),
            out_proj: linear(p / "out_proj", cfg.n_embd, cfg.n_embd, Default::default()),
            n_head: cfg.n_head,
            head_dim: cfg.head_dim(),
        }
    }
}

impl Module for CausalSelfAttention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (b, t, c) = xs.size3().expect("attention input must be [B, T, C]");
        let qkv = xs
            .apply(&self.qkv_proj)
            .view([b, t, 3, self.n_head, self.head_dim])
            .permute([2, 0, 3, 1, 4]);
        let q = qkv.narrow(0, 0, 1).squeeze_dim(0);
        let k = qkv.narrow(0, 1, 1).squeeze_dim(0);
        let v = qkv.narrow(0, 2, 1).squeeze_dim(0);

        // Scaled dot-product with a causal mask: token i may attend to <= i.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = q.matmul(&k.transpose(-2, -1)) * scale;
        let mask = Tensor::ones([t, t], (Kind::Bool, xs.device()))
            .tril(0)
            .view([1, 1, t, t]);
        att = att.masked_fill(&mask.logical_not(), f64::NEG_INFINITY);
        let att = att.softmax(-1, Kind::Float);

        att.matmul(&v)
            .permute([0, 2, 1, 3])
            .contiguous()
            .view([b, t, c])
            .apply(&self.out_proj)
    }
}

/// Position-wise MLP: expand 4x, GELU, project back.
#[derive(Debug)]
struct Mlp {
    fc: Linear,
    proj: Linear,
}

impl Mlp {
    fn new(p: &Path, cfg: &GptConfig) -> Self {
        Self {
            fc: linear(p / "fc", cfg.n_embd, 4 * cfg.n_embd, Default::default()),
            proj: linear(p / "proj", 4 * cfg.n_embd, cfg.n_embd, Default::default()),
        }
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc).gelu("none").apply(&self.proj)
    }
}

/// Pre-LN transformer block.
#[derive(Debug)]
struct Block {
    ln1: nn::LayerNorm,
    attn: CausalSelfAttention,
    ln2: nn::LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(p: &Path, cfg: &GptConfig) -> Self {
        let ln_config = nn::LayerNormConfig::default();
        Self {
            ln1: nn::layer_norm(p / "ln1", vec![cfg.n_embd], ln_config),
            attn: CausalSelfAttention::new(&(p / "attn"), cfg),
            ln2: nn::layer_norm(p / "ln2", vec![cfg.n_embd], ln_config),
            mlp: Mlp::new(&(p / "mlp"), cfg),
        }
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs + self.ln1.forward(xs).apply(&self.attn);
        let y = self.ln2.forward(&x).apply(&self.mlp);
        x + y
    }
}

/// The tiny GPT: `forward` returns next-token logits for `[B, T]` token ids.
#[derive(Debug)]
pub struct TinyGpt {
    cfg: GptConfig,
    wte: nn::Embedding,
    wpe: Tensor,
    blocks: Vec<Block>,
    ln_f: nn::LayerNorm,
    lm_head: Linear,
}

impl TinyGpt {
    pub fn new(p: &Path, cfg: &GptConfig) -> Self {
        let e_cfg = nn::EmbeddingConfig::default();
        Self {
            cfg: cfg.clone(),
            wte: embedding(p / "wte", cfg.vocab_size, cfg.n_embd, e_cfg),
            wpe: (p / "wpe").randn("w", &[cfg.block_size, cfg.n_embd], 0.0, 0.02),
            blocks: (0..cfg.n_layer)
                .map(|i| Block::new(&(p / format!("h{i}")), cfg))
                .collect(),
            ln_f: nn::layer_norm(p / "ln_f", vec![cfg.n_embd], nn::LayerNormConfig::default()),
            lm_head: linear(p / "lm_head", cfg.n_embd, cfg.vocab_size, Default::default()),
        }
    }

    /// Next-token logits of shape `[B, T, vocab_size]`.
    pub fn logits(&self, xs: &Tensor) -> Tensor {
        let (b, t) = xs.size2().expect("token ids must be [B, T]");
        assert!(t <= self.cfg.block_size, "sequence longer than block_size");
        let x = self.wte.forward(xs) + self.wpe.narrow(0, 0, t).view([1, t, -1]);
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x);
        }
        x.apply(&self.ln_f)
            .apply(&self.lm_head)
            .view([b, t, self.cfg.vocab_size])
    }

    /// Mean cross-entropy for `[B, T]` inputs against `[B, T]` target ids.
    pub fn forward_loss(&self, xs: &Tensor, targets: &Tensor) -> Tensor {
        let logits = self.logits(xs);
        let (b, t, v) = logits.size3().unwrap();
        // Slice-shaped inputs (e.g. narrow along a non-outer dim) can arrive
        // non-contiguous; cross entropy needs a flat [B*T, V] / [B*T] pair.
        logits
            .contiguous()
            .view([b * t, v])
            .cross_entropy_for_logits(&targets.to_kind(Kind::Int64).contiguous().view([b * t]))
    }
}

impl Module for TinyGpt {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.logits(xs)
    }
}

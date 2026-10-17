//! BitLinear b1.58 — ternary weights {-1, 0, +1} with absmean scaling,
//! straight-through-estimator training (BitNet b1.58 recipe, ADR ROADMAP).
//!
//! The weight stays an fp32 master (`self.w`, a normal trainable variable);
//! ternarization is *virtual*: recomputed on every forward, never stored.
//! Forward uses the ternary weights, backward flows the fp32 gradient
//! straight through (`w_ste = w + (quant(w) - w).detach()`), so the optimizer
//! always sees the true fp32 parameter. True 1.58-bit *storage* (packed
//! ternary checkpoints) and ternary matmul *kernels* are Phase-2/3 work on
//! top of this module — this file is the QAT accuracy surface.
//!
//! Quantization (absmean, BitNet b1.58):
//!   gamma = mean(|W|)
//!   W_q   = clamp(round(W / gamma), -1, 1) * gamma
//! so every effective weight is in {-gamma, 0, +gamma}.
//!
//! Kept at full precision (BitNet practice): embeddings, positional table,
//! the final LM head, and all LayerNorms — only attention qkv/out and MLP
//! fc/proj projections are ternarized (see `model.rs`).

use tch::nn::{init, Module, Path};
use tch::{Kind, Tensor};

/// A linear layer trained with ternary weights via STE.
#[derive(Debug)]
pub struct BitLinear {
    /// fp32 master weight `[out, in]` — the actual optimizer state.
    w: Tensor,
    /// fp32 bias `[out]`; biases are never quantized.
    b: Option<Tensor>,
}

impl BitLinear {
    /// Build with the same default init shape as `nn::linear` (uniform
    /// ±1/sqrt(fan_in) for weight and bias).
    pub fn new(p: &Path, in_dim: i64, out_dim: i64) -> Self {
        let bound = 1.0 / (in_dim as f64).sqrt();
        let w = p.var(
            "w",
            &[out_dim, in_dim],
            init::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        );
        let b = p.var(
            "b",
            &[out_dim],
            init::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        );
        Self { w, b: Some(b) }
    }

    /// Virtual ternary weight: `clamp(round(W / gamma), -1, 1) * gamma` with
    /// `gamma = mean(|W|)`. Differentiable only through the STE in
    /// [`Module::forward`]; call this alone for inspection/tests.
    pub fn weight_quant(&self) -> Tensor {
        let gamma = self.w.abs().mean(Kind::Float);
        let w_scaled = &self.w / &gamma;
        let w_ternary = w_scaled.round().clamp(-1.0, 1.0);
        &w_ternary * &gamma
    }

    /// The absmean scale `gamma = mean(|W|)` (for logging/tests).
    pub fn gamma(&self) -> f64 {
        self.w.abs().mean(Kind::Float).double_value(&[])
    }
}

impl Module for BitLinear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let w_q = self.weight_quant();
        // Straight-through estimator: values are ternary, gradients are not.
        let w_ste = &self.w + (w_q - &self.w).detach();
        let mut y = xs.matmul(&w_ste.transpose(-1, -2));
        if let Some(b) = &self.b {
            y = y + b;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::nn::VarStore;
    use tch::Device;

    fn bit_linear_at(w_data: &[f32], out_dim: i64, in_dim: i64) -> BitLinear {
        let vs = VarStore::new(Device::Cpu);
        let root = vs.root();
        let bl = BitLinear::new(&(root / "bl"), in_dim, out_dim);
        let src = Tensor::from_slice(w_data).view([out_dim, in_dim]);
        // A view shares the variable's storage, so copy_ writes through to
        // the leaf variable (same trick as the ZeRO-1 param gather).
        tch::no_grad(|| {
            let mut w_view = bl.w.view([-1]);
            let _ = w_view.f_copy_(&src.view([-1])).unwrap();
        });
        bl
    }

    #[test]
    fn quant_values_are_ternary_scaled_by_absmean() {
        // Arbitrary weights; effective values must be {-g, 0, +g}.
        let bl = bit_linear_at(&[0.3, -0.7, 0.1, -0.5, 0.9, -0.2], 2, 3);
        let g = bl.gamma();
        assert!(g > 0.0);
        let w_q = bl.weight_quant();
        for i in 0..6 {
            let v = w_q.double_value(&[i / 3, i % 3]);
            let in_set = (v - g).abs() < 1e-6 || (v + g).abs() < 1e-6 || v.abs() < 1e-6;
            assert!(in_set, "value {v} not in {{-g, 0, +g}} for g={g}");
        }
    }

    #[test]
    fn constant_weight_quantizes_to_itself() {
        // If |W| is constant, W/gamma = ±1, so W_q == W exactly.
        let bl = bit_linear_at(&[0.25; 6], 2, 3);
        let w_q = bl.weight_quant();
        for i in 0..6 {
            assert!((w_q.double_value(&[i / 3, i % 3]) - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn forward_is_matmul_with_quantized_weight() {
        // w all 0.4 -> gamma = 0.4, w_q = w; y = 0.4 * sum(x) + b.
        let bl = bit_linear_at(&[0.4; 4], 2, 2);
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]).view([2, 2]);
        let y = bl.forward(&x);
        // Row [1,2]: sum 3 -> 1.2 + b; row [3,4]: sum 7 -> 2.8 + b.
        let b0 = bl.b.as_ref().unwrap().double_value(&[0]);
        let b1 = bl.b.as_ref().unwrap().double_value(&[1]);
        assert!((y.double_value(&[0, 0]) - (0.4 * 3.0 + b0)).abs() < 1e-5);
        assert!((y.double_value(&[1, 1]) - (0.4 * 7.0 + b1)).abs() < 1e-5);
    }

    #[test]
    fn ste_gradient_matches_plain_matmul() {
        // The STE means dL/dw through the quantized forward must equal the
        // gradient of an unquantized matmul with the same input and upsteam
        // signal (sum of outputs), despite the ternary values.
        let bl = bit_linear_at(&[0.3, -0.7, 0.1, -0.5, 0.9, -0.2], 2, 3);
        let x = Tensor::randn([4, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let y = bl.forward(&x);
        y.sum(Kind::Float).backward();
        let got = bl.w.grad();
        assert!(got.defined());

        // Reference: y_ref = x @ w^T with the same w, same upstream sum.
        let w_ref = bl.w.detach().set_requires_grad(true);
        let y_ref = x.matmul(&w_ref.transpose(-1, -2));
        y_ref.sum(Kind::Float).backward();
        let want = w_ref.grad();
        let diff = (&got - &want).abs().max().double_value(&[]);
        assert!(diff < 1e-6, "STE gradient differs from plain matmul: {diff}");
    }

    #[test]
    fn ste_gradient_is_not_the_quantized_values_gradient() {
        // Sanity that the test above is meaningful: quantized values differ
        // from the master weights, yet gradients flow to the master.
        let bl = bit_linear_at(&[0.3, -0.7, 0.1, -0.5, 0.9, -0.2], 2, 3);
        let x = Tensor::ones([1, 3], (Kind::Float, Device::Cpu));
        let w_q = bl.weight_quant();
        let master = bl.w.detach();
        assert!(
            (&w_q - &master).abs().max().double_value(&[]) > 0.05,
            "quantized weight should differ from master for this fixture"
        );
        let y = bl.forward(&x);
        y.sum(Kind::Float).backward();
        assert!(bl.w.grad().defined(), "master weight received no gradient");
    }
}

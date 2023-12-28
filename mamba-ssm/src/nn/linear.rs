use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Linear with some more traceability
#[derive(Debug, Clone)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Self {
        let dims = weights.shape().clone();
        assert!(dims.elem_count() >= 2);
        let d1 = dims.dims()[0];
        let d2 = dims.dims()[1];
        let inner = candle_nn::Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear", d1, d2);
        Self { inner, span }
    }
}

pub fn linear(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let vb_prefix = vb.prefix();
    let inner = candle_nn::linear(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear", vb_prefix, d1, d2);
    Ok(Linear { inner, span })
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let vb_prefix = vb.prefix();
    let inner = candle_nn::linear_no_bias(d1, d2, vb)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear_no_bias", vb_prefix, d1, d2);
    Ok(Linear { inner, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

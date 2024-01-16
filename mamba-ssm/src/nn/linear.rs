use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Linear with improved speed and some more traceability.
/// The layer applies a linear transformation to the incoming data,
/// `y = x@w.t() + b` with an optional bias. However, the weights are
/// pre-transposed upon creation of the layer.
#[derive(Debug, Clone)]
pub struct Linear {
    inner: LinearInner,
    span: tracing::Span,
}

impl Linear {
    pub fn from_weights(weights: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weights.shape().clone();
        assert!(dims.elem_count() >= 2);
        let in_dim = dims.dims()[0];
        let out_dim = dims.dims()[1];
        let inner = LinearInner::new(weights, bias)?;
        let span = tracing::span!(tracing::Level::TRACE, "linear", in_dim, out_dim);
        Ok(Self { inner, span })
    }
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let vb_prefix = vb.prefix();

    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };

    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear", vb_prefix, in_dim, out_dim);
    Ok(Linear {
        inner: LinearInner::new(ws, Some(bs))?,
        span,
    })
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let vb_prefix = vb.prefix();
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let span = tracing::span!(
        tracing::Level::TRACE,
        "linear_no_bias",
        vb_prefix,
        in_dim,
        out_dim
    );
    Ok(Linear {
        inner: LinearInner::new(ws, None)?,
        span,
    })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// LinearInner is basically the same as candle_nn::Linear except the weights
/// are transposed on creation instead of every forward iteration
#[derive(Clone, Debug)]
struct LinearInner {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearInner {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            weight: weight.t()?.contiguous()?,
            bias,
        })
    }
}

impl Module for LinearInner {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = match *x.dims() {
            [b1, b2, _, _] => x.matmul(&self.weight.broadcast_left((b1, b2))?)?,
            [bsize, _, _] => x.matmul(&self.weight.broadcast_left(bsize)?)?,
            _ => x.matmul(&self.weight)?,
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

#[cfg(test)]
mod tests {
    use candle::{Module, Tensor};

    #[test]
    fn test_linear() -> candle::Result<()> {
        let device = &candle::Device::Cpu;

        let x = Tensor::randn(0f32, 1., (1, 10, 2000), &device)?;

        let var_map = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

        let candle_linear = candle_nn::linear(2000, 2000, vb.clone())?;
        let custom_linear = super::linear(2000, 2000, vb)?;

        let candle_res = candle_linear.forward(&x)?;
        let custom_res = custom_linear.forward(&x)?;

        let res_diff = (candle_res - custom_res)?.sum_all()?.to_scalar::<f32>()?;

        // The difference should be 0 but this adds a bit of leniency
        assert!(res_diff.abs() < 1e-8);
        Ok(())
    }
}

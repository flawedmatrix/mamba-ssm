use candle::{IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;

use candle::Result;

use crate::primitives::{linear, linear_no_bias, Linear};

#[derive(Clone, Debug)]
pub struct SSM {
    dt_rank: usize,

    a_log: Tensor,
    d: Tensor,

    x_proj: Linear,
    dt_proj: Linear,

    span: tracing::Span,
}

impl SSM {
    pub fn new(d_inner: usize, d_state: usize, dt_rank: usize, vb: VarBuilder) -> Result<Self> {
        let x_proj = linear_no_bias(d_inner, dt_rank + d_state * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;

        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let d = vb.get(d_inner, "D")?;

        let span = tracing::span!(tracing::Level::TRACE, "SSM", d_inner, d_state, dt_rank);

        Ok(Self {
            dt_rank,
            a_log,
            d,
            x_proj,
            dt_proj,

            span,
        })
    }
}
impl Module for SSM {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (_d_in, n) = self.a_log.dims2()?;
        let a = self.a_log.to_dtype(candle::DType::F32)?.exp()?.neg()?;
        let d = self.d.to_dtype(candle::DType::F32)?;
        let x_dbl = xs.apply(&self.x_proj)?;

        let delta = x_dbl.narrow(D::Minus1, 0, self.dt_rank)?;
        let b = x_dbl.narrow(D::Minus1, self.dt_rank, n)?;
        let c = x_dbl.narrow(D::Minus1, self.dt_rank + n, n)?;

        let delta = delta.contiguous()?.apply(&self.dt_proj)?;
        // softplus without threshold
        let delta = (delta.exp()? + 1.)?.log()?;
        let ss = selective_scan(xs, &delta, &a, &b, &c, &d)?;
        Ok(ss)
    }
}

fn selective_scan(
    u: &Tensor,
    delta: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
) -> Result<Tensor> {
    let (b_sz, l, d_in) = u.dims3()?;
    let n = a.dim(1)?;
    let delta = delta.t()?.reshape((b_sz, d_in, l, 1))?; // b d_in l 1
    let delta_a = delta.broadcast_mul(&a.reshape((1, d_in, 1, n))?)?.exp()?;
    let delta_b_u = delta
        .broadcast_mul(&b.reshape((b_sz, 1, l, n))?)?
        .broadcast_mul(&u.t()?.reshape((b_sz, d_in, l, 1))?)?;
    let mut xs = Tensor::zeros((b_sz, d_in, n), delta_a.dtype(), delta_a.device())?;
    let mut ys = Vec::with_capacity(l);
    for i in 0..l {
        xs = ((delta_a.i((.., .., i))? * xs)? + delta_b_u.i((.., .., i))?)?;
        let y = xs.matmul(&c.i((.., i, ..))?.unsqueeze(2)?)?.squeeze(2)?;
        ys.push(y)
    }
    let ys = Tensor::stack(ys.as_slice(), 1)?;
    ys + u.broadcast_mul(d)
}

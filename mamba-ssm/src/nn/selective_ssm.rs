use candle::{IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;

use candle::Result;

use crate::nn::{linear, linear_no_bias, Linear};

#[derive(Clone, Debug)]
pub struct SSM {
    dt_rank: usize,

    a_log: Tensor,
    d: Tensor,

    x_proj: Linear,
    dt_proj: Linear,

    span: tracing::Span,
    iter_span: tracing::Span,
}

impl SSM {
    pub fn new(d_inner: usize, d_state: usize, dt_rank: usize, vb: VarBuilder) -> Result<Self> {
        let x_proj = linear_no_bias(d_inner, dt_rank + d_state * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;

        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let d = vb.get(d_inner, "D")?;

        let span = tracing::span!(tracing::Level::TRACE, "SSM", d_inner, d_state, dt_rank);
        let iter_span = tracing::span!(tracing::Level::TRACE, "iter");

        Ok(Self {
            dt_rank,
            a_log,
            d,
            x_proj,
            dt_proj,

            span,
            iter_span,
        })
    }

    fn selective_scan(
        &self,
        u: &Tensor,     // The input (batch_size, seq_len, d_inner) aka (B L D)
        delta: &Tensor, // (batch_size, seq_len, d_inner) aka (B L D)
        a: &Tensor,     // (d_inner, d_state) aka (D N)
        b: &Tensor,     // (batch_size, seq_len, d_state) aka (B L N)
        c: &Tensor,     // (batch_size, seq_len, d_state) aka (B L N)
        d: &Tensor,     // (d_inner) aka (D)
    ) -> Result<Tensor> {
        let (batch_size, seq_len, d_inner) = u.dims3()?;
        let d_state = a.dim(1)?;
        let d = d.unsqueeze(1)?;

        let mut bys = Vec::with_capacity(batch_size);
        for batch in 0..batch_size {
            let mut x = Tensor::zeros((d_inner, d_state), delta.dtype(), delta.device())?; // h_(t-1), size (D N)
            let mut ys = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                let _enter_iter = self.iter_span.enter();
                // We want to avoid carrying around BLDN elements in memory, so
                // let's compute everything on the fly
                let delta_i = delta.i((batch, i, ..))?.reshape((d_inner, 1))?;
                let delta_a_i = delta_i.broadcast_mul(a)?.exp()?; // exp(Δ_i * A) (size D N)

                let b_i = b.i((batch, i, ..))?.reshape((1, d_state))?;
                let delta_b_i = delta_i.broadcast_mul(&b_i)?; // simplified discretization Δ_i * B (size D N)

                let u_i = u.i((batch, i, ..))?.reshape((d_inner, 1))?;
                let du_i = (&d * &u_i)?; // Calculate D * u (size D 1) first

                let dbu_i = delta_b_i.broadcast_mul(&u_i)?; // ΔB_i * u_i (size D N)

                x = ((delta_a_i * x)? + dbu_i)?; // h_t = (A_i * h_(t-1)) + (ΔB_i * u_i)

                let c_i = c.i((batch, i, ..))?.reshape((d_state, 1))?;
                let y = x.matmul(&c_i)?; // y = h_t * C (D N) x (N 1) -> (D 1)
                ys.push((y + du_i)?)
            }
            let ys = Tensor::cat(&ys, 1)?; // [(D,1) x L] -> (D, L)
            bys.push(ys)
        }
        Tensor::stack(&bys, 0)?.t()
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
        let ss = self.selective_scan(xs, &delta, &a, &b, &c, &d)?;
        Ok(ss)
    }
}

use candle::{IndexOp, Module, Result, Tensor, D};

use candle_nn::VarBuilder;

use crate::{
    context::Context,
    nn::{linear, linear_no_bias, Linear},
};

#[derive(Clone, Debug)]
pub struct SSM {
    dt_rank: usize,

    at: Tensor,
    d: Tensor,

    x_proj: Linear,
    dt_proj: Linear,

    ctx: Context,

    span: tracing::Span,
    iter_span: tracing::Span,
}

impl SSM {
    pub fn new(
        d_inner: usize,
        d_state: usize,
        dt_rank: usize,
        vb: VarBuilder,
        ctx: Context,
    ) -> Result<Self> {
        let x_proj = linear_no_bias(d_inner, dt_rank + d_state * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;

        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let d = vb.get(d_inner, "D")?;

        let dtype = ctx.dtype();

        // Store the transpose of A as an optimization
        let at = a_log.to_dtype(dtype)?.exp()?.neg()?.t()?;
        let d = d.to_dtype(dtype)?;

        let span = tracing::span!(tracing::Level::TRACE, "SSM", d_inner, d_state, dt_rank);
        let iter_span = tracing::span!(tracing::Level::TRACE, "iter");

        Ok(Self {
            dt_rank,
            at,
            d,
            x_proj,
            dt_proj,

            ctx,

            span,
            iter_span,
        })
    }

    fn selective_scan(
        &self,
        u: &Tensor,     // The input (batch_size, seq_len, d_inner) aka (B L D)
        delta: &Tensor, // (batch_size, seq_len, d_inner) aka (B L D)
        b: &Tensor,     // (batch_size, seq_len, d_state) aka (B L N)
        c: &Tensor,     // (batch_size, seq_len, d_state) aka (B L N)
    ) -> Result<Tensor> {
        let (batch_size, seq_len, d_inner) = u.dims3()?;

        let at = &self.at; // (d_state, d_inner) aka (N, D)
        let d_state = at.dim(0)?;

        // Precompute delta * u since they are both the same size
        let delta_u = (delta * u)?;

        let mut bys = Vec::with_capacity(batch_size);
        for batch in 0..batch_size {
            // Load the SSM state from the context, or initialize with 0s
            let mut ctx = self.ctx.pp(batch);
            let mut x = ctx.get((d_state, d_inner), "ht")?; // h_(t-1), size (N D)

            let mut ys = Vec::with_capacity(seq_len);
            for i in 0..seq_len {
                let _enter_iter = self.iter_span.enter();
                // We want to avoid carrying around BLDN elements in memory, so
                // let's compute everything on the fly with at most (N x D) elements

                let delta_i = delta.i((batch, i, ..))?.reshape((1, d_inner))?; // (size 1 D)
                let delta_a_i = delta_i.broadcast_mul(&at)?.exp()?; // exp(Δ_i * A) (size N D)

                let delta_u_i = delta_u.i((batch, i, ..))?.reshape((1, d_inner))?; // (size 1 D)
                let b_i = b.i((batch, i, ..))?.reshape((d_state, 1))?; // size (N 1)

                // Performs the following operation:
                // (B_i * u_i) x Δ_i simplified discretization
                // The (N 1) x (1 D) broadcast mul can also be done as a matmul)
                let delta_b_u_i = b_i.matmul(&delta_u_i)?; // (size N D)

                x = ((delta_a_i * x)? + delta_b_u_i)?; // h_t = (ΔA_i * h_(t-1)) + ΔBu_i

                let c_i = c.i((batch, i, ..))?.reshape((1, d_state))?;
                let y = c_i.matmul(&x)?; // y = C x h_t  { matrix mul (1 N) x (N D) -> (1 D) }

                let u_i = u.i((batch, i, ..))?;
                let du_i = (&self.d * &u_i)?.reshape((1, d_inner)); // Calculate D * u (size 1 D)
                ys.push((y + du_i)?)
            }
            ctx.set("ht", x)?; // Save off the resulting SSM to the context
            let ys = Tensor::cat(&ys, 0)?; // [(1,D) x L] -> (L, D)
            bys.push(ys)
        }
        Tensor::stack(&bys, 0)
    }
}

impl Module for SSM {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (n, _d_in) = self.at.dims2()?;

        let x_dbl = xs.apply(&self.x_proj)?;

        let delta = x_dbl.narrow(D::Minus1, 0, self.dt_rank)?;
        let b = x_dbl.narrow(D::Minus1, self.dt_rank, n)?;
        let c = x_dbl.narrow(D::Minus1, self.dt_rank + n, n)?;

        let delta = delta.contiguous()?.apply(&self.dt_proj)?;
        // softplus without threshold
        let delta = (delta.exp()? + 1.)?.log()?;
        let ss = self.selective_scan(xs, &delta, &b, &c)?;
        Ok(ss)
    }
}

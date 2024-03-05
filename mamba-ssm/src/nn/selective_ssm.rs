use candle::{CustomOp1, IndexOp, Module, Storage, Tensor, D};
use candle_nn::VarBuilder;

use candle::{bail, CpuStorage, Layout, Shape};

use candle::Result;

use rayon::prelude::*;

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

        // Optimized inference case
        if batch_size == 1 && seq_len == 1 && u.device().is_cpu() {
            let _enter_iter = self.iter_span.enter();
            return single_token_selective_scan(self.ctx.pp(0), u, &self.at, delta, b, c, &self.d);
        }
        // Else process the entire prompt

        let a = &self.at.t()?; // (d_inner, d_state) aka (D, N)
        let d_state = a.dim(1)?;

        let d = self.d.unsqueeze(1)?; // (d_inner) aka (D) -> (D 1)

        let mut bys = Vec::with_capacity(batch_size);
        for batch in 0..batch_size {
            // Load the SSM state from the context, or initialize with 0s
            let mut ctx = self.ctx.pp(batch);
            let mut x = ctx.get((d_state, d_inner), "ht")?.t()?; // h_(t-1), size (N D) -> (D N)

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

                let delta_b_u_i = delta_b_i.broadcast_mul(&u_i)?; // ΔB_i * u_i (size D N)

                x = ((delta_a_i * x)? + delta_b_u_i)?; // h_t = (ΔA_i * h_(t-1)) + ΔBu_i

                let c_i = c.i((batch, i, ..))?.reshape((d_state, 1))?;
                let y = x.matmul(&c_i)?; // y = h_t x C  { matrix mul (D N) x (N 1) -> (D 1) }
                ys.push((y + du_i)?)
            }
            ctx.set("ht", x.t()?)?; // Save off the resulting SSM to the context
            let ys = Tensor::cat(&ys, 1)?; // [(D,1) x L] -> (D, L)
            bys.push(ys)
        }
        Tensor::stack(&bys, 0)?.t()
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

fn single_token_selective_scan(
    mut ctx: Context,
    u: &Tensor,     // The input (1, 1, d_inner)
    at: &Tensor,    // (d_state, d_inner) aka (N D)
    delta: &Tensor, // (1, 1, d_inner)
    b: &Tensor,     // (1, 1, d_state)
    c: &Tensor,     // (1, 1, d_state)
    d: &Tensor,     // (d_inner) aka (D)
) -> Result<Tensor> {
    let (d_state, d_inner) = at.dims2()?;
    let old_x = ctx.get((d_state, d_inner), "ht")?; // h_(t-1), size (N D)

    // h_t = (exp(A * Δ) * h_(t-1)) + (Δ * B * u)
    let new_x = single_token_selective_scan_inner(&old_x, u, at, delta, b)?;

    let c_i = c.reshape((1, d_state))?;
    let y = c_i.matmul(&new_x)?; // y = h_t * C (N D) x (1, N) -> (1, D)

    ctx.set("ht", new_x)?; // Save off the resulting SSM to the context

    let u = u.reshape(d_inner)?;
    let du = (d * &u)?.reshape((1, d_inner));
    return (y + du)?.reshape((1, 1, d_inner));
}

fn single_token_selective_scan_inner(
    old_x: &Tensor,
    u: &Tensor,
    at: &Tensor,
    delta: &Tensor,
    b: &Tensor,
) -> candle::Result<Tensor> {
    let kernel = STSelectiveScanKernel::new(&old_x, u, &at, delta, b)?;
    let new_x = Tensor::new(&[0f32; 0], old_x.device())?;
    new_x.apply_op1(kernel)
}

struct STSelectiveScanKernel {
    x: Tensor,
    u: Tensor,
    a: Tensor,
    delta: Tensor,
    b: Tensor,
}

impl STSelectiveScanKernel {
    fn new(
        x: &Tensor,
        u: &Tensor,
        a: &Tensor,
        delta: &Tensor,
        b: &Tensor,
    ) -> candle::Result<STSelectiveScanKernel> {
        let x = x.contiguous()?;
        let u = u.contiguous()?;
        let a = a.contiguous()?;
        let delta = delta.contiguous()?;
        let b = b.contiguous()?;
        Ok(STSelectiveScanKernel { x, u, a, delta, b })
    }
}

impl CustomOp1 for STSelectiveScanKernel {
    fn name(&self) -> &'static str {
        "single_token_selective_scan_kernel"
    }

    // Forward pass for CPU only. Cuda and Metal passes to be implemented separately.
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle::Result<(CpuStorage, Shape)> {
        let (x_storage, _) = self.x.storage_and_layout();
        let (u_storage, _) = self.u.storage_and_layout();
        let (a_storage, _) = self.a.storage_and_layout();
        let (delta_storage, _) = self.delta.storage_and_layout();
        let (b_storage, _) = self.b.storage_and_layout();

        type S = Storage;
        let s = match (
            &*x_storage,
            &*u_storage,
            &*a_storage,
            &*delta_storage,
            &*b_storage,
        ) {
            (S::Cpu(x_s), S::Cpu(u_s), S::Cpu(a_s), S::Cpu(delta_s), S::Cpu(b_s)) => {
                cpu_single_token_selective_scan_inner(x_s, u_s, a_s, delta_s, b_s)?
            }
            _ => bail!("single_token_selective_scan_inner: All inputs Must be CPU storage type"),
        };
        return Ok((s, self.x.shape().clone()));
    }
}

// Assumes all of the storage is already contiguous
fn cpu_single_token_selective_scan_inner(
    x: &CpuStorage,
    u: &CpuStorage,
    a: &CpuStorage,
    delta: &CpuStorage,
    b: &CpuStorage,
) -> candle::Result<CpuStorage> {
    type C = CpuStorage;
    match (x, u, a, delta, b) {
        (C::BF16(_), C::BF16(_), C::BF16(_), C::BF16(_), C::BF16(_)) => {
            bail!("cpu_single_token_selective_scan_inner: BF16 not implemented")
        }
        (C::F16(_), C::F16(_), C::F16(_), C::F16(_), C::F16(_)) => {
            bail!("cpu_single_token_selective_scan_inner: F16 not implemented")
        }
        (C::F32(x_s), C::F32(u_s), C::F32(a_s), C::F32(delta_s), C::F32(b_s)) => Ok(C::F32(
            cpu_f32_single_token_selective_scan_inner(x_s, u_s, a_s, delta_s, b_s),
        )),
        _ => bail!("cpu_single_token_selective_scan_inner: Type not implemented"),
    }
}
fn cpu_f32_single_token_selective_scan_inner(
    x: &[f32],     // [d_inner x d_state]
    u: &[f32],     // [d_inner]
    a: &[f32],     // [d_inner x d_state]
    delta: &[f32], // [d_inner]
    b: &[f32],     // [d_state]
) -> Vec<f32> {
    let mut v = vec![0f32; x.len()];
    let d_inner = u.len();

    // Calculating x = (exp(A * Δ) * x) + (Δ * B * u)
    v.par_iter_mut().enumerate().for_each(|(idx, dst)| {
        let delta_i = delta[idx % d_inner];
        let u_i = u[idx % d_inner];
        let b_i = b[idx / d_inner];

        let x1 = (delta_i * a[idx]).exp() * x[idx];
        let x2 = delta_i * b_i * u_i;
        *dst = x1 + x2;
    });

    return v;
}

#[cfg(test)]
mod tests {
    use candle::Tensor;

    use super::single_token_selective_scan_inner;

    #[test]
    fn test_cpu_selective_scan_inner() -> candle::Result<()> {
        let device = &candle::Device::Cpu;
        // x: &[f32],     // [d_state x d_inner]
        // u: &[f32],     // [d_inner]
        // at: &[f32],     // [d_state x d_inner]
        // delta: &[f32], // [d_inner]
        // b: &[f32],     // [d_state]
        let old_x = Tensor::new(&[[1f32, 2., 3., 4., 5.], [1.1, 2.1, 3.1, 4.1, 5.1]], device)?;
        let u = Tensor::new(&[1f32, 2., 3., 4., 5.], device)?;
        let at = Tensor::new(&[[2f32, 3., 4., 5., 6.], [2.1, 3.1, 4.1, 5.1, 6.1]], device)?;
        let delta = Tensor::new(&[2f32, 3., 4., 5., 6.], device)?;
        let b = Tensor::new(&[10f32, 20.], device)?;

        // Compute expected result
        let (d_state, d_inner) = at.dims2()?;

        let delta_i = delta.reshape((1, d_inner))?;
        let delta_a_i = delta_i.broadcast_mul(&at)?.exp()?; // exp(Δ_i * A) (size N D)
        let b_i = b.reshape((d_state, 1))?;
        let delta_b_i = delta_i.broadcast_mul(&b_i)?; // simplified discretization Δ_i * B (size N D)
        let u_i = u.reshape((1, d_inner))?;
        let delta_b_u_i = delta_b_i.broadcast_mul(&u_i)?; // ΔB_i * u_i (size N D)
        let expected_x = ((delta_a_i * &old_x)? + delta_b_u_i)?; // h_t = (ΔA_i * h_(t-1)) + ΔBu_i

        let res = single_token_selective_scan_inner(&old_x, &u, &at, &delta, &b)?;

        assert_eq!(res.shape(), expected_x.shape());

        let diff = (res - expected_x)?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff.abs() == 0.0);

        Ok(())
    }
}

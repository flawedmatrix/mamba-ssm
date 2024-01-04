use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1dConfig, VarBuilder};

use crate::context::Context;

/// Faster Conv1d with some more traceability
#[derive(Debug, Clone)]
pub struct Conv1d {
    inner: Conv1dImpl,
    span: tracing::Span,
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        match self.inner {
            Conv1dImpl::TC(ref c) => c.forward(x),
            Conv1dImpl::Direct(ref c) => {
                let (_, seq_len, _) = x.dims3()?;
                c.forward(&x.t()?)?
                    .narrow(candle::D::Minus1, 0, seq_len)?
                    .t()
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Conv1dImpl {
    TC(TCConv1d),
    Direct(candle_nn::Conv1d),
}

pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vs: VarBuilder,
    ctx: Context,
) -> Result<Conv1d> {
    let span = tracing::span!(
        tracing::Level::TRACE,
        "conv1d",
        in_channels,
        out_channels,
        kernel_size
    );
    if in_channels == out_channels && cfg.groups == out_channels && kernel_size == 4 {
        let inner = TCConv1d::new(in_channels, out_channels, kernel_size, cfg, vs, ctx)?;
        Ok(Conv1d {
            inner: Conv1dImpl::TC(inner),
            span,
        })
    } else {
        // TODO: Does this case still make sense to keep?
        let inner = candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, vs)?;
        Ok(Conv1d {
            inner: Conv1dImpl::Direct(inner),
            span,
        })
    }
}

/// Winograd transform matrices for convolution F(2,4) evaluated at points
/// [0, 0.5, -1.5, 1.5] generated with the help of https://github.com/andravin/wincnn
/// See:
/// [1] Error Analysis and Improving the Accuracy of Winograd Convolution for Deep Neural Networks
///     Barbara Barabasz, Andrew Anderson, Kirk M. Soodhalter, David Gregg
///     https://arxiv.org/abs/1803.10986
/// [2] Winograd Convolution for Deep Neural Networks: Efficient Point Selection
///     Syed Asad Alam, Andrew Anderson, Barbara Barabasz, David Gregg
///     https://arxiv.org/pdf/2201.10369.pdf
///
/// Originally used [0, -1, 1, 0.5] as the set of points from [1] but now using
/// a set of points from [2] shown to have better FP error (E=4.69E-08 for 1D
/// conv of kernel size 3 for n=5)
const A_T: &[[f32; 5]; 2] = &[[1., 1., 1., 1., 0.], [0., 0.5, -1.5, 1.5, 1.]];
const G: &[[f32; 4]; 5] = &[
    [8. / 9., 0., 0., 0.],
    [-1., -0.5, -0.25, -0.125],
    [-1. / 9., 1. / 6., -0.25, 0.375],
    [2. / 9., 1. / 3., 0.5, 0.75],
    [0., 0., 0., 1.],
];
const B_T: &[[f32; 5]; 5] = &[
    [1.125, -2.25, -0.5, 1.0, 0.],
    [0., -2.25, 0., 1.0, 0.],
    [0., 0.75, -2.0, 1.0, 0.],
    [0., -0.75, 1.0, 1.0, 0.],
    [0., 1.125, -2.25, -0.5, 1.],
];

/// Basic implementation of Toom-Cook (Winograd) conv1d (F(2,4)). Only
/// implemented for the use case of Mamba.
#[derive(Debug, Clone)]
pub struct TCConv1d {
    // Winograd transform matrices
    at: Tensor,
    bt: Tensor,

    /// The cached result of multiplying G with the stack of kernels
    gk: Tensor,
    bias: Option<Tensor>,
    cfg: Conv1dConfig,

    ctx: Context,
}

impl TCConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        cfg: Conv1dConfig,
        vb: VarBuilder,
        ctx: Context,
    ) -> Result<Self> {
        if in_channels != cfg.groups {
            candle::bail!(
                "Mismatch between in_channels ({in_channels}) and groups ({})",
                cfg.groups
            );
        }
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let ws = vb.get_with_hints((out_channels, 1, kernel_size), "weight", init_ws)?;
        let bound = 1. / (in_channels as f64).sqrt();

        // Change the kernel's layout to be (kernel_size, channels)
        let kernel = ws.permute((1, 2, 0))?.squeeze(0)?;

        let dtype = ws.dtype();

        let at = Tensor::new(A_T, vb.device())?.to_dtype(dtype)?;
        let bt = Tensor::new(B_T, vb.device())?.to_dtype(dtype)?;
        // Save the result of multiplying G and the stack of kernels since we
        // don't need to use the kernel weights directly anymore
        let g = Tensor::new(G, vb.device())?.to_dtype(dtype)?;
        let gk = g.matmul(&kernel)?;

        let init_bs = candle_nn::Init::Uniform {
            lo: -bound,
            up: bound,
        };
        let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;

        Ok(Self {
            at,
            bt,
            gk,
            bias: Some(bs),
            cfg,
            ctx,
        })
    }
}

impl Module for TCConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Assume out_channels == in_channels
        let (b_size, seq_len, c_in) = x.dims3()?;

        let batches = (0..b_size)
            .map(|i| {
                let view = x.i((i, .., ..))?;
                let mut ctx = self.ctx.pp(i);
                let padding = ctx.get((self.cfg.padding, c_in), "ph")?;
                let view = Tensor::cat(&[padding, view], 0)?;

                let conv = tc_conv1d(&view, &self.at, &self.bt, &self.gk);

                // Since view is [seq_len + padding, c_in], this saves off
                // padding inputs from the end of the view
                ctx.set("ph", view.i((seq_len.., ..))?)?;

                conv
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let output = Tensor::stack(&batches, 0)?;

        match &self.bias {
            None => Ok(output),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, 1, b))?;
                Ok(output.broadcast_add(&bias)?)
            }
        }
    }
}

/// Given an (L+K-1)xD tensor, applies Toom-Cook 1D convolution on it via the
/// provided transformation matrices, returns an LxD tensor.
/// Assumes input is already left-padded and that
/// groups == in_channels == out_channels
fn tc_conv1d(input: &Tensor, at: &Tensor, bt: &Tensor, gk: &Tensor) -> Result<Tensor> {
    // TODO: Support convolutions other than F(2,4)
    let tile_size = 5;
    let tile_stride = 2;

    let tiles = tile_conv_input(&input, tile_size, tile_stride)?;
    let outputs = tiles
        .iter()
        .map(|tile| {
            let (chunk_size, _) = tile.dims2()?;
            let right_padding = tile_size - chunk_size;
            assert!(
                tile_stride > right_padding,
                "conv tiling returned too small of a chunk size {chunk_size}"
            );
            let tile = if right_padding > 0 {
                tile.pad_with_zeros(0, 0, right_padding)?
            } else {
                tile.clone()
            };
            let btd = bt.matmul(&tile)?;
            let elemwise_prod = gk.mul(&btd)?;
            let res = at.matmul(&elemwise_prod)?;
            if right_padding > 0 {
                res.narrow(0, 0, tile_stride - right_padding)
            } else {
                Ok(res)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&outputs, 0)
}

/// Given a LxD tensor, returns a vec of tensor views each with length
/// tile_size, advancing with tile_stride. Assumes input is already left-padded
/// as desired. If the input is not cleanly tilable by the tile size and tile
/// stride, the last tensor view will contain the remainders.
fn tile_conv_input(input: &Tensor, tile_size: usize, tile_stride: usize) -> Result<Vec<Tensor>> {
    let length = input.dims()[0];
    if length < tile_size {
        return Ok(vec![input.clone()]);
    }

    let tilable_size = (length - tile_size) + tile_stride;
    let remainder = tilable_size % tile_stride;
    let num_tiles = tilable_size / tile_stride;
    let mut res = vec![];
    for i in 0..num_tiles {
        res.push(input.i((tile_stride * i..(tile_stride * i) + tile_size, ..))?);
    }
    if remainder > 0 {
        res.push(input.i((tile_stride * num_tiles.., ..))?);
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use candle::{IndexOp, Tensor};

    use super::{tc_conv1d, tile_conv_input, A_T, B_T, G};

    /// For some reason candle's conv1d implementation returns the wrong output
    /// for groups > 1 in test so here's a reimplementation of it for
    /// c_in == groups
    /// There doesn't seem to be a problem with it at runtime though
    fn candle_conv1d(
        input: &Tensor,
        kernel: &Tensor,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> candle::Result<Tensor> {
        let (_c_out, c_in_k, k_size) = kernel.dims3()?;
        let (b_size, c_in, l_in) = input.dims3()?;
        if c_in != groups {
            candle::bail!("c_in must match number of groups")
        }

        let blocks = (0..groups)
            .map(|i| {
                let v = input.i((.., i, ..))?.reshape((b_size, 1, l_in))?;
                let k = kernel.i((i, .., ..))?.reshape((1, c_in_k, k_size))?;
                v.conv1d(&k, padding, stride, dilation, 1)
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Tensor::cat(&blocks, 1)
    }

    #[test]
    fn test_tc_conv1d() -> candle::Result<()> {
        let device = &candle::Device::Cpu;
        let x = Tensor::new(
            &[
                [1f32, 2., 3., 4., 5.],
                [1.1, 2.1, 3.1, 4.1, 5.1],
                [1.2, 2.2, 3.2, 4.2, 5.2],
                [1.3, 2.3, 3.3, 4.3, 5.3],
                [1.4, 2.4, 3.4, 4.4, 5.4],
                [1.5, 2.5, 3.5, 4.5, 5.5],
                [1.6, 2.6, 3.6, 4.6, 5.6],
                [1.7, 2.7, 3.7, 4.7, 5.7],
                [1.8, 2.8, 3.8, 4.8, 5.8],
                [1.9, 2.9, 3.9, 4.9, 5.9],
                [2., 3., 4., 5., 6.],
            ],
            device,
        )?;
        let (seq_len, _channels) = x.dims2()?;

        let kernels = Tensor::new(
            &[
                [0.5f32, 0.25, 0.125, 0.125, 0.125],
                [1.5, 1.25, 1.125, 1.125, 1.125],
                [2.5, 2.25, 2.125, 2.125, 2.125],
                [3.5, 3.25, 3.125, 3.125, 3.125],
            ],
            device,
        )?;
        let (kernel_size, channels) = kernels.dims2()?;

        // Compare with original candle conv1d implementation
        let x_copy = x.unsqueeze(0)?.t()?;
        let kernels_copy = kernels.unsqueeze(0)?;
        let kernel = kernels_copy.permute((2, 0, 1))?;

        let original_conv = candle_conv1d(&x_copy, &kernel, kernel_size - 1, 1, 1, channels)?
            .narrow(candle::D::Minus1, 0, seq_len)?
            .t()?
            .squeeze(0)?;

        // Perform optimized conv1d
        let at = Tensor::new(A_T, device)?;
        let bt = Tensor::new(B_T, device)?;
        let g = Tensor::new(G, device)?;
        let gk = g.matmul(&kernels)?;

        let x = x.pad_with_zeros(0, 3, 0)?;
        let conv_output = tc_conv1d(&x, &at, &bt, &gk)?;
        assert_eq!(conv_output.dims(), &[seq_len, channels]);

        println!("Original\n{}", original_conv);
        println!("Conv Output\n{}", conv_output);

        let conv_diff = (original_conv - conv_output)?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(conv_diff.abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_tile_conv_input() -> candle::Result<()> {
        // Simulate data with padding already added
        let x = Tensor::new(
            &[
                [1f32, 2., 3., 4.],
                [1.1, 2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2, 4.2],
                [1.3, 2.3, 3.3, 4.3],
                [1.4, 2.4, 3.4, 4.4],
                [1.5, 2.5, 3.5, 4.5],
                [1.6, 2.6, 3.6, 4.6],
                [1.7, 2.7, 3.7, 4.7],
                [1.8, 2.8, 3.8, 4.8],
                [1.9, 2.9, 3.9, 4.9],
            ],
            &candle::Device::Cpu,
        )?;
        let x = x.pad_with_zeros(0, 3, 0)?;

        let views = tile_conv_input(&x, 5, 2)?;

        assert_eq!(views.len(), 5);
        assert_eq!(
            views[0].to_vec2::<f32>()?,
            &[
                [0f32, 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 2., 3., 4.],
                [1.1, 2.1, 3.1, 4.1],
            ],
        );
        assert_eq!(
            views[1].to_vec2::<f32>()?,
            &[
                [0f32, 0., 0., 0.],
                [1., 2., 3., 4.],
                [1.1, 2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2, 4.2],
                [1.3, 2.3, 3.3, 4.3],
            ],
        );
        Ok(())
    }

    #[test]
    fn test_tile_conv_input_odd_size() -> candle::Result<()> {
        // Simulate data with padding already added
        let x = Tensor::new(
            &[
                [1f32, 2., 3., 4.],
                [1.1, 2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2, 4.2],
                [1.3, 2.3, 3.3, 4.3],
                [1.4, 2.4, 3.4, 4.4],
                [1.5, 2.5, 3.5, 4.5],
                [1.6, 2.6, 3.6, 4.6],
                [1.7, 2.7, 3.7, 4.7],
                [1.8, 2.8, 3.8, 4.8],
                [1.9, 2.9, 3.9, 4.9],
                [2., 3., 4., 5.],
            ],
            &candle::Device::Cpu,
        )?;
        let x = x.pad_with_zeros(0, 3, 0)?;

        let views = tile_conv_input(&x, 5, 2)?;

        assert_eq!(views.len(), 6);
        assert_eq!(
            views[0].to_vec2::<f32>()?,
            &[
                [0f32, 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 2., 3., 4.],
                [1.1, 2.1, 3.1, 4.1],
            ],
        );
        assert_eq!(
            views[5].to_vec2::<f32>()?,
            &[
                [1.7f32, 2.7, 3.7, 4.7],
                [1.8, 2.8, 3.8, 4.8],
                [1.9, 2.9, 3.9, 4.9],
                [2., 3., 4., 5.],
            ],
        );
        Ok(())
    }

    #[test]
    fn test_tile_conv_input_large_tile() -> candle::Result<()> {
        // Simulate data with padding already added
        let x = Tensor::new(
            &[
                [1f32, 2., 3., 4.],
                [1.1, 2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2, 4.2],
                [1.3, 2.3, 3.3, 4.3],
                [1.4, 2.4, 3.4, 4.4],
                [1.5, 2.5, 3.5, 4.5],
                [1.6, 2.6, 3.6, 4.6],
                [1.7, 2.7, 3.7, 4.7],
                [1.8, 2.8, 3.8, 4.8],
                [1.9, 2.9, 3.9, 4.9],
            ],
            &candle::Device::Cpu,
        )?;
        let x = x.pad_with_zeros(0, 3, 0)?;

        let views = tile_conv_input(&x, 10, 7)?;

        assert_eq!(views.len(), 2);
        assert_eq!(
            views[0].to_vec2::<f32>()?,
            &[
                [0f32, 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 2., 3., 4.],
                [1.1, 2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2, 4.2],
                [1.3, 2.3, 3.3, 4.3],
                [1.4, 2.4, 3.4, 4.4],
                [1.5, 2.5, 3.5, 4.5],
                [1.6, 2.6, 3.6, 4.6],
            ],
        );
        assert_eq!(
            views[1].to_vec2::<f32>()?,
            &[
                [1.4f32, 2.4, 3.4, 4.4],
                [1.5, 2.5, 3.5, 4.5],
                [1.6, 2.6, 3.6, 4.6],
                [1.7, 2.7, 3.7, 4.7],
                [1.8, 2.8, 3.8, 4.8],
                [1.9, 2.9, 3.9, 4.9],
            ],
        );
        Ok(())
    }
}

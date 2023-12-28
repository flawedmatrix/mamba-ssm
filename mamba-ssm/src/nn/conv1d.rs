use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1dConfig, VarBuilder};

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
) -> Result<Conv1d> {
    let span = tracing::span!(
        tracing::Level::TRACE,
        "conv1d",
        in_channels,
        out_channels,
        kernel_size
    );
    if in_channels == out_channels && cfg.groups == out_channels && kernel_size == 4 {
        let inner = TCConv1d::new(in_channels, out_channels, kernel_size, cfg, vs)?;
        Ok(Conv1d {
            inner: Conv1dImpl::TC(inner),
            span,
        })
    } else {
        let inner = candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, vs)?;
        Ok(Conv1d {
            inner: Conv1dImpl::Direct(inner),
            span,
        })
    }
}

/// Winograd transform matrices for convolution F(2,4) evaluated at points
/// [0, 0.5, -1.5, 1.5]
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
}

impl TCConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        cfg: Conv1dConfig,
        vb: VarBuilder,
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

        let at = Tensor::new(A_T, vb.device())?;
        let bt = Tensor::new(B_T, vb.device())?;
        // Save the result of multiplying G and the stack of kernels since we
        // don't need to use the kernel weights directly anymore
        let g = Tensor::new(G, vb.device())?;
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
        })
    }
}

impl Module for TCConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Assume out_channels == in_channels
        let (b_size, l_in, c_in) = x.dims3()?;

        let batches = (0..b_size)
            .map(|i| {
                let view = x.i((i, .., ..))?;
                let output = tc_conv1d(&view, self.cfg.padding, &self.at, &self.bt, &self.gk)?;
                output.reshape((1, l_in, c_in))
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let output = Tensor::cat(&batches, 0)?;

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

/// Given an LxD tensor, applies Toom-Cook 1D convolution on it via the
/// provided transformation matrices.
/// groups == in_channels == /// out_channels,
fn tc_conv1d(
    input: &Tensor,
    padding: usize,
    at: &Tensor,
    bt: &Tensor,
    gk: &Tensor,
) -> Result<Tensor> {
    // TODO: Support convolutions other than F(2,4)
    let (seq_len, _) = input.dims2()?;
    let padding = if seq_len % 2 == 1 {
        padding + 1
    } else {
        padding
    };
    let view = input.pad_with_zeros(0, padding, 0)?;

    let tiles = tile_conv_input(&view, 5, 2)?;
    let outputs = tiles
        .iter()
        .map(|tile| {
            let btd = bt.matmul(&tile)?;
            let elemwise_prod = gk.mul(&btd)?;
            at.matmul(&elemwise_prod)
        })
        .collect::<Result<Vec<_>>>()?;
    let output = Tensor::cat(&outputs, 0)?;
    if seq_len % 2 == 1 {
        Ok(output.narrow(0, 1, seq_len)?)
    } else {
        Ok(output)
    }
}

/// Given a LxD tensor, returns a vec of tensor views each with length
/// tile_size, advancing with tile_stride. Assumes input is already left-padded
/// as desired
fn tile_conv_input(input: &Tensor, tile_size: usize, tile_stride: usize) -> Result<Vec<Tensor>> {
    let length = input.dims()[0];
    if (length - tile_size) % tile_stride > 0 {
        candle::bail!("Tensor of length {length} can't be cleanly tiled with tile_size {tile_size} and stride {tile_stride}");
    }
    let num_tiles = (length - tile_size) / tile_stride + 1;
    let mut res = vec![];
    for i in 0..num_tiles {
        res.push(input.i((2 * i..(2 * i) + tile_size, ..))?);
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use candle::{IndexOp, Tensor};

    use super::{tc_conv1d, tile_conv_input, A_T, B_T, G};

    /// For some reason candle's conv1d implementation returns the wrong
    /// output for groups > 1, so here's a reimplementation of it for
    /// c_in == groups
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

        // TODO: There's something horribly wrong with the outer loop of
        // conv1d for some reason, so this is replaced with another
        // outer loop implementation.
        // let original_conv = x_copy
        //     .conv1d(&kernel, kernel_size - 1, 1, 1, channels)?
        //     .narrow(candle::D::Minus1, 0, seq_len)?
        //     .t()?
        //     .squeeze(0)?;
        let original_conv = candle_conv1d(&x_copy, &kernel, kernel_size - 1, 1, 1, channels)?
            .narrow(candle::D::Minus1, 0, seq_len)?
            .t()?
            .squeeze(0)?;

        // Perform optimized conv1d
        let at = Tensor::new(A_T, device)?;
        let bt = Tensor::new(B_T, device)?;
        let g = Tensor::new(G, device)?;
        let gk = g.matmul(&kernels)?;

        let conv_output = tc_conv1d(&x, kernel_size - 1, &at, &bt, &gk)?;
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
}

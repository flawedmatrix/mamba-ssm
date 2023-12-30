/// This is mostly an adaptation of the example mamba implementation found at
/// https://github.com/huggingface/candle/tree/37c539f2b7dfc8aa67a10b611dc12e5e0428be00/candle-examples/examples/mamba-minimal
/// which is itself based on
/// https://github.com/johnma2006/mamba-minimal/blob/master/model.py
/// Simple, minimal implementation of Mamba in one file of PyTorch.
///
/// Refer to the paper for details:
/// Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
/// https://arxiv.org/abs/2312.00752
///
use candle::{Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

use crate::context::Context;
use crate::nn::selective_ssm::SSM;
use crate::nn::{conv1d, linear_no_bias, Conv1d, Linear};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    /// The model dimension (number of channels in the input sequence)
    d_model: usize,
    n_layer: usize,
    vocab_size: usize,
    pad_vocab_size_multiple: usize,
}

impl Config {
    fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple;
        (self.vocab_size + pad - 1) / pad * pad
    }

    /// The rank of Δ
    fn dt_rank(&self) -> usize {
        (self.d_model + 15) / 16
    }

    /// Size of the conv1d kernel that's built into the model, but isn't in the
    /// config.json
    fn d_conv(&self) -> usize {
        4
    }

    /// N, the SSM state dimension
    fn d_state(&self) -> usize {
        16
    }

    /// This is the model dimension multiplied by the expansion factor 2.
    fn d_inner(&self) -> usize {
        self.d_model * 2
    }
}

#[derive(Clone, Debug)]
pub struct MambaBlock {
    in_proj: Linear,
    conv1d: Conv1d,
    ssm: SSM,
    out_proj: Linear,
}

impl MambaBlock {
    pub fn new(cfg: &Config, vb: VarBuilder, ctx: Context) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let d_conv = cfg.d_conv();

        let d_state = cfg.d_state();
        let dt_rank = cfg.dt_rank();

        let in_proj = linear_no_bias(cfg.d_model, d_inner * 2, vb.pp("in_proj"))?;
        let conv_cfg = candle_nn::Conv1dConfig {
            groups: d_inner,
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv1d = conv1d(
            d_inner,
            d_inner,
            d_conv,
            conv_cfg,
            vb.pp("conv1d"),
            ctx.pp("conv1d"),
        )?;
        let ssm = SSM::new(d_inner, d_state, dt_rank, vb.clone(), ctx.pp("ssm"))?;
        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            conv1d,
            ssm,
            out_proj,
        })
    }
}

impl Module for MambaBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Project the input to two larger inputs as per the Mamba
        // architecture in Figure 3 of the paper.
        let xs_and_res = xs.apply(&self.in_proj)?.chunk(2, D::Minus1)?;
        let (xs, res) = (&xs_and_res[0], &xs_and_res[1]);

        // Conv1d -> SiLU -> SSM
        let xs = xs.apply(&self.conv1d)?;
        let xs = candle_nn::ops::silu(&xs)?;
        let xs = xs.apply(&self.ssm)?;

        // Multiplicative gate
        let ys = (xs * candle_nn::ops::silu(res))?;
        ys.apply(&self.out_proj)
    }
}

#[derive(Clone, Debug)]
pub struct ResMambaBlock {
    mixer: MambaBlock,
    norm: RmsNorm,
}

impl ResMambaBlock {
    pub fn new(cfg: &Config, vb: VarBuilder, ctx: Context) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = MambaBlock::new(cfg, vb.pp("mixer"), ctx.pp("mamba"))?;
        Ok(Self { mixer, norm })
    }
}

impl Module for ResMambaBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.norm)?.apply(&self.mixer)? + xs
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<ResMambaBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder, ctx: Context) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.n_layer {
            let layer = ResMambaBlock::new(cfg, vb_l.pp(layer_idx), ctx.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm_f = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        let lm_head = Linear::from_weights(embedding.embeddings().clone(), None);
        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
        })
    }
}

impl Module for Model {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;
        let mut xs = self.embedding.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm_f)?
            .apply(&self.lm_head)
    }
}

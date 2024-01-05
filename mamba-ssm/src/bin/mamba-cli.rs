use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle::{DType, Device};
use clap::{ArgGroup, Parser};

use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use rand::Rng;

use mamba_ssm::{
    context::Context,
    model::{Config, Model},
    TextGeneration,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[clap(group(ArgGroup::new("prompt_options")
                .required(true)
                .args(&["prompt", "prompt_file"])
))]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long, short = 'p')]
    prompt: Option<String>,

    #[arg(long, short = 'f')]
    prompt_file: Option<PathBuf>,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value_t = 0.92)]
    top_p: f64,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 500)]
    sample_len: usize,

    #[arg(long, default_value = "./models/tokenizer.json")]
    tokenizer_file: PathBuf,

    #[arg(long, short = 'm', default_value = "./models/model.safetensors")]
    weights_file: PathBuf,

    #[arg(long, short = 'c', default_value = "./models/config.json")]
    config_file: PathBuf,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 320)]
    repeat_last_n: usize,

    /// Use BF16 floating point format for calculations instead of F32. CUDA only.
    #[arg(long)]
    bf16: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().include_args(true).build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}, num_threads: {}, cuda: {}, metal: {}, accelerate: {}, mkl: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c(),
        candle::utils::get_num_threads(),
        candle::utils::cuda_is_available(),
        candle::utils::metal_is_available(),
        candle::utils::has_accelerate(),
        candle::utils::has_mkl()
    );
    println!(
        "temp: {:.2} top-p: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.top_p, args.repeat_penalty, args.repeat_last_n
    );

    let tokenizer = Tokenizer::from_file(args.tokenizer_file).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config: Config = serde_json::from_slice(&std::fs::read(args.config_file)?)?;

    // TODO: Implement GPU-based inference
    let device = if candle::utils::cuda_is_available() {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };

    let dtype = if args.bf16 { DType::BF16 } else { DType::F32 };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[args.weights_file], dtype, &device)? };

    let ctx = Context::new(dtype, &device);

    let model = Model::new(&config, vb.pp("backbone"), ctx)?;
    println!("loaded the model in {:?}", start.elapsed());

    let seed = args.seed.unwrap_or(rand::thread_rng().gen());
    println!("generating {} tokens with seed {seed}", args.sample_len);

    let prompt = if args.prompt.is_some() {
        args.prompt.unwrap()
    } else {
        std::fs::read_to_string(args.prompt_file.unwrap())?
    };

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        seed,
        Some(args.temperature),
        Some(args.top_p),
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&prompt, args.sample_len)?;
    Ok(())
}

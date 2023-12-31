use std::str::FromStr;

use anyhow::{Error as E, Result};
use candle::{DType, Device};
use clap::Parser;

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
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 500)]
    sample_len: usize,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long, short = 'm')]
    weights_file: Option<String>,

    #[arg(long, short = 'c')]
    config_file: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
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
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => std::path::PathBuf::from_str("./models/tokenizer.json")?,
    };
    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => std::path::PathBuf::from_str("./models/config.json")?,
    };
    let weights_filename = match args.weights_file {
        Some(file) => std::path::PathBuf::from(file),
        None => std::path::PathBuf::from_str("./models/model.safetensors")?,
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;

    // TODO: Implement GPU-based inference
    let device = Device::Cpu;

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };

    let ctx = Context::new(candle::DType::F32, &device);

    let model = Model::new(&config, vb.pp("backbone"), ctx)?;
    println!("loaded the model in {:?}", start.elapsed());

    let seed = args.seed.unwrap_or(rand::thread_rng().gen());
    println!("generating {} tokens with seed {seed}", args.sample_len);

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        seed,
        Some(args.temperature),
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}

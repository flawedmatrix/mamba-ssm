#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{anyhow, Error as E, Result};

use candle::{DType, Device, Module, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

pub mod model;
use model::Model;

mod utils;
use utils::TokenOutputStream;

pub mod context;
pub mod nn;
pub mod ops;

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        if prompt.is_empty() {
            anyhow::bail!("Prompt cannot be empty");
        }
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        // Prompt processing
        let start_process = std::time::Instant::now();
        // Just process all but the last token of the prompt to give the model
        // the last token to work with when generation starts
        let (last, tokens_m1) = tokens
            .split_last()
            .ok_or(anyhow!("Tokens cannot be empty"))?;
        let mut next_token = *last;
        let input = Tensor::new(tokens_m1, &self.device)?.unsqueeze(0)?;
        let _ = self.model.forward(&input)?;
        let dt = start_process.elapsed();
        let prompt_len = tokens.len() - 1;
        println!(
            "\nPrompt processing time ({} tokens at {:.2} token/s)",
            prompt_len,
            prompt_len as f64 / dt.as_secs_f64(),
        );
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        let start_gen = std::time::Instant::now();
        for _ in 0..sample_len {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);

            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

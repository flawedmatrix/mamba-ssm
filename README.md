# mamba-ssm

Optimized inference-only implementation of [Mamba](https://arxiv.org/abs/2312.00752) based on the
[mamba-minimal candle-example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/mamba-minimal)
and @johnma2006's [mamba-minimal](https://github.com/johnma2006/mamba-minimal)

## Description
The primary goal of this project is to provide an inference backend that can run
Mamba on an Apple Silicon Macbook without having dependencies on CUDA entangled
with the code. The initial development specifically targets CPU-only as a
first-class citizen, with linear algebra routines supported by Accelerate or
Intel MKL.

The main dependency of this project is [Candle](https://github.com/huggingface/candle),
so supported platforms are mainly decided by their implementation in that
framework.

### Supported Platforms

- [x] CPU
- [x] Accelerate framework (via `--features accelerate`)
- [ ] Intel MKL (via `features mkl`)
  - It probably works, but I haven't tested it yet
- [ ] Metal
  - It does speed up inference by a lot, but the output is garbage. Probably because implementation in candle isn't stable yet.
- [ ] CUDA

### Supported Features

- [x] Inference via CLI
- [ ] FP16 support (coming soon)
- [ ] Quantized models
- [ ] Web interface for inference

## Getting Started

1. Prepare a Mamba safetensors model, config.json, and tokenizer.json.
   - For example [mamba-2.8b-slimpj](https://huggingface.co/state-spaces/mamba-2.8b-slimpj/tree/refs%2Fpr%2F1) and the tokenizer from [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b/blob/main/tokenizer.json)
2. Move these three files into the `models` folder.
3. Install Rust (https://www.rust-lang.org/), then run
```bash
$ cargo build --release
$ targets/release/mamba-cli --prompt "Mamba is the"
```

You can also specify the model and config.json used by passing flags:
```bash
targets/release/mamba-cli -m models/mamba-2.8b-slimpj/model.safetensors -c models/mamba-2.8b-slimpj/config.json -prompt "Mamba is the"
```

### Building with Apple Accelerate Framework support
```bash
$ cargo build --release --features accelerate
```

### Building with Intel MKL framework support
```bash
$ cargo build --release --features mkl
```
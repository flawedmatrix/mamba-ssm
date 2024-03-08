# mamba-ssm

Optimized inference-only implementation of [Mamba](#references) [1] written in Rust.

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
- [x] Metal
- [x] CUDA (via `features cuda`)
  - It works but no optimization was done for CUDA yet.

### Supported Features

- [x] Inference via CLI
- [ ] FP16 support (coming soon)
- [ ] Quantized models
- [ ] Web interface for inference

## Getting Started

1. Prepare a Mamba safetensors model, config.json, and tokenizer.json and move these to the `/.models` directory.
   - Run `./download.sh` to download [mamba-2.8b-slimpj](https://huggingface.co/state-spaces/mamba-2.8b-slimpj/tree/refs%2Fpr%2F1) and the tokenizer from [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b/blob/main/tokenizer.json)
2. [Install Rust](https://www.rust-lang.org), then run:

```bash
cargo build --release
target/release/mamba-cli --prompt "Mamba is the"
```

You can also specify the model and config.json used by passing flags:

```bash
target/release/mamba-cli -m models/mamba-2.8b-slimpj/model.safetensors -c models/mamba-2.8b-slimpj/config.json -prompt "Mamba is the"
```

For other usage options such as passing the prompt by file, see the usage:

```bash
target/release/mamba-cli --help
```

### Building with Apple Accelerate Framework support

```bash
cargo build --release --features accelerate
```

### Building with Intel MKL framework support

```bash
cargo build --release --features mkl
```

### Generation speed with CPU

Currently, with the Mamba 2.8b model, it generates at about 6.5 tokens/s with FP32 on CPU only on a M3 Max MBP.

```bash
$ target/release/mamba-cli --temperature 0 -n 50 -f prompt.txt
avx: false, neon: true, simd128: false, f16c: false, num_threads: 16, cuda: false, metal: false, accelerate: true, mkl: false
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
loaded the model in 1.605674125s
generating 50 tokens with seed 16889006583945703583

Prompt processing time (98 tokens at 24.68 token/s)
I am that merry wanderer of the night.
I jest to Oberon and make him smile
When I a fat and bean-fed horse beguile,
Neighing in likeness of a filly foal:
And sometime lurk I in a gossip’s bowl,
In very likeness of a roasted crab,
And when she drinks, against her lips I bob
And on her wither’d dewlap pour the ale.
I am that merry jester of the night;
When he is sick and sad, I make him smile:
If his wife be angry with him, then I
Make him laugh, as if a fool were free.
But when she
50 tokens generated (6.50 token/s)
```

## References

[1] "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    Albert Gu and Tri Dao
    https://arxiv.org/abs/2312.00752

[2] "The Annotated S4"
    Sasha Rush and Sidd Karamcheti
    https://srush.github.io/annotated-s4

[3] "Error Analysis and Improving the Accuracy of Winograd Convolution for Deep Neural Networks"
    Barbara Barabasz, Andrew Anderson, Kirk M. Soodhalter, David Gregg
    https://arxiv.org/abs/1803.10986

[4] "Winograd Convolution for Deep Neural Networks: Efficient Point Selection"
    Syed Asad Alam, Andrew Anderson, Barbara Barabasz, David Gregg
    https://arxiv.org/pdf/2201.10369.pdf

### Code references
- Original implementation: https://github.com/state-spaces/mamba
- This repo was initially adapted from code from the
  mamba-minimal candle-example:
  https://github.com/huggingface/candle/tree/main/candle-examples/examples/mamba-minimal
- Instructive minimal implementation: https://github.com/johnma2006/mamba-minimal

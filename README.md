# mamba-ssm

Implementation of [Mamba](https://arxiv.org/abs/2312.00752) based heavily on the
[mamba-minimal candle-example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/mamba-minimal)
and @johnma2006's [mamba-minimal](https://github.com/johnma2006/mamba-minimal)

## Running

```bash
$ cargo run --release -- --prompt "Mamba is the"
```

### With Intel MKL framework support
```bash
$ cargo run --release --features mkl -- --prompt "Mamba is the"
```

### With Apple Accelerate framework support
```bash
$ cargo run --release --features accelerate -- --prompt "Mamba is the"
```
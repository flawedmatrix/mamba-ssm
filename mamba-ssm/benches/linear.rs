use criterion::{criterion_group, criterion_main, Criterion};

use candle::{Device, Module, Tensor};

fn candle_gemm_mul(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (10, 2000), &device).unwrap();
    let y = Tensor::randn(0f32, 1., (2000, 2000), &device).unwrap();

    c.bench_function("gemm mul", |b| b.iter(|| x.matmul(&y).unwrap()));
}

fn candle_linear(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 10, 2000), &device).unwrap();

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let layer = candle_nn::linear(2000, 2000, vb).unwrap();

    c.bench_function("candle_nn::linear single batch", |b| {
        b.iter(|| layer.forward(&x).unwrap())
    });
}

fn candle_linear_no_bias(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 10, 2000), &device).unwrap();

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let layer = candle_nn::linear_no_bias(2000, 2000, vb).unwrap();

    c.bench_function("candle_nn::linear_no_bias single batch", |b| {
        b.iter(|| layer.forward(&x).unwrap())
    });
}

fn custom_linear(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 10, 2000), &device).unwrap();

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let layer = mamba_ssm::nn::linear(2000, 2000, vb).unwrap();

    c.bench_function("mamba_ssm::nn::linear single batch", |b| {
        b.iter(|| layer.forward(&x).unwrap())
    });
}

fn custom_linear_no_bias(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 10, 2000), &device).unwrap();

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let layer = mamba_ssm::nn::linear_no_bias(2000, 2000, vb).unwrap();

    c.bench_function("mamba_ssm::nn::linear_no_bias single batch", |b| {
        b.iter(|| layer.forward(&x).unwrap())
    });
}

criterion_group!(
    linear_benches,
    candle_gemm_mul,
    candle_linear,
    candle_linear_no_bias,
    custom_linear,
    custom_linear_no_bias,
);
criterion_main!(linear_benches);

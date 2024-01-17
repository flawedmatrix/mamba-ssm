use criterion::{criterion_group, criterion_main, Criterion};

use candle::{Device, Module, Tensor};
use mamba_ssm::context::Context;

fn candle_conv1d(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 2000, 5120), &device).unwrap();

    let conv_cfg = candle_nn::Conv1dConfig {
        groups: 5120,
        padding: 3,
        ..Default::default()
    };

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let candle_conv1d = candle_nn::conv1d(5120, 5120, 4, conv_cfg, vb).unwrap();

    c.bench_function("candle_nn::conv1d single batch", |b| {
        b.iter(|| candle_conv1d.forward(&x.t().unwrap()).unwrap().t().unwrap())
    });
}

fn custom_conv1d(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 2000, 5120), &device).unwrap();

    let conv_cfg = candle_nn::Conv1dConfig {
        groups: 5120,
        padding: 3,
        ..Default::default()
    };

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let ctx = Context::new(candle::DType::F32, &device);
    ctx.freeze();

    let custom_conv1d = mamba_ssm::nn::conv1d(5120, 5120, 4, conv_cfg, vb, ctx).unwrap();
    c.bench_function("mamba_ssm::nn::conv1d single batch", |b| {
        b.iter(|| custom_conv1d.forward(&x).unwrap())
    });
}

fn candle_conv1d_inference(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 1, 5120), &device).unwrap();

    let conv_cfg = candle_nn::Conv1dConfig {
        groups: 5120,
        padding: 3,
        ..Default::default()
    };

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let candle_conv1d = candle_nn::conv1d(5120, 5120, 4, conv_cfg, vb).unwrap();

    c.bench_function("candle_nn::conv1d inference", |b| {
        b.iter(|| candle_conv1d.forward(&x.t().unwrap()).unwrap().t().unwrap())
    });
}

fn custom_conv1d_inference(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 1, 5120), &device).unwrap();
    let cached = Tensor::randn(0f32, 1., (3, 5120), &device).unwrap();

    let conv_cfg = candle_nn::Conv1dConfig {
        groups: 5120,
        padding: 3,
        ..Default::default()
    };

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let ctx = Context::new(candle::DType::F32, &device);
    let mut ctx = ctx.pp(0);
    _ = ctx.get((3, 5120), "ph").unwrap();
    ctx.set("ph", cached).unwrap();
    ctx.freeze();

    let custom_conv1d = mamba_ssm::nn::conv1d(5120, 5120, 4, conv_cfg, vb, ctx).unwrap();
    c.bench_function("mamba_ssm::nn::conv1d inference", |b| {
        b.iter(|| custom_conv1d.forward(&x).unwrap())
    });
}

criterion_group!(
    conv1d_benches,
    candle_conv1d,
    custom_conv1d,
    candle_conv1d_inference,
    custom_conv1d_inference
);
criterion_main!(conv1d_benches);

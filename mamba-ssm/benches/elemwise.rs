use criterion::{criterion_group, criterion_main, Criterion};

use candle::{Device, Tensor};

use mamba_ssm::ops::elemwise;

fn candle_elemwise_mul(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();
    let y = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();

    c.bench_function("elemwise mul", |b| b.iter(|| x.mul(&y).unwrap()));
}

fn candle_elemwise_add(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();
    let y = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();

    c.bench_function("elemwise add", |b| b.iter(|| x.add(&y).unwrap()));
}

fn custom_elemwise_add(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();
    let y = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();

    c.bench_function("custom elemwise add", |b| {
        b.iter(|| x.apply_op2(&y, elemwise::BinaryOp::Add).unwrap())
    });
}

fn custom_elemwise_mul(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();
    let y = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();

    c.bench_function("custom elemwise mul", |b| {
        b.iter(|| x.apply_op2(&y, elemwise::BinaryOp::Mul).unwrap())
    });
}

criterion_group!(
    elemwise_benches,
    candle_elemwise_mul,
    candle_elemwise_add,
    custom_elemwise_mul,
    custom_elemwise_add,
);
criterion_main!(elemwise_benches);

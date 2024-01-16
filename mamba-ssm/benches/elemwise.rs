use criterion::{criterion_group, criterion_main, Criterion};

use candle::{Device, Tensor};

fn candle_elemwise_mul(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();
    let y = Tensor::randn(0f32, 1., (5000, 5000), &device).unwrap();

    c.bench_function("elemwise mul", |b| b.iter(|| x.mul(&y).unwrap()));
}

criterion_group!(elemwise_benches, candle_elemwise_mul);
criterion_main!(elemwise_benches);

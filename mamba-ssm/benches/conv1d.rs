use criterion::{criterion_group, criterion_main, Criterion};

use candle::{Device, Module, Tensor};

fn conv1d(c: &mut Criterion) {
    let device = Device::Cpu;

    let x = Tensor::randn(0f32, 1., (1, 2000, 5120), &device).unwrap();

    let conv_cfg = candle_nn::Conv1dConfig {
        groups: 5120,
        padding: 3,
        ..Default::default()
    };

    let var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, candle::DType::F32, &device);

    let candle_conv1d = candle_nn::conv1d(5120, 5120, 4, conv_cfg, vb.clone()).unwrap();
    let custom_conv1d = mamba_ssm::nn::conv1d(5120, 5120, 4, conv_cfg, vb).unwrap();
    c.bench_function("candle_nn::conv1d single batch", |b| {
        b.iter(|| candle_conv1d.forward(&x.t().unwrap()).unwrap().t().unwrap())
    });
    c.bench_function("mamba_ssm::nn::conv1d single batch", |b| {
        b.iter(|| custom_conv1d.forward(&x).unwrap())
    });
}

criterion_group!(benches, conv1d);
criterion_main!(benches);

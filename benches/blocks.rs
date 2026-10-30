//! Throughput of the main blocks (`cargo bench -p comrs`).
//!
//! The workflow `validation.yml` runs them when started by hand with its benchmarks
//! switch; their results are not compared with a threshold.

#![allow(missing_docs, clippy::expect_used, reason = "benchmark setup")]

use comrs::channel::optical::fiber::{FiberParams, Ssfm};
use comrs::dsp::filter::{PulseShape, fir_filter, pulse_shape};
use comrs::prelude::*;
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn mapping(c: &mut Criterion) {
    let qam = Constellation::qam(64).expect("square order").normalized();
    let mut rng = Rng::new(1);
    let bits = source::bits(6 * 4096, &mut rng);
    let x = qam.modulate(&bits).expect("whole symbols");
    let y = awgn::awgn_ebn0(&x, 12.0, 6.0, 1.0, &mut rng);
    let mut g = c.benchmark_group("mapping");
    g.throughput(Throughput::Elements(x.len() as u64));
    g.bench_function("qam64_decide", |b| b.iter(|| qam.decide(black_box(&y))));
    g.bench_function("qam64_llr_exact", |b| b.iter(|| qam.llr(black_box(&y), 0.01, None, false)));
    g.bench_function("qam64_llr_maxlog", |b| b.iter(|| qam.llr(black_box(&y), 0.01, None, true)));
    g.finish();
}

fn dsp(c: &mut Criterion) {
    let h = pulse_shape(PulseShape::Rrc { rolloff: 0.1 }, 4, 257).expect("positive sps and taps");
    let mut rng = Rng::new(2);
    let x = rng.complex_noise(1 << 16, 1.0);
    let mut g = c.benchmark_group("dsp");
    g.throughput(Throughput::Elements(x.len() as u64));
    g.bench_function("rrc_fir_257_taps", |b| b.iter(|| fir_filter(black_box(&h), black_box(&x))));
    // 64 OFDM symbols of 1024 bins, there and back: the per-call cost of the cached plans
    let plan = comrs::math::fft::plan(1024);
    let mut y = x.clone();
    g.bench_function("fft_ifft_1024_x64", |b| {
        b.iter(|| {
            for s in y.chunks_exact_mut(1024) {
                plan.forward(black_box(s));
                plan.inverse(black_box(s));
            }
        });
    });
    g.finish();
}

fn channel(c: &mut Criterion) {
    let ssfm = Ssfm::new(FiberParams::default(), 1.0);
    let mut rng = Rng::new(3);
    let x = rng.complex_noise(1 << 14, 1e-3);
    let mut g = c.benchmark_group("channel");
    g.sample_size(10);
    g.throughput(Throughput::Elements(x.len() as u64));
    g.bench_function("ssfm_80km_1km_steps", |b| b.iter(|| ssfm.propagate(black_box(&x), 64e9, None)));
    g.bench_function("ssfm_80km_1km_steps_f32", |b| b.iter(|| ssfm.propagate_f32(black_box(&x), 64e9, None)));
    g.finish();
}

criterion_group!(benches, mapping, dsp, channel);
criterion_main!(benches);

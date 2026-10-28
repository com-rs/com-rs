//! DAC and ADC (`comrs::hardware::converter`) against the closed forms of their noise:
//!
//! - the SQNR of a full-scale sine, $6.02 N + 1.76$ dB (Bennett, *Bell Syst. Tech. J.*
//!   27(3), 1948);
//! - the mean squared error of a Gaussian input against its exact integral, granular and
//!   overload cells, at several loading factors;
//! - the optimal uniform quantiser of a Gaussian input against the steps and errors of
//!   J. Max, "Quantizing for minimum distortion," *IRE Trans. Inf. Theory*, 6(1), 1960
//!   (to be checked against the paper: `references/README.md`, status needed);
//! - the effective number of bits from the SINAD of a full-scale sine (IEEE Std 1241);
//! - the SNR of a sine sampled with clock jitter, $-20 \log_{10}(2\pi f \sigma_{j})$
//!   (Walden, *IEEE J. Sel. Areas Commun.* 17(4), 1999);
//! - the response of the zero-order hold and of the analogue bandwidth;
//! - an OFDM link through an ADC: the BER of 16-QAM at the SNR of the channel and of the
//!   quantisation noise together.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::{MIN_ERRORS, within_three_sigma};
use comrs::hardware::converter::{
    Adc, Dac, Quantizer, enob_from_sinad_db, gaussian_mse, jitter_snr_db, optimal_loading, sqnr_sine_db,
    zero_order_hold_gain,
};
use comrs::mapping::constellation::ConstellationKind;
use comrs::mapping::theory::ber;
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use std::f64::consts::PI;

/// A complex exponential of amplitude `a` at `f` (cycles per sample), `n` samples.
fn tone(n: usize, f: f64, a: f64) -> Vec<C64> {
    (0..n).map(|k| C64::from_polar(a, 2.0 * PI * f * k as f64)).collect()
}

/// The ratio (dB) of the power of the tone `f` in `y` to the power of the rest.
fn tone_snr_db(y: &[C64], f: f64) -> f64 {
    let n = y.len() as f64;
    // the least-squares amplitude of the tone
    let c: C64 = y.iter().enumerate().map(|(k, v)| v * C64::from_polar(1.0, -2.0 * PI * f * k as f64)).sum::<C64>() / n;
    let signal = c.norm_sqr();
    let rest = y
        .iter()
        .enumerate()
        .map(|(k, v)| (v - c * C64::from_polar(1.0, 2.0 * PI * f * k as f64)).norm_sqr())
        .sum::<f64>()
        / n;
    10.0 * (signal / rest).log10()
}

#[test]
fn sqnr_of_a_full_scale_sine() {
    // a complex tone: two sines in quadrature. The error of one deterministic sine is not
    // exactly uniform (the SQNR of one tone strays by ±0.5 dB from the formula at some
    // frequencies and amplitudes); averaged over tones it is Δ²/12, so for an amplitude a
    // the SQNR is 6 a²/Δ² (1.5 · 4^N at full scale)
    let mut rng = Rng::new(1);
    for bits in 6..=12u32 {
        let q = Quantizer::new(bits, 1.0).unwrap();
        let (mut signal, mut noise) = (0.0, 0.0);
        for _ in 0..24 {
            let f = 0.01 + 0.48 * rng.uniform();
            let a = 0.9 + 0.1 * rng.uniform() - 1e-9;
            let mut y = tone(1 << 14, f, a);
            q.apply(&mut y);
            let r = 10f64.powf(tone_snr_db(&y, f) / 10.0);
            signal += a * a;
            noise += a * a / r;
        }
        let snr = 10.0 * (signal / noise).log10();
        let theory = sqnr_sine_db(bits) + 10.0 * (signal / 24.0).log10();
        assert!((snr - theory).abs() < 0.15, "{bits} bits: {snr} dB vs {theory}");
    }
}

#[test]
fn mse_of_a_gaussian_input_at_several_loadings() {
    // the clipping of a few samples beyond 4σ is a sizeable part of the error at large
    // loadings: four million samples keep the mean within about 1 %
    let mut rng = Rng::new(3);
    let x: Vec<f64> = (0..4_000_000).map(|_| rng.gaussian()).collect();
    for bits in [3u32, 5, 8] {
        for loading in [1.5, 2.5, 4.0, optimal_loading(bits)] {
            let q = Quantizer::new(bits, loading).unwrap();
            let sim = x.iter().map(|v| (q.quantize(*v) - v).powi(2)).sum::<f64>() / x.len() as f64;
            let exact = gaussian_mse(&q, 1.0);
            assert!((sim / exact - 1.0).abs() < 0.02, "{bits} bits, loading {loading}: {sim} vs {exact}");
        }
    }
}

#[test]
fn optimal_uniform_quantizers_of_max_1960() {
    // levels, step and mean squared error for a unit-variance Gaussian input
    for (bits, step, mse) in [(1u32, 1.596, 0.3634), (2, 0.9957, 0.1188), (3, 0.5860, 0.03744), (4, 0.3352, 0.01154)] {
        let q = Quantizer::new(bits, optimal_loading(bits)).unwrap();
        assert!((q.step() / step - 1.0).abs() < 1e-3, "{bits} bits: step {}", q.step());
        assert!((gaussian_mse(&q, 1.0) / mse - 1.0).abs() < 1e-3, "{bits} bits: mse {}", gaussian_mse(&q, 1.0));
    }
}

#[test]
fn the_enob_of_an_adc_is_that_of_its_sinad() {
    let f = 0.087_654_321;
    let mut rng = Rng::new(5);
    for (bits, enob) in [(10u32, 7.5), (12, 9.0), (8, 8.0)] {
        let adc = Adc { enob: Some(enob), ..Adc::ideal(Quantizer::new(bits, 1.0).unwrap()) };
        let y = adc.convert(&tone(1 << 16, f, 1.0 - 1e-9), 1.0, &mut rng).unwrap();
        let measured = enob_from_sinad_db(tone_snr_db(&y, f));
        assert!((measured - enob).abs() < 0.05, "{bits} bits, ENOB {enob}: {measured}");
    }
}

#[test]
fn the_snr_of_a_jittered_sine() {
    let fs = 64e9;
    let mut rng = Rng::new(7);
    for (cycles, jitter) in [(4099.0, 50e-15), (12_289.0, 100e-15)] {
        // a whole number of periods: the derivative of the periodic frame is exact
        let n = 1 << 16;
        let f = cycles / n as f64;
        let adc = Adc { jitter_s: jitter, ..Adc::ideal(Quantizer::new(24, 2.0).unwrap()) };
        let y = adc.convert(&tone(n, f, 1.0), fs, &mut rng).unwrap();
        let snr = tone_snr_db(&y, f);
        let theory = jitter_snr_db(f * fs, jitter);
        assert!((snr - theory).abs() < 0.2, "{} GHz, {jitter} s: {snr} dB vs {theory}", f * fs / 1e9);
    }
}

#[test]
fn the_responses_of_the_hold_and_of_the_bandwidth() {
    let n = 1 << 12;
    for hold in [2usize, 8] {
        let dac = Dac { hold, ..Dac::ideal(Quantizer::new(24, 2.0).unwrap()) };
        for cycles in [37.0, 501.0, 1499.0] {
            let f = cycles / n as f64; // cycles per input sample
            let y = dac.convert(&tone(n, f, 1.0), 1.0).unwrap();
            // the tone at f / hold cycles per output sample
            let fo = f / hold as f64;
            let c: C64 =
                y.iter().enumerate().map(|(k, v)| v * C64::from_polar(1.0, -2.0 * PI * fo * k as f64)).sum::<C64>()
                    / y.len() as f64;
            assert!((c.norm() - zero_order_hold_gain(f, 1.0, hold)).abs() < 1e-6, "hold {hold}, f {f}: {}", c.norm());
        }
    }
    // the analogue front end: -3 dB at its bandwidth
    let adc = Adc { bandwidth_hz: Some(0.1), ..Adc::ideal(Quantizer::new(24, 2.0).unwrap()) };
    let f = 410.0 / 4096.0;
    let y = adc.convert(&tone(4096, f, 1.0), 1.0, &mut Rng::new(0)).unwrap();
    let c: C64 =
        y.iter().enumerate().map(|(k, v)| v * C64::from_polar(1.0, -2.0 * PI * f * k as f64)).sum::<C64>() / 4096.0;
    // a Gaussian response, 1/√2 at the bandwidth
    let expected = 0.5f64.sqrt().powf((f / 0.1).powi(2));
    assert!((c.norm() / expected - 1.0).abs() < 0.01, "{} vs {expected}", c.norm());
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn an_ofdm_link_through_an_adc_has_the_ber_of_the_total_noise() {
    // 16-QAM on 256 subcarriers (unit-power samples, σ² = 1/2 per dimension), an ADC of
    // a few bits at a loading of 4 (overload negligible): the granular noise adds to the
    // noise of the channel, Gaussian after the FFT
    let c = Constellation::qam(16).unwrap().normalized();
    let ofdm = Ofdm { oversampling: 1, ..Ofdm::new(256, 256).unwrap() };
    for (bits, snr_db) in [(5u32, 25.0), (6, 22.0)] {
        let q = Quantizer::with_loading(bits, 0.5f64.sqrt(), 4.0).unwrap();
        let adc = Adc::ideal(q);
        let s2 = 1.0 / db2lin(snr_db);
        let total = s2 + 2.0 * gaussian_mse(&q, 0.5f64.sqrt());
        let theory = ber(ConstellationKind::Qam, 16, lin2db(1.0 / (4.0 * total))).unwrap();
        let (mut errors, mut bits_total, mut batch) = (0u64, 0u64, 0u64);
        while errors < MIN_ERRORS * 10 {
            let mut rng = batch_rng(11, batch);
            for _ in 0..50 {
                let b: Vec<u8> = (0..256 * 4).map(|_| rng.bit()).collect();
                let x = ofdm.modulate(&c.modulate(&b).unwrap()).unwrap();
                let y: Vec<C64> = x.iter().map(|v| v + rng.complex_gaussian() * s2.sqrt()).collect();
                let z = ofdm.demodulate(&adc.convert(&y, 1.0, &mut rng).unwrap()).unwrap();
                let d = c.demodulate(&z);
                errors += d.iter().zip(&b).filter(|(a, b)| a != b).count() as u64;
                bits_total += b.len() as u64;
            }
            batch += 1;
        }
        assert!(
            within_three_sigma(errors, bits_total, theory),
            "{bits} bits at {snr_db} dB: {errors}/{bits_total} vs {theory}"
        );
    }
}

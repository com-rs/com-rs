//! The Wiener phase noise of an oscillator (`comrs::hardware::oscillator::phase_noise`) and
//! the frequency-stability estimators (`comrs::metrics::stability`) against IEEE Std
//! 1139-2022, *IEEE Standard Definitions of Physical Quantities for Fundamental Frequency
//! and Time Metrology—Random Instabilities*, Table B.2: a phase of linewidth $\Delta\nu$
//! is a white frequency noise of one-sided density $h_{0} = \Delta\nu/\pi$ (Hz²/Hz), whose
//! Allan and modified Allan variances are $h_{0}/(2\tau)$ and $h_{0}/(4\tau)$; Table B.1:
//! the Allan deviation falls as $\tau^{-1/2}$. The Hadamard variance of eq. (A.31) is
//! $h_{0}/(2\tau)$ too, by eq. (B.2) with its transfer function
//! $\frac{16}{6}\sin^{6}(\pi\tau f)/(\pi\tau f)^{2}$ and $\int_{0}^{\infty} \sin^{6}u/u^{2}\,du =
//! 3\pi/16$: Table B.2 prints $h_{0}/(3\tau)$, which the simulation does not follow. A linewidth of
//! 100 kHz sampled at 1 GHz, 2²⁰ samples, $\tau$ from 10 to 1000 samples; within 5 %
//! (the estimates over 2²⁰ samples have a relative error below 2 % at $\tau = 1000$).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::hardware::oscillator::phase_noise;
use comrs::metrics::stability::{allan_variance, hadamard_variance, modified_allan_variance};
use comrs::prelude::*;
use std::f64::consts::PI;

#[test]
fn ieee_1139_2022_table_b2_white_frequency_noise() {
    let (linewidth, fs) = (1e5, 1e9);
    let ts = 1.0 / fs;
    let phi = phase_noise(linewidth, 1 << 20, ts, &mut Rng::new(1139)).unwrap();
    let h0 = linewidth / PI;
    for m in [10usize, 100, 1000] {
        let tau = m as f64 * ts;
        let cases = [
            ("AVAR", allan_variance(&phi, ts, m).unwrap(), h0 / (2.0 * tau)),
            ("MVAR", modified_allan_variance(&phi, ts, m).unwrap(), h0 / (4.0 * tau)),
            // eqs. (A.31) and (B.2); Table B.2 prints h0 / (3τ)
            ("HVAR", hadamard_variance(&phi, ts, m).unwrap(), h0 / (2.0 * tau)),
        ];
        for (name, got, want) in cases {
            assert!((got / want - 1.0).abs() < 0.05, "{name} at m = {m}: {got} vs {want}");
        }
    }
    // Table B.1: σ_y(τ) ∝ τ^{-1/2} for white FM
    let slope = (allan_variance(&phi, ts, 1000).unwrap() / allan_variance(&phi, ts, 10).unwrap()).log10() / 2.0 / 2.0;
    assert!((slope + 0.5).abs() < 0.03, "slope {slope}");
}

//! The frequency of a complex tone from discrete samples against D. C. Rife and R. R.
//! Boorstyn, "Single-tone parameter estimation from discrete-time observations," *IEEE
//! Trans. Inf. Theory* 20(5), 591–598, 1974: the outlier probability of the coarse DFT
//! search (eq. (59), `comrs::dsp::sync::carrier::tone_outlier_probability`) on Fig. 3, and
//! the RMS error of their maximum-likelihood estimator (`fourth_power_foe_refined` with
//! M = 1) on Fig. 4. The tests measure; the values read on the figures and their
//! tolerances are in the card `validation/cards/rife1974.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::awgn::add_noise;
use comrs::dsp::sync::carrier::{fourth_power_foe_refined, tone_outlier_probability};
use comrs::math::{bisect, cis};
use comrs::prelude::*;
use serde_json::json;
use std::f64::consts::TAU;

#[test]
fn rife_boorstyn_1974_fig_3() {
    let mut card = Card::load("rife1974").unwrap().part(&json!({"part": "outliers"})).unwrap();
    for (n, p) in
        [(16, 1e-3), (32, 1e-3), (64, 1e-3), (128, 1e-3), (256, 1e-3), (512, 1e-3), (16, 1e-7), (64, 1e-7), (256, 1e-7)]
    {
        // the SNR (dB) at which the outlier probability falls to p (it decreases with the SNR)
        let snr_db = bisect(|d| tone_outlier_probability(n, db2lin(d)) > p, -30.0, 10.0, 60);
        card.check(&json!({"part": "outliers", "n": n, "probability": p}), snr_db);
    }
    card.finish().unwrap();
}

/// The RMS error (Hz) of the estimate of a tone at 2000 Hz sampled at 4000 Hz, `n`
/// samples at `snr_db` (b0² / 2σ²), random phase; the error is taken modulo the band.
fn rms_error(n: usize, snr_db: f64, trials: usize, rng: &mut Rng) -> f64 {
    let (fs, f0) = (4000.0, 2000.0);
    let mut sq = 0.0;
    for _ in 0..trials {
        let phase = TAU * rng.uniform();
        let mut x: Vec<C64> = (0..n).map(|k| cis(TAU * f0 * k as f64 / fs + phase)).collect();
        // E|w|² = 2σ² = 1/ρ for b0 = 1
        add_noise(&mut x, 1.0 / db2lin(snr_db), rng);
        let f = fourth_power_foe_refined(&x, fs, 1).unwrap();
        let e = (f - f0 + fs / 2.0).rem_euclid(fs) - fs / 2.0;
        sq += e * e;
    }
    (sq / trials as f64).sqrt()
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn rife_boorstyn_1974_fig_4() {
    let mut rng = Rng::new(1974);
    let mut card = Card::load("rife1974").unwrap().part(&json!({"part": "rms"})).unwrap();
    for (n, snr_db) in [(16, -10.3), (16, -5.2), (16, 0.0), (16, 5.0), (128, -10.2), (64, 10.0)] {
        let v = rms_error(n, snr_db, 2000, &mut rng);
        card.check(&json!({"part": "rms", "n": n, "snr_db": snr_db}), v);
    }
    card.finish().unwrap();
}

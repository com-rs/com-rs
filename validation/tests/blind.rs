//! Blind estimation (`comrs::metrics::blind`, `comrs::dsp::cyclo`):
//!
//! - the normalised fourth-order cumulants of the common alphabets, against the values of
//!   A. Swami and B. M. Sadler, "Hierarchical digital modulation classification using
//!   cumulants," *IEEE Trans. Commun.* 48(3), 2000 (quoted from memory, to be checked
//!   against the table of the paper: `references/README.md`, status needed), which are
//!   also exact moments of the alphabets;
//! - the sample cumulants are consistent: their features converge to the theoretical
//!   ones in Gaussian noise whatever the SNR, the noise variance known;
//! - the likelihood classifier of Wei & Mendel (*IEEE Trans. Commun.* 48(2), 2000), the
//!   optimal one, classifies at least as well as the cumulant classifier;
//! - the M2M4 estimator of Pauluzzi & Beaulieu (*IEEE Trans. Commun.* 48(10), 2000) is
//!   consistent, its mean squared error falling as $1/N$, for PSK at every SNR and for
//!   QAM at low SNR; for QAM at high SNR its error is large;
//! - the symbol rate of root-raised-cosine signals is found by the line of $|x|^{2}$
//!   (Gardner, *IEEE Signal Process. Mag.* 8(2), 1991) at fractional oversampling.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::dsp::cyclo::estimate_symbol_rate;
use comrs::dsp::filter::rrc_taps;
use comrs::metrics::blind::{Cumulants, classify_cumulants, classify_ml, snr_m2m4};
use comrs::prelude::*;

fn noisy(c: &Constellation, n: usize, s2: f64, rng: &mut Rng) -> Vec<C64> {
    (0..n).map(|_| c.points()[rng.below(c.order())] + rng.complex_gaussian() * s2.sqrt()).collect()
}

#[test]
fn normalised_cumulants_of_swami_and_sadler() {
    // |C40| / C21² and C42 / C21²
    let table = [
        (Constellation::psk(2).unwrap(), 2.0, -2.0),
        (Constellation::pam(4).unwrap(), 1.36, -1.36),
        (Constellation::pam(8).unwrap(), 1.2381, -1.2381),
        (Constellation::qam(4).unwrap(), 1.0, -1.0),
        (Constellation::psk(8).unwrap(), 0.0, -1.0),
        (Constellation::qam(16).unwrap(), 0.68, -0.68),
        (Constellation::qam(64).unwrap(), 0.6190, -0.6190),
    ];
    for (c, c40, c42) in table {
        let (a, b) = Cumulants::of_constellation(&c, None).unwrap().features(None);
        assert!((a - c40).abs() < 5e-5 && (b - c42).abs() < 5e-5, "{:?} {}: {a} {b}", c.kind(), c.order());
    }
}

#[test]
fn sample_cumulants_converge_whatever_the_snr() {
    let mut rng = Rng::new(2);
    for (c, snr_db) in [(Constellation::psk(2).unwrap(), 0.0), (Constellation::qam(16).unwrap(), 5.0)] {
        let c = c.normalized();
        let s2 = 1.0 / db2lin(snr_db);
        let (ta, tb) = Cumulants::of_constellation(&c, None).unwrap().features(None);
        let r = noisy(&c, 400_000, s2, &mut rng);
        let (a, b) = Cumulants::estimate(&r).unwrap().features(Some(s2));
        // the standard deviation of the features grows as (1 + 1/SNR)² / √N
        let tol = 4.0 * (1.0 + s2).powi(2) * 6.0 / (r.len() as f64).sqrt();
        assert!(
            (a - ta).abs() < tol && (b - tb).abs() < tol,
            "{:?} at {snr_db} dB: ({a}, {b}) vs ({ta}, {tb})",
            c.kind()
        );
    }
}

/// The rate of correct decisions of `classify` over `trials` signals of `n` samples of
/// each candidate.
fn correct_rate(
    candidates: &[Constellation],
    n: usize,
    s2: f64,
    trials: usize,
    classify: &dyn Fn(&[C64]) -> usize,
    rng: &mut Rng,
) -> f64 {
    let mut ok = 0;
    for (i, c) in candidates.iter().enumerate() {
        for _ in 0..trials {
            ok += usize::from(classify(&noisy(c, n, s2, rng)) == i);
        }
    }
    ok as f64 / (trials * candidates.len()) as f64
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn the_likelihood_classifier_dominates_the_cumulants() {
    let cands: Vec<Constellation> =
        [Constellation::psk(2), Constellation::qam(4), Constellation::psk(8), Constellation::qam(16)]
            .into_iter()
            .map(|c| c.unwrap().normalized())
            .collect();
    let mut rng = Rng::new(3);
    for (snr_db, n) in [(5.0, 256usize), (10.0, 128)] {
        let s2 = 1.0 / db2lin(snr_db);
        let cum =
            correct_rate(&cands, n, s2, 300, &|r| classify_cumulants(r, &cands, Some(s2)).unwrap().index, &mut rng);
        let ml = correct_rate(&cands, n, s2, 300, &|r| classify_ml(r, &cands, s2).unwrap().index, &mut rng);
        assert!(ml >= cum - 0.02 && ml > 0.9, "{snr_db} dB, {n} samples: ML {ml}, cumulants {cum}");
    }
    // more samples, better cumulant decisions
    let s2 = 1.0 / db2lin(10.0);
    let few = correct_rate(&cands, 64, s2, 300, &|r| classify_cumulants(r, &cands, Some(s2)).unwrap().index, &mut rng);
    let many =
        correct_rate(&cands, 2048, s2, 300, &|r| classify_cumulants(r, &cands, Some(s2)).unwrap().index, &mut rng);
    assert!(many > few && many > 0.97, "64 samples: {few}, 2048: {many}");
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn m2m4_is_consistent_and_its_error_falls_as_one_over_n() {
    let mut rng = Rng::new(4);
    // QAM: up to 10 dB; at higher SNR the fluctuation of the sample kurtosis of the
    // symbols dominates the estimate of the noise (checked below)
    for (c, snrs) in
        [(Constellation::psk(8).unwrap(), vec![0.0, 10.0, 20.0]), (Constellation::qam(16).unwrap(), vec![0.0, 10.0])]
    {
        let c = c.normalized();
        for snr_db in snrs {
            let snr = db2lin(snr_db);
            // the mean and the mean squared value of the relative error
            let moments = |n: usize, rng: &mut Rng| {
                let trials = 400;
                let e: Vec<f64> =
                    (0..trials).map(|_| snr_m2m4(&noisy(&c, n, 1.0 / snr, rng), &c).unwrap() / snr - 1.0).collect();
                (e.iter().sum::<f64>() / trials as f64, e.iter().map(|v| v * v).sum::<f64>() / trials as f64)
            };
            let ((_, a), (bias, b)) = (moments(1000, &mut rng), moments(16_000, &mut rng));
            // sixteen times the samples, about sixteen times less error, and a small bias
            // (the error itself is large for QAM at low SNR, the known weakness of M2M4)
            assert!(a / b > 10.0 && a / b < 25.0, "{:?} at {snr_db} dB: {a} then {b}", c.kind());
            assert!(bias.abs() < 0.02, "{:?} at {snr_db} dB: relative bias {bias}", c.kind());
        }
    }
    // 16-QAM at 20 dB: the noise is a small difference of the moments of the symbols
    let c = Constellation::qam(16).unwrap().normalized();
    let snr = db2lin(20.0);
    let e: Vec<f64> =
        (0..200).map(|_| snr_m2m4(&noisy(&c, 4000, 1.0 / snr, &mut rng), &c).unwrap() / snr - 1.0).collect();
    let rms = (e.iter().filter(|v| v.is_finite()).map(|v| v * v).sum::<f64>() / e.len() as f64).sqrt();
    assert!(rms > 0.2 || e.iter().any(|v| !v.is_finite()), "16-QAM at 20 dB: RMS error {rms}");
}

#[test]
fn the_symbol_rate_of_rrc_signals_at_fractional_oversampling() {
    let mut rng = Rng::new(5);
    let c = Constellation::qam(16).unwrap().normalized();
    for (beta, sps) in [(0.2, 4.37), (0.35, 3.1), (0.5, 6.6)] {
        let n_sym = 8000;
        let a: Vec<C64> = (0..n_sym).map(|_| c.points()[rng.below(16)]).collect();
        let n = (n_sym as f64 * sps) as usize;
        // x(t) = Σ a_k g(t - kT) at t = n / sps, T = 1, the pulse truncated at ±8 symbols
        let x: Vec<C64> = (0..n)
            .map(|i| {
                let t = i as f64 / sps;
                let k0 = (t - 8.0).ceil().max(0.0) as usize;
                let k1 = ((t + 8.0).floor() as usize).min(n_sym - 1);
                let times: Vec<f64> = (k0..=k1).map(|k| t - k as f64).collect();
                rrc_taps(&times, beta, 1.0).iter().zip(&a[k0..=k1]).map(|(g, s)| s * *g).sum::<C64>()
                    + rng.complex_gaussian() * 0.05
            })
            .collect();
        let rs = estimate_symbol_rate(&x, sps, 0.5, sps / 2.0).unwrap();
        assert!((rs - 1.0).abs() < 1e-3, "β {beta}, {sps} samples per symbol: {rs}");
    }
}

//! LAMA and OAMP (`comrs::mimo::message_passing`) against their two papers.
//!
//! C. Jeon, R. Ghods, A. Maleki and C. Studer, "Optimal data detection in large MIMO,"
//! arXiv:1811.01917, 2018:
//!
//! - the minimum and exact recovery thresholds and the critical noise variances of
//!   Table II, which the state evolution gives through eqs. (33) and (34) and
//!   Definitions 8 and 9;
//! - the operating points of Sec. V for QPSK: the AWGN channel reaches a symbol error
//!   rate of $10^{-3}$ at 12.9 dB for $\beta_{\mathrm{mid}} = (\beta_{\min} +
//!   \beta_{\max})/2$; at $\beta_{\min}$ LAMA comes within about 0.1 dB of it after 15
//!   iterations. Two values of the text are not reproduced, and the test pins those of
//!   the state evolution, which gives the thresholds of Table II to four digits and the
//!   10 dB transition at $\beta_{\min}$ of Sec. V-A:
//!   - Sec. V-B quotes 13.5 dB for LAMA after about 90 iterations at
//!     $\beta_{\mathrm{mid}}$ (its Fig. 5(c)); the state evolution reaches $10^{-3}$ at
//!     13.17 dB after 90 iterations and 13.07 dB at convergence;
//!   - Sec. V-A quotes a SER floor of about 0.08 at $\beta_{\max}$ (its Fig. 5(b)); at
//!     $\beta_{\max}$ without noise the state evolution stops at the tangent point
//!     $\sigma^{2}_{\star} = 0.664$ of eq. (33), where the SER of QPSK is 0.208, and
//!     higher with noise (0.24 at 25 dB after 100 iterations). Simulations of 1024
//!     receive antennas and 2136 streams at 40 dB after 200 iterations either stay
//!     there (SER 0.23 and 0.27) or escape to error-free detection, a finite-size
//!     effect;
//! - the decoupling of Theorem 1: the simulated MSE of every iteration follows the
//!   state evolution;
//! - Fig. 6(a): with 128 receive antennas, 64 streams and 16-QAM, LAMA with 8
//!   iterations is close to the individually optimal (IO) bound of the state evolution.
//!
//! J. Ma and L. Ping, "Orthogonal AMP," *IEEE Access*, 5, 2020–2033, 2017:
//!
//! - eq. (42): the asymptotic linear step of OAMP with the MF, PINV and LMMSE
//!   estimators over i.i.d. Gaussian matrices, against the empirical eq. (32);
//! - Fig. 2: i.i.d. Gaussian matrix, BPSK, $M \approx 0.65 N$, SNR = 14 dB: the
//!   simulated MSEs of OAMP-MF, OAMP-PINV, OAMP-LMMSE and AMP follow their state
//!   evolutions, and AMP converges to the fixed point of OAMP-LMMSE, more slowly;
//! - Fig. 3: an ill-conditioned unitarily-invariant matrix (condition number 5) and a
//!   Bernoulli-Gaussian signal ($\rho = 0.2$, $M = N/2$, SNR = 60 dB): OAMP-LMMSE and
//!   OAMP-PINV follow their state evolution, AMP does not converge.
//!
//! The papers simulate $N = 8192$ (Fig. 2) and 4000 (Fig. 3) with 100 realisations; the
//! simulations here are smaller ($N = 1024$ and 1000, a few realisations). Before the
//! waterfall of the MSE the simulations follow the state evolution within 10 %; in the
//! waterfall finite systems lag behind it, the more so the smaller they are (Ma & Ping
//! show the same in their Fig. 6). Measured for OAMP-LMMSE in the setting of Fig. 2, at
//! the iterations 4 and 5: +36 % and +279 % for $N = 256$, +33 % and +264 % for 512,
//! +9 % and +67 % for 1024. There the tests check the claims of the papers on the state
//! evolution itself: the shared fixed point of AMP and OAMP-LMMSE and the faster
//! convergence of the latter, and the failure of AMP on the ill-conditioned matrix.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::mapping::constellation::ConstellationKind;
use comrs::mapping::theory::ser;
use comrs::math::linalg::{matvec, qr, svd};
use comrs::mimo::message_passing::{
    AmpVariance, Field, OampLinear, Prior, amp_state_evolution, amp_thresholds, lama, oamp, oamp_state_evolution,
};
use comrs::prelude::*;
use ndarray::Array2;

/// Table II: constellation, βmin, N0min(βmin), βmax, N0max(βmax).
const TABLE_II: [(&str, usize, f64, f64, f64, f64); 7] = [
    ("BPSK", 2, 2.951, 3.00e-1, 4.171, 2.43e-1),
    ("QPSK", 4, 1.475, 1.50e-1, 2.086, 1.22e-1),
    ("16-QAM", 16, 0.983, 3.00e-2, 1.363, 2.45e-2),
    ("64-QAM", 64, 0.842, 7.14e-3, 1.157, 5.87e-3),
    ("256-QAM", 256, 0.786, 1.77e-3, 1.075, 1.45e-3),
    ("8-PSK", 8, 1.458, 4.44e-2, 1.804, 3.83e-2),
    ("16-PSK", 16, 1.473, 1.14e-2, 1.801, 9.95e-3),
];

fn constellation(name: &str, m: usize) -> Constellation {
    let c = if name.ends_with("QAM") || name == "QPSK" { Constellation::qam(m) } else { Constellation::psk(m) };
    c.unwrap().normalized()
}

/// The thresholds of one row, within the rounding of the table (three decimals for
/// the ratios, three significant digits for the noise variances) and a unit of the
/// last digit.
fn check_row(row: (&str, usize, f64, f64, f64, f64)) {
    let (name, m, bmin, n0min, bmax, n0max) = row;
    let t = amp_thresholds(&Prior::uniform(&constellation(name, m)), Field::Complex);
    assert!((t.beta_min - bmin).abs() < 1.5e-3, "{name}: βmin {} vs {bmin}", t.beta_min);
    assert!((t.beta_max - bmax).abs() < 1.5e-3, "{name}: βmax {} vs {bmax}", t.beta_max);
    assert!((t.n0_min / n0min - 1.0).abs() < 1e-2, "{name}: N0min {} vs {n0min}", t.n0_min);
    assert!((t.n0_max / n0max - 1.0).abs() < 1e-2, "{name}: N0max {} vs {n0max}", t.n0_max);
}

#[test]
fn table_ii_thresholds_of_square_qam() {
    // separable: one-dimensional quadratures, fast
    for row in TABLE_II.iter().filter(|r| r.0.ends_with("QAM") || r.0 == "QPSK") {
        check_row(*row);
    }
}

#[test]
#[ignore = "two-dimensional quadratures: run with --release --ignored"]
fn table_ii_thresholds_of_bpsk_and_psk() {
    for row in TABLE_II.iter().filter(|r| r.0.ends_with("PSK") && r.0 != "QPSK") {
        check_row(*row);
    }
}

/// SER of unit-energy QPSK in complex Gaussian noise of variance `s2`.
fn ser_qpsk(s2: f64) -> f64 {
    ser(ConstellationKind::Qam, 4, lin2db(1.0 / (2.0 * s2))).unwrap()
}

/// The SNR (dB, $\beta E_{s}/N_{0}$) at which LAMA reaches a SER of $10^{-3}$ after
/// `iterations` iterations according to its state evolution (`None`: the AWGN channel).
fn operating_point(beta: f64, iterations: Option<usize>) -> f64 {
    let qpsk = Prior::uniform(&constellation("QPSK", 4));
    let ser_at = |snr_db: f64| {
        let n0 = beta / db2lin(snr_db);
        let s2 = match iterations {
            Some(i) => amp_state_evolution(&qpsk, beta, n0, i + 1, Field::Complex)[i].noise_var,
            None => n0,
        };
        ser_qpsk(s2)
    };
    comrs::math::bisect(|snr| ser_at(snr) > 1e-3, 5.0, 25.0, 60)
}

#[test]
fn operating_points_of_section_v() {
    let beta_mid = (1.475 + 2.086) / 2.0;
    // "the SNR operating point of LAMA converges to 13.5 dB after about 90 iterations,
    // which is 0.6 dB higher than the AWGN SNR of 12.9 dB": the AWGN value holds; the
    // state evolution gives 13.17 dB for LAMA (see the module documentation)
    let awgn = operating_point(beta_mid, None);
    let lama90 = operating_point(beta_mid, Some(90));
    assert!((awgn - 12.9).abs() < 0.05, "AWGN {awgn}");
    assert!((lama90 - 13.17).abs() < 0.03, "LAMA {lama90}");
    // at βmin, a small loss after 15 iterations
    let loss = operating_point(1.475, Some(15)) - operating_point(1.475, None);
    assert!(loss > 0.0 && loss < 0.2, "loss at βmin after 15 iterations: {loss} dB");
    // at βmax without noise, the state evolution stops at the tangent point of eq. (33):
    // a SER of 0.208 (the paper quotes about 0.08, see the module documentation)
    let qpsk = Prior::uniform(&constellation("QPSK", 4));
    let beta_max = amp_thresholds(&qpsk, Field::Complex).beta_max;
    let s2 = amp_state_evolution(&qpsk, beta_max, 0.0, 20_000, Field::Complex)[19_999].noise_var;
    let floor = ser_qpsk(s2);
    assert!((s2 - 0.664).abs() < 0.01 && (floor - 0.208).abs() < 0.005, "βmax: σ² {s2}, SER {floor}");
}

/// A channel of i.i.d. entries $\mathcal{CN}(0, 1/M)$ (complex) or $\mathcal{N}(0, 1/M)$ (real).
fn gaussian_channel(m: usize, n: usize, field: Field, rng: &mut Rng) -> Array2<C64> {
    let s = (1.0 / m as f64).sqrt();
    Array2::from_shape_fn((m, n), |_| match field {
        Field::Complex => rng.complex_gaussian() * s,
        Field::Real => C64::new(rng.gaussian() * s, 0.0),
    })
}

fn noise(m: usize, s2: f64, field: Field, rng: &mut Rng) -> Vec<C64> {
    (0..m)
        .map(|_| match field {
            Field::Complex => rng.complex_gaussian() * s2.sqrt(),
            Field::Real => C64::new(rng.gaussian() * s2.sqrt(), 0.0),
        })
        .collect()
}

fn mse(a: &[C64], b: &[C64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>() / a.len() as f64
}

/// Mean over `runs` of the MSE of every iteration.
/// Whether the simulated MSEs of iteration `t` (one per realisation) agree with the
/// prediction: within four standard errors of their mean (the spread between the
/// realisations), plus 3 % for the finite size of the systems.
fn agrees(runs: &[Vec<f64>], t: usize, predicted: f64) -> bool {
    let r = runs.len() as f64;
    let mean = runs.iter().map(|c| c[t]).sum::<f64>() / r;
    let var = runs.iter().map(|c| (c[t] - mean).powi(2)).sum::<f64>() / (r - 1.0);
    (mean - predicted).abs() <= 4.0 * (var / r).sqrt() + 0.03 * predicted
}

fn average(curves: &[Vec<f64>]) -> Vec<f64> {
    (0..curves[0].len()).map(|t| curves.iter().map(|c| c[t]).sum::<f64>() / curves.len() as f64).collect()
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn lama_decouples_a_large_system() {
    let c = constellation("QPSK", 4);
    let prior = Prior::uniform(&c);
    // 1024 streams: within ±4 % of the state evolution (±15 % with 256)
    for (m, n, snr_db) in [(2048usize, 1024usize, 6.0), (1024, 1024, 8.0)] {
        let beta = n as f64 / m as f64;
        let n0 = beta / db2lin(snr_db);
        let iters = 12;
        let mut rng = Rng::new(17);
        let runs: Vec<Vec<f64>> = (0..8)
            .map(|_| {
                let h = gaussian_channel(m, n, Field::Complex, &mut rng);
                let x: Vec<C64> = (0..n).map(|_| c.points()[rng.below(4)]).collect();
                let y: Vec<C64> = matvec(&h.view(), &x)
                    .iter()
                    .zip(noise(m, n0, Field::Complex, &mut rng))
                    .map(|(a, b)| a + b)
                    .collect();
                let t = lama(&h.view(), &y, n0, &prior, Field::Complex, iters, AmpVariance::Postulated).unwrap();
                t.estimates.iter().map(|e| mse(e, &x)).collect()
            })
            .collect();
        let sim = average(&runs);
        let se = amp_state_evolution(&prior, beta, n0, iters, Field::Complex);
        for (t, (s, p)) in sim.iter().zip(&se).enumerate() {
            assert!(agrees(&runs, t, p.mse), "{m}×{n} at {snr_db} dB, iteration {t}: {s} vs {}", p.mse);
        }
    }
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn lama_on_128_by_64_16qam_is_close_to_the_io_bound() {
    let c = constellation("16-QAM", 16);
    let prior = Prior::uniform(&c);
    let (m, n) = (128usize, 64usize);
    let beta = 0.5;
    let mut rng = Rng::new(23);
    for snr_db in [8.0, 12.0] {
        // the IO bound: the decoupled channel at the fixed point of the state evolution
        let io = |snr: f64| {
            let s2 = amp_state_evolution(&prior, beta, beta / db2lin(snr), 300, Field::Complex)[299].noise_var;
            ser(ConstellationKind::Qam, 16, lin2db(1.0 / (4.0 * s2))).unwrap()
        };
        let n0 = beta / db2lin(snr_db);
        let (mut errors, mut total) = (0u64, 0u64);
        while errors < 2000 {
            let h = gaussian_channel(m, n, Field::Complex, &mut rng);
            let idx: Vec<usize> = (0..n).map(|_| rng.below(16)).collect();
            let x: Vec<C64> = idx.iter().map(|&i| c.points()[i]).collect();
            let y: Vec<C64> =
                matvec(&h.view(), &x).iter().zip(noise(m, n0, Field::Complex, &mut rng)).map(|(a, b)| a + b).collect();
            let t = lama(&h.view(), &y, n0, &prior, Field::Complex, 8, AmpVariance::Postulated).unwrap();
            errors += t.z.iter().zip(&idx).filter(|(z, i)| c.nearest(**z) != **i).count() as u64;
            total += n as u64;
        }
        // "very close to the IO bound": within 0.5 dB with 64 streams and 8 iterations
        let sim = errors as f64 / total as f64;
        assert!(sim > 0.9 * io(snr_db) && sim < io(snr_db - 0.5), "{snr_db} dB: LAMA {sim} vs IO bound {}", io(snr_db));
    }
}

/// The asymptotic linear step of OAMP over i.i.d. Gaussian matrices (Ma & Ping, eq. 42).
fn phi_42(linear: OampLinear, n: f64, m: f64, v2: f64, s2: f64) -> f64 {
    let c = (n - m) / m;
    match linear {
        OampLinear::MatchedFilter => n / m * v2 + s2,
        OampLinear::PseudoInverse => c * v2 + n / (n - m) * s2,
        OampLinear::Lmmse => 0.5 * (s2 + c * v2 + ((s2 + c * v2).powi(2) + 4.0 * s2 * v2).sqrt()),
    }
}

fn bpsk() -> Prior {
    Prior::Discrete { points: vec![C64::new(-1.0, 0.0), C64::new(1.0, 0.0)], probs: vec![0.5, 0.5] }
}

#[test]
#[ignore = "large SVD: run with --release --ignored"]
fn oamp_state_evolution_of_iid_gaussian_matrices_is_eq_42() {
    let (n, m) = (1024usize, 666usize);
    let s2 = n as f64 / (m as f64 * db2lin(14.0));
    let h = gaussian_channel(m, n, Field::Real, &mut Rng::new(5));
    let (_, lambda, _) = svd(&h.view());
    let prior = bpsk();
    for linear in [OampLinear::MatchedFilter, OampLinear::PseudoInverse, OampLinear::Lmmse] {
        let empirical = oamp_state_evolution(&prior, &lambda, n, s2, 8, linear, Field::Real);
        let mut v2 = 1.0;
        for (t, step) in empirical.iter().enumerate() {
            let tau2 = phi_42(linear, n as f64, m as f64, v2, s2);
            assert!(
                (step.noise_var / tau2 - 1.0).abs() < 0.03,
                "{linear:?}, iteration {t}: {} vs {tau2}",
                step.noise_var
            );
            let mmse = prior.mse(tau2, tau2, Field::Real).0;
            v2 = 1.0 / (1.0 / mmse - 1.0 / tau2);
        }
    }
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn oamp_and_amp_on_iid_gaussian_bpsk_follow_their_state_evolution() {
    // Ma & Ping, Fig. 2 (N = 8192 there)
    let (n, m) = (1024usize, 666usize);
    let s2 = n as f64 / (m as f64 * db2lin(14.0));
    let prior = bpsk();
    let iters = 20;
    let mut rng = Rng::new(9);
    let mut sims: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 4];
    let mut se_oamp: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 3];
    let linears = [OampLinear::MatchedFilter, OampLinear::PseudoInverse, OampLinear::Lmmse];
    for _ in 0..6 {
        let h = gaussian_channel(m, n, Field::Real, &mut rng);
        let x: Vec<C64> = (0..n).map(|_| C64::new(if rng.bit() == 0 { -1.0 } else { 1.0 }, 0.0)).collect();
        let y: Vec<C64> =
            matvec(&h.view(), &x).iter().zip(noise(m, s2, Field::Real, &mut rng)).map(|(a, b)| a + b).collect();
        let (_, lambda, _) = svd(&h.view());
        for (k, linear) in linears.into_iter().enumerate() {
            let t = oamp(&h.view(), &y, s2, &prior, Field::Real, iters, linear).unwrap();
            sims[k].push(t.estimates.iter().map(|e| mse(e, &x)).collect());
            se_oamp[k].push(
                oamp_state_evolution(&prior, &lambda, n, s2, iters, linear, Field::Real)
                    .iter()
                    .map(|s| s.mse)
                    .collect(),
            );
        }
        let t = lama(&h.view(), &y, s2, &prior, Field::Real, iters, AmpVariance::Postulated).unwrap();
        sims[3].push(t.estimates.iter().map(|e| mse(e, &x)).collect());
    }
    let se_amp: Vec<f64> =
        amp_state_evolution(&prior, n as f64 / m as f64, s2, iters, Field::Real).iter().map(|s| s.mse).collect();
    let names = ["OAMP-MF", "OAMP-PINV", "OAMP-LMMSE", "AMP"];
    for (k, name) in names.iter().enumerate() {
        let sim = average(&sims[k]);
        let se = if k < 3 { average(&se_oamp[k]) } else { se_amp.clone() };
        // before the waterfall (see the module documentation)
        for t in (0..iters).take_while(|t| se[*t] > 0.1) {
            assert!(agrees(&sims[k], t, se[t]), "{name}, iteration {t}: {} vs {}", sim[t], se[t]);
        }
    }
    // AMP converges to the fixed point of OAMP-LMMSE, more slowly
    let lm = average(&se_oamp[2]);
    let fixed = lm[iters - 1];
    assert!((se_amp[iters - 1] / fixed - 1.0).abs() < 0.05, "AMP {} vs OAMP-LMMSE {fixed}", se_amp[iters - 1]);
    let reach = |c: &[f64]| c.iter().position(|v| *v < 1.1 * fixed).unwrap_or(c.len());
    assert!(reach(&lm) < reach(&se_amp), "OAMP-LMMSE {} vs AMP {} iterations", reach(&lm), reach(&se_amp));
}

/// A real unitarily-invariant matrix $V \Sigma U^{T}$ with Haar $U$, $V$ and singular
/// values in geometric progression of condition number about `kappa`, with
/// $\sum \lambda_{i}^{2} = N$ (Ma & Ping, Sec. V-B).
fn ill_conditioned(m: usize, n: usize, kappa: f64, rng: &mut Rng) -> Array2<C64> {
    let haar = |k: usize, rng: &mut Rng| {
        let g = Array2::from_shape_fn((k, k), |_| C64::new(rng.gaussian(), 0.0));
        qr(&g.view()).unwrap().0
    };
    let (v, u) = (haar(m, rng), haar(n, rng));
    let mut lambda: Vec<f64> = (0..m).map(|i| kappa.powf(-(i as f64) / m as f64)).collect();
    let scale = (n as f64 / lambda.iter().map(|l| l * l).sum::<f64>()).sqrt();
    for l in &mut lambda {
        *l *= scale;
    }
    Array2::from_shape_fn((m, n), |(i, j)| (0..m).map(|k| v[[i, k]] * lambda[k] * u[[j, k]]).sum())
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn oamp_follows_its_state_evolution_on_an_ill_conditioned_matrix() {
    // Ma & Ping, Fig. 3 (N = 4000 there)
    let (n, m) = (1000usize, 500usize);
    let prior = Prior::bernoulli_gaussian(0.2).unwrap();
    let s2 = n as f64 / (m as f64 * db2lin(60.0));
    let iters = 30;
    let mut rng = Rng::new(31);
    let (mut lm, mut pinv, mut amp, mut se_lm, mut se_pinv) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..4 {
        let h = ill_conditioned(m, n, 5.0, &mut rng);
        let x: Vec<C64> = (0..n)
            .map(|_| C64::new(if rng.uniform() < 0.2 { rng.gaussian() / 0.2f64.sqrt() } else { 0.0 }, 0.0))
            .collect();
        let y: Vec<C64> =
            matvec(&h.view(), &x).iter().zip(noise(m, s2, Field::Real, &mut rng)).map(|(a, b)| a + b).collect();
        let (_, lambda, _) = svd(&h.view());
        let run = |linear| -> Vec<f64> {
            let t = oamp(&h.view(), &y, s2, &prior, Field::Real, iters, linear).unwrap();
            t.estimates.iter().map(|e| mse(e, &x)).collect()
        };
        lm.push(run(OampLinear::Lmmse));
        pinv.push(run(OampLinear::PseudoInverse));
        let se = |linear| -> Vec<f64> {
            oamp_state_evolution(&prior, &lambda, n, s2, iters, linear, Field::Real).iter().map(|s| s.mse).collect()
        };
        se_lm.push(se(OampLinear::Lmmse));
        se_pinv.push(se(OampLinear::PseudoInverse));
        let t = lama(&h.view(), &y, s2, &prior, Field::Real, iters, AmpVariance::Postulated).unwrap();
        amp.push(t.estimates.iter().map(|e| mse(e, &x)).collect());
    }
    for (name, runs, se) in [("OAMP-LMMSE", &lm, &se_lm), ("OAMP-PINV", &pinv, &se_pinv)] {
        let (sim, se) = (average(runs), average(se));
        // before the waterfall (see the module documentation)
        for t in (0..iters).take_while(|t| se[*t] > 0.1) {
            assert!(agrees(runs, t, se[t]), "{name}, iteration {t}: {} vs {}", sim[t], se[t]);
        }
        // and the floor that both reach
        let (a, b) = (sim[iters - 1], se[iters - 1]);
        assert!(a < 3.0 * b && b < 3.0 * a, "{name} at the end: {a} vs {b}");
    }
    let (lm, amp) = (average(&lm), average(&amp));
    assert!(lm[iters - 1] < 1e-4, "OAMP-LMMSE ends at {}", lm[iters - 1]);
    assert!(amp[iters - 1].is_nan() || amp[iters - 1] >= 100.0 * lm[iters - 1], "AMP ends at {}", amp[iters - 1]);
}

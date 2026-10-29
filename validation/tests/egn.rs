//! The EGN model against Carena et al. (2014) and against the first-order perturbation
//! of the Manakov equation that it evaluates in closed form.
//!
//! - **Fig. 1(a) of Carena et al.**: one channel of PM-QPSK at 32 GBd (raised-cosine
//!   spectrum of roll-off 0.05) over spans of 100 km of SMF ($D$ = 16.7 ps/(nm·km),
//!   $\gamma$ = 1.3 /(W·km), 0.22 dB/km): the self-channel interference
//!   $\eta_{\mathrm{SCI}} = P^{-3} \int_{-R_{s}/2}^{R_{s}/2} G_{\mathrm{SCI}}(f)\\, df$ of
//!   the GN and EGN models, read on the figure at 1, 10, 20 and 50 spans. Between 2 and
//!   5 spans the published EGN curve lies up to 0.9 dB below the EGN formulas of the same
//!   paper; the Monte Carlo test below follows the formulas.
//! - **First-order perturbation**: the NLI of a periodic signal of $W$ symbols is the
//!   double sum over its spectral lines of the first-order perturbation of the Manakov
//!   equation (Carena et al., Appendix A, Eqs. (32)–(34), without the terms of constant
//!   phase). Its variance with QPSK and with Gaussian symbols, averaged by Monte Carlo,
//!   gives the ratio EGN/GN without any of the EGN formulas.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::optical::fiber::FiberParams;
use comrs::channel::optical::gn::{FormatMoments, GnModel, WdmSpectrum};
use comrs::core::rng::Rng;
use comrs::math::{C64, fft};

const QPSK: FormatMoments = FormatMoments { phi: -1.0, psi: 4.0 };

fn smf(n_spans: usize) -> FiberParams {
    FiberParams {
        n_spans,
        span_km: 100.0,
        d_ps_nm_km: 16.7,
        gamma_w_km: 1.3,
        alpha_db_km: 0.22,
        ..FiberParams::default()
    }
}

/// $\eta_{\mathrm{SCI}}$ (dB(1/W²)) of one channel at 32 GBd by Simpson's rule over the
/// band.
fn eta_sci(n_spans: usize, egn: Option<FormatMoments>, rolloff: f64) -> f64 {
    let rs = 32e9;
    let s = WdmSpectrum::uniform(vec![0.0], 1.0, rs, rolloff);
    let m = GnModel { egn, ..GnModel::new(smf(n_spans)) };
    let k = 16;
    let acc: f64 = (0..=k)
        .map(|i| {
            let w = if i == 0 || i == k {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            w * m.nli_psd(&s, -rs / 2.0 + rs * f64::from(i) / f64::from(k)).unwrap()
        })
        .sum();
    comrs::math::lin2db(acc * rs / f64::from(k) / 3.0)
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn sci_of_pm_qpsk_follows_fig_1a_of_carena_2014() {
    // (spans, GN, EGN) read on Fig. 1(a)
    for (ns, gn, egn) in [(1, 22.3, 18.1), (10, 34.5, 32.6), (20, 38.2, 36.5), (50, 42.7, 41.4)] {
        let (g, e) = (eta_sci(ns, None, 0.05), eta_sci(ns, Some(QPSK), 0.05));
        assert!((g - gn).abs() < 0.2, "{ns} spans: GN {g:.2} dB against {gn}");
        assert!((e - egn).abs() < 0.25, "{ns} spans: EGN {e:.2} dB against {egn}");
    }
}

/// The variance of the first-order NLI of the centre band, averaged over the output
/// lines, for random symbols of `draw` on `w` lines of a rectangular spectrum.
fn first_order_variance(n_spans: usize, w: usize, trials: usize, qpsk: bool, rng: &mut Rng) -> f64 {
    let f = smf(n_spans);
    let (a, b2, ls, ns) = (f.alpha_np_km(), f.beta2_s2_km(), f.span_km, n_spans as f64);
    let f0 = 32e9 / w as f64;
    let half = (w / 2) as i64;
    let mu = |m: i64, k: i64, i: i64| -> C64 {
        let kappa = 4.0 * std::f64::consts::PI.powi(2) * b2 * ((m - i) as f64 * f0) * ((k - i) as f64 * f0);
        let link = (C64::new(1.0, 0.0) - C64::new(0.0, kappa * ls).exp() * (-a * ls).exp()) / C64::new(a, -kappa);
        let x = kappa * ls / 2.0;
        let r = ns * comrs::math::diric(2.0 * x, n_spans);
        link * r * C64::new(0.0, (ns - 1.0) * x).exp()
    };
    // the kernels of every output line (the terms of constant phase, m = i or k = i, removed)
    let lines: Vec<i64> = (-half..half).collect();
    let kernels: Vec<Vec<(usize, usize, usize, C64)>> = lines
        .iter()
        .map(|&i| {
            let mut v = Vec::new();
            for (mi, &m) in lines.iter().enumerate() {
                for (ki, &k) in lines.iter().enumerate() {
                    let n = m + k - i;
                    if m != i && k != i && (-half..half).contains(&n) {
                        v.push((mi, ki, (n + half) as usize, mu(m, k, i)));
                    }
                }
            }
            v
        })
        .collect();
    let norm = (w as f64).sqrt();
    let mut acc = 0.0;
    for _ in 0..trials {
        let pol = |rng: &mut Rng| -> Vec<C64> {
            let s: Vec<C64> = (0..w)
                .map(|_| {
                    if qpsk {
                        C64::new(f64::from(rng.bit()) * 2.0 - 1.0, f64::from(rng.bit()) * 2.0 - 1.0)
                            / std::f64::consts::SQRT_2
                    } else {
                        rng.complex_gaussian()
                    }
                })
                .collect();
            // the spectral lines in increasing frequency
            let x = fft::fft(&s);
            (0..w).map(|j| x[(j + w - w / 2) % w] / norm).collect()
        };
        let (x, y) = (pol(rng), pol(rng));
        for k in &kernels {
            let q: C64 =
                k.iter().map(|(m, kk, n, g)| g * (x[*m] * x[*kk] * x[*n].conj() + x[*m] * y[*kk] * y[*n].conj())).sum();
            acc += q.norm_sqr();
        }
    }
    acc / (trials * w) as f64
}

#[test]
#[ignore = "validation against the first-order perturbation: run with --release --ignored"]
fn egn_correction_is_the_first_order_perturbation_of_qpsk() {
    // 256 symbols: a period much longer than the dispersion memory of 3 spans (about 40
    // symbols); 200 trials: about 0.1 dB of Monte Carlo spread
    let mut rng = Rng::new(1);
    for ns in [1, 2, 3] {
        let qpsk = first_order_variance(ns, 256, 200, true, &mut rng);
        let gauss = first_order_variance(ns, 256, 200, false, &mut rng);
        let monte_carlo = comrs::math::lin2db(qpsk / gauss);
        // the same ratio from the EGN formulas, rectangular spectrum
        let model = eta_sci(ns, Some(QPSK), 0.0) - eta_sci(ns, None, 0.0);
        assert!((monte_carlo - model).abs() < 0.35, "{ns} spans: Monte Carlo {monte_carlo:.2} dB, EGN {model:.2} dB");
    }
}

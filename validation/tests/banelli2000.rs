//! OFDM through a nonlinear amplifier (`comrs::dsp::nonlinear`) against P.
//! Banelli and S. Cacopardi, "Theoretical analysis and performance of OFDM signals in
//! nonlinear AWGN channels," *IEEE Trans. Commun.* 48(3), 430–441, 2000
//! (`references/2000_banelli_ofdm-nonlinear-awgn_tcom.pdf`):
//!
//! - eq. 7 (the series of the output autocorrelation, computed by quadrature for any
//!   nonlinearity) against the closed forms of eq. 28 for the soft envelope limiter of an
//!   ideally predistorted amplifier, with the coefficients $d_{i,n}$ of Table I, and
//!   against Bussgang's gain of `comrs::dsp::nonlinear::clipping_bussgang` (Ochiai & Imai
//!   2002); the coefficients sum to the output power. Eq. 28 and Table I disagree in
//!   sign: the exact integral equals eq. 28 with $+\sum_{i} d_{i,n}\gamma^{i+1}$ for the
//!   $d_{i,n}$ of Table I (or with the printed minus and the signs of Table I reversed),
//!   to $10^{-15}$; the magnitudes of Table I are exact;
//! - eq. 31: the output back-off is $\gamma/(1 - e^{-\gamma})$, which gives the 3.64 dB
//!   and 6.08 dB of Fig. 5 for input back-offs of 3 dB and 6 dB; the paper prints
//!   $\gamma \cdot (1 - e^{-\gamma})$, which gives 2.4 dB and 5.9 dB;
//! - eq. 9 and Sec. III: the distortion falling on each subcarrier of the DAB signal of
//!   Fig. 5 (mode I: 1536 subcarriers of 2048, oversampled four times) against a
//!   simulation of that signal through the limiter;
//! - Fig. 5: the BER of DQPSK against $(E_{b}/N_{0})_{\mathrm{app}}$ from the effective
//!   SNR of every subcarrier (the distortion taken as Gaussian, eqs. 17–21), at the points
//!   read on the figure, and against a Monte Carlo simulation of the whole chain.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::dsp::nonlinear::{clipping_bussgang, distortion_spectrum, output_series};
use comrs::mapping::theory::ber_dpsk;
use comrs::math::erfc;
use comrs::math::fft::{fft, ifft};
use comrs::prelude::*;
use std::f64::consts::PI;

/// Eq. 28 for 2σ² = 1, n = 0 to 4, with the d_{i,n} of Table I and the sign `sign` in
/// front of their sum (the paper prints -1; the integral needs +1).
fn eq28(gamma: f64, sign: f64) -> [f64; 5] {
    let (e, s) = ((-gamma).exp(), (PI * gamma).sqrt() * erfc(gamma.sqrt()));
    let fact = |k: u32| (1..=k).map(f64::from).product::<f64>();
    let c0 = (1.0 - e + 0.5 * s).powi(2);
    let c1 = 0.5 * (0.5 * gamma * e + 0.25 * s).powi(2);
    // Table I: d_{i,n}
    let d: [&[f64]; 3] = [&[-0.5], &[-11.0 / 4.0, 0.5], &[-125.0 / 8.0, 23.0 / 4.0, -0.5]];
    let cn = |n: u32| {
        let a = fact(2 * n) / (2f64.powi(2 * n as i32) * fact(n));
        let sum: f64 = d[n as usize - 2].iter().enumerate().map(|(i, di)| di * gamma.powi(i as i32 + 1)).sum();
        let b = fact(2 * n) / (2f64.powi(2 * n as i32 + 1) * fact(n));
        ((a + sign * sum) * gamma * e + b * s).powi(2) / (fact(n) * fact(n + 1))
    };
    [c0, c1, cn(2), cn(3), cn(4)]
}

fn limiter(gamma: f64) -> impl Fn(f64) -> C64 {
    let a = gamma.sqrt();
    move |r| C64::new(r.min(a), 0.0)
}

#[test]
fn banelli2000_eq28_table1_and_eq31() {
    for gamma_db in [0.0, 3.0, 6.0, 9.0] {
        let g = db2lin(gamma_db);
        let c = output_series(limiter(g), 80);
        let closed = eq28(g, 1.0);
        for (n, (a, b)) in c.iter().zip(&closed).enumerate() {
            assert!((a - b).abs() < 1e-6 * (1.0 + b.abs()), "γ = {gamma_db} dB, c{n}: {a} vs eq. 28 {b}");
        }
        // as printed (minus sign with Table I), c2 is wrong
        assert!(gamma_db > 3.0 || (eq28(g, -1.0)[2] - c[2]).abs() > 1e-4);
        // c0 = α² with Bussgang's gain of clipping (Ochiai & Imai 2002)
        let bg = clipping_bussgang(gamma_db);
        assert!((c[0] - bg.gain.norm_sqr()).abs() < 1e-6);
        // the coefficients sum to the output power 1 - e^{-γ}
        let p_out: f64 = c.iter().sum();
        assert!((p_out - (1.0 - (-g).exp())).abs() < 5e-4, "γ = {gamma_db} dB: Σ c_n = {p_out}");
    }
    // eq. 31: γ/(1 - e^{-γ}) gives the OBOs of Fig. 5; the printed γ(1 - e^{-γ}) does not
    let obo = |g: f64| comrs::math::lin2db(g / (1.0 - (-g).exp()));
    let printed = |g: f64| comrs::math::lin2db(g * (1.0 - (-g).exp()));
    assert!((obo(db2lin(3.0)) - 3.64).abs() < 0.01 && (obo(db2lin(6.0)) - 6.08).abs() < 0.01);
    assert!((printed(db2lin(3.0)) - 3.64).abs() > 1.0);
}

const N_FFT: usize = 2048;
const N_ON: usize = 1536;
const OVERSAMPLING: usize = 4;

/// The bins of the subcarriers (centred on 0) in the oversampled grid.
fn carrier_bins() -> Vec<usize> {
    let bins = N_FFT * OVERSAMPLING;
    (0..N_ON).map(|i| (i + bins - N_ON / 2) % bins).collect()
}

/// The per-subcarrier SDR of the theory (useful over distortion on every subcarrier).
fn sdr_theory(gamma: f64) -> Vec<f64> {
    let c = output_series(limiter(gamma), 60);
    let (u, d) = distortion_spectrum(&c, N_ON, N_FFT * OVERSAMPLING);
    carrier_bins().iter().map(|&k| u[k] / d[k]).collect()
}

/// The BER of Gray DQPSK at (Eb/N0)_app averaged over the subcarriers of SDRs `sdr`.
fn ber_theory(sdr: &[f64], ebn0_app_db: f64) -> f64 {
    let snr_app = 2.0 * db2lin(ebn0_app_db);
    sdr.iter().map(|s| ber_dpsk(4, lin2db(1.0 / (1.0 / s + 1.0 / snr_app) / 2.0)).unwrap()).sum::<f64>()
        / sdr.len() as f64
}

/// OFDM blocks of QPSK through the limiter: the received subcarriers and the transmitted
/// symbols, the oversampled signal of unit power.
fn ofdm_through_limiter(gamma: f64, blocks: usize, rng: &mut Rng) -> (Vec<Vec<C64>>, Vec<Vec<C64>>) {
    let bins = N_FFT * OVERSAMPLING;
    let a = gamma.sqrt();
    let scale = (bins as f64) / (N_ON as f64).sqrt();
    let (mut tx, mut rx) = (Vec::new(), Vec::new());
    for _ in 0..blocks {
        let x: Vec<C64> = (0..N_ON).map(|_| C64::from_polar(1.0, PI / 4.0 + PI / 2.0 * rng.below(4) as f64)).collect();
        let mut grid = vec![C64::new(0.0, 0.0); bins];
        for (&k, v) in carrier_bins().iter().zip(&x) {
            grid[k] = *v;
        }
        let s: Vec<C64> = ifft(&grid).into_iter().map(|v| v * scale).collect();
        let y: Vec<C64> = s.iter().map(|v| if v.norm() > a { v * (a / v.norm()) } else { *v }).collect();
        let yf = fft(&y);
        rx.push(carrier_bins().iter().map(|&k| yf[k] / scale).collect());
        tx.push(x);
    }
    (tx, rx)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn banelli2000_inband_distortion_and_fig5() {
    let mut rng = Rng::new(2000);
    for (gamma_db, read) in [
        (3.0, [(8.0, 6e-3), (10.0, 1.2e-3), (12.0, 1.25e-4), (14.0, 1.1e-5), (16.0, 5e-7)]),
        (6.0, [(8.0, 3.7e-3), (10.0, 3.7e-4), (12.0, 1.3e-5), (13.0, 1.3e-6), (13.0, 1.3e-6)]),
    ] {
        let g = db2lin(gamma_db);
        let sdr = sdr_theory(g);
        // simulation: α by projection, then the distortion power on every subcarrier
        let (tx, rx) = ofdm_through_limiter(g, 120, &mut rng);
        let alpha = tx.iter().flatten().zip(rx.iter().flatten()).map(|(x, y)| y * x.conj()).sum::<C64>().re
            / tx.iter().flatten().map(C64::norm_sqr).sum::<f64>();
        let c = output_series(limiter(g), 60);
        assert!((alpha * alpha / c[0] - 1.0).abs() < 0.01, "γ = {gamma_db} dB: α² = {}, c0 = {}", alpha * alpha, c[0]);
        let dist: Vec<f64> = (0..N_ON)
            .map(|k| tx.iter().zip(&rx).map(|(x, y)| (y[k] - x[k] * alpha).norm_sqr()).sum::<f64>() / tx.len() as f64)
            .collect();
        // the mean SDR in the band, and its profile: the edges (first and last 64)
        // against the centre (the middle 128)
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let sim_sdr = |r: std::ops::Range<usize>| lin2db(alpha * alpha / mean(&dist[r]));
        let th_sdr =
            |r: std::ops::Range<usize>| lin2db(1.0 / mean(&sdr[r].iter().map(|s| 1.0 / s).collect::<Vec<_>>()));
        for r in [0..N_ON, 0..64, N_ON / 2 - 64..N_ON / 2 + 64] {
            let (s, t) = (sim_sdr(r.clone()), th_sdr(r.clone()));
            assert!(
                (s - t).abs() < 0.3,
                "γ = {gamma_db} dB, carriers {r:?}: SDR {s:.2} dB simulated, {t:.2} dB theory"
            );
        }
        // the whole chain at (Eb/N0)_app = 10 dB: noise added on the subcarriers at the
        // apparent SNR, DQPSK detected from one block to the next
        let ebn0 = db2lin(10.0);
        let n0 = alpha * alpha / (2.0 * ebn0);
        let noisy: Vec<Vec<C64>> =
            rx.iter().map(|y| y.iter().map(|v| v + rng.complex_gaussian() * n0.sqrt()).collect()).collect();
        let quadrant = |z: C64| ((z.arg().rem_euclid(2.0 * PI) / (PI / 2.0)).round() as usize) % 4;
        let gray = |q: usize| q ^ (q >> 1);
        let (mut errors, mut bits) = (0u64, 0u64);
        for t in 1..tx.len() {
            for k in 0..N_ON {
                let sent = quadrant(tx[t][k] * tx[t - 1][k].conj());
                let got = quadrant(noisy[t][k] * noisy[t - 1][k].conj());
                errors += u64::from((gray(sent) ^ gray(got)).count_ones());
                bits += 2;
            }
        }
        let (sim, th) = (errors as f64 / bits as f64, ber_theory(&sdr, 10.0));
        let tol = 0.15 * th + 3.0 * (th / bits as f64).sqrt();
        assert!((sim - th).abs() < tol, "γ = {gamma_db} dB at 10 dB: simulated {sim:.3e}, theory {th:.3e}");
        // Fig. 5: the points read on the figure (a factor 1.5 on a log scale)
        for (x, v) in read {
            let b = ber_theory(&sdr, x);
            assert!((b / v).ln().abs() < 1.5f64.ln(), "γ = {gamma_db} dB at {x} dB: {b:.3e} vs {v:.1e} read");
        }
    }
}

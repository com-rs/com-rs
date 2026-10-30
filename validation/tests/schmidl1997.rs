//! Timing and frequency synchronisation of OFDM (`comrs::waveform::ofdm::sync`) against
//! T. M. Schmidl and D. C. Cox, "Robust frequency and timing synchronization for OFDM,"
//! *IEEE Trans. Commun.* 45(12), 1613–1621, 1997, in AWGN with halves of $L = 512$
//! samples (the length of their Figs. 4–6; 1000 active subcarriers of 1024, no
//! oversampling), the SNR $\sigma_{s}^{2}/\sigma_{n}^{2}$ per sample, 4000 trials:
//!
//! - at the correct timing, the mean of the timing metric is
//!   $\sigma_{s}^{4}/(\sigma_{s}^{2} + \sigma_{n}^{2})^{2}$, eq. (19), and its variance that
//!   of eq. (20), about $4/(L \cdot \mathrm{SNR})$ at high SNR;
//! - outside the preamble (noise only) it is $\chi_{2}^{2}/(2L)$, eq. (28): mean $1/L$ and
//!   variance $1/L^{2}$;
//! - the fractional frequency offset $\hat\phi/\pi$ has the variance
//!   $1/(\pi^{2} L \cdot \mathrm{SNR})$ of eq. (43), the Cramér–Rao bound (44) (Fig. 9),
//!   $2.03 \cdot 10^{-5}$ for 1000 carriers at 10 dB (Fig. 8), at high SNR: the product of
//!   the noises of the two halves, neglected by eq. (43), multiplies it by
//!   $1 + 1/(2\,\mathrm{SNR})$ (1.5 at 0 dB, which the simulation follows within 10 %).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::ofdm::sync::{preamble, schmidl_cox, synchronize};
use std::f64::consts::PI;

const TRIALS: usize = 4000;

fn setup() -> (Ofdm, Vec<C64>, usize, f64) {
    let ofdm = Ofdm { n_cp: 64, oversampling: 1, ..Ofdm::new(1024, 1000).unwrap() };
    let p = preamble(&ofdm, 1997).unwrap();
    let half = ofdm.fft_len() / 2;
    let body = &p[ofdm.n_cp..];
    let power = body.iter().map(C64::norm_sqr).sum::<f64>() / body.len() as f64;
    (ofdm, p, half, power)
}

fn mean_var(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (m, v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64)
}

#[test]
fn schmidl_cox_1997_eqs_19_20_and_28_timing_metric() {
    let (ofdm, p, l, ps) = setup();
    let mut rng = Rng::new(45);
    let lf = l as f64;
    for snr_db in [0.0, 10.0] {
        let pn = ps / db2lin(snr_db);
        let m: Vec<f64> = (0..TRIALS)
            .map(|_| {
                let r: Vec<C64> = p[ofdm.n_cp..].iter().map(|s| s + rng.complex_gaussian() * pn.sqrt()).collect();
                schmidl_cox(&r, l)[0]
            })
            .collect();
        let (mean, var) = mean_var(&m);
        let mu = ps * ps / (ps + pn).powi(2);
        let (s2, n2) = (ps, pn);
        let want_var = 2.0 * s2 * s2 * ((1.0 + mu) * s2 * n2 + (1.0 + 2.0 * mu) * n2 * n2) / (lf * (s2 + n2).powi(4));
        assert!((mean / mu - 1.0).abs() < 0.01, "{snr_db} dB: mean {mean} vs (19) {mu}");
        assert!((var / want_var - 1.0).abs() < 0.15, "{snr_db} dB: variance {var} vs (20) {want_var}");
    }
    // noise only
    let m: Vec<f64> = (0..TRIALS)
        .map(|_| {
            let r: Vec<C64> = (0..2 * l).map(|_| rng.complex_gaussian()).collect();
            schmidl_cox(&r, l)[0]
        })
        .collect();
    let (mean, var) = mean_var(&m);
    assert!((mean * lf - 1.0).abs() < 0.05 && (var * lf * lf - 1.0).abs() < 0.15, "outside: {mean} {var}");
}

#[test]
fn schmidl_cox_1997_eq_43_frequency_offset_variance() {
    let (ofdm, p, l, ps) = setup();
    let mut rng = Rng::new(43);
    let n = ofdm.fft_len() as f64;
    // Fig. 8: 1000 carriers (L = 500) at 10 dB
    assert!((1.0 / (PI * PI * 500.0 * 10.0) - 2.03e-5).abs() < 0.01e-5);
    for snr_db in [0.0, 10.0, 20.0] {
        let pn = ps / db2lin(snr_db);
        let eps = 0.2;
        let est: Vec<f64> = (0..TRIALS)
            .map(|_| {
                let r: Vec<C64> = p
                    .iter()
                    .enumerate()
                    .map(|(k, s)| {
                        s * C64::from_polar(1.0, 2.0 * PI * eps * k as f64 / n) + rng.complex_gaussian() * pn.sqrt()
                    })
                    .collect();
                synchronize(&r, &ofdm).unwrap().cfo
            })
            .collect();
        let (mean, var) = mean_var(&est);
        let snr = db2lin(snr_db);
        let crb = 1.0 / (PI * PI * l as f64 * snr);
        // the product of the noises of the two halves adds 1/(2 SNR)
        let exact = crb * (1.0 + 0.5 / snr);
        assert!((mean - eps).abs() < 5.0 * (exact / TRIALS as f64).sqrt(), "{snr_db} dB: mean {mean}");
        assert!((var / exact - 1.0).abs() < 0.1, "{snr_db} dB: variance {var} vs {exact}");
        if snr_db >= 10.0 {
            assert!((var / crb - 1.0).abs() < 0.15, "{snr_db} dB: variance {var} vs (43) {crb}");
        }
    }
}

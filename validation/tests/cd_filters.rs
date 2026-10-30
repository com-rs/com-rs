//! FIR compensation of chromatic dispersion (`comrs::dsp::compensation::cd`) against
//! A. Eghbali, H. Johansson, O. Gustafsson, S. J. Savory, "Optimal least-squares FIR
//! digital filters for compensation of chromatic dispersion in digital coherent optical
//! receivers," *J. Lightw. Technol.* 32(8), 1449–1456, 2014:
//!
//! - Table I: the dispersion factors $K$ and the lengths $N$ of Savory's filter
//!   (S. J. Savory, *Opt. Express* 16(2), 2008) of its two examples;
//! - eq. 19: over the whole band the least-squares filter is $D(n)$, checked against the
//!   integral of eq. 12 by quadrature;
//! - Sec. III-A2: the error energy $E$ of the least-squares filter falls with its length
//!   and stays below that of Savory's filter, whose error grows beyond its length rule;
//! - Fig. 8: the error of the band-limited filter falls as the penalty $\epsilon$ falls;
//! - Sec. IV-A: the band-limited filter of 31 taps against frequency sampling;
//! - Figs. 3, 5 and 9 (ignored, statistical): the BER of 16-QAM at the end of the chain of
//!   Fig. 2, with ideal root-raised-cosine filters (roll-off 0.25, two samples per
//!   symbol) as the paper assumes.
//!
//! Discrepancy: Sec. IV-A gives $E = -220$ dB for the least-squares filter and $-63$ dB
//! for frequency sampling; we find $10 \log_{10} E = -113$ and $-34$ dB, that is
//! $20 \log_{10} E = -227$ and $-68$ dB: the paper seems to quote $20 \log_{10}$ of the
//! energy, and its frequency-sampling filter (bins $2k\pi/N$, causal) differs a little
//! from ours (centred bins and taps).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::awgn::add_noise;
use comrs::dsp::compensation::cd::{
    cd_dispersion_factor, cd_fir_taps, cd_ls_error, cd_ls_taps, cd_savory_length, cd_savory_taps,
};
use comrs::dsp::filter::rrc_spectrum;
use comrs::math::cis;
use comrs::math::fft::{fft, ifft};
use comrs::prelude::*;
use std::f64::consts::PI;

/// The group-velocity dispersion (s²/km) of $D = 17$ ps/(nm km) at 1553 nm with
/// $c = 3 \cdot 10^{8}$ m/s (Table I).
fn beta2() -> f64 {
    let (d, lambda, c) = (17e-6, 1553e-9, 3e8);
    -d * lambda * lambda / (2.0 * PI * c) * 1e3
}

/// Example 1 (4000 km at 21.4 GHz) and 2 (1000 km at 80 GHz) of Table I.
const EXAMPLES: [(f64, f64); 2] = [(4000.0, 21.4e9), (1000.0, 80e9)];

fn db(e: f64) -> f64 {
    comrs::math::lin2db(e)
}

#[test]
fn eghbali2014_table1() {
    for ((z, fs), (k, n)) in EXAMPLES.iter().zip([(19.9227, 251), (69.605, 875)]) {
        let kk = cd_dispersion_factor(beta2(), *z, *fs);
        assert!((kk / k - 1.0).abs() < 5e-5, "K = {kk}, Table I {k}");
        assert_eq!(cd_savory_length(beta2(), *z, *fs), n);
    }
}

#[test]
fn eghbali2014_eq19_full_band() {
    let (z, fs) = EXAMPLES[0];
    let k = cd_dispersion_factor(beta2(), z, fs);
    let h = cd_ls_taps(beta2(), z, fs, 251, fs / 2.0, 0.0).unwrap();
    // eq. 12 by Simpson's rule
    let m = 400_000;
    for n in [-125i32, -40, 0, 7, 125] {
        let f = |i: usize| {
            let w = -PI + 2.0 * PI * i as f64 / m as f64;
            cis(k * w * w + f64::from(n) * w)
        };
        let s: C64 = (0..=m)
            .map(|i| {
                f(i) * if i == 0 || i == m {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                }
            })
            .sum();
        let d = s * (2.0 * PI / m as f64 / 3.0) / (2.0 * PI);
        let t = h[(n + 125) as usize];
        assert!((t - d).norm() < 1e-9, "n = {n}: {t} vs {d}");
    }
}

#[test]
fn eghbali2014_error_against_length() {
    for (i, (z, fs)) in EXAMPLES.iter().enumerate() {
        let (b2, band) = (beta2(), fs / 2.0);
        let n0 = cd_savory_length(b2, *z, *fs);
        let (mut last_ls, mut last_sav) = (f64::INFINITY, 0.0);
        for n in (n0..n0 + 100).step_by(10) {
            let e_ls = cd_ls_error(&cd_ls_taps(b2, *z, *fs, n, band, 0.0).unwrap(), b2, *z, *fs, band).unwrap();
            let e_sav = cd_ls_error(&cd_savory_taps(b2, *z, *fs, n).unwrap(), b2, *z, *fs, band).unwrap();
            assert!(
                e_ls < last_ls && e_ls < e_sav,
                "example {}, N = {n}: LS {}, Savory {} dB",
                i + 1,
                db(e_ls),
                db(e_sav)
            );
            // Savory's filter longer than its rule: a larger error
            if n > n0 {
                assert!(e_sav > last_sav, "example {}, N = {n}: Savory {} dB", i + 1, db(e_sav));
            }
            (last_ls, last_sav) = (e_ls, e_sav);
        }
    }
    // at N = 251 (example 1): -19.2 dB against -15.5 dB
    let (z, fs) = EXAMPLES[0];
    let e = |h: &[C64]| db(cd_ls_error(h, beta2(), z, fs, fs / 2.0).unwrap());
    let (el, es) =
        (e(&cd_ls_taps(beta2(), z, fs, 251, fs / 2.0, 0.0).unwrap()), e(&cd_savory_taps(beta2(), z, fs, 251).unwrap()));
    assert!((el + 19.19).abs() < 0.05 && (es + 15.46).abs() < 0.05, "LS {el} dB, Savory {es} dB");
}

#[test]
fn eghbali2014_fig8_penalty() {
    // the band of a roll-off 0.25 at two samples per symbol: |f| <= 0.3125 fs
    let (z, fs) = EXAMPLES[0];
    let band = 0.3125 * fs;
    for n in [101, 151] {
        let mut last = f64::INFINITY;
        for k in (2..=14).step_by(2) {
            let h = cd_ls_taps(beta2(), z, fs, n, band, 10f64.powi(-k)).unwrap();
            let e = cd_ls_error(&h, beta2(), z, fs, band).unwrap();
            assert!(e < last, "Nc = {n}, ε = 1e-{k}: {} dB", db(e));
            last = e;
        }
    }
}

#[test]
fn eghbali2014_section4a_against_frequency_sampling() {
    // z = 500 km with the parameters of example 1: K = 2.49, N = Nc = 31
    let (z, fs) = (500.0, EXAMPLES[0].1);
    assert_eq!(cd_savory_length(beta2(), z, fs), 31);
    let band = 0.3125 * fs;
    let e_ls = db(cd_ls_error(&cd_ls_taps(beta2(), z, fs, 31, band, 1e-14).unwrap(), beta2(), z, fs, band).unwrap());
    let e_fsm = db(cd_ls_error(&cd_fir_taps(beta2(), z, fs, 31), beta2(), z, fs, band).unwrap());
    // the paper: -220 and -63 dB (20 log10 E, see the module)
    assert!(e_ls < -105.0 && (e_fsm + 34.0).abs() < 1.5, "LS {e_ls} dB, FSM {e_fsm} dB");
}

/// The BER of Gray 16-QAM through the chain of Fig. 2: ideal RRC shaping (roll-off 0.25,
/// two samples per symbol), the dispersion of `z_km` at `fs`, white noise at `ebn0_db`,
/// the centred `taps` (none: back to back), the matched filter.
fn ber(taps: Option<&[C64]>, z_km: f64, fs: f64, ebn0_db: f64, n_sym: usize, rng: &mut Rng) -> f64 {
    let (sps, c) = (2usize, Constellation::qam(16).unwrap().normalized());
    let n = n_sym * sps;
    let g: Vec<f64> = rrc_spectrum(n_sym, sps, 0.25).iter().map(|v| v * (sps as f64).sqrt()).collect();
    let labels: Vec<usize> = (0..n_sym).map(|_| rng.below(16)).collect();
    let mut u = vec![C64::new(0.0, 0.0); n];
    for (k, l) in labels.iter().enumerate() {
        u[k * sps] = c.points()[*l];
    }
    let k = if taps.is_some() { cd_dispersion_factor(beta2(), z_km, fs) } else { 0.0 };
    let w = |i: usize| 2.0 * PI * ((i + n / 2) % n) as f64 / n as f64 - PI;
    let x: Vec<C64> = fft(&u).iter().enumerate().map(|(i, v)| v * g[i] * cis(-k * w(i) * w(i))).collect();
    let s2 = 1.0 / (4.0 * db2lin(ebn0_db));
    let mut y = ifft(&x);
    add_noise(&mut y, s2, rng);
    let mut yf = fft(&y);
    if let Some(t) = taps {
        let mut h = vec![C64::new(0.0, 0.0); n];
        let half = t.len() / 2;
        for (i, v) in t.iter().enumerate() {
            h[(i + n - half) % n] = *v;
        }
        for (a, b) in yf.iter_mut().zip(fft(&h)) {
            *a *= b;
        }
    }
    let r = ifft(&yf.iter().zip(&g).map(|(a, b)| a * b).collect::<Vec<_>>());
    let errors: u32 = labels.iter().enumerate().map(|(k, l)| (c.nearest(r[k * sps]) ^ l).count_ones()).sum();
    f64::from(errors) / (4 * n_sym) as f64
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn eghbali2014_fig5_and_fig9_ber_of_16qam() {
    let mut rng = Rng::new(7);
    let (z, fs) = EXAMPLES[0];
    let n_sym = 1 << 22;
    let run = |t: Option<&[C64]>, rng: &mut Rng| ber(t, z, fs, 14.0, n_sym, rng).log10();
    let sav = cd_savory_taps(beta2(), z, fs, 251).unwrap();
    let ls = cd_ls_taps(beta2(), z, fs, 251, fs / 2.0, 0.0).unwrap();
    let (b_btb, b_sav, b_ls) = (run(None, &mut rng), run(Some(&sav), &mut rng), run(Some(&ls), &mut rng));
    // Fig. 5 at N = Nc = 251, Eb/N0 = 14 dB, read on the figure: BTB -5.47, (9) -5.2,
    // (3) -4.55 (log10 BER)
    for (name, ours, paper) in [("BTB", b_btb, -5.47), ("(9)", b_ls, -5.2), ("(3)", b_sav, -4.55)] {
        assert!((ours - paper).abs() < 0.2, "{name}: log10 BER {ours}, Fig. 5 {paper}");
    }
    // Fig. 9: the band-limited filter of 119 taps does at least as well as Savory's 251;
    // 101 taps do not
    let band = 0.3125 * fs;
    let b119 = run(Some(&cd_ls_taps(beta2(), z, fs, 119, band, 1e-14).unwrap()), &mut rng);
    let b101 = run(Some(&cd_ls_taps(beta2(), z, fs, 101, band, 1e-14).unwrap()), &mut rng);
    eprintln!("log10 BER: BTB {b_btb}, (9) {b_ls}, (3) {b_sav}, 119 taps {b119}, 101 taps {b101}");
    assert!(b119 <= b_sav && b101 > b_sav, "log10 BER: 119 taps {b119}, 101 taps {b101}, Savory {b_sav}");
}

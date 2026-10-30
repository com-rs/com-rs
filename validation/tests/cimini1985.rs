//! Pilot-based gain correction of OFDM against a cochannel interferer
//! (`comrs::waveform::ofdm::correction`), by simulation of the system of L. J. Cimini,
//! Jr., "Analysis and simulation of a digital mobile channel using orthogonal frequency
//! division multiplexing," *IEEE Trans. Commun.* 33(7), 1985
//! (`references/multicarrier/1985_cimini_ofdm-digital-mobile-channel_tcom.pdf`): QPSK on N
//! subchannels at 7.5 kHz, the desired and the cochannel OFDM signals in independent
//! flat Jakes fading at 850 MHz, an ideal pilot (Sec. II-E) or one decorrelated by a
//! two-ray delay spread (Sec. II-F), BER compared at 1e-2 as in the paper (footnote 4).
//! The SIR (dB) at a BER of 1e-2, simulated and read on the figures:
//!
//! | curve | here | figure |
//! |---|---|---|
//! | N = 1, coherent QPSK (closed form) | 16.86 | 16.4 (Fig. 5), 17.0 (Fig. 7) |
//! | N = 512, 60 mph, optimum gain correction (15) | 11.3 | 11.0 |
//! | N = 512, 60 mph, gain-limited correction (16) | 12.35 | 11.9 (Fig. 5), 11.5 (Fig. 7) |
//! | N = 512, 30 mph, gain-limited | 12.9 | 12.8 |
//! | N = 128, 30 mph, gain-limited | 13.9 | 13.95 |
//!
//! The text claims "improvements in SIR of 6–7 dB" for N = 512 at 60 mph over N = 1:
//! its own Fig. 5 shows 4.5 dB at 1e-2 (16.4 - 11.9), as does the simulation (4.5 dB);
//! the other claims hold: less than 1 dB lost at 30 mph (0.5 dB), about 2 dB more with N
//! = 128 (1.55 dB here against N = 512 at 60 mph, 2.05 on the figure), and 1 dB more
//! with the optimum correction (1.0 dB). The closed forms of the SDR (Figs. 3 and 4) are
//! checked by `crates/com-waveform/src/ofdm/correction.rs::literature`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use com_validation::{log_interp, max_doppler_hz};
use comrs::channel::multipath::Tdl;
use comrs::mapping::theory::ber_rayleigh_mrc;
use comrs::math::bisect;
use comrs::math::fft::{fft, ifft};
use comrs::prelude::*;
use comrs::waveform::ofdm::correction::gain_limited;

const FS: f64 = 7_500.0;

/// The maximum Doppler shift (Hz) at 850 MHz.
fn doppler(mph: f64) -> f64 {
    max_doppler_hz(mph * 0.447_04, 850e6)
}

#[derive(Clone, Copy, Debug)]
enum Correction {
    /// Gain-limited correction (eq. (16)) with the limit (amplitude).
    Limited(f64),
    /// The optimum factor of eq. (15).
    Optimum,
}

/// BER of the OFDM QPSK of Cimini with `n` subchannels, the pilot envelope
/// $Z_{p} = \rho^{*} Z_{d} + \sqrt{1 - |\rho|^{2}} Z_{x}$ (eq. (24); `rho` = 1: ideal).
fn ber(n: usize, fd: f64, sir_db: f64, corr: Correction, rho: C64, blocks: usize, seed: u64) -> f64 {
    let tdl = Tdl::new(&[(0.0, 0.0)], fd, FS, 1, 1).unwrap();
    let mut rng = Rng::new(seed);
    let g = comrs::math::db2lin(-sir_db);
    let s = 1.0 / 2f64.sqrt();
    let qpsk = |r: &mut Rng| -> (C64, [bool; 2]) {
        let (a, b) = (r.below(2) == 1, r.below(2) == 1);
        (C64::new(if a { -s } else { s }, if b { -s } else { s }), [a, b])
    };
    let fading = |rng: &mut Rng| -> Vec<C64> {
        let t = tdl.draw(n, rng);
        (0..n).map(|m| t.coefficient(0, m, 0, 0)).collect()
    };
    let sc = (n as f64).sqrt();
    let mut errors = 0usize;
    for _ in 0..blocks {
        let (zd, zi, zx) = (fading(&mut rng), fading(&mut rng), fading(&mut rng));
        let (d, bits): (Vec<C64>, Vec<[bool; 2]>) = (0..n).map(|_| qpsk(&mut rng)).unzip();
        let i: Vec<C64> = (0..n).map(|_| qpsk(&mut rng).0).collect();
        let (dt, it) = (ifft(&d), ifft(&i));
        let r: Vec<C64> = (0..n).map(|m| (zd[m] * dt[m] + g.sqrt() * zi[m] * it[m]) * sc).collect();
        let corrected = match corr {
            Correction::Limited(eps) => {
                let zp: Vec<C64> = (0..n).map(|m| rho.conj() * zd[m] + (1.0 - rho.norm_sqr()).sqrt() * zx[m]).collect();
                gain_limited(&r, &zp, eps)
            }
            Correction::Optimum => {
                (0..n).map(|m| r[m] / (zd[m] * (1.0 + g * zi[m].norm_sqr() / zd[m].norm_sqr()))).collect()
            }
        };
        for (v, b) in fft(&corrected).iter().zip(&bits) {
            errors += usize::from((v.re < 0.0) != b[0]) + usize::from((v.im < 0.0) != b[1]);
        }
    }
    errors as f64 / (2 * n * blocks) as f64
}

/// The gain limits searched for the "best" one (dB).
const LIMITS_DB: [f64; 7] = [-20.0, -16.0, -12.0, -10.0, -8.0, -4.0, 0.0];

/// The SIR (dB) at which the BER reaches 1e-2 (the best gain limit for gain-limited
/// correction), by interpolation of log BER on a 1 dB grid.
fn sir_at_1e2(n: usize, mph: f64, optimum: bool, rho: C64) -> f64 {
    let blocks = 400_000 / n;
    let at = |s: f64, seed: u64| {
        if optimum {
            ber(n, doppler(mph), s, Correction::Optimum, rho, blocks, seed)
        } else {
            LIMITS_DB
                .iter()
                .map(|e| ber(n, doppler(mph), s, Correction::Limited(comrs::math::db2amp(*e)), rho, blocks, seed))
                .fold(1.0, f64::min)
        }
    };
    let mut prev = (2.0, at(2.0, 1));
    for k in 1..40 {
        let s = 2.0 + k as f64;
        let b = at(s, 1 + k);
        if b < 1e-2 {
            return log_interp(prev, (s, b), 1e-2);
        }
        prev = (s, b);
    }
    panic!("no SIR up to 41 dB reaches 1e-2");
}

fn flat() -> C64 {
    C64::new(1.0, 0.0)
}

/// The SIR of coherent QPSK over flat Rayleigh fading (N = 1) at a BER of 1e-2: with
/// a complex Gaussian interferer the BER is that of BPSK at the mean SNR SIR/2.
fn sir_n1() -> f64 {
    bisect(|s| ber_rayleigh_mrc(comrs::math::db2lin(s) / 2.0, 1).unwrap() > 1e-2, 0.0, 40.0, 60)
}

/// Cimini 1985, Fig. 5 and Sec. II-E (see the module).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn cimini_1985_fig_5_flat_fading() {
    let n1 = sir_n1();
    assert!((n1 - 16.86).abs() < 0.01, "{n1}");
    let opt = sir_at_1e2(512, 60.0, true, flat());
    let gl60 = sir_at_1e2(512, 60.0, false, flat());
    let gl30 = sir_at_1e2(512, 30.0, false, flat());
    let gl128 = sir_at_1e2(128, 30.0, false, flat());
    for (got, read) in [(opt, 11.0), (gl60, 11.9), (gl30, 12.8), (gl128, 13.95)] {
        assert!((got - read).abs() < 0.6, "{got} dB against {read} dB read on Fig. 5");
    }
    // the gain over N = 1: 4.5 dB, as on the figure, not the 6–7 dB of the text
    assert!((n1 - gl60 - 4.5).abs() < 0.4, "{}", n1 - gl60);
    assert!((0.0..1.0).contains(&(gl30 - gl60)), "30 mph: {}", gl30 - gl60);
    assert!((1.2..2.5).contains(&(gl128 - gl60)), "N = 128: {}", gl128 - gl60);
    assert!((0.6..1.4).contains(&(gl60 - opt)), "optimum: {}", gl60 - opt);
}

/// Cimini 1985, Fig. 6 (N = 512, 60 mph): the BER against the gain limit has its
/// minimum where the SDR of eq. (20) peaks; the minima read on the figure, 2.5e-3 at
/// -7.5 dB (SIR 15 dB), 7.5e-4 at -9 dB (17 dB) and 8e-5 at -10.3 dB (20 dB), are
/// reproduced within a factor 1.5 and 2.5 dB. Above the optimum limit the simulated BER
/// rises faster than on the figure (1.0e-2 at 0 dB and SIR 15 dB, against 4e-3 read).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn cimini_1985_fig_6_gain_limit() {
    let limits: Vec<f64> = (0..=12).map(|k| -24.0 + 2.0 * k as f64).collect();
    for (sir, ber_read, at_read) in [(15.0, 2.5e-3, -7.5), (17.0, 7.5e-4, -9.0), (20.0, 8e-5, -10.3)] {
        let curve: Vec<f64> = limits
            .iter()
            .map(|e| ber(512, doppler(60.0), sir, Correction::Limited(comrs::math::db2amp(*e)), flat(), 4000, 9))
            .collect();
        let (k, min) = curve.iter().enumerate().fold((0, 1.0), |a, (k, &b)| if b < a.1 { (k, b) } else { a });
        assert!((min / ber_read).ln().abs() < 1.5f64.ln(), "SIR {sir}: {min} against {ber_read}");
        assert!((limits[k] - at_read).abs() <= 2.5, "SIR {sir}: minimum at {} dB", limits[k]);
    }
}

/// Cimini 1985, eq. (23) and Fig. 7: the correlation of the two-ray channel ($\Delta$
/// = 50 µs) between the pilot and a subchannel 500 Hz and 1 kHz away (`Tdl` of
/// `comrs::channel::multipath`), its coherence bandwidth $1/(4\Delta)$ (not the
/// "$1/\Delta$" of the text), and the losses at a BER of 1e-2 against an ideal pilot:
/// less than 2 dB at 500 Hz (0.7 dB here, 2.0 dB on the figure) and 4.3 dB at 1 kHz
/// (4.1 dB on the figure; "5 or 6 dB" in the text).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn cimini_1985_eq_23_and_fig_7_delay_spread() {
    let delta = 50e-6;
    let two_ray = Tdl::new(&[(0.0, 0.0), (delta, 0.0)], 0.0, 1e6, 1, 1).unwrap();
    let rho_at = |f: f64| two_ray.frequency_correlation(1, (1e6 / f).round() as usize);
    for f in [500.0, 1000.0, 2000.0] {
        let s = std::f64::consts::TAU * f;
        let eq23 = (1.0 + (s * delta).cos()).sqrt() / 2f64.sqrt();
        assert!((rho_at(f).norm() - eq23).abs() < 1e-12, "{f} Hz");
        assert!((rho_at(f).arg() + s * delta / 2.0).abs() < 1e-12, "{f} Hz");
    }
    assert!((rho_at(1.0 / (4.0 * delta)).norm_sqr() - 0.5).abs() < 1e-12);
    let ideal = sir_at_1e2(512, 60.0, false, flat());
    let l500 = sir_at_1e2(512, 60.0, false, rho_at(500.0)) - ideal;
    let l1k = sir_at_1e2(512, 60.0, false, rho_at(1000.0)) - ideal;
    assert!((0.0..2.0).contains(&l500), "500 Hz: {l500} dB");
    assert!((3.5..5.0).contains(&l1k), "1 kHz: {l1k} dB");
}

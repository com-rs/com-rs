//! OFDM and FBMC/OQAM through a memoryless amplifier
//! (`comrs::waveform::{ofdm, fbmc}`, `comrs::hardware::{amplifier::Saleh,
//! nonlinear::bussgang}`) against H. Bouhadda, H. Shaiek, D. Roviras, R. Zayani,
//! Y. Medjahdi and R. Bouallegue, "Theoretical analysis of BER performance of nonlinearly
//! amplified FBMC/OQAM and OFDM signals," *EURASIP J. Adv. Signal Process.* 2014:60
//! (`references/multicarrier/2014_bouhadda_ber-nonlinear-fbmc-oqam-ofdm_jasp.pdf`): 64 subcarriers
//! (PHYDYAS, $K = 4$, for FBMC), the TWTA of eq. (13) with $A_{\mathrm{sat}} = 1$ at the
//! input back-off of eq. (14), AWGN; $E_{b}$ is that of the useful output
//! $|K|^{2}\sigma^{2}$, which puts the curves where the paper has them.
//!
//! - Figs. 10 and 11 (16QAM, the gain $K$ corrected): OFDM and FBMC/OQAM have the BER
//!   of the Gaussian model (eq. (45) with the exact $K$ and $\sigma_{d}^{2}$ of
//!   `bussgang`) within 25 %, and the floors read at 30 dB (3.5e-2 and 1.15e-3 for
//!   $\phi_{0} = 0$, 4.2e-2 and 2.0e-3 for $\pi/6$, at 4 and 8 dB) within 30 %;
//! - Fig. 7 (4QAM, $\phi_{0} = \pi/3$, $K$ not corrected): the rotation of $K$ costs
//!   FBMC/OQAM more than OFDM through the intrinsic interference; the curves labelled
//!   "IBO = 4 dB" are those of an input back-off of 8 dB, the value of the text (at 4 dB
//!   the floor of OFDM is 1.7e-2, not 1.2e-5): floors of 1.0e-5 (OFDM) and 1.5e-4 (FBMC)
//!   against 1.2e-5 and 1.6e-4 read; the "loss of ≈7 dB" of FBMC "at a BER of 5e-3"
//!   is 0.2 dB there (2.6 dB at 5e-4).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use com_validation::log_interp;
use comrs::dsp::nonlinear::bussgang;
use comrs::hardware::amplifier::Saleh;
use comrs::mapping::theory::ber_gray_qam;
use comrs::prelude::*;
use comrs::waveform::fbmc::{Fbmc, oqam_destagger, oqam_stagger};
use comrs::waveform::ofdm::Ofdm;

const M: usize = 64;

/// BER of OFDM or FBMC/OQAM (`fbmc`) with `order`-QAM through the TWTA ($A_{\mathrm{sat}} = 1$, $\phi_{0}$)
/// at the input back-off `ibo` (dB), AWGN at `ebn0_db` ($E_{b}$ of the useful output), with or
/// without the correction by Bussgang's gain.
fn ber(fbmc: bool, order: usize, phi0: f64, ibo: f64, ebn0_db: f64, correct: bool, blocks: usize, seed: u64) -> f64 {
    let c = Constellation::qam(order).unwrap().normalized();
    let bits_per = order.trailing_zeros() as usize;
    let twta = Saleh { alpha_a: 1.0, beta_a: 1.0, alpha_p: phi0, beta_p: 1.0 };
    let power = comrs::math::db2lin(-ibo);
    let k = bussgang(|r| C64::from_polar(twta.am_am(r), twta.am_pm(r)), power).gain;
    // E_b of the useful output |K|² σ² (the 'linear' curve of the paper is the AWGN one)
    let sigma_w2 = k.norm_sqr() * power / (bits_per as f64 * comrs::math::db2lin(ebn0_db));
    let mut rng = Rng::new(seed);
    let n_sym = 8; // complex symbols per subcarrier and block
    let (mut errors, mut total) = (0usize, 0usize);
    let ofdm = Ofdm { oversampling: 1, ..Ofdm::new(M, M).unwrap() };
    let fb = Fbmc::new(M, 4).unwrap();
    for _ in 0..blocks {
        let bits: Vec<u8> = (0..M * n_sym * bits_per).map(|_| rng.below(2) as u8).collect();
        let x = c.modulate(&bits).unwrap();
        let s = if fbmc { fb.modulate(&oqam_stagger(&x, M).unwrap()).unwrap() } else { ofdm.modulate(&x).unwrap() };
        let scale = power.sqrt();
        let y: Vec<C64> = s
            .iter()
            .map(|v| twta.amplify(&[v * scale])[0] / scale + rng.complex_gaussian() * (sigma_w2 / power).sqrt())
            .collect();
        let gain = if correct { k } else { C64::new(1.0, 0.0) };
        let z: Vec<C64> = if fbmc {
            // correct the gain before taking the real part
            let y2: Vec<C64> = y.iter().map(|v| v / gain).collect();
            let d = fb.demodulate(&y2).unwrap();
            oqam_destagger(&d.iter().map(|v| v.re).collect::<Vec<f64>>(), M).unwrap()
        } else {
            ofdm.demodulate(&y).unwrap().iter().map(|v| v / gain).collect()
        };
        let rx = c.demodulate(&z);
        errors += rx.iter().zip(&bits).filter(|(a, b)| a != b).count();
        total += bits.len();
    }
    errors as f64 / total as f64
}

fn theory(order: usize, phi0: f64, ibo: f64, ebn0_db: f64) -> f64 {
    let twta = Saleh { alpha_a: 1.0, beta_a: 1.0, alpha_p: phi0, beta_p: 1.0 };
    let power = comrs::math::db2lin(-ibo);
    let b = bussgang(|r| C64::from_polar(twta.am_am(r), twta.am_pm(r)), power);
    let (k, d) = (b.gain, b.distortion);
    let bits_per = order.trailing_zeros() as f64;
    let sigma_w2 = k.norm_sqr() * power / (bits_per * comrs::math::db2lin(ebn0_db));
    let es_n0 = k.norm_sqr() * power / (sigma_w2 + d);
    ber_gray_qam(order, comrs::math::lin2db(es_n0 / bits_per)).unwrap()
}

/// The $E_{b}/N_{0}$ (dB) at which `curve` crosses `target`, by interpolation of log BER on
/// a grid of 0.5 dB.
fn crossing(curve: impl Fn(f64) -> f64, target: f64) -> f64 {
    let mut prev = (0.0, curve(0.0));
    for k in 1..60 {
        let e = 0.5 * k as f64;
        let b = curve(e);
        if b < target {
            return log_interp(prev, (e, b), target);
        }
        prev = (e, b);
    }
    f64::NAN
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn bouhadda_2014_figs_10_11_corrected_gain() {
    let pi6 = std::f64::consts::FRAC_PI_6;
    for (phi0, ibo, floor) in [(0.0, 4.0, 3.5e-2), (0.0, 8.0, 1.15e-3), (pi6, 4.0, 4.2e-2), (pi6, 8.0, 2.0e-3)] {
        for e in [10.0, 20.0, 30.0] {
            let t = theory(16, phi0, ibo, e);
            for fbmc in [false, true] {
                let b = ber(fbmc, 16, phi0, ibo, e, true, 300, 1 + u64::from(fbmc));
                assert!((b / t - 1.0).abs() < 0.25, "φ0 {phi0} IBO {ibo} {e} dB FBMC {fbmc}: {b} against {t}");
            }
        }
        let t30 = theory(16, phi0, ibo, 30.0);
        assert!((t30 / floor - 1.0).abs() < 0.3, "φ0 {phi0} IBO {ibo}: floor {t30} against {floor}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn bouhadda_2014_fig_7_phase_rotation() {
    let p3 = std::f64::consts::FRAC_PI_3;
    let ofdm = ber(false, 4, p3, 8.0, 30.0, false, 12_000, 3);
    let fbmc = ber(true, 4, p3, 8.0, 30.0, false, 2000, 4);
    assert!((ofdm / 1.2e-5).ln().abs() < 1.5f64.ln(), "OFDM floor {ofdm}");
    assert!((fbmc / 1.6e-4).ln().abs() < 1.5f64.ln(), "FBMC floor {fbmc}");
    // at 4 dB of back-off the floor of OFDM is far above the curve labelled so
    assert!(ber(false, 4, p3, 4.0, 30.0, false, 300, 5) > 1e-2);
    let gap = |target: f64| {
        crossing(|e| ber(true, 4, p3, 8.0, e, false, 3000, 6), target)
            - crossing(|e| ber(false, 4, p3, 8.0, e, false, 3000, 7), target)
    };
    let (g3, g4) = (gap(5e-3), gap(5e-4));
    assert!(g3.abs() < 0.6, "gap at 5e-3: {g3} dB");
    assert!((1.8..3.5).contains(&g4), "gap at 5e-4: {g4} dB");
}

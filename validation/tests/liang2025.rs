//! The mutual information of shaped square QAM against E. M. Liang and J. M. Kahn,
//! "Probabilistic shaping distributions for optical communications," *J. Lightw.
//! Technol.*, 43(4), 1504–1521, 2025: the ultimate shaping gain of eq. (27) and the
//! finite-constellation shaping gains of Fig. 6 (the rates of `com_validation::shaping`;
//! the gains of 1024- to 65536-QAM are `#[ignore]`d).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::shaping::{mi_ask, snr_for, snr_mb};
use comrs::math::{db2lin, golden_max, lin2db};
use std::time::Instant;

/// The finite-constellation shaping gain of square $M$-QAM (Liang & Kahn 2025,
/// Sec. III-C3): the largest difference, at a fixed mutual information, between the SNR
/// of the uniform input and that of the optimised Maxwell–Boltzmann input:
/// `(gain in dB, MI in bits per QAM symbol, SNR of the MB input in dB)`. The MB law
/// $e^{-\beta |x|^{2}}$ of square QAM is the product of the MB laws of its two
/// $\sqrt{M}$-ASK quadratures, and the SNR $E[|X|^{2}]/\sigma^{2}$ of the complex channel
/// is that of each real dimension, so the MI is twice that of $\sqrt{M}$-ASK at the same
/// SNR.
fn qam_shaping_gain(m_qam: usize) -> (f64, f64, f64) {
    let m = (m_qam as f64).sqrt().round() as usize;
    // the SNRs of the uniform and of the shaped input at the MI `mi`
    let snrs = |mi: f64| {
        let unif = snr_for(&|s| mi_ask(m, 0.0, s), mi / 2.0).unwrap();
        let (shaped, _) = snr_mb(m, mi / 2.0, &|nu, s| mi_ask(m, nu, s));
        (unif, shaped)
    };
    let bits = (m_qam as f64).log2();
    let mi = golden_max(
        |mi| {
            let (unif, shaped) = snrs(mi);
            unif - shaped
        },
        2.0,
        bits - 1.0,
        30,
    );
    let (unif, shaped) = snrs(mi);
    (unif - shaped, mi, shaped)
}

/// Eq. (27) of Liang & Kahn (2025): the ultimate shaping gain of the standard coherent
/// (SC) channel, $\gamma_{s}^{(\mathrm{SC})} = \pi e / 6 \approx 1.53$ dB (also Böcherer
/// et al. 2015, Sec. III-D, after Forney et al. 1984), and the Shannon limit of Fig. 6,
/// $\log_{2}(1 + \mathrm{SNR})$, that the MI of every input stays below.
#[test]
fn liang_kahn_2025_eq27_ultimate_shaping_gain() {
    let ultimate = lin2db(std::f64::consts::PI * std::f64::consts::E / 6.0);
    assert!((ultimate - 1.53).abs() < 5e-3, "{ultimate}");
    assert!((ultimate - 1.5329).abs() < 1e-4, "{ultimate}");
    // MI of MB and uniform 256-QAM below the Shannon limit and saturating at 8 bits
    for snr_db in [10.0, 20.0, 30.0, 50.0] {
        let shannon = (1.0 + db2lin(snr_db)).log2();
        let unif = 2.0 * mi_ask(16, 0.0, snr_db);
        assert!(unif < shannon, "{snr_db} dB: {unif} vs {shannon}");
        let shaped = 2.0 * mi_ask(16, 0.02, snr_db);
        assert!(shaped < shannon, "{snr_db} dB: {shaped} vs {shannon}");
    }
    assert!((2.0 * mi_ask(16, 0.0, 50.0) - 8.0).abs() < 1e-9);
}

/// Fig. 6 of Liang & Kahn (2025): the finite-constellation shaping gain of 256-QAM over
/// the SC (AWGN) channel, "∼ 1 dB" (red arrow and text, at a mutual information of about
/// 6 bits per symbol, Sec. III-C3).
#[test]
fn liang_kahn_2025_fig6_shaping_gain_of_256qam() {
    let (gain, mi, snr) = qam_shaping_gain(256);
    println!("256-QAM: largest shaping gain {gain:.3} dB at {mi:.2} bits ({snr:.2} dB)");
    assert!((gain - 1.0).abs() < 0.1, "{gain}");
    assert!((mi - 6.0).abs() < 0.5, "{mi}");
}

/// Fig. 6 of Liang & Kahn (2025), Sec. III-C3: the finite-constellation shaping gain
/// grows from ∼ 1 dB to ∼ 1.4 dB as $M$ grows from 256 to 65536, by smaller steps at
/// every factor of 4, and that of 65536-QAM is "∼ 0.13 dB less than the ultimate shaping
/// gain" (eq. 27), i.e. ∼ 1.40 dB (blue arrow, at about 11.5 bits per symbol).
#[test]
#[ignore = "the 128- and 256-ASK quadratures take minutes: run with --ignored"]
fn liang_kahn_2025_fig6_shaping_gains_256_to_65536_qam() {
    let start = Instant::now();
    let ultimate = lin2db(std::f64::consts::PI * std::f64::consts::E / 6.0);
    let mut gains = Vec::new();
    for m in [256, 1024, 4096, 16384, 65536] {
        let (gain, mi, snr) = qam_shaping_gain(m);
        println!("{m}-QAM: largest shaping gain {gain:.3} dB at {mi:.2} bits ({snr:.2} dB)");
        gains.push(gain);
    }
    let steps: Vec<f64> = gains.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(steps.iter().all(|s| *s > 0.0), "{gains:?}");
    assert!(steps.windows(2).all(|w| w[1] < w[0]), "{steps:?}");
    let g = gains[4];
    assert!((g - 1.4).abs() < 0.1, "{g}");
    assert!(g < ultimate, "{g}");
    println!("{:.1} s", start.elapsed().as_secs_f64());
}

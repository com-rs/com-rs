//! Bit error rates of equalised OFDM over Rayleigh fading with an imperfect channel
//! estimate (`comrs::mapping::imperfect_csi`) against M.-X. Chang and Y. T. Su,
//! "Performance analysis of equalized OFDM systems in Rayleigh fading," *IEEE Trans.
//! Wireless Commun.* 1(4), 721–732, 2002 (`references/multicarrier/2002_chang_equalized-ofdm-rayleigh_twc.pdf`):
//!
//! - eqs. 31, 32 and 53 (BPSK, Gray QPSK and Gray 16-QAM equalised by an estimate jointly
//!   Gaussian with the channel, of correlations $\rho_{1}$, $\rho_{2}$) against a Monte
//!   Carlo simulation of that model, with com-rs constellations;
//! - eqs. 40 and 41 (DBPSK and Gray DQPSK over Jakes' fading) against a Monte Carlo
//!   simulation of pairs of channel samples correlated by $J_{0}(2\pi f_{D}T_{s})$;
//!
//! Figs. 3 and 4 (the floors of DBPSK and of 16-QAM at 50 dB read on the figures, and
//! the misprint of the label $f_{D}T_{s} = 0.05$ of Fig. 3, whose curve floors where
//! eq. 40 puts 0.005) are the cards `validation/cards/chang2002_fig3_dbpsk_floors.yaml`
//! and `chang2002_fig4_floors.yaml`, checked on the closed forms.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::within_three_sigma;
use comrs::mapping::imperfect_csi::{
    ber_16qam_imperfect_csi, ber_bpsk_imperfect_csi, ber_dbpsk_jakes, ber_dqpsk_jakes, ber_qpsk_imperfect_csi,
};
use comrs::math::{bessel_j0, gray};
use comrs::prelude::*;
use std::f64::consts::PI;

/// A pair (H, Ĥ): H ~ CN(0, 2σ1² = 1), Ĥ = c H + w with ½E[Ĥ H*] = μ1 + jμ2 and
/// ½E|Ĥ|² = σ2² (Chang & Su eqs. 15–19).
fn channel_and_estimate(rho1: f64, rho2: f64, sigma_ratio: f64, rng: &mut Rng) -> (C64, C64) {
    let h = rng.complex_gaussian(); // E|h|² = 1: σ1² = 1/2
    let s1 = std::f64::consts::FRAC_1_SQRT_2;
    let s2 = sigma_ratio * s1;
    let c = C64::new(rho1, rho2) * (s2 / s1); // (μ1 + jμ2)/σ1²
    let w_var = 2.0 * s2 * s2 * (1.0 - rho1 * rho1 - rho2 * rho2);
    (h, c * h + rng.complex_gaussian() * w_var.sqrt())
}

/// Bit errors of `n` symbols of `c` (unit energy) at the mean SNR per bit `ebn0`.
fn simulate(c: &Constellation, ebn0: f64, rho: (f64, f64, f64), n: usize, rng: &mut Rng) -> (u64, u64) {
    let k = c.bits_per_symbol();
    let n0 = 1.0 / (k as f64 * ebn0); // E|X|² = E|H|² = 1
    let mut errors = 0;
    for _ in 0..n {
        let i = rng.below(c.order());
        let (h, hh) = channel_and_estimate(rho.0, rho.1, rho.2, rng);
        let y = h * c.points()[i] + rng.complex_gaussian() * n0.sqrt();
        let j = c.nearest(y / hh);
        // the points are indexed by their labels
        errors += u64::from((i ^ j).count_ones());
    }
    (errors, (n * k) as u64)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn chang2002_eqs_31_32_53_against_monte_carlo() {
    let mut rng = Rng::new(2002);
    let cases = [(10.0, 0.95, 0.0, 1.0), (10.0, 0.9, 0.3, 0.8), (20.0, 0.99, 0.05, 1.1), (15.0, 0.97, -0.1, 1.0)];
    for (db, r1, r2, sr) in cases {
        let g = db2lin(db);
        let checks: [(Constellation, f64); 3] = [
            (Constellation::psk(2).unwrap(), ber_bpsk_imperfect_csi(g, r1, r2).unwrap()),
            (Constellation::qam(4).unwrap(), ber_qpsk_imperfect_csi(g, r1, r2).unwrap()),
            (Constellation::qam(16).unwrap(), ber_16qam_imperfect_csi(g, r1, r2, sr).unwrap()),
        ];
        for (c, theory) in checks {
            // BPSK and QPSK do not depend on |Ĥ| (σ2/σ1)
            let sr_used = if c.order() == 16 { sr } else { 1.0 };
            let (e, t) = simulate(&c.normalized(), g, (r1, r2, sr_used), 200_000, &mut rng);
            assert!(
                within_three_sigma(e, t, theory),
                "{}-ary at {db} dB, ρ = ({r1}, {r2}), σ2/σ1 = {sr}: {} vs {theory}",
                c.order(),
                e as f64 / t as f64
            );
        }
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn chang2002_eqs_40_41_differential_detection_over_jakes_fading() {
    let mut rng = Rng::new(41);
    for (db, fd_ts) in [(20.0, 0.01), (30.0, 0.05), (40.0, 0.005)] {
        let g = db2lin(db);
        let rho = bessel_j0(2.0 * PI * fd_ts);
        for m in [2usize, 4] {
            let k = m.trailing_zeros();
            let n0 = 1.0 / (f64::from(k) * g);
            let (mut errors, n) = (0u64, 400_000usize);
            for _ in 0..n {
                // two channel samples of correlation J0(2π fD Ts)
                let h0 = rng.complex_gaussian();
                let h1 = h0 * rho + rng.complex_gaussian() * (1.0 - rho * rho).sqrt();
                let d = rng.below(m);
                let s0 = C64::from_polar(1.0, 2.0 * PI * rng.uniform());
                let s1 = s0 * C64::from_polar(1.0, 2.0 * PI * gray(d) as f64 / m as f64);
                let r0 = h0 * s0 + rng.complex_gaussian() * n0.sqrt();
                let r1 = h1 * s1 + rng.complex_gaussian() * n0.sqrt();
                let phase = (r1 * r0.conj()).arg().rem_euclid(2.0 * PI);
                let q = ((phase * m as f64 / (2.0 * PI)).round() as usize) % m;
                // the label of the decided phase index
                let decided = (0..m).find(|&x| gray(x) == q).unwrap();
                errors += u64::from((decided ^ d).count_ones());
            }
            let theory = if m == 2 { ber_dbpsk_jakes(g, fd_ts) } else { ber_dqpsk_jakes(g, fd_ts) }.unwrap();
            assert!(
                within_three_sigma(errors, (n * k as usize) as u64, theory),
                "{m}-DPSK at {db} dB, fD Ts = {fd_ts}: {} vs {theory}",
                errors as f64 / (n * k as usize) as f64
            );
        }
    }
}

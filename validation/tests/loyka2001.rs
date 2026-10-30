//! The capacity of a MIMO channel with exponentially correlated antennas
//! (`comrs::channel::spatial::exponential_correlation`, `comrs::mimo::capacity`) against S.
//! L. Loyka, "Channel capacity of MIMO architecture using the exponential correlation
//! matrix," *IEEE Commun. Lett.* 5(9), 369–371, 2001: eq. (3), $C = \log_{2} \det(I +
//! \frac{\rho}{n} R)$ with $R_{ij} = r^{j - i}$ (eq. (10)), computed as the equal-power
//! capacity of a channel $H = L$, $L L^{H} = R$:
//!
//! - without correlation it is $n \log_{2}(1 + \rho/n)$ (eq. (16)), and it does not depend
//!   on the phase of $r$ (Sec. III);
//! - Fig. 1 ($\rho = 30$ dB): 66.6 bit/s/Hz for $n = 10$ and 219.6 for $n = 50$ at $r = 0$,
//!   about 54 for $n = 10$ at $r = 0.8$, 190 and 160 for $n = 50$ at $r = 0.6$ and 0.8 (read
//!   on the figure; within 5 %);
//! - the approximation (13) is within 3.5 % up to $r = 0.8$ (its accuracy falls as $n$
//!   and $\rho/n$ fall, Sec. III), and (14) shows that $r = 0.7$ costs about 3 dB of SNR
//!   ($-10 \log_{10}(1 - r^{2}) = 2.9$ dB): the exact capacity (3) of $r = 0.7$ is that of
//!   the uncorrelated channel 2.6 dB ($n = 10$) and 2.7 dB ($n = 50$) lower in SNR;
//! - the mean capacity of the Rayleigh channel $H = L G / \sqrt{n}$, $G$ i.i.d.
//!   $\mathcal{CN}(0, 1)$ (rows of unit mean power, $\mathbb{E}[H H^{H}] = R$, Sec. II), is
//!   below the bound (5), Telatar's closed form without correlation. The text finds it
//!   "approximately 20–40 % smaller" than the deterministic capacity; with this
//!   normalisation it is 17.5 % smaller for $n = 10$ (16–18 % up to $r = 0.8$) and 20.5 % for
//!   $n = 50$ at $r = 0$: the simulation of the paper is not described.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::spatial::exponential_correlation;
use comrs::math::bisect;
use comrs::math::linalg::{CMat, cholesky};
use comrs::mimo::capacity::{equal_power, ergodic_rayleigh};
use comrs::prelude::*;

fn factor(n: usize, r: C64) -> CMat {
    cholesky(&exponential_correlation(n, r).view()).unwrap()
}

fn capacity(n: usize, r: C64, snr: f64) -> f64 {
    equal_power(&factor(n, r).view(), snr).unwrap()
}

#[test]
fn loyka_2001_eqs_3_to_16_and_fig_1() {
    let snr = db2lin(30.0);
    for n in [10usize, 50] {
        let nf = n as f64;
        let c0 = capacity(n, C64::new(0.0, 0.0), snr);
        assert!((c0 - nf * (1.0 + snr / nf).log2()).abs() < 1e-9, "n = {n}: {c0}");
        // the phase of r does not matter
        let rotated = capacity(n, C64::from_polar(0.6, 1.1), snr);
        assert!((rotated - capacity(n, C64::new(0.6, 0.0), snr)).abs() < 1e-9);
        for r in [0.2, 0.4, 0.6, 0.8] {
            let c = capacity(n, C64::new(r, 0.0), snr);
            let s = snr / nf * (1.0 - r * r);
            let eq13 = nf * (1.0 + s).log2() + ((1.0 + snr / nf) / (1.0 + s)).log2();
            assert!((eq13 / c - 1.0).abs() < 0.035, "n = {n}, r = {r}: {c} vs (13) {eq13}");
        }
    }
    for (n, r, read) in [(10usize, 0.0, 66.6), (10, 0.8, 54.0), (50, 0.0, 219.6), (50, 0.6, 190.0), (50, 0.8, 160.0)] {
        let c = capacity(n, C64::new(r, 0.0), snr);
        assert!((c / read - 1.0).abs() < 0.05, "n = {n}, r = {r}: {c} vs {read}");
    }
    // (14): r = 0.7 is the SNR divided by 1 - r² (2.9 dB, "3 dB" in the text); the loss of
    // the exact capacity (3), the SNR drop that brings the uncorrelated capacity down to
    // that of r = 0.7, is 2.6 dB for n = 10 and 2.7 dB for n = 50
    let eq14 = -lin2db(1.0 - 0.7f64.powi(2));
    assert!((eq14 - 3.0).abs() < 0.1, "{eq14}");
    let loss = |n: usize| {
        let target = capacity(n, C64::new(0.7, 0.0), snr);
        bisect(|db| capacity(n, C64::new(0.0, 0.0), snr * db2lin(-db)) > target, 0.0, 10.0, 60)
    };
    let (l10, l50) = (loss(10), loss(50));
    assert!((l10 - 2.59).abs() < 0.01 && (l50 - 2.68).abs() < 0.01, "{l10} and {l50} dB");
    // closer to (14) as n grows
    assert!(l10 < l50 && l50 < eq14 && eq14 - l10 < 0.35, "{l10} and {l50} dB against (14) {eq14} dB");
}

#[test]
fn loyka_2001_eq_5_mean_capacity_of_the_rayleigh_channel() {
    let snr = db2lin(30.0);
    let n = 10;
    let mut rng = Rng::new(2001);
    let draws = 400;
    let mean = |r: f64, rng: &mut Rng| {
        let l = factor(n, C64::new(r, 0.0));
        (0..draws)
            .map(|_| {
                let g = CMat::from_shape_fn((n, n), |_| rng.complex_gaussian() / (n as f64).sqrt());
                equal_power(&l.dot(&g).view(), snr).unwrap()
            })
            .sum::<f64>()
            / draws as f64
    };
    // without correlation: Telatar's ergodic capacity at the SNR ρ/n per receive antenna
    let iid = mean(0.0, &mut rng);
    let telatar = ergodic_rayleigh(n, n, snr / n as f64).unwrap();
    assert!((iid / telatar - 1.0).abs() < 0.01, "{iid} vs {telatar}");
    for r in [0.0, 0.6, 0.8] {
        let (m, bound) = (mean(r, &mut rng), capacity(n, C64::new(r, 0.0), snr));
        let loss = 1.0 - m / bound;
        assert!(m < bound && (0.15..0.21).contains(&loss), "r = {r}: mean {m}, bound {bound}");
    }
}

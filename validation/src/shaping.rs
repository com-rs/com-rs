//! The information rates of shaped amplitude-shift keying shared by the tests of
//! Böcherer, Steiner & Schulte (2015) and Liang & Kahn (2025): the mutual information
//! and the rate of bit-metric decoding of $M$-ASK with a Maxwell–Boltzmann input, and the
//! SNR at which they reach a target rate.
//!
//! # References
//!
//! 1. G. Böcherer, F. Steiner and P. Schulte, "Bandwidth efficient and rate-matched
//!    low-density parity-check coded modulation," *IEEE Trans. Commun.*, 63(12),
//!    4651–4665, 2015 (arXiv:1502.02733v2), eqs. 8–10 and 50.
#![allow(clippy::expect_used, reason = "helpers of the tests: an order that is not a PAM order is a broken test")]

use comrs::mapping::constellation::{Constellation, ConstellationKind};
use comrs::mapping::source::maxwell_boltzmann;
use comrs::mapping::theory;
use comrs::math::search::regula_falsi;
use comrs::math::{golden_max, lin2db};

/// Gauss–Hermite nodes of the information rates.
pub const POINTS: usize = 40;

/// The Maxwell–Boltzmann probabilities $\propto e^{-\nu x^{2}}$ of the $M$-ASK points
/// $\pm 1, \pm 3, \dots$ (eq. 8 of Böcherer et al.), in the order of the points of
/// [`Constellation::pam`].
///
/// # Panics
/// If `m` is not a PAM order.
pub fn mb(m: usize, nu: f64) -> Vec<f64> {
    maxwell_boltzmann(&Constellation::pam(m).expect("a PAM order"), nu)
}

/// The SNR (dB) at which the increasing `rate` (bits per real dimension at an SNR in dB)
/// reaches `target`, by the Illinois regula falsi ([`regula_falsi`], 100 steps, tolerances
/// $10^{-9}$ dB and $10^{-12}$ bit) from the capacity $10 \log_{10}(2^{2R} - 1)$, which no
/// input beats, the bracket widened by steps of 2 dB; `None` when the rate stays below the
/// target over 20 dB (an entropy smaller than the target).
pub fn snr_for(rate: &dyn Fn(f64) -> f64, target: f64) -> Option<f64> {
    let mut a = lin2db(2f64.powf(2.0 * target) - 1.0);
    if rate(a) - target >= 0.0 {
        return Some(a);
    }
    let mut b = a + 1.0;
    while rate(b) - target < 0.0 {
        if b - a > 20.0 {
            return None;
        }
        a = b;
        b += 2.0;
    }
    Some(regula_falsi(|s| rate(s) - target, a, b, 1e-9, 1e-12, 100))
}

/// The mutual information (bits per real dimension) of $M$-ASK with the MB parameter
/// `nu` (0: uniform) at an SNR in dB.
///
/// # Panics
/// If `m` is not a PAM order.
pub fn mi_ask(m: usize, nu: f64, snr_db: f64) -> f64 {
    theory::mi(ConstellationKind::Pam, m, snr_db, Some(&mb(m, nu)), POINTS).expect("a PAM order")
}

/// The rate of bit-metric decoding (bits per real dimension) of $M$-ASK with the BRGC
/// labels and the MB parameter `nu` at an SNR in dB (eq. 50 of Böcherer et al.).
///
/// # Panics
/// If `m` is not a PAM order.
pub fn bmd_ask(m: usize, nu: f64, snr_db: f64) -> f64 {
    theory::gmi(ConstellationKind::Pam, m, snr_db, Some(&mb(m, nu)), POINTS).expect("a PAM order")
}

/// The SNR (dB) at which the MB input of the best parameter reaches `target` with the
/// rate `rate(nu, snr_db)` (eqs. 9–10 of Böcherer et al.: the best scaling, the power
/// fixed): `(SNR, nu)`, by a golden-section search on $\ln \nu$.
pub fn snr_mb(m: usize, target: f64, rate: &dyn Fn(f64, f64) -> f64) -> (f64, f64) {
    // nu E_s(uniform) of the optimum is about 1 at every order
    let es = ((m * m - 1) as f64) / 3.0;
    let f = |ln_nu: f64| snr_for(&|s| rate(ln_nu.exp(), s), target).unwrap_or(f64::INFINITY);
    let x = golden_max(|ln_nu| -f(ln_nu), (0.05 / es).ln(), (4.0 / es).ln(), 40);
    (f(x), x.exp())
}

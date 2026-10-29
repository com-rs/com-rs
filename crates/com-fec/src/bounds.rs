//! Information-theoretic limits of binary codes on the AWGN channel.
//!
//! With BPSK on real AWGN of variance $\sigma^{2}$ per dimension, the LLR of a
//! transmitted 0 is $L \sim \mathcal{N}(2/\sigma^{2}, 4/\sigma^{2})$, and
//!
//! $$
//! C = 1 - \mathbb{E}\left[\log_{2}(1 + e^{-L})\right], \qquad
//! V = \operatorname{Var}\left[1 - \log_{2}(1 + e^{-L})\right]
//! $$
//!
//! are the capacity and the dispersion of the binary-input AWGN channel (bits per
//! channel use; [`biawgn_capacity`], [`biawgn_dispersion`]), both by Gauss–Hermite
//! quadrature. At a finite length $n$ and $k$ information bits, the normal approximation
//! of Polyanskiy, Poor & Verdú (2010) gives the smallest achievable block error
//! probability,
//!
//! $$
//! \varepsilon \approx Q\left(\frac{n C - k + \tfrac{1}{2}\log_{2} n}{\sqrt{n V}}\right)
//! $$
//!
//! ([`normal_approximation`]), and [`shannon_limit_ebn0_db`] the $E_{b}/N_{0}$ at which a
//! rate-$R$ code can work at all ($n \to \infty$). With $E_{b}/N_{0}$ and the rate
//! $R = k/n$: $\sigma^{2} = 1 / (2 R E_{b}/N_{0})$.
//!
//! # Examples
//! ```
//! use com_fec::bounds::{shannon_limit_ebn0_db, normal_approximation};
//! // rate 1/2 BPSK: the Shannon limit is 0.187 dB
//! assert!((shannon_limit_ebn0_db(0.5) - 0.187).abs() < 0.01);
//! // a (128, 64) code cannot do much better than a block error rate of 1e-3 at 2.4 dB
//! let p = normal_approximation(128, 64, 2.4);
//! assert!(p > 5e-4 && p < 3e-3);
//! ```
//!
//! # References
//!
//! 1. C. E. Shannon, "A mathematical theory of communication," *Bell Syst. Tech. J.*,
//!    27, 379–423 and 623–656, 1948.
//! 2. Y. Polyanskiy, H. V. Poor and S. Verdú, "Channel coding rate in the finite
//!    blocklength regime," *IEEE Trans. Inf. Theory*, 56(5), 2307–2359, 2010.
//! 3. G. Liva, L. Gaudio, T. Ninacs and T. Jerkovits, "Code design for short blocks: a
//!    survey," *Proc. EuCNC*, 2016.

use com_math::{db2lin, gauss_hermite, ln_1p_exp, qfunc};

/// The noise variance per dimension of BPSK at `ebn0_db` and the rate `rate`.
pub fn bpsk_noise_variance(ebn0_db: f64, rate: f64) -> f64 {
    1.0 / (2.0 * rate * db2lin(ebn0_db))
}

/// $1 - \log_{2}(1 + e^{-L})$ without overflow.
fn info_density(l: f64) -> f64 {
    1.0 - ln_1p_exp(-l) / std::f64::consts::LN_2
}

/// The first two moments of the information density at the noise variance `noise_var`.
fn moments(noise_var: f64) -> (f64, f64) {
    let (t, w) = gauss_hermite(64);
    let (mu, s) = (2.0 / noise_var, 2.0 / noise_var.sqrt());
    let sp = std::f64::consts::PI.sqrt();
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    for (ti, wi) in t.iter().zip(&w) {
        let i = info_density(mu + std::f64::consts::SQRT_2 * s * ti);
        m1 += wi * i / sp;
        m2 += wi * i * i / sp;
    }
    (m1, m2)
}

/// Capacity of the binary-input AWGN channel (bits per channel use) at the noise
/// variance `noise_var` per dimension.
pub fn biawgn_capacity(noise_var: f64) -> f64 {
    moments(noise_var).0
}

/// Dispersion of the binary-input AWGN channel (bits² per channel use).
pub fn biawgn_dispersion(noise_var: f64) -> f64 {
    let (m1, m2) = moments(noise_var);
    (m2 - m1 * m1).max(0.0)
}

/// The normal approximation of the smallest block error probability of an $(n, k)$
/// binary code at `ebn0_db` (Polyanskiy, Poor & Verdú 2010).
pub fn normal_approximation(n: usize, k: usize, ebn0_db: f64) -> f64 {
    let (nf, kf) = (n as f64, k as f64);
    let s2 = bpsk_noise_variance(ebn0_db, kf / nf);
    let (c, v) = (biawgn_capacity(s2), biawgn_dispersion(s2));
    qfunc((nf * c - kf + 0.5 * nf.log2()) / (nf * v).sqrt())
}

/// The smallest $E_{b}/N_{0}$ (dB) at which BPSK supports the rate `rate`
/// ($n \to \infty$): the solution of $C = R$.
pub fn shannon_limit_ebn0_db(rate: f64) -> f64 {
    com_math::bisect(|e| biawgn_capacity(bpsk_noise_variance(e, rate)) < rate, -2.0, 20.0, 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_limits_and_monotonicity() {
        assert!(biawgn_capacity(1e-3) > 0.999_999);
        assert!(biawgn_capacity(1e3) < 1e-3);
        // C(σ² = 1) = 0.4859 (the BI-AWGN capacity at Es/N0 = -3 dB)
        assert!((biawgn_capacity(1.0) - 0.4859).abs() < 2e-3);
        // the rate-1/3 limit is -0.495 dB, rate 1/2 0.187 dB
        assert!((shannon_limit_ebn0_db(1.0 / 3.0) + 0.495).abs() < 0.02);
        // longer codes get closer to capacity
        assert!(normal_approximation(1024, 512, 1.5) < normal_approximation(128, 64, 1.5));
    }
}

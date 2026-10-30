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
//! [`NormalApproximation`] is the same approximation for any channel of capacity $C$ and
//! dispersion $V$, $\log_{2} M^{*}(n, \varepsilon) \approx n C - \sqrt{n V}\, Q^{-1}(\varepsilon) +
//! \frac{1}{2} \log_{2} n$, with the channels of Polyanskiy, Poor & Verdú: the BSC,
//! $C = 1 - h(\delta)$ and $V = \delta(1 - \delta) \log_{2}^{2}\frac{1 - \delta}{\delta}$; the BEC,
//! $C = 1 - \delta$ and $V = \delta(1 - \delta)$, without the logarithmic term; the AWGN
//! channel of SNR $P$ with Gaussian codebooks, $C = \frac{1}{2}\log_{2}(1 + P)$ and
//! $V = \frac{P (P + 2)}{2 (P + 1)^{2}} \log_{2}^{2} e$ (their eqs. (184), (190), (296)) — and
//! the binary-input AWGN channel above.
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
    let s2 = bpsk_noise_variance(ebn0_db, k as f64 / n as f64);
    NormalApproximation::biawgn(s2).block_error(n, k)
}

/// The normal approximation of the coding rate at finite blocklength of a channel of
/// capacity `capacity` (bits per channel use) and dispersion `dispersion` (bits² per
/// channel use), with the term $\frac{1}{2}\log_{2} n$ when `half_log` (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalApproximation {
    /// The capacity $C$ (bits per channel use).
    pub capacity: f64,
    /// The dispersion $V$ (bits² per channel use).
    pub dispersion: f64,
    /// Whether the term $\frac{1}{2}\log_{2} n$ applies (all but the BEC).
    pub half_log: bool,
}

impl NormalApproximation {
    /// The binary symmetric channel of crossover probability `delta`.
    pub fn bsc(delta: f64) -> NormalApproximation {
        let h = com_math::entropy(&[delta, 1.0 - delta]);
        let v = delta * (1.0 - delta) * ((1.0 - delta) / delta).log2().powi(2);
        NormalApproximation { capacity: 1.0 - h, dispersion: v, half_log: true }
    }

    /// The binary erasure channel of erasure probability `delta`.
    pub fn bec(delta: f64) -> NormalApproximation {
        NormalApproximation { capacity: 1.0 - delta, dispersion: delta * (1.0 - delta), half_log: false }
    }

    /// The real AWGN channel of SNR `snr_lin` with Gaussian codebooks.
    pub fn awgn(snr_lin: f64) -> NormalApproximation {
        let p = snr_lin;
        let v = p * (p + 2.0) / (2.0 * (p + 1.0).powi(2)) * std::f64::consts::LOG2_E.powi(2);
        NormalApproximation { capacity: 0.5 * (1.0 + p).log2(), dispersion: v, half_log: true }
    }

    /// The binary-input AWGN channel of noise variance `noise_var` per dimension.
    pub fn biawgn(noise_var: f64) -> NormalApproximation {
        let (m1, m2) = moments(noise_var);
        NormalApproximation { capacity: m1, dispersion: (m2 - m1 * m1).max(0.0), half_log: true }
    }

    /// $\log_{2} M^{*}(n, \varepsilon)$: the largest number of information bits of a code of
    /// length `n` at the block error probability `eps`.
    pub fn max_bits(&self, n: usize, eps: f64) -> f64 {
        let nf = n as f64;
        nf * self.capacity - (nf * self.dispersion).sqrt() * com_math::qfuncinv(eps) + self.third(nf)
    }

    /// The smallest block error probability of a code of length `n` carrying `k` bits.
    pub fn block_error(&self, n: usize, k: usize) -> f64 {
        let nf = n as f64;
        qfunc((nf * self.capacity - k as f64 + self.third(nf)) / (nf * self.dispersion).sqrt())
    }

    /// The smallest blocklength that reaches the rate `rate` (bits per channel use) at the
    /// block error probability `eps` (real-valued, by bisection), or `None` beyond
    /// capacity.
    pub fn blocklength(&self, rate: f64, eps: f64) -> Option<f64> {
        if rate >= self.capacity {
            return None;
        }
        let gap = |n: f64| {
            n * (self.capacity - rate) - (n * self.dispersion).sqrt() * com_math::qfuncinv(eps) + self.third(n)
        };
        let mut hi = 2.0;
        while gap(hi) < 0.0 {
            hi *= 2.0;
            if hi > 1e12 {
                return None;
            }
        }
        Some(com_math::bisect(|n| gap(n) < 0.0, 1.0, hi, 200))
    }

    fn third(&self, n: f64) -> f64 {
        if self.half_log { 0.5 * n.log2() } else { 0.0 }
    }
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

    #[test]
    fn the_approximation_inverts_itself() {
        let c = NormalApproximation::bsc(0.11);
        let k = c.max_bits(1000, 1e-3);
        assert!((c.block_error(1000, k.round() as usize) / 1e-3).ln().abs() < 0.2);
        assert!(NormalApproximation::bec(0.5).blocklength(0.5, 1e-3).is_none());
    }
}

/// Checks against the published tables (`references/`).
#[cfg(test)]
mod literature {
    use super::*;

    /// Y. Polyanskiy, H. V. Poor and S. Verdú, "Channel coding rate in the finite
    /// blocklength regime," IEEE Trans. Inf. Theory 56(5), 2307–2359, 2010, Table I, the
    /// column "Normal Approx.": the blocklength that reaches $R = 0.9 C$ is about 955 for
    /// the BEC(0.5) at $\varepsilon = 10^{-3}$, 3150 for the BSC(0.11), 2750 for the AWGN
    /// channel at 0 dB and $10^{-3}$, 190 at 20 dB and $10^{-6}$ (here 955.0, 3164, 2748,
    /// 188.8: within 1 %).
    #[test]
    fn polyanskiy_poor_verdu_2010_table_i() {
        let cases = [
            (NormalApproximation::bec(0.5), 1e-3, 955.0),
            (NormalApproximation::bsc(0.11), 1e-3, 3150.0),
            (NormalApproximation::awgn(1.0), 1e-3, 2750.0),
            (NormalApproximation::awgn(100.0), 1e-6, 190.0),
        ];
        for (c, eps, want) in cases {
            let n = c.blocklength(0.9 * c.capacity, eps).unwrap();
            assert!((n / want - 1.0).abs() < 0.01, "{c:?}: n = {n} vs {want}");
        }
    }
}

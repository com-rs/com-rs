//! Second- and fourth-order statistics of complex sequences.
//!
//! - [`Moments`]: the moments $\mathbb{E}[r^{2}]$, $\mathbb{E}|r|^{2}$, $\mathbb{E}[r^{4}]$,
//!   $\mathbb{E}[r^{2}|r|^{2}]$ and $\mathbb{E}|r|^{4}$, of a sample (the means) or of a
//!   discrete distribution (the points and their probabilities): the cumulants of the
//!   blind estimators, the M2M4 estimator and the radius of the constant modulus
//!   algorithm all start from them;
//! - [`Correlation`]: the cross-correlation $\sum_{k} a_{k} b_{k}^{*}$ of two sequences
//!   and their energies, with the normalised correlation
//!
//! $$
//! \rho = \frac{\left|\sum_{k} a_{k} b_{k}^{*}\right|}{\sqrt{\sum_{k} |a_{k}|^{2}
//! \sum_{k} |b_{k}|^{2}}} \in [0, 1],
//! $$
//!
//! the quality of a frame synchronisation, the effective SNR of a sequence against
//! another and the test of the singularity of a 2 × 2 equaliser.
//!
//! # Examples
//! ```
//! use com_math::C64;
//! use com_math::stats::{Correlation, Moments};
//! let a = [C64::new(1.0, 0.0), C64::new(0.0, 1.0)];
//! let m = Moments::sample(&a);
//! assert_eq!((m.m21, m.m42), (1.0, 1.0));
//! assert_eq!(m.m20, C64::new(0.0, 0.0)); // 1 and -1
//! let b: Vec<C64> = a.iter().map(|v| v * C64::new(0.0, 2.0)).collect();
//! assert!((Correlation::between(&a, &b).rho() - 1.0).abs() < 1e-15);
//! ```

use crate::C64;

/// The moments of order two and four of a complex random variable $r$ (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    /// $\mathbb{E}[r^{2}]$.
    pub m20: C64,
    /// $\mathbb{E}|r|^{2}$.
    pub m21: f64,
    /// $\mathbb{E}[r^{4}]$.
    pub m40: C64,
    /// $\mathbb{E}[r^{2}|r|^{2}]$.
    pub m41: C64,
    /// $\mathbb{E}|r|^{4}$.
    pub m42: f64,
}

impl Moments {
    /// The sample moments of `x`: the sums divided by the number of samples (`NaN`
    /// without a sample).
    pub fn sample(x: &[C64]) -> Moments {
        let n = x.len() as f64;
        let s = Moments::sums(x.iter().map(|&v| (v, 1.0)));
        Moments { m20: s.m20 / n, m21: s.m21 / n, m40: s.m40 / n, m41: s.m41 / n, m42: s.m42 / n }
    }

    /// The moments of the distribution of the points `x` with the probabilities `p` (the
    /// pairs of the shorter of the two).
    pub fn of_distribution(x: &[C64], p: &[f64]) -> Moments {
        Moments::sums(x.iter().copied().zip(p.iter().copied()))
    }

    /// The weighted sums $\sum_{k} w_{k} r_{k}^{2}$, … (a weight of 1 is exact).
    fn sums(rw: impl Iterator<Item = (C64, f64)>) -> Moments {
        let zero = C64::new(0.0, 0.0);
        let mut m = Moments { m20: zero, m21: 0.0, m40: zero, m41: zero, m42: 0.0 };
        for (x, w) in rw {
            let a = x.norm_sqr();
            let x2 = x * x;
            m.m20 += x2 * w;
            m.m21 += a * w;
            m.m40 += x2 * x2 * w;
            m.m41 += x2 * a * w;
            m.m42 += a * a * w;
        }
        m
    }
}

/// The cross-correlation of two sequences and their energies (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correlation {
    /// $\sum_{k} a_{k} b_{k}^{*}$.
    pub cross: C64,
    /// $\sum_{k} |a_{k}|^{2}$.
    pub energy_a: f64,
    /// $\sum_{k} |b_{k}|^{2}$.
    pub energy_b: f64,
}

impl Correlation {
    /// The correlation of `a` with `b` (the pairs of the shorter of the two).
    pub fn between(a: &[C64], b: &[C64]) -> Correlation {
        let mut c = Correlation { cross: C64::new(0.0, 0.0), energy_a: 0.0, energy_b: 0.0 };
        for (u, v) in a.iter().zip(b) {
            c.cross += u * v.conj();
            c.energy_a += u.norm_sqr();
            c.energy_b += v.norm_sqr();
        }
        c
    }

    /// The normalised correlation $\rho = |c| / \sqrt{E_{a} E_{b}}$, 0 when a sequence
    /// has no energy.
    pub fn rho(&self) -> f64 {
        if self.energy_a > 0.0 && self.energy_b > 0.0 {
            self.cross.norm() / (self.energy_a * self.energy_b).sqrt()
        } else {
            0.0
        }
    }

    /// Its square $\rho^{2} = |c|^{2} / (E_{a} E_{b})$ (computed as such, not as the square
    /// of [`Correlation::rho`]), 0 when a sequence has no energy.
    pub fn rho_squared(&self) -> f64 {
        if self.energy_a > 0.0 && self.energy_b > 0.0 {
            self.cross.norm_sqr() / (self.energy_a * self.energy_b)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_moments_of_a_sample_are_those_of_its_uniform_distribution() {
        let x: Vec<C64> = (0..7).map(|k| C64::new(f64::from(k) - 3.0, 0.5 * f64::from(k))).collect();
        let s = Moments::sample(&x);
        let d = Moments::of_distribution(&x, &[1.0 / 7.0; 7]);
        assert!((s.m21 - d.m21).abs() < 1e-14 && (s.m42 - d.m42).abs() < 1e-12);
        assert!((s.m40 - d.m40).norm() < 1e-12 && (s.m41 - d.m41).norm() < 1e-12 && (s.m20 - d.m20).norm() < 1e-14);
        let e = |k: i32| x.iter().map(|v| v.norm_sqr().powi(k)).sum::<f64>() / 7.0;
        assert_eq!((s.m21, s.m42), (e(1), e(2)));
    }

    #[test]
    fn the_correlation_is_one_for_proportional_sequences_and_zero_without_energy() {
        let a = [C64::new(1.0, 2.0), C64::new(-0.5, 0.1), C64::new(0.0, -1.0)];
        let b: Vec<C64> = a.iter().map(|v| v * C64::new(-1.5, 0.7)).collect();
        let c = Correlation::between(&a, &b);
        assert!((c.rho() - 1.0).abs() < 1e-15 && (c.rho_squared() - 1.0).abs() < 1e-15);
        let z = Correlation::between(&a, &[C64::new(0.0, 0.0); 3]);
        assert_eq!((z.rho(), z.rho_squared()), (0.0, 0.0));
    }
}

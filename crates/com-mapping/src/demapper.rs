//! Soft demapping: log-likelihood ratios of the bits of received symbols.
//!
//! The soft demapper is the inverse of the mapper and belongs to the same crate.
//!
//! # Model
//!
//! For every symbol and every bit position,
//!
//! $$
//! L(b) = \ln \sum_{x:\\, b = 0} p(x)\\, e^{-|y - x|^{2} / \sigma^{2}}
//!      - \ln \sum_{x:\\, b = 1} p(x)\\, e^{-|y - x|^{2} / \sigma^{2}}
//! $$
//!
//! with $\sigma^{2}$ the variance of the complex noise, one per symbol with
//! [`Constellation::llr_with_variances`]. A positive LLR favours bit 0, the library-wide
//! convention of `com_core::signal`. The sums are computed with log-sum-exp, so that
//! they stay finite at any SNR; [`Jacobian::MaxLog`] keeps only the largest term of each
//! sum.
//!
//! [`Constellation`] implements the verb traits [`Map`] and [`Demap`] on the
//! multi-stream containers, the latter with the exact (not max-log) LLRs and uniform
//! priors.
//!
//! # Examples
//! ```
//! use com_core::prelude::*;
//! use com_mapping::constellation::Constellation;
//! let c = Constellation::qam(4)?.normalized();
//! let bits = Bits::from_stream(vec![0, 1, 1, 1])?;
//! let x = c.map(&bits)?;
//! let llr = c.demap(&x, 0.01)?;
//! // positive LLR: bit 0
//! let hard: Vec<u8> = com_core::signal::hard_decisions(llr.as_slice());
//! assert_eq!(hard, bits.as_slice());
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::constellation::{Constellation, log_priors};
use com_core::signal::{Bits, Llr, Symbols};
use com_core::traits::{Demap, Map};
use com_core::{Result, ensure, same_len};
use com_math::{C64, Jacobian};

/// The metrics of one symbol over the two halves of its labels for one bit: the label
/// $i$ of a point is in half $b$ when bit `shift` of $i$ is $b$, and every half keeps its
/// largest metric $\mu_{b}$ and the sum $s_{b} = \sum_{i} e^{\ell_{i} - \mu_{b}}$, so that
///
/// $$
/// \ln \sum_{i \in b} e^{\ell_{i}} = \mu_{b} + \ln s_{b}, \qquad
/// L = \mu_{0} + \ln s_{0} - \mu_{1} - \ln s_{1}
/// $$
///
/// (with max-log, $\mu_{b}$ and $\mu_{0} - \mu_{1}$). The kernel of the bitwise demapper
/// ([`Constellation::llr`]), of the bitwise closed forms ([`crate::theory`]) and of the
/// bit-metric rates of `com_metrics`: the metric row $\ell_{i}$ (a log-likelihood plus
/// a log-prior) is theirs, the sums are these.
///
/// # Examples
/// ```
/// use com_mapping::demapper::LabelHalves;
/// use com_math::Jacobian;
/// // four labels, bit 1 (the most significant): halves {0, 1} and {2, 3}
/// let h = LabelHalves::new(&[0.0, -1.0, -3.0, -2.0], 1, Jacobian::Exact);
/// assert!((h.log_sum(0) - (1.0 + (-1f64).exp()).ln()).abs() < 1e-15);
/// assert_eq!(LabelHalves::new(&[0.0, -1.0, -3.0, -2.0], 1, Jacobian::MaxLog).llr(), 2.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelHalves {
    max: [f64; 2],
    sum: [f64; 2],
    jacobian: Jacobian,
}

impl LabelHalves {
    /// The halves of the metric row `metric` (one value per label) for bit `shift` of
    /// the labels; with [`Jacobian::MaxLog`], the maxima only.
    pub fn new(metric: &[f64], shift: usize, jacobian: Jacobian) -> LabelHalves {
        // the log-sum-exp of the two halves without allocation: maxima, then sums
        let (mut m0, mut m1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (i, &v) in metric.iter().enumerate() {
            if (i >> shift) & 1 == 0 {
                m0 = m0.max(v);
            } else {
                m1 = m1.max(v);
            }
        }
        let mut sum = [1.0, 1.0];
        if jacobian.is_exact() && (m0.is_finite() || m1.is_finite()) {
            let (mut s0, mut s1) = (0.0, 0.0);
            for (i, &v) in metric.iter().enumerate() {
                if (i >> shift) & 1 == 0 {
                    s0 += (v - m0).exp();
                } else {
                    s1 += (v - m1).exp();
                }
            }
            sum = [s0, s1];
        }
        LabelHalves { max: [m0, m1], sum, jacobian }
    }

    /// $\ln \sum_{i \in b} e^{\ell_{i}}$ over the half `bit` (0 or 1): $\mu_{b} + \ln
    /// s_{b}$, or $\mu_{b}$ with max-log or when $\mu_{b}$ is not finite.
    pub fn log_sum(&self, bit: usize) -> f64 {
        let m = self.max[bit & 1];
        if !self.jacobian.is_exact() || !m.is_finite() { m } else { m + self.sum[bit & 1].ln() }
    }

    /// The LLR of the bit, $\mu_{0} + \ln s_{0} - \mu_{1} - \ln s_{1}$ (positive for a 0),
    /// or $\mu_{0} - \mu_{1}$ with max-log or when a maximum is not finite.
    pub fn llr(&self) -> f64 {
        let [m0, m1] = self.max;
        if !self.jacobian.is_exact() || !m0.is_finite() || !m1.is_finite() {
            m0 - m1
        } else {
            m0 + self.sum[0].ln() - m1 - self.sum[1].ln()
        }
    }
}

/// The LLRs of the $b = \log_{2} M$ bits of one symbol from its metric row (the $M$
/// values $\ell_{i}$ of [`LabelHalves`]) into `out[..b]`, the most significant bit
/// first.
///
/// # Panics
/// If `out` has fewer than $b$ values.
pub fn llr_from_metrics(metric: &[f64], jacobian: Jacobian, out: &mut [f64]) {
    let b = metric.len().trailing_zeros() as usize;
    for (o, k) in out[..b].iter_mut().zip((0..b).rev()) {
        *o = LabelHalves::new(metric, k, jacobian).llr();
    }
}

impl Constellation {
    /// Log-likelihood ratios of the bits, `b` per symbol (see the module documentation).
    ///
    /// `noise_var` is the variance of the complex noise `E|w|²` (for a real
    /// constellation over a real channel, pass twice the variance of the real noise, as
    /// OptiCommPy does); `px` are the prior probabilities of the points (uniform if
    /// `None`, normalised otherwise).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a noise variance that is not positive,
    /// [`com_core::Error::Shape`] for priors of another length than the constellation,
    /// [`com_core::Error::Config`] for negative or all-zero priors.
    pub fn llr(&self, z: &[C64], noise_var: f64, px: Option<&[f64]>, jacobian: Jacobian) -> Result<Vec<f64>> {
        let mut out = vec![0.0; z.len() * self.bits_per_symbol()];
        self.llr_into(z, noise_var, px, jacobian, &mut out)?;
        Ok(out)
    }

    /// [`Constellation::llr`] into the `z.len() log2 M` values of `out` (no allocation
    /// with uniform priors).
    ///
    /// # Errors
    /// As [`Constellation::llr`], and [`com_core::Error::Shape`] for an `out` of another
    /// length.
    ///
    /// # Examples
    /// ```
    /// use com_mapping::constellation::Constellation;
    /// use com_math::Jacobian;
    /// let c = Constellation::qam(16)?.normalized();
    /// let z = [c.points()[5], c.points()[12]];
    /// let mut out = [0.0; 8];
    /// c.llr_into(&z, 0.1, None, Jacobian::Exact, &mut out)?;
    /// assert_eq!(out[..], c.llr(&z, 0.1, None, Jacobian::Exact)?[..]);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn llr_into(
        &self,
        z: &[C64],
        noise_var: f64,
        px: Option<&[f64]>,
        jacobian: Jacobian,
        out: &mut [f64],
    ) -> Result<()> {
        ensure_variance(noise_var)?;
        same_len(out.len(), z.len() * self.bits_per_symbol(), "LLRs and bits of the symbols")?;
        self.llr_kernel(z, |_| noise_var, self.log_priors_of(px)?.as_deref(), jacobian, out);
        Ok(())
    }

    /// [`Constellation::llr`] with one noise variance per symbol, `noise_vars[i]` that of
    /// `z[i]` (the output of a linear detector, of Alamouti's combiner).
    ///
    /// # Errors
    /// As [`Constellation::llr`], and [`com_core::Error::Shape`] for one variance per
    /// symbol missing.
    ///
    /// # Examples
    /// ```
    /// use com_mapping::constellation::Constellation;
    /// use com_math::Jacobian;
    /// let c = Constellation::qam(4)?.normalized();
    /// let z = [c.points()[0], c.points()[3]];
    /// let both = c.llr_with_variances(&z, &[0.1, 0.4], None, Jacobian::Exact)?;
    /// assert_eq!(both[2..], c.llr(&z[1..], 0.4, None, Jacobian::Exact)?[..]);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn llr_with_variances(
        &self,
        z: &[C64],
        noise_vars: &[f64],
        px: Option<&[f64]>,
        jacobian: Jacobian,
    ) -> Result<Vec<f64>> {
        let mut out = vec![0.0; z.len() * self.bits_per_symbol()];
        self.llr_with_variances_into(z, noise_vars, px, jacobian, &mut out)?;
        Ok(out)
    }

    /// [`Constellation::llr_with_variances`] into the `z.len() log2 M` values of `out`.
    ///
    /// # Errors
    /// As [`Constellation::llr_with_variances`], and [`com_core::Error::Shape`] for an
    /// `out` of another length.
    pub fn llr_with_variances_into(
        &self,
        z: &[C64],
        noise_vars: &[f64],
        px: Option<&[f64]>,
        jacobian: Jacobian,
        out: &mut [f64],
    ) -> Result<()> {
        same_len(noise_vars.len(), z.len(), "noise variances and symbols")?;
        for &v in noise_vars {
            ensure_variance(v)?;
        }
        same_len(out.len(), z.len() * self.bits_per_symbol(), "LLRs and bits of the symbols")?;
        self.llr_kernel(z, |i| noise_vars[i], self.log_priors_of(px)?.as_deref(), jacobian, out);
        Ok(())
    }

    /// The log-priors of `px`, checked and normalised; `None` for uniform priors, whose
    /// logarithm $-\ln M$ the kernel uses without a table.
    fn log_priors_of(&self, px: Option<&[f64]>) -> Result<Option<Vec<f64>>> {
        Ok(match px {
            None => None,
            Some(_) => Some(log_priors(&self.priors(px)?)),
        })
    }

    /// The LLRs of `z` into `out`, the noise variance of symbol `i` being `var(i)`, the
    /// log-priors `log_px` (uniform if `None`).
    fn llr_kernel(
        &self,
        z: &[C64],
        var: impl Fn(usize) -> f64,
        log_px: Option<&[f64]>,
        jacobian: Jacobian,
        out: &mut [f64],
    ) {
        let m = self.order();
        let b = self.bits_per_symbol();
        let uniform = -(m as f64).ln();
        let mut metric = vec![0.0; m];
        let points = self.points();
        for (n, (&y, o)) in z.iter().zip(out.chunks_exact_mut(b)).enumerate() {
            let sigma2 = var(n);
            for i in 0..m {
                metric[i] = -(y - points[i]).norm_sqr() / sigma2 + log_px.map_or(uniform, |l| l[i]);
            }
            llr_from_metrics(&metric, jacobian, o);
        }
    }
}

/// The LLRs $2 y / \sigma^{2}$ of real BPSK symbols ([`crate::constellation::bpsk`],
/// $0 \mapsto +1$) received in real Gaussian noise of variance `noise_var` per
/// dimension.
///
/// # Examples
/// ```
/// use com_mapping::demapper::bpsk_llr;
/// assert_eq!(bpsk_llr(&[0.5, -1.0], 0.5), [2.0, -4.0]);
/// ```
pub fn bpsk_llr(y: &[f64], noise_var: f64) -> Vec<f64> {
    y.iter().map(|v| 2.0 * v / noise_var).collect()
}

fn ensure_variance(noise_var: f64) -> Result<()> {
    ensure(noise_var > 0.0 && noise_var.is_finite(), || format!("the noise variance must be positive, got {noise_var}"))
}

impl Map for Constellation {
    fn map_into(&self, bits: &Bits, out: &mut Symbols) -> Result<()> {
        let b = self.bits_per_symbol();
        if bits.len() % b != 0 {
            return Err(com_core::Error::Shape(format!(
                "{} bits per stream is not a multiple of {b} bits per symbol",
                bits.len()
            )));
        }
        out.resize(bits.n_streams(), bits.len() / b);
        for s in 0..bits.n_streams() {
            let idx = self.indices(bits.stream(s))?;
            for (o, i) in out.stream_mut(s).iter_mut().zip(idx) {
                *o = self.points()[i];
            }
        }
        Ok(())
    }
}

impl Demap for Constellation {
    fn demap_into(&self, y: &Symbols, noise_var: f64, out: &mut Llr) -> Result<()> {
        out.resize(y.n_streams(), y.len() * self.bits_per_symbol());
        for s in 0..y.n_streams() {
            self.llr_into(y.stream(s), noise_var, None, Jacobian::Exact, out.stream_mut(s))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llr_signs_and_max_log() {
        let c = Constellation::qam(16).unwrap().normalized();
        let x = c.points()[0b1010];
        let l = c.llr(&[x], 0.01, None, Jacobian::Exact).unwrap();
        let expected = [1u8, 0, 1, 0];
        for (v, b) in l.iter().zip(expected) {
            assert!(if b == 0 { *v > 0.0 } else { *v < 0.0 }, "{l:?}");
        }
        let lm = c.llr(&[x], 0.01, None, Jacobian::MaxLog).unwrap();
        for (a, b) in l.iter().zip(&lm) {
            assert!((a - b).abs() < 1e-3 * a.abs().max(1.0));
        }
    }

    /// The demapper as written before the shared kernel: the metric row with a table of
    /// log-priors, then the halves of every bit.
    fn former_llr(c: &Constellation, z: &[C64], sigma2: f64, log_px: &[f64], max_log: bool) -> Vec<f64> {
        let (m, b) = (c.order(), c.bits_per_symbol());
        let mut out = Vec::new();
        let mut metric = vec![0.0; m];
        for &y in z {
            for i in 0..m {
                metric[i] = -(y - c.points()[i]).norm_sqr() / sigma2 + log_px[i];
            }
            for k in (0..b).rev() {
                let (mut m0, mut m1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                for (i, &v) in metric.iter().enumerate() {
                    if (i >> k) & 1 == 0 {
                        m0 = m0.max(v);
                    } else {
                        m1 = m1.max(v);
                    }
                }
                if max_log || !m0.is_finite() || !m1.is_finite() {
                    out.push(m0 - m1);
                    continue;
                }
                let (mut s0, mut s1) = (0.0, 0.0);
                for (i, &v) in metric.iter().enumerate() {
                    if (i >> k) & 1 == 0 {
                        s0 += (v - m0).exp();
                    } else {
                        s1 += (v - m1).exp();
                    }
                }
                out.push(m0 + s0.ln() - m1 - s1.ln());
            }
        }
        out
    }

    #[test]
    fn the_shared_kernel_gives_the_former_llrs_bit_for_bit() {
        let mut rng = com_core::rng::Rng::new(77);
        for c in [
            Constellation::qam(16).unwrap().normalized(),
            Constellation::psk(8).unwrap().normalized(),
            Constellation::pam(4).unwrap().normalized(),
            Constellation::qam(256).unwrap().normalized(),
        ] {
            let m = c.order();
            let z: Vec<C64> = (0..200).map(|i| c.points()[i % m] + rng.complex_gaussian() * 0.3).collect();
            let shaped: Vec<f64> = (0..m).map(|i| 1.0 + (i % 3) as f64).collect();
            let lp = log_priors(&c.priors(Some(&shaped)).unwrap());
            let uniform = vec![-(m as f64).ln(); m];
            for (px, table) in [(None, &uniform), (Some(&shaped[..]), &lp)] {
                for max_log in [false, true] {
                    for s2 in [1e-3, 0.1, 2.0] {
                        let new = c.llr(&z, s2, px, Jacobian::from_max_log(max_log)).unwrap();
                        let old = former_llr(&c, &z, s2, table, max_log);
                        assert!(new.iter().zip(&old).all(|(a, b)| a.to_bits() == b.to_bits()), "M {m}, {s2}");
                    }
                }
            }
            let mut out = vec![0.0; c.bits_per_symbol() + 1];
            assert!(c.llr_into(&z[..1], 0.1, None, Jacobian::Exact, &mut out).is_err());
        }
    }

    #[test]
    fn multi_stream_map_demap_round_trip() {
        let c = Constellation::qam(16).unwrap().normalized();
        let mut rng = com_core::rng::Rng::new(5);
        let streams: Vec<Vec<u8>> = (0..3).map(|_| (0..64).map(|_| rng.bit()).collect()).collect();
        let bits = Bits::from_streams(&streams).unwrap();
        let x = c.map(&bits).unwrap();
        assert_eq!((x.n_streams(), x.len()), (3, 16));
        let llr = c.demap(&x, 1e-3).unwrap();
        let hard: Vec<u8> = com_core::signal::hard_decisions(llr.as_slice());
        assert_eq!(hard, bits.as_slice());
        assert!(c.map(&Bits::from_stream(vec![0; 3]).unwrap()).is_err());
        let _ = C64::new(0.0, 0.0);
    }
}

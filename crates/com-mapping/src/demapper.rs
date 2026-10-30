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
//! multi-stream containers: [`Demap::demap_into`] gives the exact LLRs under uniform
//! priors, [`Demap::demap_with_into`] takes the log-priors $\ln p(x)$ of the points
//! (probabilistic shaping) and the [`Jacobian`] in a [`Demapping`]. Both write into the
//! caller's [`Llr`] and allocate nothing once its shape is stable (the metric row of a
//! symbol is on the stack up to [`STACK_POINTS`] points); with log-priors from
//! [`log_priors`] of the normalised [`Constellation::priors`], their LLRs are those of
//! [`Constellation::llr`] with the probabilities, bit for bit.
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
//! // shaped priors and max-log: the log-priors of the normalised probabilities
//! let px = [0.4, 0.3, 0.2, 0.1];
//! let lp = com_mapping::constellation::log_priors(&c.priors(Some(&px))?);
//! let shaped = c.demap_with(&x, 0.01, Demapping { log_priors: Some(&lp), jacobian: Jacobian::MaxLog })?;
//! assert_eq!(shaped.as_slice(), c.llr(x.as_slice(), 0.01, Some(&px), Jacobian::MaxLog)?);
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::constellation::{Constellation, log_priors};
use com_core::signal::{Bits, Llr, Symbols};
use com_core::traits::{Demap, Demapping, Map};
use com_core::{Result, ensure, same_len};
use com_math::{C64, Jacobian};

/// The largest constellation whose metric row (one `f64` per point, 8 KiB) the demapper
/// keeps on the stack; a larger one allocates the row once per call.
pub const STACK_POINTS: usize = 1024;

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
    /// with uniform priors up to [`STACK_POINTS`] points).
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

    /// `Ok` for log-priors of one value per point, none NaN or $+\infty$, one at least
    /// finite (a point possible).
    fn check_log_priors(&self, log_px: &[f64]) -> Result<()> {
        same_len(log_px.len(), self.order(), "log-priors and constellation points")?;
        ensure(log_px.iter().all(|v| !v.is_nan() && *v < f64::INFINITY), || {
            "the log-priors must be below +inf and not NaN".into()
        })?;
        ensure(log_px.iter().any(|v| v.is_finite()), || "the log-priors leave no point possible".into())
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
        let mut stack = [0.0; STACK_POINTS];
        let mut heap = Vec::new();
        let metric: &mut [f64] = if m <= STACK_POINTS {
            &mut stack[..m]
        } else {
            heap.resize(m, 0.0);
            &mut heap
        };
        let points = self.points();
        for (n, (&y, o)) in z.iter().zip(out.chunks_exact_mut(b)).enumerate() {
            let sigma2 = var(n);
            for i in 0..m {
                metric[i] = -(y - points[i]).norm_sqr() / sigma2 + log_px.map_or(uniform, |l| l[i]);
            }
            llr_from_metrics(metric, jacobian, o);
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

/// The soft demapper of the constellation: the kernel of [`Constellation::llr_into`] on
/// every stream, with the log-priors and the logarithm of the [`Demapping`].
impl Demap for Constellation {
    /// # Errors
    /// [`com_core::Error::Config`] for a noise variance that is not positive and finite,
    /// or log-priors with a NaN or $+\infty$, or none finite; [`com_core::Error::Shape`]
    /// for log-priors of another length than the constellation.
    fn demap_with_into(&self, y: &Symbols, noise_var: f64, demapping: Demapping<'_>, out: &mut Llr) -> Result<()> {
        ensure_variance(noise_var)?;
        if let Some(lp) = demapping.log_priors {
            self.check_log_priors(lp)?;
        }
        out.resize(y.n_streams(), y.len() * self.bits_per_symbol());
        for s in 0..y.n_streams() {
            self.llr_kernel(y.stream(s), |_| noise_var, demapping.log_priors, demapping.jacobian, out.stream_mut(s));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two streams of a 16-QAM through a fixed perturbation.
    fn received(c: &Constellation) -> Symbols {
        let p = c.points();
        let row = |k: usize| {
            (0..40)
                .map(|i| p[(7 * i + k) % 16] + C64::new(0.3 * ((i % 5) as f64 - 2.0), 0.1 * (i % 3) as f64))
                .collect()
        };
        Symbols::from_streams(&[row(0), row(3)]).unwrap()
    }

    /// The trait LLRs are those of `Constellation::llr`, bit for bit: uniform priors and
    /// shaped priors (their log-priors from `log_priors` of the normalised `priors`),
    /// exact and max-log.
    #[test]
    fn demap_with_equals_llr_with_and_without_priors() {
        let c = Constellation::qam(16).unwrap().normalized();
        let y = received(&c);
        let px: Vec<f64> = (1..=16).map(|i| f64::from(i).sqrt()).collect();
        let lp = log_priors(&c.priors(Some(&px)).unwrap());
        for jacobian in [Jacobian::Exact, Jacobian::MaxLog] {
            for (log_priors, probabilities) in [(None, None), (Some(&lp[..]), Some(&px[..]))] {
                let got = c.demap_with(&y, 0.2, Demapping { log_priors, jacobian }).unwrap();
                for s in 0..2 {
                    assert_eq!(
                        got.stream(s),
                        c.llr(y.stream(s), 0.2, probabilities, jacobian).unwrap(),
                        "{jacobian:?}"
                    );
                }
            }
        }
        assert_eq!(c.demap(&y, 0.2).unwrap(), c.demap_with(&y, 0.2, Demapping::default()).unwrap());
    }

    /// The output keeps its storage, a constellation above `STACK_POINTS` points too.
    #[test]
    fn demap_with_into_reuses_the_output() {
        for c in [Constellation::qam(16).unwrap(), Constellation::qam(4096).unwrap()] {
            let y = Symbols::from_streams(&[c.points()[..8].to_vec(), c.points()[8..16].to_vec()]).unwrap();
            let mut out = Llr::default();
            c.demap_into(&y, 0.5, &mut out).unwrap();
            let ptr = out.as_slice().as_ptr();
            c.demap_with_into(&y, 0.5, Demapping { log_priors: None, jacobian: Jacobian::MaxLog }, &mut out).unwrap();
            assert_eq!(out.as_slice().as_ptr(), ptr);
            assert_eq!(out.stream(1), c.llr(y.stream(1), 0.5, None, Jacobian::MaxLog).unwrap());
        }
    }

    /// Log-priors: one per point (`Shape`), no NaN or +inf, one finite (`Config`); an
    /// unnormalised row gives the same decisions.
    #[test]
    fn demap_with_checks_the_log_priors() {
        let c = Constellation::qam(4).unwrap().normalized();
        let y = Symbols::from_stream(c.points().to_vec()).unwrap();
        let with = |lp: &[f64]| c.demap_with(&y, 0.1, Demapping { log_priors: Some(lp), jacobian: Jacobian::Exact });
        assert!(matches!(with(&[0.0; 3]), Err(com_core::Error::Shape(_))));
        assert!(matches!(with(&[0.0, f64::NAN, 0.0, 0.0]), Err(com_core::Error::Config(_))));
        assert!(matches!(with(&[0.0, f64::INFINITY, 0.0, 0.0]), Err(com_core::Error::Config(_))));
        assert!(matches!(with(&[f64::NEG_INFINITY; 4]), Err(com_core::Error::Config(_))));
        assert!(matches!(c.demap(&y, 0.0), Err(com_core::Error::Config(_))));
        let shifted = with(&[3.0; 4]).unwrap();
        let uniform = c.demap(&y, 0.1).unwrap();
        for (a, b) in shifted.as_slice().iter().zip(uniform.as_slice()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

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

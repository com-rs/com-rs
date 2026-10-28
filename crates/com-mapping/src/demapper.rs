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
//! with $\sigma^{2}$ the variance of the complex noise (positive LLR: bit 0 more likely,
//! the library-wide convention of `com_core::signal`), computed with log-sum-exp so
//! that it stays finite at any SNR; `max_log` keeps only the largest term of each sum.
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
//! let hard: Vec<u8> = llr.as_slice().iter().map(|l| u8::from(*l < 0.0)).collect();
//! assert_eq!(hard, bits.as_slice());
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::constellation::{Constellation, log_priors};
use com_core::signal::{Bits, Llr, Symbols};
use com_core::traits::{Demap, Map};
use com_core::{Result, ensure, same_len};
use com_math::C64;
use ndarray::Array2;

impl Constellation {
    /// Log-likelihood ratios of the bits, `b` per symbol (see the module documentation).
    ///
    /// `sigma2` is the variance of the complex noise `E|w|²` (for a real constellation
    /// over a real channel, pass twice the variance of the real noise, as OptiCommPy
    /// does); `px` are the prior probabilities of the points (uniform if `None`).
    pub fn llr(&self, z: &[C64], sigma2: f64, px: Option<&[f64]>, max_log: bool) -> Vec<f64> {
        let m = self.order();
        let b = self.bits_per_symbol();
        let log_px: Vec<f64> = match px {
            Some(p) => log_priors(p),
            None => vec![-(m as f64).ln(); m],
        };
        let mut out = Vec::with_capacity(z.len() * b);
        let mut metric = vec![0.0; m];
        let points = self.points();
        for &y in z {
            for i in 0..m {
                metric[i] = -(y - points[i]).norm_sqr() / sigma2 + log_px[i];
            }
            for k in (0..b).rev() {
                // log-sum-exp of the two halves without allocation: maxima, then sums
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
            let idx = self.indices(bits.stream_slice(s))?;
            for (o, i) in out.stream_mut(s).iter_mut().zip(idx) {
                *o = self.points()[i];
            }
        }
        Ok(())
    }
}

impl Demap for Constellation {
    fn demap_into(&self, y: &Symbols, noise_var: f64, out: &mut Llr) -> Result<()> {
        ensure(noise_var > 0.0, || format!("the noise variance must be positive, got {noise_var}"))?;
        let b = self.bits_per_symbol();
        let rows: Vec<Vec<f64>> =
            (0..y.n_streams()).map(|s| self.llr(y.stream_slice(s), noise_var, None, false)).collect();
        let n = y.len() * b;
        for r in &rows {
            same_len(r.len(), n, "LLRs")?;
        }
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        *out = Llr::new(
            Array2::from_shape_vec((y.n_streams(), n), flat).map_err(|e| com_core::Error::Shape(e.to_string()))?,
        );
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
        let l = c.llr(&[x], 0.01, None, false);
        let expected = [1u8, 0, 1, 0];
        for (v, b) in l.iter().zip(expected) {
            assert!(if b == 0 { *v > 0.0 } else { *v < 0.0 }, "{l:?}");
        }
        let lm = c.llr(&[x], 0.01, None, true);
        for (a, b) in l.iter().zip(&lm) {
            assert!((a - b).abs() < 1e-3 * a.abs().max(1.0));
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
        let hard: Vec<u8> = llr.as_slice().iter().map(|l| u8::from(*l < 0.0)).collect();
        assert_eq!(hard, bits.as_slice());
        assert!(c.map(&Bits::from_stream(vec![0; 3]).unwrap()).is_err());
        let _ = C64::new(0.0, 0.0);
    }
}

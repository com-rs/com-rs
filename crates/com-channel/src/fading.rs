//! Flat fading, single-antenna and MIMO.
//!
//! [`Mimo`] draws the $n_{r} \times n_{t}$ matrix of a flat MIMO channel — Rayleigh or
//! Rician, with or without antenna correlation (see [`crate::spatial`]) — constant over
//! blocks of `coherence` samples:
//!
//! $$
//! H = \sqrt{\frac{K}{K + 1}}\\, H_{\mathrm{LOS}} + \sqrt{\frac{1}{K + 1}}\\, L_{r} G L_{t}^{T},
//! \qquad \mathbb{E}|h_{ij}|^{2} = 1,
//! $$
//!
//! and applies it to a signal of $n_{t}$ streams, $y_{k} = H x_{k}$.
//!
//! [`Rayleigh`] multiplies every stream by a circular complex Gaussian gain
//! $h \sim \mathcal{CN}(0, 1)$, constant over blocks of `coherence` samples and
//! independent from block to block and from stream to stream (`coherence = 1` is fast
//! fading). [`Rayleigh::fade`] also returns the gains, for a coherent receiver with
//! perfect channel knowledge; [`Propagate`] discards them.
//!
//! # Model
//!
//! $$
//! y_{k} = h_{\lfloor k / L_{c} \rfloor}\\, x_{k}, \qquad h \sim \mathcal{CN}(0, 1)
//! $$
//!
//! The noise is added separately (`awgn`), so that fading combines with any noise
//! model. With BPSK, coherent detection and average SNR per bit $\bar{\gamma}$,
//!
//! $$
//! P_{b} = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1 + \bar{\gamma}}}\right)
//! $$
//!
//! # Examples
//! ```
//! use com_core::prelude::*;
//! use com_channel::fading::Rayleigh;
//! let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 8], Hertz(1.0))?;
//! let (y, h) = Rayleigh { coherence: 4 }.fade(&x, &mut batch_rng(3, 0))?;
//! assert_eq!(h[0].len(), 8);
//! assert_eq!(h[0][0], h[0][3]);                       // one gain per block of 4
//! assert_eq!(y.stream(0)[5], h[0][5]);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008,
//!    Section 13.3.

use crate::spatial::Kronecker;
use com_core::ensure;
use com_core::prelude::*;
use com_core::{ensure_shape, same_len};
use com_math::linalg::CMat;
use ndarray::Array2;

/// Flat Rayleigh fading with a block coherence of `coherence` samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rayleigh {
    /// Number of consecutive samples sharing one gain (at least 1).
    pub coherence: usize,
}

impl Rayleigh {
    /// Fade `x` and return the faded signal with the gain of every stream and sample.
    ///
    /// # Errors
    /// [`Error::Config`] for a zero coherence.
    pub fn fade(&self, x: &Signal, rng: &mut SimRng) -> Result<(Signal, Vec<Vec<C64>>)> {
        ensure(self.coherence >= 1, || "the coherence must be at least one sample".into())?;
        let mut out = Signal::zeros(x.n_streams(), x.n_samples(), x.sample_rate());
        let mut gains = Vec::with_capacity(x.n_streams());
        for s in 0..x.n_streams() {
            let mut h = Vec::with_capacity(x.n_samples());
            let mut g = C64::new(0.0, 0.0);
            for k in 0..x.n_samples() {
                if k % self.coherence == 0 {
                    g = rng.complex_gaussian();
                }
                h.push(g);
            }
            for ((o, v), g) in out.as_array_mut().row_mut(s).iter_mut().zip(x.stream(s)).zip(&h) {
                *o = v * g;
            }
            gains.push(h);
        }
        Ok((out, gains))
    }
}

impl Propagate for Rayleigh {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        *out = self.fade(x, rng)?.0;
        Ok(())
    }
}

/// A flat MIMO channel of `n_rx` receive and `n_tx` transmit antennas.
#[derive(Clone, Debug, PartialEq)]
pub struct Mimo {
    /// Receive antennas.
    pub n_rx: usize,
    /// Transmit antennas.
    pub n_tx: usize,
    /// Rician factor $K$ (linear, 0 for Rayleigh fading).
    pub k_factor: f64,
    /// Line-of-sight component $H_{\mathrm{LOS}}$ (unit-modulus entries); all ones
    /// (broadside arrays) if `None`.
    pub los: Option<CMat>,
    /// Antenna correlation; independent antennas if `None`.
    pub correlation: Option<Kronecker>,
    /// Samples sharing one channel matrix (1: fast fading).
    pub coherence: usize,
}

impl Mimo {
    /// Independent Rayleigh fading, a new matrix for every sample.
    pub fn rayleigh(n_rx: usize, n_tx: usize) -> Mimo {
        Mimo { n_rx, n_tx, k_factor: 0.0, los: None, correlation: None, coherence: 1 }
    }

    /// Draw one channel matrix.
    ///
    /// # Errors
    /// [`Error::Config`] for zero antennas, a negative Rician factor, or a correlation
    /// or line-of-sight matrix of the wrong size.
    pub fn draw(&self, rng: &mut SimRng) -> Result<CMat> {
        ensure(self.n_rx >= 1 && self.n_tx >= 1, || "at least one antenna on each side".into())?;
        ensure(self.k_factor >= 0.0, || format!("the Rician factor must be non-negative, got {}", self.k_factor))?;
        let shape = (self.n_rx, self.n_tx);
        if let Some(c) = &self.correlation {
            ensure(c.rx().dim() == (self.n_rx, self.n_rx) && c.tx().dim() == (self.n_tx, self.n_tx), || {
                "the correlation matrices do not match the antennas".into()
            })?;
        }
        if let Some(l) = &self.los {
            ensure(l.dim() == shape, || "the line-of-sight matrix does not match the antennas".into())?;
        }
        let g = Array2::from_shape_fn(shape, |_| rng.complex_gaussian());
        let scattered = match &self.correlation {
            Some(c) => c.correlate(&g),
            None => g,
        };
        if self.k_factor == 0.0 {
            return Ok(scattered);
        }
        let k = self.k_factor;
        let (a, b) = ((k / (k + 1.0)).sqrt(), (1.0 / (k + 1.0)).sqrt());
        Ok(match &self.los {
            Some(l) => l.mapv(|x| x * a) + scattered.mapv(|x| x * b),
            None => scattered.mapv(|x| x * b + a),
        })
    }

    /// Apply the channel to `x` (`n_tx` streams) and return the `n_rx` received streams
    /// with the matrix of every coherence block.
    ///
    /// # Errors
    /// As [`Mimo::draw`], [`Error::Shape`] if `x` does not have `n_tx` streams, and
    /// [`Error::Config`] for a zero coherence.
    pub fn fade(&self, x: &Signal, rng: &mut SimRng) -> Result<(Signal, Vec<CMat>)> {
        ensure_shape(x.n_streams() == self.n_tx, || {
            format!("{} streams for {} transmit antennas", x.n_streams(), self.n_tx)
        })?;
        let (y, gains) = self.apply(&x.to_streams(), rng)?;
        Ok((Signal::from_streams(&y, x.sample_rate())?, gains))
    }

    /// Apply the channel to `x` (`n_tx` streams of equal length), one matrix per block
    /// of `coherence` samples: the `n_rx` received streams and the matrix of every
    /// block.
    ///
    /// # Errors
    /// As [`Mimo::draw`], [`Error::Shape`] if `x` does not have `n_tx` streams or has
    /// streams of different lengths, and [`Error::Config`] for a zero coherence.
    pub fn apply(&self, x: &[Vec<C64>], rng: &mut SimRng) -> Result<(Vec<Vec<C64>>, Vec<CMat>)> {
        ensure(self.coherence >= 1, || "the coherence must be at least one sample".into())?;
        ensure_shape(x.len() == self.n_tx, || format!("{} streams for {} transmit antennas", x.len(), self.n_tx))?;
        let n = x.first().map_or(0, Vec::len);
        for xs in x {
            same_len(xs.len(), n, "the transmitted streams")?;
        }
        let coh = self.coherence;
        let hs: Vec<CMat> = (0..n.div_ceil(coh).max(1)).map(|_| self.draw(rng)).collect::<Result<_>>()?;
        let y = (0..self.n_rx)
            .map(|r| {
                (0..n)
                    .map(|i| {
                        let h = &hs[i / coh];
                        (0..self.n_tx).map(|t| h[[r, t]] * x[t][i]).sum()
                    })
                    .collect()
            })
            .collect();
        Ok((y, hs))
    }
}

impl Propagate for Mimo {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        *out = self.fade(x, rng)?.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::exponential_correlation;

    #[test]
    fn gains_have_unit_power_and_the_requested_coherence() {
        let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 200_000], Hertz(1.0)).unwrap();
        let (_, h) = Rayleigh { coherence: 10 }.fade(&x, &mut Rng::new(1)).unwrap();
        let p = h[0].iter().map(C64::norm_sqr).sum::<f64>() / h[0].len() as f64;
        assert!((p - 1.0).abs() < 0.03, "E|h|² = {p}");
        assert!(h[0].chunks(10).all(|c| c.iter().all(|g| *g == c[0])));
        assert!(Rayleigh { coherence: 0 }.fade(&x, &mut Rng::new(1)).is_err());
    }

    #[test]
    fn mimo_entries_have_unit_power_and_the_kronecker_correlation() {
        let rho = C64::new(0.6, 0.3);
        let k = Kronecker::new(exponential_correlation(3, rho), exponential_correlation(2, rho.conj())).unwrap();
        let ch = Mimo { correlation: Some(k), ..Mimo::rayleigh(3, 2) };
        let mut rng = Rng::new(7);
        let n = 100_000;
        let (mut p, mut rr, mut rt) = (0.0, C64::new(0.0, 0.0), C64::new(0.0, 0.0));
        for _ in 0..n {
            let h = ch.draw(&mut rng).unwrap();
            p += h.iter().map(C64::norm_sqr).sum::<f64>() / 6.0;
            rr += h[[0, 0]] * h[[1, 0]].conj();
            rt += h[[2, 0]] * h[[2, 1]].conj();
        }
        let nf = f64::from(n);
        assert!((p / nf - 1.0).abs() < 0.01);
        assert!((rr / nf - rho).norm() < 0.02, "receive correlation {}", rr / nf);
        assert!((rt / nf - rho.conj()).norm() < 0.02, "transmit correlation {}", rt / nf);
    }

    #[test]
    fn rician_mean_and_block_coherence() {
        let ch = Mimo { k_factor: 3.0, coherence: 4, ..Mimo::rayleigh(2, 2) };
        let x = Signal::zeros(2, 10, Hertz(1.0));
        let (y, h) = ch.fade(&x, &mut Rng::new(1)).unwrap();
        assert_eq!((y.n_streams(), h.len()), (2, 3));
        let mut rng = Rng::new(2);
        let mean: C64 = (0..20_000).map(|_| ch.draw(&mut rng).unwrap()[[1, 0]]).sum::<C64>() / 20_000.0;
        assert!((mean - C64::new((0.75f64).sqrt(), 0.0)).norm() < 0.02);
        assert!(ch.fade(&Signal::zeros(3, 4, Hertz(1.0)), &mut Rng::new(0)).is_err());
    }

    #[test]
    fn mimo_rejects_streams_of_unequal_lengths() {
        let ch = Mimo::rayleigh(2, 2);
        let z = C64::new(1.0, 0.0);
        assert!(ch.apply(&[vec![z; 8], vec![z; 8]], &mut Rng::new(0)).is_ok());
        for x in [vec![vec![z; 8], vec![z; 5]], vec![vec![z; 5], vec![z; 8]]] {
            assert!(matches!(ch.apply(&x, &mut Rng::new(0)), Err(Error::Shape(_))));
        }
    }
}

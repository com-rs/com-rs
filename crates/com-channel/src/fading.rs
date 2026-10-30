//! Fading channels: the [`Fading`] trait, and flat fading, single-antenna and MIMO.
//!
//! [`Fading`] is the shape every random linear channel of the crate shares: it draws a
//! realisation as [`Taps`] for a number of samples and applies it,
//!
//! $$
//! y_{r}\[n\] = \sum_{l} \sum_{t} h_{l, r, t}\[n\]\, x_{t}\[n - d_{l}\],
//! $$
//!
//! with the shared checks (streams against the transmit antennas, the rate of the signal
//! against the rate the delays and Doppler shifts were put on). It is implemented by
//! [`Mimo`] (one tap at delay 0), [`crate::multipath::Tdl`],
//! [`crate::multipath::DelayDopplerModel`] and [`crate::cdl::Cdl`]; the [`Propagate`]
//! impls of the last three are [`Fading::fade_into`]. [`Rayleigh`] fades every stream by
//! its own gain (any number of streams, no antenna dimension) and stays apart.
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

use crate::multipath::{Taps, same_rate};
use crate::spatial::Kronecker;
use com_core::ensure;
use com_core::prelude::*;
use com_core::{ensure_shape, same_len};
use com_math::linalg::CMat;
use ndarray::Array2;

/// A random linear channel between `n_tx` transmit and `n_rx` receive antennas whose
/// realisations are [`Taps`] (see the module).
///
/// # Examples
/// ```
/// use com_core::prelude::*;
/// use com_channel::fading::{Fading, Mimo};
/// use com_channel::multipath::{Profile, Tdl, TdlParams};
/// /// The received power of one realisation, whatever the channel.
/// fn received_power(ch: &impl Fading, rng: &mut Rng) -> com_core::Result<f64> {
///     let fs = Hertz(ch.sample_rate().unwrap_or(1.0));
///     let x = Signal::from_streams(&vec![vec![C64::new(1.0, 0.0); 64]; ch.n_tx()], fs)?;
///     let (y, _) = ch.fade(&x, rng)?;
///     Ok(y.as_array().iter().map(C64::norm_sqr).sum::<f64>() / 64.0)
/// }
/// let mut rng = batch_rng(1, 0);
/// assert!(received_power(&Tdl::new(TdlParams { taps: Profile::Eva.taps(), doppler_hz: 10.0, sample_rate: 15.36e6, n_rx: 2, ..TdlParams::default() })?, &mut rng)? > 0.0);
/// assert!(received_power(&Mimo::rayleigh(2, 2), &mut rng)? > 0.0);
/// # Ok::<(), com_core::Error>(())
/// ```
pub trait Fading {
    /// Receive antennas.
    fn n_rx(&self) -> usize;

    /// Transmit antennas.
    fn n_tx(&self) -> usize;

    /// The sampling rate (Hz) on which the channel put its delays and Doppler shifts, the
    /// only rate of the signals it applies to; `None` for a flat channel, which applies at
    /// any rate.
    fn sample_rate(&self) -> Option<f64>;

    /// Draw a realisation over `n_samples` samples from time 0.
    ///
    /// # Errors
    /// [`Error::Config`] for a configuration the channel rejects when it draws.
    fn draw_taps(&self, n_samples: usize, rng: &mut SimRng) -> Result<Taps>;

    /// Apply a new realisation to `x` (`n_tx` streams), the samples before the signal
    /// being zero, writing the `n_rx` received streams into `out`: the taps of the
    /// realisation.
    ///
    /// # Errors
    /// [`Error::Shape`] if `x` does not have `n_tx` streams, [`Error::Config`] if it is
    /// not sampled at [`Fading::sample_rate`], and as [`Fading::draw_taps`].
    fn fade_into(&self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<Taps> {
        same_len(x.n_streams(), self.n_tx(), "streams and transmit antennas")?;
        if let Some(fs) = self.sample_rate() {
            same_rate(x.sample_rate(), fs, "the channel")?;
        }
        let taps = self.draw_taps(x.n_samples(), rng)?;
        out.set_streams(&taps.apply(&x.to_streams())?, x.sample_rate())?;
        Ok(taps)
    }

    /// [`Fading::fade_into`] into a new signal: the received streams and the taps.
    ///
    /// # Errors
    /// As [`Fading::fade_into`].
    fn fade(&self, x: &Signal, rng: &mut SimRng) -> Result<(Signal, Taps)> {
        let mut out = Signal::zeros(self.n_rx(), 0, x.sample_rate());
        let taps = self.fade_into(x, rng, &mut out)?;
        Ok((out, taps))
    }
}

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
        let mut out = Signal::zeros(x.n_streams(), x.n_samples(), x.sample_rate());
        let gains = self.fade_into(x, rng, &mut out)?;
        Ok((out, gains))
    }

    /// [`Rayleigh::fade`] into `out`, reshaped to `x` (its storage reused): the gains.
    fn fade_into(self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<Vec<Vec<C64>>> {
        ensure(self.coherence >= 1, || "the coherence must be at least one sample".into())?;
        out.reshape(x.n_streams(), x.n_samples(), x.sample_rate());
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
        Ok(gains)
    }
}

impl Propagate for Rayleigh {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        self.fade_into(x, rng, out).map(drop)
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
        self.validate()?;
        Ok(self.draw_unchecked(rng))
    }

    /// The checks of [`Mimo::draw`], done once per call of the methods that draw many
    /// matrices.
    fn validate(&self) -> Result<()> {
        ensure(self.n_rx >= 1 && self.n_tx >= 1, || "at least one antenna on each side".into())?;
        ensure(self.k_factor >= 0.0, || format!("the Rician factor must be non-negative, got {}", self.k_factor))?;
        if let Some(c) = &self.correlation {
            ensure(c.rx().dim() == (self.n_rx, self.n_rx) && c.tx().dim() == (self.n_tx, self.n_tx), || {
                "the correlation matrices do not match the antennas".into()
            })?;
        }
        if let Some(l) = &self.los {
            ensure(l.dim() == (self.n_rx, self.n_tx), || {
                "the line-of-sight matrix does not match the antennas".into()
            })?;
        }
        Ok(())
    }

    /// One channel matrix of a validated configuration ([`Mimo::validate`]).
    fn draw_unchecked(&self, rng: &mut SimRng) -> CMat {
        let g = Array2::from_shape_fn((self.n_rx, self.n_tx), |_| rng.complex_gaussian());
        let scattered = match &self.correlation {
            Some(c) => c.correlate(&g),
            None => g,
        };
        if self.k_factor == 0.0 {
            return scattered;
        }
        let k = self.k_factor;
        let (a, b) = ((k / (k + 1.0)).sqrt(), (1.0 / (k + 1.0)).sqrt());
        match &self.los {
            Some(l) => l.mapv(|x| x * a) + scattered.mapv(|x| x * b),
            None => scattered.mapv(|x| x * b + a),
        }
    }

    /// The matrices of the coherence blocks of `n` samples (at least one), the
    /// configuration checked once.
    fn draw_blocks(&self, n: usize, rng: &mut SimRng) -> Result<Vec<CMat>> {
        ensure(self.coherence >= 1, || "the coherence must be at least one sample".into())?;
        self.validate()?;
        Ok((0..n.div_ceil(self.coherence).max(1)).map(|_| self.draw_unchecked(rng)).collect())
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
        let hs = self.draw_blocks(n, rng)?;
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

/// One tap at delay 0, of the matrix of every coherence block at each of its samples
/// (stored once when the realisation is one block). The inherent [`Mimo::fade`] gives
/// the matrices of the blocks instead of the taps.
impl Fading for Mimo {
    fn n_rx(&self) -> usize {
        self.n_rx
    }

    fn n_tx(&self) -> usize {
        self.n_tx
    }

    fn sample_rate(&self) -> Option<f64> {
        None
    }

    fn draw_taps(&self, n_samples: usize, rng: &mut SimRng) -> Result<Taps> {
        let hs = self.draw_blocks(n_samples, rng)?;
        if hs.len() == 1 {
            return Taps::time_invariant(
                vec![0],
                self.n_rx,
                self.n_tx,
                n_samples,
                vec![hs[0].iter().copied().collect()],
            );
        }
        let coeffs = (0..n_samples).flat_map(|i| hs[i / self.coherence].iter().copied()).collect();
        Taps::new(vec![0], self.n_rx, self.n_tx, n_samples, vec![coeffs])
    }
}

/// The received streams of [`Mimo::apply`], which keeps one matrix per coherence block.
impl Propagate for Mimo {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        ensure_shape(x.n_streams() == self.n_tx, || {
            format!("{} streams for {} transmit antennas", x.n_streams(), self.n_tx)
        })?;
        out.set_streams(&self.apply(&x.to_streams(), rng)?.0, x.sample_rate())
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

    /// The matrices of the coherence blocks drawn as `apply` drew them before it checked
    /// the configuration once (one `draw`, with its checks, per block).
    fn blocks_drawn_one_by_one(ch: &Mimo, n: usize, rng: &mut Rng) -> Vec<CMat> {
        (0..n.div_ceil(ch.coherence).max(1)).map(|_| ch.draw(rng).unwrap()).collect()
    }

    #[test]
    fn mimo_checks_once_and_draws_the_matrices_of_one_draw_per_block() {
        let rho = C64::new(0.5, 0.2);
        let k = Kronecker::new(exponential_correlation(2, rho), exponential_correlation(3, rho)).unwrap();
        let x: Vec<Vec<C64>> = (0..3).map(|t| (0..10).map(|i| C64::new(i as f64, t as f64)).collect()).collect();
        for ch in
            [Mimo::rayleigh(2, 3), Mimo { k_factor: 2.0, coherence: 4, correlation: Some(k), ..Mimo::rayleigh(2, 3) }]
        {
            let (_, hs) = ch.apply(&x, &mut Rng::new(9)).unwrap();
            assert_eq!(hs, blocks_drawn_one_by_one(&ch, 10, &mut Rng::new(9)));
        }
        let bad = Mimo { k_factor: -1.0, ..Mimo::rayleigh(2, 3) };
        assert!(bad.apply(&x, &mut Rng::new(0)).is_err());
    }

    #[test]
    fn the_taps_of_a_mimo_channel_give_its_received_streams_bit_for_bit() {
        let x: Vec<Vec<C64>> = (0..2).map(|t| (0..9).map(|i| C64::new(i as f64, -(t as f64))).collect()).collect();
        let s = Signal::from_streams(&x, Hertz(3.0)).unwrap();
        for coherence in [1, 4, 9] {
            let ch = Mimo { k_factor: 1.5, coherence, ..Mimo::rayleigh(3, 2) };
            let (y, hs) = ch.apply(&x, &mut Rng::new(5)).unwrap();
            let (z, taps) = Fading::fade(&ch, &s, &mut Rng::new(5)).unwrap();
            assert_eq!(z.to_streams(), y);
            assert_eq!(taps.is_time_invariant(), coherence == 9);
            assert_eq!(taps.coefficient(0, 8, 2, 1), hs[8 / coherence][[2, 1]]);
            let mut out = Signal::zeros(1, 1, Hertz(1.0));
            let mut prop = ch.clone();
            prop.propagate_into(&s, &mut Rng::new(5), &mut out).unwrap();
            assert_eq!(out, z);
        }
    }
}

//! OFDM and DFT-spread OFDM.
//!
//! A block carries $K$ symbols on $K$ contiguous subcarriers centred on the carrier,
//! out of a grid of $N$ subcarriers; the time signal is computed by an IFFT of size
//! $L N$, an oversampling by $L$ that the peak measurements need (the peaks of the
//! continuous signal fall between the Nyquist-rate samples; $L = 4$ is the usual
//! choice), and a cyclic prefix of $N_{\mathrm{cp}}$ Nyquist-rate samples ($L N_{\mathrm{cp}}$
//! samples) is prepended.
//!
//! # Model
//!
//! With $X_{k}$ the symbol of subcarrier $k \in \\{-K/2, \dots, K/2 - 1\\}$ (unit average
//! energy),
//!
//! $$
//! x\[n\] = \frac{1}{\sqrt{K}} \sum_{k} X_{k}\\, e^{\\,j 2 \pi k n / (L N)}, \qquad
//! n = 0, \dots, L N - 1,
//! $$
//!
//! so that $\mathbb{E}|x\[n\]|^{2} = 1$: the samples have unit average power whatever $K$,
//! $N$ and $L$. With `dft_spread`, the $K$ data symbols are first transformed by a
//! unitary DFT of size $K$ (localised SC-FDMA, the uplink of LTE): the signal is then a
//! single-carrier signal interpolated by the subcarrier mapping, whose PAPR is that of
//! the constellation, not that of a Gaussian signal.
//!
//! The submodules build the rest of an OFDM link: [`grid`] (data and pilot resource
//! elements), [`estimation`] (LS and LMMSE channel estimation on the grid), [`sync`]
//! (Schmidl & Cox timing and frequency synchronisation) and [`detection`] (MIMO-OFDM,
//! per-subcarrier detection).
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::Ofdm;
//! use com_math::C64;
//! let ofdm = Ofdm { n_cp: 16, ..Ofdm::new(256, 200)? };   // L = 4 by default
//! let d: Vec<C64> = (0..400).map(|i| C64::new(if i % 3 == 0 { 1.0 } else { -1.0 }, 0.0)).collect();
//! let x = ofdm.modulate(&d)?;
//! assert_eq!(x.len(), 2 * 4 * (256 + 16));
//! let back = ofdm.demodulate(&x)?;
//! assert!(back.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-12));
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. R. van Nee and R. Prasad, *OFDM for Wireless Multimedia Communications*, Artech
//!    House, 2000.
//! 2. H. G. Myung, J. Lim and D. J. Goodman, "Single carrier FDMA for uplink wireless
//!    transmission," *IEEE Veh. Technol. Mag.*, 1(3), 30–38, 2006.

pub mod detection;
pub mod estimation;
pub mod grid;
pub mod link;
pub mod sync;

use com_core::signal::{Signal, Symbols};
use com_core::traits::{Demodulate, Modulate};
use com_core::units::Hertz;
use com_core::{Result, ensure, ensure_shape};
use com_math::C64;
use com_math::fft;

/// An OFDM (or DFT-spread OFDM) modulator and demodulator.
#[derive(Clone, Debug, PartialEq)]
pub struct Ofdm {
    /// Size $N$ of the subcarrier grid.
    pub n_fft: usize,
    /// Number $K$ of active (modulated) subcarriers, centred on the carrier.
    pub n_active: usize,
    /// Oversampling factor $L$ (IFFT of size $L N$).
    pub oversampling: usize,
    /// Cyclic prefix, in Nyquist-rate samples ($L N_{\mathrm{cp}}$ samples).
    pub n_cp: usize,
    /// Leave the DC subcarrier empty (the active subcarriers skip $k = 0$).
    pub dc_null: bool,
    /// DFT spreading of the data (localised SC-FDMA).
    pub dft_spread: bool,
}

impl Ofdm {
    /// $K$ active subcarriers out of $N$, oversampling 4, no prefix.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless $1 \le K \le N$.
    pub fn new(n_fft: usize, n_active: usize) -> Result<Ofdm> {
        let o = Ofdm { n_fft, n_active, oversampling: 4, n_cp: 0, dc_null: false, dft_spread: false };
        o.check()?;
        Ok(o)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for an empty or overfull grid, or a zero oversampling.
    pub fn check(&self) -> Result<()> {
        ensure(self.n_active >= 1 && self.n_active + usize::from(self.dc_null) <= self.n_fft, || {
            format!("{} active subcarriers do not fit a grid of {}", self.n_active, self.n_fft)
        })?;
        ensure(self.oversampling >= 1, || "the oversampling factor must be at least 1".into())
    }

    /// Size of the IFFT, $L N$.
    pub fn fft_len(&self) -> usize {
        self.oversampling * self.n_fft
    }

    /// Samples per block, prefix included, $L (N + N_{\mathrm{cp}})$.
    pub fn block_len(&self) -> usize {
        self.oversampling * (self.n_fft + self.n_cp)
    }

    /// Subcarrier index $k$ of the active subcarrier `i`.
    pub fn subcarrier(&self, i: usize) -> isize {
        let k = i as isize - (self.n_active / 2) as isize;
        if self.dc_null && k >= 0 { k + 1 } else { k }
    }

    /// FFT bin of the active subcarrier `i`.
    pub fn bin(&self, i: usize) -> usize {
        self.subcarrier(i).rem_euclid(self.fft_len() as isize) as usize
    }

    /// Whether an FFT bin carries an active subcarrier.
    pub fn is_active_bin(&self, bin: usize) -> bool {
        (0..self.n_active).any(|i| self.bin(i) == bin)
    }

    fn scale(&self) -> f64 {
        self.fft_len() as f64 / (self.n_active as f64).sqrt()
    }

    /// The values of the active subcarriers of `data` (the identity, or the unitary DFT
    /// of size $K$ with `dft_spread`).
    pub fn spread(&self, data: &[C64]) -> Vec<C64> {
        if self.dft_spread {
            let s = 1.0 / (data.len() as f64).sqrt();
            fft::fft(data).into_iter().map(|v| v * s).collect()
        } else {
            data.to_vec()
        }
    }

    /// The inverse of [`Ofdm::spread`].
    pub fn despread(&self, active: &[C64]) -> Vec<C64> {
        if self.dft_spread {
            let s = (active.len() as f64).sqrt();
            fft::ifft(active).into_iter().map(|v| v * s).collect()
        } else {
            active.to_vec()
        }
    }

    /// The time block (without prefix, $L N$ samples) of the values of the active
    /// subcarriers.
    ///
    /// # Panics
    /// If `active` does not have $K$ values.
    pub fn to_time(&self, active: &[C64]) -> Vec<C64> {
        assert_eq!(active.len(), self.n_active, "one value per active subcarrier");
        let mut bins = vec![C64::new(0.0, 0.0); self.fft_len()];
        for (i, v) in active.iter().enumerate() {
            bins[self.bin(i)] = *v;
        }
        let s = self.scale();
        fft::ifft(&bins).into_iter().map(|v| v * s).collect()
    }

    /// The values of the active subcarriers of a time block (without prefix).
    ///
    /// # Panics
    /// If `block` does not have $L N$ samples.
    pub fn to_active(&self, block: &[C64]) -> Vec<C64> {
        assert_eq!(block.len(), self.fft_len(), "a block of L N samples");
        let bins = fft::fft(block);
        let s = 1.0 / self.scale();
        (0..self.n_active).map(|i| bins[self.bin(i)] * s).collect()
    }

    /// Keep only the active subcarriers of a time block (an ideal band-pass filter).
    pub fn filter(&self, block: &[C64]) -> Vec<C64> {
        self.to_time(&self.to_active(block))
    }

    /// Prepend the cyclic prefix to a time block.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if the prefix is longer than the block.
    pub fn with_prefix(&self, block: &[C64]) -> Result<Vec<C64>> {
        let cp = self.oversampling * self.n_cp;
        ensure_shape(cp <= block.len(), || format!("a cyclic prefix of {cp} samples for a block of {}", block.len()))?;
        let mut out = Vec::with_capacity(block.len() + cp);
        out.extend_from_slice(&block[block.len() - cp..]);
        out.extend_from_slice(block);
        Ok(out)
    }

    /// Modulate data symbols (a multiple of $K$): $L (N + N_{\mathrm{cp}})$ samples per block.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if the number of symbols is not a multiple of $K$, and the
    /// errors of [`Ofdm::check`].
    pub fn modulate(&self, data: &[C64]) -> Result<Vec<C64>> {
        Ok(self.modulate_with(data, &NoReduction)?.0)
    }

    /// Demodulate whole blocks into data symbols.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if the number of samples is not a multiple of the block length.
    pub fn demodulate(&self, samples: &[C64]) -> Result<Vec<C64>> {
        let blocks = samples.len() / self.block_len().max(1);
        self.demodulate_with(samples, &NoReduction, &vec![0; blocks])
    }

    /// The PAPR (dB) of every whole block of `samples`, the cyclic prefix excluded
    /// ([`com_math::papr_db`] of the useful part).
    pub fn papr_db_per_block(&self, samples: &[C64]) -> Vec<f64> {
        let cp = self.oversampling * self.n_cp;
        samples.chunks_exact(self.block_len()).map(|b| com_math::papr_db(&b[cp..])).collect()
    }

    /// Modulate with a PAPR reduction technique: the samples and the side information
    /// of every block (the selected candidate of SLM or PTS, 0 otherwise).
    ///
    /// # Errors
    /// As [`Ofdm::modulate`], with the number of data symbols per block of the technique,
    /// and the errors of the technique.
    pub fn modulate_with(&self, data: &[C64], reduction: &dyn PaprReduction) -> Result<(Vec<C64>, Vec<usize>)> {
        self.check()?;
        let k = reduction.data_len(self);
        ensure(k >= 1, || "no data subcarrier".into())?;
        ensure_shape(data.len() % k == 0, || format!("{} symbols is not a multiple of {k} per block", data.len()))?;
        let mut out = Vec::with_capacity(data.len() / k * self.block_len());
        let mut side = Vec::with_capacity(data.len() / k);
        for chunk in data.chunks(k) {
            let (block, s) = reduction.reduce(self, chunk)?;
            out.extend(self.with_prefix(&block)?);
            side.push(s);
        }
        Ok((out, side))
    }

    /// Demodulate blocks transmitted with a PAPR reduction technique and its side
    /// information.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for a partial block or a side information of another length.
    pub fn demodulate_with(&self, samples: &[C64], reduction: &dyn PaprReduction, side: &[usize]) -> Result<Vec<C64>> {
        self.check()?;
        let len = self.block_len();
        ensure_shape(samples.len() % len == 0, || {
            format!("{} samples is not a multiple of the block length {len}", samples.len())
        })?;
        ensure_shape(side.len() == samples.len() / len, || {
            format!("{} side informations for {} blocks", side.len(), samples.len() / len)
        })?;
        let cp = self.oversampling * self.n_cp;
        let mut out = Vec::new();
        for (block, &s) in samples.chunks(len).zip(side) {
            let mut b = block[cp..].to_vec();
            reduction.restore_time(&mut b);
            out.extend(reduction.recover(self, self.to_active(&b), s)?);
        }
        Ok(out)
    }
}

/// A PAPR reduction technique applied block by block by [`Ofdm::modulate_with`].
pub trait PaprReduction {
    /// Data symbols per block (all the active subcarriers by default).
    fn data_len(&self, ofdm: &Ofdm) -> usize {
        ofdm.n_active
    }

    /// The time block (without prefix) of one block of data, and its side information.
    ///
    /// # Errors
    /// For a configuration the technique cannot handle.
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)>;

    /// Undo a memoryless transformation of the time block at the receiver (the
    /// expansion of companding); nothing by default.
    fn restore_time(&self, _block: &mut [C64]) {}

    /// The data of a block from the values of its active subcarriers and its side
    /// information (despreading only, by default).
    ///
    /// # Errors
    /// For an invalid side information.
    fn recover(&self, ofdm: &Ofdm, active: Vec<C64>, _side: usize) -> Result<Vec<C64>> {
        Ok(ofdm.despread(&active))
    }
}

/// Plain OFDM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoReduction;

impl PaprReduction for NoReduction {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        Ok((ofdm.to_time(&ofdm.spread(data)), 0))
    }
}

/// OFDM as a [`Modulate`] block: one stream of data symbols into one stream of samples
/// at the rate of the oversampled grid (unit Nyquist-rate sample rate × $L$).
impl Modulate for Ofdm {
    fn modulate_into(&self, x: &Symbols, out: &mut Signal) -> Result<()> {
        ensure(x.n_streams() == 1, || format!("OFDM modulates one stream, not {}", x.n_streams()))?;
        *out = Signal::from_stream(Ofdm::modulate(self, x.stream_slice(0))?, Hertz(self.oversampling as f64))?;
        Ok(())
    }
}

impl Demodulate for Ofdm {
    fn demodulate_into(&mut self, y: &Signal, out: &mut Symbols) -> Result<()> {
        ensure(y.n_streams() == 1, || format!("OFDM demodulates one stream, not {}", y.n_streams()))?;
        *out = Symbols::from_stream(Ofdm::demodulate(self, y.stream(0))?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn round_trip_unit_power_and_band_occupancy() {
        let mut rng = Rng::new(1);
        for (dc_null, spread) in [(false, false), (true, false), (false, true)] {
            let o = Ofdm { n_cp: 8, dc_null, dft_spread: spread, ..Ofdm::new(64, 48).unwrap() };
            let d: Vec<C64> = (0..48 * 200).map(|_| rng.complex_gaussian()).collect();
            let x = o.modulate(&d).unwrap();
            assert!(o.demodulate(&x).unwrap().iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-10));
            let p = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
            assert!((p - 1.0).abs() < 0.03, "power {p}");
            // nothing outside the active bins
            let block = &x[8 * 4..8 * 4 + o.fft_len()];
            let spec = fft::fft(block);
            for (b, v) in spec.iter().enumerate() {
                if !o.is_active_bin(b) {
                    assert!(v.norm() < 1e-9);
                }
            }
            if dc_null {
                assert!(!o.is_active_bin(0));
            }
        }
        assert!(Ofdm::new(64, 65).is_err() && Ofdm::new(64, 0).is_err());
    }

    #[test]
    fn the_traits_wrap_the_methods() {
        let o = Ofdm::new(32, 20).unwrap();
        let d = Symbols::from_stream(vec![C64::new(1.0, 0.0); 40]).unwrap();
        let x = Modulate::modulate(&o, &d).unwrap();
        assert_eq!(x.n_samples(), 2 * 128);
        let mut o2 = o.clone();
        let back = Demodulate::demodulate(&mut o2, &x).unwrap();
        assert!(back.as_slice().iter().all(|v| (v - C64::new(1.0, 0.0)).norm() < 1e-12));
    }
}

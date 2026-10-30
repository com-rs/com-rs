//! Universal filtered multicarrier (UFMC): OFDM filtered per sub-band.
//!
//! # Model
//!
//! $K = BQ$ contiguous subcarriers centred on the carrier, out of a grid of $N$, are
//! grouped in $B$ sub-bands of $Q$ subcarriers. Every sub-band is brought to the time
//! domain by an IDFT of size $N$ (no cyclic prefix) and filtered by a filter of $L$ taps
//! centred on it; the symbol is the sum of the $B$ filtered components, $N + L - 1$
//! samples, and consecutive symbols do not overlap (Schaich & Wild 2014, Sec. 3):
//!
//! $$
//! \mathbf{x} = \sum_{i=1}^{B} \mathbf{F}_{i} \mathbf{V}_{i} \mathbf{s}_{i},
//! $$
//!
//! with $\mathbf{V}_{i}$ the columns of the IDFT of the sub-band and $\mathbf{F}_{i}$ the
//! Toeplitz matrix of the filter $f_{i}\[l\] = w\[l\]\, e^{j 2\pi c_{i} l / N}$, $w$ a
//! Dolph–Chebyshev window ([`com_dsp::window::chebyshev`]) normalised to unit energy and
//! $c_{i}$ the centre of the sub-band. The receiver appends zeros up to $2N$ samples and
//! takes the DFT: the even bins $2k$ carry the subcarriers $k$, $\mathbf{s}_{k}$ times the
//! response of its filter at $k/N$, which a one-tap equaliser removes. The other
//! sub-bands vanish there exactly (the IDFT of a sub-band is zero at the bins of the
//! others): UFMC is orthogonal in the complex sense, unlike FBMC/OQAM, and needs no
//! prefix; the filters lower the sidelobes between sub-bands.
//!
//! The symbols on the subcarriers have unit energy: the IDFT is scaled by
//! $1/\sqrt{K}$ as in [`crate::ofdm`].
//!
//! # Examples
//! ```
//! use com_waveform::ufmc::Ufmc;
//! use com_math::C64;
//! let u = Ufmc::new(64, 4, 12, 16, 40.0)?; // 48 subcarriers, L = 16
//! let d: Vec<C64> = (0..96).map(|i| C64::new(if i % 3 == 0 { 1.0 } else { -1.0 }, 1.0)).collect();
//! let x = u.modulate(&d)?;
//! assert_eq!(x.len(), 2 * (64 + 15));
//! let back = u.demodulate(&x)?;
//! assert!(back.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-9));
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. F. Schaich and T. Wild, "Waveform contenders for 5G — OFDM vs. FBMC vs. UFMC,"
//!    *Proc. 6th Int. Symp. Communications, Control and Signal Processing (ISCCSP)*,
//!    2014.
//! 2. F. J. Harris, "On the use of windows for harmonic analysis with the discrete
//!    Fourier transform," *Proc. IEEE*, 66(1), 51–83, 1978.

use com_core::signal::{Signal, Symbols};
use com_core::traits::{Demodulate, Modulate};
use com_core::units::Hertz;
use com_core::{Result, ensure, ensure_shape};
use com_dsp::filter::convolve_full;
use com_dsp::window::chebyshev;
use com_math::{C64, cis, fft};

/// A UFMC modem (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Ufmc {
    /// Size of the grid and of the IDFT, $N$.
    pub n_fft: usize,
    /// Sub-bands $B$.
    pub n_subbands: usize,
    /// Subcarriers per sub-band, $Q$.
    pub subband_size: usize,
    /// Taps of the filters, $L$ (1: plain OFDM without prefix).
    pub filter_len: usize,
    /// Attenuation of the sidelobes of the Dolph–Chebyshev filters (dB).
    pub sidelobe_db: f64,
}

impl Ufmc {
    /// `n_subbands` sub-bands of `subband_size` subcarriers on a grid of `n_fft`,
    /// filtered by Dolph–Chebyshev filters of `filter_len` taps whose sidelobes are
    /// `sidelobe_db` below their main lobe.
    ///
    /// # Errors
    /// As [`Ufmc::check`].
    pub fn new(
        n_fft: usize,
        n_subbands: usize,
        subband_size: usize,
        filter_len: usize,
        sidelobe_db: f64,
    ) -> Result<Ufmc> {
        let u = Ufmc { n_fft, n_subbands, subband_size, filter_len, sidelobe_db };
        u.check()?;
        Ok(u)
    }

    /// Check the configuration.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the sub-bands do not fit the grid, or if the
    /// filter has no tap or is longer than $N + 1$ taps (a symbol must fit the DFT of
    /// $2N$ of the receiver).
    pub fn check(&self) -> Result<()> {
        let (b, q, n) = (self.n_subbands, self.subband_size, self.n_fft);
        ensure(b >= 1 && q >= 1 && b * q <= n, || format!("{b} sub-bands of {q} do not fit a grid of {n}"))?;
        ensure(self.filter_len >= 1 && self.filter_len <= n + 1, || {
            format!("a filter of {} taps for a grid of {n}: between 1 and N + 1", self.filter_len)
        })?;
        ensure(self.sidelobe_db.is_finite(), || "the sidelobe attenuation must be finite".into())
    }

    /// The number of active subcarriers $K = BQ$.
    pub fn n_active(&self) -> usize {
        self.n_subbands * self.subband_size
    }

    /// Samples per symbol, $N + L - 1$.
    pub fn block_len(&self) -> usize {
        self.n_fft + self.filter_len - 1
    }

    /// The subcarrier index (centred on the carrier) of the active subcarrier `i`.
    pub fn subcarrier(&self, i: usize) -> isize {
        i as isize - (self.n_active() / 2) as isize
    }

    /// The Dolph–Chebyshev window of unit energy.
    fn window(&self) -> Vec<f64> {
        let w = chebyshev(self.filter_len, self.sidelobe_db);
        let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        w.into_iter().map(|v| v / norm).collect()
    }

    /// The taps of the filter of sub-band `b`: the window shifted to the centre of the
    /// sub-band.
    pub fn subband_taps(&self, b: usize) -> Vec<C64> {
        self.shifted(&self.window(), b)
    }

    fn shifted(&self, window: &[f64], b: usize) -> Vec<C64> {
        let first = self.subcarrier(b * self.subband_size) as f64;
        let centre = first + (self.subband_size as f64 - 1.0) / 2.0;
        // centred in time too, so that the response is real at the centre of the band
        let mid = (window.len() as f64 - 1.0) / 2.0;
        window
            .iter()
            .enumerate()
            .map(|(l, w)| cis(std::f64::consts::TAU * centre * (l as f64 - mid) / self.n_fft as f64) * *w)
            .collect()
    }

    /// The response of the filter of the sub-band of the active subcarrier `i` at its
    /// frequency, the gain the receiver divides by.
    pub fn response(&self, i: usize) -> C64 {
        self.response_of(&self.subband_taps(i / self.subband_size), i)
    }

    fn response_of(&self, taps: &[C64], i: usize) -> C64 {
        let k = self.subcarrier(i) as f64;
        taps.iter().enumerate().map(|(l, f)| f * cis(-std::f64::consts::TAU * k * l as f64 / self.n_fft as f64)).sum()
    }

    /// Modulate the symbols `data`, $K$ per UFMC symbol.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] unless `data` holds whole symbols, and the errors of
    /// [`Ufmc::check`].
    pub fn modulate(&self, data: &[C64]) -> Result<Vec<C64>> {
        self.check()?;
        let (n, k) = (self.n_fft, self.n_active());
        ensure_shape(data.len() % k == 0, || format!("{} symbols for {k} subcarriers", data.len()))?;
        let scale = n as f64 / (k as f64).sqrt();
        let window = self.window();
        let filters: Vec<Vec<C64>> = (0..self.n_subbands).map(|b| self.shifted(&window, b)).collect();
        let mut out = Vec::with_capacity(data.len() / k * self.block_len());
        for symbol in data.chunks_exact(k) {
            let mut x = vec![C64::new(0.0, 0.0); self.block_len()];
            for (b, f) in filters.iter().enumerate() {
                let mut grid = vec![C64::new(0.0, 0.0); n];
                for q in 0..self.subband_size {
                    let i = b * self.subband_size + q;
                    grid[self.subcarrier(i).rem_euclid(n as isize) as usize] = symbol[i] * scale;
                }
                for (xt, v) in x.iter_mut().zip(convolve_full(f, &fft::ifft(&grid))) {
                    *xt += v;
                }
            }
            out.extend(x);
        }
        Ok(out)
    }

    /// Demodulate whole symbols: zeros up to $2N$, DFT, the even bins divided by the
    /// response of the filters.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] unless `samples` holds whole symbols, and the errors
    /// of [`Ufmc::check`].
    pub fn demodulate(&self, samples: &[C64]) -> Result<Vec<C64>> {
        self.check()?;
        let (n, len) = (self.n_fft, self.block_len());
        ensure_shape(samples.len() % len == 0, || format!("{} samples for symbols of {len}", samples.len()))?;
        let scale = n as f64 / (self.n_active() as f64).sqrt();
        let window = self.window();
        let filters: Vec<Vec<C64>> = (0..self.n_subbands).map(|b| self.shifted(&window, b)).collect();
        let gains: Vec<C64> =
            (0..self.n_active()).map(|i| self.response_of(&filters[i / self.subband_size], i) * scale).collect();
        let mut out = Vec::with_capacity(samples.len() / len * self.n_active());
        for block in samples.chunks_exact(len) {
            let mut buf = block.to_vec();
            buf.resize(2 * n, C64::new(0.0, 0.0));
            let spec = fft::fft(&buf);
            for (i, g) in gains.iter().enumerate() {
                let bin = (2 * self.subcarrier(i)).rem_euclid(2 * n as isize) as usize;
                out.push(spec[bin] / g);
            }
        }
        Ok(out)
    }
}

impl Modulate for Ufmc {
    fn modulate_into(&self, x: &Symbols, out: &mut Signal) -> Result<()> {
        ensure(x.n_streams() == 1, || format!("UFMC modulates one stream, not {}", x.n_streams()))?;
        out.set_stream(&Ufmc::modulate(self, x.stream_slice(0))?, Hertz(1.0));
        Ok(())
    }
}

impl Demodulate for Ufmc {
    fn demodulate_into(&mut self, y: &Signal, out: &mut Symbols) -> Result<()> {
        ensure(y.n_streams() == 1, || format!("UFMC demodulates one stream, not {}", y.n_streams()))?;
        out.set_stream(&Ufmc::demodulate(self, y.stream(0))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_sub_bands_are_orthogonal_and_the_filters_unit_energy() {
        let u = Ufmc::new(128, 6, 12, 32, 50.0).unwrap();
        let e: f64 = u.subband_taps(0).iter().map(C64::norm_sqr).sum();
        assert!((e - 1.0).abs() < 1e-12);
        // one sub-band alone leaves nothing on the others
        let mut d = vec![C64::new(0.0, 0.0); u.n_active()];
        for v in &mut d[24..36] {
            *v = C64::new(1.0, -1.0);
        }
        let y = u.demodulate(&u.modulate(&d).unwrap()).unwrap();
        assert!(y.iter().zip(&d).all(|(a, b)| (a - b).norm() < 1e-10));
    }

    #[test]
    fn the_sidelobes_fall_faster_than_those_of_ofdm() {
        // the spectrum of random symbols on the first sub-band only, 8 subcarriers past
        // its edge relative to its centre, against the same sub-band without filter
        // (L = 1: plain OFDM)
        let mut r = Rng::new(2);
        let mut level = |u: &Ufmc| {
            let n = u.n_fft;
            let mut p = vec![0.0; 8 * n];
            for _ in 0..200 {
                let mut d = vec![C64::new(0.0, 0.0); u.n_active()];
                for v in &mut d[..12] {
                    *v = C64::new(if r.below(2) == 0 { 1.0 } else { -1.0 }, if r.below(2) == 0 { 1.0 } else { -1.0 });
                }
                let mut x = u.modulate(&d).unwrap();
                x.resize(8 * n, C64::new(0.0, 0.0));
                for (a, b) in p.iter_mut().zip(fft::fft(&x)) {
                    *a += b.norm_sqr();
                }
            }
            // half-way between subcarriers, where the sinc of OFDM has its sidelobes
            let at = |k: isize| p[(8 * k + 4).rem_euclid(8 * n as isize) as usize];
            let first = u.subcarrier(0);
            com_math::lin2db(at(first - 8) / at(first + 6))
        };
        let ufmc = level(&Ufmc::new(128, 4, 12, 33, 60.0).unwrap());
        let ofdm = level(&Ufmc::new(128, 4, 12, 1, 60.0).unwrap());
        assert!(ufmc < ofdm - 20.0, "UFMC {ufmc} dB, OFDM {ofdm} dB");
    }

    #[test]
    fn filters_longer_than_the_receiver_dft_are_rejected() {
        assert!(Ufmc::new(64, 4, 12, 65, 40.0).is_ok());
        assert!(Ufmc::new(64, 4, 12, 66, 40.0).is_err());
        assert!(Ufmc::new(64, 4, 12, 0, 40.0).is_err());
        assert!(Ufmc::new(64, 6, 12, 16, 40.0).is_err());
    }
}

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
//! let u = Ufmc::new(64, 4, 12, 16, 40.0)?;            // 48 subcarriers, L = 16
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

use com_core::{Result, ensure, ensure_shape};
use com_dsp::window::chebyshev;
use com_math::{C64, cis, fft};

/// A UFMC modem (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Ufmc {
    fft_len: usize,
    subbands: usize,
    subband_size: usize,
    window: Vec<f64>,
}

impl Ufmc {
    /// `subbands` sub-bands of `subband_size` subcarriers on a grid of `fft_len`,
    /// filtered by a Dolph–Chebyshev filter of `filter_len` taps whose sidelobes are
    /// `sidelobe_db` below its main lobe.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the sub-bands do not fit the grid or for a filter
    /// of no tap.
    pub fn new(
        fft_len: usize,
        subbands: usize,
        subband_size: usize,
        filter_len: usize,
        sidelobe_db: f64,
    ) -> Result<Ufmc> {
        ensure(subbands >= 1 && subband_size >= 1 && subbands * subband_size <= fft_len, || {
            format!("{subbands} sub-bands of {subband_size} do not fit a grid of {fft_len}")
        })?;
        ensure(filter_len >= 1, || "the filter needs at least one tap".into())?;
        let w = chebyshev(filter_len, sidelobe_db);
        let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok(Ufmc { fft_len, subbands, subband_size, window: w.into_iter().map(|v| v / norm).collect() })
    }

    /// The number of active subcarriers $K = BQ$.
    pub fn n_active(&self) -> usize {
        self.subbands * self.subband_size
    }

    /// Samples per symbol, $N + L - 1$.
    pub fn symbol_len(&self) -> usize {
        self.fft_len + self.window.len() - 1
    }

    /// The subcarrier index (centred on the carrier) of the active subcarrier `i`.
    pub fn subcarrier(&self, i: usize) -> isize {
        i as isize - (self.n_active() / 2) as isize
    }

    /// The filter of sub-band `b`: the window shifted to the centre of the sub-band.
    pub fn filter(&self, b: usize) -> Vec<C64> {
        let first = self.subcarrier(b * self.subband_size) as f64;
        let centre = first + (self.subband_size as f64 - 1.0) / 2.0;
        // centred in time too, so that the response is real at the centre of the band
        let mid = (self.window.len() as f64 - 1.0) / 2.0;
        self.window
            .iter()
            .enumerate()
            .map(|(l, w)| cis(std::f64::consts::TAU * centre * (l as f64 - mid) / self.fft_len as f64) * *w)
            .collect()
    }

    /// The response of the filter of the sub-band of the active subcarrier `i` at its
    /// frequency, the gain the receiver divides by.
    pub fn response(&self, i: usize) -> C64 {
        let k = self.subcarrier(i) as f64;
        self.filter(i / self.subband_size)
            .iter()
            .enumerate()
            .map(|(l, f)| f * cis(-std::f64::consts::TAU * k * l as f64 / self.fft_len as f64))
            .sum()
    }

    /// Modulate the symbols `data`, $K$ per UFMC symbol.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] unless `data` holds whole symbols.
    pub fn modulate(&self, data: &[C64]) -> Result<Vec<C64>> {
        let (n, k) = (self.fft_len, self.n_active());
        ensure_shape(data.len() % k == 0, || format!("{} symbols for {k} subcarriers", data.len()))?;
        let scale = n as f64 / (k as f64).sqrt();
        let filters: Vec<Vec<C64>> = (0..self.subbands).map(|b| self.filter(b)).collect();
        let mut out = Vec::with_capacity(data.len() / k * self.symbol_len());
        for symbol in data.chunks_exact(k) {
            let mut x = vec![C64::new(0.0, 0.0); self.symbol_len()];
            for (b, f) in filters.iter().enumerate() {
                let mut grid = vec![C64::new(0.0, 0.0); n];
                for q in 0..self.subband_size {
                    let i = b * self.subband_size + q;
                    grid[self.subcarrier(i).rem_euclid(n as isize) as usize] = symbol[i] * scale;
                }
                let v = fft::ifft(&grid);
                for (t, vt) in v.iter().enumerate() {
                    for (l, fl) in f.iter().enumerate() {
                        x[t + l] += vt * fl;
                    }
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
    /// [`com_core::Error::Shape`] unless `samples` holds whole symbols.
    pub fn demodulate(&self, samples: &[C64]) -> Result<Vec<C64>> {
        let (n, len) = (self.fft_len, self.symbol_len());
        ensure_shape(samples.len() % len == 0, || format!("{} samples for symbols of {len}", samples.len()))?;
        ensure(len <= 2 * n, || format!("a filter of {} taps is longer than the FFT of {n}", self.window.len()))?;
        let scale = n as f64 / (self.n_active() as f64).sqrt();
        let gains: Vec<C64> = (0..self.n_active()).map(|i| self.response(i) * scale).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_sub_bands_are_orthogonal_and_the_filters_unit_energy() {
        let u = Ufmc::new(128, 6, 12, 32, 50.0).unwrap();
        let e: f64 = u.filter(0).iter().map(C64::norm_sqr).sum();
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
            let n = u.fft_len;
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
            10.0 * (at(first - 8) / at(first + 6)).log10()
        };
        let ufmc = level(&Ufmc::new(128, 4, 12, 33, 60.0).unwrap());
        let ofdm = level(&Ufmc::new(128, 4, 12, 1, 60.0).unwrap());
        assert!(ufmc < ofdm - 20.0, "UFMC {ufmc} dB, OFDM {ofdm} dB");
    }
}

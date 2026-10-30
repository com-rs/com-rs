//! Orthogonal time frequency space (OTFS) modulation with rectangular pulses and one
//! cyclic prefix per frame.
//!
//! # Model
//!
//! An $M \times N$ grid $X$ of symbols in the delay–Doppler plane ($M$ delay bins, the
//! subcarriers; $N$ Doppler bins, the symbols of the frame) is spread by the inverse
//! symplectic finite Fourier transform (ISFFT) to the time–frequency plane and sent by
//! the Heisenberg transform, an OFDM modulator with rectangular pulses. Both together
//! reduce to an IDFT along the Doppler axis (Raviteja et al. 2019, eqs. (1)–(2)):
//!
//! $$
//! \mathbf{s} = \operatorname{vec}(X F_{N}^{H}) = (F_{N}^{H} \otimes I_{M})\, \mathbf{x},
//! \qquad \mathbf{x} = \operatorname{vec}(X),
//! $$
//!
//! the samples $s\[nM + m\]$ of the $N$ symbols one after the other, preceded by a single
//! cyclic prefix for the whole frame. The receiver drops the prefix and applies
//! $(F_{N} \otimes I_{M})$ (Wigner transform and SFFT, eq. (11)).
//!
//! A channel of $P$ paths of gains $h_{i}$, integer delays $l_{i} < M$ and Doppler
//! shifts $k_{i}$ (in bins of $1/(NT)$), $r\[n\] = \sum_{i} h_{i} e^{j 2\pi k_{i}(n -
//! l_{i})/(MN)} s\[\[n - l_{i}\]_{MN}\]$ after the prefix (eq. (6)), becomes in the
//! delay–Doppler plane $\mathbf{y} = H_{\mathrm{eff}} \mathbf{x} + \mathbf{w}$ with the
//! sparse matrix of Theorem 1, [`Otfs::effective_channel`] ([`EffectiveChannel`]: $P$
//! entries per row, applied in $O(MNP)$ operations, [`EffectiveChannel::to_dense`] for
//! the detectors that take a matrix),
//! the 2D circular convolution $Y\[m, n\] = \sum_{i} h_{i} X\[\[m - l_{i}\]_{M},
//! \[n - k_{i}\]_{N}\]$ of ideal pulses (eq. (30)) up to the phases
//! $\alpha_{i}(m, n)$ of Remark 1. Fractional Doppler shifts spread over all the Doppler
//! bins ([`Otfs::transform_channel`] takes any time-domain channel matrix). The
//! channel itself is drawn in `com-channel` (`multipath::DelayDoppler`), its paths are
//! [`com_core::channel::Path`]; the detectors are those of `com-mimo` on
//! $H_{\mathrm{eff}}$.
//!
//! # Examples
//! ```
//! use com_waveform::otfs::Otfs;
//! use com_math::C64;
//! let o = Otfs { n_cp: 2, ..Otfs::new(8, 4)? };
//! let x: Vec<C64> = (0..32).map(|i| C64::new((i % 3) as f64 - 1.0, (i % 2) as f64)).collect();
//! let s = o.modulate(&x)?;
//! assert_eq!(s.len(), 32 + 2);
//! assert!(o.demodulate(&s)?.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. R. Hadani, S. Rakib, M. Tsatsanis, A. Monk, A. J. Goldsmith, A. F. Molisch and R.
//!    Calderbank, "Orthogonal time frequency space modulation," *Proc. IEEE WCNC*,
//!    2017 (arXiv:1808.00519).
//! 2. P. Raviteja, Y. Hong, E. Viterbo and E. Biglieri, "Practical pulse-shaping
//!    waveforms for reduced-cyclic-prefix OTFS," *IEEE Trans. Veh. Technol.*, 68(1),
//!    957–961, 2019.
//! 3. Z. Wei, W. Yuan, S. Li, J. Yuan, G. Bharatula, R. Hadani and L. Hanzo,
//!    "Orthogonal time-frequency space modulation: a promising next-generation
//!    waveform," *IEEE Wireless Commun.*, 28(4), 136–144, 2021.

use crate::ofdm::check_rate;
use com_core::channel::Path;
use com_core::prelude::*;
use com_core::{ensure, ensure_shape};
use com_dsp::signal::{extend_with_prefix, strip_prefix};
use com_math::linalg::CMat;
use com_math::{C64, cis, fft};

/// An OTFS modem: `n_fft` delay bins $M$ (the subcarriers), `n_symbols` Doppler bins $N$
/// (the symbols of a frame) and one prefix of `n_cp` samples per frame (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Otfs {
    /// Delay bins $M$, the subcarriers of the Heisenberg transform.
    pub n_fft: usize,
    /// Doppler bins $N$, the symbols of a frame.
    pub n_symbols: usize,
    /// The cyclic prefix of a frame, in samples (OTFS has no oversampling: one sample
    /// per delay bin).
    pub n_cp: usize,
    /// The rate of the samples, $M \Delta f$: the rate of the [`Signal`] of [`Modulate`]
    /// and the rate [`Demodulate`] expects; 1 Hz by default (time in samples).
    pub sample_rate: Hertz,
}

impl Otfs {
    /// $M$ delay bins and $N$ Doppler bins, no prefix, at the sample rate 1 Hz.
    ///
    /// # Errors
    /// [`Error::Config`] for an empty grid.
    pub fn new(n_fft: usize, n_symbols: usize) -> Result<Otfs> {
        let o = Otfs { n_fft, n_symbols, n_cp: 0, sample_rate: Hertz(1.0) };
        o.check()?;
        Ok(o)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// [`Error::Config`] for an empty grid, a prefix longer than the frame or a sample
    /// rate that is not positive and finite.
    pub fn check(&self) -> Result<()> {
        ensure(self.n_fft >= 1 && self.n_symbols >= 1, || {
            format!("an OTFS grid needs at least one bin, got {} × {}", self.n_fft, self.n_symbols)
        })?;
        ensure(self.n_cp <= self.n_fft * self.n_symbols, || {
            format!("a prefix of {} samples is longer than the frame of {}", self.n_cp, self.n_fft * self.n_symbols)
        })?;
        check_rate(self.sample_rate)
    }

    /// Samples per frame, $MN$ plus the prefix.
    pub fn block_len(&self) -> usize {
        self.n_fft * self.n_symbols + self.n_cp
    }

    /// The unitary DFT ($\pm$) along the Doppler axis of `v[n * M + m]`.
    fn doppler_dft(&self, v: &[C64], inverse: bool) -> Vec<C64> {
        let (m, n) = (self.n_fft, self.n_symbols);
        let mut out = vec![C64::new(0.0, 0.0); m * n];
        let mut col = vec![C64::new(0.0, 0.0); n];
        for d in 0..m {
            for (k, c) in col.iter_mut().enumerate() {
                *c = v[k * m + d];
            }
            if inverse {
                fft::ifft_unitary_in_place(&mut col);
            } else {
                fft::fft_unitary_in_place(&mut col);
            }
            for (k, c) in col.iter().enumerate() {
                out[k * m + d] = *c;
            }
        }
        out
    }

    /// Modulate the frames of `x`, each $MN$ symbols $x\[nM + m\] = X\[m, n\]$ (the
    /// columns of $X$ one after the other), with the prefix.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid configuration, [`Error::Shape`] unless `x` holds
    /// whole frames.
    pub fn modulate(&self, x: &[C64]) -> Result<Vec<C64>> {
        self.check()?;
        let mn = self.n_fft * self.n_symbols;
        ensure_shape(x.len() % mn == 0, || format!("{} symbols for frames of {mn}", x.len()))?;
        let mut out = Vec::with_capacity(x.len() / mn * self.block_len());
        for frame in x.chunks_exact(mn) {
            extend_with_prefix(&mut out, &self.doppler_dft(frame, true), self.n_cp)?;
        }
        Ok(out)
    }

    /// Demodulate whole frames: drop the prefix, DFT along the Doppler axis.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid configuration, [`Error::Shape`] unless `r` holds
    /// whole frames.
    pub fn demodulate(&self, r: &[C64]) -> Result<Vec<C64>> {
        self.check()?;
        let len = self.block_len();
        ensure_shape(r.len() % len == 0, || format!("{} samples for frames of {len}", r.len()))?;
        let mut out = Vec::with_capacity(r.len() / len * (len - self.n_cp));
        for frame in r.chunks_exact(len) {
            out.extend(self.doppler_dft(strip_prefix(frame, self.n_cp)?, false));
        }
        Ok(out)
    }

    /// The effective delay–Doppler channel of `paths` at [`Otfs::sample_rate`]: the
    /// $MN \times MN$ matrix of Theorem 1 of Raviteja et al. (2019), eq. (12), with
    /// $z = e^{j 2\pi/(MN)}$ and the Doppler shifts in bins $k_{i} = \nu_{i} MN / f_{s}$
    /// (of either sign),
    ///
    /// $$
    /// T^{(i)}(p, q) = \begin{cases} e^{-j 2\pi n/N} z^{k_{i}([m - l_{i}]_{M})} &
    /// m < l_{i}, \\\\ z^{k_{i}([m - l_{i}]_{M})} & m \ge l_{i}, \end{cases}
    /// \qquad q = [m - l_{i}]_{M} + M[n - k_{i}]_{N},
    /// $$
    ///
    /// $p = nM + m$, zero elsewhere, summed with the gains: one entry per path in every
    /// row ([`EffectiveChannel`]). The time of a path counts from the first sample of the
    /// frame, its prefix included, as in `com_channel::multipath::DelayDoppler`: the gain
    /// of eq. (6), whose time starts after the prefix, is
    /// $h_{i} e^{j 2\pi k_{i} N_{\mathrm{cp}}/(MN)}$. The model assumes a prefix at least
    /// as long as the largest delay.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid configuration, a delay of $M$ samples or more, or
    /// a Doppler shift that is not a whole number of bins ([`Otfs::transform_channel`]
    /// takes any channel).
    ///
    /// # Examples
    /// ```
    /// use com_waveform::otfs::Otfs;
    /// use com_core::channel::Path;
    /// use com_core::units::Hertz;
    /// use com_math::C64;
    /// // M N = 32 samples at 32 kHz: one Doppler bin is 1 kHz
    /// let o = Otfs { sample_rate: Hertz(32e3), ..Otfs::new(8, 4)? };
    /// let h = o.effective_channel(&[Path { gain: C64::new(1.0, 0.0), delay: 1, doppler_hz: 1e3 }])?;
    /// assert_eq!((h.size(), h.paths()), (32, 1));
    /// let x: Vec<C64> = (0..32).map(|i| C64::new(i as f64, 0.0)).collect();
    /// let y = h.apply(&x)?;
    /// let dense = h.to_dense();
    /// assert!((0..32).all(|p| ((0..32).map(|q| dense[[p, q]] * x[q]).sum::<C64>() - y[p]).norm() < 1e-12));
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn effective_channel(&self, paths: &[Path]) -> Result<EffectiveChannel> {
        self.check()?;
        let (m, n) = (self.n_fft, self.n_symbols);
        let mn = m * n;
        let n_paths = paths.len();
        let mut cols = vec![0usize; mn * n_paths];
        let mut vals = vec![C64::new(0.0, 0.0); mn * n_paths];
        for (i, path) in paths.iter().enumerate() {
            let l = path.delay;
            ensure(l < m, || format!("a delay of {l} bins needs more than {m} delay bins"))?;
            let bins = path.doppler_hz * mn as f64 / self.sample_rate.0;
            ensure((bins - bins.round()).abs() < 1e-9, || {
                format!("a Doppler shift of {bins} bins is not a whole bin: use transform_channel")
            })?;
            let k = bins.round() as i64;
            let gain = path.gain * cis(std::f64::consts::TAU * k as f64 * self.n_cp as f64 / mn as f64);
            for p in 0..mn {
                let (row_n, row_m) = (p / m, p % m);
                let shifted = (row_m + m - l) % m;
                let q = shifted + m * ((row_n as i64 - k).rem_euclid(n as i64) as usize);
                let z = cis(std::f64::consts::TAU * k as f64 * shifted as f64 / mn as f64);
                let wrap =
                    if row_m < l { cis(-std::f64::consts::TAU * row_n as f64 / n as f64) } else { C64::new(1.0, 0.0) };
                cols[p * n_paths + i] = q;
                vals[p * n_paths + i] = gain * z * wrap;
            }
        }
        Ok(EffectiveChannel { size: mn, n_paths, cols, vals })
    }

    /// The effective channel $(F_{N} \otimes I_{M}) H (F_{N}^{H} \otimes I_{M})$ of a
    /// time-domain channel matrix `h` ($MN \times MN$, acting on the frame without its
    /// prefix), for any delays and Doppler shifts (eq. (13)).
    ///
    /// $F_{N}^{H} \otimes I_{M}$ is symmetric, so the right product transforms every row
    /// of $H$ and the left one every column of the result, each by $M$ FFTs of $N$
    /// points: $O((MN)^{2} \log N)$ operations.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `h` is $MN \times MN$.
    pub fn transform_channel(&self, h: &CMat) -> Result<CMat> {
        let mn = self.n_fft * self.n_symbols;
        ensure_shape(h.nrows() == mn && h.ncols() == mn, || {
            format!("a {}×{} channel for frames of {mn}", h.nrows(), h.ncols())
        })?;
        // (F^H ⊗ I) is symmetric: row p of H (F^H ⊗ I) is (F^H ⊗ I) applied to row p of H
        let mut right = CMat::zeros((mn, mn));
        for (p, row) in h.rows().into_iter().enumerate() {
            let t = self.doppler_dft(&row.to_vec(), true);
            right.row_mut(p).iter_mut().zip(t).for_each(|(o, v)| *o = v);
        }
        // then (F ⊗ I) on every column
        let mut out = CMat::zeros((mn, mn));
        for (q, col) in right.columns().into_iter().enumerate() {
            let t = self.doppler_dft(&col.to_vec(), false);
            out.column_mut(q).iter_mut().zip(t).for_each(|(o, v)| *o = v);
        }
        Ok(out)
    }
}

/// The effective delay–Doppler channel of [`Otfs::effective_channel`]: an $MN \times MN$
/// matrix of $P$ entries per row, one per path (the entries of paths that land on the
/// same column add up).
#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveChannel {
    size: usize,
    n_paths: usize,
    /// The column of every entry, row after row, $P$ per row.
    cols: Vec<usize>,
    /// The value of every entry.
    vals: Vec<C64>,
}

impl EffectiveChannel {
    /// The size $MN$ of the square matrix.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The number $P$ of entries per row (the paths).
    pub fn paths(&self) -> usize {
        self.n_paths
    }

    /// The entries `(column, value)` of row `p`, one per path.
    ///
    /// # Panics
    /// If `p` is not below [`EffectiveChannel::size`].
    pub fn row(&self, p: usize) -> impl Iterator<Item = (usize, C64)> + '_ {
        let r = p * self.n_paths..(p + 1) * self.n_paths;
        self.cols[r.clone()].iter().copied().zip(self.vals[r].iter().copied())
    }

    /// $\mathbf{y} = H_{\mathrm{eff}} \mathbf{x}$, in $O(MNP)$ operations.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `x` has $MN$ values.
    pub fn apply(&self, x: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(x.len() == self.size, || format!("{} values for an effective channel of {}", x.len(), self.size))?;
        Ok((0..self.size).map(|p| self.row(p).map(|(q, v)| v * x[q]).sum()).collect())
    }

    /// The dense $MN \times MN$ matrix ($16 (MN)^{2}$ bytes: 16 MB at $32 \times 32$), for
    /// the detectors that take a matrix.
    pub fn to_dense(&self) -> CMat {
        let mut h = CMat::zeros((self.size, self.size));
        // path after path, as the entries of a path were added to the matrix
        for i in 0..self.n_paths {
            for p in 0..self.size {
                h[[p, self.cols[p * self.n_paths + i]]] += self.vals[p * self.n_paths + i];
            }
        }
        h
    }
}

/// OTFS as a [`Modulate`] block: one stream of delay–Doppler symbols into one stream of
/// samples at [`Otfs::sample_rate`].
impl Modulate for Otfs {
    fn modulate_into(&self, x: &Symbols, out: &mut Signal) -> Result<()> {
        ensure(x.n_streams() == 1, || format!("OTFS modulates one stream, not {}", x.n_streams()))?;
        out.set_stream(&Otfs::modulate(self, x.stream_slice(0))?, self.sample_rate);
        Ok(())
    }
}

/// OTFS as a [`Demodulate`] block: one stream of samples at [`Otfs::sample_rate`].
impl Demodulate for Otfs {
    /// # Errors
    /// [`Error::Config`] for several streams or a signal at another sample rate, and the
    /// errors of [`Otfs::demodulate`].
    fn demodulate_into(&mut self, y: &Signal, out: &mut Symbols) -> Result<()> {
        crate::ofdm::check_input(y, self.sample_rate, "OTFS")?;
        out.set_stream(&Otfs::demodulate(self, y.stream(0))?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_traits_carry_the_rate_of_the_modem() {
        let mut o = Otfs { n_cp: 2, sample_rate: Hertz(480e3), ..Otfs::new(8, 4).unwrap() };
        let x = Symbols::from_stream((0..32).map(|i| C64::new(f64::from(i), 1.0)).collect()).unwrap();
        let s = Modulate::modulate(&o, &x).unwrap();
        assert_eq!(s.sample_rate(), Hertz(480e3));
        assert!(
            Demodulate::demodulate(&mut o, &s)
                .unwrap()
                .as_slice()
                .iter()
                .zip(x.as_slice())
                .all(|(a, b)| (a - b).norm() < 1e-12)
        );
        let wrong = Signal::from_stream(s.stream(0).to_vec(), Hertz(1.0)).unwrap();
        assert!(Demodulate::demodulate(&mut o, &wrong).is_err());
        assert!(Otfs { sample_rate: Hertz(f64::NAN), ..o }.check().is_err());
    }

    #[test]
    fn a_channel_without_delay_or_doppler_is_the_identity_and_rows_are_sparse() {
        // one Doppler bin is 1 Hz at a sampling rate of MN = 24 Hz
        let o = Otfs { sample_rate: Hertz(24.0), ..Otfs::new(6, 4).unwrap() };
        let h = o.effective_channel(&paths_of(&[(C64::new(0.5, 0.0), 0, 0.0)])).unwrap().to_dense();
        for p in 0..24 {
            for q in 0..24 {
                assert_eq!(h[[p, q]], if p == q { C64::new(0.5, 0.0) } else { C64::new(0.0, 0.0) });
            }
        }
        let paths = [(C64::new(1.0, 0.0), 0, 0.0), (C64::new(0.3, 0.2), 2, 1.0), (C64::new(-0.4, 0.1), 5, -2.0)];
        let sparse = o.effective_channel(&paths_of(&paths)).unwrap();
        let h = sparse.to_dense();
        for p in 0..24 {
            assert_eq!((0..24).filter(|&q| h[[p, q]].norm() > 0.0).count(), 3);
            assert_eq!((0..24).filter(|&q| h[[q, p]].norm() > 0.0).count(), 3);
            assert_eq!(sparse.row(p).count(), 3);
        }
        assert!(o.effective_channel(&paths_of(&[(C64::new(1.0, 0.0), 0, 0.5)])).is_err());
        assert!(sparse.apply(&[C64::new(0.0, 0.0); 23]).is_err());
        assert!(Otfs { n_cp: 25, ..o }.check().is_err());
    }

    /// Paths from `(gain, delay, doppler_hz)`.
    fn paths_of(p: &[(C64, usize, f64)]) -> Vec<Path> {
        p.iter().map(|&(gain, delay, doppler_hz)| Path { gain, delay, doppler_hz }).collect()
    }

    /// The effective channel of Theorem 1 as it was written before it was sparse: a dense
    /// matrix, path after path.
    fn dense_theorem_1(o: &Otfs, paths: &[(C64, usize, f64)], sample_rate: f64) -> CMat {
        let (m, n) = (o.n_fft, o.n_symbols);
        let mn = m * n;
        let mut h = CMat::zeros((mn, mn));
        for &(gain, l, doppler_hz) in paths {
            let k = (doppler_hz * mn as f64 / sample_rate).round() as i64;
            let gain = gain * cis(std::f64::consts::TAU * k as f64 * o.n_cp as f64 / mn as f64);
            for p in 0..mn {
                let (row_n, row_m) = (p / m, p % m);
                let shifted = (row_m + m - l) % m;
                let q = shifted + m * ((row_n as i64 - k).rem_euclid(n as i64) as usize);
                let z = cis(std::f64::consts::TAU * k as f64 * shifted as f64 / mn as f64);
                let wrap =
                    if row_m < l { cis(-std::f64::consts::TAU * row_n as f64 / n as f64) } else { C64::new(1.0, 0.0) };
                h[[p, q]] += gain * z * wrap;
            }
        }
        h
    }

    #[test]
    fn the_sparse_channel_is_the_former_dense_matrix() {
        let mut rng = com_core::rng::Rng::new(8);
        for (m, n, cp) in [(6usize, 4usize, 0usize), (16, 8, 4), (5, 7, 3)] {
            let mn = (m * n) as f64;
            let o = Otfs { n_cp: cp, sample_rate: Hertz(mn * 1e3), ..Otfs::new(m, n).unwrap() };
            // whole bins of 1 kHz, two paths on the same delay and Doppler (they add up)
            let paths = [
                (rng.complex_gaussian(), 0usize, 0.0),
                (rng.complex_gaussian(), 2, 1e3),
                (rng.complex_gaussian(), m - 1, -2e3),
                (rng.complex_gaussian(), 2, 1e3),
            ];
            let sparse = o.effective_channel(&paths_of(&paths)).unwrap();
            assert_eq!(sparse.to_dense(), dense_theorem_1(&o, &paths, mn * 1e3), "{m} × {n}");
            let x: Vec<C64> = (0..m * n).map(|_| rng.complex_gaussian()).collect();
            let dense = sparse.to_dense();
            for (p, y) in sparse.apply(&x).unwrap().iter().enumerate() {
                let want: C64 = (0..m * n).map(|q| dense[[p, q]] * x[q]).sum();
                assert!((y - want).norm() < 1e-12);
            }
        }
    }

    /// The effective channel as it was computed before: column $q$ is
    /// $(F_{N} \otimes I_{M}) H (F_{N}^{H} \otimes I_{M}) e_{q}$, a dense product per column.
    fn transform_channel_by_columns(o: &Otfs, h: &CMat) -> CMat {
        let mn = o.n_fft * o.n_symbols;
        let mut out = CMat::zeros((mn, mn));
        for q in 0..mn {
            let mut e = vec![C64::new(0.0, 0.0); mn];
            e[q] = C64::new(1.0, 0.0);
            let s = o.doppler_dft(&e, true);
            let hs: Vec<C64> = (0..mn).map(|p| (0..mn).map(|j| h[[p, j]] * s[j]).sum()).collect();
            for (p, v) in o.doppler_dft(&hs, false).into_iter().enumerate() {
                out[[p, q]] = v;
            }
        }
        out
    }

    #[test]
    fn the_transformed_channel_is_the_former_product_and_theorem_1_for_whole_bins() {
        let mut rng = com_core::rng::Rng::new(5);
        for (m, n) in [(4usize, 3usize), (8, 4), (5, 7)] {
            let o = Otfs::new(m, n).unwrap();
            let mn = m * n;
            let h = CMat::from_shape_fn((mn, mn), |_| rng.complex_gaussian());
            let (fast, slow) = (o.transform_channel(&h).unwrap(), transform_channel_by_columns(&o, &h));
            let err = fast.iter().zip(&slow).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-12, "{m} × {n}: {err}");
        }
        // a cyclic channel of whole delay and Doppler bins (no prefix): Theorem 1
        let o = Otfs::new(6, 4).unwrap();
        let paths = [(C64::new(1.0, 0.0), 0usize, 0.0), (C64::new(0.3, 0.2), 2, 1.0), (C64::new(-0.4, 0.1), 5, -2.0)];
        let mut h = CMat::zeros((24, 24));
        for &(g, l, k) in &paths {
            for p in 0..24 {
                h[[p, (p + 24 - l) % 24]] += g * cis(std::f64::consts::TAU * k * (p as f64 - l as f64) / 24.0);
            }
        }
        let theorem = Otfs { sample_rate: Hertz(24.0), ..o };
        let (a, b) =
            (o.transform_channel(&h).unwrap(), theorem.effective_channel(&paths_of(&paths)).unwrap().to_dense());
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).norm() < 1e-12));
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    type Matrix = [[C64; 4]; 4];

    /// P. Raviteja, Y. Hong, E. Viterbo and E. Biglieri, IEEE Trans. Veh. Technol. 68(1),
    /// 2019, the example after Theorem 1: $M = N = 2$, the matrices $T^{(i)}$ of the
    /// four channels $(k, l) = (0, 0), (0, 1), (1, 0), (1, 1)$, entry by entry.
    #[test]
    fn raviteja_2019_example_of_theorem_1() {
        // MN = 4: one bin is 1 Hz at 4 Hz
        let o = Otfs { sample_rate: Hertz(4.0), ..Otfs::new(2, 2).unwrap() };
        let (z0, one) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
        let (j4, mj4, mj2) =
            (cis(std::f64::consts::FRAC_PI_2), cis(-std::f64::consts::FRAC_PI_2), cis(-std::f64::consts::PI));
        let cases: [((usize, f64), Matrix); 4] = [
            ((0, 0.0), [[one, z0, z0, z0], [z0, one, z0, z0], [z0, z0, one, z0], [z0, z0, z0, one]]),
            ((1, 0.0), [[z0, one, z0, z0], [one, z0, z0, z0], [z0, z0, z0, mj2], [z0, z0, one, z0]]),
            ((0, 1.0), [[z0, z0, one, z0], [z0, z0, z0, j4], [one, z0, z0, z0], [z0, j4, z0, z0]]),
            ((1, 1.0), [[z0, z0, z0, j4], [z0, z0, one, z0], [z0, mj4, z0, z0], [one, z0, z0, z0]]),
        ];
        for ((l, k), want) in cases {
            let t = o.effective_channel(&[Path { gain: one, delay: l, doppler_hz: k }]).unwrap().to_dense();
            for p in 0..4 {
                for q in 0..4 {
                    assert!((t[[p, q]] - want[p][q]).norm() < 1e-12, "l {l} k {k} ({p}, {q}): {}", t[[p, q]]);
                }
            }
        }
    }
}

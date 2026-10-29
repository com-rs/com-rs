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
//! sparse matrix of Theorem 1, [`Otfs::effective_channel`]: $P$ nonzero entries per row,
//! the 2D circular convolution $Y\[m, n\] = \sum_{i} h_{i} X\[\[m - l_{i}\]_{M},
//! \[n - k_{i}\]_{N}\]$ of ideal pulses (eq. (30)) up to the phases
//! $\alpha_{i}(m, n)$ of Remark 1. Fractional Doppler shifts spread over all the Doppler
//! bins ([`Otfs::transform_channel`] takes any time-domain channel matrix). The
//! channel itself is drawn in `com-channel` (`multipath::DelayDoppler`); the
//! detectors are those of `com-mimo` on $H_{\mathrm{eff}}$.
//!
//! # Examples
//! ```
//! use com_waveform::otfs::Otfs;
//! use com_math::C64;
//! let o = Otfs::new(8, 4, 2)?;
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

use com_core::{Result, ensure, ensure_shape};
use com_math::linalg::CMat;
use com_math::{C64, cis, fft};

/// An OTFS modem of $M$ delay bins and $N$ Doppler bins (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Otfs {
    m: usize,
    n: usize,
    cp: usize,
}

impl Otfs {
    /// `m` delay bins (subcarriers), `n` Doppler bins (symbols), a prefix of `cp`
    /// samples for the frame.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for an empty grid.
    pub fn new(m: usize, n: usize, cp: usize) -> Result<Otfs> {
        ensure(m >= 1 && n >= 1, || format!("an OTFS grid needs at least one bin, got {m} × {n}"))?;
        Ok(Otfs { m, n, cp })
    }

    /// The number of delay bins $M$.
    pub fn delay_bins(&self) -> usize {
        self.m
    }

    /// The number of Doppler bins $N$.
    pub fn doppler_bins(&self) -> usize {
        self.n
    }

    /// The prefix (samples).
    pub fn cp(&self) -> usize {
        self.cp
    }

    /// Samples per frame, $MN$ plus the prefix.
    pub fn frame_len(&self) -> usize {
        self.m * self.n + self.cp
    }

    /// The unitary DFT ($\pm$) along the Doppler axis of `v[n * M + m]`.
    fn doppler_dft(&self, v: &[C64], inverse: bool) -> Vec<C64> {
        let (m, n) = (self.m, self.n);
        let plan = fft::plan(n);
        let mut out = vec![C64::new(0.0, 0.0); m * n];
        let mut col = vec![C64::new(0.0, 0.0); n];
        let scale = (n as f64).sqrt();
        for d in 0..m {
            for (k, c) in col.iter_mut().enumerate() {
                *c = v[k * m + d];
            }
            if inverse {
                plan.inverse(&mut col);
                for c in &mut col {
                    *c *= scale;
                }
            } else {
                plan.forward(&mut col);
                for c in &mut col {
                    *c /= scale;
                }
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
    /// [`com_core::Error::Shape`] unless `x` holds whole frames.
    pub fn modulate(&self, x: &[C64]) -> Result<Vec<C64>> {
        let mn = self.m * self.n;
        ensure_shape(x.len() % mn == 0, || format!("{} symbols for frames of {mn}", x.len()))?;
        let mut out = Vec::with_capacity(x.len() / mn * self.frame_len());
        for frame in x.chunks_exact(mn) {
            let s = self.doppler_dft(frame, true);
            out.extend_from_slice(&s[mn - self.cp.min(mn)..]);
            out.extend(s);
        }
        Ok(out)
    }

    /// Demodulate whole frames: drop the prefix, DFT along the Doppler axis.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] unless `r` holds whole frames.
    pub fn demodulate(&self, r: &[C64]) -> Result<Vec<C64>> {
        let len = self.frame_len();
        ensure_shape(r.len() % len == 0, || format!("{} samples for frames of {len}", r.len()))?;
        Ok(r.chunks_exact(len).flat_map(|f| self.doppler_dft(&f[self.cp.min(self.m * self.n)..], false)).collect())
    }

    /// The effective delay–Doppler channel of paths `(gain, delay, doppler)` with
    /// integer delays $l < M$ and Doppler shifts $k$ (bins, of either sign): the
    /// $MN \times MN$ matrix of Theorem 1 of Raviteja et al. (2019), eq. (12), with
    /// $z = e^{j 2\pi/(MN)}$,
    ///
    /// $$
    /// T^{(i)}(p, q) = \begin{cases} e^{-j 2\pi n/N} z^{k_{i}([m - l_{i}]_{M})} &
    /// m < l_{i}, \\\\ z^{k_{i}([m - l_{i}]_{M})} & m \ge l_{i}, \end{cases}
    /// \qquad q = [m - l_{i}]_{M} + M[n - k_{i}]_{N},
    /// $$
    ///
    /// $p = nM + m$, zero elsewhere, summed with the gains $h_{i}$.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a delay of $M$ or more.
    pub fn effective_channel(&self, paths: &[(C64, usize, i64)]) -> Result<CMat> {
        let (m, n) = (self.m, self.n);
        let mn = m * n;
        let mut h = CMat::zeros((mn, mn));
        for &(gain, l, k) in paths {
            ensure(l < m, || format!("a delay of {l} bins needs more than {m} delay bins"))?;
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
        Ok(h)
    }

    /// The effective channel $(F_{N} \otimes I_{M}) H (F_{N}^{H} \otimes I_{M})$ of a
    /// time-domain channel matrix `h` ($MN \times MN$, acting on the frame without its
    /// prefix), for any delays and Doppler shifts (eq. (13)).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] unless `h` is $MN \times MN$.
    pub fn transform_channel(&self, h: &CMat) -> Result<CMat> {
        let mn = self.m * self.n;
        ensure_shape(h.nrows() == mn && h.ncols() == mn, || {
            format!("a {}×{} channel for frames of {mn}", h.nrows(), h.ncols())
        })?;
        // columns: H (F^H ⊗ I) e_q, then rows: (F ⊗ I)
        let mut out = CMat::zeros((mn, mn));
        for q in 0..mn {
            let mut e = vec![C64::new(0.0, 0.0); mn];
            e[q] = C64::new(1.0, 0.0);
            let s = self.doppler_dft(&e, true);
            let hs: Vec<C64> = (0..mn).map(|p| (0..mn).map(|j| h[[p, j]] * s[j]).sum()).collect();
            for (p, v) in self.doppler_dft(&hs, false).into_iter().enumerate() {
                out[[p, q]] = v;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_channel_without_delay_or_doppler_is_the_identity_and_rows_are_sparse() {
        let o = Otfs::new(6, 4, 0).unwrap();
        let h = o.effective_channel(&[(C64::new(0.5, 0.0), 0, 0)]).unwrap();
        for p in 0..24 {
            for q in 0..24 {
                assert_eq!(h[[p, q]], if p == q { C64::new(0.5, 0.0) } else { C64::new(0.0, 0.0) });
            }
        }
        let paths = [(C64::new(1.0, 0.0), 0, 0), (C64::new(0.3, 0.2), 2, 1), (C64::new(-0.4, 0.1), 5, -2)];
        let h = o.effective_channel(&paths).unwrap();
        for p in 0..24 {
            assert_eq!((0..24).filter(|&q| h[[p, q]].norm() > 0.0).count(), 3);
            assert_eq!((0..24).filter(|&q| h[[q, p]].norm() > 0.0).count(), 3);
        }
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
        let o = Otfs::new(2, 2, 0).unwrap();
        let (z0, one) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
        let (j4, mj4, mj2) =
            (cis(std::f64::consts::FRAC_PI_2), cis(-std::f64::consts::FRAC_PI_2), cis(-std::f64::consts::PI));
        let cases: [((usize, i64), Matrix); 4] = [
            ((0, 0), [[one, z0, z0, z0], [z0, one, z0, z0], [z0, z0, one, z0], [z0, z0, z0, one]]),
            ((1, 0), [[z0, one, z0, z0], [one, z0, z0, z0], [z0, z0, z0, mj2], [z0, z0, one, z0]]),
            ((0, 1), [[z0, z0, one, z0], [z0, z0, z0, j4], [one, z0, z0, z0], [z0, j4, z0, z0]]),
            ((1, 1), [[z0, z0, z0, j4], [z0, z0, one, z0], [z0, mj4, z0, z0], [one, z0, z0, z0]]),
        ];
        for ((l, k), want) in cases {
            let t = o.effective_channel(&[(one, l, k)]).unwrap();
            for p in 0..4 {
                for q in 0..4 {
                    assert!((t[[p, q]] - want[p][q]).norm() < 1e-12, "l {l} k {k} ({p}, {q}): {}", t[[p, q]]);
                }
            }
        }
    }
}

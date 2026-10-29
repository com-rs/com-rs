//! Filter-bank multicarrier with offset QAM (FBMC/OQAM, staggered multitone).
//!
//! # Model
//!
//! $M$ subcarriers spaced by $1/T$ ($T = M$ samples) carry real symbols $a_{m,n}$ every
//! $T/2$, each on a time and frequency shift of a real, symmetric prototype filter $h$
//! of $L = KM - 1$ taps ($K$ the overlapping factor; Bouhadda et al. 2014, eqs.
//! (3)–(4); Nissel & Rupp 2017, eq. (2)):
//!
//! $$
//! s\[k\] = \sum_{n} \sum_{m=0}^{M-1} a_{m,n}\, g_{m,n}\[k\], \qquad
//! g_{m,n}\[k\] = h\[k - nM/2\]\, e^{j 2\pi m (k - nM/2 - D/2)/M}\, e^{j \frac{\pi}{2}(m + n)},
//! $$
//!
//! with $D = L - 1$, so that the exponential is centred on the filter. The pulses are
//! orthogonal in the real sense only, $\Re\langle g_{m,n}, g_{m',n'} \rangle =
//! \delta_{mm'}\delta_{nn'}$ (nearly, for PHYDYAS: below $5 \cdot 10^{-4}$ per
//! neighbour for $K = 4$, $5 \cdot 10^{-3}$ for $K = 3$, $0.05$ for $K = 2$):
//! the receiver projects on $g_{m,n}$ and keeps the real part,
//! the imaginary part being the intrinsic interference of the neighbours
//! ([`Fbmc::interference`]). A complex QAM symbol is split into two real ones on
//! consecutive half-symbols ([`oqam_stagger`]), the real part first on the even
//! subcarriers and the imaginary part first on the odd ones, so that FBMC/OQAM carries
//! as many complex symbols as OFDM without a cyclic prefix ($TF = 1$).
//!
//! The pulses have unit energy: with QAM symbols of unit energy (real symbols of
//! variance 1/2) on all subcarriers the signal has unit power, and complex noise of
//! variance $\sigma^{2}$ per sample leaves a variance $\sigma^{2}/2$ on every real
//! estimate, the SNR of OFDM per dimension (Nissel & Rupp 2017, eq. (18)).
//!
//! The modulator and the demodulator are polyphase: one FFT of size $M$ per
//! half-symbol and $L$ products.
//!
//! [`Prototype::Phydyas`] is the filter of the PHYDYAS project (Bellanger 2010),
//! sampled from its frequency response $H_{0} = 1$, $H_{1}, \dots, H_{K-1}$:
//! $h\[k\] = H_{0} + 2\sum_{i=1}^{K-1} (-1)^{i} H_{i} \cos\left(\frac{2\pi i (k+1)}{KM}\right)$,
//! $k = 0, \dots, KM - 2$, normalised to unit energy; for $K = 4$, $H_{1} = 0.97195983$,
//! $H_{2} = 1/\sqrt{2}$, $H_{3} = 0.23514695$ (the transmultiplexer response of Table 1
//! of Bouhadda et al. 2014 to three decimals, checked by [`Fbmc::interference`]).
//!
//! # Examples
//! ```
//! use com_waveform::fbmc::{Fbmc, oqam_stagger, oqam_destagger};
//! use com_math::C64;
//! let f = Fbmc::new(16, 4)?;
//! // 6 complex symbols per subcarrier: 12 real half-symbols
//! let c: Vec<C64> = (0..16 * 6).map(|i| C64::new(if i % 3 == 0 { 1.0 } else { -1.0 }, if i % 5 == 0 { -1.0 } else { 1.0 })).collect();
//! let a = oqam_stagger(&c, 16);
//! let s = f.modulate(&a)?;
//! assert_eq!(s.len(), f.len(12));
//! let y = f.demodulate(&s, 12)?;
//! let back = oqam_destagger(&y.iter().map(|v| v.re).collect::<Vec<f64>>(), 16);
//! // PHYDYAS is nearly perfect reconstruction: a residual of a few 1e-3
//! assert!(back.iter().zip(&c).all(|(a, b)| (a - b).norm() < 1e-2));
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. B. Farhang-Boroujeny, "OFDM versus filter bank multicarrier," *IEEE Signal
//!    Process. Mag.*, 28(3), 92–112, 2011.
//! 2. M. Bellanger, "FBMC physical layer: a primer," PHYDYAS project (FP7-ICT
//!    211887), 2010.
//! 3. H. Bouhadda, H. Shaiek, D. Roviras, R. Zayani, Y. Medjahdi and R. Bouallegue,
//!    "Theoretical analysis of BER performance of nonlinearly amplified FBMC/OQAM and
//!    OFDM signals," *EURASIP J. Adv. Signal Process.*, 2014:60.
//! 4. R. Nissel and M. Rupp, "OFDM and FBMC-OQAM in doubly-selective channels:
//!    calculating the bit error probability," *IEEE Commun. Lett.*, 21(6), 1297–1300,
//!    2017.

use com_core::{Error, Result, ensure, ensure_shape};
use com_math::{C64, cis, fft};

/// The prototype filter of an [`Fbmc`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prototype {
    /// The PHYDYAS filter of Bellanger (overlapping factors 2, 3 and 4).
    Phydyas,
}

impl Prototype {
    /// The $KM - 1$ taps of unit energy for `subcarriers` $M$ and overlapping factor
    /// `overlap` $K$.
    ///
    /// # Errors
    /// [`Error::Config`] for an overlapping factor the prototype does not define.
    pub fn taps(self, subcarriers: usize, overlap: usize) -> Result<Vec<f64>> {
        let freq: &[f64] = match (self, overlap) {
            (Prototype::Phydyas, 2) => &[1.0, std::f64::consts::FRAC_1_SQRT_2],
            (Prototype::Phydyas, 3) => &[1.0, 0.911_437_83, 0.411_437_83],
            (Prototype::Phydyas, 4) => &[1.0, 0.971_959_83, std::f64::consts::FRAC_1_SQRT_2, 0.235_146_95],
            _ => {
                return Err(Error::Config(format!(
                    "no PHYDYAS filter for an overlapping factor of {overlap}: 2, 3 or 4"
                )));
            }
        };
        let km = (overlap * subcarriers) as f64;
        let h: Vec<f64> = (0..overlap * subcarriers - 1)
            .map(|k| {
                let tail: f64 = freq
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(i, hi)| {
                        let sign = if i % 2 == 1 { -1.0 } else { 1.0 };
                        2.0 * sign * hi * (std::f64::consts::TAU * i as f64 * (k + 1) as f64 / km).cos()
                    })
                    .sum();
                freq[0] + tail
            })
            .collect();
        let norm = h.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok(h.into_iter().map(|v| v / norm).collect())
    }
}

/// An FBMC/OQAM modem (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Fbmc {
    subcarriers: usize,
    overlap: usize,
    taps: Vec<f64>,
}

impl Fbmc {
    /// `subcarriers` $M$ (even) with the PHYDYAS prototype of overlapping factor
    /// `overlap`.
    ///
    /// # Errors
    /// [`Error::Config`] for an odd or zero number of subcarriers or an unknown
    /// overlapping factor.
    pub fn new(subcarriers: usize, overlap: usize) -> Result<Fbmc> {
        Fbmc::with_prototype(subcarriers, overlap, Prototype::Phydyas)
    }

    /// [`Fbmc::new`] with another prototype.
    ///
    /// # Errors
    /// As [`Fbmc::new`].
    pub fn with_prototype(subcarriers: usize, overlap: usize, prototype: Prototype) -> Result<Fbmc> {
        ensure(subcarriers >= 2 && subcarriers % 2 == 0, || {
            format!("FBMC/OQAM needs an even number of subcarriers, got {subcarriers}")
        })?;
        Ok(Fbmc { subcarriers, overlap, taps: prototype.taps(subcarriers, overlap)? })
    }

    /// The number of subcarriers $M$.
    pub fn subcarriers(&self) -> usize {
        self.subcarriers
    }

    /// The overlapping factor $K$.
    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// The prototype filter (unit energy, $KM - 1$ taps).
    pub fn taps(&self) -> &[f64] {
        &self.taps
    }

    /// The number of samples of `half_symbols` real symbols per subcarrier:
    /// $(n - 1) M/2 + L$.
    pub fn len(&self, half_symbols: usize) -> usize {
        if half_symbols == 0 { 0 } else { (half_symbols - 1) * self.subcarriers / 2 + self.taps.len() }
    }

    /// The pulse $g_{m,n}$ on its support: `(first sample, values)`.
    pub fn pulse(&self, m: usize, n: usize) -> (usize, Vec<C64>) {
        let mm = self.subcarriers as f64;
        let centre = (self.taps.len() - 1) as f64 / 2.0;
        let phase = cis(std::f64::consts::FRAC_PI_2 * ((m + n) % 4) as f64);
        let values = self
            .taps
            .iter()
            .enumerate()
            .map(|(i, h)| cis(std::f64::consts::TAU * m as f64 * (i as f64 - centre) / mm) * phase * *h)
            .collect();
        (n * self.subcarriers / 2, values)
    }

    /// The intrinsic interference $\langle g_{m+\Delta m, n+\Delta n}, g_{m,n} \rangle =
    /// \sum_{k} g_{m+\Delta m,n+\Delta n}\[k\]\, g_{m,n}^{*}\[k\]$ that the symbol
    /// $(m + \Delta m, n + \Delta n)$ leaves on the estimate of $(m, n)$: 1 for
    /// $\Delta m = \Delta n = 0$, purely imaginary otherwise (Bouhadda et al. 2014, eq. (7)
    /// and Table 1). The subcarrier index is taken modulo $M$.
    pub fn interference(&self, m: usize, n: usize, dm: isize, dn: isize) -> C64 {
        let mm = self.subcarriers as isize;
        let m2 = (m as isize + dm).rem_euclid(mm) as usize;
        // a common time origin far enough for negative shifts
        let base = 4 * self.overlap;
        let (n1, n2) = (n + base, (n as isize + base as isize + dn) as usize);
        let (s1, g1) = self.pulse(m, n1);
        let (s2, g2) = self.pulse(m2, n2);
        let (start, end) = (s1.max(s2), (s1 + g1.len()).min(s2 + g2.len()));
        // the phase of the pulses depends on n modulo 4: shifting both by `base` (a
        // multiple of 4) changes nothing
        (start..end).map(|k| g2[k - s2] * g1[k - s1].conj()).sum()
    }

    /// Modulate real symbols `a` laid out half-symbol after half-symbol,
    /// `a[n * M + m]`.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `a` holds whole half-symbols.
    pub fn modulate(&self, a: &[f64]) -> Result<Vec<C64>> {
        let m = self.subcarriers;
        ensure_shape(a.len() % m == 0, || format!("{} real symbols for {m} subcarriers", a.len()))?;
        let half_symbols = a.len() / m;
        let shift = (self.taps.len() - 1) / 2;
        let plan = fft::plan(m);
        let mut s = vec![C64::new(0.0, 0.0); self.len(half_symbols)];
        let mut v = vec![C64::new(0.0, 0.0); m];
        for (n, row) in a.chunks_exact(m).enumerate() {
            for (k, (vk, ak)) in v.iter_mut().zip(row).enumerate() {
                *vk = cis(std::f64::consts::FRAC_PI_2 * ((k + n) % 4) as f64) * *ak;
            }
            // Σ_m c_m e^{j2πmj/M}: the inverse FFT divides by M
            plan.inverse(&mut v);
            let offset = n * m / 2;
            for (i, h) in self.taps.iter().enumerate() {
                s[offset + i] += v[(i + m * self.overlap - shift) % m] * (h * m as f64);
            }
        }
        Ok(s)
    }

    /// Project `r` on every pulse of `half_symbols` half-symbols: $y_{m,n} =
    /// \langle r, g_{m,n} \rangle$, laid out `y[n * M + m]`; the real parts estimate the
    /// symbols, the imaginary parts hold the intrinsic interference.
    ///
    /// # Errors
    /// [`Error::Shape`] if `r` is shorter than [`Fbmc::len`] of `half_symbols`.
    pub fn demodulate(&self, r: &[C64], half_symbols: usize) -> Result<Vec<C64>> {
        let m = self.subcarriers;
        ensure_shape(r.len() >= self.len(half_symbols), || {
            format!("{} samples for {half_symbols} half-symbols ({} needed)", r.len(), self.len(half_symbols))
        })?;
        let shift = (self.taps.len() - 1) / 2;
        let plan = fft::plan(m);
        let mut y = Vec::with_capacity(half_symbols * m);
        let mut w = vec![C64::new(0.0, 0.0); m];
        for n in 0..half_symbols {
            w.fill(C64::new(0.0, 0.0));
            let offset = n * m / 2;
            for (i, h) in self.taps.iter().enumerate() {
                w[(i + m * self.overlap - shift) % m] += r[offset + i] * *h;
            }
            plan.forward(&mut w);
            for (k, wk) in w.iter().enumerate() {
                y.push(wk * cis(-std::f64::consts::FRAC_PI_2 * ((k + n) % 4) as f64));
            }
        }
        Ok(y)
    }
}

/// Split complex symbols `c[k * M + m]` ($k$ the complex symbol, $m$ the subcarrier) into
/// real symbols on twice as many half-symbols, `a[n * M + m]`: the real part on the even
/// half-symbol and the imaginary part on the odd one for the even subcarriers, the
/// reverse for the odd subcarriers.
///
/// # Panics
/// If `c` does not hold whole symbols of `m` subcarriers.
pub fn oqam_stagger(c: &[C64], m: usize) -> Vec<f64> {
    assert!(m > 0 && c.len() % m == 0, "whole symbols of {m} subcarriers");
    let mut a = vec![0.0; 2 * c.len()];
    for (k, row) in c.chunks_exact(m).enumerate() {
        for (sub, v) in row.iter().enumerate() {
            let (first, second) = if sub % 2 == 0 { (v.re, v.im) } else { (v.im, v.re) };
            a[2 * k * m + sub] = first;
            a[(2 * k + 1) * m + sub] = second;
        }
    }
    a
}

/// The inverse of [`oqam_stagger`]: complex symbols from the real estimates
/// `a[n * M + m]` of an even number of half-symbols.
///
/// # Panics
/// If `a` does not hold an even number of half-symbols of `m` subcarriers.
pub fn oqam_destagger(a: &[f64], m: usize) -> Vec<C64> {
    assert!(m > 0 && a.len() % (2 * m) == 0, "an even number of half-symbols of {m} subcarriers");
    let mut c = Vec::with_capacity(a.len() / 2);
    for pair in a.chunks_exact(2 * m) {
        let (first, second) = pair.split_at(m);
        for sub in 0..m {
            c.push(if sub % 2 == 0 { C64::new(first[sub], second[sub]) } else { C64::new(second[sub], first[sub]) });
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_pulses_are_real_orthogonal_and_of_unit_energy() {
        for (m, k) in [(8usize, 2usize), (16, 3), (32, 4)] {
            let f = Fbmc::new(m, k).unwrap();
            assert!((f.taps().iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
            for m0 in [0usize, 1, 5] {
                for n0 in [0usize, 1, 2, 3] {
                    for dm in -2isize..=2 {
                        for dn in -(2 * k as isize)..=(2 * k as isize) {
                            let v = f.interference(m0, n0, dm, dn);
                            let want = if dm == 0 && dn == 0 { 1.0 } else { 0.0 };
                            // PHYDYAS is only nearly perfect reconstruction, the closer the
                            // longer the filter
                            let tol = match k {
                                2 => 0.05,
                                3 => 5e-3,
                                _ => 5e-4,
                            };
                            assert!((v.re - want).abs() < tol, "M {m} K {k} ({m0},{n0}) + ({dm},{dn}): {v}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn modulation_is_the_sum_of_the_pulses_and_demodulation_their_projection() {
        let f = Fbmc::new(8, 4).unwrap();
        let mut r = Rng::new(3);
        let n = 5;
        let a: Vec<f64> = (0..8 * n).map(|_| r.gaussian()).collect();
        let s = f.modulate(&a).unwrap();
        let mut direct = vec![C64::new(0.0, 0.0); f.len(n)];
        for nn in 0..n {
            for m in 0..8 {
                let (start, g) = f.pulse(m, nn);
                for (i, v) in g.iter().enumerate() {
                    direct[start + i] += v * a[nn * 8 + m];
                }
            }
        }
        assert!(s.iter().zip(&direct).all(|(x, y)| (x - y).norm() < 1e-12));
        let y = f.demodulate(&s, n).unwrap();
        for nn in 0..n {
            for m in 0..8 {
                let (start, g) = f.pulse(m, nn);
                let proj: C64 = g.iter().enumerate().map(|(i, v)| s[start + i] * v.conj()).sum();
                assert!((y[nn * 8 + m] - proj).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn unit_power_and_half_the_noise_on_each_real_estimate() {
        let (m, n) = (64usize, 400usize);
        let f = Fbmc::new(m, 4).unwrap();
        let mut r = Rng::new(9);
        let a: Vec<f64> = (0..m * n).map(|_| if r.below(2) == 0 { 0.5f64.sqrt() } else { -0.5f64.sqrt() }).collect();
        let s = f.modulate(&a).unwrap();
        // the steady part, away from the ramps
        let mid = &s[f.taps().len()..s.len() - f.taps().len()];
        let p = mid.iter().map(C64::norm_sqr).sum::<f64>() / mid.len() as f64;
        assert!((p - 1.0).abs() < 0.02, "{p}");
        let sigma2: f64 = 0.1;
        let noisy: Vec<C64> = s.iter().map(|v| v + r.complex_gaussian() * sigma2.sqrt()).collect();
        let y = f.demodulate(&noisy, n).unwrap();
        let err: Vec<f64> = y.iter().zip(&a).map(|(v, a)| v.re - a).collect();
        let var = err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64;
        assert!((var / (sigma2 / 2.0) - 1.0).abs() < 0.03, "{var}");
    }

    #[test]
    fn the_spectrum_is_confined_where_that_of_ofdm_is_not() {
        // Farhang-Boroujeny 2011, Sec. "FBMC": the subcarrier filters of FBMC leave
        // almost nothing beyond the neighbouring subcarriers, the sinc of OFDM decays as
        // 1/f. One subcarrier, 16 half-symbols, 8 subcarriers away (half-way between
        // subcarriers): more than 50 dB below its peak for PHYDYAS, about 25 dB for OFDM.
        let (m, n) = (32usize, 16usize);
        let f = Fbmc::new(m, 4).unwrap();
        let mut a = vec![0.0; m * n];
        let mut r = Rng::new(4);
        let mut p = vec![0.0; 16 * m];
        for _ in 0..50 {
            for nn in 0..n {
                a[nn * m] = if r.below(2) == 0 { 1.0 } else { -1.0 };
            }
            let mut s = f.modulate(&a).unwrap();
            s.resize(16 * m * 8, C64::new(0.0, 0.0));
            let spec = fft::fft(&s);
            // bins of the zero-padded transform: subcarrier k at 8 k (the grid is 8 times finer)
            for (i, v) in p.iter_mut().enumerate() {
                *v += spec[i * 8].norm_sqr();
            }
        }
        let at = |k: f64| p[((k * 16.0).round() as isize).rem_euclid(16 * m as isize) as usize];
        let fbmc_db = 10.0 * (at(8.5) / at(0.0)).log10();
        // OFDM: a rectangular symbol of M samples, |sinc(8.5)|²
        let ofdm_db = 20.0 * com_math::sinc(8.5).abs().log10();
        assert!(fbmc_db < -50.0 && ofdm_db > -30.0, "FBMC {fbmc_db} dB, OFDM {ofdm_db} dB");
    }

    #[test]
    fn staggering_round_trips() {
        let c: Vec<C64> = (0..12).map(|i| C64::new(i as f64, -(i as f64) - 0.5)).collect();
        let a = oqam_stagger(&c, 4);
        assert_eq!(a.len(), 24);
        assert_eq!((a[0], a[4], a[1], a[5]), (0.0, -0.5, -1.5, 1.0));
        assert_eq!(oqam_destagger(&a, 4), c);
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    /// H. Bouhadda et al., EURASIP J. Adv. Signal Process. 2014:60, Table 1: the
    /// transmultiplexer response of the PHYDYAS filter ($K = 4$) around a symbol
    /// $(m_{0}, n_{0})$, to the three decimals printed. With the phases of the module
    /// (those of Nissel & Rupp, $\frac{\pi}{2}(m + n)$, with $m_{0}$ and $n_{0}$ even), the
    /// rows $m_{0} - 1$ and $m_{0} + 1$ are those of the table exchanged: the table counts
    /// the subcarriers the other way.
    #[test]
    fn bouhadda_2014_table_1() {
        let f = Fbmc::new(64, 4).unwrap();
        let rows: [(isize, [f64; 7]); 3] = [
            (1, [0.043, 0.125, 0.206, 0.239, 0.206, 0.125, 0.043]),
            (0, [-0.067, 0.0, -0.564, 0.0, 0.564, 0.0, 0.067]),
            (-1, [0.043, -0.125, 0.206, -0.239, 0.206, -0.125, 0.043]),
        ];
        for (dm, row) in rows {
            for (j, want) in row.iter().enumerate() {
                let dn = j as isize - 3;
                let v = f.interference(10, 10, dm, dn);
                if dm == 0 && dn == 0 {
                    assert!((v - 1.0).norm() < 1e-3);
                } else {
                    assert!(v.re.abs() < 5e-4 && (v.im - want).abs() < 5e-4, "({dm}, {dn}): {v} against {want}j");
                }
            }
        }
    }
}

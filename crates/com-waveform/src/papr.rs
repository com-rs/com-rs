//! Peak-to-average power ratio (PAPR) of OFDM: its distribution, and the classical
//! reduction techniques.
//!
//! The PAPR of a block $x$ is $\max_{n} |x\[n\]|^{2} / \mathbb{E}|x\[n\]|^{2}$, measured on
//! the oversampled signal ([`crate::ofdm::Ofdm`] with $L \ge 4$). Its complementary
//! cumulative distribution (CCDF) $\Pr(\mathrm{PAPR} > \gamma)$ is the standard figure of
//! merit: an OFDM signal is nearly Gaussian, and a peak is rare but large.
//!
//! # Distribution
//!
//! - Nyquist-rate samples of $N$ independent Rayleigh amplitudes (van Nee & Prasad):
//!   [`ccdf_nyquist`], $1 - (1 - e^{-\gamma})^{N}$;
//! - the continuous (oversampled) signal, for large $N$ (Ochiai & Imai 2001):
//!   [`ccdf_oversampled`], $1 - \exp\left(-N \sqrt{\pi \gamma / 3}\\, e^{-\gamma}\right)$;
//! - SLM with $U$ independent candidates: [`ccdf_slm`], the CCDF raised to the power $U$.
//!
//! # Clipping
//!
//! An envelope limiter acting on a complex Gaussian signal is, by Bussgang's theorem, a
//! gain plus an uncorrelated distortion, whose closed form
//! ([`com_dsp::nonlinear::clipping_bussgang`]) gives the signal-to-distortion ratio, the
//! error floor of clipping.
//!
//! # Reduction techniques
//!
//! | Technique | Principle | Distortion | Side information | Reference |
//! |---|---|---|---|---|
//! | [`Clip`] | envelope limiter | in band and out of band | none | Li & Cimini 1998 |
//! | [`ClipFilter`] | repeated clipping and out-of-band filtering | in band | none | Armstrong 2002 |
//! | [`Slm`] | best of $U$ phase-rotated copies | none | $\log_{2} U$ bits | Bäuml, Fischer & Huber 1996 |
//! | [`Pts`] | best phase combination of $V$ subblocks | none | $(V - 1)\log_{2} W$ bits | Müller & Huber 1997 |
//! | [`ToneReservation`] | peak-cancelling signal on reserved tones | none (rate loss) | none | Tellado 2000 |
//! | [`Ace`] | outer constellation points pushed outwards | none (power increase) | none | Krongold & Jones 2003 |
//! | [`Companding`] | μ-law or exponential compression of the envelope, expansion at the receiver | yes | none | Wang, Tjhung & Ng 1999; Jiang, Yang & Song 2005 |
//!
//! DFT-spread OFDM ([`crate::ofdm::Ofdm::dft_spread`], Myung, Lim & Goodman 2006)
//! removes most of the PAPR at the source.
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::Ofdm;
//! use com_waveform::papr::{ccdf_oversampled, Slm};
//! use com_math::{papr_db, C64};
//! let ofdm = Ofdm::new(256, 256)?;
//! let qpsk: Vec<C64> = (0..256).map(|i| C64::new(if i % 3 == 0 { 1.0 } else { -1.0 }, if i % 5 < 2 { 1.0 } else { -1.0 }) / 2f64.sqrt()).collect();
//! let (x, side) = ofdm.modulate_with(&qpsk, &Slm { candidates: 16, seed: 1 })?;
//! assert!(papr_db(&x) <= papr_db(&ofdm.modulate(&qpsk)?));
//! assert_eq!(ofdm.demodulate_with(&x, &Slm { candidates: 16, seed: 1 }, &side)?.len(), 256);
//! assert!((ccdf_oversampled(256, 10.0) - 0.0369).abs() < 1e-3);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. R. van Nee and R. Prasad, *OFDM for Wireless Multimedia Communications*, Artech
//!    House, 2000, ch. 6.
//! 2. H. Ochiai and H. Imai, "On the distribution of the peak-to-average power ratio in
//!    OFDM signals," *IEEE Trans. Commun.*, 49(2), 282–289, 2001.
//! 3. H. Ochiai and H. Imai, "Performance analysis of deliberately clipped OFDM signals,"
//!    *IEEE Trans. Commun.*, 50(1), 89–101, 2002.
//! 4. D. Dardari, V. Tralli and A. Vaccari, "A theoretical characterization of nonlinear
//!    distortion effects in OFDM systems," *IEEE Trans. Commun.*, 48(10), 1755–1764, 2000.
//! 5. J. J. Bussgang, "Crosscorrelation functions of amplitude-distorted Gaussian
//!    signals," MIT RLE Tech. Rep. 216, 1952.
//! 6. S. H. Han and J. H. Lee, "An overview of peak-to-average power ratio reduction
//!    techniques for multicarrier transmission," *IEEE Wireless Commun.*, 12(2), 56–65,
//!    2005.
//! 7. X. Li and L. J. Cimini, "Effects of clipping and filtering on the performance of
//!    OFDM," *IEEE Commun. Lett.*, 2(5), 131–133, 1998.
//! 8. J. Armstrong, "Peak-to-average power reduction for OFDM by repeated clipping and
//!    frequency domain filtering," *Electron. Lett.*, 38(5), 246–247, 2002.
//! 9. R. W. Bäuml, R. F. H. Fischer and J. B. Huber, "Reducing the peak-to-average power
//!    ratio of multicarrier modulation by selected mapping," *Electron. Lett.*, 32(22),
//!    2056–2057, 1996.
//! 10. S. H. Müller and J. B. Huber, "OFDM with reduced peak-to-average power ratio by
//!     optimum combination of partial transmit sequences," *Electron. Lett.*, 33(5),
//!     368–369, 1997.
//! 11. J. Tellado, *Multicarrier Modulation with Low PAR: Applications to DSL and
//!     Wireless*, Kluwer, 2000.
//! 12. B. S. Krongold and D. L. Jones, "PAR reduction in OFDM via active constellation
//!     extension," *IEEE Trans. Broadcast.*, 49(3), 258–268, 2003.
//! 13. X. Wang, T. T. Tjhung and C. S. Ng, "Reduction of peak-to-average power ratio of
//!     OFDM system using a companding technique," *IEEE Trans. Broadcast.*, 45(3),
//!     303–307, 1999.
//! 14. T. Jiang, Y. Yang and Y.-H. Song, "Exponential companding technique for PAPR
//!     reduction in OFDM systems," *IEEE Trans. Broadcast.*, 51(2), 244–248, 2005.
//! 15. H. G. Myung, J. Lim and D. J. Goodman, "Single carrier FDMA for uplink wireless
//!     transmission," *IEEE Veh. Technol. Mag.*, 1(3), 30–38, 2006.

use crate::ofdm::{Ofdm, PaprReduction};
use com_core::rng::Rng;
use com_core::{Error, Result, ensure};
use com_dsp::signal::map_envelope;
use com_mapping::constellation::Constellation;
use com_math::{C64, cis, db2lin, papr_db};
use std::f64::consts::PI;

// ------------------------------------------------------------------ distribution

/// $\Pr(\mathrm{PAPR} > \gamma)$ of $n$ independent Rayleigh samples (Nyquist rate),
/// $1 - (1 - e^{-\gamma})^{n}$ (van Nee & Prasad 2000; Ochiai & Imai 2001, eq. (11)).
pub fn ccdf_nyquist(n: usize, papr_db: f64) -> f64 {
    let g = db2lin(papr_db);
    1.0 - (1.0 - (-g).exp()).powi(n as i32)
}

/// $\Pr(\mathrm{PAPR} > \gamma)$ of the continuous OFDM signal of $n$ subcarriers,
/// $1 - \exp(-n \sqrt{\pi\gamma/3}\\, e^{-\gamma})$ (Ochiai & Imai 2001, eq. (27), the
/// asymptotic form of the level-crossing analysis, within the resolution of the
/// figures of $n \geq 100$).
pub fn ccdf_oversampled(n: usize, papr_db: f64) -> f64 {
    let g = db2lin(papr_db);
    (1.0 - (-(n as f64) * (PI * g / 3.0).sqrt() * (-g).exp()).exp()).clamp(0.0, 1.0)
}

/// The CCDF after SLM with `candidates` independent candidates: that of one candidate
/// raised to the power $U$ (oversampled signal).
pub fn ccdf_slm(n: usize, candidates: usize, papr_db: f64) -> f64 {
    ccdf_oversampled(n, papr_db).powi(candidates as i32)
}

// ------------------------------------------------------------------ clipping

/// Limit the envelope to `a`, keeping the phase.
fn limit(x: &mut [C64], a: f64) {
    map_envelope(x, |r| C64::new(if r > a { a / r } else { 1.0 }, 0.0));
}

fn amplitude(ratio_db: f64) -> f64 {
    com_math::db2amp(ratio_db)
}

/// Clipping of the envelope at `ratio_db` above the (unit) average power
/// (Li & Cimini 1998): in-band distortion and out-of-band regrowth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clip {
    /// Clipping ratio (dB).
    pub ratio_db: f64,
}

impl PaprReduction for Clip {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        let mut x = ofdm.to_time(&ofdm.spread(data))?;
        limit(&mut x, amplitude(self.ratio_db));
        Ok((x, 0))
    }
}

/// Repeated clipping and frequency-domain filtering (Armstrong 2002): every iteration
/// clips the envelope and removes the out-of-band components, which regrows some
/// peaks; a few iterations reach the target PAPR without any out-of-band emission.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipFilter {
    /// Clipping ratio (dB).
    pub ratio_db: f64,
    /// Clipping-and-filtering iterations.
    pub iterations: usize,
}

impl PaprReduction for ClipFilter {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        let mut x = ofdm.to_time(&ofdm.spread(data))?;
        let a = amplitude(self.ratio_db);
        for _ in 0..self.iterations {
            limit(&mut x, a);
            x = ofdm.filter(&x)?;
        }
        Ok((x, 0))
    }
}

// ------------------------------------------------------------------ selected mapping

/// Selected mapping (Bäuml, Fischer & Huber 1996): $U$ copies of the block rotated by
/// known phase sequences (the first one is the original), the one of lowest PAPR is
/// sent and its index is the side information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slm {
    /// Number of candidates $U$.
    pub candidates: usize,
    /// Seed of the phase sequences (shared by transmitter and receiver).
    pub seed: u64,
}

impl Slm {
    /// The phase sequence of candidate `u` ($\pm 1, \pm j$; all ones for `u = 0`).
    pub fn phases(&self, u: usize, k: usize) -> Vec<C64> {
        if u == 0 {
            return vec![C64::new(1.0, 0.0); k];
        }
        let mut r = com_core::rng::member(self.seed, u as u64);
        (0..k).map(|_| cis(PI / 2.0 * r.below(4) as f64)).collect()
    }
}

impl PaprReduction for Slm {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        ensure(self.candidates >= 1, || "SLM needs at least one candidate".into())?;
        let active = ofdm.spread(data);
        let mut best = (f64::INFINITY, Vec::new(), 0);
        for u in 0..self.candidates {
            let p = self.phases(u, active.len());
            let x = ofdm.to_time(&active.iter().zip(&p).map(|(a, b)| a * b).collect::<Vec<_>>())?;
            let v = papr_db(&x);
            if v < best.0 {
                best = (v, x, u);
            }
        }
        Ok((best.1, best.2))
    }

    fn recover(&self, ofdm: &Ofdm, active: Vec<C64>, side: usize) -> Result<Vec<C64>> {
        ensure(side < self.candidates, || format!("SLM side information {side} out of {}", self.candidates))?;
        let p = self.phases(side, active.len());
        Ok(ofdm.despread(&active.iter().zip(&p).map(|(a, b)| a * b.conj()).collect::<Vec<_>>()))
    }
}

// ------------------------------------------------------------------ partial transmit sequences

/// How the subcarriers are split into the subblocks of [`Pts`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    /// Contiguous subblocks.
    Adjacent,
    /// Subcarrier $i$ in subblock $i \bmod V$.
    Interleaved,
    /// A balanced random assignment (seeded): the best PAPR reduction.
    Pseudorandom(u64),
}

/// Partial transmit sequences (Müller & Huber 1997): the subcarriers are split into $V$
/// subblocks whose time signals are combined with phase factors from
/// $\\{e^{j 2\pi w / W}\\}$, the first fixed to one; the combination of lowest PAPR is
/// sent (exhaustive search over $W^{V-1}$ combinations) and its index is the side
/// information.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pts {
    /// Number of subblocks $V$.
    pub subblocks: usize,
    /// Number of phase factors $W$ (2: $\pm 1$, 4: $\pm 1, \pm j$).
    pub phases: usize,
    /// The partition of the subcarriers.
    pub partition: Partition,
}

impl Pts {
    /// The subblock of every active subcarrier.
    pub fn assignment(&self, k: usize) -> Vec<usize> {
        let v = self.subblocks.max(1);
        match self.partition {
            Partition::Adjacent => (0..k).map(|i| i * v / k).collect(),
            Partition::Interleaved => (0..k).map(|i| i % v).collect(),
            Partition::Pseudorandom(seed) => {
                let mut a: Vec<usize> = (0..k).map(|i| i % v).collect();
                Rng::new(seed).shuffle(&mut a);
                a
            }
        }
    }

    /// The phase factor of every subblock for the combination `index`.
    pub fn factors(&self, index: usize) -> Vec<C64> {
        let mut f = vec![C64::new(1.0, 0.0)];
        let mut rest = index;
        for _ in 1..self.subblocks {
            f.push(cis(2.0 * PI * (rest % self.phases) as f64 / self.phases as f64));
            rest /= self.phases;
        }
        f
    }

    fn combinations(&self) -> Result<usize> {
        ensure(self.subblocks >= 1 && self.phases >= 1, || "PTS needs subblocks and phases".into())?;
        let n = (1..self.subblocks).try_fold(1usize, |acc, _| acc.checked_mul(self.phases));
        n.filter(|n| *n <= 1 << 20).ok_or_else(|| Error::Config("too many PTS combinations".into()))
    }
}

impl PaprReduction for Pts {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        let combinations = self.combinations()?;
        let active = ofdm.spread(data);
        let assign = self.assignment(active.len());
        let parts = (0..self.subblocks)
            .map(|v| {
                let masked: Vec<C64> =
                    active.iter().zip(&assign).map(|(a, &b)| if b == v { *a } else { C64::new(0.0, 0.0) }).collect();
                ofdm.to_time(&masked)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut best = (f64::INFINITY, 0usize);
        let mut x = vec![C64::new(0.0, 0.0); ofdm.fft_len()];
        for index in 0..combinations {
            let f = self.factors(index);
            for (n, v) in x.iter_mut().enumerate() {
                *v = parts.iter().zip(&f).map(|(p, b)| p[n] * b).sum();
            }
            let p = papr_db(&x);
            if p < best.0 {
                best = (p, index);
            }
        }
        let f = self.factors(best.1);
        let x = (0..ofdm.fft_len()).map(|n| parts.iter().zip(&f).map(|(p, b)| p[n] * b).sum()).collect();
        Ok((x, best.1))
    }

    fn recover(&self, ofdm: &Ofdm, active: Vec<C64>, side: usize) -> Result<Vec<C64>> {
        ensure(side < self.combinations()?, || format!("PTS side information {side} out of range"))?;
        let f = self.factors(side);
        let assign = self.assignment(active.len());
        Ok(ofdm.despread(&active.iter().zip(&assign).map(|(a, &v)| a * f[v].conj()).collect::<Vec<_>>()))
    }
}

// ------------------------------------------------------------------ tone reservation

/// Tone reservation (Tellado 2000): a few subcarriers carry no data but a
/// peak-cancelling signal. Every iteration projects the excess of the peaks above the
/// target amplitude (the clipping noise) onto the reserved tones, by least squares, and
/// subtracts it: a gradient descent on the peak excess that never touches the data
/// tones.
#[derive(Clone, Debug, PartialEq)]
pub struct ToneReservation {
    /// Indices of the reserved tones among the active subcarriers.
    pub reserved: Vec<usize>,
    /// Target amplitude, in dB above the average power.
    pub ratio_db: f64,
    /// Iterations.
    pub iterations: usize,
    /// Step of every iteration (1: the full projection).
    pub step: f64,
}

impl ToneReservation {
    /// `count` reserved tones drawn at random (seeded) among the `n_active` subcarriers,
    /// the usual choice (a random set gives a peaky cancelling kernel).
    pub fn random(n_active: usize, count: usize, seed: u64, ratio_db: f64) -> ToneReservation {
        let mut idx: Vec<usize> = (0..n_active).collect();
        Rng::new(seed).shuffle(&mut idx);
        let mut reserved: Vec<usize> = idx.into_iter().take(count.min(n_active)).collect();
        reserved.sort_unstable();
        ToneReservation { reserved, ratio_db, iterations: 20, step: 1.0 }
    }

    fn mask(&self, ofdm: &Ofdm) -> Result<Vec<bool>> {
        ensure(!ofdm.dft_spread, || "tone reservation needs plain OFDM".into())?;
        let mut m = vec![false; ofdm.n_active];
        for &i in &self.reserved {
            ensure(i < ofdm.n_active, || format!("reserved tone {i} is not an active subcarrier"))?;
            m[i] = true;
        }
        Ok(m)
    }
}

impl PaprReduction for ToneReservation {
    fn data_len(&self, ofdm: &Ofdm) -> usize {
        ofdm.n_active.saturating_sub(self.reserved.len())
    }

    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        let mask = self.mask(ofdm)?;
        let mut it = data.iter();
        let active: Vec<C64> = mask
            .iter()
            .map(|&r| if r { C64::new(0.0, 0.0) } else { *it.next().unwrap_or(&C64::new(0.0, 0.0)) })
            .collect();
        let mut x = ofdm.to_time(&active)?;
        let a = amplitude(self.ratio_db);
        for _ in 0..self.iterations {
            let mut clipped = x.clone();
            limit(&mut clipped, a);
            if clipped.iter().zip(&x).all(|(c, v)| c == v) {
                break;
            }
            let noise: Vec<C64> = clipped.iter().zip(&x).map(|(c, v)| c - v).collect();
            let projected: Vec<C64> = ofdm
                .to_active(&noise)?
                .into_iter()
                .zip(&mask)
                .map(|(v, &r)| if r { v } else { C64::new(0.0, 0.0) })
                .collect();
            for (v, c) in x.iter_mut().zip(ofdm.to_time(&projected)?) {
                *v += c * self.step;
            }
        }
        Ok((x, 0))
    }

    fn recover(&self, ofdm: &Ofdm, active: Vec<C64>, _side: usize) -> Result<Vec<C64>> {
        let mask = self.mask(ofdm)?;
        Ok(active.into_iter().zip(&mask).filter(|(_, r)| !**r).map(|(v, _)| v).collect())
    }
}

// ------------------------------------------------------------------ active constellation extension

/// Active constellation extension (Krongold & Jones 2003), clipping-based: the peaks
/// are clipped, and the resulting change of every subcarrier is kept only where it
/// moves an outer constellation point further out (which can only increase its
/// distance to the decision boundaries); a few iterations converge. For square QAM.
#[derive(Clone, Debug, PartialEq)]
pub struct Ace {
    /// The constellation of the data (square QAM or QPSK).
    pub constellation: Constellation,
    /// Clipping ratio of every iteration (dB).
    pub ratio_db: f64,
    /// Iterations.
    pub iterations: usize,
    /// Gain applied to the allowed extension at every iteration.
    pub step: f64,
}

impl PaprReduction for Ace {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        ensure(!ofdm.dft_spread, || "active constellation extension needs plain OFDM".into())?;
        let pts = self.constellation.points();
        let max_re = pts.iter().map(|p| p.re.abs()).fold(0.0, f64::max);
        let max_im = pts.iter().map(|p| p.im.abs()).fold(0.0, f64::max);
        let tol = 1e-9 * max_re.max(max_im).max(1e-300);
        let original: Vec<C64> = data.iter().map(|d| pts[self.constellation.nearest(*d)]).collect();
        let mut active = data.to_vec();
        let mut x = ofdm.to_time(&active)?;
        let a = amplitude(self.ratio_db);
        for _ in 0..self.iterations {
            let mut clipped = x.clone();
            limit(&mut clipped, a);
            let noise: Vec<C64> = clipped.iter().zip(&x).map(|(c, v)| c - v).collect();
            if noise.iter().all(|v| v.norm_sqr() == 0.0) {
                break;
            }
            for ((v, c), p) in active.iter_mut().zip(ofdm.to_active(&noise)?).zip(&original) {
                let re = if p.re.abs() >= max_re - tol && c.re * p.re > 0.0 { c.re } else { 0.0 };
                let im = if p.im.abs() >= max_im - tol && c.im * p.im > 0.0 { c.im } else { 0.0 };
                *v += C64::new(re, im) * self.step;
            }
            x = ofdm.to_time(&active)?;
        }
        Ok((x, 0))
    }
}

// ------------------------------------------------------------------ companding

/// Companding of the envelope at the transmitter and expansion at the receiver; the
/// compressed signal keeps the unit average power.
///
/// The expansion is exact without noise but amplifies the noise where the law is flat:
/// the exponential law of degree 2 squeezes every amplitude above about twice the RMS
/// into a narrow range below its maximum, and expanding them multiplies the noise by
/// 20 to 150 — an error floor. A receiver can instead skip the expansion and treat the
/// compression as a distortion (Bussgang gain plus noise), as for clipping.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Companding {
    /// μ-law (Wang, Tjhung & Ng 1999): $c(r) = V \ln(1 + \mu r/V) / \ln(1 + \mu)$, with
    /// $V$ at `peak_db` above the average power, times the gain that keeps the unit
    /// average power of a Rayleigh envelope. The gain is computed once per block, by a
    /// 64-point Gauss–Legendre rule within $10^{-13}$ (relative) of its integral for
    /// $\mu/V \le 10^{5}$; it differs from that of the former 6000-point Simpson rule,
    /// computed per sample, by that rule's error: $3 \cdot 10^{-7}$ (relative) at
    /// $\mu = 4$ and 9 dB, up to $2.6 \cdot 10^{-4}$ at $\mu = 255$ and 0 dB.
    MuLaw {
        /// Compression parameter $\mu$.
        mu: f64,
        /// Reference amplitude $V$ (dB above the average power).
        peak_db: f64,
    },
    /// Exponential companding (Jiang, Yang & Song 2005):
    /// $c(r) = \left[\alpha (1 - e^{-r^{2}})\right]^{1/d}$ with
    /// $\alpha = ((d + 2)/d)^{d/2}$, which turns the Rayleigh envelope into one of fixed
    /// distribution (uniform power for $d = 2$, a PAPR of 3 dB).
    Exponential {
        /// Degree $d$.
        degree: f64,
    },
}

/// The 64 nodes and weights of Gauss–Legendre on $[-1, 1]$ of the μ-law normalisation,
/// computed once.
fn mu_law_nodes() -> &'static (Vec<f64>, Vec<f64>) {
    static NODES: std::sync::OnceLock<(Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();
    NODES.get_or_init(|| com_math::gauss_legendre(64))
}

/// A companding law with the constants of its parameters computed once (the
/// normalisation of the μ-law is an integral): [`PaprReduction::reduce`] and
/// [`PaprReduction::restore_time`] build it once per block.
#[derive(Clone, Copy, Debug)]
enum Law {
    MuLaw { mu: f64, v: f64, gain: f64 },
    Exponential { degree: f64, alpha: f64 },
}

impl Law {
    fn compress(&self, r: f64) -> f64 {
        match *self {
            Law::MuLaw { mu, v, gain } => gain * v * (1.0 + mu * r / v).ln() / (1.0 + mu).ln(),
            Law::Exponential { degree: d, alpha } => (alpha * (1.0 - (-r * r).exp())).powf(1.0 / d),
        }
    }

    fn expand(&self, r: f64) -> f64 {
        match *self {
            Law::MuLaw { mu, v, gain } => {
                let c = r / gain;
                v / mu * ((1.0 + mu).powf(c / v) - 1.0)
            }
            Law::Exponential { degree: d, alpha } => {
                let u = (1.0 - r.powf(d) / alpha).max(1e-16);
                (-u.ln()).sqrt()
            }
        }
    }
}

impl Companding {
    /// The gain $g = 1/\sqrt{E\\{c(R)^{2}\\}}$ of the μ-law that keeps the unit average power,
    /// for a Rayleigh envelope $R$ of unit power ($R^{2} \sim \mathrm{Exp}(1)$) and $V$ =
    /// `v`. With $a = \mu/V$ and the change of variable $s = \ln(1 + a r)$,
    ///
    /// $$
    /// E\\{c(R)^{2}\\} = \frac{V^{2}}{\ln^{2}(1 + \mu)} \int_{0}^{\infty} 2 r e^{-r^{2}}
    /// \ln^{2}(1 + a r)\\, dr = \frac{V^{2}}{\ln^{2}(1 + \mu)} \int_{0}^{\ln(1 + a r_{\max})}
    /// \frac{2 r e^{-r^{2}} s^{2} e^{s}}{a}\\, ds, \qquad r = \frac{e^{s} - 1}{a},
    /// $$
    ///
    /// an entire integrand in $s$, integrated by the 64-point Gauss–Legendre rule up to
    /// $r_{\max} = 7$ (the tail beyond holds $e^{-49}$ of the power): within $10^{-13}$
    /// (relative) of the integral for $a \le 10^{5}$. The former 6000-point Simpson rule on
    /// $t = r^{2}$ erred by $6 \cdot 10^{-7}$ at $\mu = 4$, $V$ at 9 dB, and by up to
    /// $5 \cdot 10^{-4}$ at $\mu = 255$ (the $\sqrt{t}$ of its integrand); the gain differs
    /// from that of the former rule by half these errors.
    fn mu_law_gain(mu: f64, v: f64) -> f64 {
        let a = mu / v;
        let half = (a * 7.0).ln_1p() / 2.0;
        let (x, w) = mu_law_nodes();
        let integral = x
            .iter()
            .zip(w)
            .map(|(x, w)| {
                let s = (x + 1.0) * half;
                let r = s.exp_m1() / a;
                w * 2.0 * r * (-r * r).exp() * s * s * s.exp() / a
            })
            .sum::<f64>()
            * half;
        let k = v / (1.0 + mu).ln();
        1.0 / (k * k * integral).sqrt()
    }

    /// The law with the constants of its parameters.
    fn law(&self) -> Law {
        match *self {
            Companding::MuLaw { mu, peak_db } => {
                let v = amplitude(peak_db);
                Law::MuLaw { mu, v, gain: Companding::mu_law_gain(mu, v) }
            }
            Companding::Exponential { degree } => {
                Law::Exponential { degree, alpha: ((degree + 2.0) / degree).powf(degree / 2.0) }
            }
        }
    }

    /// Compress an envelope. Each call computes the normalisation of the μ-law (64
    /// points); [`PaprReduction::reduce`] computes it once per block.
    pub fn compress(&self, r: f64) -> f64 {
        self.law().compress(r)
    }

    /// Expand an envelope (the inverse of [`Companding::compress`]).
    pub fn expand(&self, r: f64) -> f64 {
        self.law().expand(r)
    }

    fn apply(x: &mut [C64], f: impl Fn(f64) -> f64) {
        map_envelope(x, |r| C64::new(f(r) / r, 0.0));
    }
}

impl PaprReduction for Companding {
    fn reduce(&self, ofdm: &Ofdm, data: &[C64]) -> Result<(Vec<C64>, usize)> {
        if let Companding::MuLaw { mu, .. } = self {
            ensure(*mu > 0.0, || "μ must be positive".into())?;
        }
        if let Companding::Exponential { degree } = self {
            ensure(*degree > 0.0, || "the degree must be positive".into())?;
        }
        let mut x = ofdm.to_time(&ofdm.spread(data))?;
        let law = self.law();
        Companding::apply(&mut x, |r| law.compress(r));
        Ok((x, 0))
    }

    fn restore_time(&self, block: &mut [C64]) {
        let law = self.law();
        Companding::apply(block, |r| law.expand(r));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_math::power;

    fn qam_data(c: &Constellation, n: usize, seed: u64) -> Vec<C64> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| c.points()[r.below(c.order())]).collect()
    }

    /// The PAPR exceeded by 5 % of the blocks (the tail is what the techniques reduce).
    fn papr_95(ofdm: &Ofdm, red: &dyn PaprReduction, data: &[C64]) -> f64 {
        let k = red.data_len(ofdm);
        let blocks = data.len() / k;
        let (x, _) = ofdm.modulate_with(&data[..blocks * k], red).unwrap();
        let mut p: Vec<f64> = x.chunks(ofdm.block_len()).map(papr_db).collect();
        p.sort_by(f64::total_cmp);
        p[(0.95 * p.len() as f64) as usize]
    }

    #[test]
    fn the_ccdf_of_ofdm_follows_ochiai_and_imai() {
        let ofdm = Ofdm::new(256, 256).unwrap();
        let c = Constellation::qam(4).unwrap().normalized();
        let blocks = 4000;
        let x = ofdm.modulate(&qam_data(&c, 256 * blocks, 1)).unwrap();
        let p: Vec<f64> = x.chunks(ofdm.block_len()).map(papr_db).collect();
        for g in [8.0, 9.0, 10.0] {
            let emp = p.iter().filter(|v| **v > g).count() as f64 / p.len() as f64;
            let th = ccdf_oversampled(256, g);
            assert!((emp / th - 1.0).abs() < 0.25, "{g} dB: {emp} vs {th}");
            // the Nyquist-rate formula underestimates the peaks of the continuous signal
            assert!(ccdf_nyquist(256, g) < th);
        }
    }

    #[test]
    fn bussgang_matches_a_monte_carlo() {
        let mut r = Rng::new(2);
        let x: Vec<C64> = (0..400_000).map(|_| r.complex_gaussian()).collect();
        for cr in [0.0, 3.0, 6.0] {
            let mut y = x.clone();
            limit(&mut y, amplitude(cr));
            let alpha: C64 = y.iter().zip(&x).map(|(a, b)| a * b.conj()).sum::<C64>() / power(&x) / x.len() as f64;
            let th = com_dsp::nonlinear::clipping_bussgang(cr);
            assert!((alpha - th.gain).norm() < 3e-3, "{cr} dB: {alpha} vs {}", th.gain);
            assert!((power(&y) - th.output_power).abs() < 3e-3);
            let d: f64 = y.iter().zip(&x).map(|(a, b)| (a - b * th.gain).norm_sqr()).sum::<f64>() / x.len() as f64;
            assert!((d / th.distortion - 1.0).abs() < 0.03);
        }
    }

    #[test]
    fn every_technique_reduces_the_papr_and_lossless_ones_recover_the_data() {
        let ofdm = Ofdm::new(128, 128).unwrap();
        let c = Constellation::qam(16).unwrap().normalized();
        let data = qam_data(&c, 128 * 200, 3);
        let plain = papr_95(&ofdm, &crate::ofdm::NoReduction, &data);
        let techniques: Vec<(&str, Box<dyn PaprReduction>, bool)> = vec![
            ("clip", Box::new(Clip { ratio_db: 4.0 }), false),
            ("icf", Box::new(ClipFilter { ratio_db: 4.0, iterations: 4 }), false),
            ("slm", Box::new(Slm { candidates: 8, seed: 4 }), true),
            ("pts", Box::new(Pts { subblocks: 4, phases: 4, partition: Partition::Pseudorandom(5) }), true),
            ("tr", Box::new(ToneReservation::random(128, 12, 6, 4.0)), true),
            ("ace", Box::new(Ace { constellation: c.clone(), ratio_db: 4.0, iterations: 10, step: 1.0 }), false),
            ("exp", Box::new(Companding::Exponential { degree: 2.0 }), true),
            ("mu", Box::new(Companding::MuLaw { mu: 4.0, peak_db: 9.0 }), true),
        ];
        for (name, t, lossless) in &techniques {
            let k = t.data_len(&ofdm);
            let d = &data[..(data.len() / k) * k];
            let reduced = papr_95(&ofdm, t.as_ref(), d);
            assert!(reduced < plain - 1.0, "{name}: {reduced} vs {plain}");
            let (x, side) = ofdm.modulate_with(d, t.as_ref()).unwrap();
            let back = ofdm.demodulate_with(&x, t.as_ref(), &side).unwrap();
            if *lossless {
                assert!(back.iter().zip(d).all(|(a, b)| (a - b).norm() < 1e-6), "{name}");
            } else {
                // ACE moves outer points outwards only: the decisions are unchanged
                assert!(
                    back.iter().zip(d).filter(|(a, b)| c.nearest(**a) != c.nearest(**b)).count() < d.len() / 50,
                    "{name}"
                );
            }
        }
        // the exponential companding of degree 2 caps the PAPR at 3 dB
        let (x, _) = ofdm.modulate_with(&data, &Companding::Exponential { degree: 2.0 }).unwrap();
        assert!(papr_db(&x) <= com_math::lin2db(2.0) + 0.3);
    }

    /// The μ-law as it was computed before: the gain by Simpson's rule on 6000 intervals
    /// of $t = r^{2}$ for every sample.
    fn mu_law_compress_simpson(mu: f64, peak_db: f64, r: f64) -> f64 {
        let v = amplitude(peak_db);
        let c = |t: f64| v * (1.0 + mu * t.sqrt() / v).ln() / (1.0 + mu).ln();
        let gain = 1.0 / com_math::simpson(|t| c(t).powi(2) * (-t).exp(), 0.0, 60.0, 6000).sqrt();
        gain * v * (1.0 + mu * r / v).ln() / (1.0 + mu).ln()
    }

    #[test]
    fn the_mu_law_gain_is_the_integral_and_follows_the_former_simpson_rule() {
        // E{c(R)²} by mpmath.quad at 30 digits (mpmath 1.3), and the error of the former
        // 6000-point Simpson rule on the gain
        for (mu, peak_db, power, simpson_error) in [
            (4.0, 9.0, 2.089_224_506_453_305, 3.5e-7),
            (255.0, 0.0, 0.912_706_126_294_248_1, 2.6e-4),
            (255.0, 12.0, 8.026_021_096_858_328, 1.1e-4),
            (100.0, 3.0, 1.531_119_965_389_956, 1.2e-4),
            (10.0, 6.0, 1.910_143_981_626_004_6, 3e-6),
        ] {
            // the gain g normalises E{(g c(R))²} to 1
            let gain = Companding::mu_law_gain(mu, amplitude(peak_db));
            let e = 1.0 / (gain * gain);
            assert!((e / power - 1.0).abs() < 1e-13, "μ {mu}, {peak_db} dB: {e} vs {power}");
            let law = Companding::MuLaw { mu, peak_db };
            for r in [0.01, 0.3, 1.0, 2.0, 4.5] {
                let (new, old) = (law.compress(r), mu_law_compress_simpson(mu, peak_db, r));
                assert!((new / old - 1.0).abs() < simpson_error, "μ {mu}, {peak_db} dB, r {r}: {new} vs {old}");
                assert!((law.expand(new) / r - 1.0).abs() < 1e-12);
            }
        }
        // the unit average power of a Rayleigh envelope, by Monte Carlo
        let law = Companding::MuLaw { mu: 255.0, peak_db: 0.0 }.law();
        let mut rng = Rng::new(11);
        let n = 400_000;
        let p = (0..n).map(|_| law.compress(rng.complex_gaussian().norm()).powi(2)).sum::<f64>() / f64::from(n);
        // the power of the compressed envelope has a standard deviation below 1 / sqrt(n)
        assert!((p - 1.0).abs() < 5e-3, "{p}");
    }

    #[test]
    fn ace_never_moves_a_point_across_a_decision_boundary() {
        let ofdm = Ofdm::new(64, 64).unwrap();
        let c = Constellation::qam(16).unwrap().normalized();
        let data = qam_data(&c, 64 * 50, 7);
        let ace = Ace { constellation: c.clone(), ratio_db: 3.0, iterations: 20, step: 1.0 };
        let (x, side) = ofdm.modulate_with(&data, &ace).unwrap();
        let back = ofdm.demodulate_with(&x, &ace, &side).unwrap();
        assert_eq!(c.decide(&back), c.decide(&data));
    }

    #[test]
    fn dft_spread_ofdm_has_a_lower_papr() {
        let c = Constellation::qam(4).unwrap().normalized();
        let data = qam_data(&c, 64 * 300, 8);
        let ofdm = Ofdm::new(256, 64).unwrap();
        let sc = Ofdm { dft_spread: true, ..ofdm.clone() };
        let p0 = papr_95(&ofdm, &crate::ofdm::NoReduction, &data);
        let p1 = papr_95(&sc, &crate::ofdm::NoReduction, &data);
        assert!(p1 < p0 - 2.0, "{p1} vs {p0}");
    }
}

/// Checks against the published closed forms (`references/`).
#[cfg(test)]
mod literature {
    use super::*;

    /// H. Ochiai and H. Imai, "On the distribution of the peak-to-average power ratio in
    /// OFDM signals," IEEE Trans. Commun. 49(2), 2001: eq. (11) (p. 284), the Nyquist-rate
    /// distribution `1 - (1 - e^{-λ})^N`, and eq. (27) (p. 287), the asymptotic crest
    /// factor distribution `exp(-√(π/3) N r e^{-r²})` with `r² = λ`; the text (p. 287)
    /// notes that even for `N = 100 000` the PAPR exceeds 15 dB with a probability of
    /// about `1e-8`.
    #[test]
    fn ochiai_2001_eqs_11_and_27() {
        let lambda = db2lin(10.0);
        // a relative tolerance: powi rounds differently when the compiler folds it
        // (found under the instrumentation of code coverage)
        let nyquist = 1.0 - (1.0 - (-lambda).exp()).powi(256);
        assert!((ccdf_nyquist(256, 10.0) / nyquist - 1.0).abs() < 1e-12);
        let r = lambda.sqrt();
        let crest = (-(PI / 3.0).sqrt() * 256.0 * r * (-r * r).exp()).exp();
        assert!((ccdf_oversampled(256, 10.0) / (1.0 - crest) - 1.0).abs() < 1e-12);
        let p15 = ccdf_oversampled(100_000, 15.0);
        assert!(p15 > 3e-9 && p15 < 3e-8, "{p15:e}");
    }
}

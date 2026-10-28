//! Adaptive equalisation (OptiCommPy `optic.dsp.equalization`).
//!
//! Polarisation demultiplexing is the `2×2` case: there is no optical-specific
//! equaliser.
//!
//! # Adaptive MIMO equaliser
//!
//! [`AdaptiveEqualizer`] is the `n×n` butterfly FIR equaliser of OptiCommPy's
//! `mimoAdaptEqualizer` (time domain, one update per output symbol). The input of `n`
//! modes sampled at `sps` samples per symbol is padded with `⌊L/2⌋` zeros at both ends;
//! output symbol `k` of mode `m` is
//!
//! `y_m[k] = Σ_N Σ_t h_{m,N}[t] x_N[k·sps + t]`,
//!
//! with the centre taps of the diagonal filters initialised at 1. After every output the
//! taps follow `h_{m,N} ← h_{m,N} + μ e_m x_N*` (a stochastic gradient) with the error
//! of the current [`Algorithm`]:
//!
//! - `Cma` (Godard): `e = (R - |y|²) y`, `R = E|s|⁴/E|s|²`;
//! - `Rde` (radius-directed): `e = (R̂² - |y|²) y` with `R̂` the constellation radius
//!   nearest to `|y|`;
//! - `Nlms` (data-aided): `e = (s - y)/‖x‖²` with the reference symbols `s`;
//! - `DdLms` (decision-directed): `e = ŝ - y` with `ŝ` the nearest constellation point.
//!
//! A [`Schedule`] runs several algorithms in sequence (for instance CMA for
//! pre-convergence, then DD-LMS), over several passes of the block if wanted.
//!
//! # The singularity of blind 2 × 2 equalisation
//!
//! Blind criteria such as the CMA are satisfied by any output of constant modulus: both
//! outputs of a polarisation demultiplexer can converge to the same polarisation (the
//! singularity of Kikuchi's analysis), most often when the input polarisations are
//! mixed near 45°. [`Singularity::Kikuchi`] is his two-stage initialisation: during the
//! first `n` output symbols only output 0 adapts; then the filters of output 1 are set
//! to the unitary complement of those of output 0, as the inverse of a unitary Jones
//! matrix requires,
//!
//! $$
//! h_{yx}\[t\] = -h_{xy}^{*}\[L - 1 - t\], \qquad h_{yy}\[t\] = h_{xx}^{*}\[L - 1 - t\],
//! $$
//!
//! (the conjugate of the frequency response, the taps reversed about the centre), so
//! that output 1 starts on the other polarisation; all filters adapt from then on.
//!
//! # Examples
//! ```
//! use com_dsp::equalizer::{AdaptiveEqualizer, Algorithm};
//! use com_core::rng::Rng;
//! use com_math::C64;
//! // QPSK through a 2-tap channel at 1 sample per symbol, equalised blindly by CMA
//! let qpsk: Vec<C64> = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
//!     .iter().map(|&(a, b)| C64::new(a, b) / 2f64.sqrt()).collect();
//! let mut rng = Rng::new(1);
//! let s: Vec<C64> = (0..4000).map(|_| qpsk[rng.below(4)]).collect();
//! let x: Vec<C64> = (0..s.len()).map(|n| s[n] + if n > 0 { s[n - 1] * 0.3 } else { C64::new(0.0, 0.0) }).collect();
//! let eq = AdaptiveEqualizer { n_taps: 11, sps: 1, mu: 1e-3, ..AdaptiveEqualizer::new(Algorithm::Cma) };
//! let out = eq.run(&[x], None, &qpsk).unwrap();
//! let tail = &out.symbols[0][3000..];
//! let spread = tail.iter().map(|v| (v.norm_sqr() - 1.0).abs()).sum::<f64>() / tail.len() as f64;
//! assert!(spread < 0.05);                     // constant modulus restored
//! ```

use com_core::{Error, Result, ensure, ensure_shape};
use com_math::C64;
use com_math::nearest::Slicer;

/// An update rule of the adaptive equaliser.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Constant modulus algorithm (blind).
    Cma,
    /// Radius-directed equalisation (blind, multi-ring constellations).
    Rde,
    /// Normalised LMS on reference symbols (data-aided).
    Nlms,
    /// Decision-directed LMS.
    DdLms,
}

impl std::str::FromStr for Algorithm {
    type Err = Error;

    /// `cma`, `rde`, `nlms` or `dd-lms` (`dd_lms`, `ddlms`, `dd`), case-insensitive.
    fn from_str(s: &str) -> Result<Algorithm> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "cma" => Ok(Algorithm::Cma),
            "rde" => Ok(Algorithm::Rde),
            "nlms" => Ok(Algorithm::Nlms),
            "ddlms" | "dd" => Ok(Algorithm::DdLms),
            other => Err(Error::Config(format!("unknown algorithm {other:?} (cma, rde, nlms, dd-lms)"))),
        }
    }
}

/// The protection of a blind 2 × 2 equaliser against the singularity (see the module).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Singularity {
    /// No protection: the filters start from the identity.
    #[default]
    None,
    /// Kikuchi's two-stage initialisation: output 0 adapts alone during this number of
    /// output symbols, then output 1 starts as its unitary complement.
    Kikuchi(usize),
}

/// A sequence of `(algorithm, number of output symbols)`; the last entry runs to the end
/// of the block, and every pass of the block restarts the sequence.
pub type Schedule = Vec<(Algorithm, usize)>;

/// The `n×n` butterfly adaptive equaliser (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveEqualizer {
    /// Number of taps of every filter.
    pub n_taps: usize,
    /// Input samples per symbol (the output is at one sample per symbol).
    pub sps: usize,
    /// Step size `μ`.
    pub mu: f64,
    /// Step sizes of the stages of the schedule (the last one holds for the following
    /// stages, as OptiCommPy's list of `mu`); empty: `mu` for every stage.
    pub stage_mu: Vec<f64>,
    /// The algorithms and their durations.
    pub schedule: Schedule,
    /// Number of passes over the block (the taps are kept from one pass to the next).
    pub passes: usize,
    /// The protection against the singularity (2 × 2 only).
    pub singularity: Singularity,
}

impl AdaptiveEqualizer {
    /// 15 taps, 2 samples per symbol, `μ = 10⁻³`, one pass of `alg`.
    pub fn new(alg: Algorithm) -> AdaptiveEqualizer {
        AdaptiveEqualizer {
            n_taps: 15,
            sps: 2,
            mu: 1e-3,
            stage_mu: Vec::new(),
            schedule: vec![(alg, usize::MAX)],
            passes: 1,
            singularity: Singularity::None,
        }
    }
}

/// The output of [`AdaptiveEqualizer::run`].
#[derive(Clone, Debug, PartialEq)]
pub struct EqualizerOutput {
    /// Equalised symbols of every mode (last pass).
    pub symbols: Vec<Vec<C64>>,
    /// Final taps, `taps[m + N·n]` the filter from input mode `N` to output mode `m`.
    pub taps: Vec<Vec<C64>>,
    /// Squared error `|e|²` of every mode and output symbol (last pass).
    pub error: Vec<Vec<f64>>,
}

impl AdaptiveEqualizer {
    /// Equalise the modes `x` (equal lengths). `reference` gives the transmitted symbols
    /// of every mode (needed by `Nlms`); `points` are the constellation points
    /// (normalised to unit average energy inside), for the radii and the decisions.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid configuration or a missing reference,
    /// [`Error::Shape`] for modes of different lengths, a block shorter than the
    /// equaliser or a reference too short.
    pub fn run(&self, x: &[Vec<C64>], reference: Option<&[Vec<C64>]>, points: &[C64]) -> Result<EqualizerOutput> {
        let n = x.len();
        ensure(n >= 1, || "at least one mode".into())?;
        ensure(self.n_taps >= 1 && self.sps >= 1, || "n_taps and sps must be positive".into())?;
        ensure(!self.schedule.is_empty(), || "empty schedule".into())?;
        let len = x[0].len();
        ensure_shape(x.iter().all(|m| m.len() == len), || {
            format!("the modes have different lengths: {:?}", x.iter().map(Vec::len).collect::<Vec<_>>())
        })?;
        let pad = self.n_taps / 2;
        let padded: Vec<Vec<C64>> = x
            .iter()
            .map(|m| {
                let mut p = vec![C64::new(0.0, 0.0); pad];
                p.extend_from_slice(m);
                p.extend(std::iter::repeat_n(C64::new(0.0, 0.0), pad));
                p
            })
            .collect();
        let plen = padded[0].len();
        ensure_shape(plen >= self.n_taps, || {
            format!("a block of {len} samples is shorter than the {} taps of the equaliser", self.n_taps)
        })?;
        let n_out = (plen - self.n_taps) / self.sps + 1;
        if self.schedule.iter().any(|(a, _)| *a == Algorithm::Nlms) {
            let r = reference.ok_or_else(|| Error::Config("NLMS needs reference symbols".into()))?;
            ensure_shape(r.len() == n && r.iter().all(|v| v.len() >= n_out), || {
                format!(
                    "the reference needs {n} modes of at least {n_out} symbols, got {} modes of {:?} symbols",
                    r.len(),
                    r.iter().map(Vec::len).collect::<Vec<_>>()
                )
            })?;
        }
        ensure(!points.is_empty(), || "empty constellation".into())?;
        let scale = 1.0 / (points.iter().map(C64::norm_sqr).sum::<f64>() / points.len() as f64).sqrt();
        let pts: Vec<C64> = points.iter().map(|p| p * scale).collect();
        let slicer = Slicer::new(&pts);
        let e2: f64 = pts.iter().map(C64::norm_sqr).sum::<f64>() / pts.len() as f64;
        let e4: f64 = pts.iter().map(|p| p.norm_sqr().powi(2)).sum::<f64>() / pts.len() as f64;
        let r_cma = e4 / e2;
        let mut radii: Vec<f64> = pts.iter().map(|p| p.norm()).collect();
        radii.sort_by(f64::total_cmp);
        radii.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        let mut h = vec![vec![C64::new(0.0, 0.0); self.n_taps]; n * n];
        for m in 0..n {
            h[m + m * n][self.n_taps / 2] = C64::new(1.0, 0.0);
        }
        let mut out = vec![vec![C64::new(0.0, 0.0); n_out]; n];
        let mut err = vec![vec![0.0; n_out]; n];
        let kikuchi = match self.singularity {
            Singularity::Kikuchi(k) => {
                ensure(n == 2, || "the Kikuchi initialisation is for 2 × 2 equalisers".into())?;
                Some(k)
            }
            Singularity::None => None,
        };
        for pass in 0..self.passes.max(1) {
            let mut stage = 0usize;
            let mut left = self.schedule[0].1;
            for k in 0..n_out {
                while left == 0 && stage + 1 < self.schedule.len() {
                    stage += 1;
                    left = self.schedule[stage].1;
                }
                left = left.saturating_sub(1);
                let alg = self.schedule[stage].0;
                let mu = self.stage_mu.get(stage).or(self.stage_mu.last()).copied().unwrap_or(self.mu);
                // the second stage of Kikuchi's initialisation: output 1 as the unitary
                // complement of output 0
                let alone = pass == 0 && kikuchi.is_some_and(|k0| k < k0);
                if pass == 0 && kikuchi == Some(k) {
                    let l = self.n_taps;
                    for t in 0..l {
                        h[1][t] = -h[2][l - 1 - t].conj();
                        h[3][t] = h[0][l - 1 - t].conj();
                    }
                }
                let base = k * self.sps;
                let mut y = vec![C64::new(0.0, 0.0); n];
                for (mo, ym) in y.iter_mut().enumerate() {
                    for (mi, xin) in padded.iter().enumerate() {
                        let taps = &h[mo + mi * n];
                        for t in 0..self.n_taps {
                            *ym += taps[t] * xin[base + t];
                        }
                    }
                }
                let norm2: Vec<f64> =
                    padded.iter().map(|xin| xin[base..base + self.n_taps].iter().map(C64::norm_sqr).sum()).collect();
                for mo in 0..n {
                    let ym = y[mo];
                    let (e, e_abs2) = match alg {
                        Algorithm::Cma => {
                            let d = r_cma - ym.norm_sqr();
                            (ym * d, d * d)
                        }
                        Algorithm::Rde => {
                            let r = ym.norm();
                            let rr = radii
                                .iter()
                                .copied()
                                .min_by(|a, b| (a - r).abs().total_cmp(&(b - r).abs()))
                                .unwrap_or(1.0);
                            let d = rr * rr - ym.norm_sqr();
                            (ym * d, d * d)
                        }
                        Algorithm::Nlms => {
                            let Some(r) = reference else { unreachable!("checked before the passes") };
                            let s = r[mo][k];
                            let d = s - ym;
                            (d, d.norm_sqr())
                        }
                        Algorithm::DdLms => {
                            let d = pts[slicer.nearest(ym)] - ym;
                            (d, d.norm_sqr())
                        }
                    };
                    out[mo][k] = ym;
                    err[mo][k] = e_abs2;
                    if alone && mo > 0 {
                        continue;
                    }
                    for (mi, xin) in padded.iter().enumerate() {
                        let scale = if alg == Algorithm::Nlms { mu / norm2[mi].max(1e-300) } else { mu };
                        let taps = &mut h[mo + mi * n];
                        for t in 0..self.n_taps {
                            taps[t] += e * xin[base + t].conj() * scale;
                        }
                    }
                }
            }
        }
        Ok(EqualizerOutput { symbols: out, taps: h, error: err })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn nlms_converges_to_the_inverse_and_dd_lms_tracks() {
        // 16-QAM, unit energy
        let c: Vec<C64> =
            (0..16).map(|i| C64::new(f64::from(2 * (i % 4) - 3), f64::from(2 * (i / 4) - 3)) / 10f64.sqrt()).collect();
        let mut rng = Rng::new(3);
        let s: Vec<C64> = (0..6000).map(|_| c[rng.below(16)]).collect();
        let x: Vec<C64> = (0..s.len())
            .map(|n| {
                s[n] * C64::new(0.9, 0.1) + if n > 0 { s[n - 1] * C64::new(0.2, -0.1) } else { C64::new(0.0, 0.0) }
            })
            .collect();
        let eq = AdaptiveEqualizer {
            n_taps: 9,
            sps: 1,
            mu: 2e-2,
            stage_mu: Vec::new(),
            schedule: vec![(Algorithm::Nlms, 2000), (Algorithm::DdLms, usize::MAX)],
            passes: 1,
            singularity: Singularity::None,
        };
        let out = eq.run(&[x], Some(std::slice::from_ref(&s)), &c).unwrap();
        let tail = 5000..6000;
        let mse: f64 = tail.clone().map(|k| (out.symbols[0][k] - s[k]).norm_sqr()).sum::<f64>() / 1000.0;
        assert!(mse < 1e-3, "mse {mse}");
    }

    #[test]
    fn kikuchi_initialisation_avoids_the_singularity_of_the_45_degree_rotation() {
        let qpsk: Vec<C64> = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
            .iter()
            .map(|&(a, b)| C64::new(a, b) / 2f64.sqrt())
            .collect();
        let (c, sn) = (std::f64::consts::FRAC_PI_4.cos(), std::f64::consts::FRAC_PI_4.sin());
        let corr = |kikuchi: Singularity, seed: u64| {
            let mut rng = Rng::new(seed);
            let n = 6000;
            let a: Vec<C64> = (0..n).map(|_| qpsk[rng.below(4)]).collect();
            let b: Vec<C64> = (0..n).map(|_| qpsk[rng.below(4)]).collect();
            let x: Vec<C64> = a.iter().zip(&b).map(|(u, v)| u * c - v * sn + rng.complex_gaussian() * 0.05).collect();
            let y: Vec<C64> = a.iter().zip(&b).map(|(u, v)| u * sn + v * c + rng.complex_gaussian() * 0.05).collect();
            let eq = AdaptiveEqualizer {
                n_taps: 5,
                sps: 1,
                mu: 2e-3,
                singularity: kikuchi,
                ..AdaptiveEqualizer::new(Algorithm::Cma)
            };
            let out = eq.run(&[x, y], None, &qpsk).unwrap();
            let (u, v) = (&out.symbols[0][4000..], &out.symbols[1][4000..]);
            let cc: C64 = u.iter().zip(v).map(|(p, q)| p * q.conj()).sum();
            let e = |w: &[C64]| w.iter().map(C64::norm_sqr).sum::<f64>();
            cc.norm() / (e(u) * e(v)).sqrt()
        };
        // from the identity, many realisations end with both outputs on one
        // polarisation (correlation above 0.5); none with Kikuchi's initialisation
        let singular = |init: Singularity| (0..20).filter(|&s| corr(init, s) > 0.5).count();
        assert!(singular(Singularity::None) >= 5);
        assert_eq!(singular(Singularity::Kikuchi(2000)), 0);
    }
}

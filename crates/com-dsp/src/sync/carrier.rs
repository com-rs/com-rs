//! Carrier recovery (OptiCommPy `optic.dsp.carrierRecovery`).
//!
//! # Frequency offset
//!
//! [`fourth_power_foe`] removes the `M`-fold symmetry of the constellation by raising
//! the signal to the power `M` (4 for QAM), takes the frequency of the largest bin of
//! its spectrum and divides it by `M` (OptiCommPy `fourthPowerFOE`); the estimate is
//! limited to the bin spacing `f_s/(M N)`. [`fourth_power_foe_refined`] refines it by a
//! golden-section search of the same periodogram between the neighbouring bins.
//!
//! # Phase
//!
//! - [`viterbi_viterbi`]: `θ̂[n] = -unwrap(arg(Σ_window y^M)/M, 2π/M) - π/4`
//!   (OptiCommPy `viterbi`, a centred moving average of `N` samples);
//! - [`bps`]: blind phase search (Pfau, Hoffmann and Noé, JLT 27(8), 2009): every test
//!   phase `φ_b = b (π/2)/B`, `b = 0..B-1`, rotates the samples, the squared distances
//!   to the nearest constellation points are summed over a window of `2N + 1` samples,
//!   and the phase of the smallest sum is kept (OptiCommPy `bps`, values in `[0, π/2)`,
//!   to be unwrapped with [`unwrap`] and a period of `π/2`);
//! - [`ddpll`]: the decision-directed phase-locked loop of OptiCommPy (`ddpll`):
//!   detector `Im(y ŝ*)` on the derotated sample, a proportional-integral loop filter
//!   with coefficients from `τ₁`, `τ₂` and the bilinear transform, a VCO of gain `K_v`;
//!   known pilots replace the decisions where given.
//!
//! Every phase estimator returns the correction `θ[n]` to apply with
//! `derotate(x, θ, +1.0)`, that is `x[n] e^{jθ[n]}`; the blind estimators leave an
//! ambiguity of a multiple of `2π/M` (`π/2` for QAM), resolved by pilots or differential
//! coding.
//!
//! # Examples
//! ```
//! use com_dsp::sync::carrier::*;
//! use com_core::rng::Rng;
//! use com_math::C64;
//! // 16-QAM of unit energy
//! let c: Vec<C64> = (0..16).map(|i| C64::new((2 * (i % 4) - 3) as f64, (2 * (i / 4) - 3) as f64) / 10f64.sqrt()).collect();
//! let mut rng = Rng::new(5);
//! let s: Vec<C64> = (0..4096).map(|_| c[rng.below(16)]).collect();
//! let fs = 32e9;
//! let y: Vec<C64> = s.iter().enumerate()
//!     .map(|(n, v)| v * com_math::cis(2.0 * std::f64::consts::PI * 150e6 * n as f64 / fs + 0.3))
//!     .collect();
//! let (z, fo) = fourth_power_foe_refined(&y, fs, 4)?;
//! assert!((fo - 150e6).abs() < 1e6);
//! let theta = unwrap(&bps(&z, 20, &c, 32)?, std::f64::consts::FRAC_PI_2);
//! let out = derotate(&z, &theta, 1.0)?;
//! // what is left is a multiple of π/2, resolved by pilots in a real receiver
//! let err = (out[2000] / s[2000]).arg().rem_euclid(std::f64::consts::FRAC_PI_2);
//! assert!(err < 0.05 || err > std::f64::consts::FRAC_PI_2 - 0.05);
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::filter::moving_average;
use crate::signal::freq_shift;
use com_core::{Result, ensure, same_len};
use com_math::fft;
use com_math::{C64, cis, golden_max};
use std::f64::consts::PI;

/// Multiply every sample by `e^{j sign θ[n]}`.
///
/// # Errors
/// [`com_core::Error::Shape`] unless there is one phase per sample.
pub fn derotate(x: &[C64], theta: &[f64], sign: f64) -> Result<Vec<C64>> {
    same_len(x.len(), theta.len(), "derotate: samples and phases")?;
    Ok(x.iter().zip(theta).map(|(v, t)| v * cis(sign * t)).collect())
}

/// Unwrap a phase sequence with the given period (`numpy.unwrap(period=…)`): jumps
/// larger than half a period are folded by multiples of the period.
pub fn unwrap(phi: &[f64], period: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(phi.len());
    let mut offset = 0.0;
    for (i, &p) in phi.iter().enumerate() {
        if i > 0 {
            let d = p - phi[i - 1];
            let dd = (d + period / 2.0).rem_euclid(period) - period / 2.0;
            // numpy keeps +period/2 for an exact jump of +period/2
            let dd = if dd == -period / 2.0 && d > 0.0 { period / 2.0 } else { dd };
            offset += dd - d;
        }
        out.push(p + offset);
    }
    out
}

/// Frequency offset by the `M`-th power spectrum (largest bin), and the corrected signal.
///
/// # Errors
/// [`com_core::Error::Config`] for an empty signal, $M = 0$ or a sampling rate that is
/// not positive and finite.
pub fn fourth_power_foe(x: &[C64], fs: f64, m: u32) -> Result<(Vec<C64>, f64)> {
    ensure(!x.is_empty(), || "fourth_power_foe: the signal is empty".into())?;
    ensure(m >= 1, || "fourth_power_foe: the symmetry M must be at least 1".into())?;
    ensure(fs > 0.0 && fs.is_finite(), || format!("fourth_power_foe: the sampling rate must be positive, got {fs}"))?;
    let n = x.len();
    let xm: Vec<C64> = x.iter().map(|v| v.powu(m)).collect();
    let spec = fft::fft(&xm);
    let k = spec.iter().enumerate().max_by(|a, b| a.1.norm().total_cmp(&b.1.norm())).map_or(0, |(k, _)| k);
    let fo = fft::fftfreq(n, 1.0 / fs)[k] / m as f64;
    Ok((freq_shift(x, -fo, fs), fo))
}

/// [`fourth_power_foe`] refined by a golden-section search of the periodogram of `x^M`
/// between the neighbouring bins (accuracy far below the bin spacing).
///
/// # Errors
/// As [`fourth_power_foe`].
pub fn fourth_power_foe_refined(x: &[C64], fs: f64, m: u32) -> Result<(Vec<C64>, f64)> {
    let (_, coarse) = fourth_power_foe(x, fs, m)?;
    let n = x.len();
    let xm: Vec<C64> = x.iter().map(|v| v.powu(m)).collect();
    let periodogram = |f: f64| -> f64 {
        let step = cis(-2.0 * PI * f * m as f64 / fs);
        let mut rot = C64::new(1.0, 0.0);
        let mut acc = C64::new(0.0, 0.0);
        for &v in &xm {
            acc += v * rot;
            rot *= step;
        }
        acc.norm_sqr()
    };
    let df = fs / (n as f64 * m as f64);
    let fo = golden_max(periodogram, coarse - df, coarse + df, 60);
    Ok((freq_shift(x, -fo, fs), fo))
}

/// Viterbi–Viterbi phase estimates (radians, to be applied with `derotate(…, +1)`).
pub fn viterbi_viterbi(x: &[C64], n: usize, m: u32) -> Vec<f64> {
    let xm: Vec<C64> = x.iter().map(|v| v.powu(m)).collect();
    let avg = moving_average(&xm, n);
    let ang: Vec<f64> = avg.iter().map(|v| v.arg() / m as f64).collect();
    unwrap(&ang, 2.0 * PI / m as f64).into_iter().map(|p| -p - PI / 4.0).collect()
}

/// Blind phase search with `b` test phases in `[0, π/2)` and a window of `2n + 1`
/// samples; returns the selected test phase of every sample (to be applied with
/// `derotate(…, +1)` after unwrapping with a period of `π/2`).
///
/// # Errors
/// [`com_core::Error::Config`] without a constellation point or a test phase.
pub fn bps(x: &[C64], n: usize, points: &[C64], b: usize) -> Result<Vec<f64>> {
    ensure(!points.is_empty() && b >= 1, || "bps needs constellation points and at least one test phase".into())?;
    let len = x.len();
    let w = 2 * n + 1;
    let phases: Vec<f64> = (0..b).map(|i| i as f64 * (PI / 2.0) / b as f64).collect();
    let rot: Vec<C64> = phases.iter().map(|&p| cis(p)).collect();
    let mut padded = vec![C64::new(0.0, 0.0); n];
    padded.extend_from_slice(x);
    padded.extend(std::iter::repeat_n(C64::new(0.0, 0.0), n));
    let slicer = com_math::nearest::Slicer::new(points);
    let mut dmin = vec![0.0; w * b];
    let mut sums = vec![0.0; b];
    let mut out = vec![0.0; len];
    for (k, &v) in padded.iter().enumerate() {
        let slot = k % w;
        for p in 0..b {
            let z = v * rot[p];
            let d = slicer.distance2(z);
            sums[p] += d - dmin[slot * b + p];
            dmin[slot * b + p] = d;
        }
        if k >= 2 * n {
            let best = (0..b).min_by(|&i, &j| sums[i].total_cmp(&sums[j])).unwrap_or(0);
            out[k - 2 * n] = phases[best];
        }
    }
    Ok(out)
}

/// Decision-directed PLL (OptiCommPy `ddpll`); returns the phase correction `θ[n]` to
/// apply with `derotate(…, +1)`.
///
/// `ts` is the symbol period, `kv` the VCO gain, `tau1`, `tau2` the time constants of
/// the loop filter; `pilots` gives the indices and the known symbols used instead of the
/// decisions.
pub fn ddpll(
    x: &[C64],
    ts: f64,
    kv: f64,
    tau1: f64,
    tau2: f64,
    points: &[C64],
    pilots: Option<(&[usize], &[C64])>,
) -> Vec<f64> {
    let n = x.len();
    let b1 = ts / (2.0 * tau1) * (1.0 - 1.0 / (ts / (2.0 * tau2)).tan());
    let b2 = ts / (2.0 * tau1) * (1.0 + 1.0 / (ts / (2.0 * tau2)).tan());
    let mut known: Vec<Option<C64>> = vec![None; n];
    if let Some((idx, sym)) = pilots {
        for (&i, &s) in idx.iter().zip(sym) {
            if i < n {
                known[i] = Some(s);
            }
        }
    }
    let slicer = com_math::nearest::Slicer::new(points);
    let mut phase = vec![0.0; n];
    let (mut ud, mut uf) = (0.0, 0.0);
    for k in 0..n {
        let ud1 = ud;
        let y = x[k] * cis(phase[k]);
        let decided = match known[k] {
            Some(s) => s,
            None if points.is_empty() => C64::new(0.0, 0.0),
            None => points[slicer.nearest(y)],
        };
        ud = (y * decided.conj()).im;
        uf += b1 * ud1 + b2 * ud;
        if k + 1 < n {
            phase[k + 1] = phase[k] - kv * uf;
        }
    }
    phase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwrap_matches_numpy() {
        let p = vec![0.0, 3.0, -3.0, -0.5];
        let u = unwrap(&p, 2.0 * PI);
        assert!((u[2] - (-3.0 + 2.0 * PI)).abs() < 1e-12);
        assert!((u[3] - (-0.5 + 2.0 * PI)).abs() < 1e-12);
    }

    #[test]
    fn viterbi_and_ddpll_track_a_static_phase() {
        let qpsk: Vec<C64> = (0..4).map(|k| cis(PI / 4.0 + k as f64 * PI / 2.0)).collect();
        let mut r = com_core::rng::Rng::new(9);
        let s: Vec<C64> = (0..2000).map(|_| qpsk[r.below(4)]).collect();
        let y: Vec<C64> = s.iter().map(|v| v * cis(0.2)).collect();
        // y⁴ = -e^{j0.8}: arg/4 = (0.8 - π)/4, so θ = -(0.8 - π)/4 - π/4 = -0.2
        let th = viterbi_viterbi(&y, 31, 4);
        assert!((th[1000] + 0.2).abs() < 1e-9, "{}", th[1000]);
        assert!((derotate(&y, &th, 1.0).unwrap()[1000] - s[1000]).norm() < 1e-9);
        let ph = ddpll(&y, 1.0, 0.3, 1.0 / 0.1, 1.0 / 0.05, &qpsk, None);
        let z = derotate(&y, &ph, 1.0).unwrap();
        assert!((z[1500] / s[1500]).arg().abs() < 1e-3);
    }

    #[test]
    fn derotate_rejects_a_phase_per_sample_mismatch() {
        let x = [C64::new(1.0, 0.0); 3];
        assert!(matches!(derotate(&x, &[0.0; 2], 1.0), Err(com_core::Error::Shape(_))));
        assert!(matches!(derotate(&x, &[0.0; 4], 1.0), Err(com_core::Error::Shape(_))));
    }

    #[test]
    fn fourth_power_foe_rejects_an_empty_signal_instead_of_panicking() {
        for refined in [false, true] {
            let foe = if refined { fourth_power_foe_refined } else { fourth_power_foe };
            assert!(matches!(foe(&[], 1.0, 4), Err(com_core::Error::Config(_))));
            assert!(foe(&[C64::new(1.0, 0.0); 8], 1.0, 0).is_err());
            assert!(foe(&[C64::new(1.0, 0.0); 8], 0.0, 4).is_err());
            assert!(foe(&[C64::new(1.0, 0.0); 8], f64::NAN, 4).is_err());
            // a single sample is a valid, if uninformative, block
            assert!(foe(&[C64::new(1.0, 0.0)], 1.0, 4).unwrap().1.is_finite());
        }
    }
}

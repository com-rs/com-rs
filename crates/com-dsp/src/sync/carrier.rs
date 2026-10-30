//! Carrier recovery (OptiCommPy `optic.dsp.carrierRecovery`).
//!
//! # Frequency offset
//!
//! [`fourth_power_foe`] removes the `M`-fold symmetry of the constellation by raising
//! the signal to the power `M` (4 for QAM), takes the frequency of the largest bin of
//! its spectrum and divides it by `M` (OptiCommPy `fourthPowerFOE`); the estimate is
//! limited to the bin spacing `f_s/(M N)`. [`fourth_power_foe_refined`] refines it by a
//! golden-section search of the same periodogram between the neighbouring bins. Both
//! only estimate: the caller removes the offset with [`crate::signal::freq_shift`]
//! (`freq_shift(x, -fo, fs)`).
//!
//! # Phase
//!
//! - [`viterbi_viterbi`]: `θ̂[n] = -unwrap(arg(Σ_window y^M)/M, 2π/M) - π/M`
//!   (OptiCommPy `viterbi`, a centred moving average of `N` samples, written there for
//!   QPSK and square QAM with `π/4`): the reference is a constellation whose `M`-th
//!   power is negative, $s^{M} < 0$ (QPSK and square QAM, $M = 4$; the $M$-PSK
//!   $e^{j\pi(2i+1)/M}$; BPSK $\pm j$ with $M = 2$), so that the estimate removes the
//!   rotation up to a multiple of `2π/M`; for a constellation whose `M`-th power is
//!   positive (BPSK $\pm 1$, the $M$-PSK $e^{j 2\pi i/M}$) add `π/M`;
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
//! let fo = fourth_power_foe_refined(&y, fs, 4)?;
//! assert!((fo - 150e6).abs() < 1e6);
//! let z = com_dsp::signal::freq_shift(&y, -fo, fs);
//! let theta = unwrap(&bps(&z, 20, &c, 32)?, std::f64::consts::FRAC_PI_2);
//! let out = derotate(&z, &theta, 1.0)?;
//! // what is left is a multiple of π/2, resolved by pilots in a real receiver
//! let err = (out[2000] / s[2000]).arg().rem_euclid(std::f64::consts::FRAC_PI_2);
//! assert!(err < 0.05 || err > std::f64::consts::FRAC_PI_2 - 0.05);
//! # Ok::<(), com_core::Error>(())
//! ```

use crate::filter::moving_average;
use com_core::{Result, ensure, ensure_shape, same_len};
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

/// The frequency offset (Hz) by the `M`-th power spectrum (its largest bin); remove it
/// with `freq_shift(x, -fo, fs)` ([`crate::signal::freq_shift`]).
///
/// # Errors
/// [`com_core::Error::Shape`] for an empty signal, [`com_core::Error::Config`] for
/// $M = 0$ or a sampling rate that is not positive and finite.
pub fn fourth_power_foe(x: &[C64], fs: f64, m: u32) -> Result<f64> {
    ensure_shape(!x.is_empty(), || "fourth_power_foe: the signal is empty".into())?;
    ensure(m >= 1, || "fourth_power_foe: the symmetry M must be at least 1".into())?;
    ensure(fs > 0.0 && fs.is_finite(), || format!("fourth_power_foe: the sampling rate must be positive, got {fs}"))?;
    let n = x.len();
    let xm: Vec<C64> = x.iter().map(|v| v.powu(m)).collect();
    let spec = fft::fft(&xm);
    let k = spec.iter().enumerate().max_by(|a, b| a.1.norm().total_cmp(&b.1.norm())).map_or(0, |(k, _)| k);
    Ok(fft::fftfreq(n, 1.0 / fs)[k] / m as f64)
}

/// [`fourth_power_foe`] refined by a golden-section search of the periodogram of `x^M`
/// between the neighbouring bins (accuracy far below the bin spacing): the frequency
/// offset (Hz).
///
/// # Errors
/// As [`fourth_power_foe`].
pub fn fourth_power_foe_refined(x: &[C64], fs: f64, m: u32) -> Result<f64> {
    let coarse = fourth_power_foe(x, fs, m)?;
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
    Ok(golden_max(periodogram, coarse - df, coarse + df, 60))
}

/// The Cramér–Rao bound of the variance of an unbiased estimate of the frequency of a
/// complex tone of unknown phase from $N$ samples at the SNR $\rho = b_{0}^{2}/(2\sigma^{2})$
/// per sample (`snr_lin`, linear; $\sigma^{2}$ the noise variance of each of the real and
/// imaginary parts), in cycles per sample squared: Rife & Boorstyn (1974), eq. (17),
/// $12\sigma^{2}/(b_{0}^{2} N (N^{2} - 1))$ rad² per sample², that is
/// $3/(2\pi^{2} \rho N (N^{2} - 1))$ (multiply by $f_{s}^{2}$ for Hz²).
///
/// [`fourth_power_foe_refined`] with $M = 1$ is their maximum-likelihood estimator (the
/// largest bin of the DFT, then the maximum of $|A(\omega)|$ next to it), which meets the
/// bound above the threshold.
pub fn tone_frequency_crb(n: usize, snr_lin: f64) -> f64 {
    let nf = n as f64;
    3.0 / (2.0 * PI * PI * snr_lin * nf * (nf * nf - 1.0))
}

/// The probability that the largest bin of the $N$-point DFT of a complex tone in noise
/// is not the bin of the tone (an outlier of the coarse search), at the SNR $\rho$ per
/// sample (`snr`, linear): Rife & Boorstyn (1974), eqs. (57)–(59), the magnitude of the
/// bin of the tone Rician and the $N - 1$ others Rayleigh,
///
/// $$
/// 1 - q = \int_{0}^{\infty} \left[1 - e^{-N x^{2}/2\sigma^{2}}\right]^{N-1}
/// \frac{N x}{\sigma^{2}} e^{-N (x^{2} + b_{0}^{2})/2\sigma^{2}}
/// I_{0}\!\left(\frac{N b_{0} x}{\sigma^{2}}\right) dx,
/// $$
///
/// by Simpson's rule ([`com_math::simpson`], with [`com_math::bessel_i0e`] against the
/// overflow of $I_{0}$), accurate to about $10^{-13}$ absolute.
pub fn tone_outlier_probability(n: usize, snr: f64) -> f64 {
    // b0 = 1, sigma^2 = 1/(2 rho): N/sigma^2 = 2 rho N
    let nf = n as f64;
    let a = 2.0 * snr * nf;
    let s = 1.0 / a.sqrt();
    let hit = com_math::simpson(
        |x| {
            // 1 - e^{-y}, without cancellation for a small y
            let others = (-(-(0.5 * a * x * x)).exp_m1()).powi(n as i32 - 1);
            others * a * x * (-(0.5 * a) * (x - 1.0) * (x - 1.0)).exp() * com_math::bessel_i0e(a * x)
        },
        0.0,
        1.0 + 12.0 * s,
        4000,
    );
    (1.0 - hit).clamp(0.0, 1.0)
}

/// The RMS error (Hz) of the maximum-likelihood frequency of a tone at half the sampling
/// rate `fs`, from $N$ samples at the SNR $\rho$ (`snr`, linear), with its outliers:
/// Rife & Boorstyn (1974), eq. (50), $q f_{s}^{2}/12$ (an outlier is uniform over the
/// band) plus $(1 - q)$ times the bound of [`tone_frequency_crb`], $q$ by
/// [`tone_outlier_probability`].
pub fn tone_frequency_rms(n: usize, snr: f64, fs: f64) -> f64 {
    let q = tone_outlier_probability(n, snr);
    (fs * fs * (q / 12.0 + (1.0 - q) * tone_frequency_crb(n, snr))).sqrt()
}

/// Viterbi–Viterbi phase estimates (radians, to be applied with `derotate(…, +1)`) over
/// a centred window of `n` samples, for a constellation of symmetry `m` whose `m`-th
/// power is negative (see the module: `-π/M` is the reference phase; OptiCommPy's `π/4`
/// for $M = 4$).
///
/// # Errors
/// [`com_core::Error::Config`] for $M = 0$.
pub fn viterbi_viterbi(x: &[C64], n: usize, m: u32) -> Result<Vec<f64>> {
    ensure(m >= 1, || "viterbi_viterbi: the symmetry M must be at least 1".into())?;
    let xm: Vec<C64> = x.iter().map(|v| v.powu(m)).collect();
    let avg = moving_average(&xm, n);
    let mf = m as f64;
    let ang: Vec<f64> = avg.iter().map(|v| v.arg() / mf).collect();
    Ok(unwrap(&ang, 2.0 * PI / mf).into_iter().map(|p| -p - PI / mf).collect())
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
///
/// # Errors
/// [`com_core::Error::Shape`] for pilots with another number of indices than of
/// symbols, or an index beyond the signal (they were dropped silently before).
pub fn ddpll(
    x: &[C64],
    ts: f64,
    kv: f64,
    tau1: f64,
    tau2: f64,
    points: &[C64],
    pilots: Option<(&[usize], &[C64])>,
) -> Result<Vec<f64>> {
    let n = x.len();
    let b1 = ts / (2.0 * tau1) * (1.0 - 1.0 / (ts / (2.0 * tau2)).tan());
    let b2 = ts / (2.0 * tau1) * (1.0 + 1.0 / (ts / (2.0 * tau2)).tan());
    let mut known: Vec<Option<C64>> = vec![None; n];
    if let Some((idx, sym)) = pilots {
        same_len(idx.len(), sym.len(), "ddpll: pilot indices and symbols")?;
        for (&i, &s) in idx.iter().zip(sym) {
            ensure_shape(i < n, || format!("ddpll: the pilot index {i} is beyond the {n} samples"))?;
            known[i] = Some(s);
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
    Ok(phase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tone_bounds_have_their_limits() {
        use com_math::db2lin;
        // eq. (46): N = 16, 10 dB, f_s = 4000 Hz: 7.72 Hz
        assert!((4000.0 * tone_frequency_crb(16, 10.0).sqrt() - 7.7201).abs() < 1e-3);
        // no signal: every bin is equally likely; a strong one: never an outlier
        let q = tone_outlier_probability(16, 1e-6);
        assert!((q - 15.0 / 16.0).abs() < 1e-3, "{q}");
        assert!(tone_outlier_probability(64, db2lin(10.0)) < 1e-12);
        let rms = tone_frequency_rms(64, db2lin(10.0), 4000.0);
        // above the threshold, the bound (q is accurate to about 1e-13 absolute)
        assert!((rms / (4000.0 * tone_frequency_crb(64, db2lin(10.0)).sqrt()) - 1.0).abs() < 1e-6);
    }

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
        let th = viterbi_viterbi(&y, 31, 4).unwrap();
        assert!((th[1000] + 0.2).abs() < 1e-9, "{}", th[1000]);
        assert!((derotate(&y, &th, 1.0).unwrap()[1000] - s[1000]).norm() < 1e-9);
        let ph = ddpll(&y, 1.0, 0.3, 1.0 / 0.1, 1.0 / 0.05, &qpsk, None).unwrap();
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
            assert!(matches!(foe(&[], 1.0, 4), Err(com_core::Error::Shape(_))));
            assert!(foe(&[C64::new(1.0, 0.0); 8], 1.0, 0).is_err());
            assert!(foe(&[C64::new(1.0, 0.0); 8], 0.0, 4).is_err());
            assert!(foe(&[C64::new(1.0, 0.0); 8], f64::NAN, 4).is_err());
            // a single sample is a valid, if uninformative, block
            assert!(foe(&[C64::new(1.0, 0.0)], 1.0, 4).unwrap().is_finite());
        }
    }

    #[test]
    fn viterbi_viterbi_takes_the_reference_phase_pi_over_m() {
        // BPSK on ±j (its square is -1) rotated by 0.2: M = 2 removes the rotation
        let mut r = com_core::rng::Rng::new(3);
        let y: Vec<C64> = (0..500).map(|_| C64::new(0.0, if r.bit() == 1 { 1.0 } else { -1.0 }) * cis(0.2)).collect();
        let th = viterbi_viterbi(&y, 21, 2).unwrap();
        // the residual phase is a multiple of π
        let res = (th[250] + 0.2).rem_euclid(PI);
        assert!(res < 1e-9 || PI - res < 1e-9, "{}", th[250]);
        assert!(matches!(viterbi_viterbi(&y, 21, 0), Err(com_core::Error::Config(_))));
    }

    #[test]
    fn ddpll_rejects_pilots_it_would_have_dropped() {
        let qpsk: Vec<C64> = (0..4).map(|k| cis(PI / 4.0 + k as f64 * PI / 2.0)).collect();
        let x = vec![qpsk[0]; 8];
        let run = |idx: &[usize], sym: &[C64]| ddpll(&x, 1.0, 0.3, 10.0, 20.0, &qpsk, Some((idx, sym)));
        assert!(matches!(run(&[0, 2], &qpsk[..1]), Err(com_core::Error::Shape(_))));
        assert!(matches!(run(&[0, 8], &qpsk[..2]), Err(com_core::Error::Shape(_))));
        assert!(run(&[0, 7], &qpsk[..2]).is_ok());
    }
}

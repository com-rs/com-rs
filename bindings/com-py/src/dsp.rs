//! Digital signal processing: filters, resampling, dispersion compensation, adaptive
//! equalisation, carrier and clock recovery.

use crate::comm::PyConstellation;
use crate::fiber::PyFiber;
use crate::{CArr, CMat, FArr, FMat, IntoPy, carr, cmat, cvec, farr, fvec, mat, uvec};
use comrs::dsp::compensation::cd;
use comrs::dsp::equalizer::{AdaptiveEqualizer, Algorithm, Singularity};
use comrs::dsp::filter::{LowPass, PulseShape};
use comrs::dsp::sync::timing::GardnerLoop;
use comrs::dsp::sync::{carrier, timing as clock};
use comrs::dsp::{filter, resample, signal};
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Taps of a transmit pulse at `sps` samples per symbol, normalised to a unit sum
/// (OptiCommPy `pulseShape`).
///
/// The defaults are those of the ``pulse_shape`` block of the specifications (1025
/// taps, roll-off 0.1). OptiCommPy's defaults are reached by explicit arguments:
/// ``pulse_shape("rrc", sps, n_taps=4096, rolloff=0.01)`` gives its taps (an even
/// length, see below).
///
/// Args:
///     kind (str): `"rect"`, `"nrz"`, `"rc"` (raised cosine) or `"rrc"` (root raised
///         cosine).
///     sps (int): samples per symbol.
///     n_taps (int): length of the raised-cosine pulses, on a symmetric grid of
///         `n_taps` points over `±⌊n_taps/2⌋` samples (ignored by `rect` and `nrz`).
///         An even length gives OptiCommPy's taps, which fall between the sampling
///         instants; an odd length puts them on the sampling instants (OptiCommPy's
///         odd grid is shifted by about half a sample).
///     rolloff (float): roll-off of the raised-cosine pulses.
///
/// Returns:
///     numpy.ndarray: the taps (float64), summing to one.
///
/// Examples:
///     >>> import comrs
///     >>> h = comrs.pulse_shape("rrc", 4)
///     >>> h.size, round(float(h.sum()), 12)
///     (1025, 1.0)
#[pyfunction]
#[pyo3(signature = (kind, sps, *, n_taps = 1025, rolloff = 0.1))]
fn pulse_shape<'py>(py: Python<'py>, kind: &str, sps: usize, n_taps: usize, rolloff: f64) -> PyResult<FArr<'py>> {
    let shape = PulseShape::parse(kind, rolloff).py()?;
    if sps == 0 || n_taps == 0 {
        return Err(PyValueError::new_err("sps and n_taps must be positive"));
    }
    Ok(farr(py, filter::pulse_shape(shape, sps, n_taps)))
}

/// Root of the raised-cosine spectrum on the `n_symbols·sps` bins of a periodic block
/// (1 in the passband): shaping and matched filtering with it are exactly Nyquist.
///
/// Args:
///     n_symbols (int): symbols of the block.
///     sps (int): samples per symbol.
///     rolloff (float): roll-off factor.
///
/// Returns:
///     numpy.ndarray: `n_symbols·sps` gains in FFT order (float64).
#[pyfunction]
fn rrc_spectrum(py: Python<'_>, n_symbols: usize, sps: usize, rolloff: f64) -> FArr<'_> {
    farr(py, filter::rrc_spectrum(n_symbols, sps, rolloff))
}

/// FIR filtering with the `same` alignment (OptiCommPy `firFilter`); the taps may be
/// real or complex.
///
/// Args:
///     h (array_like): the taps.
///     x (array_like): the signal.
///
/// Returns:
///     numpy.ndarray: the filtered signal, of the length of `x` (complex128).
#[pyfunction]
fn fir_filter<'py>(py: Python<'py>, h: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    let (h, x) = (cvec(h)?, cvec(x)?);
    Ok(carr(py, py.detach(|| filter::fir_filter_complex(&h, &x))))
}

/// Full linear convolution (direct for short filters, by FFT otherwise).
///
/// Args:
///     h (array_like): the taps.
///     x (array_like): the signal.
///
/// Returns:
///     numpy.ndarray: `len(h) + len(x) - 1` samples (complex128).
#[pyfunction]
fn convolve<'py>(py: Python<'py>, h: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    let (h, x) = (cvec(h)?, cvec(x)?);
    Ok(carr(py, py.detach(|| filter::convolve_full(&h, &x))))
}

/// Insert `factor - 1` zeros after every sample.
///
/// Args:
///     x (array_like): the signal.
///     factor (int): the upsampling factor.
///
/// Returns:
///     numpy.ndarray: `factor·len(x)` samples (complex128).
#[pyfunction]
fn upsample<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, factor: usize) -> PyResult<CArr<'py>> {
    Ok(carr(py, resample::upsample(&cvec(x)?, factor)))
}

/// Keep the samples `offset, offset + factor, …`.
///
/// Args:
///     x (array_like): the signal.
///     factor (int): the downsampling factor.
///     offset (int): index of the first sample kept.
///
/// Returns:
///     numpy.ndarray: the kept samples (complex128).
#[pyfunction]
#[pyo3(signature = (x, factor, *, offset = 0))]
fn downsample<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, factor: usize, offset: usize) -> PyResult<CArr<'py>> {
    Ok(carr(py, resample::downsample(&cvec(x)?, factor, offset)))
}

/// Decimate from `sps_in` to `sps_out` samples per symbol at the phase of maximum
/// variance (OptiCommPy `decimate`).
///
/// Args:
///     x (array_like): the signal at `sps_in` samples per symbol.
///     sps_in (int): input samples per symbol.
///     sps_out (int): output samples per symbol (a divisor of `sps_in`).
///
/// Returns:
///     numpy.ndarray: the signal at `sps_out` samples per symbol (complex128).
#[pyfunction]
fn decimate<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, sps_in: usize, sps_out: usize) -> PyResult<CArr<'py>> {
    Ok(carr(py, resample::decimate(&cvec(x)?, sps_in, sps_out).py()?))
}

/// Band-limited resampling of a periodic block to `n_out` samples (FFT).
///
/// Args:
///     x (array_like): the block.
///     n_out (int): output length.
///
/// Returns:
///     numpy.ndarray: `n_out` samples (complex128).
#[pyfunction]
fn resample_fft<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, n_out: usize) -> PyResult<CArr<'py>> {
    Ok(carr(py, resample::resample_fft(&cvec(x)?, n_out)))
}

/// Low-pass FIR of cut-off `fc` at the sampling rate `fs`, unit DC gain (OptiCommPy
/// `lowPassFIR`).
///
/// Args:
///     fc (float): cut-off frequency (Hz).
///     fs (float): sampling rate (Hz).
///     n_taps (int): number of taps.
///     kind (str): `"rect"` (truncated sinc) or `"gauss"` (Gaussian, 3 dB at `fc`).
///
/// Returns:
///     numpy.ndarray: the taps (float64).
#[pyfunction]
#[pyo3(signature = (fc, fs, n_taps, *, kind = "rect"))]
fn low_pass_fir<'py>(py: Python<'py>, fc: f64, fs: f64, n_taps: usize, kind: &str) -> PyResult<FArr<'py>> {
    let k: LowPass = kind.parse().py()?;
    Ok(farr(py, filter::low_pass_fir(fc, fs, n_taps, k)))
}

/// Scale to unit mean power.
///
/// Args:
///     x (array_like): the signal.
///
/// Returns:
///     numpy.ndarray: the scaled signal (complex128).
#[pyfunction]
fn pnorm<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    Ok(carr(py, signal::pnorm(&cvec(x)?)))
}

/// Scale to unit peak amplitude.
///
/// Args:
///     x (array_like): the signal.
///
/// Returns:
///     numpy.ndarray: the scaled signal (complex128).
#[pyfunction]
fn anorm<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    Ok(carr(py, signal::anorm(&cvec(x)?)))
}

/// Frequency shift `x[n] e^{j2πΔf n/fs}`.
///
/// Args:
///     x (array_like): the signal.
///     delta_f (float): the shift `Δf` (Hz).
///     fs (float): sampling rate (Hz).
///
/// Returns:
///     numpy.ndarray: the shifted signal (complex128).
#[pyfunction]
fn freq_shift<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, delta_f: f64, fs: f64) -> PyResult<CArr<'py>> {
    Ok(carr(py, signal::freq_shift(&cvec(x)?, delta_f, fs)))
}

/// Centred moving average over `n` samples with zeros outside (OptiCommPy
/// `movingAverage`).
///
/// Args:
///     x (array_like): the signal.
///     n (int): the window length (samples).
///
/// Returns:
///     numpy.ndarray: the averaged signal, of the length of `x` (complex128).
#[pyfunction]
fn moving_average<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, n: usize) -> PyResult<CArr<'py>> {
    Ok(carr(py, filter::moving_average(&cvec(x)?, n)))
}

/// Delay a periodic block by `delay` seconds (FFT phase ramp).
///
/// Args:
///     x (array_like): the block.
///     delay (float): the delay (s), fractional delays included.
///     fs (float): sampling rate (Hz).
///
/// Returns:
///     numpy.ndarray: the delayed block (complex128).
#[pyfunction]
fn delay_fft<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, delay: f64, fs: f64) -> PyResult<CArr<'py>> {
    Ok(carr(py, resample::delay_fft(&cvec(x)?, delay, fs)))
}

/// Compensate the dispersion of `length_km` of `fiber` on a block sampled at `fs`
/// (frequency domain, exact on a periodic block).
///
/// Args:
///     x (array_like): the field.
///     fiber (FiberParams): the fibre (its dispersion).
///     length_km (float): the length to compensate (km).
///     fs (float): sampling rate (Hz).
///
/// Returns:
///     numpy.ndarray: the compensated field (complex128).
#[pyfunction]
fn cdc<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fiber: &PyFiber, length_km: f64, fs: f64) -> PyResult<CArr<'py>> {
    let (x, f) = (cvec(x)?, fiber.to_rust());
    Ok(carr(py, py.detach(|| cd::cdc(&x, f.beta2_s2_km(), length_km, fs))))
}

/// Taps of the dispersion-compensating FIR (OptiCommPy `edc` rule,
/// `2⌈6.67 |β₂| L R_s² (f_s/R_s)⌉`).
///
/// Args:
///     fiber (FiberParams): the fibre.
///     length_km (float): the length to compensate (km).
///     symbol_rate (float): symbol rate (Bd).
///     fs (float): sampling rate (Hz).
///
/// Returns:
///     int: the number of taps.
#[pyfunction]
fn cd_taps_needed(fiber: &PyFiber, length_km: f64, symbol_rate: f64, fs: f64) -> usize {
    cd::cd_taps_needed(fiber.to_rust().beta2_s2_km(), length_km, symbol_rate, fs)
}

/// Centred taps of the dispersion compensation of `length_km` (frequency sampling).
///
/// Args:
///     fiber (FiberParams): the fibre.
///     length_km (float): the length to compensate (km).
///     fs (float): sampling rate (Hz).
///     n_taps (int): number of taps (see `cd_taps_needed`).
///
/// Returns:
///     numpy.ndarray: the taps (complex128).
#[pyfunction]
fn cd_fir_taps<'py>(py: Python<'py>, fiber: &PyFiber, length_km: f64, fs: f64, n_taps: usize) -> CArr<'py> {
    carr(py, cd::cd_fir_taps(fiber.to_rust().beta2_s2_km(), length_km, fs, n_taps))
}

/// The dispersion factor `K = -β₂ L fs²/2` of the digital compensation `e^{jKω²}`
/// (`ω` in radians per sample; Savory 2008, Eghbali et al. 2014).
///
/// Args:
///     fiber (FiberParams): the fibre.
///     length_km (float): the length to compensate (km).
///     fs (float): sampling rate (Hz).
///
/// Returns:
///     float: K.
#[pyfunction]
fn cd_dispersion_factor(fiber: &PyFiber, length_km: f64, fs: f64) -> f64 {
    cd::cd_dispersion_factor(fiber.to_rust().beta2_s2_km(), length_km, fs)
}

/// Centred taps of Savory's dispersion-compensating FIR, `√(j/(4Kπ)) e^{-jn²/(4K)}`
/// (S. J. Savory, *Opt. Express* 16(2), 2008).
///
/// Args:
///     fiber (FiberParams): the fibre.
///     length_km (float): the length to compensate (km).
///     fs (float): sampling rate (Hz).
///     n_taps (int | None): odd number of taps; None: Savory's rule `2⌊2π|K|⌋ + 1`.
///
/// Returns:
///     numpy.ndarray: the taps (complex128).
#[pyfunction]
#[pyo3(signature = (fiber, length_km, fs, n_taps = None))]
fn cd_savory_taps<'py>(
    py: Python<'py>,
    fiber: &PyFiber,
    length_km: f64,
    fs: f64,
    n_taps: Option<usize>,
) -> PyResult<CArr<'py>> {
    let b2 = fiber.to_rust().beta2_s2_km();
    let n = n_taps.unwrap_or_else(|| cd::cd_savory_length(b2, length_km, fs));
    Ok(carr(py, cd::cd_savory_taps(b2, length_km, fs, n).py()?))
}

/// Centred taps of the least-squares dispersion-compensating FIR over the band
/// `|f| <= band_hz` (Eghbali, Johansson, Gustafsson & Savory, *J. Lightw. Technol.*
/// 32(8), 2014).
///
/// Over the whole band the taps are closed-form; over the band of the signal
/// (`fs (1 + rolloff) / (2 sps)`) fewer taps reach the same error, at the price of a
/// large response outside the band, which the receive filter must reject.
///
/// Args:
///     fiber (FiberParams): the fibre.
///     length_km (float): the length to compensate (km).
///     fs (float): sampling rate (Hz).
///     n_taps (int): odd number of taps.
///     band_hz (float | None): half the band of the fit (Hz); None: `fs / 2`.
///     epsilon (float): the penalty on the energy of the taps (the paper uses 1e-14
///         for a band-limited fit).
///
/// Returns:
///     numpy.ndarray: the taps (complex128).
#[pyfunction]
#[pyo3(signature = (fiber, length_km, fs, n_taps, *, band_hz = None, epsilon = 0.0))]
fn cd_ls_taps<'py>(
    py: Python<'py>,
    fiber: &PyFiber,
    length_km: f64,
    fs: f64,
    n_taps: usize,
    band_hz: Option<f64>,
    epsilon: f64,
) -> PyResult<CArr<'py>> {
    let b2 = fiber.to_rust().beta2_s2_km();
    Ok(carr(py, cd::cd_ls_taps(b2, length_km, fs, n_taps, band_hz.unwrap_or(fs / 2.0), epsilon).py()?))
}

/// The energy of the error `E = (1/2π) ∫ |H(e^{jω}) - e^{jKω²}|² dω` of centred
/// dispersion-compensating taps over the band `|f| <= band_hz` (Eghbali et al. 2014,
/// eq. 6).
///
/// Args:
///     taps (array_like): the centred taps (odd in number).
///     fiber (FiberParams): the fibre.
///     length_km (float): the length to compensate (km).
///     fs (float): sampling rate (Hz).
///     band_hz (float | None): half the band (Hz); None: `fs / 2`.
///
/// Returns:
///     float: E.
#[pyfunction]
#[pyo3(signature = (taps, fiber, length_km, fs, *, band_hz = None))]
fn cd_ls_error(
    taps: &Bound<'_, PyAny>,
    fiber: &PyFiber,
    length_km: f64,
    fs: f64,
    band_hz: Option<f64>,
) -> PyResult<f64> {
    let b2 = fiber.to_rust().beta2_s2_km();
    cd::cd_ls_error(&cvec(taps)?, b2, length_km, fs, band_hz.unwrap_or(fs / 2.0)).py()
}

/// The `n×n` butterfly adaptive equaliser (OptiCommPy `mimoAdaptEqualizer`).
///
/// Output symbol `k` of mode `m` is `Σ_N Σ_t h_{m,N}[t] x_N[k·sps + t]` (input padded
/// with `⌊n_taps/2⌋` zeros at both ends, diagonal centre taps initialised at 1); the
/// taps follow `h ← h + μ e x*` with the error of the current algorithm: CMA
/// `(R - |y|²) y`, RDE `(R̂² - |y|²) y`, NLMS `(s - y)/‖x‖²` (needs `reference`), DD-LMS
/// `ŝ - y`.
///
/// Args:
///     x (array_like): input of shape `(modes, samples)` (a 1-D array is one mode).
///     constellation (Constellation): the alphabet (normalised inside).
///     algorithm (str | list[tuple[str, int]] | None): one algorithm (`"cma"` by
///         default), or a schedule of `(algorithm, output symbols)` whose last entry
///         runs to the end of every pass.
///     n_taps (int): taps per filter.
///     sps (int): input samples per symbol (the output is at 1 sample per symbol).
///     mu (float | list[float] | None): step size, or one per stage of the schedule
///         (the last one holds for the following stages); 1e-3 by default.
///     passes (int): passes over the block (the taps carry over).
///     reference (array_like, optional): transmitted symbols `(modes, symbols)`.
///     kikuchi (int, optional): Kikuchi's two-stage initialisation of a 2 × 2
///         equaliser against the singularity: output 0 adapts alone during this number
///         of symbols, then output 1 starts as its unitary complement.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: the symbols `(modes,
///     symbols)`, the final taps `(modes², n_taps)` with row `m + N·modes` the filter
///     from input `N` to output `m`, and the squared errors `(modes, symbols)`.
#[pyfunction]
#[pyo3(signature = (x, constellation, *, algorithm = None, n_taps = 15, sps = 2, mu = None, passes = 1, reference = None, kikuchi = None))]
#[allow(clippy::too_many_arguments)]
fn adaptive_equalizer<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    algorithm: Option<&Bound<'py, PyAny>>,
    n_taps: usize,
    sps: usize,
    mu: Option<&Bound<'py, PyAny>>,
    passes: usize,
    reference: Option<&Bound<'py, PyAny>>,
    kikuchi: Option<usize>,
) -> PyResult<(CMat<'py>, CMat<'py>, FMat<'py>)> {
    let modes =
        if x.getattr("ndim").and_then(|n| n.extract::<usize>()).unwrap_or(2) == 1 { vec![cvec(x)?] } else { cmat(x)? };
    let schedule = match algorithm {
        None => vec![(Algorithm::Cma, usize::MAX)],
        Some(a) if a.is_instance_of::<PyString>() => vec![(a.extract::<String>()?.parse().py()?, usize::MAX)],
        Some(a) => {
            let steps: Vec<(String, usize)> = a.extract()?;
            steps.into_iter().map(|(s, n)| Ok((s.parse::<Algorithm>().py()?, n))).collect::<PyResult<_>>()?
        }
    };
    let reference = match reference {
        None => None,
        Some(r) if r.getattr("ndim").and_then(|n| n.extract::<usize>()).unwrap_or(2) == 1 => Some(vec![cvec(r)?]),
        Some(r) => Some(cmat(r)?),
    };
    let singularity = kikuchi.map_or(Singularity::None, Singularity::Kikuchi);
    let stage_mu: Vec<f64> = match mu {
        None => vec![1e-3],
        Some(m) => m.extract::<Vec<f64>>().or_else(|_| m.extract::<f64>().map(|v| vec![v]))?,
    };
    if stage_mu.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("mu needs at least one step size"));
    }
    let eq = AdaptiveEqualizer { n_taps, sps, mu: stage_mu[0], stage_mu, schedule, passes, singularity };
    let c = &constellation.inner;
    let out = py.detach(|| eq.run(&modes, reference.as_deref(), c.points())).py()?;
    Ok((mat(py, out.symbols)?, mat(py, out.taps)?, mat(py, out.error)?))
}

/// Multiply every sample by `e^{j·sign·θ[n]}`.
///
/// Args:
///     x (array_like): the signal.
///     theta (array_like): one phase per sample (rad).
///     sign (float): `+1` to apply the corrections of the carrier recoveries, `-1` to
///         undo them.
///
/// Returns:
///     numpy.ndarray: the rotated signal (complex128).
#[pyfunction]
#[pyo3(signature = (x, theta, *, sign = 1.0))]
fn derotate<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, theta: &Bound<'py, PyAny>, sign: f64) -> PyResult<CArr<'py>> {
    let (x, t) = (cvec(x)?, fvec(theta)?);
    if x.len() != t.len() {
        return Err(PyValueError::new_err(format!("{} samples but {} phases", x.len(), t.len())));
    }
    Ok(carr(py, carrier::derotate(&x, &t, sign)))
}

/// Unwrap a phase sequence with a given period (`numpy.unwrap(period=…)`).
///
/// Args:
///     phi (array_like): the phases (rad).
///     period (float): the period of the wraps (rad).
///
/// Returns:
///     numpy.ndarray: the unwrapped phases (float64).
#[pyfunction]
#[pyo3(signature = (phi, *, period = std::f64::consts::TAU))]
fn unwrap<'py>(py: Python<'py>, phi: &Bound<'py, PyAny>, period: f64) -> PyResult<FArr<'py>> {
    Ok(farr(py, carrier::unwrap(&fvec(phi)?, period)))
}

/// Frequency offset of an `m`-fold symmetric signal by its `m`-th power spectrum
/// (OptiCommPy `fourthPowerFOE` with `refined=False`).
///
/// Args:
///     x (array_like): signal at one sample per symbol.
///     fs (float): its sampling rate (Hz).
///     m (int): the symmetry (4 for QAM).
///     refined (bool): refine the periodogram peak by a golden-section search between
///         the neighbouring bins (the default, as the ``foe`` block); otherwise the
///         estimate is on the FFT grid, as OptiCommPy's.
///
/// Returns:
///     tuple[numpy.ndarray, float]: the corrected signal and the offset (Hz).
#[pyfunction]
#[pyo3(signature = (x, fs, *, m = 4, refined = true))]
fn fourth_power_foe<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    fs: f64,
    m: u32,
    refined: bool,
) -> PyResult<(CArr<'py>, f64)> {
    let x = cvec(x)?;
    let (y, fo) = py.detach(|| {
        if refined { carrier::fourth_power_foe_refined(&x, fs, m) } else { carrier::fourth_power_foe(&x, fs, m) }
    });
    Ok((carr(py, y), fo))
}

/// Viterbi–Viterbi phase estimates (OptiCommPy `viterbi`), to apply with
/// `derotate(x, θ)`.
///
/// Args:
///     x (array_like): signal at one sample per symbol.
///     n (int): half-length of the averaging window (symbols).
///     m (int): the symmetry (4 for QAM).
///
/// Returns:
///     numpy.ndarray: one phase per symbol (rad).
#[pyfunction]
#[pyo3(signature = (x, *, n = 35, m = 4))]
fn viterbi_viterbi<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, n: usize, m: u32) -> PyResult<FArr<'py>> {
    Ok(farr(py, carrier::viterbi_viterbi(&cvec(x)?, n, m)))
}

/// Blind phase search (Pfau, Hoffmann and Noé 2009; OptiCommPy `bps`).
///
/// Every test phase `b (π/2)/B` rotates the samples; the squared distances to the
/// nearest points are summed over `2n + 1` samples and the phase of the smallest sum is
/// kept. The constellation is normalised inside, so the signal should have unit mean
/// power. Unwrap the result with a period of `π/2`, then apply it with
/// `derotate(x, θ)`.
///
/// Args:
///     x (array_like): signal at one sample per symbol, unit mean power.
///     constellation (Constellation): the alphabet.
///     n (int): half-length of the window (symbols).
///     b (int): number of test phases `B`.
///
/// Returns:
///     numpy.ndarray: one phase per symbol (rad, in `[0, π/2)`).
#[pyfunction]
#[pyo3(signature = (x, constellation, *, n = 35, b = 64))]
fn bps<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    n: usize,
    b: usize,
) -> PyResult<FArr<'py>> {
    let x = cvec(x)?;
    let pts = constellation.inner.normalized().points().to_vec();
    Ok(farr(py, py.detach(|| carrier::bps(&x, n, &pts, b)).py()?))
}

/// Decision-directed phase-locked loop (OptiCommPy `ddpll`): the phase correction to
/// apply with `derotate(x, θ)`.
///
/// Args:
///     x (array_like): signal at one sample per symbol.
///     constellation (Constellation): the alphabet (normalised inside).
///     fs (float): sampling rate, the symbol rate (Hz); OptiCommPy's `Ts` is `1/fs`.
///     kv (float): VCO gain.
///     tau1 (float): first time constant of the loop filter (s).
///     tau2 (float): second time constant of the loop filter (s).
///     pilot_indices (array_like, optional): positions of known symbols.
///     pilot_symbols (array_like, optional): the known symbols, used instead of the
///         decisions at those positions.
///
/// Returns:
///     numpy.ndarray: one phase per symbol (rad).
#[pyfunction]
#[pyo3(signature = (x, constellation, *, fs = 32e9, kv = 0.1, tau1 = 1.0 / (2.0 * std::f64::consts::PI * 10e6), tau2 = 1.0 / (2.0 * std::f64::consts::PI * 10e6), pilot_indices = None, pilot_symbols = None))]
#[allow(clippy::too_many_arguments)]
fn ddpll<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    fs: f64,
    kv: f64,
    tau1: f64,
    tau2: f64,
    pilot_indices: Option<&Bound<'py, PyAny>>,
    pilot_symbols: Option<&Bound<'py, PyAny>>,
) -> PyResult<FArr<'py>> {
    let x = cvec(x)?;
    let pts = constellation.inner.normalized().points().to_vec();
    let pilots = match (pilot_indices, pilot_symbols) {
        (Some(i), Some(s)) => {
            let (i, s) = (uvec(i)?, cvec(s)?);
            if i.len() != s.len() || i.iter().any(|&k| k >= x.len()) {
                return Err(PyValueError::new_err("pilot indices and symbols must match and lie in the signal"));
            }
            Some((i, s))
        }
        (None, None) => None,
        _ => return Err(PyValueError::new_err("give both pilot_indices and pilot_symbols, or neither")),
    };
    if fs <= 0.0 {
        return Err(PyValueError::new_err(format!("the sampling rate must be positive, got {fs}")));
    }
    let theta =
        carrier::ddpll(&x, 1.0 / fs, kv, tau1, tau2, &pts, pilots.as_ref().map(|(i, s)| (i.as_slice(), s.as_slice())));
    Ok(farr(py, theta))
}

fn array<const N: usize>(x: &Bound<'_, PyAny>) -> PyResult<[C64; N]> {
    cvec(x)?.try_into().map_err(|v: Vec<C64>| PyValueError::new_err(format!("expected {N} samples, got {}", v.len())))
}

/// Gardner timing error `Re{x[1]* (x[2] - x[0])}` of three samples at 2 sps.
///
/// Args:
///     x (array_like): three consecutive samples.
///
/// Returns:
///     float: the timing error.
#[pyfunction]
fn gardner_ted(x: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(clock::gardner_ted(&array::<3>(x)?))
}

/// Gardner timing error for Nyquist pulses, `|x[1]|² (|x[0]|² - |x[2]|²)`.
///
/// Args:
///     x (array_like): three consecutive samples.
///
/// Returns:
///     float: the timing error.
#[pyfunction]
fn gardner_ted_nyquist(x: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(clock::gardner_ted_nyquist(&array::<3>(x)?))
}

/// Cubic Lagrange (Farrow) interpolation of four samples at `μ ∈ [-1, 1]`
/// (`μ = 0` returns `x[2]`; OptiCommPy `interpolator`).
///
/// Args:
///     x (array_like): four consecutive samples.
///     mu (float): the fractional position.
///
/// Returns:
///     complex: the interpolated sample.
#[pyfunction]
fn interpolate_cubic(x: &Bound<'_, PyAny>, mu: f64) -> PyResult<C64> {
    Ok(clock::interpolate_cubic(&array::<4>(x)?, mu))
}

/// Gardner clock recovery at 2 samples per symbol (OptiCommPy `gardnerClockRecovery`).
///
/// Args:
///     x (array_like): the signal at (about) 2 samples per symbol.
///     kp (float): proportional gain of the loop filter.
///     ki (float): integral gain of the loop filter.
///     nyquist (bool): use the detector for Nyquist pulses.
///     max_ppm (float): largest clock deviation handled (ppm, sizes the output).
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the resampled signal (2 sps) and the
///     fractional sampling instant of every output sample.
#[pyfunction]
#[pyo3(signature = (x, *, kp = 1e-3, ki = 1e-6, nyquist = true, max_ppm = 500.0))]
fn gardner_clock_recovery<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    kp: f64,
    ki: f64,
    nyquist: bool,
    max_ppm: f64,
) -> PyResult<(CArr<'py>, FArr<'py>)> {
    let x = cvec(x)?;
    let out = py.detach(|| GardnerLoop { kp, ki, nyquist, max_ppm }.run(&x));
    Ok((carr(py, out.signal), farr(py, out.timing)))
}

/// Register the DSP functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(pulse_shape, m)?,
        wrap_pyfunction!(rrc_spectrum, m)?,
        wrap_pyfunction!(fir_filter, m)?,
        wrap_pyfunction!(convolve, m)?,
        wrap_pyfunction!(upsample, m)?,
        wrap_pyfunction!(downsample, m)?,
        wrap_pyfunction!(decimate, m)?,
        wrap_pyfunction!(resample_fft, m)?,
        wrap_pyfunction!(low_pass_fir, m)?,
        wrap_pyfunction!(pnorm, m)?,
        wrap_pyfunction!(anorm, m)?,
        wrap_pyfunction!(freq_shift, m)?,
        wrap_pyfunction!(moving_average, m)?,
        wrap_pyfunction!(delay_fft, m)?,
        wrap_pyfunction!(cdc, m)?,
        wrap_pyfunction!(cd_taps_needed, m)?,
        wrap_pyfunction!(cd_fir_taps, m)?,
        wrap_pyfunction!(cd_dispersion_factor, m)?,
        wrap_pyfunction!(cd_savory_taps, m)?,
        wrap_pyfunction!(cd_ls_taps, m)?,
        wrap_pyfunction!(cd_ls_error, m)?,
        wrap_pyfunction!(adaptive_equalizer, m)?,
        wrap_pyfunction!(derotate, m)?,
        wrap_pyfunction!(unwrap, m)?,
        wrap_pyfunction!(fourth_power_foe, m)?,
        wrap_pyfunction!(viterbi_viterbi, m)?,
        wrap_pyfunction!(bps, m)?,
        wrap_pyfunction!(ddpll, m)?,
        wrap_pyfunction!(gardner_ted, m)?,
        wrap_pyfunction!(gardner_ted_nyquist, m)?,
        wrap_pyfunction!(interpolate_cubic, m)?,
        wrap_pyfunction!(gardner_clock_recovery, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}

//! DAC and ADC: quantisation, effective number of bits, clock jitter, bandwidth and
//! zero-order hold, with the closed forms of their noise.

use crate::rng::PyRng;
use crate::{CArr, IntoPy, carr, cvec};
use comrs::hardware::converter::{self as conv, Adc, Dac, Quantizer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The quantiser of `bits` bits: the given full scale, or `loading` (the optimal one
/// for a Gaussian input by default) times the RMS value of one dimension of `x`.
fn quantizer(
    x: &[num_complex::Complex64],
    bits: u32,
    full_scale: Option<f64>,
    loading: Option<f64>,
) -> PyResult<Quantizer> {
    match full_scale {
        Some(a) => Quantizer::new(bits, a).py(),
        None => {
            let l = loading.unwrap_or_else(|| conv::optimal_loading(bits));
            let rms = conv::rms_per_dimension(x);
            if rms <= 0.0 {
                return Err(PyValueError::new_err("the signal is zero: give the full scale"));
            }
            Quantizer::with_loading(bits, rms, l).py()
        }
    }
}

/// An analogue-to-digital converter on the in-phase and quadrature components: the
/// front-end bandwidth (Gaussian, 3 dB at ``bandwidth_hz``), the sampling-clock jitter
/// (first order), the noise of the effective number of bits, then a uniform mid-rise
/// quantiser of ``bits`` bits with clipping at the full scale.
///
/// Args:
///     x (array_like): the analogue signal (complex).
///     bits (int): resolution `N`.
///     fs (float): sampling rate (Hz).
///     full_scale (float, optional): the full scale `A` of each dimension (the levels
///         span `±(A - Δ/2)`, `Δ = 2A/2^N`).
///     loading (float, optional): without ``full_scale``, `A` is ``loading`` times the
///         RMS value of one dimension of ``x``; by default the loading that minimises the
///         error of a Gaussian input (``optimal_loading``).
///     enob (float, optional): effective number of bits (`≤ N`).
///     jitter_s (float): RMS clock jitter (s).
///     bandwidth_hz (float, optional): 3 dB bandwidth of the front end (Hz).
///     rng (Rng, optional): the generator, needed with ``enob`` or ``jitter_s``.
///
/// Returns:
///     numpy.ndarray: the digital samples (complex128).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> comrs.adc(np.array([0.3 - 0.01j]), 3, full_scale=1.0).tolist()
///     [(0.375-0.125j)]
#[pyfunction]
#[pyo3(signature = (
    x, bits, *, fs=1.0, full_scale=None, loading=None, enob=None, jitter_s=0.0, bandwidth_hz=None, rng=None
))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn adc<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    bits: u32,
    fs: f64,
    full_scale: Option<f64>,
    loading: Option<f64>,
    enob: Option<f64>,
    jitter_s: f64,
    bandwidth_hz: Option<f64>,
    rng: Option<PyRefMut<'_, PyRng>>,
) -> PyResult<CArr<'py>> {
    let x = cvec(x)?;
    let a = Adc { quantizer: quantizer(&x, bits, full_scale, loading)?, enob, jitter_s, bandwidth_hz };
    let random = enob.is_some() || jitter_s > 0.0;
    let y = match rng {
        Some(mut r) => {
            let inner = &mut r.inner;
            py.detach(|| a.convert(&x, fs, inner)).py()?
        }
        None if !random => a.convert(&x, fs, &mut comrs::core::rng::Rng::new(0)).py()?,
        None => return Err(PyValueError::new_err("adc: enob and jitter_s need an rng")),
    };
    Ok(carr(py, y))
}

/// A digital-to-analogue converter: a uniform mid-rise quantiser of ``bits`` bits on the
/// in-phase and quadrature components, a zero-order hold of ``hold`` output samples per
/// input sample, then the analogue bandwidth (Gaussian, 3 dB at ``bandwidth_hz``).
///
/// Args:
///     x (array_like): the digital samples (complex).
///     bits (int): resolution `N`.
///     fs (float): input sampling rate (Hz); the output is at ``hold`` × ``fs``.
///     full_scale (float, optional): the full scale of each dimension.
///     loading (float, optional): without ``full_scale``, the full scale in RMS values
///         of one dimension of ``x`` (``optimal_loading`` by default).
///     hold (int): output samples per input sample.
///     bandwidth_hz (float, optional): 3 dB bandwidth of the output (Hz).
///
/// Returns:
///     numpy.ndarray: the analogue signal (complex128), ``hold`` × ``len(x)`` samples.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> comrs.dac(np.array([0.3 + 0.0j]), 3, full_scale=1.0, hold=2).real.tolist()
///     [0.375, 0.375]
#[pyfunction]
#[pyo3(signature = (x, bits, *, fs=1.0, full_scale=None, loading=None, hold=1, bandwidth_hz=None))]
#[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python function")]
fn dac<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    bits: u32,
    fs: f64,
    full_scale: Option<f64>,
    loading: Option<f64>,
    hold: usize,
    bandwidth_hz: Option<f64>,
) -> PyResult<CArr<'py>> {
    let x = cvec(x)?;
    let d = Dac { quantizer: quantizer(&x, bits, full_scale, loading)?, hold, bandwidth_hz };
    Ok(carr(py, py.detach(|| d.convert(&x, fs)).py()?))
}

/// The SQNR of a full-scale sine through an ideal quantiser: `6.02 N + 1.76` dB
/// (Bennett 1948).
///
/// Args:
///     bits (int): resolution `N`.
///
/// Returns:
///     float: the SQNR (dB).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_sqnr_sine_db(8), 2)
///     49.93
#[pyfunction]
fn theory_sqnr_sine_db(bits: u32) -> f64 {
    conv::sqnr_sine_db(bits)
}

/// The exact mean squared error of a uniform mid-rise quantiser of ``bits`` bits and full
/// scale ``full_scale`` for a Gaussian input of standard deviation ``sigma`` (one
/// dimension): granular and overload (clipping) noise together.
///
/// Args:
///     bits (int): resolution `N`.
///     full_scale (float): the full scale `A`.
///     sigma (float): standard deviation of the input.
///
/// Returns:
///     float: the mean squared error.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_quantization_mse(2, comrs.optimal_loading(2)), 4)
///     0.1188
#[pyfunction]
#[pyo3(signature = (bits, full_scale, sigma=1.0))]
fn theory_quantization_mse(bits: u32, full_scale: f64, sigma: f64) -> PyResult<f64> {
    Ok(conv::gaussian_mse(&Quantizer::new(bits, full_scale).py()?, sigma))
}

/// The loading factor `A/σ` that minimises the error of a Gaussian input (the optimal
/// uniform quantiser of Max 1960).
///
/// Args:
///     bits (int): resolution `N`.
///
/// Returns:
///     float: the optimal loading factor.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.optimal_loading(1), 3)
///     1.596
#[pyfunction]
fn optimal_loading(bits: u32) -> f64 {
    conv::optimal_loading(bits)
}

/// The SNR of a sine of frequency ``f`` sampled with an RMS clock jitter ``jitter_s``:
/// `-20 log10(2π f σ_j)` dB (Walden 1999).
///
/// Args:
///     f (float): frequency of the sine (Hz).
///     jitter_s (float): RMS jitter (s).
///
/// Returns:
///     float: the SNR (dB).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_jitter_snr_db(1e9, 100e-15), 1)
///     64.0
#[pyfunction]
fn theory_jitter_snr_db(f: f64, jitter_s: f64) -> f64 {
    conv::jitter_snr_db(f, jitter_s)
}

/// The effective number of bits of a SINAD measured on a full-scale sine:
/// `(SINAD - 1.76) / 6.02` (IEEE Std 1241).
///
/// Args:
///     sinad_db (float): the SINAD (dB).
///
/// Returns:
///     float: the ENOB (bits).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.enob_from_sinad_db(comrs.theory_sqnr_sine_db(10)), 6)
///     10.0
#[pyfunction]
fn enob_from_sinad_db(sinad_db: f64) -> f64 {
    conv::enob_from_sinad_db(sinad_db)
}

/// The amplitude response of a zero-order hold of ``hold`` output samples per input
/// sample at the input rate ``fs``: `|sin(π f/f_s) / (H sin(π f/(H f_s)))|`, which tends
/// to `|sinc(f/f_s)|`.
///
/// Args:
///     f (float): frequency (Hz).
///     fs (float): input sampling rate (Hz).
///     hold (int): output samples per input sample.
///
/// Returns:
///     float: the gain.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.zero_order_hold_gain(0.5, 1.0, 1000), 4)
///     0.6366
#[pyfunction]
fn zero_order_hold_gain(f: f64, fs: f64, hold: usize) -> f64 {
    conv::zero_order_hold_gain(f, fs, hold)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adc, m)?)?;
    m.add_function(wrap_pyfunction!(dac, m)?)?;
    m.add_function(wrap_pyfunction!(theory_sqnr_sine_db, m)?)?;
    m.add_function(wrap_pyfunction!(theory_quantization_mse, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_loading, m)?)?;
    m.add_function(wrap_pyfunction!(theory_jitter_snr_db, m)?)?;
    m.add_function(wrap_pyfunction!(enob_from_sinad_db, m)?)?;
    m.add_function(wrap_pyfunction!(zero_order_hold_gain, m)?)?;
    Ok(())
}

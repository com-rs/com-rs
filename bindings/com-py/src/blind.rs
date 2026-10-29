//! Blind estimation: cumulants, modulation classification, the M2M4 SNR, the symbol
//! rate by cyclostationarity.

use crate::comm::PyConstellation;
use crate::{IntoPy, cvec, map_real};
use comrs::dsp::cyclo;
use comrs::metrics::blind::{self, Cumulants};
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn cumulant_dict<'py>(py: Python<'py>, c: &Cumulants) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("c20", c.c20)?;
    d.set_item("c21", c.c21)?;
    d.set_item("c40", c.c40)?;
    d.set_item("c41", c.c41)?;
    d.set_item("c42", c.c42)?;
    d.set_item("c40_normalized", c.c40_normalized())?;
    d.set_item("c42_normalized", c.c42_normalized())?;
    Ok(d)
}

/// The sample cumulants of order two and four of a zero-mean signal, and the normalised
/// `C40/C21²`, `C42/C21²` (Swami & Sadler 2000).
///
/// Args:
///     x (array_like): the samples (complex).
///
/// Returns:
///     dict: ``c20``, ``c21``, ``c40``, ``c41``, ``c42``, ``c40_normalized``,
///     ``c42_normalized``.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.Constellation("qam", 4).normalized()
///     >>> round(comrs.cumulants(np.tile(c.points, 100))["c42_normalized"], 6)
///     -1.0
#[pyfunction]
fn cumulants<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    cumulant_dict(py, &Cumulants::estimate(&cvec(x)?).py()?)
}

/// The cumulants of an alphabet (its theoretical features), with the probabilities of
/// its points.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (uniform by default).
///
/// Returns:
///     dict: as :func:`cumulants`.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_cumulants(comrs.Constellation("qam", 16))["c42_normalized"], 2)
///     -0.68
#[pyfunction]
#[pyo3(signature = (constellation, *, px=None))]
fn theory_cumulants<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let px = px.map(crate::fvec).transpose()?;
    cumulant_dict(py, &Cumulants::of_constellation(&constellation.inner, px.as_deref()).py()?)
}

/// Blind modulation classification at one sample per symbol, timing and carrier
/// recovered.
///
/// Methods: `"ml"` (the likelihood, the scale and the noise estimated by M2M4 under
/// every candidate), `"ml_known"` (the likelihood with the noise variance ``noise_var``
/// known and the symbols of unit energy, Wei & Mendel 2000), `"cumulants"` (the nearest
/// fourth-order features `|C40|/C21²`, `C42/C21²`, Swami & Sadler 2000; with
/// ``noise_var``, normalised by `C21 - noise_var`).
///
/// Args:
///     x (array_like): the samples (complex).
///     candidates (list[Constellation]): the candidate alphabets (scaled to unit energy).
///     method (str): the classifier.
///     noise_var (float, optional): the noise variance, for `"ml_known"` and
///         `"cumulants"`.
///
/// Returns:
///     dict: ``index`` (the decided candidate) and ``scores`` (one per candidate, the
///     largest wins).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> rng = comrs.Rng(1)
///     >>> cands = [comrs.Constellation(k, m).normalized() for k, m in [("psk", 2), ("qam", 4), ("qam", 16)]]
///     >>> x = cands[2].points[rng.integers(16, 2000)] + rng.complex_noise(2000, 0.02)
///     >>> comrs.classify_modulation(x, cands)["index"]
///     2
#[pyfunction]
#[pyo3(signature = (x, candidates, method="ml", *, noise_var=None))]
fn classify_modulation<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    candidates: Vec<PyRef<'py, PyConstellation>>,
    method: &str,
    noise_var: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let x = cvec(x)?;
    let cands: Vec<_> = candidates.iter().map(|c| c.inner.normalized()).collect();
    let d = match method {
        "ml" => py.detach(|| blind::classify_ml_blind(&x, &cands)).py()?,
        "ml_known" => {
            let n = noise_var.ok_or_else(|| PyValueError::new_err("ml_known needs noise_var"))?;
            py.detach(|| blind::classify_ml(&x, &cands, n)).py()?
        }
        "cumulants" => blind::classify_cumulants(&x, &cands, noise_var).py()?,
        other => return Err(PyValueError::new_err(format!("unknown method {other:?}: ml, ml_known or cumulants"))),
    };
    let out = PyDict::new(py);
    out.set_item("index", d.index)?;
    out.set_item("scores", d.scores)?;
    Ok(out)
}

/// The M2M4 estimate of the SNR (Pauluzzi & Beaulieu 2000), generalised to alphabets of
/// several amplitudes by their kurtosis; it degrades at high SNR for QAM.
///
/// Args:
///     x (array_like): the samples (complex), at one sample per symbol.
///     constellation (Constellation): the alphabet.
///
/// Returns:
///     float: the SNR (dB); ``inf`` when the moments give no noise.
///
/// Examples:
///     >>> import comrs
///     >>> rng = comrs.Rng(2)
///     >>> c = comrs.Constellation("psk", 8).normalized()
///     >>> x = c.points[rng.integers(8, 100_000)] + rng.complex_noise(100_000, 0.1)
///     >>> round(comrs.snr_m2m4_db(x, c), 1)
///     10.0
#[pyfunction]
fn snr_m2m4_db(x: &Bound<'_, PyAny>, constellation: &PyConstellation) -> PyResult<f64> {
    let snr = blind::snr_m2m4(&cvec(x)?, &constellation.inner).py()?;
    Ok(comrs::math::lin2db(snr))
}

/// The Cramér–Rao bound on the normalised mean squared error of an unbiased SNR
/// estimator from known PSK symbols (Pauluzzi & Beaulieu 2000, eq. (65) in complex AWGN,
/// eq. (67), twice it, in real AWGN).
///
/// Args:
///     snr_db (float | array_like): the true SNR (dB).
///     n_symbols (int): the symbols of a block.
///     samples_per_symbol (int): independent samples per symbol.
///     real (bool): real AWGN instead of complex.
///
/// Returns:
///     float | numpy.ndarray: the bound on ``E[(ρ̂ - ρ)²] / ρ²``.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.snr_crb_nmse(30.0, 64), 4)  # tends to 1/64
///     0.0157
#[pyfunction]
#[pyo3(signature = (snr_db, n_symbols, *, samples_per_symbol=1, real=false))]
fn snr_crb_nmse(
    snr_db: &Bound<'_, PyAny>,
    n_symbols: usize,
    samples_per_symbol: usize,
    real: bool,
) -> PyResult<Py<PyAny>> {
    if n_symbols == 0 || samples_per_symbol == 0 {
        return Err(PyValueError::new_err("n_symbols and samples_per_symbol must be positive"));
    }
    map_real(snr_db, |s| Ok(blind::snr_crb_nmse(comrs::math::db2lin(s), n_symbols, samples_per_symbol, real)))
}

/// The symbol rate of a linearly modulated signal from the strongest line of the
/// spectrum of its instantaneous power (cyclostationarity, Gardner 1991); the pulse must
/// have an excess bandwidth.
///
/// Args:
///     x (array_like): the samples (complex).
///     fs (float): the sampling rate (Hz).
///     f_min (float): the lowest symbol rate searched (Hz).
///     f_max (float): the highest symbol rate searched (Hz, at most `fs/2`).
///
/// Returns:
///     float: the symbol rate (Hz).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> n = np.arange(4096)
///     >>> x = np.sqrt(1 + 0.3 * np.cos(np.pi * n / 4)).astype(complex)
///     >>> round(comrs.estimate_symbol_rate(x, 1.0, 0.05, 0.2), 6)
///     0.125
#[pyfunction]
fn estimate_symbol_rate(x: &Bound<'_, PyAny>, fs: f64, f_min: f64, f_max: f64) -> PyResult<f64> {
    cyclo::estimate_symbol_rate(&cvec(x)?, fs, f_min, f_max).py()
}

/// The cyclic autocorrelation `R^α(τ) = mean x[n+τ] x*[n] e^{-j2π α n/f_s}`.
///
/// Args:
///     x (array_like): the samples (complex).
///     alpha (float): the cycle frequency (Hz).
///     lag (int): the lag `τ` (samples).
///     fs (float): the sampling rate (Hz).
///
/// Returns:
///     complex: the cyclic autocorrelation.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> x = np.ones(100, dtype=complex)
///     >>> abs(comrs.cyclic_autocorrelation(x, 0.0, 0, 1.0))
///     1.0
#[pyfunction]
#[pyo3(signature = (x, alpha, lag=0, fs=1.0))]
fn cyclic_autocorrelation(x: &Bound<'_, PyAny>, alpha: f64, lag: usize, fs: f64) -> PyResult<C64> {
    Ok(cyclo::cyclic_autocorrelation(&cvec(x)?, alpha, lag, fs))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cumulants, m)?)?;
    m.add_function(wrap_pyfunction!(theory_cumulants, m)?)?;
    m.add_function(wrap_pyfunction!(classify_modulation, m)?)?;
    m.add_function(wrap_pyfunction!(snr_m2m4_db, m)?)?;
    m.add_function(wrap_pyfunction!(snr_crb_nmse, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_symbol_rate, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_autocorrelation, m)?)?;
    Ok(())
}

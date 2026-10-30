//! Spectral estimation (Welch's PSD, the averaged periodogram) and the adjacent-channel
//! leakage ratio.

use crate::{FArr, IntoPy, cvec, farr, fvec};
use comrs::metrics::spectrum;
use pyo3::prelude::*;

/// Welch's power spectral density (periodic Hann window, 50 % overlap), centred.
///
/// Args:
///     x (array_like): the signal.
///     fs (float): sampling rate (Hz).
///     nfft (int): length of a segment.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the frequencies (Hz) and the density
///     (power per Hz), which integrates to the average power.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, *, nfft=1024))]
fn psd<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64, nfft: usize) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let v = cvec(x)?;
    let (f, s) = py.detach(|| spectrum::psd(&v, fs, nfft)).py()?;
    Ok((farr(py, f), farr(py, s)))
}

/// The averaged periodogram of non-overlapping segments of `nfft` samples (no window):
/// with `nfft` the block length of a cyclic signal (OFDM without prefix), it has no
/// leakage, and its out-of-band part is the regrowth of a nonlinearity.
///
/// Args:
///     x (array_like): the signal.
///     fs (float): sampling rate (Hz).
///     nfft (int): length of a segment.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the frequencies (Hz) and the density
///     (power per Hz).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, *, nfft=1024))]
fn periodogram<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64, nfft: usize) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let v = cvec(x)?;
    let (f, s) = py.detach(|| spectrum::periodogram(&v, fs, nfft)).py()?;
    Ok((farr(py, f), farr(py, s)))
}

/// The adjacent channel leakage ratio (dB) of a density (`psd` or `periodogram`): power
/// in `[-B/2, B/2)` over the larger power of the channels of width `B` centred at
/// `±offset`.
///
/// Args:
///     freqs (array_like): the frequencies of the density (Hz).
///     density (array_like): the density.
///     bandwidth (float): the channel width `B` (Hz).
///     offset (float): the offset of the adjacent channels (Hz).
///
/// Returns:
///     float: the ACLR (dB).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn aclr_db(freqs: &Bound<'_, PyAny>, density: &Bound<'_, PyAny>, bandwidth: f64, offset: f64) -> PyResult<f64> {
    spectrum::aclr_db(&fvec(freqs)?, &fvec(density)?, bandwidth, offset).py()
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(psd, m)?)?;
    m.add_function(wrap_pyfunction!(periodogram, m)?)?;
    m.add_function(wrap_pyfunction!(aclr_db, m)?)?;
    Ok(())
}

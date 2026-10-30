//! Closed forms of OFDM links with impairments: equalisation with an imperfect channel
//! estimate over Rayleigh fading (Chang & Su 2002), differential detection over Jakes'
//! fading, and the distortion of a nonlinear amplifier driven by an OFDM signal
//! (Banelli & Cacopardi 2000).

use crate::{FArr, IntoPy, count, cvec, farr, fvec, map_real};
use comrs::dsp::nonlinear;
use comrs::mapping::constellation::Constellation;
use comrs::mapping::imperfect_csi as csi;
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The bit error rate over flat Rayleigh fading of a receiver that equalises with a
/// channel estimate jointly Gaussian with the channel, of correlations `rho1` (real) and
/// `rho2` (imaginary) and of standard deviation ratio `sigma_ratio` = σ2/σ1 (M.-X. Chang
/// and Y. T. Su, *IEEE Trans. Wireless Commun.* 1(4), 2002, eqs. 31, 32, 53; every linear
/// pilot-assisted estimate has this form). `rho1 = 1`, `rho2 = 0`, `sigma_ratio = 1`:
/// the perfect channel.
///
/// Args:
///     modulation (str): ``"bpsk"``, ``"qpsk"`` (Gray) or ``"16qam"`` (Gray), any name of
///         a constellation (``"16-QAM"``, ``"QPSK"``).
///     ebn0_db (float | array_like): the mean `Eb/N0` (dB).
///     rho1 (float): the real correlation coefficient.
///     rho2 (float): the imaginary correlation coefficient.
///     sigma_ratio (float): σ2/σ1 (16-QAM only: PSK does not depend on `|Ĥ|`).
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every `Eb/N0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> p = comrs.theory_ber_imperfect_csi("16qam", 50.0, 0.999)
///     >>> round(p, 4)          # the error floor of Chang & Su's Fig. 4
///     0.0034
#[pyfunction]
#[pyo3(signature = (modulation, ebn0_db, rho1, rho2 = 0.0, sigma_ratio = 1.0))]
fn theory_ber_imperfect_csi(
    modulation: &str,
    ebn0_db: &Bound<'_, PyAny>,
    rho1: f64,
    rho2: f64,
    sigma_ratio: f64,
) -> PyResult<Py<PyAny>> {
    let c = Constellation::from_name(modulation).py()?;
    map_real(ebn0_db, |e| csi::ber_imperfect_csi(&c, comrs::math::db2lin(e), rho1, rho2, sigma_ratio).py())
}

/// The bit error rate of DBPSK (`m = 2`) or Gray DQPSK (`m = 4`) over Jakes' Rayleigh
/// fading of normalised Doppler `fd_ts` (Chang & Su 2002, eqs. 40, 41): an error floor
/// of `(1 - J0(2π fd Ts))/2` for DBPSK.
///
/// Args:
///     m (int): 2 or 4.
///     ebn0_db (float | array_like): the mean `Eb/N0` (dB).
///     fd_ts (float): the maximum Doppler frequency times the symbol duration.
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every `Eb/N0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn theory_ber_dpsk_jakes(m: usize, ebn0_db: &Bound<'_, PyAny>, fd_ts: f64) -> PyResult<Py<PyAny>> {
    let g = |e: f64| comrs::math::db2lin(e);
    match m {
        2 => map_real(ebn0_db, |e| csi::ber_dbpsk_jakes(g(e), fd_ts).py()),
        4 => map_real(ebn0_db, |e| csi::ber_dqpsk_jakes(g(e), fd_ts).py()),
        _ => Err(PyValueError::new_err(format!("m is 2 or 4, got {m}"))),
    }
}

/// The coefficients `c_n` of the output autocorrelation of a memoryless envelope
/// nonlinearity `f` driven by a circular complex Gaussian signal of unit power
/// (P. Banelli and S. Cacopardi, *IEEE Trans. Commun.* 48(3), 2000, eq. 7): `c_0` is
/// the useful power `|α|²` (Bussgang), the term `n` has the input spectrum convolved
/// `2n + 1` times with itself, and the coefficients sum to the output power.
///
/// Args:
///     f (callable): the complex output envelope `A(r) exp(jΦ(r))` of an input envelope
///         `r` (vectorised over a NumPy array).
///     n_terms (int): the number of coefficients.
///
/// Returns:
///     numpy.ndarray: `c_0, …, c_{n_terms - 1}`.
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.envelope_series(lambda r: np.minimum(r, np.sqrt(2.0)), 60)  # 3 dB IBO
///     >>> round(float(c.sum()), 4), round(float(1 - np.exp(-2.0)), 4)
///     (0.8647, 0.8647)
#[pyfunction]
#[pyo3(signature = (f, n_terms = 60))]
fn envelope_series<'py>(py: Python<'py>, f: &Bound<'py, PyAny>, n_terms: i64) -> PyResult<FArr<'py>> {
    let n_terms = count(n_terms, "n_terms")?;
    let r = farr(py, nonlinear::series_grid());
    let values: Vec<C64> = cvec(&f.call1((r,))?)?;
    Ok(farr(py, nonlinear::output_series_sampled(&values, n_terms)))
}

/// The power of the useful term and of the distortion on each of `bins` frequency bins,
/// for the coefficients `c` of `envelope_series` and an input flat on the `occupied`
/// bins centred on bin 0 (an OFDM signal of `occupied` subcarriers oversampled to
/// `bins`; Banelli & Cacopardi, eq. 9).
///
/// Args:
///     c (array_like): the coefficients `c_n`.
///     occupied (int): the occupied bins.
///     bins (int): the bins of the circular grid (FFT order: bin 0 first).
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the useful and the distortion power per bin.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn distortion_spectrum<'py>(
    py: Python<'py>,
    c: &Bound<'py, PyAny>,
    occupied: usize,
    bins: usize,
) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let (u, d) = nonlinear::distortion_spectrum(&fvec(c)?, occupied, bins).py()?;
    Ok((farr(py, u), farr(py, d)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theory_ber_imperfect_csi, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_dpsk_jakes, m)?)?;
    m.add_function(wrap_pyfunction!(envelope_series, m)?)?;
    m.add_function(wrap_pyfunction!(distortion_spectrum, m)?)?;
    Ok(())
}

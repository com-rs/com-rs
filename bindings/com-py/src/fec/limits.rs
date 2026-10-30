//! The finite-length limits and the BPSK helpers of the code tests: the BI-AWGN capacity,
//! the normal approximation, the Shannon limit.

use super::*;

/// Capacity of the binary-input AWGN channel for a code rate (the noise variance is
/// `1/(2 R Eb/N0)`).
///
/// Args:
///     ebn0_db (float | array_like): `Eb/N0` (dB).
///     rate (float): the code rate `R`.
///
/// Returns:
///     float | numpy.ndarray: the capacity (bits per channel use) at every `Eb/N0`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_biawgn_capacity(comrs.theory_shannon_limit_ebn0_db(0.5), 0.5), 6)
///     0.5
#[pyfunction]
fn theory_biawgn_capacity(ebn0_db: &Bound<'_, PyAny>, rate: f64) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::biawgn_capacity(bounds::bpsk_noise_variance(e, rate))))
}

/// The normal approximation of the smallest block error probability of an `(n, k)`
/// binary code over BPSK and AWGN (Polyanskiy, Poor & Verdú 2010).
///
/// Args:
///     n (int): block length.
///     k (int): information bits.
///     ebn0_db (float | array_like): `Eb/N0` (dB).
///
/// Returns:
///     float | numpy.ndarray: the block error probability at every `Eb/N0`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> p = comrs.theory_normal_approximation(256, 128, [1.0, 3.0])
///     >>> bool(p[0] > p[1])
///     True
#[pyfunction]
fn theory_normal_approximation(n: usize, k: usize, ebn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::normal_approximation(n, k, e)))
}

/// The smallest `Eb/N0` at which BPSK supports a code rate.
///
/// Args:
///     rate (float | array_like): the code rate `R`.
///
/// Returns:
///     float | numpy.ndarray: the limit (dB).
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_shannon_limit_ebn0_db(0.5), 3)
///     0.187
#[pyfunction]
fn theory_shannon_limit_ebn0_db(rate: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(rate, |r| Ok(bounds::shannon_limit_ebn0_db(r)))
}

/// The noise variance per dimension of BPSK at `E_b/N_0` for a code of rate `rate`:
/// `σ² = 1 / (2 R E_b/N_0)` (unit-energy symbols).
///
/// Args:
///     ebn0_db (float | array_like): `E_b/N_0` (dB).
///     rate (float): the code rate.
///
/// Returns:
///     float | numpy.ndarray: `σ²` at every `E_b/N_0`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.bpsk_noise_variance(0.0, 0.5)
///     1.0
#[pyfunction]
fn bpsk_noise_variance(ebn0_db: &Bound<'_, PyAny>, rate: f64) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::bpsk_noise_variance(e, rate)))
}

/// The LLRs of BPSK (`0 ↦ +1`, `1 ↦ -1`) received in real Gaussian noise of variance
/// `noise_var` per dimension: `2y/σ²` (positive for a 0, as every decoder of the package
/// expects).
///
/// Args:
///     y (array_like): the received real samples.
///     noise_var (float): the noise variance per dimension `σ²`.
///
/// Returns:
///     numpy.ndarray: one LLR per sample (float64).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.bpsk_llr([1.0, -0.5], 0.5).tolist()
///     [4.0, -2.0]
#[pyfunction]
fn bpsk_llr<'py>(py: Python<'py>, y: &Bound<'py, PyAny>, noise_var: f64) -> PyResult<FArr<'py>> {
    Ok(farr(py, comrs::mapping::demapper::bpsk_llr(&fvec(y)?, noise_var)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theory_biawgn_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(bpsk_noise_variance, m)?)?;
    m.add_function(wrap_pyfunction!(bpsk_llr, m)?)?;
    m.add_function(wrap_pyfunction!(theory_normal_approximation, m)?)?;
    m.add_function(wrap_pyfunction!(theory_shannon_limit_ebn0_db, m)?)?;
    Ok(())
}

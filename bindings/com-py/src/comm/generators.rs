//! The sources: random bits, PRBS and symbols.

use super::*;

/// Independent fair bits.
///
/// Args:
///     n (int): number of bits.
///     rng (Rng): the generator.
///
/// Returns:
///     numpy.ndarray: `n` bits (uint8).
///
/// Raises:
///     ValueError: for a negative `n`.
///
/// Examples:
///     >>> import comrs
///     >>> b = comrs.bits(8, comrs.Rng(1))
///     >>> b.dtype, b.size, set(b.tolist()) <= {0, 1}
///     (dtype('uint8'), 8, True)
#[pyfunction]
fn bits<'py>(py: Python<'py>, n: i64, mut rng: PyRefMut<'py, PyRng>) -> PyResult<BArr<'py>> {
    Ok(sources::bits(count(n, "n")?, &mut rng.inner).into_pyarray(py))
}

/// A maximal-length pseudo-random binary sequence (ITU-T O.150 polynomials).
///
/// Args:
///     order (int): 7, 9, 11, 15, 20, 23 or 31 (period `2^order - 1`).
///     length (int): number of bits.
///     seed (int): initial register state (non-zero after masking).
///
/// Returns:
///     numpy.ndarray: `length` bits (uint8).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> int(comrs.prbs(7, 127).sum())
///     64
#[pyfunction]
#[pyo3(signature = (order, length, *, seed = 1))]
fn prbs(py: Python<'_>, order: u32, length: i64, seed: u64) -> PyResult<BArr<'_>> {
    Ok(sources::prbs(order, count(length, "length")?, seed).py()?.into_pyarray(py))
}

/// Random symbols and their indices.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     n (int): number of symbols.
///     rng (Rng): the generator.
///     px (array_like, optional): probabilities of the points (uniform by default).
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the indices (int64) and the symbols
///     (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (constellation, n, rng, *, px = None))]
fn symbols<'py>(
    py: Python<'py>,
    constellation: &PyConstellation,
    n: i64,
    mut rng: PyRefMut<'py, PyRng>,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<(IArr<'py>, CArr<'py>)> {
    let px = px.map(crate::fvec).transpose()?;
    let (idx, x) = sources::symbols(&constellation.inner, count(n, "n")?, px.as_deref(), &mut rng.inner).py()?;
    Ok((iarr(py, idx), carr(py, x)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bits, m)?)?;
    m.add_function(wrap_pyfunction!(prbs, m)?)?;
    m.add_function(wrap_pyfunction!(symbols, m)?)?;
    Ok(())
}

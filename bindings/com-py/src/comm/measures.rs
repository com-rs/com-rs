//! The measured metrics: error rates, SNR, EVM and the information rates.

use super::*;

/// Bit error rate of two bit sequences of the same length.
///
/// Args:
///     rx_bits (array_like): the decided bits.
///     tx_bits (array_like): the transmitted bits.
///
/// Returns:
///     float: the fraction of different bits.
///
/// Raises:
///     ValueError: for sequences of different lengths.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.ber([0, 1, 1, 0], [0, 1, 0, 0])
///     0.25
#[pyfunction]
fn ber(rx_bits: &Bound<'_, PyAny>, tx_bits: &Bound<'_, PyAny>) -> PyResult<f64> {
    metrics::ber(&bvec(rx_bits)?, &bvec(tx_bits)?).py()
}

/// SNR (dB) of a received sequence against the transmitted one,
/// `10 log10(P_tx/P_{rx-tx})`.
///
/// Args:
///     rx (array_like): the received samples.
///     tx (array_like): the transmitted samples, of the same length.
///
/// Returns:
///     float: the SNR (dB).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn snr_db(rx: &Bound<'_, PyAny>, tx: &Bound<'_, PyAny>) -> PyResult<f64> {
    metrics::snr_db(&cvec(rx)?, &cvec(tx)?).py()
}

/// BER, SER and SNR of a received symbol sequence (OptiCommPy `fastBERcalc`).
///
/// The received sequence is rotated by `mean(tx/rx)` for QAM and PSK and normalised to
/// unit power, then decided on the normalised constellation.
///
/// Args:
///     rx (array_like): the received symbols.
///     tx (array_like): the transmitted symbols.
///     constellation (Constellation): the alphabet.
///
/// Returns:
///     tuple[float, float, float]: BER, SER and SNR (dB).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn fast_ber_calc(
    rx: &Bound<'_, PyAny>,
    tx: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
) -> PyResult<(f64, f64, f64)> {
    let r = metrics::fast_ber_calc(&cvec(rx)?, &cvec(tx)?, &constellation.inner).py()?;
    Ok((r.ber, r.ser, r.snr_db))
}

/// Error vector magnitude as a power ratio `E|y - x̂|² / E|x̂|²` (OptiCommPy `calcEVM`);
/// the EVM in percent is `100·√evm`.
///
/// With `tx`, `x̂` is the transmitted sequence (after the same preprocessing as
/// `fast_ber_calc`); without, the nearest points of the normalised constellation.
///
/// Args:
///     symbols (array_like): the received symbols.
///     constellation (Constellation): the alphabet.
///     tx (array_like, optional): the transmitted symbols.
///
/// Returns:
///     float: the EVM as a power ratio.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (symbols, constellation, *, tx = None))]
fn evm(symbols: &Bound<'_, PyAny>, constellation: &PyConstellation, tx: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    let tx = tx.map(cvec).transpose()?;
    metrics::evm(&cvec(symbols)?, &constellation.inner, tx.as_deref()).py()
}

/// Generalised mutual information of bit-wise decoding (OptiCommPy `monteCarloGMI`).
///
/// Args:
///     rx (array_like): the received symbols.
///     tx (array_like): the transmitted symbols.
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (probabilistic
///         shaping); uniform by default. The constellation is normalised to unit energy
///         under them and the LLRs include them.
///
/// Returns:
///     tuple[float, float]: GMI (bits/symbol) and normalised GMI (per bit of
///     information, ``GMI / H(X)``).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (rx, tx, constellation, *, px = None))]
fn monte_carlo_gmi(
    rx: &Bound<'_, PyAny>,
    tx: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    px: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, f64)> {
    let px = px.map(crate::fvec).transpose()?;
    metrics::monte_carlo_gmi(&cvec(rx)?, &cvec(tx)?, &constellation.inner, px.as_deref()).map(|g| (g.gmi, g.ngmi)).py()
}

/// Symbol-wise mutual information for a Gaussian channel of the measured noise
/// variance (OptiCommPy `monteCarloMI`; the transmitted symbols are snapped to the
/// constellation, so that the estimate never exceeds `H(X)`).
///
/// Args:
///     rx (array_like): the received symbols.
///     tx (array_like): the transmitted symbols.
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points (uniform by
///         default).
///
/// Returns:
///     float: the mutual information (bits/symbol).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (rx, tx, constellation, *, px = None))]
fn monte_carlo_mi(
    rx: &Bound<'_, PyAny>,
    tx: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    px: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let px = px.map(crate::fvec).transpose()?;
    metrics::monte_carlo_mi(&cvec(rx)?, &cvec(tx)?, &constellation.inner, px.as_deref()).py()
}

/// Achievable rates of a receiver with its own posterior `q(x|z) ∝ e^{-|z - x|²/σ̂²}`.
///
/// The rate of a decoder that uses the receiver's beliefs (mismatched decoding), not a
/// genie's: `air = H(X) - E[-log2 q(x|z)]` on the unit-energy constellation, without
/// any gain correction, and its bit-wise version with the LLRs of `q`. Scaling the
/// exponent by `s` and maximising gives `air_opt` (per block); `calibration =
/// σ̂²/mean|z - x|²` is 1 for a calibrated receiver, below 1 for an over-confident one.
///
/// Args:
///     z (array_like): the receiver's output, one sample per symbol.
///     indices (array_like): the transmitted indices.
///     constellation (Constellation): the alphabet (normalised inside).
///     noise_var (float): the noise variance `σ̂²` the receiver believes in.
///     px (array_like, optional): prior probabilities of the points; uniform by default.
///
/// Returns:
///     dict: ``air``, ``air_bmd``, ``air_opt``, ``air_bmd_opt`` (bits/symbol), ``scale``,
///     ``scale_bmd`` (the best scalings) and ``calibration``.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (z, indices, constellation, noise_var, *, px = None))]
fn posterior_rates<'py>(
    py: Python<'py>,
    z: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    noise_var: f64,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let (z, idx) = (cvec(z)?, uvec(indices)?);
    let px = px.map(crate::fvec).transpose()?;
    let c = &constellation.inner;
    let r = py.detach(|| metrics::posterior_rates(&z, &idx, c, noise_var, px.as_deref())).py()?;
    let d = pyo3::types::PyDict::new(py);
    for (k, v) in [
        ("air", r.air),
        ("air_bmd", r.air_bmd),
        ("air_opt", r.air_opt),
        ("air_bmd_opt", r.air_bmd_opt),
        ("scale", r.scale),
        ("scale_bmd", r.scale_bmd),
        ("calibration", r.calibration),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Mutual information and GMI with the least-squares channel gain: consistent at every
/// SNR, unlike the power normalisation of `monte_carlo_mi` and `monte_carlo_gmi`.
///
/// The model is `y = a x + w` with `x` the points at `indices` of the constellation
/// normalised to unit energy under the priors ``px`` (uniform by default),
/// `a = Σ y x*/Σ|x|²` and `σ² = mean|y - a x|²`; the rates use the priors.
///
/// Args:
///     rx (array_like): the received symbols.
///     indices (array_like): the transmitted indices.
///     constellation (Constellation): the alphabet.
///     px (array_like, optional): prior probabilities of the points.
///
/// Returns:
///     dict: ``mi``, ``gmi`` (bits/symbol), ``ngmi`` (per bit of information),
///     ``snr_db`` (the effective SNR, dB) and ``entropy`` (``H(X)``, bits).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (rx, indices, constellation, *, px = None))]
fn information_rates<'py>(
    py: Python<'py>,
    rx: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    constellation: &PyConstellation,
    px: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let px = px.map(crate::fvec).transpose()?;
    let r = metrics::information_rates(&cvec(rx)?, &uvec(indices)?, &constellation.inner, px.as_deref()).py()?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("mi", r.mi)?;
    d.set_item("gmi", r.gmi)?;
    d.set_item("ngmi", r.ngmi)?;
    d.set_item("snr_db", r.snr_db)?;
    d.set_item("entropy", r.entropy)?;
    Ok(d)
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ber, m)?)?;
    m.add_function(wrap_pyfunction!(snr_db, m)?)?;
    m.add_function(wrap_pyfunction!(fast_ber_calc, m)?)?;
    m.add_function(wrap_pyfunction!(evm, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_gmi, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_mi, m)?)?;
    m.add_function(wrap_pyfunction!(information_rates, m)?)?;
    m.add_function(wrap_pyfunction!(posterior_rates, m)?)?;
    Ok(())
}

//! The closed forms of the error and information rates of the constellations.

use super::*;

/// Theoretical BER over AWGN with Gray mapping (OptiCommPy `theoryBER`, exact for BPSK).
///
/// Args:
///     m (int): order.
///     ebn0_db (float | array_like): `E_b/N_0` (dB).
///     kind (str): `"qam"`, `"psk"` or `"pam"`.
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every `E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_ber(4, 0.0), 6)
///     0.07865
///     >>> comrs.theory_ber(16, [8.0, 12.0], "qam").shape
///     (2,)
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, kind = "qam"))]
fn theory_ber(m: usize, ebn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
    let k = self::kind(kind)?;
    map_real(ebn0_db, |e| metrics::theory_ber(k, m, e).py())
}

/// Exact BER of Gray-mapped PAM, square QAM or rectangular QAM over AWGN (K. Cho and
/// D. Yoon, *IEEE Trans. Commun.* 50(7), 2002, eqs. (9)–(22)).
///
/// Args:
///     m (int): order of the PAM (`kind="pam"`), of the square QAM (`kind="qam"`), or the
///         number of in-phase levels of a rectangular QAM (`kind="qam"` with `mq`).
///     ebn0_db (float | array_like): `E_b/N_0` (dB).
///     kind (str): `"pam"` or `"qam"`.
///     mq (int | None): the number of quadrature levels of a rectangular `m × mq` QAM.
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every `E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_ber_gray(4, 0.0, "qam"), 6)
///     0.07865
///     >>> bool(comrs.theory_ber_gray(8, 12.0, "qam", mq=4) < comrs.theory_ber_gray(64, 12.0))
///     True
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, kind = "qam", *, mq = None))]
fn theory_ber_gray(m: usize, ebn0_db: &Bound<'_, PyAny>, kind: &str, mq: Option<usize>) -> PyResult<Py<PyAny>> {
    use comrs::mapping::constellation::ConstellationKind as Kind;
    use comrs::mapping::theory as th;
    match (self::kind(kind)?, mq) {
        (Kind::Pam, None) => map_real(ebn0_db, |e| th::ber_gray_pam(m, e).py()),
        (Kind::Qam, None) => map_real(ebn0_db, |e| th::ber_gray_qam(m, e).py()),
        (Kind::Qam, Some(j)) => map_real(ebn0_db, |e| th::ber_gray_rect_qam(m, j, e).py()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "the exact Gray BER is for kind \"pam\" or \"qam\" (mq with qam only), got {kind:?}"
        ))),
    }
}

/// BER of BPSK (or Gray QPSK, per bit) with maximal-ratio combining of ``branches``
/// independent Rayleigh-faded branches (J. G. Proakis and M. Salehi, *Digital
/// Communications*, 5th ed., eq. (13.4-15)):
/// `P_b = ((1 - μ)/2)^L Σ_{k<L} C(L-1+k, k) ((1 + μ)/2)^k`, `μ = √(γ/(1 + γ))`, with
/// `γ` the mean `E_b/N_0` of one branch; one branch is the flat Rayleigh channel,
/// `P_b = (1 - μ)/2`.
///
/// Args:
///     ebn0_db (float | array_like): the mean `E_b/N_0` of one branch (dB).
///     branches (int): the number of branches `L` (at least 1).
///
/// Returns:
///     float | numpy.ndarray: the bit error probability at every `E_b/N_0`.
///
/// Raises:
///     ValueError: for no branch.
///
/// Examples:
///     >>> import comrs, math
///     >>> g = comrs.db2lin(10.0)
///     >>> round(comrs.theory_ber_rayleigh_mrc(10.0) / (0.5 * (1 - math.sqrt(g / (1 + g)))), 12)
///     1.0
///     >>> bool(comrs.theory_ber_rayleigh_mrc(10.0, branches=4) < 1e-5)
///     True
#[pyfunction]
#[pyo3(signature = (ebn0_db, *, branches = 1))]
fn theory_ber_rayleigh_mrc(ebn0_db: &Bound<'_, PyAny>, branches: usize) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| theory::ber_rayleigh_mrc(comrs::math::db2lin(e), branches).py())
}

/// Density of the phase error of a carrier in circular Gaussian noise (S. O. Rice 1948;
/// eq. (8) of Ahmed, Thompson and Zeidler, MILCOM 2007).
///
/// Args:
///     theta (float | array_like): phase error (rad), in `(-π, π]`.
///     snr_lin (float): `A²/E|w|²` (linear).
///
/// Returns:
///     float | numpy.ndarray: the density at every `theta`.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs, math
///     >>> round(comrs.theory_phase_pdf(1.0, 0.0) * 2 * math.pi, 12)
///     1.0
#[pyfunction]
fn theory_phase_pdf(theta: &Bound<'_, PyAny>, snr_lin: f64) -> PyResult<Py<PyAny>> {
    map_real(theta, |t| Ok(comrs::mapping::theory::phase_pdf(t, snr_lin)))
}

/// Theoretical SER over AWGN (exact for square QAM and PAM, a tight bound for PSK).
///
/// Args:
///     m (int): order.
///     ebn0_db (float | array_like): `E_b/N_0` (dB).
///     kind (str): `"qam"`, `"psk"` or `"pam"`.
///
/// Returns:
///     float | numpy.ndarray: the symbol error probability at every `E_b/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_ser(2, 0.0, "psk"), 6)
///     0.07865
///     >>> bool(comrs.theory_ser(16, 10.0) > comrs.theory_ber(16, 10.0))
///     True
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, kind = "qam"))]
fn theory_ser(m: usize, ebn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
    let k = self::kind(kind)?;
    map_real(ebn0_db, |e| metrics::theory_ser(k, m, e).py())
}

/// Theoretical SER over AWGN as a function of `E_s/N_0`: :func:`theory_ser` with
/// `E_b/N_0 = E_s/N_0 - 10 log10(log2 M)`, for the links whose SNR is per symbol.
///
/// Args:
///     m (int): order.
///     esn0_db (float | array_like): `E_s/N_0` (dB).
///     kind (str): `"qam"`, `"psk"` or `"pam"`.
///
/// Returns:
///     float | numpy.ndarray: the symbol error probability at every `E_s/N_0`.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs, math
///     >>> a = comrs.theory_ser_esn0(4, 10.0 + 10 * math.log10(2))
///     >>> round(a / comrs.theory_ser(4, 10.0), 12)
///     1.0
#[pyfunction]
#[pyo3(signature = (m, esn0_db, kind = "qam"))]
fn theory_ser_esn0(m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
    let k = self::kind(kind)?;
    map_real(esn0_db, |e| theory::ser_esn0(k, m, e).py())
}

/// Mutual information of a constellation over AWGN (OptiCommPy `theoryMI`) or Rayleigh
/// fading known at the receiver, by Gauss–Hermite quadrature.
///
/// Args:
///     m (int): order.
///     snr_db (float | array_like): `E_s/σ²` (dB); PAM and OOK see a real channel, the
///         others a circular complex one, as in OptiCommPy.
///     kind (str): `"qam"`, `"psk"`, `"pam"` or `"ook"`.
///     px (array_like, optional): probabilities of the points.
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the mutual information (bits/symbol) at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_mi(4, 30.0), 6)
///     2.0
#[pyfunction]
#[pyo3(signature = (m, snr_db, kind = "qam", *, px = None, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_mi(
    m: usize,
    snr_db: &Bound<'_, PyAny>,
    kind: &str,
    px: Option<&Bound<'_, PyAny>>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let px = px.map(crate::fvec).transpose()?;
    let (c, f) = (Constellation::new(self::kind(kind)?, m).py()?, self::fading(fading)?);
    map_real(snr_db, |s| theory::constellation_mi(&c, s, px.as_deref(), f, points).py())
}

/// Generalised mutual information of a Gray-labelled constellation over AWGN
/// (OptiCommPy `theoryGMI`) or Rayleigh fading known at the receiver, by Gauss–Hermite
/// quadrature.
///
/// Args:
///     m (int): order.
///     snr_db (float | array_like): `E_s/σ²` (dB), as `theory_mi`.
///     kind (str): `"qam"`, `"psk"`, `"pam"` or `"ook"`.
///     px (array_like, optional): probabilities of the points.
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the GMI (bits/symbol) at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (m, snr_db, kind = "qam", *, px = None, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_gmi(
    m: usize,
    snr_db: &Bound<'_, PyAny>,
    kind: &str,
    px: Option<&Bound<'_, PyAny>>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let px = px.map(crate::fvec).transpose()?;
    let (c, f) = (Constellation::new(self::kind(kind)?, m).py()?, self::fading(fading)?);
    map_real(snr_db, |s| theory::constellation_gmi(&c, s, px.as_deref(), f, points).py())
}

/// The CM capacity (bits per channel use) of a constellation with uniform inputs over
/// AWGN or Rayleigh fading known at the receiver (Caire, Taricco & Biglieri 1998, eq.
/// (10)): :func:`theory_mi` of any constellation, uniform inputs.
///
/// Args:
///     constellation (Constellation): the points (normalised inside).
///     snr_db (float | array_like): `E_s/σ²` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the capacity at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_cm_capacity(comrs.Constellation("qam", 4), 30.0), 6)
///     2.0
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_cm_capacity(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| theory::constellation_mi(&constellation.inner, s, None, f, points).py())
}

/// The BICM capacity (bits per channel use) of a labelled constellation with uniform
/// inputs (Caire, Taricco & Biglieri 1998, eq. (14)): the GMI of the bit-wise decoder,
/// labels as `constellation` gives them (Gray, or `Constellation.set_partitioning`):
/// :func:`theory_gmi` of any labelled constellation, uniform inputs.
///
/// Args:
///     constellation (Constellation): the labelled points.
///     snr_db (float | array_like): `E_s/σ²` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the capacity at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_bicm_capacity(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| theory::constellation_gmi(&constellation.inner, s, None, f, points).py())
}

/// The CM cutoff rate (bits per channel use) with uniform inputs, in closed form
/// (Caire, Taricco & Biglieri 1998, eq. (17)).
///
/// Args:
///     constellation (Constellation): the points.
///     snr_db (float | array_like): `E_s/σ²` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///
/// Returns:
///     float | numpy.ndarray: the cutoff rate at every SNR.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs, math
///     >>> r0 = comrs.theory_cm_cutoff_rate(comrs.Constellation("psk", 2), 0.0)
///     >>> round(r0 - (1 - math.log2(1 + math.exp(-1))), 12)
///     0.0
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn"))]
fn theory_cm_cutoff_rate(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| Ok(bicm::cm_cutoff_rate(&constellation.inner, s, f)))
}

/// The BICM cutoff rate (bits per channel use) of a labelled constellation, `m(1 -
/// log2(1 + B))` with the Bhattacharyya factor `B` (Caire, Taricco & Biglieri 1998,
/// eqs. (19), (21)).
///
/// Args:
///     constellation (Constellation): the labelled points.
///     snr_db (float | array_like): `E_s/σ²` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the cutoff rate at every SNR.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, *, fading = "awgn", points = theory::DEFAULT_QUADRATURE_POINTS))]
fn theory_bicm_cutoff_rate(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| bicm::bicm_cutoff_rate(&constellation.inner, s, f, points).py())
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theory_ber, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ser, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ser_esn0, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_gray, m)?)?;
    m.add_function(wrap_pyfunction!(theory_ber_rayleigh_mrc, m)?)?;
    m.add_function(wrap_pyfunction!(theory_phase_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(theory_mi, m)?)?;
    m.add_function(wrap_pyfunction!(theory_gmi, m)?)?;
    m.add_function(wrap_pyfunction!(theory_cm_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(theory_bicm_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(theory_cm_cutoff_rate, m)?)?;
    m.add_function(wrap_pyfunction!(theory_bicm_cutoff_rate, m)?)?;
    Ok(())
}

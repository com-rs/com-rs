//! Sources, modulation and metrics.

use crate::rng::PyRng;
use crate::{BArr, CArr, FArr, IArr, IntoPy, bvec, carr, count, cvec, farr, iarr, kind, map_real, uvec};
use comrs::mapping::bicm;
use comrs::mapping::constellation::{self as modulation, Constellation};
use comrs::mapping::source as sources;
use comrs::mapping::theory::{self, Fading};
use comrs::metrics;
use numpy::IntoPyArray;
use pyo3::prelude::*;

/// A constellation with Gray labels: point `i` carries the bits of `i`, MSB first.
///
/// The points and their labels are those of OptiCommPy (`GrayMapping`): square QAM
/// as the product of two PAM listed row by row from the top in a "snake" order, PSK
/// on `e^{j2πk/M}`, PAM on `-(M-1), …, -1, 1, …, M-1`, OOK on `{0, 1}`; the `i`-th
/// point of the construction order gets the Gray label `i ^ (i >> 1)`. The grids keep
/// their integer spacing; call `normalized()` for unit mean energy.
///
/// Args:
///     kind (str): `"qam"`, `"psk"`, `"pam"` or `"ook"`.
///     m (int): the order, a power of two (`2` for OOK).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> c = comrs.Constellation("qam", 16).normalized()
///     >>> round(c.energy, 12)
///     1.0
///     >>> bits = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
///     >>> c.demodulate(c.modulate(bits)).tolist()
///     [0, 0, 0, 0, 1, 1, 1, 1]
#[pyclass(name = "Constellation", module = "comrs")]
#[derive(Clone)]
pub struct PyConstellation {
    /// The Rust constellation.
    pub inner: Constellation,
}

#[pymethods]
impl PyConstellation {
    #[new]
    fn new(kind: &str, m: usize) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::new(self::kind(kind)?, m).py()? })
    }

    /// A custom constellation from its points, labelled by their index.
    ///
    /// Args:
    ///     points (array_like): the `M` points (complex), `M` a power of two.
    ///
    /// Returns:
    ///     Constellation: the constellation (kind `"custom"`).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Constellation.from_points([1, -1]).order
    ///     2
    #[staticmethod]
    fn from_points(points: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::from_points(cvec(points)?).py()? })
    }

    /// The points of `Constellation(kind, m)` with a set-partitioning labelling
    /// (Ungerboeck): natural labels for PSK, Fig. 2(a) of Caire, Taricco & Biglieri
    /// (1998) for 16-QAM.
    ///
    /// Args:
    ///     kind (str): `"psk"` or `"qam"` (16-QAM only).
    ///     m (int): the order.
    ///
    /// Returns:
    ///     Constellation: the constellation (kind `"custom"`).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.Constellation.set_partitioning("qam", 16).points[0b1001]
    ///     np.complex128(-3+3j)
    #[staticmethod]
    fn set_partitioning(kind: &str, m: usize) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::set_partitioning(self::kind(kind)?, m).py()? })
    }

    /// A constellation from the name the specifications use (``"16-QAM"``, ``"QPSK"``,
    /// ``"8PSK"``, ``"4-PAM"``, ``"OOK"``; case, dashes and spaces are ignored), with
    /// its integer grid (call `normalized()` for unit energy).
    ///
    /// Args:
    ///     name (str): the name.
    ///
    /// Returns:
    ///     Constellation: the constellation.
    ///
    /// Raises:
    ///     ValueError: for an unknown family or an invalid order.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> c = comrs.Constellation.from_name("16-QAM")
    ///     >>> c.kind, c.order
    ///     ('qam', 16)
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        Ok(PyConstellation { inner: Constellation::from_name(name).py()? })
    }

    /// str: The family, `"qam"`, `"psk"`, `"pam"`, `"ook"` or `"custom"`.
    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().name()
    }

    /// int: The number of points `M`.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// int: Bits per symbol, `log2 M`.
    #[getter]
    fn bits_per_symbol(&self) -> usize {
        self.inner.bits_per_symbol()
    }

    /// numpy.ndarray: The points, in label order (a copy, complex128).
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.points().to_vec())
    }

    /// float: Mean energy of the points under uniform probabilities.
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy()
    }

    /// Mean energy of the points under the probabilities `px`.
    ///
    /// Args:
    ///     px (array_like): the probabilities of the points.
    ///
    /// Returns:
    ///     float: `Σ p_i |s_i|²`.
    fn energy_with(&self, px: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.inner.energy_with(&crate::fvec(px)?))
    }

    /// The same constellation scaled to unit mean energy.
    ///
    /// Returns:
    ///     Constellation: the normalised constellation.
    fn normalized(&self) -> Self {
        PyConstellation { inner: self.inner.normalized() }
    }

    /// The same constellation with every point multiplied by `s`.
    ///
    /// Args:
    ///     s (float): the scale.
    ///
    /// Returns:
    ///     Constellation: the scaled constellation.
    fn scaled(&self, s: f64) -> Self {
        PyConstellation { inner: self.inner.scaled(s) }
    }

    /// Symbol indices of a bit sequence (`log2 M` bits per symbol, MSB first).
    ///
    /// Args:
    ///     bits (array_like): the bits, a multiple of `log2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the indices (int64).
    ///
    /// Raises:
    ///     ValueError: if the length is not a multiple of `log2 M`.
    fn indices<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let idx = self.inner.indices(&bvec(bits)?).py()?;
        Ok(iarr(py, idx))
    }

    /// Symbols of a bit sequence (OptiCommPy `modulateGray`).
    ///
    /// Args:
    ///     bits (array_like): the bits, a multiple of `log2 M`.
    ///
    /// Returns:
    ///     numpy.ndarray: one symbol per `log2 M` bits (complex128).
    fn modulate<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&bvec(bits)?).py()?))
    }

    /// Symbols of an index sequence.
    ///
    /// Args:
    ///     indices (array_like): the indices, below `M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the points (complex128).
    fn symbols<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let idx = uvec(indices)?;
        if let Some(bad) = idx.iter().find(|&&i| i >= self.inner.order()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("index {bad} out of range")));
        }
        Ok(carr(py, self.inner.symbols(&idx)))
    }

    /// Index of the nearest point of every sample (minimum-distance decisions).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the indices (int64).
    fn decide<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let z = cvec(z)?;
        Ok(iarr(py, self.inner.decide(&z)))
    }

    /// The nearest point of every sample.
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the points (complex128).
    fn project<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.project(&cvec(z)?)))
    }

    /// Hard-decision bits of every sample (OptiCommPy `demodulateGray`).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: `log2 M` bits per sample (uint8).
    fn demodulate<'py>(&self, py: Python<'py>, z: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.demodulate(&cvec(z)?).into_pyarray(py))
    }

    /// Bit log-likelihood ratios `ln p(b=0|y) - ln p(b=1|y)`, `log2 M` per sample.
    ///
    /// The likelihood of point `s` is `exp(-|y - s|²/σ²)` (OptiCommPy `calcLLR`).
    ///
    /// Args:
    ///     z (array_like): received samples.
    ///     noise_var (float): complex noise variance `E|w|²` (for a real constellation
    ///         over a real channel, twice the variance of the real noise).
    ///     px (array_like, optional): prior probabilities of the points.
    ///     max_log (bool): max-log approximation instead of log-sum-exp.
    ///
    /// Returns:
    ///     numpy.ndarray: `log2 M` LLRs per sample, MSB first (float64).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> c = comrs.Constellation("qam", 4).normalized()
    ///     >>> llr = c.llr(c.modulate([0, 1]), 0.1)
    ///     >>> bool(llr[0] > 0 and llr[1] < 0)
    ///     True
    #[pyo3(signature = (z, noise_var, *, px = None, max_log = false))]
    fn llr<'py>(
        &self,
        py: Python<'py>,
        z: &Bound<'py, PyAny>,
        noise_var: f64,
        px: Option<&Bound<'py, PyAny>>,
        max_log: bool,
    ) -> PyResult<FArr<'py>> {
        let z = cvec(z)?;
        let px = px.map(crate::fvec).transpose()?;
        let c = &self.inner;
        let v = py.detach(|| c.llr(&z, noise_var, px.as_deref(), max_log)).py()?;
        Ok(farr(py, v))
    }

    fn __len__(&self) -> usize {
        self.inner.order()
    }

    fn __repr__(&self) -> String {
        format!("comrs.Constellation('{}', {}, energy={:.6})", self.kind(), self.inner.order(), self.inner.energy())
    }
}

/// The reflected binary Gray code of `n_bits` bits: `g[i] = i ^ (i >> 1)`.
///
/// Args:
///     n_bits (int): the number of bits.
///
/// Returns:
///     numpy.ndarray: the `2^n_bits` code words (int64).
///
/// Raises:
///     ValueError: for more than 24 bits (more than 16 777 216 code words).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.gray_code(2).tolist()
///     [0, 1, 3, 2]
#[pyfunction]
fn gray_code(py: Python<'_>, n_bits: u32) -> PyResult<IArr<'_>> {
    if n_bits > MAX_GRAY_BITS {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_bits must be at most {MAX_GRAY_BITS}, got {n_bits}"
        )));
    }
    Ok(iarr(py, modulation::gray_code(n_bits)))
}

/// The largest Gray code that `gray_code` builds (`2^n_bits` words are allocated: 40 bits
/// would abort the process on the allocation).
const MAX_GRAY_BITS: u32 = 24;

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
/// Examples:
///     >>> import comrs
///     >>> int(comrs.prbs(7, 127).sum())
///     64
#[pyfunction]
#[pyo3(signature = (order, length, *, seed = 1))]
fn prbs(py: Python<'_>, order: u32, length: i64, seed: u64) -> PyResult<BArr<'_>> {
    Ok(sources::prbs(order, count(length, "length")?, seed).py()?.into_pyarray(py))
}

/// Maxwell–Boltzmann probabilities `p(x) ∝ exp(-λ|x|²)` of the points.
///
/// Args:
///     constellation (Constellation): the alphabet.
///     lam (float): the shaping parameter `λ` (0 for uniform).
///
/// Returns:
///     numpy.ndarray: one probability per point, summing to one (float64).
#[pyfunction]
fn maxwell_boltzmann<'py>(py: Python<'py>, constellation: &PyConstellation, lam: f64) -> FArr<'py> {
    farr(py, sources::maxwell_boltzmann(&constellation.inner, lam))
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
#[pyfunction]
#[pyo3(signature = (symbols, constellation, *, tx = None))]
fn evm(symbols: &Bound<'_, PyAny>, constellation: &PyConstellation, tx: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    let tx = tx.map(cvec).transpose()?;
    metrics::evm(&cvec(symbols)?, &constellation.inner, tx.as_deref()).py()
}

/// The entropy ``H = -Σ p log2 p`` of a distribution (bits): the information per symbol of
/// a shaped source.
///
/// Args:
///     px (array_like): the probabilities.
///
/// Returns:
///     float: the entropy (bits).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.entropy([0.25, 0.25, 0.25, 0.25])
///     2.0
#[pyfunction]
fn entropy(px: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(metrics::entropy(&crate::fvec(px)?))
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
#[pyfunction]
#[pyo3(signature = (rx, tx, constellation, *, px = None))]
fn monte_carlo_gmi(
    rx: &Bound<'_, PyAny>,
    tx: &Bound<'_, PyAny>,
    constellation: &PyConstellation,
    px: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, f64)> {
    let px = px.map(crate::fvec).transpose()?;
    metrics::monte_carlo_gmi(&cvec(rx)?, &cvec(tx)?, &constellation.inner, px.as_deref()).py()
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
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_ber_gray(4, 0.0, "qam"), 6)
///     0.07865
///     >>> bool(comrs.theory_ber_gray(8, 12.0, "qam", mq=4) < comrs.theory_ber_gray(64, 12.0))
///     True
#[pyfunction]
#[pyo3(signature = (m, ebn0_db, kind = "qam", *, mq = None))]
fn theory_ber_gray(m: usize, ebn0_db: &Bound<'_, PyAny>, kind: &str, mq: Option<usize>) -> PyResult<Py<PyAny>> {
    use comrs::mapping::theory as th;
    match (kind.to_ascii_lowercase().as_str(), mq) {
        ("pam", None) => map_real(ebn0_db, |e| th::ber_gray_pam(m, e).py()),
        ("qam", None) => map_real(ebn0_db, |e| th::ber_gray_qam(m, e).py()),
        ("qam", Some(j)) => map_real(ebn0_db, |e| th::ber_gray_rect_qam(m, j, e).py()),
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
///     >>> bool(comrs.theory_ber_rayleigh_mrc(10.0, 4) < 1e-5)
///     True
#[pyfunction]
#[pyo3(signature = (ebn0_db, branches = 1))]
fn theory_ber_rayleigh_mrc(ebn0_db: &Bound<'_, PyAny>, branches: usize) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| theory::ber_rayleigh_mrc(comrs::math::db2lin(e), branches).py())
}

/// Density of the phase error of a carrier in circular Gaussian noise (S. O. Rice 1948;
/// eq. (8) of Ahmed, Thompson and Zeidler, MILCOM 2007).
///
/// Args:
///     theta (float | array_like): phase error (rad), in `(-π, π]`.
///     snr (float): `A²/E|w|²` (linear).
///
/// Returns:
///     float | numpy.ndarray: the density at every `theta`.
///
/// Examples:
///     >>> import comrs, math
///     >>> round(comrs.theory_phase_pdf(1.0, 0.0) * 2 * math.pi, 12)
///     1.0
#[pyfunction]
fn theory_phase_pdf(theta: &Bound<'_, PyAny>, snr: f64) -> PyResult<Py<PyAny>> {
    map_real(theta, |t| Ok(comrs::mapping::theory::phase_pdf(t, snr)))
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
/// Examples:
///     >>> import comrs
///     >>> round(comrs.theory_mi(4, 30.0), 6)
///     2.0
#[pyfunction]
#[pyo3(signature = (m, snr_db, kind = "qam", *, px = None, fading = "awgn", points = 40))]
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
#[pyfunction]
#[pyo3(signature = (m, snr_db, kind = "qam", *, px = None, fading = "awgn", points = 40))]
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

fn fading(name: &str) -> PyResult<Fading> {
    name.parse().py()
}

/// The CM capacity (bits per channel use) of a constellation with uniform inputs over
/// AWGN or Rayleigh fading known at the receiver (Caire, Taricco & Biglieri 1998, eq.
/// (10)): `theory_mi` of a constellation object.
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
/// Examples:
///     >>> import comrs
///     >>> round(comrs.cm_capacity(comrs.Constellation("qam", 4), 30.0), 6)
///     2.0
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, fading = "awgn", *, points = 16))]
fn cm_capacity(
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
/// `theory_gmi` of a constellation object.
///
/// Args:
///     constellation (Constellation): the labelled points.
///     snr_db (float | array_like): `E_s/σ²` (dB).
///     fading (str): `"awgn"` or `"rayleigh"`.
///     points (int): quadrature nodes per dimension.
///
/// Returns:
///     float | numpy.ndarray: the capacity at every SNR.
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, fading = "awgn", *, points = 16))]
fn bicm_capacity(
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
/// Examples:
///     >>> import comrs, math
///     >>> r0 = comrs.cm_cutoff_rate(comrs.Constellation("psk", 2), 0.0)
///     >>> round(r0 - (1 - math.log2(1 + math.exp(-1))), 12)
///     0.0
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, fading = "awgn"))]
fn cm_cutoff_rate(constellation: &PyConstellation, snr_db: &Bound<'_, PyAny>, fading: &str) -> PyResult<Py<PyAny>> {
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
#[pyfunction]
#[pyo3(signature = (constellation, snr_db, fading = "awgn", *, points = 16))]
fn bicm_cutoff_rate(
    constellation: &PyConstellation,
    snr_db: &Bound<'_, PyAny>,
    fading: &str,
    points: usize,
) -> PyResult<Py<PyAny>> {
    let f = self::fading(fading)?;
    map_real(snr_db, |s| bicm::bicm_cutoff_rate(&constellation.inner, s, f, points).py())
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

/// Register sources, modulation and metrics.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConstellation>()?;
    for f in [
        wrap_pyfunction!(gray_code, m)?,
        wrap_pyfunction!(bits, m)?,
        wrap_pyfunction!(prbs, m)?,
        wrap_pyfunction!(maxwell_boltzmann, m)?,
        wrap_pyfunction!(symbols, m)?,
        wrap_pyfunction!(ber, m)?,
        wrap_pyfunction!(snr_db, m)?,
        wrap_pyfunction!(fast_ber_calc, m)?,
        wrap_pyfunction!(evm, m)?,
        wrap_pyfunction!(monte_carlo_gmi, m)?,
        wrap_pyfunction!(entropy, m)?,
        wrap_pyfunction!(monte_carlo_mi, m)?,
        wrap_pyfunction!(theory_ber, m)?,
        wrap_pyfunction!(theory_ser, m)?,
        wrap_pyfunction!(theory_ber_gray, m)?,
        wrap_pyfunction!(theory_ber_rayleigh_mrc, m)?,
        wrap_pyfunction!(theory_phase_pdf, m)?,
        wrap_pyfunction!(theory_mi, m)?,
        wrap_pyfunction!(theory_gmi, m)?,
        wrap_pyfunction!(cm_capacity, m)?,
        wrap_pyfunction!(bicm_capacity, m)?,
        wrap_pyfunction!(cm_cutoff_rate, m)?,
        wrap_pyfunction!(bicm_cutoff_rate, m)?,
        wrap_pyfunction!(information_rates, m)?,
        wrap_pyfunction!(posterior_rates, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}

//! Continuous-phase modulation (CPFSK, MSK, GMSK) and its detectors.

use super::*;

/// The modulation index `p/q` of a string `"p/q"` or a pair `(p, q)`.
fn modulation_index(h: &Bound<'_, PyAny>) -> PyResult<(u32, u32)> {
    if let Ok(s) = h.extract::<String>() {
        let h: ModulationIndex = s.parse().py()?;
        return Ok((h.num, h.den));
    }
    h.extract::<(u32, u32)>()
        .map_err(|_| PyValueError::new_err("h must be a fraction \"p/q\" or a pair of integers (p, q)"))
}

/// A continuous-phase modulator (CPFSK, MSK, GMSK, raised-cosine CPM) and its
/// detectors.
///
/// An `M`-ary symbol sequence `a_k ∈ {±1, ±3, …, ±(M-1)}` is carried by the phase of a
/// unit-envelope signal `s(t) = exp(jφ(t))`, `φ(t) = 2πh Σ_k a_k q(t - kT)`, where
/// `h = p/q` is the modulation index and `q(t)` the integral of a frequency pulse of
/// `L` symbols and area 1/2 (Proakis & Salehi, *Digital Communications*, 5th ed.,
/// Sec. 3.3-2). The pulses: `"rec"` (rectangular; 1REC is CPFSK and MSK is 1REC with
/// `h = 1/2`), `"rc"` (raised cosine) and `"gmsk"` (the Gaussian pulse of Murota &
/// Hirade, *IEEE Trans. Commun.* 29(7), 1981, of normalised bandwidth `bt`, truncated
/// to `L` symbols). The trellis has `2q M^(L-1)` states.
///
/// Symbols are **indices** `i ∈ {0, …, M-1}`, the level being `a = 2i - (M-1)` (see
/// :meth:`levels`): for binary CPM the bits themselves are the indices, the bit 1
/// sending the level +1. With MSK, :func:`msk_precode` turns information bits into the
/// indices whose signal is OQPSK with half-sinusoid pulses, and :func:`msk_bits` reads
/// them back from the phase states of :meth:`ml_detect_path`, with the BER of BPSK.
///
/// Args:
///     h (str | tuple[int, int]): modulation index, `"p/q"` or `(p, q)` with
///         `0 < p < q ≤ 64` (`"1/2"` by default).
///     pulse (str): `"rec"`, `"rc"` or `"gmsk"`.
///     length (int): pulse length `L` in symbols (1 to 8).
///     bt (float): bandwidth–time product of the Gaussian pulse.
///     sps (int): samples per symbol (at least 2).
///     order (int): alphabet size `M`, 2 or 4.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> msk = comrs.Cpm("1/2", "rec", sps=4)
///     >>> a = np.array([1, 0, 0, 1, 1, 0])
///     >>> s = msk.modulate(a)
///     >>> np.allclose(np.abs(s), 1.0)
///     True
///     >>> msk.ml_detect(s).tolist()
///     [1, 0, 0, 1, 1, 0]
///     >>> comrs.Cpm.gmsk(0.3, length=3, sps=8).n_states
///     16
#[pyclass(name = "Cpm", module = "comrs")]
pub struct PyCpm {
    inner: Cpm,
}

impl PyCpm {
    fn build(cfg: CpmConfig) -> PyResult<Self> {
        Ok(PyCpm { inner: cfg.build().py()? })
    }
}

#[pymethods]
impl PyCpm {
    #[new]
    #[pyo3(signature = (h=None, pulse="rec", *, length=1, bt=0.3, sps=4, order=2))]
    fn new(
        h: Option<&Bound<'_, PyAny>>,
        pulse: &str,
        length: usize,
        bt: f64,
        sps: usize,
        order: usize,
    ) -> PyResult<Self> {
        let (h_num, h_den) = h.map_or(Ok((1, 2)), modulation_index)?;
        let pulse = FrequencyPulse::from_name(pulse, bt).py()?;
        PyCpm::build(CpmConfig { order, h_num, h_den, pulse, length, sps })
    }

    /// MSK: binary 1REC with `h = 1/2`.
    ///
    /// Args:
    ///     sps (int): samples per symbol.
    ///
    /// Returns:
    ///     Cpm: the modulator.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    #[staticmethod]
    #[pyo3(signature = (sps=4))]
    fn msk(sps: usize) -> PyResult<Self> {
        PyCpm::build(CpmConfig::msk(sps))
    }

    /// GMSK: binary Gaussian pulse of bandwidth `bt` over `length` symbols, `h = 1/2`.
    ///
    /// Args:
    ///     bt (float): bandwidth–time product `BT` of the Gaussian filter.
    ///     length (int): pulse length `L` in symbols.
    ///     sps (int): samples per symbol.
    ///
    /// Returns:
    ///     Cpm: the modulator.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    #[staticmethod]
    #[pyo3(signature = (bt=0.3, *, length=3, sps=4))]
    fn gmsk(bt: f64, length: usize, sps: usize) -> PyResult<Self> {
        PyCpm::build(CpmConfig::gmsk(bt, length, sps))
    }

    /// float: The modulation index `h = p/q` (in lowest terms).
    #[getter]
    fn h(&self) -> f64 {
        self.inner.h()
    }

    /// tuple[int, int]: The numerator and the denominator `(p, q)` of the modulation
    /// index.
    #[getter]
    fn h_fraction(&self) -> (u32, u32) {
        let c = self.inner.config();
        (c.h_num, c.h_den)
    }

    /// int: The alphabet size `M`.
    #[getter]
    fn order(&self) -> usize {
        self.inner.config().order
    }

    /// str: The pulse name, `"rec"`, `"rc"` or `"gmsk"`.
    #[getter]
    fn pulse(&self) -> &'static str {
        self.inner.config().pulse.name()
    }

    /// float | None: The bandwidth–time product of the Gaussian pulse (`None` for
    /// another pulse).
    #[getter]
    fn bt(&self) -> Option<f64> {
        match self.inner.config().pulse {
            FrequencyPulse::Gaussian { bt } => Some(bt),
            _ => None,
        }
    }

    /// int: The pulse length `L` in symbols.
    #[getter]
    fn length(&self) -> usize {
        self.inner.config().length
    }

    /// int: Samples per symbol.
    #[getter]
    fn sps(&self) -> usize {
        self.inner.config().sps
    }

    /// int: The number of terminal phases, `2q` (the trellis visits `q` of them when
    /// `p` is even).
    #[getter]
    fn n_phase_states(&self) -> usize {
        self.inner.n_phase_states()
    }

    /// int: The number of trellis states, `2q M^(L-1)`.
    #[getter]
    fn n_states(&self) -> usize {
        self.inner.n_states()
    }

    /// numpy.ndarray: The terminal phases of the trellis (rad), `kπ/q` for `k < 2q`.
    #[getter]
    fn phase_states<'py>(&self, py: Python<'py>) -> FArr<'py> {
        let q = f64::from(self.inner.config().h_den);
        farr(py, (0..self.inner.n_phase_states()).map(|k| std::f64::consts::PI * k as f64 / q).collect())
    }

    /// numpy.ndarray: The tabulated phase response `q(kT/sps)` for `k < L·sps`.
    #[getter]
    fn phase_response<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.phase_response().to_vec())
    }

    /// The symbol levels `a = 2i - (M-1)` of symbol indices (for binary CPM, the bits
    /// `b` give the levels `2b - 1`).
    ///
    /// Args:
    ///     indices (array_like): symbol indices below `M`.
    ///
    /// Returns:
    ///     numpy.ndarray: the levels `±1, ±3, …` (float64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn levels<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        let m = self.inner.config().order;
        let idx = uvec(indices)?;
        if idx.iter().any(|&i| i >= m) {
            return Err(PyValueError::new_err(format!("symbol indices must be below {m}")));
        }
        Ok(farr(py, idx.into_iter().map(|i| self.inner.symbol_value(i)).collect()))
    }

    /// The phase `φ(t)` at every sample, the terminal phase reduced modulo `2π`.
    ///
    /// Args:
    ///     symbols (array_like): symbol indices below `M`.
    ///
    /// Returns:
    ///     numpy.ndarray: `sps` phase samples (rad) per symbol.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn phase<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.phase(&uvec(symbols)?).py()?))
    }

    /// The unit-envelope signal `exp(jφ(t))` at `sps` samples per symbol.
    ///
    /// Args:
    ///     symbols (array_like): symbol indices below `M` (for binary CPM, the bits).
    ///
    /// Returns:
    ///     numpy.ndarray: `sps · len(symbols)` complex samples.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&uvec(symbols)?).py()?))
    }

    /// Coherent maximum-likelihood sequence detection (Viterbi algorithm over the phase
    /// trellis), from the initial state of the modulator to any state.
    ///
    /// Args:
    ///     samples (array_like): a whole number of symbols of received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the symbol indices of the most likely sequence (int64).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn ml_detect<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let r = cvec(samples)?;
        Ok(iarr(py, py.detach(|| self.inner.ml_detect(&r)).py()?))
    }

    /// The maximum-likelihood sequence with its terminal phase states.
    ///
    /// Args:
    ///     samples (array_like): a whole number of symbols of received samples.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: the symbol indices, and the terminal phase states
    ///     in units of `π/q` (modulo `2q`) at the start of every interval and after
    ///     the last one (one more value than symbols).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn ml_detect_path<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<(IArr<'py>, IArr<'py>)> {
        let r = cvec(samples)?;
        let path = py.detach(|| self.inner.ml_detect_path(&r)).py()?;
        Ok((iarr(py, path.symbols), iarr(py, path.phase_states)))
    }

    /// The quadrature detector of MSK: the boundary bits `d_n` of the equivalent OQPSK
    /// signal (the information bits of :func:`msk_precode`), each decided from the
    /// correlation with a half-sinusoid over two symbols, with the BER of antipodal
    /// signalling over AWGN (Proakis & Salehi, Sec. 3.3-2).
    ///
    /// Args:
    ///     samples (array_like): a whole number of symbols of an MSK signal.
    ///
    /// Returns:
    ///     numpy.ndarray: the bits (uint8).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn msk_detect<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.msk_detect(&cvec(samples)?).py()?.into_pyarray(py))
    }

    /// The one-symbol differential phase detector of binary CPM: after an optional
    /// Gaussian low-pass predetection filter of normalised 3 dB bandwidth `bt`, the
    /// sign of `Im(r_n r_{n-1}*)` sampled once per symbol decides `a_n` (Simon & Wang,
    /// *IEEE Trans. Veh. Technol.* 33(4), 1984). Exact for MSK without noise; the
    /// partial response of GMSK closes the eye.
    ///
    /// Args:
    ///     samples (array_like): a whole number of symbols of received samples.
    ///     bt (float, optional): receive bandwidth–time product (no filter by default;
    ///         about 0.5 is the best choice for MSK over AWGN).
    ///
    /// Returns:
    ///     numpy.ndarray: the symbol indices (int64; for binary CPM, the bits).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    #[pyo3(signature = (samples, *, bt=None))]
    fn differential_detect<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
        bt: Option<f64>,
    ) -> PyResult<IArr<'py>> {
        let r = cvec(samples)?;
        Ok(iarr(py, py.detach(|| self.inner.differential_detect(&r, bt)).py()?))
    }

    /// The normalised minimum squared Euclidean distance over the pairs of sequences
    /// that diverge at `t = 0` and merge within `max_len` symbols (the bound `d_B²` of
    /// Anderson, Aulin & Sundberg, *Digital Phase Modulation*, 1986, ch. 3; Aulin &
    /// Sundberg, *IEEE Trans. Commun.* 29(3), 1981, for full response): antipodal
    /// signalling and MSK have `d² = 2`, and the asymptotic gain over BPSK is
    /// `10 log10(d²/2)` dB.
    ///
    /// Args:
    ///     max_len (int): the longest merge considered (the search is exhaustive,
    ///         `(2M-1)^max_len` sequences).
    ///
    /// Returns:
    ///     float: `d²` (infinite if no sequence merges within `max_len` symbols).
    ///
    /// Raises:
    ///     ValueError: if `(2M-1)^max_len` exceeds 2^20 sequences.
    #[pyo3(signature = (*, max_len=3))]
    fn min_distance2(&self, py: Python<'_>, max_len: usize) -> PyResult<f64> {
        py.detach(|| self.inner.min_distance2(max_len)).py()
    }

    fn __repr__(&self) -> String {
        let c = self.inner.config();
        let pulse = match c.pulse {
            FrequencyPulse::Gaussian { bt } => format!("\"gmsk\", bt={bt}"),
            _ => format!("{:?}", self.pulse()),
        };
        format!(
            "Cpm(h=\"{}/{}\", pulse={pulse}, length={}, sps={}, order={})",
            c.h_num, c.h_den, c.length, c.sps, c.order
        )
    }
}

/// The differential precoder of MSK: the bits `b_1 … b_N` give the boundary signs
/// `d_n = 2b_n - 1` (`d_0 = +1`) and the CPM symbols `a_n = d_n d_{n+1}`, as indices
/// `(a_n + 1)/2`, so that the signal is OQPSK with half-sinusoid pulses carrying `d_n`
/// (Proakis & Salehi, Sec. 3.3-2); the ML detector then has the BER of BPSK.
///
/// Args:
///     bits (array_like): the information bits.
///
/// Returns:
///     numpy.ndarray: the symbol indices for :meth:`Cpm.modulate` (int64).
///
/// Raises:
///     ValueError: for bits that are not 0 or 1, or not a 1-D array.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.msk_precode([1, 1, 0, 0, 1]).tolist()
///     [1, 1, 0, 1, 0]
#[pyfunction]
fn msk_precode<'py>(py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
    Ok(iarr(py, cpm::msk_precode(&bvec(bits)?)))
}

/// The information bits of MSK from the terminal phase states of a survivor path
/// (the second array of :meth:`Cpm.ml_detect_path`, in units of `π/2`): the inverse of
/// :func:`msk_precode`.
///
/// Args:
///     phase_states (array_like): `N + 1` phase states.
///
/// Returns:
///     numpy.ndarray: the `N` bits (uint8).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comrs
///     >>> msk = comrs.Cpm.msk(4)
///     >>> bits = [1, 0, 1, 1, 0, 0, 0, 1]
///     >>> _, states = msk.ml_detect_path(msk.modulate(comrs.msk_precode(bits)))
///     >>> comrs.msk_bits(states).tolist() == bits
///     True
#[pyfunction]
fn msk_bits<'py>(py: Python<'py>, phase_states: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
    Ok(cpm::msk_bits(&uvec(phase_states)?).into_pyarray(py))
}

/// The power spectral density of MSK at the normalised frequency `fT`, unit power and
/// `T = 1` (Proakis & Salehi, Sec. 3.4-3):
/// `S(f) = (16/π²) [cos(2πfT) / (1 - 16 f²T²)]²`, finite at `fT = ±1/4`, with nulls at
/// `fT = ±3/4, ±5/4, …` and a 3 dB bandwidth of about `0.59/T`.
///
/// Args:
///     ft (float | array_like): normalised frequencies `fT`.
///
/// Returns:
///     float | numpy.ndarray: the density, which integrates to one over all frequencies.
///
/// Raises:
///     ValueError: for values that are not real numbers.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.msk_psd(0.0), 6)
///     1.621139
///     >>> comrs.msk_psd([0.25, 0.75]).round(6).tolist()
///     [1.0, 0.0]
#[pyfunction]
fn msk_psd(ft: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ft, |f| Ok(cpm::msk_psd(f)))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCpm>()?;
    m.add_function(wrap_pyfunction!(msk_precode, m)?)?;
    m.add_function(wrap_pyfunction!(msk_bits, m)?)?;
    m.add_function(wrap_pyfunction!(msk_psd, m)?)?;
    Ok(())
}

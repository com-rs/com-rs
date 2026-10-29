//! Constant-envelope waveforms: continuous-phase modulation (CPFSK, MSK, GMSK),
//! constant-envelope OFDM and chirp spread spectrum.

use crate::comm::PyConstellation;
use crate::kind;
use crate::{BArr, CArr, FArr, IArr, IntoPy, bvec, carr, cvec, farr, fvec, iarr, map_real, uvec};
use comrs::math::C64;
use comrs::waveform::ce_ofdm::{CeOfdm, Receiver, Unwrap, gamp::Gamp, theory};
use comrs::waveform::cpm::{self, Cpm, CpmConfig, FrequencyPulse};
use comrs::waveform::css::{Css, Waveform, theory as css_theory};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The modulation index `p/q` of a string `"p/q"` or a pair `(p, q)`.
fn modulation_index(h: &Bound<'_, PyAny>) -> PyResult<(u32, u32)> {
    if let Ok(s) = h.extract::<String>() {
        return s
            .split_once('/')
            .and_then(|(p, q)| Some((p.trim().parse::<u32>().ok()?, q.trim().parse::<u32>().ok()?)))
            .ok_or_else(|| PyValueError::new_err(format!("h must be a fraction \"p/q\", got {s:?}")));
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
    #[pyo3(signature = (*, max_len=3))]
    fn min_distance2(&self, py: Python<'_>, max_len: usize) -> f64 {
        py.detach(|| self.inner.min_distance2(max_len))
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

/// A constant-envelope OFDM (CE-OFDM) modulator and phase demodulator (Thompson et al.
/// 2008).
///
/// After Thompson, Ahmed, Proakis, Zeidler & Geile, "Constant envelope OFDM," *IEEE
/// Trans. Commun.* 56(8), 2008.
///
/// A block carries `K` data symbols `d` (unit mean energy) on the subcarriers
/// `1..K` of a DFT of size `N = 2J(K+1)`, `J` the oversampling factor; the real OFDM
/// signal `x = (F d + F* d*)/√(2K)` has a unit variance and the samples are
/// `s_n = exp(j2πh x_n)`, so that the phase has the standard deviation `2πh`, `h`
/// being the modulation index (Sec. II). An optional cyclic prefix of `N_cp` samples
/// is prepended. The receiver (Sec. III, Fig. 1) takes the phase of the samples,
/// unwrapped along the stream or not, removes the prefix and projects it on the data
/// bins: `d̂ = G ψ/(2πh)`.
///
/// **SNR convention.** `E_s/N_0` is per data symbol: a block of energy `N + N_cp`
/// carries `K` symbols, so the noise variance per sample is
/// `σ² = (N + N_cp)/(K E_s/N_0)` (:meth:`noise_variance`; `sps = (N + N_cp)/K` samples
/// per data symbol). Above the threshold of the
/// phase demodulator (a carrier-to-noise ratio near 10 dB, Sec. III-B), every data
/// subcarrier sees the SNR `(2πh)² E_s/N_0 N/(N + N_cp)` (:meth:`snr`), and the error
/// rates are those of the constellation over AWGN at that SNR (:meth:`ser`,
/// :meth:`ber`). For `M`-PAM on every real subcarrier this is eq. (15) of the paper,
/// `BER = 2(M-1)/(M log₂M) Q(2πh √(6 log₂M E_b/N_0 / (M²-1)))`, the
/// nearest-neighbour term of the Gray BER; :meth:`ber` uses the exact Gray BER of the
/// square QAM of order `M²` whose real and imaginary parts are the two PAM symbols
/// (within 1 % of eq. (15) once the BER is under 1e-2).
///
/// Args:
///     n_data (int): data subcarriers `K` per block.
///     h (float): modulation index (phase standard deviation `2πh`).
///     oversampling (int): `J`, the DFT has `N = 2J(K+1)` points.
///     n_cp (int): cyclic prefix (samples).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> ce = comrs.CeOfdm(16, h=0.1, oversampling=2)
///     >>> c = comrs.Constellation("qam", 16).normalized()
///     >>> _, d = comrs.symbols(c, 64, comrs.Rng(1))
///     >>> s = ce.modulate(d)
///     >>> s.size == 4 * ce.n_fft and np.allclose(np.abs(s), 1.0)
///     True
///     >>> np.allclose(ce.demodulate(s), d)
///     True
#[pyclass(name = "CeOfdm", module = "comrs")]
pub struct PyCeOfdm {
    inner: CeOfdm,
}

fn unwrap(name: &str) -> PyResult<Unwrap> {
    name.parse::<Unwrap>().py()
}

/// The receiver of a name (`arctangent`, `linear`, `enhanced`), the arctangent one with
/// the given unwrapping.
fn receiver(name: &str, unwrap: &str) -> PyResult<Receiver> {
    Ok(match name.parse::<Receiver>().py()? {
        Receiver::Arctangent(_) => Receiver::Arctangent(self::unwrap(unwrap)?),
        other => other,
    })
}

#[pymethods]
impl PyCeOfdm {
    #[new]
    #[pyo3(signature = (n_data, h, *, oversampling=4, n_cp=0))]
    fn new(n_data: usize, h: f64, oversampling: usize, n_cp: usize) -> PyResult<Self> {
        let inner = CeOfdm { n_data, oversampling, h, n_cp };
        // validate through the theory, which checks the parameters
        theory::snr(&inner, 0.0).py()?;
        Ok(PyCeOfdm { inner })
    }

    /// int: Data subcarriers `K` per block.
    #[getter]
    fn n_data(&self) -> usize {
        self.inner.n_data
    }

    /// float: The modulation index `h`.
    #[getter]
    fn h(&self) -> f64 {
        self.inner.h
    }

    /// int: The oversampling factor `J`.
    #[getter]
    fn oversampling(&self) -> usize {
        self.inner.oversampling
    }

    /// int: The cyclic prefix `N_cp` (samples).
    #[getter]
    fn n_cp(&self) -> usize {
        self.inner.n_cp
    }

    /// int: The DFT size `N = 2J(K+1)`.
    #[getter]
    fn n_fft(&self) -> usize {
        self.inner.n_fft()
    }

    /// int: Samples per block, `N + N_cp`.
    #[getter]
    fn block_len(&self) -> usize {
        self.inner.block_len()
    }

    /// float: Samples per data symbol, `(N + N_cp)/K`.
    #[getter]
    fn sps(&self) -> f64 {
        self.inner.samples_per_symbol()
    }

    /// The noise variance per sample for an `E_s/N_0` per data symbol and a unit
    /// envelope, `σ² = (N + N_cp)/(K E_s/N_0)`.
    ///
    /// Args:
    ///     esn0_db (float | array_like): `E_s/N_0` (dB) per data symbol.
    ///
    /// Returns:
    ///     float | numpy.ndarray: `σ²` (for `Rng.complex_noise`).
    fn noise_variance(&self, esn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        map_real(esn0_db, |e| Ok(self.inner.noise_variance(e)))
    }

    /// The real OFDM signal `x = (F d + F* d*)/√(2K)` of every block (no prefix).
    ///
    /// Args:
    ///     symbols (array_like): a multiple of `K` data symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: `N` real samples per block.
    fn ofdm_signal<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.ofdm_signal(&cvec(symbols)?).py()?))
    }

    /// The transmitted samples `exp(j2πh x)` with the cyclic prefix of every block.
    ///
    /// Args:
    ///     symbols (array_like): a multiple of `K` data symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: `N + N_cp` unit-envelope samples per block.
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&cvec(symbols)?).py()?))
    }

    /// The symbol estimates of every block.
    ///
    /// Args:
    ///     samples (array_like): a multiple of `N + N_cp` received samples.
    ///     unwrap (str): for the arctangent receiver, `"none"` (the principal phase, wraps
    ///         are errors) or `"stream"` (the phase unwrapped along the stream, the phase
    ///         unwrapper of the paper).
    ///     receiver (str): `"arctangent"` (`d̂ = G ψ/(2πh)`, Thompson et al. 2008),
    ///         `"linear"` (`Im r`) or `"enhanced"` (`Im r (2 - Re r)`), the low-complexity
    ///         receivers of Ahmed and Zeidler (2015), divided by their Bussgang gain.
    ///
    /// Returns:
    ///     numpy.ndarray: `K` complex estimates per block.
    #[pyo3(signature = (samples, *, unwrap="none", receiver="arctangent"))]
    fn demodulate<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
        unwrap: &str,
        receiver: &str,
    ) -> PyResult<CArr<'py>> {
        let r = self::receiver(receiver, unwrap)?;
        Ok(carr(py, self.inner.demodulate_with(&cvec(samples)?, r).py()?))
    }

    /// The posterior means of the symbols of every block by the GAMP receiver of López
    /// Morales, Dinis and García Armada (2022), near the optimum where the phase leaves
    /// `(-π, π]`.
    ///
    /// Args:
    ///     samples (array_like): a multiple of `N + N_cp` received samples.
    ///     noise_var (float): `σ²` per sample (:meth:`noise_variance`).
    ///     constellation (Constellation): the symbols as transmitted (square QAM, PAM or
    ///         BPSK).
    ///     iterations (int): iterations per block.
    ///     damping (float): damping factor in `(0, 1]`.
    ///     points (int): points of the quadrature of the output step.
    ///
    /// Returns:
    ///     numpy.ndarray: `K` complex estimates per block.
    #[pyo3(signature = (samples, noise_var, constellation, *, iterations=50, damping=0.5, points=64))]
    #[allow(clippy::too_many_arguments, reason = "the keyword arguments of the Python method")]
    fn gamp<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
        noise_var: f64,
        constellation: &PyConstellation,
        iterations: usize,
        damping: f64,
        points: usize,
    ) -> PyResult<CArr<'py>> {
        let g = Gamp { iterations, damping, points };
        Ok(carr(py, g.detect(&self.inner, &cvec(samples)?, &constellation.inner, noise_var).py()?))
    }

    /// The phase of the receiver before the DFT, prefix included.
    ///
    /// Args:
    ///     samples (array_like): received samples.
    ///     unwrap (str): `"none"` or `"stream"`, as :meth:`demodulate`.
    ///
    /// Returns:
    ///     numpy.ndarray: one phase (rad) per sample.
    #[pyo3(signature = (samples, *, unwrap="none"))]
    fn received_phase<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>, unwrap: &str) -> PyResult<FArr<'py>> {
        let u = self::unwrap(unwrap)?;
        Ok(farr(py, self.inner.received_phase(&cvec(samples)?, u)))
    }

    /// The SNR of every data subcarrier in the linear regime (linear scale),
    /// `(2πh)² E_s/N_0 N/(N + N_cp)`.
    ///
    /// Args:
    ///     esn0_db (float | array_like): `E_s/N_0` (dB) per data symbol.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the SNR (linear).
    fn snr(&self, esn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        map_real(esn0_db, |e| theory::snr(&self.inner, e).py())
    }

    /// The SER of the linear regime: that of a constellation of `m` points over AWGN
    /// at :meth:`snr` (:func:`theory_ser`).
    ///
    /// Args:
    ///     m (int): constellation order.
    ///     esn0_db (float | array_like): `E_s/N_0` (dB) per data symbol.
    ///     kind (str): `"qam"`, `"psk"` or `"pam"`.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the SER.
    #[pyo3(signature = (m, esn0_db, kind="qam"))]
    fn ser(&self, m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
        let k = self::kind(kind)?;
        map_real(esn0_db, |e| theory::ser(&self.inner, k, m, e).py())
    }

    /// The BER of the linear regime, Gray mapping: the exact Gray BER of a square QAM
    /// (eq. (15) of Thompson et al. with its further neighbours), else
    /// :func:`theory_ber`.
    ///
    /// Args:
    ///     m (int): constellation order.
    ///     esn0_db (float | array_like): `E_s/N_0` (dB) per data symbol.
    ///     kind (str): `"qam"`, `"psk"` or `"pam"`.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the BER.
    #[pyo3(signature = (m, esn0_db, kind="qam"))]
    fn ber(&self, m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str) -> PyResult<Py<PyAny>> {
        let k = self::kind(kind)?;
        map_real(esn0_db, |e| theory::ber(&self.inner, k, m, e).py())
    }

    /// The expected periodogram `E|S_k|²/N` of a transmitted block under the Gaussian
    /// model (Guerreiro et al. 2015): the bins average to 1, bin 0 holds the carrier line.
    ///
    /// Returns:
    ///     numpy.ndarray: `N` values, in the order of the DFT.
    fn psd<'py>(&self, py: Python<'py>) -> PyResult<FArr<'py>> {
        Ok(farr(py, theory::psd(&self.inner).py()?))
    }

    /// The SNR of every data subcarrier with the receiver: flat for the arctangent receiver
    /// above its threshold, Bussgang's analysis for the linear receivers.
    ///
    /// Args:
    ///     esn0_db (float): `E_s/N_0` (dB) per data symbol.
    ///     receiver (str): `"arctangent"`, `"linear"` or `"enhanced"`.
    ///
    /// Returns:
    ///     numpy.ndarray: `K` SNRs (linear).
    #[pyo3(signature = (esn0_db, receiver="arctangent"))]
    fn subcarrier_snr<'py>(&self, py: Python<'py>, esn0_db: f64, receiver: &str) -> PyResult<FArr<'py>> {
        let r = self::receiver(receiver, "none")?;
        Ok(farr(py, theory::subcarrier_snr(&self.inner, r, esn0_db).py()?))
    }

    /// The BER of a Gray-mapped constellation with the receiver, averaged over the
    /// subcarriers of :meth:`subcarrier_snr`.
    ///
    /// Args:
    ///     m (int): constellation order.
    ///     esn0_db (float | array_like): `E_s/N_0` (dB) per data symbol.
    ///     kind (str): `"qam"`, `"psk"` or `"pam"`.
    ///     receiver (str): `"arctangent"`, `"linear"` or `"enhanced"`.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the BER.
    #[pyo3(signature = (m, esn0_db, kind="qam", receiver="arctangent"))]
    fn ber_receiver(&self, m: usize, esn0_db: &Bound<'_, PyAny>, kind: &str, receiver: &str) -> PyResult<Py<PyAny>> {
        let (k, r) = (self::kind(kind)?, self::receiver(receiver, "none")?);
        map_real(esn0_db, |e| theory::ber_receiver(&self.inner, r, k, m, e).py())
    }

    /// The change of the estimates of the arctangent receiver caused by a cycle slip of
    /// `+2π` from sample `start` (after the prefix) to the end of the block (Ahmed et al.
    /// 2012, eqs. (11)–(14)).
    ///
    /// Args:
    ///     start (int): first sample of the slip, `0 ≤ start < N`.
    ///
    /// Returns:
    ///     numpy.ndarray: `K` complex changes.
    fn cycle_slip_response<'py>(&self, py: Python<'py>, start: usize) -> PyResult<CArr<'py>> {
        Ok(carr(py, theory::cycle_slip_response(&self.inner, start).py()?))
    }

    fn __repr__(&self) -> String {
        let c = &self.inner;
        format!("CeOfdm(n_data={}, h={}, oversampling={}, n_cp={})", c.n_data, c.h, c.oversampling, c.n_cp)
    }
}

/// The powers `e^{-φ²} φ^{2i}/i!` of the intermodulation products of a phase modulator
/// driven by a Gaussian signal of standard deviation `φ = 2πh` (Guerreiro et al. 2015,
/// eqs. (12)–(14); Pacheco and Hatzinakos 2006, eq. (2)): the carrier line, the useful
/// term, then the products of order `i`.
///
/// Args:
///     two_pi_h (float): `2πh`.
///     n (int): number of powers.
///
/// Returns:
///     numpy.ndarray: `P_0 … P_{n-1}`.
///
/// Examples:
///     >>> import comrs, math
///     >>> round(float(comrs.ce_ofdm_imp_powers(1.0, 3)[0]), 6) == round(math.exp(-1), 6)
///     True
#[pyfunction]
fn ce_ofdm_imp_powers<'py>(py: Python<'py>, two_pi_h: f64, n: usize) -> FArr<'py> {
    farr(py, theory::imp_powers(two_pi_h, n))
}

/// The optimum asymptotic gain `G` of CE-OFDM over linear OFDM in AWGN, `(2πh)²`, or
/// `(2πh)² erf(θ_max/(√2 · 2πh))` with the phase clipped to `±θ_max` (Guerreiro et al.
/// 2015, eq. (20); 2016, eq. (7)).
///
/// Args:
///     two_pi_h (float): `2πh`.
///     theta_max (float | None): phase clipping (rad).
///
/// Returns:
///     float: `G` (linear).
///
/// Examples:
///     >>> import comrs, math
///     >>> round(10 * math.log10(comrs.ce_ofdm_optimum_gain(1.5)), 2)
///     3.52
#[pyfunction]
#[pyo3(signature = (two_pi_h, theta_max=None))]
fn ce_ofdm_optimum_gain(two_pi_h: f64, theta_max: Option<f64>) -> f64 {
    theory::optimum_gain(two_pi_h, theta_max)
}

/// The optimum asymptotic BER of QPSK CE-OFDM in AWGN, `Q(√(G 2E_b/N_0))` (Guerreiro et
/// al. 2016, eq. (8)).
///
/// Args:
///     two_pi_h (float): `2πh`.
///     ebn0_db (float | array_like): `E_b/N_0` (dB).
///
/// Returns:
///     float | numpy.ndarray: the BER.
#[pyfunction]
fn ce_ofdm_ber_optimum(two_pi_h: f64, ebn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(theory::ber_optimum(two_pi_h, e)))
}

/// The occupied bandwidth `B_y/(2B_m)` of a carrier phase-modulated by a Gaussian message
/// of standard deviation `sigma` with a flat spectrum (Pacheco and Hatzinakos 2006,
/// eqs. (2)–(5), Fig. 1), the carrier line removed.
///
/// Args:
///     sigma (float): standard deviation of the phase (rad).
///     fraction (float): fraction of the power in the band.
///
/// Returns:
///     float: `B_y/(2B_m)`.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.pm_occupied_bandwidth(1.1), 1)
///     2.0
#[pyfunction]
#[pyo3(signature = (sigma, fraction=0.99))]
fn pm_occupied_bandwidth(sigma: f64, fraction: f64) -> PyResult<f64> {
    theory::occupied_bandwidth(sigma, fraction).py()
}

/// Chirp spread spectrum, the frequency-shift chirp modulation of LoRa (Chiani and
/// Elzanaty 2019; Afisiadis et al. 2020): `M = 2^SF` chirps of `M` chips, a
/// noncoherent DFT receiver. LoRa is a trademark of Semtech; this models the published
/// descriptions for research.
///
/// Args:
///     sf (int): spreading factor.
///     oversampling (int): samples per chip.
///     waveform (str): `"lora"`, or for comparison `"vangelista"` (the discrete chirp of
///         Vangelista 2017) and `"fsk"` (one sample per chip).
///
/// Examples:
///     >>> import comrs
///     >>> css = comrs.Css(7)
///     >>> s = css.modulate([0, 5, 127])
///     >>> s.size, css.demodulate(s).tolist()
///     (384, [0, 5, 127])
#[pyclass(name = "Css", module = "comrs")]
pub struct PyCss {
    inner: Css,
}

#[pymethods]
impl PyCss {
    #[new]
    #[pyo3(signature = (sf, *, oversampling=1, waveform="lora"))]
    fn new(sf: u32, oversampling: usize, waveform: &str) -> PyResult<Self> {
        let inner = Css { sf, oversampling, waveform: waveform.parse::<Waveform>().py()? };
        inner.symbol(0).py()?;
        Ok(PyCss { inner })
    }

    /// int: The spreading factor.
    #[getter]
    fn sf(&self) -> u32 {
        self.inner.sf
    }

    /// int: The number of symbols and of chips per symbol, `M = 2^SF`.
    #[getter]
    fn m(&self) -> usize {
        self.inner.m()
    }

    /// int: Samples per symbol, `J M`.
    #[getter]
    fn symbol_len(&self) -> usize {
        self.inner.symbol_len()
    }

    /// The samples of one symbol.
    ///
    /// Args:
    ///     a (int): the symbol, `0 ≤ a < M`.
    ///
    /// Returns:
    ///     numpy.ndarray: `J M` unit-envelope samples.
    fn symbol<'py>(&self, py: Python<'py>, a: usize) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.symbol(a).py()?))
    }

    /// The samples of a sequence of symbols.
    ///
    /// Args:
    ///     symbols (array_like): integers in `[0, M)`.
    ///
    /// Returns:
    ///     numpy.ndarray: `J M` samples per symbol.
    fn modulate<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&uvec(symbols)?).py()?))
    }

    /// The energy of the `M` candidates of every received symbol (squared magnitudes of
    /// the DFT of the dechirped samples).
    ///
    /// Args:
    ///     samples (array_like): a multiple of `J M` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: `M` energies per symbol.
    fn energies<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.energies(&cvec(samples)?).py()?))
    }

    /// The noncoherent decisions.
    ///
    /// Args:
    ///     samples (array_like): a multiple of `J M` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: one symbol per `J M` samples.
    fn demodulate<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(iarr(py, self.inner.demodulate(&cvec(samples)?).py()?))
    }

    /// One symbol of a same-SF interferer offset by `tau` chips: the end of its symbol
    /// `s1`, then its symbol `s2` (Afisiadis et al. 2020, eq. (21)).
    ///
    /// Args:
    ///     s1 (int): the symbol that ends in the window.
    ///     s2 (int): the symbol that starts `tau` chips in.
    ///     tau (float): offset in chips, `0 ≤ tau < M`.
    ///
    /// Returns:
    ///     numpy.ndarray: `M` samples.
    fn interferer<'py>(&self, py: Python<'py>, s1: usize, s2: usize, tau: f64) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.interferer(s1, s2, tau).py()?))
    }

    fn __repr__(&self) -> String {
        let c = &self.inner;
        format!("Css(sf={}, oversampling={}, waveform={:?})", c.sf, c.oversampling, c.waveform)
    }
}

/// The SER of the noncoherent CSS receiver over AWGN at the SNR per sample (dB), exact
/// (Afisiadis et al. 2020, eq. (16)); `approximation="rice"` or `"gumbel"` gives their
/// eqs. (17) and (18).
///
/// Args:
///     sf (int): spreading factor.
///     snr_db (float | array_like): SNR per sample (dB).
///     approximation (str | None): `None`, `"rice"` or `"gumbel"`.
///
/// Returns:
///     float | numpy.ndarray: the SER.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.css_ser(7, -8.0), 5)
///     0.00161
#[pyfunction]
#[pyo3(signature = (sf, snr_db, approximation=None))]
fn css_ser(sf: u32, snr_db: &Bound<'_, PyAny>, approximation: Option<&str>) -> PyResult<Py<PyAny>> {
    let f: fn(u32, f64) -> comrs::Result<f64> = match approximation {
        None => css_theory::ser,
        Some("rice") => css_theory::ser_rice,
        Some("gumbel") => css_theory::ser_gumbel,
        Some(other) => {
            return Err(PyValueError::new_err(format!("approximation must be rice or gumbel, got {other:?}")));
        }
    };
    map_real(snr_db, |s| f(sf, s).py())
}

/// The BER of the noncoherent CSS receiver over AWGN, `SER M/(2(M - 1))`.
///
/// Args:
///     sf (int): spreading factor.
///     snr_db (float | array_like): SNR per sample (dB).
///
/// Returns:
///     float | numpy.ndarray: the BER.
#[pyfunction]
fn css_ber(sf: u32, snr_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(snr_db, |s| css_theory::ber(sf, s).py())
}

/// The cross-correlation of two continuous-time LoRa waveforms (Chiani and Elzanaty 2019,
/// eq. (7)).
///
/// Args:
///     sf (int): spreading factor.
///     l (int): first symbol.
///     m (int): second symbol.
///
/// Returns:
///     complex: `C(l, m)`.
#[pyfunction]
fn css_cross_correlation(sf: u32, l: usize, m: usize) -> PyResult<C64> {
    css_theory::cross_correlation(sf, l, m).py()
}

/// The largest real cross-correlation of the continuous-time waveforms (Chiani and
/// Elzanaty 2019, Table I).
///
/// Args:
///     sf (int): spreading factor.
///
/// Returns:
///     float: `max Re C(l, m)` over `l ≠ m`.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.css_max_cross_correlation(7), 3)
///     0.045
#[pyfunction]
fn css_max_cross_correlation(sf: u32) -> PyResult<f64> {
    css_theory::max_cross_correlation(sf).py()
}

/// The continuous part of the power spectrum of LoRa, `G^c(f) B`, at `f` in units of `B`
/// (Chiani and Elzanaty 2019, eqs. (20)–(25)).
///
/// Args:
///     sf (int): spreading factor.
///     f (array_like): frequencies in units of `B`.
///
/// Returns:
///     numpy.ndarray: one value per frequency.
#[pyfunction]
fn css_continuous_spectrum<'py>(py: Python<'py>, sf: u32, f: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
    Ok(farr(py, css_theory::continuous_spectrum(sf, &fvec(f)?).py()?))
}

/// The lines of the spectrum of LoRa at `n B/M`, `|n| ≤ n_max` (Chiani and Elzanaty 2019,
/// eq. (21)); they hold `1/M` of the power.
///
/// Args:
///     sf (int): spreading factor.
///     n_max (int): largest index `|n|` of a line.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the frequencies (units of `B`) and powers.
#[pyfunction]
fn css_spectral_lines<'py>(py: Python<'py>, sf: u32, n_max: usize) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let (f, p): (Vec<f64>, Vec<f64>) = css_theory::spectral_lines(sf, n_max).py()?.into_iter().unzip();
    Ok((farr(py, f), farr(py, p)))
}

/// The band (units of `B`) that holds `fraction` of the power of LoRa (Chiani and
/// Elzanaty 2019, Table I, `B99`).
///
/// Args:
///     sf (int): spreading factor.
///     fraction (float): fraction of the power.
///     span (float): half-width (units of `B`) of the computed spectrum.
///
/// Returns:
///     float: the band in units of `B`.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.css_occupied_bandwidth(5), 2)
///     1.18
#[pyfunction]
#[pyo3(signature = (sf, fraction=0.99, span=3.0))]
fn css_occupied_bandwidth(sf: u32, fraction: f64, span: f64) -> PyResult<f64> {
    css_theory::occupied_bandwidth(sf, fraction, span).py()
}

/// The pattern `R_k` that a same-SF interferer leaves after the dechirping (Afisiadis et
/// al. 2020, eq. (25)).
///
/// Args:
///     sf (int): spreading factor.
///     s1 (int): the interfering symbol that ends in the window.
///     s2 (int): the interfering symbol that starts `tau` chips in.
///     tau (float): offset in chips, `0 ≤ tau < M`.
///
/// Returns:
///     numpy.ndarray: `M` complex values.
#[pyfunction]
fn css_interference_pattern<'py>(py: Python<'py>, sf: u32, s1: usize, s2: usize, tau: f64) -> PyResult<CArr<'py>> {
    Ok(carr(py, css_theory::interference_pattern(sf, s1, s2, tau).py()?))
}

/// The approximate SER (or, with `frame`, the frame error rate) under AWGN and a same-SF
/// interferer of power `pi_db` relative to the signal (Afisiadis et al. 2020, eqs.
/// (61)–(67), with the strongest bin at `-round(τ)`).
///
/// Args:
///     sf (int): spreading factor.
///     snr_db (float): SNR per sample (dB).
///     pi_db (float): power of the interferer (dB).
///     frame (int | None): symbols per frame for the frame error rate.
///     eps (float): step of the offset (chips).
///
/// Returns:
///     float: the SER or the FER.
#[pyfunction]
#[pyo3(signature = (sf, snr_db, pi_db, *, frame=None, eps=0.2))]
fn css_ser_interference(sf: u32, snr_db: f64, pi_db: f64, frame: Option<usize>, eps: f64) -> PyResult<f64> {
    match frame {
        None => css_theory::ser_interference(sf, snr_db, pi_db, eps),
        Some(f) => css_theory::fer_interference(sf, snr_db, pi_db, f, eps),
    }
    .py()
}

/// Register the constant-envelope waveform bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCpm>()?;
    m.add_class::<PyCeOfdm>()?;
    m.add_class::<PyCss>()?;
    m.add_function(wrap_pyfunction!(css_ser, m)?)?;
    m.add_function(wrap_pyfunction!(css_ber, m)?)?;
    m.add_function(wrap_pyfunction!(css_cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(css_max_cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(css_continuous_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(css_spectral_lines, m)?)?;
    m.add_function(wrap_pyfunction!(css_occupied_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(css_interference_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(css_ser_interference, m)?)?;
    m.add_function(wrap_pyfunction!(ce_ofdm_imp_powers, m)?)?;
    m.add_function(wrap_pyfunction!(ce_ofdm_optimum_gain, m)?)?;
    m.add_function(wrap_pyfunction!(ce_ofdm_ber_optimum, m)?)?;
    m.add_function(wrap_pyfunction!(pm_occupied_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(msk_precode, m)?)?;
    m.add_function(wrap_pyfunction!(msk_bits, m)?)?;
    m.add_function(wrap_pyfunction!(msk_psd, m)?)?;
    Ok(())
}

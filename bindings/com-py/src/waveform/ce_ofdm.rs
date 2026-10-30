//! Constant-envelope OFDM: the modulator, its receivers and its closed forms.

use super::*;

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
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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
        inner.check().py()?;
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
    ///
    /// Raises:
    ///     ValueError: for values that are not real numbers.
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
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
/// Raises:
///     ValueError: for a negative count.
///
/// Examples:
///     >>> import comrs, math
///     >>> round(float(comrs.ce_ofdm_imp_powers(1.0, 3)[0]), 6) == round(math.exp(-1), 6)
///     True
#[pyfunction]
fn ce_ofdm_imp_powers<'py>(py: Python<'py>, two_pi_h: f64, n: i64) -> PyResult<FArr<'py>> {
    Ok(farr(py, theory::imp_powers(two_pi_h, count(n, "n")?)))
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
///
/// Raises:
///     ValueError: for values that are not real numbers.
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
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCeOfdm>()?;
    m.add_function(wrap_pyfunction!(ce_ofdm_imp_powers, m)?)?;
    m.add_function(wrap_pyfunction!(ce_ofdm_optimum_gain, m)?)?;
    m.add_function(wrap_pyfunction!(ce_ofdm_ber_optimum, m)?)?;
    m.add_function(wrap_pyfunction!(pm_occupied_bandwidth, m)?)?;
    Ok(())
}

//! Chirp spread spectrum (the modulation of LoRa) and its closed forms.

use super::*;

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
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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
        let inner = Css {
            sf,
            oversampling,
            waveform: waveform.parse::<Waveform>().py()?,
            sample_rate: comrs::core::units::Hertz(1.0),
        };
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
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
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
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
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
#[pyfunction]
fn css_spectral_lines<'py>(py: Python<'py>, sf: u32, n_max: i64) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let lines = css_theory::spectral_lines(sf, count(n_max, "n_max")?).py()?;
    let (f, p): (Vec<f64>, Vec<f64>) = lines.into_iter().unzip();
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
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
#[pyfunction]
#[pyo3(signature = (sf, snr_db, pi_db, *, frame=None, eps=0.2))]
fn css_ser_interference(sf: u32, snr_db: f64, pi_db: f64, frame: Option<usize>, eps: f64) -> PyResult<f64> {
    match frame {
        None => css_theory::ser_interference(sf, snr_db, pi_db, eps),
        Some(f) => css_theory::fer_interference(sf, snr_db, pi_db, f, eps),
    }
    .py()
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    Ok(())
}

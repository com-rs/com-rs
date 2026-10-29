//! Filtered multicarrier waveforms (FBMC/OQAM, UFMC) and Bussgang's decomposition of a
//! memoryless amplifier driven by a Gaussian signal.

use crate::{CArr, FArr, IntoPy, carr, cvec, farr, fvec};
use comrs::hardware::amplifier::{Rapp, Saleh};
use comrs::hardware::gaussian_input;
use comrs::waveform::fbmc::{self, Fbmc};
use comrs::waveform::ufmc::Ufmc;
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// FBMC/OQAM (staggered multitone) with the PHYDYAS prototype filter.
///
/// `n_subcarriers` subcarriers carry real symbols every half symbol on pulses of unit
/// energy, orthogonal in the real sense; `oqam_stagger` splits complex symbols into
/// them. The modulator and the demodulator are polyphase (one FFT per half symbol).
///
/// Args:
///     n_subcarriers (int): `M`, even.
///     overlap (int): the overlapping factor `K` of the prototype (2, 3 or 4).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> f = comrs.Fbmc(16, 4)
///     >>> c = np.exp(2j * np.pi * np.arange(16 * 6) / 7)
///     >>> y = f.demodulate(f.modulate(comrs.oqam_stagger(c, 16)), 12)
///     >>> back = comrs.oqam_destagger(y.real, 16)
///     >>> bool(np.max(np.abs(back - c)) < 1e-2)
///     True
#[pyclass(name = "Fbmc", module = "comrs")]
pub struct PyFbmc {
    inner: Fbmc,
}

#[pymethods]
impl PyFbmc {
    #[new]
    #[pyo3(signature = (n_subcarriers, overlap=4))]
    fn new(n_subcarriers: usize, overlap: usize) -> PyResult<Self> {
        Ok(PyFbmc { inner: Fbmc::new(n_subcarriers, overlap).py()? })
    }

    /// int: The number of subcarriers `M`.
    #[getter]
    fn n_subcarriers(&self) -> usize {
        self.inner.subcarriers()
    }

    /// numpy.ndarray: The prototype filter (unit energy, `KM - 1` taps).
    #[getter]
    fn taps<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.taps().to_vec())
    }

    /// The number of samples of `half_symbols` half-symbols.
    ///
    /// Args:
    ///     half_symbols (int): real symbols per subcarrier.
    ///
    /// Returns:
    ///     int: `(n - 1) M / 2 + KM - 1`.
    fn length(&self, half_symbols: usize) -> usize {
        self.inner.len(half_symbols)
    }

    /// Modulate real symbols laid out half-symbol after half-symbol (`a[n * M + m]`).
    ///
    /// Args:
    ///     a (array_like): the real symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: the samples (complex128).
    fn modulate<'py>(&self, py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&fvec(a)?).py()?))
    }

    /// Project the samples on every pulse: the real parts estimate the symbols, the
    /// imaginary parts hold the intrinsic interference.
    ///
    /// Args:
    ///     r (array_like): the received samples.
    ///     half_symbols (int): the number of half-symbols to demodulate.
    ///
    /// Returns:
    ///     numpy.ndarray: `y[n * M + m]` (complex128).
    fn demodulate<'py>(&self, py: Python<'py>, r: &Bound<'py, PyAny>, half_symbols: usize) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.demodulate(&cvec(r)?, half_symbols).py()?))
    }

    /// The intrinsic interference that the symbol `(m + dm, n + dn)` leaves on the
    /// estimate of `(m, n)` (Bouhadda et al. 2014, Table 1).
    ///
    /// Args:
    ///     m (int): subcarrier.
    ///     n (int): half-symbol.
    ///     dm (int): subcarrier offset.
    ///     dn (int): half-symbol offset.
    ///
    /// Returns:
    ///     complex: 1 at the symbol itself, purely imaginary elsewhere.
    fn interference(&self, m: usize, n: usize, dm: isize, dn: isize) -> C64 {
        self.inner.interference(m, n, dm, dn)
    }

    fn __repr__(&self) -> String {
        format!("Fbmc(n_subcarriers={}, overlap={})", self.inner.subcarriers(), self.inner.overlap())
    }
}

/// Split complex symbols (`c[k * M + m]`) into the real symbols of OQAM on twice as
/// many half-symbols.
///
/// Args:
///     c (array_like): complex symbols, whole symbols of `m` subcarriers.
///     m (int): subcarriers.
///
/// Returns:
///     numpy.ndarray: the real symbols (float64).
#[pyfunction]
fn oqam_stagger<'py>(py: Python<'py>, c: &Bound<'py, PyAny>, m: usize) -> PyResult<FArr<'py>> {
    let c = cvec(c)?;
    if m == 0 || c.len() % m != 0 {
        return Err(PyValueError::new_err("whole symbols of m subcarriers"));
    }
    Ok(farr(py, fbmc::oqam_stagger(&c, m)))
}

/// The inverse of `oqam_stagger`.
///
/// Args:
///     a (array_like): real estimates of an even number of half-symbols.
///     m (int): subcarriers.
///
/// Returns:
///     numpy.ndarray: the complex symbols (complex128).
#[pyfunction]
fn oqam_destagger<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, m: usize) -> PyResult<CArr<'py>> {
    let a = fvec(a)?;
    if m == 0 || a.len() % (2 * m) != 0 {
        return Err(PyValueError::new_err("an even number of half-symbols of m subcarriers"));
    }
    Ok(carr(py, fbmc::oqam_destagger(&a, m)))
}

/// Universal filtered multicarrier (Schaich & Wild 2014): OFDM filtered per sub-band by
/// Dolph–Chebyshev filters, no cyclic prefix, `n_fft + filter_len - 1` samples per
/// symbol.
///
/// Args:
///     n_fft (int): grid size `N`.
///     subbands (int): `B`.
///     subband_size (int): subcarriers per sub-band `Q`.
///     filter_len (int): `L`.
///     sidelobe_db (float): attenuation of the sidelobes of the filter (dB).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> u = comrs.Ufmc(64, 4, 12, 16, 40.0)
///     >>> d = np.exp(2j * np.pi * np.arange(96) / 5)
///     >>> np.allclose(u.demodulate(u.modulate(d)), d)
///     True
#[pyclass(name = "Ufmc", module = "comrs")]
pub struct PyUfmc {
    inner: Ufmc,
}

#[pymethods]
impl PyUfmc {
    #[new]
    #[pyo3(signature = (n_fft, subbands, subband_size, filter_len, sidelobe_db=40.0))]
    fn new(n_fft: usize, subbands: usize, subband_size: usize, filter_len: usize, sidelobe_db: f64) -> PyResult<Self> {
        Ok(PyUfmc { inner: Ufmc::new(n_fft, subbands, subband_size, filter_len, sidelobe_db).py()? })
    }

    /// int: Samples per symbol, `N + L - 1`.
    #[getter]
    fn symbol_len(&self) -> usize {
        self.inner.symbol_len()
    }

    /// Modulate `K = B Q` symbols per UFMC symbol.
    ///
    /// Args:
    ///     data (array_like): the symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: the samples (complex128).
    fn modulate<'py>(&self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&cvec(data)?).py()?))
    }

    /// Demodulate whole symbols (zeros to `2N`, DFT, even bins, divided by the filters).
    ///
    /// Args:
    ///     samples (array_like): the received samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the symbols (complex128).
    fn demodulate<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.demodulate(&cvec(samples)?).py()?))
    }
}

/// Bussgang's decomposition `y = K x + d` of a memoryless amplifier driven by a circular
/// complex Gaussian signal: the complex gain and the power of the distortion.
///
/// Args:
///     model (str): `"rapp"` (gain 1, saturation `saturation`, smoothness `p`) or
///         `"saleh"` (`alpha_a`, `beta_a`, `alpha_p`, `beta_p`).
///     power (float): the input power.
///     saturation (float): Rapp's saturation amplitude.
///     p (float): Rapp's smoothness.
///     alpha_a (float): Saleh's AM/AM gain.
///     beta_a (float): Saleh's AM/AM compression.
///     alpha_p (float): Saleh's AM/PM gain (rad).
///     beta_p (float): Saleh's AM/PM compression.
///
/// Returns:
///     tuple[complex, float]: `K` and `σ_d²`.
///
/// Examples:
///     >>> import comrs
///     >>> k, d = comrs.amplifier_bussgang("saleh", 10 ** -0.4, alpha_p=3.141592653589793 / 6)
///     >>> round(k.real, 4), round(k.imag, 4)
///     (0.5917, 0.1106)
#[pyfunction]
#[pyo3(signature = (model, power, *, saturation=1.0, p=1.0, alpha_a=1.0, beta_a=1.0, alpha_p=0.0, beta_p=1.0))]
#[allow(clippy::too_many_arguments, reason = "the parameters of both amplifier models, by keyword")]
fn amplifier_bussgang(
    model: &str,
    power: f64,
    saturation: f64,
    p: f64,
    alpha_a: f64,
    beta_a: f64,
    alpha_p: f64,
    beta_p: f64,
) -> PyResult<(C64, f64)> {
    match model {
        "rapp" => {
            let a = Rapp { gain: 1.0, saturation, smoothness: p };
            Ok(gaussian_input::bussgang(|r| C64::new(a.am_am(r), 0.0), power))
        }
        "saleh" => {
            let a = Saleh { alpha_a, beta_a, alpha_p, beta_p };
            Ok(gaussian_input::bussgang(|r| C64::from_polar(a.am_am(r), a.am_pm(r)), power))
        }
        other => Err(PyValueError::new_err(format!("unknown amplifier {other:?}: rapp, saleh"))),
    }
}

/// The Dolph–Chebyshev window (as `scipy.signal.windows.chebwin`).
///
/// Args:
///     n (int): length.
///     attenuation_db (float): level of the sidelobes below the main lobe (dB).
///
/// Returns:
///     numpy.ndarray: the window, maximum 1 (float64).
#[pyfunction]
fn chebyshev_window(py: Python<'_>, n: usize, attenuation_db: f64) -> FArr<'_> {
    farr(py, comrs::dsp::window::chebyshev(n, attenuation_db))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFbmc>()?;
    m.add_class::<PyUfmc>()?;
    m.add_function(wrap_pyfunction!(oqam_stagger, m)?)?;
    m.add_function(wrap_pyfunction!(oqam_destagger, m)?)?;
    m.add_function(wrap_pyfunction!(amplifier_bussgang, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_window, m)?)?;
    Ok(())
}

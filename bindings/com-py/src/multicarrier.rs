//! Filtered multicarrier waveforms (FBMC/OQAM, UFMC) and Bussgang's decomposition of a
//! memoryless amplifier driven by a Gaussian signal.

use crate::{CArr, FArr, IntoPy, carr, cvec, farr, fvec};
use comrs::dsp::nonlinear;
use comrs::hardware::amplifier::{Model, Rapp, Saleh};
use comrs::waveform::fbmc::{self, Fbmc};
use comrs::waveform::ufmc::Ufmc;
use num_complex::Complex64 as C64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// FBMC/OQAM (staggered multitone) with the PHYDYAS prototype filter.
///
/// `n_fft` subcarriers carry real symbols every half symbol on pulses of unit
/// energy, orthogonal in the real sense; `oqam_stagger` splits complex symbols into
/// them. The modulator and the demodulator are polyphase (one FFT per half symbol).
///
/// Args:
///     n_fft (int): the number of subcarriers `M`, even.
///     overlap (int): the overlapping factor `K` of the prototype (2, 3 or 4).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> f = comrs.Fbmc(16, overlap=4)
///     >>> c = np.exp(2j * np.pi * np.arange(16 * 6) / 7)
///     >>> y = f.demodulate(f.modulate(comrs.oqam_stagger(c, 16)))
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
    #[pyo3(signature = (n_fft, *, overlap=4))]
    fn new(n_fft: usize, overlap: usize) -> PyResult<Self> {
        Ok(PyFbmc { inner: Fbmc::new(n_fft, overlap).py()? })
    }

    /// int: The number of subcarriers `M`.
    #[getter]
    fn n_fft(&self) -> usize {
        self.inner.n_fft()
    }

    /// int: The overlapping factor `K`.
    #[getter]
    fn overlap(&self) -> usize {
        self.inner.overlap()
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
    fn n_samples(&self, half_symbols: usize) -> usize {
        self.inner.n_samples(half_symbols)
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
    ///     r (array_like): the received samples, `n_samples(n)` for `n` half-symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: `y[n * M + m]` (complex128).
    fn demodulate<'py>(&self, py: Python<'py>, r: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.demodulate(&cvec(r)?).py()?))
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
        format!("Fbmc(n_fft={}, overlap={})", self.inner.n_fft(), self.inner.overlap())
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
    Ok(farr(py, fbmc::oqam_stagger(&cvec(c)?, m).py()?))
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
    Ok(carr(py, fbmc::oqam_destagger(&fvec(a)?, m).py()?))
}

/// Universal filtered multicarrier (Schaich & Wild 2014): OFDM filtered per sub-band by
/// Dolph–Chebyshev filters, no cyclic prefix, `n_fft + filter_len - 1` samples per
/// symbol.
///
/// Args:
///     n_fft (int): grid size `N`.
///     n_subbands (int): `B`.
///     subband_size (int): subcarriers per sub-band `Q`.
///     filter_len (int): `L`.
///     sidelobe_db (float): attenuation of the sidelobes of the filter (dB).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> u = comrs.Ufmc(64, 4, 12, 16, sidelobe_db=40.0)
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
    #[pyo3(signature = (n_fft, n_subbands, subband_size, filter_len, *, sidelobe_db=40.0))]
    fn new(
        n_fft: usize,
        n_subbands: usize,
        subband_size: usize,
        filter_len: usize,
        sidelobe_db: f64,
    ) -> PyResult<Self> {
        Ok(PyUfmc { inner: Ufmc::new(n_fft, n_subbands, subband_size, filter_len, sidelobe_db).py()? })
    }

    /// int: Samples per symbol, `N + L - 1`.
    #[getter]
    fn block_len(&self) -> usize {
        self.inner.block_len()
    }

    /// int: Active subcarriers `K = B Q`.
    #[getter]
    fn n_active(&self) -> usize {
        self.inner.n_active()
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
/// The parameters of each model default to those of :func:`pa_rapp` and
/// :func:`pa_saleh` (Rapp with `p = 2`, Saleh's fit of 1981).
///
/// Args:
///     model (str): `"rapp"` (gain 1, `saturation`, `smoothness`) or
///         `"saleh"` (`alpha_a`, `beta_a`, `alpha_p`, `beta_p`).
///     power (float): the input power.
///     saturation (float): Rapp's saturation amplitude.
///     smoothness (float): Rapp's smoothness `p`.
///     alpha_a (float): Saleh's AM/AM gain.
///     beta_a (float): Saleh's AM/AM compression.
///     alpha_p (float): Saleh's AM/PM gain (rad).
///     beta_p (float): Saleh's AM/PM compression.
///
/// Returns:
///     tuple[complex, float]: `K` and `σ_d²`.
///
/// Raises:
///     ValueError: for an unknown model.
///
/// Examples:
///     The TWTA of Bouhadda et al. (2014, Table 3) at 4 dB of input back-off, `φ₀ = π/6`:
///
///     >>> import comrs, math
///     >>> k, d = comrs.amplifier_bussgang("saleh", 10 ** -0.4, alpha_a=1.0, beta_a=1.0,
///     ...                                 alpha_p=math.pi / 6, beta_p=1.0)
///     >>> round(k.real, 4), round(k.imag, 4)
///     (0.5917, 0.1106)
#[pyfunction]
#[pyo3(signature = (
    model,
    power,
    *,
    saturation = Rapp::default().saturation,
    smoothness = Rapp::default().smoothness,
    alpha_a = Saleh::default().alpha_a,
    beta_a = Saleh::default().beta_a,
    alpha_p = Saleh::default().alpha_p,
    beta_p = Saleh::default().beta_p,
))]
#[allow(clippy::too_many_arguments, reason = "the parameters of both amplifier models, by keyword")]
fn amplifier_bussgang(
    model: &str,
    power: f64,
    saturation: f64,
    smoothness: f64,
    alpha_a: f64,
    beta_a: f64,
    alpha_p: f64,
    beta_p: f64,
) -> PyResult<(C64, f64)> {
    let b = match model.parse::<Model>().py()? {
        Model::Rapp => nonlinear::bussgang_of(&Rapp { saturation, smoothness, ..Rapp::default() }, power),
        Model::Saleh => nonlinear::bussgang_of(&Saleh { alpha_a, beta_a, alpha_p, beta_p }, power),
        other => return Err(PyValueError::new_err(format!("no Bussgang decomposition for {other:?}"))),
    };
    Ok((b.gain, b.distortion))
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

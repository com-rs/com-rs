//! Orthogonal time frequency space and the delay–Doppler channel it is designed for.

use crate::rng::PyRng;
use crate::{CArr, IntoPy, carr, cvec};
use comrs::channel::multipath::DelayDopplerModel;
use comrs::waveform::otfs::Otfs;
use num_complex::Complex64 as C64;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

/// Orthogonal time frequency space (Raviteja et al. 2019) with rectangular pulses: the
/// symbols of the `n_fft × n_symbols` delay–Doppler grid (`x[n * M + m]`, delay `m`,
/// Doppler `n`) through the ISFFT and the Heisenberg transform, one cyclic prefix per
/// frame.
///
/// Args:
///     n_fft (int): delay bins `M` (subcarriers).
///     n_symbols (int): Doppler bins `N` (symbols of a frame).
///     n_cp (int): the cyclic prefix of a frame (samples).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> o = comrs.Otfs(16, 8, n_cp=4)
///     >>> x = np.exp(2j * np.pi * np.arange(128) / 7)
///     >>> s = o.modulate(x)
///     >>> s.size == o.block_len, bool(np.allclose(o.demodulate(s), x))
///     (True, True)
#[pyclass(name = "Otfs", module = "comrs")]
pub struct PyOtfs {
    inner: Otfs,
}

#[pymethods]
impl PyOtfs {
    #[new]
    #[pyo3(signature = (n_fft, n_symbols, *, n_cp=0))]
    fn new(n_fft: usize, n_symbols: usize, n_cp: usize) -> PyResult<Self> {
        let inner = Otfs { n_cp, ..Otfs::new(n_fft, n_symbols).py()? };
        inner.check().py()?;
        Ok(PyOtfs { inner })
    }

    /// int: Delay bins `M`.
    #[getter]
    fn n_fft(&self) -> usize {
        self.inner.n_fft
    }

    /// int: Doppler bins `N`.
    #[getter]
    fn n_symbols(&self) -> usize {
        self.inner.n_symbols
    }

    /// int: The cyclic prefix of a frame (samples).
    #[getter]
    fn n_cp(&self) -> usize {
        self.inner.n_cp
    }

    /// int: Samples per frame, `MN` plus the prefix.
    #[getter]
    fn block_len(&self) -> usize {
        self.inner.block_len()
    }

    /// Modulate whole frames of `MN` delay–Doppler symbols.
    ///
    /// Args:
    ///     x (array_like): the symbols, `x[n * M + m]` frame after frame.
    ///
    /// Returns:
    ///     numpy.ndarray: the samples, prefixes included (complex128).
    fn modulate<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.modulate(&cvec(x)?).py()?))
    }

    /// Demodulate whole frames: drop the prefix, DFT along the Doppler axis.
    ///
    /// Args:
    ///     r (array_like): a multiple of `block_len` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the delay–Doppler samples (complex128).
    fn demodulate<'py>(&self, py: Python<'py>, r: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.demodulate(&cvec(r)?).py()?))
    }

    /// The effective `MN × MN` delay–Doppler channel of paths (Raviteja et al. 2019,
    /// Theorem 1, eq. (12)), the prefix phase included.
    ///
    /// Args:
    ///     paths (list[tuple[complex, int, float]]): `(gain, delay in samples, Doppler
    ///         in Hz)`, as `delay_doppler_paths` draws them; the Doppler shifts must be
    ///         whole bins `ν M N / fs`.
    ///     fs (float): the sampling rate (Hz).
    ///
    /// Returns:
    ///     numpy.ndarray: `(MN, MN)` (complex128).
    fn effective_channel<'py>(
        &self,
        py: Python<'py>,
        paths: Vec<(C64, usize, f64)>,
        fs: f64,
    ) -> PyResult<Bound<'py, PyArray2<C64>>> {
        Ok(self.inner.effective_channel(&paths, fs).py()?.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Otfs(n_fft={}, n_symbols={}, n_cp={})", self.inner.n_fft, self.inner.n_symbols, self.inner.n_cp)
    }
}

/// One realisation of a delay–Doppler channel: one path per tap of a power-delay
/// profile, with a Rayleigh gain and a Doppler shift `ν_max cos θ`, `θ` uniform on
/// `[-π, π)` (`DelayDopplerModel` of `com_channel::multipath`).
///
/// Args:
///     profile (str | array_like): a profile name (as `power_delay_profile`) or taps
///         `(n_taps, 2)`, `(delay in s, power in dB)`.
///     max_doppler_hz (float): the largest Doppler shift (Hz).
///     fs (float): the sampling rate (Hz).
///     rng (Rng): the generator.
///
/// Returns:
///     list[tuple[complex, int, float]]: `(gain, delay in samples, Doppler in Hz)` per
///     path.
#[pyfunction]
fn delay_doppler_paths(
    profile: &Bound<'_, PyAny>,
    max_doppler_hz: f64,
    fs: f64,
    mut rng: PyRefMut<'_, PyRng>,
) -> PyResult<Vec<(C64, usize, f64)>> {
    let taps = crate::ofdm_link::profile_or_taps(profile)?;
    let model = DelayDopplerModel::new(&taps, max_doppler_hz, fs).py()?;
    Ok(model.draw(&mut rng.inner).paths().iter().map(|p| (p.gain, p.delay, p.doppler_hz)).collect())
}

/// Register the OTFS bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOtfs>()?;
    m.add_function(wrap_pyfunction!(delay_doppler_paths, m)?)?;
    Ok(())
}

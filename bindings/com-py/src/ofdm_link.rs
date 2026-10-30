//! OFDM links: multipath channels, resource grid, channel estimation and
//! synchronisation.

use crate::ofdm::PyOfdm;
use crate::rng::PyRng;
use crate::{
    CArr, FArr, FMat, IArr, IntoPy, carr, cvec, farr, fvec, iarr, mat, streams_in, streams_out, taps_in, taps_out, uvec,
};
use comrs::channel::multipath::{Profile, Taps, Tdl, rms_delay_spread};
use comrs::core::signal::Signal;
use comrs::core::units::Hertz;
use comrs::waveform::ofdm::estimation::{FrequencyInterpolation, PowerDelayProfile, estimate};
use comrs::waveform::ofdm::grid::{PilotPattern, ResourceGrid};
use comrs::waveform::ofdm::sync;
use ndarray::Array3;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

fn profile_taps(name: &str) -> PyResult<Vec<(f64, f64)>> {
    Ok(name.parse::<Profile>().py()?.taps())
}

/// The taps of a profile given by its name (`"eva"`) or as an `(n, 2)` array of
/// `(delay in s, power in dB)`.
pub(crate) fn profile_or_taps(profile: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    if profile.is_instance_of::<PyString>() { profile_taps(&profile.extract::<String>()?) } else { taps_in(profile) }
}

/// The taps of a power-delay profile of the standards.
///
/// Args:
///     name (str): `"epa"`, `"eva"`, `"etu"` (3GPP TS 36.101, Annex B.2),
///         `"pedestrian_a"` or `"vehicular_a"` (ITU-R M.1225).
///
/// Returns:
///     numpy.ndarray: the taps `(n_taps, 2)`, column 0 the delays (s) and column 1 the
///     powers (dB).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> taps = comrs.power_delay_profile("epa")
///     >>> taps.shape, round(float(taps[-1, 0]) * 1e9)
///     ((7, 2), 410)
#[pyfunction]
fn power_delay_profile<'py>(py: Python<'py>, name: &str) -> PyResult<FMat<'py>> {
    taps_out(py, profile_taps(name)?)
}

/// The RMS delay spread of a power-delay profile.
///
/// Args:
///     taps (array_like): the taps `(n_taps, 2)`, `(delay in s, power in dB)` (as
///         `power_delay_profile` and `cdl_profile` return them).
///
/// Returns:
///     float: the RMS delay spread (s).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.rms_delay_spread(comrs.power_delay_profile("eva")) * 1e9)
///     357
#[pyfunction]
#[pyo3(name = "rms_delay_spread")]
fn py_rms_delay_spread(taps: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(rms_delay_spread(&taps_in(taps)?))
}

/// A tapped-delay-line multipath channel with Clarke/Jakes Doppler (sum of sinusoids of
/// Zheng & Xiao 2003), SISO or MIMO: every tap is Rayleigh with its power, the delays
/// are rounded to the sampling grid and the powers normalised.
///
/// Args:
///     profile (str | array_like): a profile name (see `power_delay_profile`) or taps
///         `(n_taps, 2)` of `(delay in s, power in dB)`.
///     fs (float): sampling rate (Hz).
///     doppler_hz (float): maximum Doppler shift (Hz), 0 for a static realisation.
///     n_rx (int): receive antennas.
///     n_tx (int): transmit antennas.
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> ch = comrs.TdlChannel("eva", 15.36e6)
///     >>> y, taps = ch.fade(np.ones(64), comrs.Rng(1))
///     >>> y.shape, int(taps.delays[-1])
///     ((64,), 39)
#[pyclass(name = "TdlChannel", module = "comrs")]
pub struct PyTdl {
    inner: Tdl,
}

/// The taps of one realisation of a `TdlChannel` or a `CdlChannel`: one tap per
/// delay, with its coefficients at every sample on every link.
#[pyclass(name = "ChannelTaps", module = "comrs")]
pub struct PyTaps {
    inner: Taps,
    n_rx: usize,
    n_tx: usize,
}

impl PyTaps {
    /// Wrap the taps of a realisation between `n_tx` and `n_rx` antennas.
    pub(crate) fn new(inner: Taps, n_rx: usize, n_tx: usize) -> PyTaps {
        PyTaps { inner, n_rx, n_tx }
    }
}

#[pymethods]
impl PyTdl {
    #[new]
    #[pyo3(signature = (profile, fs, *, doppler_hz=0.0, n_rx=1, n_tx=1))]
    fn new(profile: &Bound<'_, PyAny>, fs: f64, doppler_hz: f64, n_rx: usize, n_tx: usize) -> PyResult<Self> {
        let taps = profile_or_taps(profile)?;
        Ok(PyTdl { inner: Tdl::new(&taps, doppler_hz, fs, n_rx, n_tx).py()? })
    }

    /// numpy.ndarray: Delays of the taps (samples, int64).
    #[getter]
    fn delays<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.delays().to_vec())
    }

    /// numpy.ndarray: Normalised powers of the taps (unit sum, float64).
    #[getter]
    fn powers<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.powers().to_vec())
    }

    /// int: Receive antennas.
    #[getter]
    fn n_rx(&self) -> usize {
        self.inner.n_rx
    }

    /// int: Transmit antennas.
    #[getter]
    fn n_tx(&self) -> usize {
        self.inner.n_tx
    }

    /// Apply the channel.
    ///
    /// Args:
    ///     x (array_like): the transmitted streams, `(n,)` for one transmit antenna or
    ///         `(n_tx, n)`.
    ///     rng (Rng): the generator of the realisation.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, ChannelTaps]: the received streams (`(n,)` or
    ///     `(n_rx, n)`) and the taps of the realisation.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn fade(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        mut rng: PyRefMut<'_, PyRng>,
    ) -> PyResult<(Py<PyAny>, PyTaps)> {
        let (rows, one_d) = streams_in(x)?;
        let s = Signal::from_streams(&rows, Hertz(self.inner.sample_rate())).py()?;
        let (y, taps) = self.inner.fade(&s, &mut rng.inner).py()?;
        Ok((streams_out(py, &y, one_d), PyTaps::new(taps, self.inner.n_rx, self.inner.n_tx)))
    }

    fn __repr__(&self) -> String {
        format!("TdlChannel({} taps, n_rx={}, n_tx={})", self.inner.delays().len(), self.inner.n_rx, self.inner.n_tx)
    }
}

#[pymethods]
impl PyTaps {
    /// numpy.ndarray: Delays of the taps (samples, int64).
    #[getter]
    fn delays<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.delays().to_vec())
    }

    /// int: Number of samples of the realisation.
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    /// The coefficients of every tap.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n_taps, n_samples, n_rx, n_tx)` (complex128):
    ///     `coefficients[l, n, r, t]` is tap `l` at sample `n` between transmit antenna
    ///     `t` and receive antenna `r`.
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn coefficients(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (n_taps, n, n_rx, n_tx) = (self.inner.delays().len(), self.inner.n_samples(), self.n_rx, self.n_tx);
        let a =
            ndarray::Array4::from_shape_fn((n_taps, n, n_rx, n_tx), |(l, k, r, t)| self.inner.coefficient(l, k, r, t));
        Ok(a.into_pyarray(py).into_any().unbind())
    }

    /// The frequency response at one sample on the bins of an FFT.
    ///
    /// Args:
    ///     n (int): the sample.
    ///     bins (array_like): the FFT bins (as `Ofdm.bins`).
    ///     fft_len (int): the size of the transform.
    ///
    /// Returns:
    ///     numpy.ndarray: `(len(bins),)` for SISO, `(len(bins), n_rx, n_tx)` otherwise
    ///     (complex128).
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
    fn frequency_response(
        &self,
        py: Python<'_>,
        n: usize,
        bins: &Bound<'_, PyAny>,
        fft_len: usize,
    ) -> PyResult<Py<PyAny>> {
        let bins = uvec(bins)?;
        let h = self.inner.frequency_response(n, &bins, fft_len);
        if self.n_rx == 1 && self.n_tx == 1 {
            return Ok(carr(py, h).into_any().unbind());
        }
        let a = Array3::from_shape_vec((bins.len(), self.n_rx, self.n_tx), h)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(a.into_pyarray(py).into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "ChannelTaps({} taps, {} samples, n_rx={}, n_tx={})",
            self.inner.delays().len(),
            self.inner.n_samples(),
            self.n_rx,
            self.n_tx
        )
    }
}

/// An OFDM resource grid of `n_subcarriers` × `n_symbols` with pilots (Coleri et al.
/// 2002). The grid is stored symbol after symbol, so that
/// `Ofdm.modulate(grid.map(data))` sends it.
///
/// Args:
///     n_subcarriers (int): subcarriers of the grid.
///     n_symbols (int): OFDM symbols of the grid.
///     pattern (str): `"block"` (every subcarrier of one symbol in `period`), `"comb"`
///         (one subcarrier in `spacing`, every symbol) or `"lattice"` (one subcarrier
///         in `freq` of one symbol in `time`).
///     period (int): symbol period of the block pattern.
///     spacing (int): subcarrier spacing of the comb pattern.
///     freq (int): subcarrier spacing of the lattice pattern.
///     time (int): symbol spacing of the lattice pattern.
///     seed (int): seed of the pilot symbols.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
///
/// Examples:
///     >>> import comrs
///     >>> grid = comrs.ResourceGrid(12, 4, pattern="comb", spacing=4)
///     >>> grid.n_data, grid.pilot_positions().shape
///     (36, (12, 2))
#[pyclass(name = "ResourceGrid", module = "comrs")]
pub struct PyGrid {
    pub(crate) inner: ResourceGrid,
}

#[pymethods]
impl PyGrid {
    #[new]
    #[pyo3(signature = (n_subcarriers, n_symbols, *, pattern="lattice", period=4, spacing=4, freq=3, time=4, seed=0))]
    #[allow(clippy::too_many_arguments, reason = "the grid and the parameters of its pilot pattern, by keyword")]
    fn new(
        n_subcarriers: usize,
        n_symbols: usize,
        pattern: &str,
        period: usize,
        spacing: usize,
        freq: usize,
        time: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let pattern = PilotPattern::from_name(pattern, period, spacing, freq, time).py()?;
        Ok(PyGrid { inner: ResourceGrid { n_subcarriers, n_symbols, pattern, seed } })
    }

    /// int: Data REs per grid.
    #[getter]
    fn n_data(&self) -> usize {
        self.inner.n_data()
    }

    /// int: Subcarriers of the grid.
    #[getter]
    fn n_subcarriers(&self) -> usize {
        self.inner.n_subcarriers
    }

    /// int: OFDM symbols of the grid.
    #[getter]
    fn n_symbols(&self) -> usize {
        self.inner.n_symbols
    }

    /// The pilot resource elements.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n_pilots, 2)` (int64), the subcarrier `k` and the symbol `l`
    ///     of every pilot.
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter (the message names it).
    fn pilot_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
        mat(py, self.inner.pilot_positions().into_iter().map(|(k, l)| vec![k as i64, l as i64]).collect())
    }

    /// Whether a resource element carries a pilot.
    ///
    /// Args:
    ///     k (int): the subcarrier.
    ///     l (int): the OFDM symbol.
    ///
    /// Returns:
    ///     bool: whether `(k, l)` is a pilot.
    fn is_pilot(&self, k: usize, l: usize) -> bool {
        self.inner.is_pilot(k, l)
    }

    /// The grid (symbol after symbol) with `data` and the pilots.
    ///
    /// Args:
    ///     data (array_like): `n_data` data symbols.
    ///
    /// Returns:
    ///     numpy.ndarray: `n_subcarriers · n_symbols` symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn map<'py>(&self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.inner.map(&cvec(data)?).py()?))
    }

    /// The data resource elements of a grid.
    ///
    /// Args:
    ///     grid (array_like): a whole grid, symbol after symbol.
    ///
    /// Returns:
    ///     numpy.ndarray: the `n_data` data symbols (complex128).
    ///
    /// Raises:
    ///     ValueError: for an invalid parameter or input (the message names it).
    fn extract<'py>(&self, py: Python<'py>, grid: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let g = cvec(grid)?;
        if g.len() != self.inner.n_subcarriers * self.inner.n_symbols {
            return Err(PyValueError::new_err("not a whole grid"));
        }
        Ok(carr(py, self.inner.extract(&g).py()?))
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourceGrid({} × {}, {} data REs)",
            self.inner.n_subcarriers,
            self.inner.n_symbols,
            self.inner.n_data()
        )
    }
}

/// Channel estimation on a resource grid: LS at the pilots, then `"linear"`
/// interpolation, or `"lmmse"` across the subcarriers (van de Beek et al. 1995) with
/// the power-delay profile `delays`, `powers`, or the robust uniform profile over
/// `max_delay` samples (Edfors et al. 1998); linear in time.
///
/// Args:
///     grid (ResourceGrid): the grid and its pilots.
///     rx (array_like): the received grid (the demodulated active subcarriers, symbol
///         after symbol).
///     noise_var (float): noise variance per resource element.
///     method (str): `"linear"`, `"lmmse"`, or the interpolations compared by Coleri et
///         al. (2002): `"second_order"`, `"spline"`, `"lowpass"` (MATLAB `interp`) and
///         `"time_domain"` (zero padding of the IDFT, pilots on 0, L, 2L, …).
///     delays (array_like, optional): delays of the profile (samples of the IFFT of
///         `n_fft`).
///     powers (array_like, optional): powers of the profile (linear).
///     max_delay (int, optional): the length of the uniform profile (samples).
///     n_fft (int, optional): the size of the IFFT (needed by `"lmmse"`, with `delays`
///         and `powers` or `max_delay`).
///
/// Returns:
///     numpy.ndarray: the channel estimate at every resource element (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
#[pyo3(signature = (grid, rx, noise_var, method="linear", *, delays=None, powers=None, max_delay=None, n_fft=None))]
#[allow(clippy::too_many_arguments, reason = "the grid, the method and its profile, by keyword")]
fn ofdm_estimate<'py>(
    py: Python<'py>,
    grid: &PyGrid,
    rx: &Bound<'py, PyAny>,
    noise_var: f64,
    method: &str,
    delays: Option<&Bound<'py, PyAny>>,
    powers: Option<&Bound<'py, PyAny>>,
    max_delay: Option<usize>,
    n_fft: Option<usize>,
) -> PyResult<CArr<'py>> {
    let pdp = match (delays, powers, max_delay, n_fft) {
        (None, None, None, None) => None,
        (Some(d), Some(p), None, Some(n)) => Some(PowerDelayProfile::new(fvec(d)?, &fvec(p)?, n).py()?),
        (None, None, Some(m), Some(n)) => Some(PowerDelayProfile::uniform(m, n).py()?),
        _ => {
            return Err(PyValueError::new_err(
                "give delays, powers and n_fft (a measured profile), or max_delay and n_fft (the uniform profile)",
            ));
        }
    };
    let m = FrequencyInterpolation::from_name(method, pdp).py()?;
    let r = cvec(rx)?;
    Ok(carr(py, py.detach(|| estimate(&grid.inner, &r, noise_var, &m)).py()?))
}

/// The preamble of Schmidl & Cox (1997): two identical halves.
///
/// Args:
///     ofdm (Ofdm): the modulator.
///     seed (int): seed of the symbols.
///
/// Returns:
///     numpy.ndarray: one OFDM block with its prefix (complex128).
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
#[pyfunction]
#[pyo3(signature = (ofdm, *, seed=0))]
fn ofdm_preamble<'py>(py: Python<'py>, ofdm: &PyOfdm, seed: u64) -> PyResult<CArr<'py>> {
    Ok(carr(py, sync::preamble(&ofdm.inner, seed).py()?))
}

/// The timing metric of Schmidl & Cox, `|P(d)|²/R(d)²`.
///
/// Args:
///     r (array_like): the received samples.
///     half (int): the length of a half of the preamble (samples).
///
/// Returns:
///     numpy.ndarray: the metric at every delay (float64).
///
/// Raises:
///     ValueError: for an input that is not an array of numbers of the expected shape.
#[pyfunction]
fn schmidl_cox<'py>(py: Python<'py>, r: &Bound<'py, PyAny>, half: usize) -> PyResult<FArr<'py>> {
    Ok(farr(py, sync::schmidl_cox(&cvec(r)?, half)))
}

/// Timing and fractional frequency offset of the preamble of `ofdm` in `r` (Schmidl &
/// Cox 1997).
///
/// Args:
///     r (array_like): the received samples.
///     ofdm (Ofdm): the modulator of the preamble.
///
/// Returns:
///     dict: ``timing`` (the middle of the plateau, inside the cyclic prefix,
///     samples), ``cfo`` (subcarrier spacings), ``peak`` (the metric).
///
/// Raises:
///     ValueError: for an invalid parameter or input (the message names it).
#[pyfunction]
fn ofdm_synchronize<'py>(py: Python<'py>, r: &Bound<'py, PyAny>, ofdm: &PyOfdm) -> PyResult<Bound<'py, PyDict>> {
    let e = sync::synchronize(&cvec(r)?, &ofdm.inner).py()?;
    let d = PyDict::new(py);
    d.set_item("timing", e.timing)?;
    d.set_item("cfo", e.cfo)?;
    d.set_item("peak", e.peak)?;
    Ok(d)
}

/// Register the OFDM link bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTdl>()?;
    m.add_class::<PyTaps>()?;
    m.add_class::<PyGrid>()?;
    m.add_function(wrap_pyfunction!(power_delay_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_rms_delay_spread, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_preamble, m)?)?;
    m.add_function(wrap_pyfunction!(schmidl_cox, m)?)?;
    m.add_function(wrap_pyfunction!(ofdm_synchronize, m)?)?;
    Ok(())
}

//! Clustered delay line channels of 3GPP TR 38.901 (CDL-A to CDL-E).

use crate::ofdm_link::PyTaps;
use crate::rng::PyRng;
use crate::{FArr, FMat, IArr, IntoPy, count, farr, iarr, streams_in, streams_out, taps_out};
use comrs::channel::cdl::{Array, Cdl, Profile};
use comrs::core::signal::Signal;
use comrs::core::units::Hertz;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn profile(name: &str) -> PyResult<Profile> {
    name.parse::<Profile>().py()
}

/// The power-delay profile of a CDL profile of 3GPP TR 38.901 for a desired RMS delay
/// spread: one tap per cluster.
///
/// The five profiles are the Tables 7.7.1-1 to 7.7.1-5 of TR 38.901 V17.0.0: CDL-A,
/// CDL-B and CDL-C without line of sight (23, 23 and 24 clusters), CDL-D and CDL-E
/// with a specular component in the first cluster (13 and 14 clusters). The delays of
/// the tables are normalised to a unit RMS delay spread and scaled by the desired one
/// (Sec. 7.7.3, eq. 7.7-1), ``tau_n = tau_n,model * DS``; the powers are the table
/// entries normalised to a unit sum, the specular row of CDL-D/E added to its cluster.
/// The result is the tap array `TdlChannel` takes, for a CDL profile with Clarke/Jakes
/// fading instead of the ray geometry of `CdlChannel`.
///
/// Args:
///     name (str): `"CDL-A"` to `"CDL-E"` (or `"a"` to `"e"`, case-insensitive).
///     delay_spread_s (float): desired RMS delay spread (s).
///
/// Returns:
///     numpy.ndarray: the taps `(n_clusters, 2)`, column 0 the delays (s) and column 1
///     the powers (dB), as `power_delay_profile`.
///
/// Examples:
///     >>> import comrs
///     >>> taps = comrs.cdl_profile("CDL-C", 100e-9)
///     >>> taps.shape, round(comrs.rms_delay_spread(taps) * 1e9, 1)
///     ((24, 2), 100.0)
#[pyfunction]
fn cdl_profile<'py>(py: Python<'py>, name: &str, delay_spread_s: f64) -> PyResult<FMat<'py>> {
    if delay_spread_s < 0.0 {
        return Err(PyValueError::new_err(format!("negative delay spread {delay_spread_s}")));
    }
    taps_out(py, profile(name)?.pdp(delay_spread_s))
}

/// The Rician K-factor (dB) of the first cluster of a CDL profile.
///
/// Tables 7.7.1-4 and 7.7.1-5 of 3GPP TR 38.901 V17.0.0 give cluster 1 of CDL-D and
/// CDL-E as two rows, a specular (LOS) path and a Laplacian cluster; the K-factor is
/// their power difference, 13.3 dB for CDL-D (−0.2 against −13.5 dB) and 22 dB for
/// CDL-E (−0.03 against −22.03 dB). CDL-A, B and C have no line of sight.
///
/// Args:
///     name (str): `"CDL-A"` to `"CDL-E"`.
///
/// Returns:
///     float | None: the K-factor (dB), `None` without line of sight.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.cdl_k_factor_db("CDL-D"), 6), comrs.cdl_k_factor_db("CDL-A")
///     (13.3, None)
#[pyfunction]
fn cdl_k_factor_db(name: &str) -> PyResult<Option<f64>> {
    Ok(profile(name)?.k_factor_db())
}

/// A clustered delay line channel of 3GPP TR 38.901 (CDL-A to CDL-E) between two
/// uniform linear arrays, with the Doppler of a moving receiver.
///
/// Every cluster of the table is a tap of 20 rays (Table 7.5-3 gives the ray offsets
/// `alpha_m` scaled by the per-cluster angular spreads of the last rows of the table),
/// each ray with its departure and arrival angles, its random initial phases, the
/// cross-polarisation ratio of the table and the phase it accumulates across the
/// elements of the arrays, so the spatial correlation follows from the geometry
/// (Sec. 7.5 step 11, eq. 7.5-22). The delays of the table are scaled by the desired
/// RMS delay spread (Sec. 7.7.3, eq. 7.7-1) and rounded to the sampling grid; the
/// cluster powers are normalised to a unit sum (the specular component of CDL-D/E,
/// eq. 7.5-29, counted in the first cluster), so the mean gain of every link is 1.
/// The receiver moves along the array axis at `velocity_mps`; the Doppler shift of a
/// ray is `v cos(angle) f_c / c`, 0 for a static channel, which is then constant in
/// time.
///
/// Args:
///     profile (str): `"CDL-A"` to `"CDL-E"` (Tables 7.7.1-1 to 7.7.1-5).
///     delay_spread_s (float): desired RMS delay spread (s), Sec. 7.7.3.
///     fs (float): sampling rate (Hz).
///     n_rx (int): elements of the receive array (pairs when `dual_polarized`).
///     n_tx (int): elements of the transmit array (pairs when `dual_polarized`).
///     spacing (float): element spacing of both arrays (wavelengths).
///     velocity_mps (float): speed of the receiver along the array axis (m/s).
///     fc_hz (float): carrier frequency (Hz), the wavelength of the Doppler.
///     dual_polarized (bool): cross-polarised pairs (slants ±45°) instead of vertical
///         elements: the arrays then have `2 n_tx` and `2 n_rx` antennas.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> ch = comrs.CdlChannel("CDL-C", 100e-9, 30.72e6, n_rx=4, n_tx=2)
///     >>> y, taps = ch.fade(np.ones((2, 256)), comrs.Rng(1))
///     >>> y.shape, len(taps.delays), ch.n_rx
///     ((4, 256), 24, 4)
#[pyclass(name = "CdlChannel", module = "comrs")]
pub struct PyCdl {
    inner: Cdl,
}

#[pymethods]
impl PyCdl {
    #[new]
    #[pyo3(signature = (profile, delay_spread_s, fs, *, n_rx=1, n_tx=1, spacing=0.5, velocity_mps=0.0, fc_hz=3.5e9, dual_polarized=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        profile: &str,
        delay_spread_s: f64,
        fs: f64,
        n_rx: usize,
        n_tx: usize,
        spacing: f64,
        velocity_mps: f64,
        fc_hz: f64,
        dual_polarized: bool,
    ) -> PyResult<Self> {
        if spacing <= 0.0 {
            return Err(PyValueError::new_err(format!("the element spacing must be positive, got {spacing}")));
        }
        let array = |n: usize| if dual_polarized { Array::linear_dual(n, spacing) } else { Array::linear(n, spacing) };
        let mut inner = Cdl::new(self::profile(profile)?, delay_spread_s, fs, array(n_tx), array(n_rx)).py()?;
        inner.velocity = [0.0, velocity_mps, 0.0];
        inner.carrier_frequency = fc_hz;
        if velocity_mps != 0.0 && fc_hz <= 0.0 {
            return Err(PyValueError::new_err("the carrier frequency must be positive for a moving receiver"));
        }
        Ok(PyCdl { inner })
    }

    /// numpy.ndarray: Delays of the clusters on the sampling grid (samples, int64), one
    /// tap each.
    #[getter]
    fn delays<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.delays().to_vec())
    }

    /// numpy.ndarray: Normalised powers of the clusters (unit sum, the specular
    /// component of CDL-D/E counted in the first; float64).
    #[getter]
    fn powers<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.powers().to_vec())
    }

    /// int: Receive antennas.
    #[getter]
    fn n_rx(&self) -> usize {
        self.inner.n_rx()
    }

    /// int: Transmit antennas.
    #[getter]
    fn n_tx(&self) -> usize {
        self.inner.n_tx()
    }

    /// float: The maximum Doppler shift `|v| f_c / c` (Hz).
    #[getter]
    fn max_doppler_hz(&self) -> f64 {
        self.inner.max_doppler()
    }

    /// Apply the channel to transmitted streams whose samples before the signal are
    /// zero.
    ///
    /// Args:
    ///     x (array_like): the transmitted streams, `(n,)` for one transmit antenna or
    ///         `(n_tx, n)`.
    ///     rng (Rng): the generator of the ray coupling and phases.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, ChannelTaps]: the received streams (`(n,)` for a SISO
    ///     channel given a 1-D input, `(n_rx, n)` otherwise) and the taps of the
    ///     realisation.
    fn fade(
        &self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        mut rng: PyRefMut<'_, PyRng>,
    ) -> PyResult<(Py<PyAny>, PyTaps)> {
        let (rows, one_d) = streams_in(x)?;
        let s = Signal::from_streams(&rows, Hertz(self.inner.sample_rate())).py()?;
        let (y, taps) = self.inner.fade(&s, &mut rng.inner).py()?;
        Ok((streams_out(py, &y, one_d), PyTaps::new(taps, self.inner.n_rx(), self.inner.n_tx())))
    }

    /// Draw the taps of a realisation without applying them.
    ///
    /// Args:
    ///     n (int): number of samples.
    ///     rng (Rng): the generator.
    ///
    /// Returns:
    ///     ChannelTaps: the taps, whose `coefficients()` give every link at every sample.
    fn draw(&self, n: i64, mut rng: PyRefMut<'_, PyRng>) -> PyResult<PyTaps> {
        let taps = self.inner.draw(count(n, "n")?, &mut rng.inner).py()?;
        Ok(PyTaps::new(taps, self.inner.n_rx(), self.inner.n_tx()))
    }

    fn __repr__(&self) -> String {
        format!(
            "CdlChannel({} taps, n_rx={}, n_tx={}, max_doppler_hz={})",
            self.inner.delays().len(),
            self.inner.n_rx(),
            self.inner.n_tx(),
            self.inner.max_doppler()
        )
    }
}

/// Register the CDL bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCdl>()?;
    m.add_function(wrap_pyfunction!(cdl_profile, m)?)?;
    m.add_function(wrap_pyfunction!(cdl_k_factor_db, m)?)?;
    Ok(())
}

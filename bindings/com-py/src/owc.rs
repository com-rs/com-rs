//! Optical wireless channels: the indoor channel of LED lights and infrared emitters
//! with the reflections of a room, the turbulence, pointing errors and weather of
//! free-space optics, and the PIN-FET receiver of visible-light links.

use crate::rng::PyRng;
use crate::{FArr, IntoPy, farr, fvec, map_real};
use comrs::channel::owc::atmospheric::{self as fso, GammaGamma, PointingError, Turbulence, Wave};
use comrs::channel::owc::indoor::{
    self, Detector, Emitter, Illumination, ImpulseResponse, Reflectance, Resolution, Room,
};
use comrs::hardware::optical::PinFetReceiver;
use num_complex::Complex64 as C64;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A point or a direction `(x, y, z)`.
fn vec3(x: &Bound<'_, PyAny>, what: &str) -> PyResult<[f64; 3]> {
    let v = fvec(x)?;
    v.try_into().map_err(|_| PyValueError::new_err(format!("{what} must have 3 coordinates")))
}

/// `n` points `(n, 3)`, or one `(3,)` repeated `n` times (`n = 0`: as many as given).
fn points(x: &Bound<'_, PyAny>, n: usize, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let a = crate::as_numpy(x, "float64")?;
    let a = a.cast::<numpy::PyArrayDyn<f64>>().map_err(|_| PyValueError::new_err(format!("{what}: real values")))?;
    let ro = a.readonly();
    let v = ro.as_array();
    match v.shape() {
        [3] => Ok(vec![[v[[0]], v[[1]], v[[2]]]; n.max(1)]),
        [m, 3] if n == 0 || *m == n => Ok(v.outer_iter().map(|r| [r[0], r[1], r[2]]).collect()),
        _ => Err(PyValueError::new_err(format!("{what} must be (3,) or (n, 3), n the number of emitters"))),
    }
}

/// The emitters of `positions` `(n, 3)`, `directions` `(3,)` or `(n, 3)`, and a mode and
/// a power each shared or per emitter.
fn emitters(
    positions: &Bound<'_, PyAny>,
    directions: &Bound<'_, PyAny>,
    mode: &Bound<'_, PyAny>,
    power_w: &Bound<'_, PyAny>,
) -> PyResult<Vec<Emitter>> {
    let pos = points(positions, 0, "positions")?;
    let dir = points(directions, pos.len(), "directions")?;
    let per = |x: &Bound<'_, PyAny>, what: &str| -> PyResult<Vec<f64>> {
        let v = fvec(&crate::as_numpy(x, "float64")?.call_method1("reshape", (-1,))?)?;
        match v.len() {
            1 => Ok(vec![v[0]; pos.len()]),
            k if k == pos.len() => Ok(v),
            _ => Err(PyValueError::new_err(format!("{what} must be a scalar or one per emitter"))),
        }
    };
    let (modes, powers) = (per(mode, "mode")?, per(power_w, "power_w")?);
    Ok((0..pos.len()).map(|i| Emitter::new(pos[i], dir[i], modes[i], powers[i])).collect())
}

/// The Lambertian mode `n = -ln 2 / ln cos Φ½` of a source of semi-angle at half power
/// `Φ½` (rad): 1 for 60°, 0.65 for 70°.
///
/// Args:
///     semi_angle_rad (float): the semi-angle at half power, in `(0, π/2)`.
///
/// Returns:
///     float: the mode.
#[pyfunction]
fn lambertian_mode(semi_angle_rad: f64) -> PyResult<f64> {
    indoor::lambertian_mode(semi_angle_rad).py()
}

/// The unit vector of elevation `elevation_rad` (positive up) and azimuth `azimuth_rad`
/// (from the x axis, towards y).
///
/// Args:
///     elevation_rad (float): the elevation (rad), `-π/2` pointing down.
///     azimuth_rad (float): the azimuth (rad).
///
/// Returns:
///     numpy.ndarray: `(3,)` (float64).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> bool(np.allclose(comrs.owc_direction(-np.pi / 2, 0.0), [0, 0, -1]))
///     True
#[pyfunction]
fn owc_direction(py: Python<'_>, elevation_rad: f64, azimuth_rad: f64) -> FArr<'_> {
    farr(py, indoor::direction(elevation_rad, azimuth_rad).to_vec())
}

/// The illuminance (lx) at `point`, on a surface of normal `normal`, of LED chips of
/// centre luminous intensity `intensity_cd` (Komine & Nakagawa 2004, eq. (5)).
///
/// Args:
///     positions (array_like): `(n, 3)` positions of the chips (m).
///     directions (array_like): `(3,)` or `(n, 3)` their axes.
///     mode (float | array_like): the Lambertian mode, shared or per chip.
///     intensity_cd (float): the luminous intensity on the axis (cd).
///     point (array_like): `(3,)` the point lit.
///     normal (array_like): `(3,)` the normal of the surface.
///
/// Returns:
///     float: the illuminance (lx).
#[pyfunction]
fn owc_illuminance(
    positions: &Bound<'_, PyAny>,
    directions: &Bound<'_, PyAny>,
    mode: &Bound<'_, PyAny>,
    intensity_cd: f64,
    point: &Bound<'_, PyAny>,
    normal: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let one = 1.0f64.into_pyobject(positions.py())?.into_any();
    let e = emitters(positions, directions, mode, &one)?;
    Ok(indoor::illuminance_lx(&e, intensity_cd, vec3(point, "point")?, vec3(normal, "normal")?))
}

/// The received optical power of an indoor channel, bin by bin (see
/// `OwcIllumination.impulse_responses`).
#[pyclass(name = "OwcImpulseResponse", module = "comrs")]
#[derive(Clone)]
pub struct PyImpulseResponse {
    inner: ImpulseResponse,
}

#[pymethods]
impl PyImpulseResponse {
    /// float: The width of the time bins (s).
    #[getter]
    fn bin_s(&self) -> f64 {
        self.inner.bin_s
    }

    /// numpy.ndarray: The power received in each bin (W), for the powers of the emitters.
    #[getter]
    fn power_w<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.power_w.clone())
    }

    /// The total received power (W), the DC gain for an emitter of 1 W.
    ///
    /// Returns:
    ///     float: the power (W).
    fn total_w(&self) -> f64 {
        self.inner.total_w()
    }

    /// The start of the first bin that holds power (s); `None` for no power.
    ///
    /// Returns:
    ///     float | None: the time (s).
    fn first_arrival_s(&self) -> Option<f64> {
        self.inner.first_arrival_s()
    }

    /// The frequency response `H(f) = Σ P_i exp(-j2πf(i + ½)Δt)` (Barry et al. 1993,
    /// eq. (15)).
    ///
    /// Args:
    ///     f_hz (array_like): the frequencies (Hz).
    ///
    /// Returns:
    ///     numpy.ndarray: `H(f)` (complex128).
    fn frequency_response<'py>(
        &self,
        py: Python<'py>,
        f_hz: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let f = fvec(&crate::as_numpy(f_hz, "float64")?.call_method1("reshape", (-1,))?)?;
        Ok(f.iter().map(|&x| self.inner.frequency_response(x)).collect::<Vec<_>>().into_pyarray(py))
    }

    /// The −3 dB bandwidth (Hz): the first frequency at which `|H(f)|/H(0)` falls to
    /// `10^(-3/20)`.
    ///
    /// Args:
    ///     step_hz (float): the step of the search (Hz), refined by bisection.
    ///     max_hz (float): the largest frequency searched (Hz).
    ///
    /// Returns:
    ///     float | None: the bandwidth, or None if `|H|` does not fall within `max_hz`.
    #[pyo3(signature = (step_hz=1e5, max_hz=1e9))]
    fn bandwidth_3db_hz(&self, step_hz: f64, max_hz: f64) -> Option<f64> {
        self.inner.bandwidth_3db_hz(step_hz, max_hz)
    }

    /// The power of on–off keying with rectangular pulses of `bit_s` received inside the
    /// bit started at `origin_s` and after it (Komine & Nakagawa 2004, eqs. (17), (19)).
    ///
    /// Args:
    ///     bit_s (float): the duration of a bit (s).
    ///     origin_s (float): the start of the bit (s), usually `first_arrival_s()`.
    ///
    /// Returns:
    ///     tuple[float, float]: `(signal, intersymbol interference)` (W).
    fn bit_split(&self, bit_s: f64, origin_s: f64) -> (f64, f64) {
        self.inner.bit_split(bit_s, origin_s)
    }

    fn __add__(&self, other: &PyImpulseResponse) -> PyResult<PyImpulseResponse> {
        Ok(PyImpulseResponse { inner: self.inner.add(&other.inner).py()? })
    }

    fn __repr__(&self) -> String {
        format!(
            "OwcImpulseResponse(bin_s={:e}, bins={}, total_w={:e})",
            self.inner.bin_s,
            self.inner.power_w.len(),
            self.inner.total_w()
        )
    }
}

/// An empty rectangular room `[0, L] × [0, W] × [0, H]` with Lambertian walls (Barry et
/// al. 1993): north and south are the walls `x = L` and `x = 0`, west and east `y = W`
/// and `y = 0`.
///
/// Args:
///     size_m (array_like): `(L, W, H)` (m).
///     reflectance (float | dict): one reflectivity for every surface, or a dict of
///         `north`, `south`, `east`, `west`, `ceiling`, `floor` (missing ones: 0).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> room = comrs.OwcRoom([5.0, 5.0, 3.0], 0.8)
///     >>> lit = room.illuminate([2.5, 2.5, 3.0], [0, 0, -1], 1.0, 1.0,
///     ...                       bin_s=0.2e-9, partitions=[[25, 25, 15]])
///     >>> h = lit.impulse_responses([0.5, 1.0, 0.0], [0, 0, 1], 1e-4, np.radians(85))
///     >>> round(h[0].total_w() * 1e6, 2)  # the line of sight (Barry et al., Fig. 3)
///     1.23
#[pyclass(name = "OwcRoom", module = "comrs")]
pub struct PyRoom {
    inner: Room,
}

#[pymethods]
impl PyRoom {
    #[new]
    fn new(size_m: &Bound<'_, PyAny>, reflectance: &Bound<'_, PyAny>) -> PyResult<Self> {
        let size = vec3(size_m, "size_m")?;
        let reflectance = if let Ok(d) = reflectance.cast::<PyDict>() {
            let get = |k: &str| -> PyResult<f64> { d.get_item(k)?.map_or(Ok(0.0), |v| v.extract()) };
            for k in d.keys() {
                let k: String = k.extract()?;
                if !["north", "south", "east", "west", "ceiling", "floor"].contains(&k.as_str()) {
                    return Err(PyValueError::new_err(format!("unknown surface {k:?}")));
                }
            }
            Reflectance {
                north: get("north")?,
                south: get("south")?,
                east: get("east")?,
                west: get("west")?,
                ceiling: get("ceiling")?,
                floor: get("floor")?,
            }
        } else {
            Room::uniform(size, reflectance.extract()?).py()?.reflectance
        };
        let inner = Room { size_m: size, reflectance };
        inner.check().py()?;
        Ok(PyRoom { inner })
    }

    /// Carry the light of emitters to the reflecting elements, once for any number of
    /// detectors.
    ///
    /// Args:
    ///     positions (array_like): `(3,)` or `(n, 3)` positions (m).
    ///     directions (array_like): `(3,)` or `(n, 3)` axes (see `owc_direction`).
    ///     mode (float | array_like): the Lambertian mode (see `lambertian_mode`).
    ///     power_w (float | array_like): the optical power (W).
    ///     bin_s (float): the width of the time bins (s).
    ///     partitions (list[tuple[int, int, int]]): the elements `(N_x, N_y, N_z)` per
    ///         dimension, one partition per order of reflection.
    ///
    /// Returns:
    ///     OwcIllumination: the light on the elements.
    #[pyo3(signature = (positions, directions, mode, power_w, *, bin_s, partitions))]
    #[allow(clippy::too_many_arguments, reason = "the emitters as arrays and the resolution")]
    fn illuminate(
        &self,
        py: Python<'_>,
        positions: &Bound<'_, PyAny>,
        directions: &Bound<'_, PyAny>,
        mode: &Bound<'_, PyAny>,
        power_w: &Bound<'_, PyAny>,
        bin_s: f64,
        partitions: Vec<[usize; 3]>,
    ) -> PyResult<PyIllumination> {
        let e = emitters(positions, directions, mode, power_w)?;
        let room = self.inner;
        let inner = py.detach(|| room.illuminate(&e, &Resolution::new(bin_s, partitions))).py()?;
        Ok(PyIllumination { inner })
    }

    fn __repr__(&self) -> String {
        format!("OwcRoom(size_m={:?}, reflectance={:?})", self.inner.size_m, self.inner.reflectance)
    }
}

/// The light of a set of emitters carried to the elements of a room
/// (`OwcRoom.illuminate`).
#[pyclass(name = "OwcIllumination", module = "comrs")]
pub struct PyIllumination {
    inner: Illumination,
}

#[pymethods]
impl PyIllumination {
    /// The impulse responses at a detector: the line of sight, then each order of
    /// reflection, on the same bins.
    ///
    /// Args:
    ///     position (array_like): `(3,)` the detector (m).
    ///     direction (array_like): `(3,)` its normal.
    ///     area_m2 (float): its area (m²).
    ///     fov_rad (float): its field of view (rad).
    ///     filter_gain (float): the gain of the optical filter.
    ///     concentrator_index (float | None): the refractive index of a concentrator of
    ///         gain `n²/sin²Ψ`.
    ///
    /// Returns:
    ///     list[OwcImpulseResponse]: one response per order, from the line of sight.
    #[pyo3(signature = (position, direction, area_m2, fov_rad, *, filter_gain=1.0, concentrator_index=None))]
    fn impulse_responses(
        &self,
        position: &Bound<'_, PyAny>,
        direction: &Bound<'_, PyAny>,
        area_m2: f64,
        fov_rad: f64,
        filter_gain: f64,
        concentrator_index: Option<f64>,
    ) -> PyResult<Vec<PyImpulseResponse>> {
        let det = Detector {
            filter_gain,
            concentrator_index,
            ..Detector::new(vec3(position, "position")?, vec3(direction, "direction")?, area_m2, fov_rad)
        };
        Ok(self.inner.impulse_responses(&det).py()?.into_iter().map(|inner| PyImpulseResponse { inner }).collect())
    }
}

/// The Rytov variance `σ₁² = 1.23 C_n² k^{7/6} L^{11/6}` of a plane wave.
///
/// Args:
///     cn2 (float): the refractive-index structure parameter (m^-2/3).
///     wavelength_m (float): the wavelength (m).
///     length_m (float): the length of the link (m).
///
/// Returns:
///     float: `σ₁²`.
#[pyfunction]
fn rytov_variance(cn2: f64, wavelength_m: f64, length_m: f64) -> f64 {
    fso::rytov_variance(cn2, wavelength_m, length_m)
}

/// The attenuation (dB/km) of haze and fog for a visibility `visibility_km` (Kim et al.
/// 2001).
///
/// Args:
///     visibility_km (float): the visibility (km).
///     wavelength_nm (float): the wavelength (nm).
///
/// Returns:
///     float: the attenuation (dB/km).
#[pyfunction]
fn fog_attenuation_db_per_km(visibility_km: f64, wavelength_nm: f64) -> PyResult<f64> {
    fso::attenuation_db_per_km(visibility_km, wavelength_nm).py()
}

/// The gamma–gamma distribution of the intensity, of unit mean (Al-Habash, Andrews &
/// Phillips 2001).
///
/// Args:
///     alpha (float): the effective number of large-scale cells.
///     beta (float): the effective number of small-scale cells.
///
/// Examples:
///     >>> import comrs
///     >>> g = comrs.GammaGamma.from_rytov(25.0, "plane")
///     >>> round(g.alpha, 2), round(g.beta, 2), round(g.scintillation_index(), 3)
///     (8.05, 1.03, 1.214)
#[pyclass(name = "GammaGamma", module = "comrs")]
#[derive(Clone, Copy)]
pub struct PyGammaGamma {
    inner: GammaGamma,
}

#[pymethods]
impl PyGammaGamma {
    #[new]
    fn new(alpha: f64, beta: f64) -> PyResult<Self> {
        Ok(PyGammaGamma { inner: GammaGamma::new(alpha, beta).py()? })
    }

    /// The parameters predicted by the Rytov variance for a negligible inner scale.
    ///
    /// Args:
    ///     rytov (float): `σ₁²`.
    ///     wave (str): `"plane"` or `"spherical"`.
    ///
    /// Returns:
    ///     GammaGamma: the distribution.
    #[staticmethod]
    #[pyo3(signature = (rytov, wave="plane"))]
    fn from_rytov(rytov: f64, wave: &str) -> PyResult<Self> {
        Ok(PyGammaGamma { inner: GammaGamma::from_rytov(rytov, wave.parse::<Wave>().py()?).py()? })
    }

    /// float: `α`.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// float: `β`.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    /// The scintillation index `1/α + 1/β + 1/(αβ)`.
    ///
    /// Returns:
    ///     float: the scintillation index.
    fn scintillation_index(&self) -> f64 {
        self.inner.scintillation_index()
    }

    /// The density of the intensity.
    ///
    /// Args:
    ///     i (float | array_like): the intensities (of unit mean).
    ///
    /// Returns:
    ///     float | numpy.ndarray: the density, of the shape of `i`.
    fn pdf(&self, i: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        map_real(i, |x| Ok(self.inner.pdf(x)))
    }

    /// Independent draws of the intensity (a product of two gamma variables).
    ///
    /// Args:
    ///     rng (Rng): the generator.
    ///     n (int): the number of draws.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n,)` (float64).
    fn draw<'py>(&self, py: Python<'py>, mut rng: PyRefMut<'_, PyRng>, n: usize) -> FArr<'py> {
        farr(py, (0..n).map(|_| self.inner.draw(&mut rng.inner)).collect())
    }

    fn __repr__(&self) -> String {
        format!("GammaGamma(alpha={}, beta={})", self.inner.alpha, self.inner.beta)
    }
}

/// The misalignment of a Gaussian beam on a circular aperture, a Rayleigh radial
/// displacement (Farid & Hranilovic 2007).
///
/// Args:
///     beam_radius_m (float): the beam radius at the receiver (m).
///     aperture_radius_m (float): the radius of the aperture (m).
///     jitter_m (float): the standard deviation of the displacement per direction (m).
#[pyclass(name = "PointingError", module = "comrs")]
#[derive(Clone, Copy)]
pub struct PyPointingError {
    inner: PointingError,
}

#[pymethods]
impl PyPointingError {
    #[new]
    fn new(beam_radius_m: f64, aperture_radius_m: f64, jitter_m: f64) -> PyResult<Self> {
        Ok(PyPointingError { inner: PointingError::new(beam_radius_m, aperture_radius_m, jitter_m).py()? })
    }

    /// The fraction `A₀ = erf²(v)` collected without displacement.
    ///
    /// Returns:
    ///     float: `A₀`.
    fn a0(&self) -> f64 {
        self.inner.a0()
    }

    /// The equivalent beam radius (m).
    ///
    /// Returns:
    ///     float: the radius (m).
    fn equivalent_radius_m(&self) -> f64 {
        self.inner.equivalent_radius_m()
    }

    /// The fraction of the power collected at a radial displacement.
    ///
    /// Args:
    ///     r_m (float | array_like): the displacements (m).
    ///
    /// Returns:
    ///     float | numpy.ndarray: the fraction, of the shape of `r_m`.
    fn collected(&self, r_m: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        map_real(r_m, |x| Ok(self.inner.collected(x)))
    }

    /// The density of the collected fraction.
    ///
    /// Args:
    ///     h (float | array_like): the fractions, in `(0, A₀]`.
    ///
    /// Returns:
    ///     float | numpy.ndarray: the density, of the shape of `h`.
    fn pdf(&self, h: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        map_real(h, |x| Ok(self.inner.pdf(x)))
    }

    /// Independent draws of the collected fraction (a Rayleigh displacement).
    ///
    /// Args:
    ///     rng (Rng): the generator.
    ///     n (int): the number of draws.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n,)` (float64).
    fn draw<'py>(&self, py: Python<'py>, mut rng: PyRefMut<'_, PyRng>, n: usize) -> FArr<'py> {
        farr(py, (0..n).map(|_| self.inner.draw(&mut rng.inner)).collect())
    }
}

/// A free-space optical link: a deterministic loss, turbulence and pointing errors, the
/// gain of the intensity `h = h_l h_a h_p`.
///
/// Args:
///     loss_db (float): the loss of the weather and of the geometry (dB).
///     turbulence (GammaGamma | str | None): a gamma–gamma model, `"lognormal"` (with
///         `scintillation_index`), `"negative-exponential"`, or None.
///     scintillation_index (float | None): the scintillation index of the log-normal
///         model.
///     pointing (PointingError | None): the pointing errors.
///
/// Examples:
///     >>> import comrs
///     >>> link = comrs.Fso(3.0, turbulence=comrs.GammaGamma.from_rytov(1.0))
///     >>> h = link.gains(comrs.Rng(1), 100_000)
///     >>> round(float(h.mean()), 2)  # 10^(-0.3), the turbulence of unit mean
///     0.5
#[pyclass(name = "Fso", module = "comrs")]
pub struct PyFso {
    inner: fso::Fso,
}

#[pymethods]
impl PyFso {
    #[new]
    #[pyo3(signature = (loss_db=0.0, *, turbulence=None, scintillation_index=None, pointing=None))]
    fn new(
        loss_db: f64,
        turbulence: Option<&Bound<'_, PyAny>>,
        scintillation_index: Option<f64>,
        pointing: Option<PyRef<'_, PyPointingError>>,
    ) -> PyResult<Self> {
        let turbulence = match turbulence {
            Some(t) if t.is_instance_of::<PyGammaGamma>() => {
                if scintillation_index.is_some() {
                    return Err(PyValueError::new_err("scintillation_index is that of the log-normal model"));
                }
                Turbulence::GammaGamma(t.cast::<PyGammaGamma>()?.borrow().inner)
            }
            Some(t) => Turbulence::from_name(&t.extract::<String>()?, scintillation_index).py()?,
            None => Turbulence::from_name("none", scintillation_index).py()?,
        };
        let inner = fso::Fso { loss_db, turbulence, pointing: pointing.map(|p| p.inner) };
        inner.check().py()?;
        Ok(PyFso { inner })
    }

    /// The scintillation index of the turbulence.
    ///
    /// Returns:
    ///     float: the scintillation index.
    fn scintillation_index(&self) -> f64 {
        self.inner.turbulence.scintillation_index()
    }

    /// Independent draws of the gain of the intensity, one per frame.
    ///
    /// Args:
    ///     rng (Rng): the generator.
    ///     n (int): the number of draws.
    ///
    /// Returns:
    ///     numpy.ndarray: `(n,)` (float64).
    fn gains<'py>(&self, py: Python<'py>, mut rng: PyRefMut<'_, PyRng>, n: usize) -> PyResult<FArr<'py>> {
        Ok(farr(py, (0..n).map(|_| self.inner.gain(&mut rng.inner)).collect::<comrs::Result<Vec<_>>>().py()?))
    }
}

/// A PIN photodiode with a FET preamplifier: shot and thermal noise (Komine & Nakagawa
/// 2004, eqs. (18)–(21)).
///
/// Args:
///     responsivity (float): the responsivity (A/W).
///     background_current_a (float): the background current of the ambient light (A).
///     open_loop_gain (float): the open-loop voltage gain.
///     capacitance_f_per_m2 (float): the fixed capacitance per unit area (F/m²).
///     channel_noise_factor (float): the FET channel noise factor.
///     transconductance_s (float): the FET transconductance (S).
///     temperature_k (float): the absolute temperature (K).
///     i2 (float): the noise bandwidth factor `I₂`.
///     i3 (float): the noise bandwidth factor `I₃`.
///
/// Examples:
///     >>> import comrs
///     >>> rx = comrs.PinFetReceiver.komine2004()
///     >>> rx.responsivity
///     0.53
#[pyclass(name = "PinFetReceiver", module = "comrs")]
#[derive(Clone)]
pub struct PyPinFet {
    inner: PinFetReceiver,
}

wrapper_methods! {
    PyPinFet {
        /// float: Responsivity (A/W) (read and write).
        responsivity / set_responsivity: f64,
        /// float: Background current of the ambient light (A) (read and write).
        background_current_a / set_background_current_a: f64,
        /// float: Open-loop voltage gain (read and write).
        open_loop_gain / set_open_loop_gain: f64,
        /// float: Fixed capacitance per unit area (F/m²) (read and write).
        capacitance_f_per_m2 / set_capacitance_f_per_m2: f64,
        /// float: FET channel noise factor (read and write).
        channel_noise_factor / set_channel_noise_factor: f64,
        /// float: FET transconductance (S) (read and write).
        transconductance_s / set_transconductance_s: f64,
        /// float: Absolute temperature (K) (read and write).
        temperature_k / set_temperature_k: f64,
        /// float: Noise bandwidth factor `I₂` (read and write).
        i2 / set_i2: f64,
        /// float: Noise bandwidth factor `I₃` (read and write).
        i3 / set_i3: f64,
    }

    #[new]
    #[allow(clippy::too_many_arguments, reason = "one argument per field of the receiver")]
    fn new(
        responsivity: f64,
        background_current_a: f64,
        open_loop_gain: f64,
        capacitance_f_per_m2: f64,
        channel_noise_factor: f64,
        transconductance_s: f64,
        temperature_k: f64,
        i2: f64,
        i3: f64,
    ) -> Self {
        PyPinFet {
            inner: PinFetReceiver {
                responsivity,
                background_current_a,
                open_loop_gain,
                capacitance_f_per_m2,
                channel_noise_factor,
                transconductance_s,
                temperature_k,
                i2,
                i3,
            },
        }
    }

    /// The receiver of Komine & Nakagawa (2004), Tables II–III.
    ///
    /// Returns:
    ///     PinFetReceiver: the receiver.
    #[staticmethod]
    fn komine2004() -> Self {
        PyPinFet { inner: PinFetReceiver::komine2004() }
    }

    /// The variance of the shot noise (A²) of the signal and of the ambient light.
    ///
    /// Args:
    ///     signal_w (float): the power received inside the bit (W).
    ///     isi_w (float): the power spilled from the previous bits (W).
    ///     bandwidth_hz (float): the noise bandwidth (Hz).
    ///
    /// Returns:
    ///     float: the variance (A²).
    fn shot_variance(&self, signal_w: f64, isi_w: f64, bandwidth_hz: f64) -> f64 {
        self.inner.shot_variance(signal_w, isi_w, bandwidth_hz)
    }

    /// The variance of the thermal noise (A²) of the feedback resistor and of the FET.
    ///
    /// Args:
    ///     area_m2 (float): the area of the photodiode (m²).
    ///     bandwidth_hz (float): the noise bandwidth (Hz).
    ///
    /// Returns:
    ///     float: the variance (A²).
    fn thermal_variance(&self, area_m2: f64, bandwidth_hz: f64) -> f64 {
        self.inner.thermal_variance(area_m2, bandwidth_hz)
    }

    /// The SNR (dB) of on–off keying with intersymbol interference (their eq. (16)).
    ///
    /// Args:
    ///     signal_w (float): the power received inside the bit (W).
    ///     isi_w (float): the power spilled from the previous bits (W).
    ///     area_m2 (float): the area of the photodiode (m²).
    ///     bandwidth_hz (float): the noise bandwidth (Hz).
    ///
    /// Returns:
    ///     float: the SNR (dB).
    fn ook_snr_db(&self, signal_w: f64, isi_w: f64, area_m2: f64, bandwidth_hz: f64) -> f64 {
        self.inner.ook_snr_db(signal_w, isi_w, area_m2, bandwidth_hz)
    }

    fn __repr__(&self) -> String {
        let r = &self.inner;
        format!(
            "comrs.PinFetReceiver({}, {:e}, {}, {:e}, {}, {}, {}, {}, {})",
            r.responsivity,
            r.background_current_a,
            r.open_loop_gain,
            r.capacitance_f_per_m2,
            r.channel_noise_factor,
            r.transconductance_s,
            r.temperature_k,
            r.i2,
            r.i3
        )
    }

    #[allow(clippy::type_complexity, reason = "the nine positional arguments of the constructor")]
    fn __getnewargs__(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let r = &self.inner;
        (
            r.responsivity,
            r.background_current_a,
            r.open_loop_gain,
            r.capacitance_f_per_m2,
            r.channel_noise_factor,
            r.transconductance_s,
            r.temperature_k,
            r.i2,
            r.i3,
        )
    }
}

/// Register the optical wireless bindings.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lambertian_mode, m)?)?;
    m.add_function(wrap_pyfunction!(owc_direction, m)?)?;
    m.add_function(wrap_pyfunction!(owc_illuminance, m)?)?;
    m.add_function(wrap_pyfunction!(rytov_variance, m)?)?;
    m.add_function(wrap_pyfunction!(fog_attenuation_db_per_km, m)?)?;
    m.add_class::<PyRoom>()?;
    m.add_class::<PyIllumination>()?;
    m.add_class::<PyImpulseResponse>()?;
    m.add_class::<PyGammaGamma>()?;
    m.add_class::<PyPointingError>()?;
    m.add_class::<PyFso>()?;
    m.add_class::<PyPinFet>()?;
    Ok(())
}

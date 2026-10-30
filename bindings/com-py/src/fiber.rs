//! Fibre propagation: fibre parameters, the EDFA, the linear fibre, the split-step
//! Fourier method (one polarisation) and the Manakov equation (two polarisations).

use crate::rng::PyRng;
use crate::{CArr, IntoPy, carr, cvec, json, rng_of};
use comrs::channel::optical::edfa::Edfa;
use comrs::channel::optical::fiber::FiberParams;
use comrs::channel::optical::fiber::{self as channels, Amplification, Precision, Ssfm};
use comrs::channel::optical::manakov::ManakovSsfm;
use comrs::channel::optical::pmd::{Manakov, PmdLink};
use comrs::channel::optical::raman::Raman;
use comrs::core::signal::DualPol;
use comrs::math::nlse::StepRule;
use comrs::spec::blocks::BuildContext;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The parameters of a fibre link of identical spans, each followed by an amplifier.
///
/// Engineering units of OptiCommPy: attenuation in dB/km, dispersion parameter `D` in
/// ps/(nm·km), nonlinear coefficient `γ` in 1/(W·km), lengths in km, carrier frequency
/// in Hz, noise figure in dB. The defaults describe standard single-mode fibre at
/// 193.1 THz with 80 km spans. Every attribute is readable and writable.
///
/// Args:
///     alpha_db_km (float): attenuation (dB/km).
///     d_ps_nm_km (float): dispersion parameter `D` (ps/(nm·km)).
///     gamma_w_km (float): nonlinear coefficient `γ` (1/(W·km)).
///     span_km (float): span length (km).
///     n_spans (int): number of spans.
///     fc_hz (float): carrier frequency (Hz).
///     nf_db (float): noise figure of the amplifiers (dB).
///
/// Examples:
///     >>> import comrs
///     >>> f = comrs.FiberParams(n_spans=10)
///     >>> f.total_km
///     800.0
///     >>> round(f.beta2_ps2_km, 2)
///     -21.75
#[pyclass(name = "FiberParams", module = "comrs", get_all, set_all)]
#[derive(Clone)]
pub struct PyFiber {
    /// float: Attenuation (dB/km) (read and write).
    pub alpha_db_km: f64,
    /// float: Dispersion parameter `D` (ps/(nm·km)) (read and write).
    pub d_ps_nm_km: f64,
    /// float: Nonlinear coefficient `γ` (1/(W·km)) (read and write).
    pub gamma_w_km: f64,
    /// float: Span length (km) (read and write).
    pub span_km: f64,
    /// int: Number of spans (read and write).
    pub n_spans: usize,
    /// float: Carrier frequency (Hz) (read and write).
    pub fc_hz: f64,
    /// float: Noise figure of the amplifiers (dB) (read and write).
    pub nf_db: f64,
}

impl PyFiber {
    /// The Rust parameters.
    pub fn to_rust(&self) -> FiberParams {
        FiberParams {
            alpha_db_km: self.alpha_db_km,
            d_ps_nm_km: self.d_ps_nm_km,
            gamma_w_km: self.gamma_w_km,
            span_km: self.span_km,
            n_spans: self.n_spans,
            fc_hz: self.fc_hz,
            nf_db: self.nf_db,
        }
    }

    fn from_rust(f: &FiberParams) -> PyFiber {
        PyFiber {
            alpha_db_km: f.alpha_db_km,
            d_ps_nm_km: f.d_ps_nm_km,
            gamma_w_km: f.gamma_w_km,
            span_km: f.span_km,
            n_spans: f.n_spans,
            fc_hz: f.fc_hz,
            nf_db: f.nf_db,
        }
    }
}

#[pymethods]
impl PyFiber {
    #[new]
    #[pyo3(signature = (*, alpha_db_km = 0.2, d_ps_nm_km = 17.0, gamma_w_km = 1.3, span_km = 80.0, n_spans = 1, fc_hz = 193.1e12, nf_db = 4.5))]
    fn new(
        alpha_db_km: f64,
        d_ps_nm_km: f64,
        gamma_w_km: f64,
        span_km: f64,
        n_spans: usize,
        fc_hz: f64,
        nf_db: f64,
    ) -> Self {
        PyFiber { alpha_db_km, d_ps_nm_km, gamma_w_km, span_km, n_spans, fc_hz, nf_db }
    }

    /// float: Total length (km).
    #[getter]
    fn total_km(&self) -> f64 {
        self.to_rust().total_km()
    }

    /// float: Power attenuation `α` (neper/km).
    #[getter]
    fn alpha_np_km(&self) -> f64 {
        self.to_rust().alpha_np_km()
    }

    /// float: Wavelength `c/f_c` (m).
    #[getter]
    fn wavelength_m(&self) -> f64 {
        self.to_rust().wavelength_m()
    }

    /// float: Group-velocity dispersion `β₂ = -Dλ²/(2πc)` (s²/km).
    #[getter]
    fn beta2_s2_km(&self) -> f64 {
        self.to_rust().beta2_s2_km()
    }

    /// float: Group-velocity dispersion `β₂` (ps²/km).
    #[getter]
    fn beta2_ps2_km(&self) -> f64 {
        self.to_rust().beta2_ps2_km()
    }

    /// float: Linear power gain `e^{αL}` that compensates the loss of one span.
    #[getter]
    fn span_gain(&self) -> f64 {
        self.to_rust().span_gain()
    }

    /// float: Span gain (dB).
    #[getter]
    fn span_gain_db(&self) -> f64 {
        self.to_rust().span_gain_db()
    }

    /// float: Effective length of a span `(1 - e^{-αL})/α` (km).
    #[getter]
    fn effective_length_km(&self) -> f64 {
        self.to_rust().effective_length_km()
    }

    /// float: Accumulated dispersion `D·L_total` (ps/nm).
    #[getter]
    fn accumulated_dispersion_ps_nm(&self) -> f64 {
        self.to_rust().accumulated_dispersion_ps_nm()
    }

    /// Dispersion memory of the link for a signal of bandwidth `bandwidth_hz`,
    /// `2π |β₂| L B fs`.
    ///
    /// Args:
    ///     bandwidth_hz (float): bandwidth of the signal (Hz).
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     float: the memory (samples at `fs`).
    fn dispersion_memory_samples(&self, bandwidth_hz: f64, fs: f64) -> f64 {
        self.to_rust().dispersion_memory_samples(bandwidth_hz, fs)
    }

    fn __repr__(&self) -> String {
        format!(
            "comrs.FiberParams(alpha_db_km={}, d_ps_nm_km={}, gamma_w_km={}, span_km={}, n_spans={}, fc_hz={:e}, nf_db={})",
            self.alpha_db_km, self.d_ps_nm_km, self.gamma_w_km, self.span_km, self.n_spans, self.fc_hz, self.nf_db
        )
    }
}

/// An ideal erbium-doped fibre amplifier: flat gain and white ASE.
///
/// The spontaneous-emission factor is `n_sp = (G·NF - 1)/(2(G - 1))` and the ASE adds
/// complex Gaussian noise of variance `(G - 1) n_sp h f_c f_s` per sample at the
/// sampling rate `f_s` (OptiCommPy `edfa`).
///
/// Args:
///     gain_db (float): gain (dB, positive).
///     nf_db (float): noise figure (dB, at least 3 dB).
///     fc_hz (float): carrier frequency (Hz).
#[pyclass(name = "Edfa", module = "comrs", get_all, set_all)]
#[derive(Clone)]
pub struct PyEdfa {
    /// float: Gain (dB) (read and write).
    pub gain_db: f64,
    /// float: Noise figure (dB) (read and write).
    pub nf_db: f64,
    /// float: Carrier frequency (Hz) (read and write).
    pub fc_hz: f64,
}

impl PyEdfa {
    fn to_rust(&self) -> Edfa {
        Edfa { gain_db: self.gain_db, nf_db: self.nf_db, fc_hz: self.fc_hz }
    }
}

#[pymethods]
impl PyEdfa {
    #[new]
    #[pyo3(signature = (*, gain_db = 20.0, nf_db = 4.5, fc_hz = 193.1e12))]
    fn new(gain_db: f64, nf_db: f64, fc_hz: f64) -> Self {
        PyEdfa { gain_db, nf_db, fc_hz }
    }

    /// float: Spontaneous-emission factor `n_sp`.
    #[getter]
    fn n_sp(&self) -> f64 {
        self.to_rust().n_sp()
    }

    /// float: ASE power spectral density `(G - 1) n_sp h f_c` (W/Hz).
    #[getter]
    fn ase_psd(&self) -> f64 {
        self.to_rust().ase_psd()
    }

    /// Variance of the ASE per sample.
    ///
    /// Args:
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     float: the variance `E|w|²` (W).
    fn noise_variance(&self, fs: f64) -> f64 {
        self.to_rust().noise_variance(fs)
    }

    /// Amplify a field and add the ASE.
    ///
    /// Args:
    ///     x (array_like): the field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng): the generator of the ASE.
    ///
    /// Returns:
    ///     numpy.ndarray: the amplified field (complex128).
    ///
    /// Raises:
    ///     ValueError: for a gain that is not positive or a noise figure below 3 dB.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: f64,
        mut rng: PyRefMut<'py, PyRng>,
    ) -> PyResult<CArr<'py>> {
        Ok(carr(py, self.to_rust().apply(&cvec(x)?, fs, &mut rng.inner).py()?))
    }

    fn __repr__(&self) -> String {
        format!("comrs.Edfa(gain_db={}, nf_db={}, fc_hz={:e})", self.gain_db, self.nf_db, self.fc_hz)
    }
}

/// Loss and dispersion of `length_km` of fibre on a block sampled at `fs`
/// (OptiCommPy `linearFiberChannel`).
///
/// Args:
///     x (array_like): the field (√W).
///     fiber (FiberParams): the fibre.
///     length_km (float): the length (km).
///     fs (float): sampling rate (Hz).
///
/// Returns:
///     numpy.ndarray: the field at the output (complex128).
#[pyfunction]
fn linear_fiber_channel<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    fiber: &PyFiber,
    length_km: f64,
    fs: f64,
) -> PyResult<CArr<'py>> {
    let (x, f) = (cvec(x)?, fiber.to_rust());
    Ok(carr(py, py.detach(|| channels::linear_fiber_channel(&x, &f, length_km, fs))))
}

/// The Raman amplification of a ``raman`` mapping (the keys of ``fiber.raman`` in a
/// specification).
pub(crate) fn raman(py: Python<'_>, v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Raman>> {
    let Some(v) = v else { return Ok(None) };
    if v.is_none() {
        return Ok(None);
    }
    let j = json(py, v)?;
    BuildContext::default().raman(Some(&serde_json::json!({ "raman": j }))).py()
}

/// The symmetric split-step Fourier method of OptiCommPy (`ssfm`), one polarisation.
///
/// Every span is cut into `⌊L_span/h⌋` steps; a step applies half of the linear
/// operator (loss and dispersion) in the frequency domain, the nonlinear phase
/// `γ|A|²h`, then the other half. After every span:
///
/// - `"edfa"`: an EDFA of gain `α L` and the noise figure of the fibre (ASE drawn from
///   `rng`, noiseless without one);
/// - `"ideal"`: a noiseless gain `e^{αL/2}` on the field;
/// - `"none"`: nothing.
///
/// Args:
///     fiber (FiberParams): the link.
///     step_km (float): step size `h` (km).
///     amplification (str): `"edfa"`, `"ideal"` or `"none"`.
///     raman (dict, optional): Raman pumps and ISRS in every span, the keys of
///         ``fiber.raman`` in a specification (see :func:`raman_profile`); the amplifier
///         after a span then restores every frequency to its launch power.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> link = comrs.Ssfm(comrs.FiberParams(n_spans=2), step_km=1.0, amplification="ideal")
///     >>> x = np.exp(2j * np.pi * np.arange(64) / 64) * 1e-2
///     >>> np.allclose(link.backpropagate(link.propagate(x, 32e9), 32e9, steps_per_span=80), x)
///     True
#[pyclass(name = "Ssfm", module = "comrs")]
pub struct PySsfm {
    inner: Ssfm,
}

#[pymethods]
impl PySsfm {
    #[new]
    #[pyo3(signature = (fiber, *, step_km = 1.0, amplification = "edfa", raman = None))]
    fn new(
        py: Python<'_>,
        fiber: &PyFiber,
        step_km: f64,
        amplification: &str,
        raman: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(PySsfm {
            inner: Ssfm {
                fiber: fiber.to_rust(),
                step_km,
                amplification: amplification.parse::<Amplification>().py()?,
                raman: self::raman(py, raman)?,
            },
        })
    }

    /// FiberParams: The link (a copy).
    #[getter]
    fn fiber(&self) -> PyFiber {
        PyFiber::from_rust(&self.inner.fiber)
    }

    /// float: The step size (km).
    #[getter]
    fn step_km(&self) -> f64 {
        self.inner.step_km
    }

    /// Propagate a field over the link.
    ///
    /// Args:
    ///     x (array_like): the field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng, optional): the generator of the ASE (noiseless without one).
    ///     precision (str): ``"f64"``, or ``"f32"`` (about twice as fast, relative error
    ///         about ``1e-6 √steps``; the amplifiers stay in double precision).
    ///
    /// Returns:
    ///     numpy.ndarray: the field at the end of the link (complex128).
    #[pyo3(signature = (x, fs, rng = None, *, precision = "f64"))]
    fn propagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: f64,
        rng: Option<PyRefMut<'py, PyRng>>,
        precision: &str,
    ) -> PyResult<CArr<'py>> {
        let precision: Precision = precision.parse().py()?;
        let x = cvec(x)?;
        let mut rng = rng;
        let r = rng_of(&mut rng);
        let s = &self.inner;
        let y = py.detach(|| s.propagate_with(&x, fs, r, precision));
        Ok(carr(py, y.py()?))
    }

    /// Digital back-propagation with `steps_per_span` symmetric steps per span (the
    /// split step with `-α`, `-β₂`, `-γ`, spans in reverse order, amplifier gain removed
    /// first). With the forward step and no noise it inverts `propagate` exactly; with
    /// coarser steps the nonlinear phase of a step uses its effective length
    /// `2 sinh(αδ/2)/α`.
    ///
    /// Args:
    ///     x (array_like): the received field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     steps_per_span (int): steps per span (1 by default, as the ``dbp`` block).
    ///     nonlinear_scale (float): ξ, multiplies the nonlinear phase (the optimised DBP
    ///         with few steps per span, Ip & Kahn 2008; 0 gives the linear
    ///         back-propagation).
    ///
    /// Returns:
    ///     numpy.ndarray: the back-propagated field (complex128).
    #[pyo3(signature = (x, fs, *, steps_per_span = 1, nonlinear_scale = 1.0))]
    fn backpropagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        fs: f64,
        steps_per_span: usize,
        nonlinear_scale: f64,
    ) -> PyResult<CArr<'py>> {
        let x = cvec(x)?;
        let s = &self.inner;
        Ok(carr(py, py.detach(|| s.backpropagate_scaled(&x, fs, steps_per_span, nonlinear_scale)).py()?))
    }
}

/// Dual-polarisation propagation by the Manakov equation with polarisation-mode
/// dispersion (coarse-step model).
///
/// Every span is cut into `sections_per_span` sections; a section applies the loss and
/// the dispersion, a random rotation and a differential group delay (the PMD), then the
/// averaged Kerr phase `(8/9) γ L_eff (|A_x|² + |A_y|²)`. The amplification after every
/// span is an EDFA (ASE from `rng` if given).
///
/// Args:
///     fiber (FiberParams): the link.
///     pmd_ps_sqrt_km (float): PMD coefficient (ps/√km); 0 for none.
///     sections_per_span (int): sections per span.
///     rng (Rng, optional): draws the PMD realisation (needed when the PMD is not 0).
#[pyclass(name = "Manakov", module = "comrs")]
pub struct PyManakov {
    inner: Manakov,
}

#[pymethods]
impl PyManakov {
    #[new]
    #[pyo3(signature = (fiber, *, pmd_ps_sqrt_km = 0.0, sections_per_span = 10, rng = None))]
    fn new(
        fiber: &PyFiber,
        pmd_ps_sqrt_km: f64,
        sections_per_span: usize,
        rng: Option<PyRefMut<'_, PyRng>>,
    ) -> PyResult<Self> {
        let f = fiber.to_rust();
        let pmd = match (pmd_ps_sqrt_km, rng) {
            (0.0, _) => PmdLink::none(&f, sections_per_span).py()?,
            (p, Some(mut r)) => PmdLink::random(&f, p, sections_per_span, &mut r.inner).py()?,
            (_, None) => return Err(PyValueError::new_err("a PMD realisation needs an rng")),
        };
        Ok(PyManakov { inner: Manakov::new(f, pmd) })
    }

    /// float: RMS differential group delay of the PMD realisation (s).
    #[getter]
    fn rms_dgd_s(&self) -> f64 {
        self.inner.pmd.rms_dgd_s()
    }

    /// Propagate the two polarisations.
    ///
    /// Args:
    ///     x (array_like): the field of the first polarisation (√W).
    ///     y (array_like): the field of the second polarisation (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng, optional): the generator of the ASE (noiseless without one).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: the fields of the two polarisations.
    #[pyo3(signature = (x, y, fs, rng = None))]
    fn propagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        fs: f64,
        rng: Option<PyRefMut<'py, PyRng>>,
    ) -> PyResult<(CArr<'py>, CArr<'py>)> {
        let mut u = dual_pol(x, y)?;
        let mut rng = rng;
        let r = rng_of(&mut rng);
        let m = &self.inner;
        py.detach(|| m.propagate(&mut u, fs, r)).py()?;
        Ok(split(py, u))
    }

    /// Back-propagation with the known PMD realisation (inverts `propagate` without
    /// noise).
    ///
    /// Args:
    ///     x (array_like): the received field of the first polarisation (√W).
    ///     y (array_like): the received field of the second polarisation (√W).
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: the two back-propagated fields.
    fn backpropagate_with_pmd<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        fs: f64,
    ) -> PyResult<(CArr<'py>, CArr<'py>)> {
        let mut u = dual_pol(x, y)?;
        let m = &self.inner;
        py.detach(|| m.backpropagate_with_pmd(&mut u, fs));
        Ok(split(py, u))
    }

    /// Undo the lumped PMD of the link (inverse of its frequency response), leaving the
    /// dispersion and the nonlinearity.
    ///
    /// Args:
    ///     x (array_like): the field of the first polarisation (√W).
    ///     y (array_like): the field of the second polarisation (√W).
    ///     fs (float): sampling rate (Hz).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: the two fields without the PMD.
    fn equalize_pmd<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        fs: f64,
    ) -> PyResult<(CArr<'py>, CArr<'py>)> {
        let mut u = dual_pol(x, y)?;
        let m = &self.inner;
        py.detach(|| m.equalize_pmd(&mut u, fs));
        Ok(split(py, u))
    }
}

/// Dual-polarisation nonlinear propagation over the link: the Manakov equation solved
/// by the symmetric split-step method of OptiCommPy's ``manakovSSF``.
///
/// Each step is a half linear step, the nonlinear phase
/// ``(8/9) γ h (P(z) + P(z+h))/2`` (``P = |A_x|² + |A_y|²``, the trapezoidal rule,
/// iterated until the relative change of the fields falls below ``tolerance``) and a half
/// linear step. The step is fixed (``step_km``) or adapted so that the largest nonlinear
/// phase of a step is ``max_nl_phase`` (Sinkin et al. 2003). An amplifier follows every
/// span. Polarisation-mode dispersion is neglected (see :class:`Manakov`).
///
/// Args:
///     fiber (FiberParams): the link.
///     max_nl_phase (float): largest nonlinear phase rotation per step (rad).
///     step_km (float, optional): a fixed step (km) instead of the adaptive one.
///     max_iterations (int): trapezoidal iterations per step at most.
///     tolerance (float): convergence tolerance of the iterations.
///     amplification (str): ``"edfa"``, ``"ideal"`` or ``"none"``.
///     raman (dict, optional): Raman pumps and ISRS in every span (see
///         :func:`raman_profile`).
#[pyclass(name = "ManakovSsfm", module = "comrs")]
pub struct PyManakovSsfm {
    inner: ManakovSsfm,
}

#[pymethods]
impl PyManakovSsfm {
    #[new]
    #[pyo3(signature = (fiber, *, max_nl_phase = 2e-2, step_km = None, max_iterations = 10, tolerance = 1e-5, amplification = "edfa", raman = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        fiber: &PyFiber,
        max_nl_phase: f64,
        step_km: Option<f64>,
        max_iterations: usize,
        tolerance: f64,
        amplification: &str,
        raman: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let rule = step_km.map_or(StepRule::NonlinearPhase(max_nl_phase), StepRule::Fixed);
        Ok(PyManakovSsfm {
            inner: ManakovSsfm {
                fiber: fiber.to_rust(),
                rule,
                max_iterations,
                tolerance,
                amplification: amplification.parse::<Amplification>().py()?,
                raman: self::raman(py, raman)?,
            },
        })
    }

    /// Propagate the two polarisations.
    ///
    /// Args:
    ///     x (array_like): the field of the first polarisation (√W).
    ///     y (array_like): the field of the second polarisation (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng, optional): the generator of the ASE (noiseless without one).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray, int]: the two fields and the number of
    ///     steps.
    #[pyo3(signature = (x, y, fs, rng = None))]
    fn propagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        fs: f64,
        rng: Option<PyRefMut<'py, PyRng>>,
    ) -> PyResult<(CArr<'py>, CArr<'py>, usize)> {
        let mut u = dual_pol(x, y)?;
        let mut rng = rng;
        let r = rng_of(&mut rng);
        let m = &self.inner;
        let steps = py.detach(|| m.propagate(&mut u, fs, r)).py()?;
        Ok((carr(py, u.x), carr(py, u.y), steps))
    }

    /// Dual-polarisation digital back-propagation with `steps_per_span` symmetric steps
    /// per span (the Manakov equation with `-α`, `-β₂`, `-γ`, the amplifier gain removed
    /// before each span, the nonlinear phase of a step on its effective length).
    ///
    /// Args:
    ///     x (array_like): the received field of the first polarisation (√W).
    ///     y (array_like): the received field of the second polarisation (√W).
    ///     fs (float): sampling rate (Hz).
    ///     steps_per_span (int): steps per span.
    ///     nonlinear_scale (float): ξ, multiplies the nonlinear phase (0: dispersion
    ///         compensation only).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: the two back-propagated fields.
    #[pyo3(signature = (x, y, fs, *, steps_per_span = 1, nonlinear_scale = 1.0))]
    fn backpropagate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        fs: f64,
        steps_per_span: usize,
        nonlinear_scale: f64,
    ) -> PyResult<(CArr<'py>, CArr<'py>)> {
        let (mut x, mut y) = (cvec(x)?, cvec(y)?);
        let d = self.inner.dbp(steps_per_span, nonlinear_scale);
        py.detach(|| d.run_dual(&mut x, &mut y, fs)).py()?;
        Ok((carr(py, x), carr(py, y)))
    }
}

/// The two polarisations of a dual-polarisation field from two arrays.
fn dual_pol(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<DualPol> {
    DualPol::new(cvec(x)?, cvec(y)?).py()
}

/// The two polarisations as two arrays.
fn split(py: Python<'_>, u: DualPol) -> (CArr<'_>, CArr<'_>) {
    (carr(py, u.x), carr(py, u.y))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFiber>()?;
    m.add_class::<PyEdfa>()?;
    m.add_class::<PySsfm>()?;
    m.add_class::<PyManakov>()?;
    m.add_class::<PyManakovSsfm>()?;
    m.add_function(wrap_pyfunction!(linear_fiber_channel, m)?)?;
    Ok(())
}

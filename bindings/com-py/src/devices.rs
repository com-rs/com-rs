//! Noise sources and optoelectronic devices: AWGN, phase noise, lasers, modulators,
//! photodiodes, the 90° hybrid and the coherent receiver.

use crate::rng::PyRng;
use crate::{CArr, FArr, IntoPy, carr, count, cvec, farr, fvec};
use comrs::channel::awgn as awgn_ch;
use comrs::dsp::filter::LowPass;
use comrs::hardware::optical::{self as devices, Iqm, Laser, Mzm, Photodiode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Add circular complex Gaussian noise for an SNR of `snr_db` within the bandwidth
/// `bandwidth` of a signal sampled at `fs` (OptiCommPy `awgn`):
/// `σ² = (fs/bandwidth) P_x / SNR`.
///
/// Args:
///     x (array_like): the signal.
///     snr_db (float): the SNR (dB).
///     rng (Rng): the generator.
///     fs (float): sampling rate (Hz).
///     bandwidth_hz (float): the bandwidth of the SNR (Hz); with the default `fs = 1`,
///         `bandwidth_hz = 1`, the SNR is per sample.
///
/// Returns:
///     numpy.ndarray: the noisy signal (complex128).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> x = np.ones(100_000, dtype=complex)
///     >>> y = comrs.awgn(x, 10.0, comrs.Rng(1))
///     >>> round(comrs.snr_db(y, x))
///     10
#[pyfunction]
#[pyo3(signature = (x, snr_db, rng, *, fs = 1.0, bandwidth_hz = 1.0))]
fn awgn<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    snr_db: f64,
    mut rng: PyRefMut<'py, PyRng>,
    fs: f64,
    bandwidth_hz: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, awgn_ch::awgn(&cvec(x)?, snr_db, fs, bandwidth_hz, &mut rng.inner)))
}

/// Add noise for an `E_b/N_0` of `ebn0_db`: `σ² = P_x · sps / (b · E_b/N_0)` for `b`
/// bits per symbol at `sps` samples per symbol.
///
/// Args:
///     x (array_like): the signal.
///     ebn0_db (float): `E_b/N_0` (dB).
///     bits_per_symbol (float): information bits per symbol `b` (fractional for a
///         coded or shaped signal).
///     rng (Rng): the generator.
///     sps (float): samples per symbol.
///
/// Returns:
///     numpy.ndarray: the noisy signal (complex128).
///
/// Examples:
///     >>> import comrs
///     >>> c = comrs.Constellation("qam", 4).normalized()
///     >>> b = comrs.bits(2 * 50_000, comrs.Rng(1))
///     >>> y = comrs.awgn_ebn0(c.modulate(b), 6.0, 2, comrs.Rng(2))
///     >>> abs(comrs.ber(c.demodulate(y), b) / comrs.theory_ber(4, 6.0) - 1) < 0.15
///     True
#[pyfunction]
#[pyo3(signature = (x, ebn0_db, bits_per_symbol, rng, *, sps = 1.0))]
fn awgn_ebn0<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    ebn0_db: f64,
    bits_per_symbol: f64,
    mut rng: PyRefMut<'py, PyRng>,
    sps: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, awgn_ch::awgn_ebn0(&cvec(x)?, ebn0_db, bits_per_symbol, sps, &mut rng.inner)))
}

/// Wiener phase noise of a laser: a random walk starting at 0 whose increments have
/// the variance `2π Δν / fs`.
///
/// Args:
///     linewidth_hz (float): the linewidth `Δν` (Hz).
///     n (int): number of samples.
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator.
///
/// Returns:
///     numpy.ndarray: the phase of every sample (rad).
#[pyfunction]
fn phase_noise<'py>(
    py: Python<'py>,
    linewidth_hz: f64,
    n: i64,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
) -> PyResult<FArr<'py>> {
    if fs <= 0.0 {
        return Err(PyValueError::new_err(format!("the sampling rate must be positive, got {fs}")));
    }
    let n = count(n, "n")?;
    Ok(farr(py, comrs::hardware::oscillator::phase_noise(linewidth_hz, n, 1.0 / fs, &mut rng.inner).py()?))
}

/// The field of a laser: power, Wiener phase noise, relative intensity noise and
/// frequency offset (OptiCommPy `basicLaserModel`).
///
/// Args:
///     n (int): number of samples.
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator.
///     power_dbm (float): output power (dBm).
///     linewidth_hz (float): linewidth (Hz).
///     rin_var (float): variance of the relative intensity noise.
///     freq_shift_hz (float): frequency offset (Hz).
///
/// Returns:
///     numpy.ndarray: `n` samples of the field (√W, complex128).
#[pyfunction]
#[pyo3(signature = (n, fs, rng, *, power_dbm = 10.0, linewidth_hz = 100e3, rin_var = 0.0, freq_shift_hz = 0.0))]
#[allow(clippy::too_many_arguments)]
fn laser<'py>(
    py: Python<'py>,
    n: i64,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
    power_dbm: f64,
    linewidth_hz: f64,
    rin_var: f64,
    freq_shift_hz: f64,
) -> PyResult<CArr<'py>> {
    let n = count(n, "n")?;
    Ok(carr(py, Laser { power_dbm, linewidth_hz, rin_var, freq_shift_hz }.field(n, fs, &mut rng.inner).py()?))
}

/// Phase modulator `E e^{jπu/V_π}` (OptiCommPy `pm`).
///
/// Args:
///     e (array_like): the optical field (√W).
///     u (array_like): the driving voltage (V), one value per sample.
///     vpi (float): the half-wave voltage `V_π` (V).
///
/// Returns:
///     numpy.ndarray: the modulated field (complex128).
#[pyfunction]
#[pyo3(signature = (e, u, *, vpi = 2.0))]
fn phase_modulator<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    vpi: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, devices::phase_modulator(&cvec(e)?, &fvec(u)?, vpi).py()?))
}

/// Push-pull Mach–Zehnder modulator (OptiCommPy `mzm`):
/// `E [c_I cos θ + j c_Q sin θ]` with `θ = π(u + V_b)/(2V_π)` and the arm coefficients
/// of the extinction ratio (`cos θ` for an infinite one); the bias `-V_π` is the null
/// (carrier suppression), `-V_π/2` the quadrature point.
///
/// Args:
///     e (array_like): the optical field (√W).
///     u (array_like): the driving voltage (V), one value per sample.
///     vpi (float): the half-wave voltage `V_π` (V).
///     vb (float): the bias voltage `V_b` (V).
///     er_db (float): the extinction ratio (dB).
///
/// Returns:
///     numpy.ndarray: the modulated field (complex128).
#[pyfunction]
#[pyo3(signature = (e, u, *, vpi = 2.0, vb = -1.0, er_db = 60.0))]
fn mzm<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    vpi: f64,
    vb: f64,
    er_db: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Mzm { vpi, vb, er_db }.modulate(&cvec(e)?, &fvec(u)?).py()?))
}

/// IQ modulator (OptiCommPy `iqm`): two MZMs driven by `Re u` and `Im u` on the halves
/// of the field and recombined with the phase `π V_φ/V_π`.
///
/// Args:
///     e (array_like): the optical field (√W).
///     u (array_like): the complex driving signal (V), one value per sample.
///     vpi (float): the half-wave voltage `V_π` of both MZMs (V).
///     vb_i (float): the bias of the in-phase MZM (V).
///     vb_q (float): the bias of the quadrature MZM (V).
///     vphi (float): the voltage of the phase shifter `V_φ` (V).
///     er_i_db (float): the extinction ratio of the in-phase MZM (dB).
///     er_q_db (float): the extinction ratio of the quadrature MZM (dB).
///
/// Returns:
///     numpy.ndarray: the modulated field (complex128).
#[pyfunction]
#[pyo3(signature = (e, u, *, vpi = 2.0, vb_i = -2.0, vb_q = -2.0, vphi = 1.0, er_i_db = 60.0, er_q_db = 60.0))]
#[allow(clippy::too_many_arguments)]
fn iqm<'py>(
    py: Python<'py>,
    e: &Bound<'py, PyAny>,
    u: &Bound<'py, PyAny>,
    vpi: f64,
    vb_i: f64,
    vb_q: f64,
    vphi: f64,
    er_i_db: f64,
    er_q_db: f64,
) -> PyResult<CArr<'py>> {
    let m = Iqm { vpi, vb_i, vb_q, vphi, er_i_db, er_q_db };
    Ok(carr(py, m.modulate(&cvec(e)?, &cvec(u)?).py()?))
}

/// A PIN photodiode (OptiCommPy `photodiode`): responsivity, shot noise
/// `2q(I + I_d)B`, thermal noise `4k_BTB/R_L`, and a low-pass FIR of bandwidth `B`.
///
/// Every attribute is readable and writable; `ideal=True` gives `R|E|²` without noise
/// nor filter.
///
/// Args:
///     responsivity (float): responsivity (A/W).
///     temperature_c (float): temperature (°C).
///     dark_current (float): dark current (A).
///     load_ohm (float): load resistance (Ω).
///     bandwidth_hz (float): electrical bandwidth `B` (Hz).
///     n_taps (int): taps of the bandwidth-limiting filter (made odd).
///     filter_kind (str): the filter, `"rect"` or `"gauss"` (see `low_pass_fir`).
///     shot_noise (bool): add shot noise.
///     thermal_noise (bool): add thermal noise.
///     bandwidth_limitation (bool): filter to the bandwidth.
///     ideal (bool): ideal square-law detection (overrides the three flags above).
#[pyclass(name = "Photodiode", module = "comrs", get_all, set_all)]
#[derive(Clone)]
pub struct PyPhotodiode {
    /// float: Responsivity (A/W) (read and write).
    pub responsivity: f64,
    /// float: Temperature (°C) (read and write).
    pub temperature_c: f64,
    /// float: Dark current (A) (read and write).
    pub dark_current: f64,
    /// float: Load resistance (Ω) (read and write).
    pub load_ohm: f64,
    /// float: Electrical bandwidth `B` (Hz) (read and write).
    pub bandwidth_hz: f64,
    /// int: Taps of the bandwidth-limiting filter (made odd) (read and write).
    pub n_taps: usize,
    /// str: The filter, `"rect"` or `"gauss"` (read and write).
    pub filter_kind: String,
    /// bool: Add shot noise (read and write).
    pub shot_noise: bool,
    /// bool: Add thermal noise (read and write).
    pub thermal_noise: bool,
    /// bool: Filter to the bandwidth (read and write).
    pub bandwidth_limitation: bool,
    /// bool: Ideal square-law detection (overrides the three flags above) (read and write).
    pub ideal: bool,
}

impl PyPhotodiode {
    fn to_rust(&self) -> PyResult<Photodiode> {
        let filter: LowPass = self.filter_kind.parse().py()?;
        Ok(Photodiode {
            responsivity: self.responsivity,
            temperature_c: self.temperature_c,
            dark_current: self.dark_current,
            load_ohm: self.load_ohm,
            bandwidth_hz: self.bandwidth_hz,
            n_taps: self.n_taps,
            filter,
            shot_noise: self.shot_noise,
            thermal_noise: self.thermal_noise,
            bandwidth_limitation: self.bandwidth_limitation,
            ideal: self.ideal,
        })
    }
}

#[pymethods]
impl PyPhotodiode {
    #[new]
    #[pyo3(signature = (*, responsivity = 1.0, temperature_c = 25.0, dark_current = 5e-9, load_ohm = 50.0, bandwidth_hz = 30e9, n_taps = 255, filter_kind = "rect".to_string(), shot_noise = true, thermal_noise = true, bandwidth_limitation = true, ideal = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        responsivity: f64,
        temperature_c: f64,
        dark_current: f64,
        load_ohm: f64,
        bandwidth_hz: f64,
        n_taps: usize,
        filter_kind: String,
        shot_noise: bool,
        thermal_noise: bool,
        bandwidth_limitation: bool,
        ideal: bool,
    ) -> Self {
        PyPhotodiode {
            responsivity,
            temperature_c,
            dark_current,
            load_ohm,
            bandwidth_hz,
            n_taps,
            filter_kind,
            shot_noise,
            thermal_noise,
            bandwidth_limitation,
            ideal,
        }
    }

    /// Photocurrent of an optical field (one polarisation).
    ///
    /// Args:
    ///     e (array_like): the field (√W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng): the generator of the noises.
    ///
    /// Returns:
    ///     numpy.ndarray: the photocurrent (A, float64).
    ///
    /// Raises:
    ///     ValueError: if the detector is not ideal and `fs < 2B`.
    fn detect<'py>(
        &self,
        py: Python<'py>,
        e: &Bound<'py, PyAny>,
        fs: f64,
        mut rng: PyRefMut<'py, PyRng>,
    ) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.to_rust()?.detect(&cvec(e)?, fs, &mut rng.inner).py()?))
    }

    /// Photocurrent of an optical power.
    ///
    /// Args:
    ///     p (array_like): the optical power of every sample (W).
    ///     fs (float): sampling rate (Hz).
    ///     rng (Rng): the generator of the noises.
    ///
    /// Returns:
    ///     numpy.ndarray: the photocurrent (A, float64).
    fn detect_power<'py>(
        &self,
        py: Python<'py>,
        p: &Bound<'py, PyAny>,
        fs: f64,
        mut rng: PyRefMut<'py, PyRng>,
    ) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.to_rust()?.detect_power(&fvec(p)?, fs, &mut rng.inner).py()?))
    }
}

fn photodiode_or_ideal(pd: Option<&PyPhotodiode>) -> PyResult<Photodiode> {
    pd.map_or(Ok(Photodiode::ideal()), PyPhotodiode::to_rust)
}

/// Balanced detection `i(E₁) - i(E₂)`.
///
/// Args:
///     e1 (array_like): the field on the first photodiode (√W).
///     e2 (array_like): the field on the second photodiode (√W).
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator of the noises.
///     photodiode (Photodiode, optional): the photodiodes (ideal by default).
///
/// Returns:
///     numpy.ndarray: the photocurrent (A, float64).
#[pyfunction]
#[pyo3(signature = (e1, e2, fs, rng, *, photodiode = None))]
fn balanced_photodiode<'py>(
    py: Python<'py>,
    e1: &Bound<'py, PyAny>,
    e2: &Bound<'py, PyAny>,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
    photodiode: Option<&PyPhotodiode>,
) -> PyResult<FArr<'py>> {
    let pd = photodiode_or_ideal(photodiode)?;
    Ok(farr(py, devices::balanced_photodiode(&cvec(e1)?, &cvec(e2)?, &pd, fs, &mut rng.inner).py()?))
}

/// The 90° optical hybrid (OptiCommPy `hybrid_2x4_90deg`): the four outputs
/// `(E_s - E_lo)/2`, `j(E_s + E_lo)/2`, `(jE_s - E_lo)/2` and `(-E_s + jE_lo)/2`.
///
/// Args:
///     es (array_like): the signal field (√W).
///     elo (array_like): the field of the local oscillator (√W).
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: the four
///     output fields.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn hybrid_2x4<'py>(
    py: Python<'py>,
    es: &Bound<'py, PyAny>,
    elo: &Bound<'py, PyAny>,
) -> PyResult<(CArr<'py>, CArr<'py>, CArr<'py>, CArr<'py>)> {
    let [a, b, c, d] = devices::hybrid_2x4(&cvec(es)?, &cvec(elo)?).py()?;
    Ok((carr(py, a), carr(py, b), carr(py, c), carr(py, d)))
}

/// Single-polarisation coherent receiver (OptiCommPy `coherentReceiver`): the hybrid
/// and two balanced photodiodes; equals `R E_s E_lo*` with ideal photodiodes.
///
/// Args:
///     es (array_like): the signal field (√W).
///     elo (array_like): the field of the local oscillator (√W).
///     fs (float): sampling rate (Hz).
///     rng (Rng): the generator of the noises.
///     photodiode (Photodiode, optional): the photodiodes (ideal by default).
///
/// Returns:
///     numpy.ndarray: the in-phase and quadrature photocurrents as one complex signal
///     (A).
#[pyfunction]
#[pyo3(signature = (es, elo, fs, rng, *, photodiode = None))]
fn coherent_receiver<'py>(
    py: Python<'py>,
    es: &Bound<'py, PyAny>,
    elo: &Bound<'py, PyAny>,
    fs: f64,
    mut rng: PyRefMut<'py, PyRng>,
    photodiode: Option<&PyPhotodiode>,
) -> PyResult<CArr<'py>> {
    let pd = photodiode_or_ideal(photodiode)?;
    Ok(carr(py, devices::coherent_receiver(&cvec(es)?, &cvec(elo)?, &pd, fs, &mut rng.inner).py()?))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhotodiode>()?;
    for f in [
        wrap_pyfunction!(awgn, m)?,
        wrap_pyfunction!(awgn_ebn0, m)?,
        wrap_pyfunction!(phase_noise, m)?,
        wrap_pyfunction!(laser, m)?,
        wrap_pyfunction!(phase_modulator, m)?,
        wrap_pyfunction!(mzm, m)?,
        wrap_pyfunction!(iqm, m)?,
        wrap_pyfunction!(balanced_photodiode, m)?,
        wrap_pyfunction!(hybrid_2x4, m)?,
        wrap_pyfunction!(coherent_receiver, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}

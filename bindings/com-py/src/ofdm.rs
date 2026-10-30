//! OFDM, PAPR reduction, power amplifiers, digital predistortion and spectral metrics.

use crate::comm::PyConstellation;
use crate::{CArr, FArr, IArr, IntoPy, carr, cvec, farr, fvec, iarr, map_real, uvec};
use comrs::dsp::compensation::dpd::MemoryPolynomialDpd;
use comrs::hardware::amplifier::{MemoryPolynomial, Rapp, Saleh};
use comrs::metrics::spectrum;
use comrs::waveform::ofdm::correction;
use comrs::waveform::ofdm::{NoReduction, Ofdm, PaprReduction};
use comrs::waveform::papr::{self, Ace, Clip, ClipFilter, Companding, Partition, Pts, Slm, ToneReservation};
use num_complex::Complex64 as C64;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// An OFDM (or DFT-spread OFDM) modulator and demodulator.
///
/// `n_subcarriers` active subcarriers centred on the carrier, out of a grid of `n_fft`;
/// the IFFT has `oversampling · n_fft` points (4 by default, as the PAPR measurements
/// need), and the samples have unit average power. With `dft_spread`, the data are
/// first spread by a unitary DFT (localised SC-FDMA).
///
/// Args:
///     n_fft (int): grid size `N`.
///     n_subcarriers (int): active subcarriers `K ≤ N` (the name of the ``ofdm_tx``
///         block).
///     oversampling (int): `L`.
///     n_cp (int): cyclic prefix, in Nyquist-rate samples.
///     dc_null (bool): leave the DC subcarrier empty.
///     dft_spread (bool): DFT-spread OFDM.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> ofdm = comrs.Ofdm(64, 48, n_cp=8)
///     >>> ofdm.n_subcarriers, ofdm.bins.shape
///     (48, (48,))
///     >>> d = np.exp(2j * np.pi * np.arange(96) / 7)
///     >>> np.allclose(ofdm.demodulate(ofdm.modulate(d)), d)
///     True
#[pyclass(name = "Ofdm", module = "comrs")]
pub struct PyOfdm {
    pub(crate) inner: Ofdm,
}

/// A PAPR reduction technique for `Ofdm.modulate` and `Ofdm.demodulate`; build it with
/// one of the static methods (`clip`, `clip_filter`, `slm`, `pts`, `tone_reservation`,
/// `ace`, `mu_law`, `exponential`).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> ofdm = comrs.Ofdm(64, 64)
///     >>> d = np.exp(2j * np.pi * comrs.Rng(1).uniform(64 * 20))
///     >>> x, side = ofdm.modulate(d, comrs.PaprReduction.slm(8))
///     >>> bool(np.median(ofdm.papr_db(x)) < np.median(ofdm.papr_db(ofdm.modulate(d))))
///     True
///     >>> np.allclose(ofdm.demodulate(x, comrs.PaprReduction.slm(8), side=side), d)
///     True
#[pyclass(name = "PaprReduction", module = "comrs")]
pub struct PyPaprReduction {
    inner: Box<dyn PaprReduction + Send + Sync>,
    name: String,
}

fn reduction<'a>(r: &'a Option<PyRef<'_, PyPaprReduction>>) -> &'a (dyn PaprReduction + Send + Sync) {
    match r {
        Some(r) => r.inner.as_ref(),
        None => &NoReduction,
    }
}

#[pymethods]
impl PyOfdm {
    #[new]
    #[pyo3(signature = (n_fft, n_subcarriers, *, oversampling=4, n_cp=0, dc_null=false, dft_spread=false))]
    fn new(
        n_fft: usize,
        n_subcarriers: usize,
        oversampling: usize,
        n_cp: usize,
        dc_null: bool,
        dft_spread: bool,
    ) -> PyResult<Self> {
        let inner =
            Ofdm { n_active: n_subcarriers, oversampling, n_cp, dc_null, dft_spread, ..Ofdm::new(n_fft, 1).py()? };
        inner.check().py()?;
        Ok(PyOfdm { inner })
    }

    /// int: Grid size `N`.
    #[getter]
    fn n_fft(&self) -> usize {
        self.inner.n_fft
    }

    /// int: Active subcarriers `K`.
    #[getter]
    fn n_subcarriers(&self) -> usize {
        self.inner.n_active
    }

    /// int: Oversampling factor `L`.
    #[getter]
    fn oversampling(&self) -> usize {
        self.inner.oversampling
    }

    /// int: Cyclic prefix (Nyquist-rate samples).
    #[getter]
    fn n_cp(&self) -> usize {
        self.inner.n_cp
    }

    /// numpy.ndarray: FFT bins of the active subcarriers (int64), for the channel
    /// frequency responses (`ChannelTaps.frequency_response`).
    #[getter]
    fn bins<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, (0..self.inner.n_active).map(|i| self.inner.bin(i)).collect())
    }

    /// int: Samples per block, prefix included.
    #[getter]
    fn block_len(&self) -> usize {
        self.inner.block_len()
    }

    /// int: Samples of the useful part of a block, `L N`.
    #[getter]
    fn fft_len(&self) -> usize {
        self.inner.fft_len()
    }

    /// Data symbols per block with a reduction technique.
    ///
    /// Args:
    ///     reduction (PaprReduction, optional): the technique.
    ///
    /// Returns:
    ///     int: `K` without a technique, fewer with tone reservation.
    #[pyo3(signature = (reduction=None))]
    fn data_len(&self, reduction: Option<PyRef<'_, PyPaprReduction>>) -> usize {
        self::reduction(&reduction).data_len(&self.inner)
    }

    /// Modulate data symbols (whole blocks).
    ///
    /// Args:
    ///     data (array_like): a multiple of `data_len(reduction)` data symbols.
    ///     reduction (PaprReduction, optional): a PAPR reduction technique.
    ///
    /// Returns:
    ///     numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]: the samples
    ///     (complex128); with a reduction technique, the samples and the side
    ///     information of every block (the SLM or PTS candidate, int64).
    #[pyo3(signature = (data, reduction=None))]
    fn modulate(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        reduction: Option<PyRef<'_, PyPaprReduction>>,
    ) -> PyResult<Py<PyAny>> {
        let d = cvec(data)?;
        let with = reduction.is_some();
        let r = self::reduction(&reduction);
        let (x, side) = py.detach(|| self.inner.modulate_with(&d, r)).py()?;
        if with {
            let side: Vec<i64> = side.into_iter().map(|s| s as i64).collect();
            Ok((carr(py, x), side.into_pyarray(py)).into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(carr(py, x).into_any().unbind())
        }
    }

    /// Demodulate whole blocks.
    ///
    /// Args:
    ///     samples (array_like): a multiple of `block_len` samples.
    ///     reduction (PaprReduction, optional): the technique of the transmitter (the
    ///         expansion of companding is applied).
    ///     side (array_like, optional): the side information of SLM and PTS.
    ///
    /// Returns:
    ///     numpy.ndarray: the data symbols (complex128).
    #[pyo3(signature = (samples, reduction=None, *, side=None))]
    fn demodulate<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
        reduction: Option<PyRef<'py, PyPaprReduction>>,
        side: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<CArr<'py>> {
        let x = cvec(samples)?;
        let blocks = x.len() / self.inner.block_len().max(1);
        let side = match side {
            Some(s) => uvec(s)?,
            None => vec![0; blocks],
        };
        let r = self::reduction(&reduction);
        Ok(carr(py, py.detach(|| self.inner.demodulate_with(&x, r, &side)).py()?))
    }

    /// The PAPR of every block (the prefix is excluded).
    ///
    /// Args:
    ///     samples (array_like): a multiple of `block_len` samples.
    ///
    /// Returns:
    ///     numpy.ndarray: the PAPR of every block (dB).
    fn papr_db<'py>(&self, py: Python<'py>, samples: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.papr_db_per_block(&cvec(samples)?)))
    }

    fn __repr__(&self) -> String {
        let o = &self.inner;
        format!(
            "Ofdm(n_fft={}, n_subcarriers={}, oversampling={}, n_cp={}, dc_null={}, dft_spread={})",
            o.n_fft, o.n_active, o.oversampling, o.n_cp, o.dc_null, o.dft_spread
        )
    }
}

fn boxed(r: impl PaprReduction + Send + Sync + 'static, name: String) -> PyPaprReduction {
    PyPaprReduction { inner: Box::new(r), name }
}

#[pymethods]
impl PyPaprReduction {
    /// Clipping of the envelope above the average power (Li & Cimini 1998).
    ///
    /// Args:
    ///     ratio_db (float): the clipping ratio (dB above the average power).
    ///
    /// Returns:
    ///     PaprReduction: the technique.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PaprReduction.clip(4.0)
    ///     PaprReduction.clip(4 dB)
    #[staticmethod]
    fn clip(ratio_db: f64) -> Self {
        boxed(Clip { ratio_db }, format!("clip({ratio_db} dB)"))
    }

    /// Repeated clipping and out-of-band filtering (Armstrong 2002).
    ///
    /// Args:
    ///     ratio_db (float): the clipping ratio (dB).
    ///     iterations (int): clipping and filtering passes.
    ///
    /// Returns:
    ///     PaprReduction: the technique.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PaprReduction.clip_filter(4.0, iterations=2)
    ///     PaprReduction.clip_filter(4 dB, 2)
    #[staticmethod]
    #[pyo3(signature = (ratio_db, *, iterations=4))]
    fn clip_filter(ratio_db: f64, iterations: usize) -> Self {
        boxed(ClipFilter { ratio_db, iterations }, format!("clip_filter({ratio_db} dB, {iterations})"))
    }

    /// Selected mapping (Bäuml, Fischer & Huber 1996).
    ///
    /// Args:
    ///     candidates (int): number of phase sequences.
    ///     seed (int): seed of the phase sequences.
    ///
    /// Returns:
    ///     PaprReduction: the technique (its side information is the candidate).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PaprReduction.slm(8)
    ///     PaprReduction.slm(8)
    #[staticmethod]
    #[pyo3(signature = (candidates, *, seed=0))]
    fn slm(candidates: usize, seed: u64) -> Self {
        boxed(Slm { candidates, seed }, format!("slm({candidates})"))
    }

    /// Partial transmit sequences (Müller & Huber 1997).
    ///
    /// Args:
    ///     subblocks (int): number of subblocks.
    ///     phases (int): number of phase factors.
    ///     partition (str): `"adjacent"`, `"interleaved"` or `"pseudorandom"`.
    ///     seed (int): seed of the pseudorandom partition.
    ///
    /// Returns:
    ///     PaprReduction: the technique (its side information is the candidate).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PaprReduction.pts(4, phases=2)
    ///     PaprReduction.pts(4, 2)
    #[staticmethod]
    #[pyo3(signature = (subblocks, *, phases=4, partition="pseudorandom", seed=0))]
    fn pts(subblocks: usize, phases: usize, partition: &str, seed: u64) -> PyResult<Self> {
        let partition = Partition::from_name(partition, seed).py()?;
        Ok(boxed(Pts { subblocks, phases, partition }, format!("pts({subblocks}, {phases})")))
    }

    /// Tone reservation on random tones (Tellado 2000).
    ///
    /// Args:
    ///     n_subcarriers (int): active subcarriers of the modulator.
    ///     n_tones (int): reserved tones among them.
    ///     ratio_db (float): the target clipping ratio (dB).
    ///     seed (int): seed of the choice of the tones.
    ///     iterations (int): iterations of the gradient search.
    ///     step (float): its step.
    ///
    /// Returns:
    ///     PaprReduction: the technique (`Ofdm.data_len` drops the reserved tones).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> ofdm = comrs.Ofdm(128, 128)
    ///     >>> ofdm.data_len(comrs.PaprReduction.tone_reservation(128, 12, 4.0))
    ///     116
    #[staticmethod]
    #[pyo3(signature = (n_subcarriers, n_tones, ratio_db, *, seed=0, iterations=20, step=1.0))]
    fn tone_reservation(
        n_subcarriers: usize,
        n_tones: usize,
        ratio_db: f64,
        seed: u64,
        iterations: usize,
        step: f64,
    ) -> Self {
        let tr =
            ToneReservation { iterations, step, ..ToneReservation::random(n_subcarriers, n_tones, seed, ratio_db) };
        boxed(tr, format!("tone_reservation({n_tones} tones, {ratio_db} dB)"))
    }

    /// Active constellation extension, clipping-based (Krongold & Jones 2003).
    ///
    /// Args:
    ///     constellation (Constellation): the alphabet of the data.
    ///     ratio_db (float): the clipping ratio (dB).
    ///     iterations (int): iterations.
    ///     step (float): step of the extension.
    ///
    /// Returns:
    ///     PaprReduction: the technique.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PaprReduction.ace(comrs.Constellation("qam", 4), 4.0)
    ///     PaprReduction.ace(4 dB, 10)
    #[staticmethod]
    #[pyo3(signature = (constellation, ratio_db, *, iterations=10, step=1.0))]
    fn ace(constellation: &PyConstellation, ratio_db: f64, iterations: usize, step: f64) -> Self {
        let ace = Ace { constellation: constellation.inner.clone(), ratio_db, iterations, step };
        boxed(ace, format!("ace({ratio_db} dB, {iterations})"))
    }

    /// μ-law companding (Wang, Tjhung & Ng 1999).
    ///
    /// Args:
    ///     mu (float): the parameter `μ`.
    ///     peak_db (float): the peak of the compander (dB above the average power).
    ///
    /// Returns:
    ///     PaprReduction: the technique (`Ofdm.demodulate` expands).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PaprReduction.mu_law(255.0)
    ///     PaprReduction.mu_law(255)
    #[staticmethod]
    #[pyo3(signature = (mu, *, peak_db=9.0))]
    fn mu_law(mu: f64, peak_db: f64) -> Self {
        boxed(Companding::MuLaw { mu, peak_db }, format!("mu_law({mu})"))
    }

    /// Exponential companding (Jiang, Yang & Song 2005).
    ///
    /// Args:
    ///     degree (float): the degree of the compander.
    ///
    /// Returns:
    ///     PaprReduction: the technique (`Ofdm.demodulate` expands).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PaprReduction.exponential(2.0)
    ///     PaprReduction.exponential(2)
    #[staticmethod]
    #[pyo3(signature = (degree=2.0))]
    fn exponential(degree: f64) -> Self {
        boxed(Companding::Exponential { degree }, format!("exponential({degree})"))
    }

    fn __repr__(&self) -> String {
        format!("PaprReduction.{}", self.name)
    }
}

/// The PAPR of a block, or of every block of `block_len` samples.
///
/// Args:
///     x (array_like): the samples.
///     block_len (int, optional): the length of a block.
///
/// Returns:
///     float | numpy.ndarray: the PAPR (dB), of `x` or of every block.
#[pyfunction]
#[pyo3(signature = (x, *, block_len=None))]
fn papr_db(py: Python<'_>, x: &Bound<'_, PyAny>, block_len: Option<usize>) -> PyResult<Py<PyAny>> {
    let v = cvec(x)?;
    match block_len {
        None => Ok(comrs::math::papr_db(&v).into_pyobject(py)?.into_any().unbind()),
        Some(n) if n > 0 => Ok(farr(py, v.chunks_exact(n).map(comrs::math::papr_db).collect()).into_any().unbind()),
        Some(_) => Err(PyValueError::new_err("block_len must be positive")),
    }
}

/// The empirical complementary distribution `P(X > t)` of `values` at every threshold.
///
/// Args:
///     values (array_like): the samples of `X`.
///     thresholds (array_like): the thresholds `t`.
///
/// Returns:
///     numpy.ndarray: `P(X > t)` at every threshold (float64).
#[pyfunction]
fn ccdf<'py>(py: Python<'py>, values: &Bound<'py, PyAny>, thresholds: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
    Ok(farr(py, spectrum::ccdf(&fvec(values)?, &fvec(thresholds)?)))
}

/// The theoretical CCDF of the PAPR of OFDM with `n` subcarriers:
/// `1 - exp(-n √(πγ/3) e^(-γ))` for the continuous signal (Ochiai & Imai 2001), or
/// `1 - (1 - e^(-γ))^n` at the Nyquist rate (van Nee & Prasad 2000); with
/// `candidates`, the CCDF after SLM (raised to that power).
///
/// Args:
///     n (int): number of subcarriers.
///     papr_db (float | array_like): the thresholds `γ` (dB).
///     oversampled (bool): the continuous signal (else the Nyquist-rate samples).
///     candidates (int): SLM candidates (1 without SLM).
///
/// Returns:
///     float | numpy.ndarray: `P(PAPR > γ)` at every threshold.
///
/// Examples:
///     >>> import comrs
///     >>> p = comrs.theory_papr_ccdf(256, [8.0, 10.0])
///     >>> bool(p[0] > p[1] > 0)
///     True
#[pyfunction]
#[pyo3(signature = (n, papr_db, *, oversampled=true, candidates=1))]
fn theory_papr_ccdf(n: usize, papr_db: &Bound<'_, PyAny>, oversampled: bool, candidates: usize) -> PyResult<Py<PyAny>> {
    map_real(papr_db, |t| {
        Ok(if oversampled { papr::ccdf_slm(n, candidates, t) } else { papr::ccdf_slm_nyquist(n, candidates, t) })
    })
}

/// The Bussgang decomposition `y = α x + d` of clipping at `ratio_db` above the average
/// power of a complex Gaussian signal (Ochiai & Imai 2002).
///
/// Args:
///     ratio_db (float): the clipping ratio (dB above the average power).
///
/// Returns:
///     dict: ``gain`` (α), ``output_power``, ``distortion`` (σ_d²), ``sdr_db``.
#[pyfunction]
fn clipping_bussgang(py: Python<'_>, ratio_db: f64) -> PyResult<Bound<'_, PyDict>> {
    let b = comrs::dsp::nonlinear::clipping_bussgang(ratio_db);
    let d = PyDict::new(py);
    d.set_item("gain", b.gain.re)?;
    d.set_item("output_power", b.output_power)?;
    d.set_item("distortion", b.distortion)?;
    d.set_item("sdr_db", b.sdr_db(1.0))?;
    Ok(d)
}

/// Welch's power spectral density (periodic Hann window, 50 % overlap), centred.
///
/// Args:
///     x (array_like): the signal.
///     fs (float): sampling rate (Hz).
///     nfft (int): length of a segment.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the frequencies (Hz) and the density
///     (power per Hz), which integrates to the average power.
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, *, nfft=1024))]
fn psd<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64, nfft: usize) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let v = cvec(x)?;
    let (f, s) = py.detach(|| spectrum::psd(&v, fs, nfft)).py()?;
    Ok((farr(py, f), farr(py, s)))
}

/// The averaged periodogram of non-overlapping segments of `nfft` samples (no window):
/// with `nfft` the block length of a cyclic signal (OFDM without prefix), it has no
/// leakage, and its out-of-band part is the regrowth of a nonlinearity.
///
/// Args:
///     x (array_like): the signal.
///     fs (float): sampling rate (Hz).
///     nfft (int): length of a segment.
///
/// Returns:
///     tuple[numpy.ndarray, numpy.ndarray]: the frequencies (Hz) and the density
///     (power per Hz).
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, *, nfft=1024))]
fn periodogram<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, fs: f64, nfft: usize) -> PyResult<(FArr<'py>, FArr<'py>)> {
    let v = cvec(x)?;
    let (f, s) = py.detach(|| spectrum::periodogram(&v, fs, nfft)).py()?;
    Ok((farr(py, f), farr(py, s)))
}

/// The adjacent channel leakage ratio (dB) of a density (`psd` or `periodogram`): power
/// in `[-B/2, B/2)` over the larger power of the channels of width `B` centred at
/// `±offset`.
///
/// Args:
///     freqs (array_like): the frequencies of the density (Hz).
///     density (array_like): the density.
///     bandwidth (float): the channel width `B` (Hz).
///     offset (float): the offset of the adjacent channels (Hz).
///
/// Returns:
///     float: the ACLR (dB).
#[pyfunction]
fn aclr_db(freqs: &Bound<'_, PyAny>, density: &Bound<'_, PyAny>, bandwidth: f64, offset: f64) -> PyResult<f64> {
    spectrum::aclr_db(&fvec(freqs)?, &fvec(density)?, bandwidth, offset).py()
}

/// Rapp's solid-state amplifier: `A(r) = g r / (1 + (g r / A_sat)^(2p))^(1/(2p))`, no
/// AM/PM (Rapp 1991).
///
/// Args:
///     x (array_like): the input envelope.
///     gain (float): the small-signal gain `g`.
///     saturation (float): the saturation amplitude `A_sat`.
///     smoothness (float): the smoothness `p`.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
#[pyfunction]
#[pyo3(signature = (x, *, gain=1.0, saturation=1.0, smoothness=2.0))]
fn pa_rapp<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    gain: f64,
    saturation: f64,
    smoothness: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Rapp { gain, saturation, smoothness }.amplify(&cvec(x)?).py()?))
}

/// The static predistorter of Rapp's amplifier: the inverse AM/AM, up to `max_output`
/// times the saturation (the cascade is then an ideal envelope limiter).
///
/// Args:
///     x (array_like): the input envelope.
///     gain (float): the small-signal gain of the amplifier.
///     saturation (float): its saturation amplitude.
///     smoothness (float): its smoothness.
///     max_output (float): the largest output, relative to the saturation.
///
/// Returns:
///     numpy.ndarray: the predistorted input (complex128).
#[pyfunction]
#[pyo3(signature = (x, *, gain=1.0, saturation=1.0, smoothness=2.0, max_output=0.99))]
fn rapp_predistort<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    gain: f64,
    saturation: f64,
    smoothness: f64,
    max_output: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Rapp { gain, saturation, smoothness }.predistort(&cvec(x)?, max_output).py()?))
}

/// Saleh's travelling-wave tube amplifier (AM/AM and AM/PM; Saleh 1981, default
/// coefficients of his fit).
///
/// Args:
///     x (array_like): the input envelope.
///     alpha_a (float): AM/AM coefficient `α_a`.
///     beta_a (float): AM/AM coefficient `β_a`.
///     alpha_p (float): AM/PM coefficient `α_φ`.
///     beta_p (float): AM/PM coefficient `β_φ`.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
#[pyfunction]
#[pyo3(signature = (x, *, alpha_a=2.1587, beta_a=1.1517, alpha_p=4.0033, beta_p=9.1040))]
fn pa_saleh<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    alpha_a: f64,
    beta_a: f64,
    alpha_p: f64,
    beta_p: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, Saleh { alpha_a, beta_a, alpha_p, beta_p }.amplify(&cvec(x)?).py()?))
}

/// The memory-polynomial amplifier of Ding et al. (2004): orders 1, 3, 5, memory 3.
///
/// Args:
///     x (array_like): the input envelope.
///
/// Returns:
///     numpy.ndarray: the output (complex128).
#[pyfunction]
fn pa_memory_polynomial<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
    Ok(carr(py, MemoryPolynomial::ding2004().amplify(&cvec(x)?)))
}

/// A memory-polynomial digital predistorter, trained by indirect learning (Eun & Powers
/// 1997; Ding et al. 2004; Morgan et al. 2006).
///
/// Args:
///     orders (list[int]): nonlinearity orders (odd: 1, 3, 5, …).
///     memory (int): memory depth `M` (samples).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> g = 1.0513 + 0.0904j                      # small-signal gain of the amplifier
///     >>> x = 0.2 * comrs.Rng(1).complex_noise(20_000, 1.0)
///     >>> dpd = comrs.MemoryPolynomialDpd([1, 3, 5], 3)
///     >>> dpd.train_indirect(x, comrs.pa_memory_polynomial, gain=g, iterations=2)
///     >>> nmse = lambda y: float(np.mean(np.abs(y / g - x) ** 2) / np.mean(np.abs(x) ** 2))
///     >>> nmse(comrs.pa_memory_polynomial(dpd.apply(x))) < nmse(comrs.pa_memory_polynomial(x)) / 5
///     True
///
/// The drive must stay where the amplifier is monotonic: beyond, the post-inverse that
/// indirect learning fits has no meaning and the iterations diverge.
#[pyclass(name = "MemoryPolynomialDpd", module = "comrs")]
pub struct PyDpd {
    inner: MemoryPolynomialDpd,
}

#[pymethods]
impl PyDpd {
    #[new]
    #[pyo3(signature = (orders=vec![1, 3, 5], memory=3))]
    fn new(orders: Vec<u32>, memory: usize) -> Self {
        PyDpd { inner: MemoryPolynomialDpd::new(orders, memory) }
    }

    /// numpy.ndarray: The coefficients, order after order, tap after tap (complex128).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> CArr<'py> {
        carr(py, self.inner.coefficients())
    }

    /// Predistort a block.
    ///
    /// Args:
    ///     x (array_like): the input envelope.
    ///
    /// Returns:
    ///     numpy.ndarray: the predistorted envelope (complex128).
    fn apply<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<CArr<'py>> {
        let v = cvec(x)?;
        Ok(carr(py, py.detach(|| self.inner.apply(&v))))
    }

    /// Fit the polynomial mapping `x` to `y` by regularised least squares.
    ///
    /// Args:
    ///     x (array_like): the input of the polynomial.
    ///     y (array_like): the output it should give.
    ///     regularization (float): the Tikhonov regularisation.
    #[pyo3(signature = (x, y, *, regularization=1e-9))]
    fn fit(&mut self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, regularization: f64) -> PyResult<()> {
        self.inner.fit(&cvec(x)?, &cvec(y)?, regularization).py()
    }

    /// Indirect learning against an amplifier.
    ///
    /// Args:
    ///     x (array_like): the training signal.
    ///     pa (callable): the amplifier, an array to an array.
    ///     gain (complex): the target complex gain of the cascade.
    ///     iterations (int): iterations of the indirect learning.
    ///     regularization (float): the regularisation of every fit.
    #[pyo3(signature = (x, pa, *, gain=C64::new(1.0, 0.0), iterations=3, regularization=1e-9))]
    fn train_indirect(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        pa: &Bound<'_, PyAny>,
        gain: C64,
        iterations: usize,
        regularization: f64,
    ) -> PyResult<()> {
        let x = cvec(x)?;
        // the exception raised by `pa`, passed through the Rust error
        let mut raised = None;
        let amplify = |z: &[C64]| {
            pa.call1((carr(py, z.to_vec()),)).and_then(|y| cvec(&y)).map_err(|e| {
                raised = Some(e);
                comrs::Error::Config("the amplifier raised an exception".into())
            })
        };
        let r = self.inner.train_indirect(&x, amplify, gain, iterations, regularization);
        match raised {
            Some(e) => Err(e),
            None => r.py(),
        }
    }
}

/// Pilot-based gain correction limited to `eps` (Cimini 1985, eq. (16)): `r / (max(|p|,
/// ε) e^{j arg p})` for every sample.
///
/// Args:
///     r (array_like): the received samples.
///     pilot (array_like): the complex envelope given by the pilot.
///     eps_db (float): the limit, an amplitude relative to the RMS gain (dB, `20 log10`);
///         `-inf` for unlimited correction.
///
/// Returns:
///     numpy.ndarray: the corrected samples (complex128).
#[pyfunction]
fn gain_limited_correction<'py>(
    py: Python<'py>,
    r: &Bound<'py, PyAny>,
    pilot: &Bound<'py, PyAny>,
    eps_db: f64,
) -> PyResult<CArr<'py>> {
    Ok(carr(py, correction::gain_limited(&cvec(r)?, &cvec(pilot)?, comrs::math::db2amp(eps_db))))
}

/// The signal-to-distortion ratio (dB) of gain-limited correction against a cochannel
/// interferer in Rayleigh fading (Cimini 1985, eq. (25); eq. (20) for an ideal pilot).
///
/// Args:
///     sir_db (float | array_like): the signal-to-interference ratio (dB).
///     eps_db (float): the gain limit relative to the RMS gain (dB, `20 log10`).
///     rho (complex): the correlation `E[Z_d Z_p*]` of the data and pilot envelopes.
///
/// Returns:
///     float | numpy.ndarray: the SDR (dB) at every SIR.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.sdr_gain_limited(25.0, -17.0), 1)
///     17.7
#[pyfunction]
#[pyo3(signature = (sir_db, eps_db, rho = C64::new(1.0, 0.0)))]
fn sdr_gain_limited(sir_db: &Bound<'_, PyAny>, eps_db: f64, rho: C64) -> PyResult<Py<PyAny>> {
    let eps = comrs::math::db2amp(eps_db);
    map_real(sir_db, |s| Ok(comrs::math::lin2db(correction::sdr_gain_limited(comrs::math::db2lin(s), eps, rho))))
}

/// The signal-to-distortion ratio (dB) of the optimum gain correction against a
/// cochannel interferer in Rayleigh fading (Cimini 1985, eq. (21)).
///
/// Args:
///     sir_db (float | array_like): the signal-to-interference ratio (dB).
///
/// Returns:
///     float | numpy.ndarray: the SDR (dB) at every SIR.
#[pyfunction]
fn sdr_optimum_correction(sir_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(sir_db, |s| Ok(comrs::math::lin2db(correction::sdr_optimum(comrs::math::db2lin(s)))))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOfdm>()?;
    m.add_class::<PyPaprReduction>()?;
    m.add_class::<PyDpd>()?;
    m.add_function(wrap_pyfunction!(papr_db, m)?)?;
    m.add_function(wrap_pyfunction!(ccdf, m)?)?;
    m.add_function(wrap_pyfunction!(theory_papr_ccdf, m)?)?;
    m.add_function(wrap_pyfunction!(clipping_bussgang, m)?)?;
    m.add_function(wrap_pyfunction!(psd, m)?)?;
    m.add_function(wrap_pyfunction!(periodogram, m)?)?;
    m.add_function(wrap_pyfunction!(aclr_db, m)?)?;
    m.add_function(wrap_pyfunction!(pa_rapp, m)?)?;
    m.add_function(wrap_pyfunction!(rapp_predistort, m)?)?;
    m.add_function(wrap_pyfunction!(pa_saleh, m)?)?;
    m.add_function(wrap_pyfunction!(pa_memory_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(gain_limited_correction, m)?)?;
    m.add_function(wrap_pyfunction!(sdr_gain_limited, m)?)?;
    m.add_function(wrap_pyfunction!(sdr_optimum_correction, m)?)?;
    Ok(())
}

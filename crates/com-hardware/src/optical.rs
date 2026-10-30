//! Optoelectronic devices (OptiCommPy `optic.models.devices`).
//!
//! # Transmitter
//!
//! - [`Laser`]: a continuous-wave field of power `P` with the Wiener phase noise of its
//!   linewidth, an optional relative intensity noise and frequency offset;
//! - [`phase_modulator`]: `E_out = E_in e^{jπ u/V_π}`;
//! - [`Mzm`]: a push-pull Mach–Zehnder modulator with a finite extinction ratio,
//!   `E_out = E_in [c_I cos θ + j c_Q sin θ]`, `θ = π (u + V_b)/(2 V_π)`, with
//!   `c_{I,Q} = (√(1+γ) ± √(1-γ))/2` and `γ = 2√ER/(ER+1)` (`ER` linear): an infinite
//!   extinction ratio gives the ideal `cos θ`;
//! - [`Iqm`]: two MZMs in parallel, each fed with half of the input field, the Q branch
//!   rotated by the phase modulator biased at `V_φ` (`π/2` for `V_φ = V_π/2`).
//!
//! # Receiver
//!
//! - [`Photodiode`]: photocurrent `i = R |E|²` (summed over the polarisations), with
//!   shot noise `σ² = 2q(i + I_d)B`, thermal noise `σ² = 4kTB/R_L`, both white over the
//!   simulation bandwidth and scaled by `f_s/(2B)`, then a low-pass FIR of bandwidth `B`;
//!   [`Photodiode::ideal`] keeps `R|E|²` only;
//! - [`hybrid_2x4`]: the 90° optical hybrid of OptiCommPy (`opticalHybrid2x4`);
//! - [`coherent_receiver`]: hybrid and two balanced photodiode pairs; with ideal
//!   photodiodes its output is exactly `R E_s E_LO*`;
//! - [`Photodiode::detect_signal`] and [`coherent_receiver_signal`]: the receivers on a
//!   [`Signal`] (the polarisations of a `DualPol` as its streams), the sampling rate read
//!   from the signal;
//! - [`PinFetReceiver`]: the p-i-n photodiode and FET transimpedance amplifier of the
//!   optical wireless receivers, the variances of its shot noise (signal and ambient
//!   light) and thermal noise (feedback resistor and FET channel), and the SNR of on–off
//!   keying with intersymbol interference (Komine & Nakagawa 2004, eqs. (16)–(21)).
//!
//! # Examples
//! ```
//! use com_hardware::optical::{coherent_receiver, Photodiode, Mzm};
//! use com_core::rng::Rng;
//! use com_math::C64;
//! let es = vec![C64::new(0.3, -0.2); 4];
//! let lo = vec![C64::new(1.0, 0.0); 4];
//! let s = coherent_receiver(&es, &lo, &Photodiode::ideal(), 64e9, &mut Rng::new(0)).unwrap();
//! assert!((s[0] - C64::new(0.3, -0.2)).norm() < 1e-12);            // R E_s E_LO*
//! // an ideal MZM at the quadrature point (V_b = -V_π/2) passes 1/√2 of the field,
//! // at the null (V_b = -V_π) nothing
//! let m = Mzm { vpi: 2.0, vb: -1.0, er_db: f64::INFINITY };
//! let y = m.modulate(&[C64::new(1.0, 0.0)], &[0.0]).unwrap();
//! assert!((y[0].norm() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
//! let null = Mzm { vb: -2.0, ..m }.modulate(&[C64::new(1.0, 0.0)], &[0.0]).unwrap();
//! assert!(null[0].norm() < 1e-15);
//! ```

use crate::oscillator::{PhaseOrigin, increment_sd, wiener_phase};
use com_core::rng::Rng;
use com_core::signal::Signal;
use com_core::{Result, ensure, ensure_shape, same_len};
use com_dsp::filter::{LowPass, fir_filter, low_pass_fir};
use com_math::{BOLTZMANN, C64, ELEMENTARY_CHARGE, ZERO_CELSIUS, cis, db2lin, dbm2w};
use std::f64::consts::PI;

/// A continuous-wave laser.
#[derive(Clone, Debug, PartialEq)]
pub struct Laser {
    /// Output power (dBm).
    pub power_dbm: f64,
    /// Linewidth (Hz): the variance of the phase increments is `2πΔν/f_s`.
    pub linewidth_hz: f64,
    /// Relative intensity noise: variance of the relative power fluctuation per sample
    /// (`P (1 + r[n])`, `r ~ N(0, rin_var)`, clipped at -1).
    pub rin_var: f64,
    /// Frequency offset from the simulation centre frequency (Hz).
    pub freq_shift_hz: f64,
}

impl Default for Laser {
    /// 10 dBm, 100 kHz linewidth, no intensity noise, no frequency offset.
    fn default() -> Self {
        Laser { power_dbm: 10.0, linewidth_hz: 100e3, rin_var: 0.0, freq_shift_hz: 0.0 }
    }
}

impl Laser {
    /// The unit-amplitude carrier of the laser, `n` samples at the sampling rate `fs`:
    /// the Wiener phase of its linewidth ([`wiener_phase`] with the increments of
    /// [`increment_sd`], $\phi_{0} = 0$) and its frequency offset,
    /// $e^{j(\phi_{k} + 2\pi \Delta f k / f_{s})}$. A zero linewidth draws nothing from
    /// `rng`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a negative linewidth or a sampling rate that is not
    /// positive.
    ///
    /// # Examples
    /// ```
    /// use com_hardware::optical::Laser;
    /// use com_core::rng::Rng;
    /// let laser = Laser { linewidth_hz: 0.0, freq_shift_hz: 1e9, ..Laser::default() };
    /// let c = laser.carrier(8, 8e9, &mut Rng::new(1))?;
    /// assert!((c[2] - com_math::cis(std::f64::consts::FRAC_PI_2)).norm() < 1e-15);   // a quarter turn
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn carrier(&self, n: usize, fs: f64, rng: &mut Rng) -> Result<Vec<C64>> {
        let sd = increment_sd(self.linewidth_hz, fs)?;
        let phi = if sd > 0.0 { wiener_phase(sd, n, PhaseOrigin::Zero, rng) } else { vec![0.0; n] };
        Ok(phi.iter().enumerate().map(|(k, p)| cis(p + 2.0 * PI * self.freq_shift_hz * k as f64 / fs)).collect())
    }

    /// `n` samples of the field at the sampling rate `fs`: the [`Laser::carrier`] times
    /// $\sqrt{P (1 + r_{k})}$, the relative intensity noise $r_{k}$ drawn after the
    /// phase.
    ///
    /// # Errors
    /// As [`Laser::carrier`]; [`com_core::Error::Config`] for a negative or non-finite
    /// intensity-noise variance.
    pub fn field(&self, n: usize, fs: f64, rng: &mut Rng) -> Result<Vec<C64>> {
        ensure(self.rin_var >= 0.0 && self.rin_var.is_finite(), || {
            format!("the intensity-noise variance must be non-negative, got {}", self.rin_var)
        })?;
        let p = dbm2w(self.power_dbm);
        let carrier = self.carrier(n, fs, rng)?;
        Ok(carrier
            .into_iter()
            .map(|c| {
                let r = if self.rin_var > 0.0 { (1.0 + self.rin_var.sqrt() * rng.gaussian()).max(0.0) } else { 1.0 };
                c * (p * r).sqrt()
            })
            .collect())
    }
}

fn check_drive(e: &[C64], u: &[f64]) -> Result<()> {
    same_len(e.len(), u.len(), "optical field and driving signal")
}

/// `Ok` for a half-wave voltage that is positive and finite.
fn check_vpi(vpi: f64) -> Result<()> {
    ensure(vpi > 0.0 && vpi.is_finite(), || format!("the half-wave voltage must be positive, got {vpi} V"))
}

/// `Ok` for an extinction ratio that is non-negative (infinite for an ideal modulator).
fn check_er(er_db: f64) -> Result<()> {
    ensure(er_db >= 0.0, || format!("the extinction ratio must be non-negative, got {er_db} dB"))
}

/// Phase modulator `E e^{jπu/V_π}`.
///
/// # Errors
/// [`com_core::Error::Shape`] if the field and the drive have different lengths;
/// [`com_core::Error::Config`] for a half-wave voltage that is not positive.
pub fn phase_modulator(e: &[C64], u: &[f64], vpi: f64) -> Result<Vec<C64>> {
    check_drive(e, u)?;
    check_vpi(vpi)?;
    Ok(e.iter().zip(u).map(|(&x, &v)| x * cis(PI * v / vpi)).collect())
}

/// A push-pull Mach–Zehnder modulator.
#[derive(Clone, Debug, PartialEq)]
pub struct Mzm {
    /// Half-wave voltage (V).
    pub vpi: f64,
    /// Bias voltage (V): `-V_π` is the null of the field response (carrier
    /// suppression, the bias of coherent modulation), `-V_π/2` the quadrature point.
    pub vb: f64,
    /// Extinction ratio (dB); `f64::INFINITY` for an ideal modulator.
    pub er_db: f64,
}

impl Default for Mzm {
    /// `V_π = 2 V`, `V_b = -1 V`, 60 dB extinction ratio (OptiCommPy defaults).
    fn default() -> Self {
        Mzm { vpi: 2.0, vb: -1.0, er_db: 60.0 }
    }
}

impl Mzm {
    /// Validate the parameters.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a half-wave voltage that is not positive and finite,
    /// a bias that is not finite, or a negative (or NaN) extinction ratio.
    pub fn check(&self) -> Result<()> {
        check_vpi(self.vpi)?;
        ensure(self.vb.is_finite(), || format!("the bias must be finite, got {} V", self.vb))?;
        check_er(self.er_db)
    }

    /// The arm coefficients `(c_I, c_Q)` of the extinction ratio.
    pub fn coefficients(&self) -> (f64, f64) {
        if self.er_db.is_infinite() {
            return (1.0, 0.0);
        }
        let er = db2lin(self.er_db);
        let g = 2.0 * er.sqrt() / (er + 1.0);
        (((1.0 + g).sqrt() + (1.0 - g).sqrt()) / 2.0, ((1.0 + g).sqrt() - (1.0 - g).sqrt()) / 2.0)
    }

    /// Modulate the field `e` with the driving voltage `u`.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if the field and the drive have different lengths; the
    /// errors of [`Mzm::check`].
    pub fn modulate(&self, e: &[C64], u: &[f64]) -> Result<Vec<C64>> {
        check_drive(e, u)?;
        self.check()?;
        let (ci, cq) = self.coefficients();
        Ok(e.iter()
            .zip(u)
            .map(|(&x, &v)| {
                let th = (v + self.vb) / (2.0 * self.vpi) * PI;
                x * C64::new(ci * th.cos(), cq * th.sin())
            })
            .collect())
    }
}

/// An IQ modulator: two MZMs and a phase shift between them.
#[derive(Clone, Debug, PartialEq)]
pub struct Iqm {
    /// Half-wave voltage of the three modulators (V).
    pub vpi: f64,
    /// Bias of the I modulator (V).
    pub vb_i: f64,
    /// Bias of the Q modulator (V).
    pub vb_q: f64,
    /// Bias of the phase modulator between the branches (V).
    pub vphi: f64,
    /// Extinction ratio of the I modulator (dB).
    pub er_i_db: f64,
    /// Extinction ratio of the Q modulator (dB).
    pub er_q_db: f64,
}

impl Default for Iqm {
    /// OptiCommPy defaults: `V_π = 2`, biases at the null (`-2 V`), `V_φ = 1 V` (π/2),
    /// 60 dB extinction ratios.
    fn default() -> Self {
        Iqm { vpi: 2.0, vb_i: -2.0, vb_q: -2.0, vphi: 1.0, er_i_db: 60.0, er_q_db: 60.0 }
    }
}

impl Iqm {
    /// Modulate `e` with the complex driving signal `u` (`Re u` on I, `Im u` on Q).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if the field and the drive have different lengths;
    /// [`com_core::Error::Config`] for the parameters [`Mzm::check`] refuses (and a
    /// phase bias that is not finite).
    pub fn modulate(&self, e: &[C64], u: &[C64]) -> Result<Vec<C64>> {
        ensure(self.vphi.is_finite(), || format!("the phase bias must be finite, got {} V", self.vphi))?;
        same_len(e.len(), u.len(), "optical field and driving signal")?;
        let half: Vec<C64> = e.iter().map(|v| v * std::f64::consts::FRAC_1_SQRT_2).collect();
        let ui: Vec<f64> = u.iter().map(|v| v.re).collect();
        let uq: Vec<f64> = u.iter().map(|v| v.im).collect();
        let i = Mzm { vpi: self.vpi, vb: self.vb_i, er_db: self.er_i_db }.modulate(&half, &ui)?;
        let q = Mzm { vpi: self.vpi, vb: self.vb_q, er_db: self.er_q_db }.modulate(&half, &uq)?;
        let rot = cis(PI * self.vphi / self.vpi);
        Ok(i.iter().zip(&q).map(|(a, b)| a + b * rot).collect())
    }
}

/// A PIN photodiode with shot and thermal noise and a finite bandwidth.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools, reason = "the switches of OptiCommPy's photodiode parameters")]
pub struct Photodiode {
    /// Responsivity (A/W).
    pub responsivity: f64,
    /// Temperature (K).
    pub temperature_k: f64,
    /// Dark current (A).
    pub dark_current_a: f64,
    /// Load resistance (Ω).
    pub load_ohm: f64,
    /// Electrical bandwidth (Hz).
    pub bandwidth_hz: f64,
    /// Number of taps of the low-pass filter (odd).
    pub n_taps: usize,
    /// Shape of the low-pass filter.
    pub filter: LowPass,
    /// Add shot noise.
    pub shot_noise: bool,
    /// Add thermal noise.
    pub thermal_noise: bool,
    /// Apply the bandwidth limitation.
    pub bandwidth_limitation: bool,
    /// Ideal detection `R|E|²` (no noise, no filter): overrides the three flags above.
    pub ideal: bool,
}

impl Default for Photodiode {
    /// OptiCommPy defaults: 1 A/W, 25 °C (298.15 K), 5 nA, 50 Ω, 30 GHz, 255-tap rectangular filter.
    fn default() -> Self {
        Photodiode {
            responsivity: 1.0,
            temperature_k: 25.0 + ZERO_CELSIUS,
            dark_current_a: 5e-9,
            load_ohm: 50.0,
            bandwidth_hz: 30e9,
            n_taps: 255,
            filter: LowPass::Rect,
            shot_noise: true,
            thermal_noise: true,
            bandwidth_limitation: true,
            ideal: false,
        }
    }
}

impl Photodiode {
    /// An ideal photodiode of unit responsivity.
    pub fn ideal() -> Photodiode {
        Photodiode { ideal: true, ..Photodiode::default() }
    }

    /// Photocurrent of a field (one polarisation) sampled at `fs`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the detector is not ideal and `fs < 2B`.
    pub fn detect(&self, e: &[C64], fs: f64, rng: &mut Rng) -> Result<Vec<f64>> {
        self.detect_power(&e.iter().map(C64::norm_sqr).collect::<Vec<_>>(), fs, rng)
    }

    /// Photocurrent of the field `x` summed over its streams (the polarisations of a
    /// [`com_core::signal::DualPol`], `to_signal`), at the sampling rate of the signal:
    /// [`Photodiode::detect_power`] of $\sum_{s} |x_{s}\[k\]|^{2}$.
    ///
    /// # Errors
    /// As [`Photodiode::detect_power`].
    ///
    /// # Examples
    /// ```
    /// use com_hardware::optical::Photodiode;
    /// use com_core::prelude::*;
    /// let x = Signal::from_streams(&[vec![C64::new(0.3, 0.4); 4], vec![C64::new(0.0, 0.5); 4]], Hertz(64e9))?;
    /// let i = Photodiode::ideal().detect_signal(&x, &mut Rng::new(1))?;
    /// assert!((i[0] - 0.5).abs() < 1e-15);                       // |x|² + |y|²
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn detect_signal(&self, x: &Signal, rng: &mut Rng) -> Result<Vec<f64>> {
        let p: Vec<f64> =
            (0..x.n_samples()).map(|k| (0..x.n_streams()).map(|s| x.stream(s)[k].norm_sqr()).sum()).collect();
        self.detect_power(&p, x.sample_rate().0, rng)
    }

    /// Photocurrent of an optical power (W per sample), e.g. summed over two polarisations.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a non-positive responsivity or a sampling rate below twice the bandwidth.
    pub fn detect_power(&self, p: &[f64], fs: f64, rng: &mut Rng) -> Result<Vec<f64>> {
        ensure(self.responsivity > 0.0, || "the responsivity must be positive".into())?;
        let mut i: Vec<f64> = p.iter().map(|v| self.responsivity * v).collect();
        if self.ideal {
            return Ok(i);
        }
        let b = self.bandwidth_hz;
        ensure(fs >= 2.0 * b, || format!("the sampling rate {fs} Hz is below twice the bandwidth {b} Hz"))?;
        if self.shot_noise {
            for v in &mut i {
                let s2 = shot_noise_variance(*v + self.dark_current_a, b);
                *v += (fs * s2 / (2.0 * b)).max(0.0).sqrt() * rng.gaussian();
            }
        }
        if self.thermal_noise {
            let s2 = thermal_noise_variance(self.temperature_k, self.load_ohm, b);
            let s = (fs * s2 / (2.0 * b)).sqrt();
            for v in &mut i {
                *v += s * rng.gaussian();
            }
        }
        if self.bandwidth_limitation {
            let n = self.n_taps | 1;
            let h = low_pass_fir(b, fs, n, self.filter);
            let c: Vec<C64> = i.iter().map(|&v| C64::new(v, 0.0)).collect();
            i = fir_filter(&h, &c).into_iter().map(|v| v.re).collect();
        }
        Ok(i)
    }
}

/// A p-i-n photodiode followed by a FET transimpedance amplifier, the receiver of
/// indoor optical wireless links: shot noise of the received and ambient light, and
/// thermal noise of the feedback resistor and of the FET channel (Komine & Nakagawa
/// 2004, eqs. (18), (20)–(21), after Kahn & Barry 1997):
///
/// $$
/// \sigma_{\mathrm{shot}}^{2} = 2q\gamma (P_{s} + P_{\mathrm{ISI}}) B + 2 q I_{bg} I_{2} B,
/// \qquad
/// \sigma_{\mathrm{th}}^{2} = \frac{8\pi k T}{G} \eta A I_{2} B^{2} +
///   \frac{16 \pi^{2} k T \Gamma}{g_{m}} \eta^{2} A^{2} I_{3} B^{3},
/// $$
///
/// and the SNR of on–off keying $\gamma^{2} P_{s}^{2} / (\sigma_{\mathrm{shot}}^{2} +
/// \sigma_{\mathrm{th}}^{2} + \gamma^{2} P_{\mathrm{ISI}}^{2})$ (their eqs. (16)–(18)), with
/// the powers $P_{s}$ and $P_{\mathrm{ISI}}$ inside the bit and spilled out of it
/// (`com_channel::owc::indoor::ImpulseResponse::bit_split`).
///
/// # Examples
/// ```
/// use com_hardware::optical::PinFetReceiver;
/// let rx = PinFetReceiver::komine2004();
/// // 1 mW on 1 cm² at 100 Mb/s: the ambient light dominates the noise
/// let (a, b) = (1e-4, 100e6);
/// let ambient = 2.0 * com_math::ELEMENTARY_CHARGE * rx.background_current_a * rx.i2 * b;
/// assert!(ambient > 0.5 * (rx.shot_variance(1e-3, 0.0, b) + rx.thermal_variance(a, b)));
/// // doubling the power gains a little less than 6 dB: the shot noise of the signal grows
/// let gain = rx.ook_snr_db(2e-3, 0.0, a, b) - rx.ook_snr_db(1e-3, 0.0, a, b);
/// assert!(gain > 5.0 && gain < 6.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinFetReceiver {
    /// Responsivity $\gamma$ (A/W).
    pub responsivity: f64,
    /// Background current $I_{bg}$ of the ambient light (A).
    pub background_current_a: f64,
    /// Open-loop voltage gain $G$.
    pub open_loop_gain: f64,
    /// Fixed capacitance of the photodiode per unit area $\eta$ (F/m²).
    pub capacitance_f_per_m2: f64,
    /// FET channel noise factor $\Gamma$.
    pub channel_noise_factor: f64,
    /// FET transconductance $g_{m}$ (S).
    pub transconductance_s: f64,
    /// Absolute temperature $T$ (K).
    pub temperature_k: f64,
    /// Noise bandwidth factor $I_{2}$.
    pub i2: f64,
    /// Noise bandwidth factor $I_{3}$.
    pub i3: f64,
}

impl PinFetReceiver {
    /// The receiver of Komine & Nakagawa (2004), Tables II–III: 0.53 A/W, 5100 µA of
    /// background current, $G = 10$, 112 pF/cm², $\Gamma = 1.5$, 30 mS, 298 K,
    /// $I_{2} = 0.562$, $I_{3} = 0.0868$ (their text quotes 0.54 A/W and 295 K).
    pub fn komine2004() -> PinFetReceiver {
        PinFetReceiver {
            responsivity: 0.53,
            background_current_a: 5100e-6,
            open_loop_gain: 10.0,
            capacitance_f_per_m2: 112e-12 / 1e-4,
            channel_noise_factor: 1.5,
            transconductance_s: 30e-3,
            temperature_k: 298.0,
            i2: 0.562,
            i3: 0.0868,
        }
    }

    /// The variance of the shot noise (A²) for the received powers `signal_w` and
    /// `isi_w` and the noise bandwidth `bandwidth_hz`.
    pub fn shot_variance(&self, signal_w: f64, isi_w: f64, bandwidth_hz: f64) -> f64 {
        shot_noise_variance(self.responsivity * (signal_w + isi_w), bandwidth_hz)
            + shot_noise_variance(self.background_current_a * self.i2, bandwidth_hz)
    }

    /// The variance of the thermal noise (A²) of a photodiode of area `area_m2`.
    pub fn thermal_variance(&self, area_m2: f64, bandwidth_hz: f64) -> f64 {
        let (k_t, b) = (BOLTZMANN * self.temperature_k, bandwidth_hz);
        let c = self.capacitance_f_per_m2 * area_m2;
        8.0 * PI * k_t / self.open_loop_gain * c * self.i2 * b * b
            + 16.0 * PI * PI * k_t * self.channel_noise_factor / self.transconductance_s * c * c * self.i3 * b * b * b
    }

    /// The SNR (dB) of on–off keying for the received powers `signal_w` and `isi_w`.
    pub fn ook_snr_db(&self, signal_w: f64, isi_w: f64, area_m2: f64, bandwidth_hz: f64) -> f64 {
        let s = (self.responsivity * signal_w).powi(2);
        let n = self.shot_variance(signal_w, isi_w, bandwidth_hz)
            + self.thermal_variance(area_m2, bandwidth_hz)
            + (self.responsivity * isi_w).powi(2);
        com_math::lin2db(s / n)
    }
}

/// The variance $2 q I B$ (A²) of the shot noise of the mean current `current_a` $I$ in
/// the noise bandwidth `bandwidth_hz` $B$, the shot noise of [`Photodiode`] and of
/// [`PinFetReceiver`].
///
/// # Examples
/// ```
/// use com_hardware::optical::shot_noise_variance;
/// let v = shot_noise_variance(1e-3, 1e9);                    // 1 mA over 1 GHz
/// assert!((v - 2.0 * com_math::ELEMENTARY_CHARGE * 1e-3 * 1e9).abs() < 1e-30);
/// ```
pub fn shot_noise_variance(current_a: f64, bandwidth_hz: f64) -> f64 {
    2.0 * ELEMENTARY_CHARGE * current_a * bandwidth_hz
}

/// The variance $4 k T B / R$ (A²) of the thermal (Johnson–Nyquist) noise current of a
/// resistance `resistance_ohm` $R$ at the absolute temperature `temperature_k` $T$ in the
/// noise bandwidth `bandwidth_hz` $B$, the thermal noise of the load of [`Photodiode`].
///
/// # Examples
/// ```
/// use com_hardware::optical::thermal_noise_variance;
/// // 50 Ω at 25 °C over 30 GHz: about 1e-11 A²
/// let v = thermal_noise_variance(25.0 + com_math::ZERO_CELSIUS, 50.0, 30e9);
/// assert!((v / 9.88e-12 - 1.0).abs() < 1e-3);
/// ```
pub fn thermal_noise_variance(temperature_k: f64, resistance_ohm: f64, bandwidth_hz: f64) -> f64 {
    4.0 * BOLTZMANN * temperature_k * bandwidth_hz / resistance_ohm
}

/// Balanced detection `i(E₁) - i(E₂)`.
///
/// # Errors
/// [`com_core::Error::Shape`] for inputs of different lengths; the errors of [`Photodiode::detect`].
pub fn balanced_photodiode(e1: &[C64], e2: &[C64], pd: &Photodiode, fs: f64, rng: &mut Rng) -> Result<Vec<f64>> {
    same_len(e1.len(), e2.len(), "balanced detector inputs")?;
    let a = pd.detect(e1, fs, rng)?;
    let b = pd.detect(e2, fs, rng)?;
    Ok(a.iter().zip(&b).map(|(x, y)| x - y).collect())
}

/// The four outputs of the 90° optical hybrid for the signal `es` and the local
/// oscillator `elo` (OptiCommPy `opticalHybrid2x4`):
///
/// `E₀ = (E_s - E_LO)/2`, `E₁ = j(E_s + E_LO)/2`, `E₂ = (jE_s - E_LO)/2`, `E₃ = (-E_s + jE_LO)/2`.
///
/// # Errors
/// [`com_core::Error::Shape`] if the signal and the local oscillator have different lengths.
pub fn hybrid_2x4(es: &[C64], elo: &[C64]) -> Result<[Vec<C64>; 4]> {
    same_len(es.len(), elo.len(), "signal and local oscillator")?;
    let j = C64::new(0.0, 1.0);
    let h = 0.5;
    Ok([
        es.iter().zip(elo).map(|(s, l)| (s - l) * h).collect(),
        es.iter().zip(elo).map(|(s, l)| j * (s + l) * h).collect(),
        es.iter().zip(elo).map(|(s, l)| (j * s - l) * h).collect(),
        es.iter().zip(elo).map(|(s, l)| (-s + j * l) * h).collect(),
    ])
}

/// Single-polarisation coherent receiver: 90° hybrid and two balanced pairs,
/// `s = i(E₁) - i(E₀) + j (i(E₂) - i(E₃))` (OptiCommPy `coherentReceiver` without
/// front-end impairments). With ideal photodiodes `s = R E_s E_LO*`.
///
/// # Errors
/// As [`hybrid_2x4`] and [`balanced_photodiode`].
pub fn coherent_receiver(es: &[C64], elo: &[C64], pd: &Photodiode, fs: f64, rng: &mut Rng) -> Result<Vec<C64>> {
    let [e0, e1, e2, e3] = hybrid_2x4(es, elo)?;
    let si = balanced_photodiode(&e1, &e0, pd, fs, rng)?;
    let sq = balanced_photodiode(&e2, &e3, pd, fs, rng)?;
    Ok(si.iter().zip(&sq).map(|(&a, &b)| C64::new(a, b)).collect())
}

/// [`coherent_receiver`] on every stream of the field `es` (the polarisations of a
/// [`com_core::signal::DualPol`]), at its sampling rate, with the local oscillator `elo`:
/// one stream shared by all, or one per stream.
///
/// # Errors
/// [`com_core::Error::Shape`] for a local oscillator of another length or number of
/// streams; [`com_core::Error::Config`] for another sampling rate; the errors of
/// [`coherent_receiver`].
///
/// # Examples
/// ```
/// use com_hardware::optical::{coherent_receiver_signal, Photodiode};
/// use com_core::prelude::*;
/// let es = Signal::from_streams(&[vec![C64::new(0.3, -0.2); 4], vec![C64::new(0.1, 0.0); 4]], Hertz(64e9))?;
/// let lo = Signal::from_stream(vec![C64::new(1.0, 0.0); 4], Hertz(64e9))?;
/// let s = coherent_receiver_signal(&es, &lo, &Photodiode::ideal(), &mut Rng::new(0))?;
/// assert!((s.stream(1)[0] - C64::new(0.1, 0.0)).norm() < 1e-12);   // R E_s E_LO*
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn coherent_receiver_signal(es: &Signal, elo: &Signal, pd: &Photodiode, rng: &mut Rng) -> Result<Signal> {
    ensure(es.sample_rate() == elo.sample_rate(), || {
        format!("a local oscillator at {} Hz for a field at {} Hz", elo.sample_rate().0, es.sample_rate().0)
    })?;
    ensure_shape(elo.n_streams() == 1 || elo.n_streams() == es.n_streams(), || {
        format!("{} local oscillators for {} streams: one, or one per stream", elo.n_streams(), es.n_streams())
    })?;
    let mut out = Signal::zeros(0, 0, es.sample_rate());
    es.try_map_streams(&mut out, |s, x| {
        let lo = elo.stream(if elo.n_streams() == 1 { 0 } else { s });
        coherent_receiver(x, lo, pd, es.sample_rate().0, rng)
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::units::Hertz;

    #[test]
    fn invalid_modulators_are_refused_instead_of_giving_nan() {
        let e = [C64::new(1.0, 0.0); 3];
        assert!(phase_modulator(&e, &[0.0; 3], 0.0).is_err());
        for m in [
            Mzm { vpi: 0.0, ..Mzm::default() },
            Mzm { vb: f64::NAN, ..Mzm::default() },
            Mzm { er_db: -3.0, ..Mzm::default() },
        ] {
            assert!(m.modulate(&e, &[0.0; 3]).is_err(), "{m:?}");
        }
        assert!(Mzm { er_db: f64::INFINITY, ..Mzm::default() }.check().is_ok());
        assert!(Iqm { vpi: -2.0, ..Iqm::default() }.modulate(&e, &e).is_err());
        assert!(Iqm { vphi: f64::INFINITY, ..Iqm::default() }.modulate(&e, &e).is_err());
    }

    #[test]
    fn the_receivers_on_a_signal_are_those_of_the_slices() {
        let mut r = Rng::new(4);
        let pols: Vec<Vec<C64>> = (0..2).map(|_| (0..256).map(|_| r.complex_gaussian() * 1e-2).collect()).collect();
        let fs = 128e9;
        let x = Signal::from_streams(&pols, Hertz(fs)).unwrap();
        let pd = Photodiode::default();
        let power: Vec<f64> = (0..256).map(|k| pols[0][k].norm_sqr() + pols[1][k].norm_sqr()).collect();
        assert_eq!(
            pd.detect_signal(&x, &mut Rng::new(1)).unwrap(),
            pd.detect_power(&power, fs, &mut Rng::new(1)).unwrap()
        );
        let lo = Signal::from_stream(vec![C64::new(0.1, 0.0); 256], Hertz(fs)).unwrap();
        let (mut a, mut b) = (Rng::new(2), Rng::new(2));
        let s = coherent_receiver_signal(&x, &lo, &pd, &mut a).unwrap();
        for (k, p) in pols.iter().enumerate() {
            assert_eq!(s.stream(k), &coherent_receiver(p, lo.stream(0), &pd, fs, &mut b).unwrap()[..]);
        }
        let slow = Signal::from_stream(vec![C64::new(0.1, 0.0); 256], Hertz(fs / 2.0)).unwrap();
        assert!(coherent_receiver_signal(&x, &slow, &pd, &mut a).is_err());
    }
}

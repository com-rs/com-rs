//! Optical fibre: link parameters, the linear channel and the split-step Fourier method
//! (OptiCommPy `optic.models.channels`).
//!
//! # Parameters
//!
//! [`FiberParams`] uses the engineering units of OptiCommPy: attenuation `α` in dB/km,
//! dispersion parameter `D` in ps/(nm·km), nonlinear coefficient `γ` in 1/(W·km),
//! lengths in km, carrier frequency in Hz, amplifier noise figure in dB. Its methods
//! give the quantities of the propagation equation:
//!
//! - `α [1/km] = α_dB / (10 log10 e)` (power attenuation in neper/km);
//! - `β₂ [s²/km] = -D λ² / (2π c)`, with `λ = c/f_c`: since 1 ps/(nm·km) is
//!   numerically 1 s/km², `β₂ = -D λ_km² / (2π c_km)` in s²/km with `λ` in km and `c` in
//!   km/s;
//! - the span gain `G = e^{α L}` that restores the power after each span;
//! - the effective length `L_eff = (1 - e^{-α L})/α`;
//! - [`FiberParams::medium`], the coefficients of [`com_math::nlse`].
//!
//! # Propagation
//!
//! - [`linear_fiber_channel`]: loss and chromatic dispersion of a fibre length, exact in
//!   the frequency domain on the block (OptiCommPy `linearFiberChannel`):
//!   `A(L, ω) = A(0, ω) exp(-αL/2 + j β₂ ω² L/2)`;
//! - [`Ssfm`]: the symmetric split-step Fourier method of OptiCommPy `ssfm` for one
//!   polarisation (`n = ⌈L/h⌉` equal steps of `L/n` per span, `h` itself when it
//!   divides the span; per step: half linear step, nonlinear phase `γ|A|² h`, half
//!   linear step; after each span an EDFA, an ideal gain or nothing),
//!   a thin wrapper that supplies the parameters of the link to the solver
//!   [`com_math::nlse::SplitStep`]; [`Ssfm::propagate_f32`] runs it in single
//!   precision. [`Ssfm::backpropagate`] is a shortcut to the digital back-propagation
//!   of `com_dsp::compensation::dbp` with the parameters of the link.
//!
//! # Model
//!
//! $$
//! \frac{\partial A}{\partial z} = -\frac{\alpha}{2} A
//!   - j \frac{\beta_{2}}{2} \frac{\partial^{2} A}{\partial t^{2}}
//!   + j \gamma |A|^{2} A
//!     $$
//!
//! # Examples
//! ```
//! use com_channel::optical::fiber::{linear_fiber_channel, Amplification, FiberParams, Ssfm};
//! use com_math::C64;
//! let f = FiberParams { span_km: 100.0, n_spans: 10, ..FiberParams::default() };
//! assert_eq!(f.total_km(), 1000.0);
//! assert!((f.span_gain_db() - 20.0).abs() < 1e-9);       // 0.2 dB/km × 100 km
//! assert!((f.effective_length_km() - 21.5).abs() < 0.1);
//! let fiber = FiberParams { alpha_db_km: 0.0, gamma_w_km: 0.0, ..FiberParams::default() };
//! let x: Vec<C64> = (0..64).map(|n| C64::new((n % 3) as f64, (n % 2) as f64)).collect();
//! // without loss and nonlinearity the split step is the linear channel
//! let a = linear_fiber_channel(&x, &fiber, 80.0, 64e9);
//! let mut b = x.clone();
//! Ssfm { fiber: fiber.clone(), step_km: 10.0, amplification: Amplification::None, raman: None }
//!     .propagate(&mut b, 64e9, None)
//!     .unwrap();
//! assert!(a.iter().zip(&b).all(|(u, v)| (u - v).norm() < 1e-9));
//! ```
//!
//! # References
//!
//! 1. G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013.

use super::edfa::Edfa;
use super::raman::{FieldSpan, Raman};
use com_core::ensure;
use com_core::prelude::*;
use com_dsp::compensation::dbp::Dbp;
use com_math::fft;
use com_math::nlse::{Medium, SplitStep, SplitStepF32, Steps};
use com_math::{SPEED_OF_LIGHT, db2lin, db2np, effective_length};
use num_complex::Complex32 as C32;

/// A fibre link of identical spans, each followed by an amplifier.
#[derive(Clone, Debug, PartialEq)]
pub struct FiberParams {
    /// Attenuation (dB/km).
    pub alpha_db_km: f64,
    /// Dispersion parameter `D` (ps/(nm·km)); positive for standard fibre at 1550 nm.
    pub d_ps_nm_km: f64,
    /// Nonlinear coefficient `γ` (1/(W·km)).
    pub gamma_w_km: f64,
    /// Length of one span (km).
    pub span_km: f64,
    /// Number of spans.
    pub n_spans: usize,
    /// Carrier frequency (Hz).
    pub fc_hz: f64,
    /// Noise figure of the amplifiers (dB).
    pub nf_db: f64,
}

impl Default for FiberParams {
    /// Standard single-mode fibre: 0.2 dB/km, 17 ps/(nm·km), 1.3 /(W·km), one span of
    /// 80 km, 193.1 THz, amplifiers of noise figure 4.5 dB.
    fn default() -> Self {
        FiberParams {
            alpha_db_km: 0.2,
            d_ps_nm_km: 17.0,
            gamma_w_km: 1.3,
            span_km: 80.0,
            n_spans: 1,
            fc_hz: 193.1e12,
            nf_db: 4.5,
        }
    }
}

impl FiberParams {
    /// The coefficients of the propagation equation (`α` in Np/km, `β₂` in s²/km, `γ`).
    pub fn medium(&self) -> Medium {
        Medium { alpha_np_km: self.alpha_np_km(), beta2_s2_km: self.beta2_s2_km(), gamma_w_km: self.gamma_w_km }
    }

    /// Total length (km).
    pub fn total_km(&self) -> f64 {
        self.span_km * self.n_spans as f64
    }

    /// Power attenuation `α` (1/km).
    pub fn alpha_np_km(&self) -> f64 {
        db2np(self.alpha_db_km)
    }

    /// Wavelength (m).
    pub fn wavelength_m(&self) -> f64 {
        SPEED_OF_LIGHT / self.fc_hz
    }

    /// Group-velocity dispersion `β₂` (s²/km).
    pub fn beta2_s2_km(&self) -> f64 {
        let c_km = SPEED_OF_LIGHT / 1e3;
        let lambda_km = c_km / self.fc_hz;
        -self.d_ps_nm_km * lambda_km * lambda_km / (2.0 * std::f64::consts::PI * c_km)
    }

    /// Group-velocity dispersion `β₂` (ps²/km).
    pub fn beta2_ps2_km(&self) -> f64 {
        self.beta2_s2_km() * 1e24
    }

    /// Linear gain `e^{αL}` that compensates the loss of one span.
    pub fn span_gain(&self) -> f64 {
        (self.alpha_np_km() * self.span_km).exp()
    }

    /// Span gain in dB, `α_dB L`.
    pub fn span_gain_db(&self) -> f64 {
        self.alpha_db_km * self.span_km
    }

    /// The EDFA after every span: gain `α_dB L`, the noise figure and carrier of the link.
    pub fn span_edfa(&self) -> Edfa {
        Edfa { gain_db: self.span_gain_db(), nf_db: self.nf_db, fc_hz: self.fc_hz }
    }

    /// Effective length of one span `(1 - e^{-αL})/α` (km).
    pub fn effective_length_km(&self) -> f64 {
        effective_length(self.alpha_np_km(), self.span_km)
    }

    /// Accumulated dispersion of the link `D L` (ps/nm).
    pub fn accumulated_dispersion_ps_nm(&self) -> f64 {
        self.d_ps_nm_km * self.total_km()
    }

    /// Number of samples spanned by the dispersion of the link at the sampling rate `fs`
    /// for a signal of bandwidth `bandwidth_hz`: `2π |β₂| L B fs`.
    pub fn dispersion_memory_samples(&self, bandwidth_hz: f64, fs: f64) -> f64 {
        2.0 * std::f64::consts::PI * self.beta2_s2_km().abs() * self.total_km() * bandwidth_hz * fs
    }
}

/// Loss and dispersion of `length_km` of fibre on a block sampled at `fs`
/// (OptiCommPy `linearFiberChannel`).
pub fn linear_fiber_channel(x: &[C64], fiber: &FiberParams, length_km: f64, fs: f64) -> Vec<C64> {
    let w = fft::angular_frequencies(x.len(), fs);
    fft::filter(x, fiber.medium().linear_response(&w, length_km))
}

/// The amplification after every span of a split-step simulation.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Amplification {
    /// An EDFA of gain `α L` and the noise figure of the fibre parameters (adds ASE when
    /// a generator is given).
    Edfa,
    /// A noiseless gain `e^{αL/2}` on the field.
    Ideal,
    /// No amplification.
    None,
}

impl std::str::FromStr for Amplification {
    type Err = Error;

    /// `edfa`, `ideal` or `none` (case-insensitive).
    fn from_str(s: &str) -> Result<Amplification> {
        match s.to_ascii_lowercase().as_str() {
            "edfa" => Ok(Amplification::Edfa),
            "ideal" => Ok(Amplification::Ideal),
            "none" => Ok(Amplification::None),
            other => Err(Error::Config(format!("amplification must be edfa, ideal or none, got {other:?}"))),
        }
    }
}

impl Amplification {
    /// The field gain after a span: the EDFA restores the span `α L` of `fiber`, the
    /// ideal gain restores the simulated `length_km`, and `None` gives 1.
    pub fn field_gain(&self, fiber: &FiberParams, length_km: f64) -> f64 {
        match self {
            Amplification::Edfa => db2lin(fiber.span_gain_db()).sqrt(),
            Amplification::Ideal => (fiber.alpha_np_km() / 2.0 * length_km).exp(),
            Amplification::None => 1.0,
        }
    }

    /// Amplify the field `e` (√W, sampled at `fs`) after a span: the EDFA of `fiber`
    /// draws its ASE from `rng` (noiseless without one), the ideal gain restores the
    /// simulated `length_km`.
    ///
    /// # Errors
    /// As [`Edfa::apply`].
    pub fn apply(
        &self,
        e: &mut [C64],
        fiber: &FiberParams,
        length_km: f64,
        fs: f64,
        rng: Option<&mut Rng>,
    ) -> Result<()> {
        match self {
            Amplification::Edfa => {
                let amp = fiber.span_edfa();
                match rng {
                    Some(r) => amp.apply(e, fs, r)?,
                    None => amp.apply_noiseless(e),
                }
            }
            Amplification::Ideal => {
                let s = self.field_gain(fiber, length_km);
                for v in e.iter_mut() {
                    *v *= s;
                }
            }
            Amplification::None => {}
        }
        Ok(())
    }
}

/// The loop over the spans of a link, shared by the propagators ([`Ssfm`],
/// `manakov::ManakovSsfm`, `pmd::ManakovPmd`): for each of the `fiber.n_spans` spans,
/// `span(k, fields, raman_span)` propagates the polarisations `fields` over span `k`,
/// then each polarisation, in order, is amplified with its noise drawn from `rng`:
///
/// - without Raman amplification, by [`Amplification::apply`] over `length_km` (the
///   simulated length of the span);
/// - with `raman`, by the restoring gain of the Raman profile of the span, which is
///   solved for the fields at its input over `fiber` and handed to `span` (its
///   distributed gain).
///
/// Returns the sum of the values of `span` (the steps taken).
///
/// # Errors
/// The errors of `span`, of the Raman solver and of the amplifier.
pub(crate) fn amplified_spans(
    fiber: &FiberParams,
    length_km: f64,
    amplification: &Amplification,
    raman: Option<&Raman>,
    fields: &mut [&mut Vec<C64>],
    fs: f64,
    mut rng: Option<&mut Rng>,
    mut span: impl FnMut(usize, &mut [&mut Vec<C64>], Option<&FieldSpan>) -> Result<usize>,
) -> Result<usize> {
    let mut total = 0;
    for k in 0..fiber.n_spans {
        if let Some(raman) = raman {
            let views: Vec<&[C64]> = fields.iter().map(|f| f.as_slice()).collect();
            let profile = raman.field_span(fiber, &views, fs)?;
            total += span(k, fields, Some(&profile))?;
            for f in fields.iter_mut() {
                profile.amplify(f, fs, amplification, fiber.nf_db, rng.as_deref_mut());
            }
        } else {
            total += span(k, fields, None)?;
            for f in fields.iter_mut() {
                amplification.apply(f, fiber, length_km, fs, rng.as_deref_mut())?;
            }
        }
    }
    Ok(total)
}

/// The arithmetic of the split steps of [`Ssfm::propagate_with`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Double precision ([`Ssfm::propagate`]).
    #[default]
    F64,
    /// Single precision for the steps, double for the amplifiers
    /// ([`Ssfm::propagate_f32`]).
    F32,
}

impl std::str::FromStr for Precision {
    type Err = Error;

    /// `f64` or `f32` (case-insensitive).
    fn from_str(s: &str) -> Result<Precision> {
        match s.to_ascii_lowercase().as_str() {
            "f64" => Ok(Precision::F64),
            "f32" => Ok(Precision::F32),
            _ => Err(Error::Config(format!("precision must be f64 or f32, got {s:?}"))),
        }
    }
}

/// The symmetric split-step Fourier method of OptiCommPy (`ssfm`), one polarisation.
#[derive(Clone, Debug, PartialEq)]
pub struct Ssfm {
    /// The link (spans, loss, dispersion, nonlinearity, noise figure).
    pub fiber: FiberParams,
    /// Step size `h` (km): the span of length $L$ is covered by $n = \lceil L / h \rceil$
    /// equal steps of $L / n \le h$ (the step `h` itself when it divides the span), so
    /// that the simulated length is the span the amplifier compensates.
    pub step_km: f64,
    /// Amplification after every span.
    pub amplification: Amplification,
    /// Distributed Raman amplification and ISRS in every span (see [`super::raman`]):
    /// the amplifier after the span then restores every frequency to its launch power.
    pub raman: Option<Raman>,
}

impl Ssfm {
    /// A simulation of `fiber` with steps of `step_km` and EDFAs.
    pub fn new(fiber: FiberParams, step_km: f64) -> Ssfm {
        Ssfm { fiber, step_km, amplification: Amplification::Edfa, raman: None }
    }

    /// The steps of a span: their number $n = \lceil L / h \rceil$ and their length
    /// $L / n$, which add up to the span $L$. The length is `h` itself when $n h = L$
    /// (within $10^{-9}$), $L / n$ otherwise; the back-propagation recognises these steps
    /// by their number ([`Ssfm::dbp`]).
    ///
    /// # Errors
    /// [`Error::Config`] for a non-positive or non-finite step, or a step longer than
    /// the span.
    fn steps(&self) -> Result<(usize, f64)> {
        let (h, span) = (self.step_km, self.fiber.span_km);
        ensure(h > 0.0 && h.is_finite(), || format!("step must be positive, got {h}"))?;
        let ratio = span / h;
        ensure(ratio.is_finite() && ratio >= 1.0 - 1e-9, || format!("the step {h} km is longer than the span"))?;
        let whole = ratio.round();
        let divides = (whole - ratio).abs() <= 1e-9 * ratio;
        let n = if divides { whole as usize } else { ratio.ceil() as usize };
        // n h = L within rounding: the step as given
        let step = if divides && (span / h).floor() as usize == n { h } else { span / n as f64 };
        Ok((n, step))
    }

    /// Propagate a field (√W) sampled at `fs` over the link, in place; the ASE of the
    /// EDFAs is drawn from `rng` (noiseless amplifiers without one).
    ///
    /// # Errors
    /// [`Error::Config`] for a non-positive step or a step longer than the span.
    pub fn propagate(&self, x: &mut Vec<C64>, fs: f64, rng: Option<&mut Rng>) -> Result<()> {
        let (n_steps, h) = self.steps()?;
        let simulated_km = n_steps as f64 * h;
        // with Raman amplification the profile of every span is solved for the slices of
        // the field at its input, over the simulated length, and gives the loss
        let (fiber, medium) = match self.raman {
            Some(_) => {
                let fiber = FiberParams { span_km: simulated_km, ..self.fiber.clone() };
                let medium = Medium { alpha_np_km: 0.0, ..fiber.medium() };
                (fiber, medium)
            }
            None => (self.fiber.clone(), self.fiber.medium()),
        };
        let steps = SplitStep::new(&medium, Steps::new(n_steps, h), x.len(), fs);
        amplified_spans(
            &fiber,
            simulated_km,
            &self.amplification,
            self.raman.as_ref(),
            &mut [x],
            fs,
            rng,
            |_, f, raman| {
                match raman {
                    Some(span) => steps.run_with_gain(f[0], &|z0, z1, out| span.field_gain(z0, z1, out)),
                    None => steps.run(f[0]),
                }
                Ok(0)
            },
        )?;
        Ok(())
    }

    /// [`Ssfm::propagate`] or [`Ssfm::propagate_f32`], as `precision` says.
    ///
    /// # Errors
    /// The errors of the method called.
    pub fn propagate_with(&self, x: &mut Vec<C64>, fs: f64, rng: Option<&mut Rng>, precision: Precision) -> Result<()> {
        match precision {
            Precision::F64 => self.propagate(x, fs, rng),
            Precision::F32 => self.propagate_f32(x, fs, rng),
        }
    }

    /// The digital back-propagation of this link with `steps_per_span` steps per span
    /// and the nonlinear scale `xi` (see `com_dsp::compensation::dbp`): the amplifier
    /// gain of [`Ssfm::amplification`] is removed before each span, and with the
    /// forward steps (`steps_per_span = ⌈L/h⌉`, see [`Ssfm::step_km`]) it is the exact
    /// inverse of [`Ssfm::propagate`] without noise. It back-propagates the power profile
    /// of the lumped amplifiers: [`Ssfm::raman`] is ignored.
    pub fn dbp(&self, steps_per_span: usize, xi: f64) -> Dbp {
        let f = &self.fiber;
        let forward = self.steps().ok().map(|(n, h)| Steps::new(n, h));
        let mut dbp = Dbp { steps_per_span, xi, forward, ..Dbp::new(f.medium(), f.span_km, f.n_spans, 1.0) };
        dbp.span_gain = self.amplification.field_gain(f, steps_per_span as f64 * dbp.step_km());
        dbp
    }

    /// Digital back-propagation with `steps_per_span` symmetric steps per span and the
    /// full nonlinear phase ([`Ssfm::dbp`] with `ξ = 1`).
    ///
    /// # Errors
    /// [`Error::Config`] for zero steps.
    pub fn backpropagate(&self, x: &[C64], fs: f64, steps_per_span: usize) -> Result<Vec<C64>> {
        self.backpropagate_scaled(x, fs, steps_per_span, 1.0)
    }

    /// [`Ssfm::backpropagate`] with the nonlinear phase of every step multiplied by
    /// `xi` (`xi = 0` gives the linear back-propagation).
    ///
    /// # Errors
    /// [`Error::Config`] for zero steps or a negative `xi`.
    pub fn backpropagate_scaled(&self, x: &[C64], fs: f64, steps_per_span: usize, xi: f64) -> Result<Vec<C64>> {
        self.dbp(steps_per_span, xi).run(x, fs)
    }

    /// [`Ssfm::propagate`] in single precision: the field, the FFTs, the linear
    /// operator and the nonlinear phase are computed in `f32`, the amplifiers in `f64`.
    ///
    /// The rounding error grows like `ε √K` with `ε ≈ 6·10⁻⁸` and `K` the number of
    /// steps: below `10⁻⁴` relative for thousands of steps, far below the ASE of any
    /// practical link.
    ///
    /// # Errors
    /// As [`Ssfm::propagate`], and [`Error::Config`] with Raman amplification (double
    /// precision only).
    pub fn propagate_f32(&self, x: &mut Vec<C64>, fs: f64, rng: Option<&mut Rng>) -> Result<()> {
        ensure(self.raman.is_none(), || "the Raman amplification is simulated in double precision only".into())?;
        let (n_steps, h) = self.steps()?;
        let n = x.len();
        if n == 0 {
            return Ok(());
        }
        let f = &self.fiber;
        let steps = SplitStepF32::new(&f.medium(), Steps::new(n_steps, h), n, fs);
        let simulated_km = n_steps as f64 * h;
        // the ideal gain in single precision, with the steps; the EDFA (and its ASE) in
        // double precision, between the spans, on the field rounded to single precision
        let (amplification, ideal_gain) = match self.amplification {
            Amplification::Ideal => (Amplification::None, Some(self.amplification.field_gain(f, simulated_km) as f32)),
            ref other => (other.clone(), None),
        };
        let to_f32 = |e: &[C64]| -> Vec<C32> { e.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect() };
        let to_f64 = |e: &[C32]| -> Vec<C64> { e.iter().map(|v| C64::new(f64::from(v.re), f64::from(v.im))).collect() };
        *x = to_f64(&to_f32(x));
        amplified_spans(f, simulated_km, &amplification, None, &mut [&mut *x], fs, rng, |_, fields, _| {
            let mut e32 = to_f32(fields[0]);
            steps.run(&mut e32);
            if let Some(g) = ideal_gain {
                for v in &mut e32 {
                    *v *= g;
                }
            }
            *fields[0] = to_f64(&e32);
            Ok(0)
        })?;
        *x = to_f64(&to_f32(x));
        Ok(())
    }
}

/// The split step on every stream of a [`Signal`] independently (ASE drawn from `rng`).
impl Propagate for Ssfm {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        let fs = x.sample_rate().0;
        x.try_map_streams(out, |_, xs| {
            let mut y = xs.to_vec();
            Ssfm::propagate(self, &mut y, fs, Some(&mut *rng)).map(|()| y)
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use com_math::power;

    /// `n` samples of complex Gaussian noise limited to half the band (so that the split
    /// step is accurate) and scaled to the power `p` (for the tests of the optical
    /// channels).
    pub(crate) fn signal(n: usize, seed: u64, p: f64) -> Vec<C64> {
        let mut r = Rng::new(seed);
        let x: Vec<C64> = (0..n).map(|_| r.complex_gaussian()).collect();
        let mask = fft::fftfreq(n, 1.0).into_iter().map(|f| C64::new(if f.abs() > 0.25 { 0.0 } else { 1.0 }, 0.0));
        let y = fft::filter(&x, mask);
        let s = (p / power(&y)).sqrt();
        y.iter().map(|v| v * s).collect()
    }

    /// [`Ssfm::propagate`] on a copy of `x`.
    pub(crate) fn propagated(s: &Ssfm, x: &[C64], fs: f64, rng: Option<&mut Rng>) -> Vec<C64> {
        let mut y = x.to_vec();
        s.propagate(&mut y, fs, rng).unwrap();
        y
    }

    /// [`Ssfm::propagate_f32`] on a copy of `x`.
    fn propagated_f32(s: &Ssfm, x: &[C64], fs: f64, rng: Option<&mut Rng>) -> Vec<C64> {
        let mut y = x.to_vec();
        s.propagate_f32(&mut y, fs, rng).unwrap();
        y
    }

    #[test]
    fn single_precision_split_step_matches_double_precision() {
        let fiber = FiberParams { n_spans: 3, ..FiberParams::default() };
        let mut rng = Rng::new(4);
        let x: Vec<C64> = rng.complex_noise(4096, 2e-3);
        for amplification in [Amplification::Edfa, Amplification::Ideal, Amplification::None] {
            let s = Ssfm { fiber: fiber.clone(), step_km: 0.5, amplification, raman: None };
            let a = propagated(&s, &x, 64e9, None);
            let b = propagated_f32(&s, &x, 64e9, None);
            let err: f64 = a.iter().zip(&b).map(|(u, v)| (u - v).norm_sqr()).sum::<f64>().sqrt();
            let nrm: f64 = a.iter().map(C64::norm_sqr).sum::<f64>().sqrt();
            assert!(err / nrm < 1e-4, "{:?}: {}", s.amplification, err / nrm);
        }
        // with ASE, the same draws in both precisions
        let s = Ssfm::new(fiber, 1.0);
        let a = propagated(&s, &x, 64e9, Some(&mut Rng::new(1)));
        let b = propagated_f32(&s, &x, 64e9, Some(&mut Rng::new(1)));
        let err: f64 = a.iter().zip(&b).map(|(u, v)| (u - v).norm_sqr()).sum::<f64>();
        assert!(err / a.iter().map(C64::norm_sqr).sum::<f64>() < 1e-8);
    }

    #[test]
    fn dbp_inverts_the_split_step() {
        let fiber = FiberParams { n_spans: 2, ..FiberParams::default() };
        let x = signal(256, 1, 5e-3);
        let sim = Ssfm::new(fiber, 2.0);
        let y = propagated(&sim, &x, 64e9, None);
        let z = sim.backpropagate(&y, 64e9, 40).unwrap();
        assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-9 * a.norm().max(1e-3)));
    }

    #[test]
    fn the_split_step_covers_the_whole_span_when_the_step_does_not_divide_it() {
        // 80 km in steps of 3 km: ⌈80/3⌉ = 27 steps of 80/27 km, so that the EDFA (α L =
        // 16 dB) restores the launch power exactly (26 steps of 3 km gave +0.4 dB per span)
        let fiber = FiberParams { n_spans: 10, gamma_w_km: 0.0, ..FiberParams::default() };
        let x = signal(512, 3, 1e-3);
        for amplification in [Amplification::Edfa, Amplification::Ideal] {
            let s = Ssfm { fiber: fiber.clone(), step_km: 3.0, amplification, raman: None };
            // the single precision rounds the power of 270 steps by about 1e-4 dB
            for (y, tol) in [(propagated(&s, &x, 64e9, None), 1e-9), (propagated_f32(&s, &x, 64e9, None), 1e-3)] {
                let gain_db = 10.0 * (power(&y) / power(&x)).log10();
                assert!(gain_db.abs() < tol, "{:?}: {gain_db} dB", s.amplification);
            }
        }
        // and the linear channel of the whole link is reproduced
        let lossless = FiberParams { alpha_db_km: 0.0, n_spans: 1, gamma_w_km: 0.0, ..FiberParams::default() };
        let s = Ssfm { fiber: lossless.clone(), step_km: 3.0, amplification: Amplification::None, raman: None };
        let a = linear_fiber_channel(&x, &lossless, 80.0, 64e9);
        let b = propagated(&s, &x, 64e9, None);
        assert!(a.iter().zip(&b).all(|(u, v)| (u - v).norm() < 1e-12), "the steps do not add up to 80 km");
        assert_eq!(s.steps().unwrap().0, 27);
        let exact = Ssfm { step_km: 0.4, ..s.clone() }.steps().unwrap();
        assert_eq!(exact, (200, 0.4));
        assert!(Ssfm { step_km: 81.0, ..s.clone() }.steps().is_err());
        assert!(Ssfm { step_km: 0.0, ..s }.steps().is_err());
    }

    #[test]
    fn dbp_inverts_the_split_step_when_the_step_does_not_divide_the_span() {
        let fiber = FiberParams { n_spans: 2, ..FiberParams::default() };
        let x = signal(256, 1, 5e-3);
        for step in [3.0, 7.0, 2.6] {
            let sim = Ssfm::new(fiber.clone(), step);
            let y = propagated(&sim, &x, 64e9, None);
            let z = sim.backpropagate(&y, 64e9, (80.0 / step).ceil() as usize).unwrap();
            assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-9 * a.norm().max(1e-3)), "step {step}");
        }
    }

    /// The span loops of `Ssfm::propagate` (lumped or Raman amplification) and
    /// `Ssfm::propagate_f32` as they were written before the span driver.
    pub(crate) mod former {
        use super::*;
        use crate::optical::raman::Raman;

        pub(crate) fn propagate(s: &Ssfm, x: &[C64], fs: f64, mut rng: Option<&mut Rng>) -> Vec<C64> {
            let (n_steps, h) = s.steps().unwrap();
            if let Some(raman) = &s.raman {
                return propagate_raman(s, raman, x, fs, rng);
            }
            let f = &s.fiber;
            let steps = SplitStep::new(&f.medium(), Steps::new(n_steps, h), x.len(), fs);
            let mut e = x.to_vec();
            for _ in 0..f.n_spans {
                steps.run(&mut e);
                s.amplification.apply(&mut e, f, n_steps as f64 * h, fs, rng.as_deref_mut()).unwrap();
            }
            e
        }

        fn propagate_raman(s: &Ssfm, raman: &Raman, x: &[C64], fs: f64, mut rng: Option<&mut Rng>) -> Vec<C64> {
            let (n_steps, h) = s.steps().unwrap();
            let fiber = FiberParams { span_km: n_steps as f64 * h, ..s.fiber.clone() };
            let medium = Medium { alpha_np_km: 0.0, ..fiber.medium() };
            let steps = SplitStep::new(&medium, Steps::new(n_steps, h), x.len(), fs);
            let mut e = x.to_vec();
            for _ in 0..fiber.n_spans {
                let span = raman.field_span(&fiber, &[&e], fs).unwrap();
                steps.run_with_gain(&mut e, &|z0, z1, out| span.field_gain(z0, z1, out));
                span.amplify(&mut e, fs, &s.amplification, fiber.nf_db, rng.as_deref_mut());
            }
            e
        }

        pub(crate) fn propagate_f32(s: &Ssfm, x: &[C64], fs: f64, mut rng: Option<&mut Rng>) -> Vec<C64> {
            let (n_steps, h) = s.steps().unwrap();
            let f = &s.fiber;
            let steps = SplitStepF32::new(&f.medium(), Steps::new(n_steps, h), x.len(), fs);
            let simulated_km = n_steps as f64 * h;
            let ideal_gain = s.amplification.field_gain(f, simulated_km) as f32;
            let mut e: Vec<C32> = x.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
            for _ in 0..f.n_spans {
                steps.run(&mut e);
                match s.amplification {
                    Amplification::Edfa => {
                        let mut e64: Vec<C64> = e.iter().map(|v| C64::new(f64::from(v.re), f64::from(v.im))).collect();
                        s.amplification.apply(&mut e64, f, simulated_km, fs, rng.as_deref_mut()).unwrap();
                        e = e64.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
                    }
                    Amplification::Ideal => e.iter_mut().for_each(|v| *v *= ideal_gain),
                    Amplification::None => {}
                }
            }
            e.iter().map(|v| C64::new(f64::from(v.re), f64::from(v.im))).collect()
        }
    }

    /// The span driver gives the fields of the former loops bit for bit: every
    /// amplification, with and without ASE, in double and single precision, and with
    /// Raman pumps.
    #[test]
    fn the_span_driver_is_the_former_span_loops() {
        use crate::optical::raman::{Direction, Raman, RamanPump};
        let fiber = FiberParams { n_spans: 3, span_km: 40.0, ..FiberParams::default() };
        let x = signal(512, 7, 4e-3);
        for amplification in [Amplification::Edfa, Amplification::Ideal, Amplification::None] {
            let s = Ssfm { fiber: fiber.clone(), step_km: 3.0, amplification, raman: None };
            for seed in [None, Some(5)] {
                let (mut a, mut b) = (seed.map(Rng::new), seed.map(Rng::new));
                assert_eq!(propagated(&s, &x, 64e9, a.as_mut()), former::propagate(&s, &x, 64e9, b.as_mut()));
                let (mut a, mut b) = (seed.map(Rng::new), seed.map(Rng::new));
                assert_eq!(
                    propagated_f32(&s, &x, 64e9, a.as_mut()),
                    former::propagate_f32(&s, &x, 64e9, b.as_mut()),
                    "{:?}",
                    s.amplification
                );
            }
        }
        let pump = RamanPump::at_wavelength(1450.0, 0.3, Direction::Counter);
        let raman = Raman { pumps: vec![pump], step_km: 1.0, ..Raman::default() };
        for amplification in [Amplification::Edfa, Amplification::None] {
            let s = Ssfm { fiber: fiber.clone(), step_km: 4.0, amplification, raman: Some(raman.clone()) };
            let (mut a, mut b) = (Rng::new(3), Rng::new(3));
            assert_eq!(propagated(&s, &x, 64e9, Some(&mut a)), former::propagate(&s, &x, 64e9, Some(&mut b)));
        }
    }

    #[test]
    fn opticommpy_beta2() {
        // OptiCommPy: β2 = -(D λ²)/(2π c) with D = 16, λ = c/193.1 THz
        let f = FiberParams { d_ps_nm_km: 16.0, ..FiberParams::default() };
        let c = 299_792.458;
        let lam = c / 193.1e12;
        let b2 = -(16.0 * lam * lam) / (2.0 * std::f64::consts::PI * c);
        assert!((f.beta2_s2_km() / b2 - 1.0).abs() < 1e-14);
        assert!((f.alpha_np_km() - 0.2 * 10f64.ln() / 10.0).abs() < 1e-15);
    }
}

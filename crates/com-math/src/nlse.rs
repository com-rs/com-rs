//! The symmetric split-step Fourier solvers of the nonlinear Schrödinger equation:
//! scalar ([`SplitStep`], [`SplitStepF32`], both built from the [`Steps`] of a span)
//! and dual-polarisation Manakov ([`ManakovSplitStep`]).
//!
//! The same solver propagates a field through a fibre (`com_channel::optical::fiber`)
//! and back-propagates it at the receiver (`com_dsp::compensation::dbp`, with the
//! coefficients negated), so it lives here, below both.
//!
//! # Model
//!
//! $$
//! \frac{\partial A}{\partial z} = -\frac{\alpha}{2} A
//!   - j \frac{\beta_{2}}{2} \frac{\partial^{2} A}{\partial t^{2}}
//!   + j \gamma |A|^{2} A
//!     $$
//!
//! Each step of length $h$ is a half linear step, the nonlinear phase
//! $\gamma |A|^{2} h_{\mathrm{NL}}$ and a half linear step, with the linear operator applied
//! in the frequency domain:
//!
//! $$
//! \tilde{A}(\omega) \leftarrow \tilde{A}(\omega)\\, e^{-\alpha h / 4}\\, e^{j \beta_{2} \omega^{2} h / 4}
//! $$
//!
//! The nonlinear length $h_{\mathrm{NL}}$ is the step itself for the forward model; a
//! back-propagation with coarse steps uses the effective length of the step instead.
//!
//! # The Manakov equation
//!
//! Averaged over the fast random rotations of the birefringence, the two polarisations
//! $A_{x}$, $A_{y}$ obey
//!
//! $$
//! \frac{\partial A_{x,y}}{\partial z} = -\frac{\alpha}{2} A_{x,y} -
//!   j \frac{\beta_{2}}{2} \frac{\partial^{2} A_{x,y}}{\partial t^{2}} +
//!   j \frac{8}{9} \gamma \left(|A_{x}|^{2} + |A_{y}|^{2}\right) A_{x,y}.
//! $$
//!
//! [`ManakovSplitStep`] is the symmetric scheme of OptiCommPy's `manakovSSF`: a half
//! linear step, the nonlinear phase
//!
//! $$
//! \phi(t) = \frac{8}{9} \gamma\, \frac{P(z, t) + P(z + h, t)}{2}\, h, \qquad
//! P = |A_{x}|^{2} + |A_{y}|^{2},
//! $$
//!
//! (the trapezoidal rule, iterated until the step converges since $P(z + h)$ depends
//! on the step itself) and a half linear step. The step is either fixed or adapted so
//! that the largest nonlinear phase of a step is a given rotation (the nonlinear phase
//! rotation method of Sinkin et al., which concentrates the steps where the power is
//! high, at the beginning of the spans).
//!
//! # Examples
//! ```
//! use com_math::nlse::{Medium, SplitStep, Steps};
//! use com_math::C64;
//! // without loss and nonlinearity, forward then negated propagation is the identity
//! let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 0.0 };
//! let x: Vec<C64> = (0..64).map(|n| C64::new((n % 3) as f64, (n % 2) as f64)).collect();
//! let mut e = x.clone();
//! SplitStep::new(&m, Steps::new(8, 10.0), 64, 64e9).run(&mut e);
//! SplitStep::new(&m.negated(), Steps::new(8, 10.0), 64, 64e9).run(&mut e);
//! assert!(x.iter().zip(&e).all(|(a, b)| (a - b).norm() < 1e-9));
//! ```
//!
//! # References
//!
//! 1. G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013.
//! 2. E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments
//!    using digital backpropagation," *J. Lightw. Technol.*, 26(20), 2008.
//! 3. D. Marcuse, C. R. Menyuk and P. K. A. Wai, "Application of the Manakov-PMD
//!    equation to studies of signal propagation in optical fibers with randomly varying
//!    birefringence," *J. Lightw. Technol.*, 15(9), 1735–1746, 1997.
//! 4. O. V. Sinkin, R. Holzlöhner, J. Zweck and C. R. Menyuk, "Optimization of the
//!    split-step Fourier method in modeling optical-fiber communications systems," *J.
//!    Lightw. Technol.*, 21(1), 61–68, 2003.

use crate::fft::{self, Fft};
use crate::{C64, cis};
use com_core::{Error, Result, ensure, same_len};
use num_complex::Complex32 as C32;

/// The coefficients of the scalar NLSE, per kilometre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Medium {
    /// Power attenuation `α` (Np/km).
    pub alpha_np_km: f64,
    /// Group-velocity dispersion `β₂` (s²/km).
    pub beta2_s2_km: f64,
    /// Nonlinear coefficient `γ` (1/(W·km)).
    pub gamma_w_km: f64,
}

impl Medium {
    /// The medium with `α`, `β₂` and `γ` reversed: propagating through it undoes a
    /// propagation through `self` (digital back-propagation).
    pub fn negated(&self) -> Medium {
        Medium { alpha_np_km: -self.alpha_np_km, beta2_s2_km: -self.beta2_s2_km, gamma_w_km: -self.gamma_w_km }
    }

    /// The linear response `e^{-αL/2} e^{jβ₂ω²L/2}` of `length_km` of the medium at
    /// every angular frequency (loss and dispersion of the field).
    pub fn linear_response(&self, omega: &[f64], length_km: f64) -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); omega.len()];
        self.linear_response_into(omega, length_km, &mut out);
        out
    }

    /// [`Medium::linear_response`] into `out` (one value per angular frequency).
    fn linear_response_into(&self, omega: &[f64], length_km: f64, out: &mut [C64]) {
        let (a, b2) = (self.alpha_np_km, self.beta2_s2_km);
        for (o, wk) in out.iter_mut().zip(omega) {
            *o = (-(a / 2.0) * length_km).exp() * cis(b2 / 2.0 * wk * wk * length_km);
        }
    }

    /// The half linear step `e^{-αh/4} e^{jβ₂ω²h/4}` at every angular frequency.
    fn half_step(&self, omega: &[f64], h: f64) -> Vec<C64> {
        self.linear_response(omega, h / 2.0)
    }

    /// [`Medium::half_step`] into `out`.
    fn half_step_into(&self, omega: &[f64], h: f64, out: &mut [C64]) {
        self.linear_response_into(omega, h / 2.0, out);
    }
}

/// The steps of one span: `n_steps` symmetric steps of `step_km`, the nonlinear phase
/// of each over `nl_length_km` (the step itself for the forward model, the effective
/// length of a coarse step for a back-propagation). Shared by [`SplitStep`] and
/// [`SplitStepF32`], and given by a forward model to its back-propagation
/// (`com_dsp::compensation::dbp::Dbp::forward`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steps {
    /// Number of steps.
    pub n_steps: usize,
    /// Length of a step (km).
    pub step_km: f64,
    /// Nonlinear length of a step (km).
    pub nl_length_km: f64,
}

impl Steps {
    /// The steps of the forward model: `n_steps` of `step_km`, whose nonlinear length is
    /// the step.
    pub fn new(n_steps: usize, step_km: f64) -> Steps {
        Steps { n_steps, step_km, nl_length_km: step_km }
    }

    /// The length covered, $n h$ (km).
    pub fn length_km(&self) -> f64 {
        self.n_steps as f64 * self.step_km
    }
}

/// The [`Steps`] of one span in double precision.
#[derive(Clone, Debug)]
pub struct SplitStep {
    lin: Vec<C64>,
    /// `None` for an empty block (nothing to propagate).
    plan: Option<Fft>,
    gamma_w_km: f64,
    steps: Steps,
}

impl SplitStep {
    /// The `steps` through `medium` for a block of `n` samples at `fs` (no plan for an
    /// empty block: the steps then leave it empty).
    pub fn new(medium: &Medium, steps: Steps, n: usize, fs: f64) -> SplitStep {
        let omega = fft::angular_frequencies(n, fs);
        SplitStep {
            lin: medium.half_step(&omega, steps.step_km),
            plan: (n > 0).then(|| fft::plan(n)),
            gamma_w_km: medium.gamma_w_km,
            steps,
        }
    }

    /// The steps.
    pub fn steps(&self) -> Steps {
        self.steps
    }

    /// Propagate `e` (√W, time domain) through the steps, in place.
    ///
    /// # Panics
    /// If `e` has another length than the block of [`SplitStep::new`].
    pub fn run(&self, e: &mut [C64]) {
        let Some(p) = self.plan.as_ref() else {
            assert!(e.is_empty(), "a field for an empty block");
            return;
        };
        let (g, h) = (self.gamma_w_km, self.steps.nl_length_km);
        p.forward(e);
        for _ in 0..self.steps.n_steps {
            mul(e, &self.lin);
            p.inverse(e);
            for v in e.iter_mut() {
                *v *= cis(g * v.norm_sqr() * h);
            }
            p.forward(e);
            mul(e, &self.lin);
        }
        p.inverse(e);
    }

    /// The steps for the Manakov equation: both polarisations (√W, time domain, equal
    /// lengths) in place, with the nonlinear phase
    /// $\frac{8}{9} \gamma (|A_{x}|^{2} + |A_{y}|^{2}) h_{\mathrm{NL}}$ at the middle of every
    /// step (the back-propagation of a dual-polarisation link).
    ///
    /// # Panics
    /// If the polarisations have different lengths, or another length than the block of
    /// [`SplitStep::new`].
    pub fn run_dual(&self, x: &mut [C64], y: &mut [C64]) {
        assert_eq!(x.len(), y.len(), "the polarisations must have the same length");
        let Some(p) = self.plan.as_ref() else {
            assert!(x.is_empty(), "a field for an empty block");
            return;
        };
        let k = 8.0 / 9.0 * self.gamma_w_km * self.steps.nl_length_km;
        p.forward(x);
        p.forward(y);
        for _ in 0..self.steps.n_steps {
            mul(x, &self.lin);
            mul(y, &self.lin);
            p.inverse(x);
            p.inverse(y);
            manakov_rotation(x, y, k);
            p.forward(x);
            p.forward(y);
            mul(x, &self.lin);
            mul(y, &self.lin);
        }
        p.inverse(x);
        p.inverse(y);
    }

    /// [`SplitStep::run`] with a distributed gain: before the nonlinear phase and after
    /// it, every half step also multiplies the spectrum by `gain(z_0, z_1, g)`, the gain
    /// of the field at every frequency (FFT order) from $z_{0}$ to $z_{1}$ (km, from the
    /// start of the steps). The medium then carries no attenuation: `gain` holds the
    /// loss as well (the power profile of a Raman-pumped span, `com_channel::optical::raman`).
    /// As in [`SplitStep::run`], the field stays in the frequency domain from the end of
    /// a step to the start of the next: two FFTs per step.
    ///
    /// # Panics
    /// If `e` has another length than the block of [`SplitStep::new`].
    pub fn run_with_gain(&self, e: &mut [C64], gain: FieldGain<'_>) {
        let Some(p) = self.plan.as_ref() else {
            assert!(e.is_empty(), "a field for an empty block");
            return;
        };
        let (g, h) = (self.gamma_w_km, self.steps.nl_length_km);
        let step = self.steps.step_km;
        let mut buf = vec![0.0; e.len()];
        p.forward(e);
        for k in 0..self.steps.n_steps {
            let z0 = k as f64 * step;
            mul(e, &self.lin);
            gain(z0, z0 + step / 2.0, &mut buf);
            scale(e, &buf);
            p.inverse(e);
            for v in e.iter_mut() {
                *v *= cis(g * v.norm_sqr() * h);
            }
            p.forward(e);
            mul(e, &self.lin);
            gain(z0 + step / 2.0, z0 + step, &mut buf);
            scale(e, &buf);
        }
        p.inverse(e);
    }
}

/// The gain of the field at every frequency bin (FFT order) between two positions of a
/// span (km): `gain(z_0, z_1, out)` writes $\sqrt{G(z_{1}, f) / G(z_{0}, f)}$ into `out`.
pub type FieldGain<'a> = &'a dyn Fn(f64, f64, &mut [f64]);

/// [`SplitStep`] in single precision: the field, the FFTs, the linear operator and the
/// nonlinear phase are computed in `f32` (about twice as fast), with the same [`Steps`]
/// (a back-propagation included); the plans and their work space are cached per thread
/// as those of [`fft::plan`].
///
/// The rounding error grows like `ε √K` with `ε ≈ 6·10⁻⁸` and `K` the number of
/// steps: below `10⁻⁴` relative for thousands of steps.
#[derive(Clone)]
pub struct SplitStepF32 {
    lin: Vec<C32>,
    lin_n: Vec<C32>,
    /// `None` for an empty block.
    plan: Option<fft::FftF32>,
    gamma_h: f32,
    steps: Steps,
}

impl std::fmt::Debug for SplitStepF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplitStepF32")
            .field("n", &self.lin.len())
            .field("gamma_h", &self.gamma_h)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

impl SplitStepF32 {
    /// The `steps` through `medium` for a block of `n` samples at `fs` (no plan for an
    /// empty block).
    pub fn new(medium: &Medium, steps: Steps, n: usize, fs: f64) -> SplitStepF32 {
        let omega = fft::angular_frequencies(n, fs);
        let half = medium.half_step(&omega, steps.step_km);
        // the 1/N of the inverse transform is folded into the first half step
        let inv_n = 1.0 / n.max(1) as f64;
        let lin = half.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
        let lin_n = half.iter().map(|v| v * inv_n).map(|v| C32::new(v.re as f32, v.im as f32)).collect();
        SplitStepF32 {
            lin,
            lin_n,
            plan: (n > 0).then(|| fft::plan_f32(n)),
            gamma_h: (medium.gamma_w_km * steps.nl_length_km) as f32,
            steps,
        }
    }

    /// The steps.
    pub fn steps(&self) -> Steps {
        self.steps
    }

    /// Propagate `e` through the steps, in place.
    ///
    /// # Panics
    /// If `e` has another length than the block of [`SplitStepF32::new`].
    pub fn run(&self, e: &mut [C32]) {
        let Some(p) = self.plan.as_ref() else {
            assert!(e.is_empty(), "a field for an empty block");
            return;
        };
        assert_eq!(e.len(), self.lin.len(), "FFT length mismatch");
        let n = e.len();
        p.forward(e);
        for _ in 0..self.steps.n_steps {
            e.iter_mut().zip(&self.lin_n).for_each(|(u, l)| *u *= l);
            p.inverse_unnormalized(e);
            for v in e.iter_mut() {
                let (sn, cs) = (self.gamma_h * v.norm_sqr()).sin_cos();
                *v *= C32::new(cs, sn);
            }
            p.forward(e);
            e.iter_mut().zip(&self.lin).for_each(|(u, l)| *u *= l);
        }
        p.inverse_unnormalized(e);
        let s = 1.0 / n as f32;
        for v in e.iter_mut() {
            *v *= s;
        }
    }
}

/// The step size of [`ManakovSplitStep`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum StepRule {
    /// Steps of a fixed length (km); the last step of a span is shortened.
    Fixed(f64),
    /// Steps whose largest nonlinear phase $\max_{t} \phi(t)$ is this rotation (rad).
    NonlinearPhase(f64),
}

/// The symmetric split-step solver of the Manakov equation (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManakovSplitStep {
    /// The fibre.
    pub medium: Medium,
    /// The step size.
    pub rule: StepRule,
    /// Largest number of trapezoidal iterations per step.
    pub max_iterations: usize,
    /// Convergence tolerance of the iterations: the relative change of the fields.
    pub tolerance: f64,
}

/// The buffers of [`ManakovSplitStep`], allocated once per run.
struct ManakovWork {
    /// The angular frequencies of the bins.
    omega: Vec<f64>,
    /// $P(z)$, the total power at the start of the step.
    p0: Vec<f64>,
    /// The total power of the last iteration, then the phase $\phi$ of the trapezoid.
    p1: Vec<f64>,
    phi: Vec<f64>,
    /// The half linear step, and its two versions with a distributed gain.
    lin: Vec<C64>,
    lin1: Vec<C64>,
    lin2: Vec<C64>,
    gain: Vec<f64>,
    /// The fields after the first half step, and those of the current iteration.
    xh: Vec<C64>,
    yh: Vec<C64>,
    xf: Vec<C64>,
    yf: Vec<C64>,
    /// The fields of the last iteration (the field at z at the start of a step).
    xc: Vec<C64>,
    yc: Vec<C64>,
}

impl ManakovSplitStep {
    /// The solver of OptiCommPy's defaults: steps of 0.02 rad of nonlinear phase, at
    /// most 10 iterations, tolerance $10^{-5}$.
    pub fn new(medium: Medium) -> ManakovSplitStep {
        ManakovSplitStep { medium, rule: StepRule::NonlinearPhase(2e-2), max_iterations: 10, tolerance: 1e-5 }
    }

    /// Propagate the two polarisations (√W, time domain, equal lengths) over
    /// `length_km` of fibre sampled at `fs`, in place: the number of steps (none for a
    /// length that is not positive).
    ///
    /// # Errors
    /// [`Error::Shape`] if the polarisations have different lengths, [`Error::Config`]
    /// for a fixed step or a rotation that is not positive or a length that is not
    /// finite, [`Error::Numerical`] if a step vanishes (a non-finite field).
    pub fn run(&self, x: &mut [C64], y: &mut [C64], length_km: f64, fs: f64) -> Result<usize> {
        self.run_inner(x, y, length_km, fs, None)
    }

    /// [`ManakovSplitStep::run`] with a distributed gain (see
    /// [`SplitStep::run_with_gain`]): the medium carries no attenuation, `gain` holds the
    /// power profile of the span.
    ///
    /// # Errors
    /// As [`ManakovSplitStep::run`].
    pub fn run_with_gain(
        &self,
        x: &mut [C64],
        y: &mut [C64],
        length_km: f64,
        fs: f64,
        gain: FieldGain<'_>,
    ) -> Result<usize> {
        self.run_inner(x, y, length_km, fs, Some(gain))
    }

    /// The parameter of the step rule, checked: the fixed step (km) or the rotation (rad).
    fn rule_parameter(&self) -> Result<f64> {
        let (step, what) = match self.rule {
            StepRule::Fixed(h) => (h, "fixed step (km)"),
            StepRule::NonlinearPhase(r) => (r, "nonlinear rotation (rad)"),
        };
        ensure(step > 0.0, || format!("the {what} must be positive, got {step}"))?;
        Ok(step)
    }

    /// The length of the next step at `z` with `rest` km left and the phase `phi`
    /// (rad/km) of the step.
    ///
    /// # Errors
    /// [`Error::Numerical`] for a step that does not advance `z`.
    fn step_length(&self, phi: &[f64], z: f64, rest: f64) -> Result<f64> {
        let h = match self.rule {
            StepRule::Fixed(hf) => hf.min(rest),
            StepRule::NonlinearPhase(rot) => {
                let m = phi.iter().copied().fold(0.0, f64::max);
                if m > 0.0 && rest >= rot / m { rot / m } else { rest }
            }
        };
        // a step that does not advance (an infinite power, a step below the precision
        // of z) would never reach the end of the fibre
        if z + h <= z || h.is_nan() {
            return Err(Error::Numerical(format!("the split step vanishes at z = {z} km (step {h} km)")));
        }
        Ok(h)
    }

    fn run_inner(
        &self,
        x: &mut [C64],
        y: &mut [C64],
        length_km: f64,
        fs: f64,
        gain: Option<FieldGain<'_>>,
    ) -> Result<usize> {
        same_len(x.len(), y.len(), "the polarisations")?;
        self.rule_parameter()?;
        ensure(length_km.is_finite(), || format!("the length must be finite, got {length_km} km"))?;
        let n = x.len();
        if n == 0 || length_km <= 0.0 {
            return Ok(0);
        }
        let p = fft::plan(n);
        let zero = C64::new(0.0, 0.0);
        let mut w = ManakovWork {
            omega: fft::angular_frequencies(n, fs),
            p0: vec![0.0; n],
            p1: vec![0.0; n],
            phi: vec![0.0; n],
            lin: vec![zero; n],
            lin1: vec![zero; n],
            lin2: vec![zero; n],
            gain: vec![0.0; n],
            xh: vec![zero; n],
            yh: vec![zero; n],
            xf: vec![zero; n],
            yf: vec![zero; n],
            xc: x.to_vec(),
            yc: y.to_vec(),
        };
        let k = 8.0 / 9.0 * self.medium.gamma_w_km;
        let mut z = 0.0;
        let mut steps = 0;
        while z < length_km {
            total_power(x, y, &mut w.p0);
            total_power(&w.xc, &w.yc, &mut w.p1);
            trapezoid(&w.p0, &w.p1, k, &mut w.phi);
            let h = self.step_length(&w.phi, z, length_km - z)?;
            self.medium.half_step_into(&w.omega, h, &mut w.lin);
            // the two half steps differ only with a distributed gain
            if let Some(g) = gain {
                for (half, z0, z1) in [(&mut w.lin1, z, z + h / 2.0), (&mut w.lin2, z + h / 2.0, z + h)] {
                    g(z0, z1, &mut w.gain);
                    for ((o, l), b) in half.iter_mut().zip(&w.lin).zip(&w.gain) {
                        *o = l * b;
                    }
                }
            }
            let (lin1, lin2) = if gain.is_some() { (&w.lin1, &w.lin2) } else { (&w.lin, &w.lin) };
            w.xh.copy_from_slice(x);
            w.yh.copy_from_slice(y);
            for u in [&mut w.xh, &mut w.yh] {
                p.forward(u);
                mul(u, lin1);
                p.inverse(u);
            }
            for it in 0..self.max_iterations.max(1) {
                for (o, (v, f)) in w.xf.iter_mut().zip(w.xh.iter().zip(&w.phi)) {
                    *o = v * cis(f * h);
                }
                for (o, (v, f)) in w.yf.iter_mut().zip(w.yh.iter().zip(&w.phi)) {
                    *o = v * cis(f * h);
                }
                for u in [&mut w.xf, &mut w.yf] {
                    p.forward(u);
                    mul(u, lin2);
                    p.inverse(u);
                }
                let diff: f64 =
                    w.xf.iter().zip(&w.xc).chain(w.yf.iter().zip(&w.yc)).map(|(u, v)| (u - v).norm_sqr()).sum();
                let norm: f64 = w.xc.iter().chain(&w.yc).map(C64::norm_sqr).sum();
                std::mem::swap(&mut w.xc, &mut w.xf);
                std::mem::swap(&mut w.yc, &mut w.yf);
                if norm == 0.0 || (diff / norm).sqrt() < self.tolerance || it + 1 == self.max_iterations {
                    break;
                }
                total_power(&w.xc, &w.yc, &mut w.p1);
                trapezoid(&w.p0, &w.p1, k, &mut w.phi);
            }
            x.copy_from_slice(&w.xc);
            y.copy_from_slice(&w.yc);
            z += h;
            steps += 1;
        }
        Ok(steps)
    }
}

/// $P = |A_{x}|^{2} + |A_{y}|^{2}$ at every sample.
fn total_power(u: &[C64], v: &[C64], out: &mut [f64]) {
    for (o, (p, q)) in out.iter_mut().zip(u.iter().zip(v)) {
        *o = p.norm_sqr() + q.norm_sqr();
    }
}

/// The phase per kilometre of the trapezoidal rule, $\phi = k (P_{0} + P_{1}) / 2$.
fn trapezoid(p0: &[f64], p1: &[f64], k: f64, phi: &mut [f64]) {
    for (f, (u, v)) in phi.iter_mut().zip(p0.iter().zip(p1)) {
        *f = k * (u + v) / 2.0;
    }
}

/// The nonlinear step of the Manakov equation with a phase that does not change during
/// the step: both polarisations rotated by the same phase,
///
/// $$
/// A_{x,y}(t) \leftarrow A_{x,y}(t)\, e^{j k (|A_{x}(t)|^{2} + |A_{y}(t)|^{2})},
/// $$
///
/// with $k = \frac{8}{9} \gamma h_{\mathrm{NL}}$ (negated to undo it). The step of
/// [`SplitStep::run_dual`] and of the Manakov-PMD equation of
/// `com_channel::optical::pmd::ManakovPmd`.
///
/// # Panics
/// If the polarisations have different lengths.
///
/// # Examples
/// ```
/// use com_math::{C64, nlse::manakov_rotation};
/// let (mut x, mut y) = ([C64::new(1.0, 0.0)], [C64::new(0.0, 1.0)]);
/// manakov_rotation(&mut x, &mut y, std::f64::consts::FRAC_PI_4);
/// assert!((x[0] - C64::new(0.0, 1.0)).norm() < 1e-15 && (y[0] - C64::new(-1.0, 0.0)).norm() < 1e-15);
/// ```
pub fn manakov_rotation(x: &mut [C64], y: &mut [C64], k: f64) {
    assert_eq!(x.len(), y.len(), "the polarisations must have the same length");
    for (u, v) in x.iter_mut().zip(y.iter_mut()) {
        let e = cis(k * (u.norm_sqr() + v.norm_sqr()));
        *u *= e;
        *v *= e;
    }
}

fn mul(a: &mut [C64], b: &[C64]) {
    for (u, v) in a.iter_mut().zip(b) {
        *u *= v;
    }
}

fn scale(a: &mut [C64], b: &[f64]) {
    for (u, v) in a.iter_mut().zip(b) {
        *u *= v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_field_propagates_to_an_empty_field_instead_of_panicking() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        let steps = Steps { n_steps: 3, step_km: 1.0, nl_length_km: 0.8 };
        let s = SplitStep::new(&m, steps, 0, 64e9);
        let (mut x, mut y): (Vec<C64>, Vec<C64>) = (Vec::new(), Vec::new());
        s.run(&mut x);
        s.run_dual(&mut x, &mut y);
        s.run_with_gain(&mut x, &|_, _, _| {});
        SplitStepF32::new(&m, steps, 0, 64e9).run(&mut []);
        assert!(x.is_empty() && y.is_empty() && s.steps() == steps);
        assert!((steps.length_km() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn a_uniform_gain_of_the_loss_is_the_attenuation_of_the_medium() {
        // run_with_gain on the lossless medium with the field gain e^{-α Δz / 2} at every
        // frequency is run on the lossy medium (to the rounding of the factorisation)
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        let lossless = Medium { alpha_np_km: 0.0, ..m };
        let x: Vec<C64> = (0..256).map(|n| cis(0.3 * n as f64) * (1e-2 * (1 + n % 5) as f64)).collect();
        let steps = Steps::new(40, 0.5);
        let (mut a, mut b) = (x.clone(), x);
        SplitStep::new(&m, steps, 256, 64e9).run(&mut a);
        let loss = |z0: f64, z1: f64, out: &mut [f64]| out.fill((-(0.046 / 2.0) * (z1 - z0)).exp());
        SplitStep::new(&lossless, steps, 256, 64e9).run_with_gain(&mut b, &loss);
        let peak = a.iter().map(|v| v.norm()).fold(0.0, f64::max);
        assert!(a.iter().zip(&b).all(|(u, v)| (u - v).norm() < 1e-13 * peak));
    }

    #[test]
    fn lossless_split_step_conserves_energy() {
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        let mut e: Vec<C64> = (0..256).map(|n| cis(0.1 * n as f64) * (1e-2 * (n % 7) as f64)).collect();
        let e0: f64 = e.iter().map(C64::norm_sqr).sum();
        SplitStep::new(&m, Steps::new(80, 1.0), 256, 64e9).run(&mut e);
        let e1: f64 = e.iter().map(C64::norm_sqr).sum();
        assert!((e1 / e0 - 1.0).abs() < 1e-12);
    }

    /// Two polarisations of about 30 GHz of bandwidth at 256 GS/s, of mean power 10 mW.
    fn dual_field(n: usize) -> (Vec<C64>, Vec<C64>) {
        let w = |k: usize, f: f64, ph: f64| (2.0 * std::f64::consts::PI * f * k as f64 / n as f64 + ph).sin();
        let x = (0..n)
            .map(|k| C64::new(w(k, 61.0, 0.3), w(k, 23.0, 1.1)) * (0.07 * (1.0 + 0.5 * w(k, 7.0, 0.0))))
            .collect();
        let y = (0..n)
            .map(|k| C64::new(w(k, 41.0, 2.0), w(k, 89.0, 0.4)) * (0.07 * (1.0 + 0.5 * w(k, 5.0, 1.0))))
            .collect();
        (x, y)
    }

    #[test]
    fn manakov_with_one_polarisation_is_the_scalar_equation_with_8_9_gamma() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        let (x0, _) = dual_field(512);
        let (mut x, mut y) = (x0.clone(), vec![C64::new(0.0, 0.0); 512]);
        let s = ManakovSplitStep { rule: StepRule::Fixed(0.01), max_iterations: 20, tolerance: 1e-12, medium: m };
        s.run(&mut x, &mut y, 20.0, 128e9).unwrap();
        let mut e = x0;
        let scalar = Medium { gamma_w_km: 8.0 / 9.0 * 1.3, ..m };
        SplitStep::new(&scalar, Steps::new(2000, 0.01), 512, 128e9).run(&mut e);
        let err = x.iter().zip(&e).map(|(u, v)| (u - v).norm()).fold(0.0, f64::max);
        let peak = e.iter().map(|v| v.norm()).fold(0.0, f64::max);
        assert!(err < 1e-4 * peak, "{}", err / peak);
        assert!(y.iter().all(|v| v.norm() == 0.0));
    }

    #[test]
    fn manakov_rejects_a_step_that_does_not_advance_instead_of_looping() {
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        for rule in [
            StepRule::Fixed(0.0),
            StepRule::Fixed(-1.0),
            StepRule::Fixed(f64::NAN),
            StepRule::NonlinearPhase(0.0),
            StepRule::NonlinearPhase(-2e-2),
        ] {
            let (mut x, mut y) = dual_field(64);
            let s = ManakovSplitStep { rule, ..ManakovSplitStep::new(m) };
            assert!(matches!(s.run(&mut x, &mut y, 1.0, 256e9), Err(Error::Config(_))), "{rule:?}");
        }
        let s = ManakovSplitStep::new(m);
        let (mut x, mut y) = dual_field(64);
        assert!(matches!(s.run(&mut x, &mut y, f64::INFINITY, 256e9), Err(Error::Config(_))));
        assert!(matches!(s.run(&mut x, &mut y[..32], 1.0, 256e9), Err(Error::Shape(_))));
        assert_eq!(s.run(&mut x, &mut y, 0.0, 256e9).unwrap(), 0);
        // an infinite power: a nonlinear step of zero length
        x[0] = C64::new(f64::INFINITY, 0.0);
        assert!(matches!(s.run(&mut x, &mut y, 1.0, 256e9), Err(Error::Numerical(_))));
    }

    #[test]
    fn manakov_conserves_the_total_energy_and_converges_with_the_step() {
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        let run = |rule| {
            let (mut x, mut y) = dual_field(1024);
            let s = ManakovSplitStep { rule, ..ManakovSplitStep::new(m) };
            let steps = s.run(&mut x, &mut y, 80.0, 256e9).unwrap();
            (x, y, steps)
        };
        let (x0, y0) = dual_field(1024);
        let e0: f64 = x0.iter().chain(&y0).map(C64::norm_sqr).sum();
        let (xa, ya, na) = run(StepRule::NonlinearPhase(2e-2));
        let (xb, yb, nb) = run(StepRule::NonlinearPhase(2e-4));
        let e1: f64 = xa.iter().chain(&ya).map(C64::norm_sqr).sum();
        assert!((e1 / e0 - 1.0).abs() < 1e-10);
        assert!(nb > 50 * na);
        // second order: the error of the coarse solution is small against the fine one
        let err: f64 = xa.iter().zip(&xb).chain(ya.iter().zip(&yb)).map(|(u, v)| (u - v).norm_sqr()).sum();
        assert!((err / e0).sqrt() < 1e-3, "{}", (err / e0).sqrt());
    }

    #[test]
    fn fundamental_soliton_keeps_its_shape() {
        // T0 = 10 ps, β₂ = -20 ps²/km, P0 = |β₂| / (γ T0²): |A(z, t)| = |A(0, t)|
        let (t0, b2, gamma): (f64, f64, f64) = (10e-12, -20e-24, 1.0);
        let p0 = b2.abs() / (gamma * t0 * t0);
        let (n, fs) = (1024, 2e12);
        let a0: Vec<C64> = (0..n)
            .map(|k| {
                let t = (k as f64 - n as f64 / 2.0) / fs;
                C64::new(p0.sqrt() / (t / t0).cosh(), 0.0)
            })
            .collect();
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: b2, gamma_w_km: gamma };
        let mut a = a0.clone();
        // two dispersion lengths (L_D = T0²/|β₂| = 5 km)
        SplitStep::new(&m, Steps::new(1000, 0.01), n, fs).run(&mut a);
        let err = a.iter().zip(&a0).map(|(u, v)| (u.norm() - v.norm()).abs()).fold(0.0, f64::max);
        assert!(err < 1e-3 * p0.sqrt(), "relative envelope error {}", err / p0.sqrt());
    }
}

//! Digital back-propagation.
//!
//! The receiver runs the split-step solver of [`com_math::nlse`] through the negated
//! medium, span after span in reverse order, removing the amplifier gain first. Given the
//! [`Steps`] of the forward model ([`Dbp::forward`]) and as many steps per span, it
//! uses those steps and inverts the forward split step exactly without noise.
//!
//! # Model
//!
//! With coarser steps of length $\delta$, the nonlinear phase of a step is
//! $\xi \gamma |A|^{2} L_{\mathrm{eff}}(\delta)$ with the effective length of the symmetric step
//!
//! $$
//! L_{\mathrm{eff}}(\delta) = \frac{2 \sinh(\alpha \delta / 2)}{\alpha}
//! $$
//!
//! (the power at the middle of the step times $L_{\mathrm{eff}}$ is the power integrated over
//! the step); with the step length itself, one step per span would apply about 60 % of
//! the nonlinear phase of an 80 km span. The scale $\xi$ is the classical tuning of DBP
//! with few steps per span, where the full phase over-compensates at high symbol rates
//! and an optimised $\xi < 1$ performs better; $\xi = 0$ gives the linear
//! back-propagation.
//!
//! [`Dbp::run_dual`] back-propagates the two polarisations of a dual-polarisation link
//! through the Manakov equation (nonlinear coefficient $\frac{8}{9}\gamma$ on the total
//! power), with the same steps and the same effective length.
//!
//! # Examples
//! ```
//! use com_dsp::compensation::dbp::Dbp;
//! use com_math::nlse::{Medium, SplitStep, Steps};
//! use com_math::C64;
//! let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
//! let x: Vec<C64> = (0..128).map(|n| C64::new(1e-2 * (n % 3) as f64, 0.0)).collect();
//! let mut y = x.clone();
//! let forward = Steps::new(40, 2.0);                         // one 80 km span
//! SplitStep::new(&m, forward, 128, 64e9).run(&mut y);
//! let dbp = Dbp { forward: Some(forward), steps_per_span: 40, ..Dbp::new(m, 80.0, 1, 1.0) };
//! let z = dbp.run(&y, 64e9).unwrap();
//! assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-12));
//! ```
//!
//! # References
//!
//! 1. E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments
//!    using digital backpropagation," *J. Lightw. Technol.*, 26(20), 2008.

use com_core::{Result, ensure};
use com_math::C64;
use com_math::nlse::{Medium, SplitStep, Steps};

/// Digital back-propagation of a link of `n_spans` identical spans.
#[derive(Clone, Debug, PartialEq)]
pub struct Dbp {
    /// The forward medium of the link (it is negated inside).
    pub medium: Medium,
    /// Length of a span (km).
    pub span_km: f64,
    /// Number of spans.
    pub n_spans: usize,
    /// Field gain of the amplifier after every span (removed before each span).
    pub span_gain: f64,
    /// Symmetric steps per span.
    pub steps_per_span: usize,
    /// Scale `ξ ≥ 0` of the nonlinear phase.
    pub xi: f64,
    /// The steps of a span of the forward model, if known: with as many steps per span
    /// (`steps_per_span == forward.n_steps`, compared as integers) the back-propagation
    /// uses exactly those steps and their nonlinear length (the exact inverse).
    pub forward: Option<Steps>,
}

impl Dbp {
    /// One step per span, the full nonlinear phase, the effective length.
    pub fn new(medium: Medium, span_km: f64, n_spans: usize, span_gain: f64) -> Dbp {
        Dbp { medium, span_km, n_spans, span_gain, steps_per_span: 1, xi: 1.0, forward: None }
    }

    /// The forward steps, when they are those of the back-propagation.
    fn exact(&self) -> Option<Steps> {
        self.forward.filter(|f| f.n_steps == self.steps_per_span)
    }

    /// The step length (km): that of the forward model with as many steps, the span
    /// divided by the steps otherwise.
    pub fn step_km(&self) -> f64 {
        match self.exact() {
            Some(f) => f.step_km,
            None => self.span_km / self.steps_per_span as f64,
        }
    }

    /// The steps of one span for a block of `n` samples at `fs` (an empty block has
    /// nothing to propagate).
    fn steps(&self, n: usize, fs: f64) -> Result<SplitStep> {
        ensure(self.steps_per_span >= 1, || "at least one step per span".into())?;
        ensure(self.xi >= 0.0, || format!("the nonlinear scale must be non-negative, got {}", self.xi))?;
        let h = self.step_km();
        let a = self.medium.alpha_np_km;
        // nonlinear length of a step: that of the forward model for the exact inverse,
        // the effective length of the symmetric step otherwise
        let h_nl = self.xi
            * match self.exact() {
                Some(f) => f.nl_length_km,
                None if a * h < 1e-12 => h,
                None => 2.0 * (a * h / 2.0).sinh() / a,
            };
        let steps = Steps { n_steps: self.steps_per_span, step_km: h, nl_length_km: h_nl };
        Ok(SplitStep::new(&self.medium.negated(), steps, n, fs))
    }

    /// Back-propagate a field (√W) sampled at `fs`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for zero steps or a negative `xi`.
    pub fn run(&self, x: &[C64], fs: f64) -> Result<Vec<C64>> {
        let steps = self.steps(x.len(), fs)?;
        let mut e = x.to_vec();
        for _ in 0..self.n_spans {
            for v in &mut e {
                *v /= self.span_gain;
            }
            steps.run(&mut e);
        }
        Ok(e)
    }

    /// Back-propagate both polarisations (√W) of a dual-polarisation link sampled at
    /// `fs` through the Manakov equation (nonlinear coefficient $\frac{8}{9}\gamma$ on
    /// the total power), in place.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for zero steps or a negative `xi`,
    /// [`com_core::Error::Shape`] for polarisations of different lengths.
    pub fn run_dual(&self, x: &mut [C64], y: &mut [C64], fs: f64) -> Result<()> {
        com_core::same_len(x.len(), y.len(), "the polarisations")?;
        let steps = self.steps(x.len(), fs)?;
        for _ in 0..self.n_spans {
            for v in x.iter_mut().chain(y.iter_mut()) {
                *v /= self.span_gain;
            }
            steps.run_dual(x, y);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_math::nlse::{ManakovSplitStep, StepRule};

    #[test]
    fn an_empty_field_is_back_propagated_to_an_empty_field_instead_of_panicking() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        assert_eq!(Dbp::new(m, 80.0, 1, 1.0).run(&[], 64e9).unwrap(), Vec::<C64>::new());
        let (mut x, mut y) = (Vec::new(), Vec::new());
        Dbp { steps_per_span: 4, ..Dbp::new(m, 80.0, 2, 3.0) }.run_dual(&mut x, &mut y, 64e9).unwrap();
        assert!(x.is_empty() && y.is_empty());
    }

    #[test]
    fn the_forward_steps_are_recognised_by_their_number_not_by_a_floor_of_lengths() {
        // 80 km in 115 steps: 80/115 is not representable, and the floor of 80 over it
        // may be 114
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3 };
        let forward = Steps::new(115, 80.0 / 115.0);
        let x: Vec<C64> = (0..64).map(|n| C64::new(1e-2 * (n % 5) as f64, 1e-3 * (n % 2) as f64)).collect();
        let mut y = x.clone();
        SplitStep::new(&m, forward, 64, 64e9).run(&mut y);
        let dbp = Dbp { forward: Some(forward), steps_per_span: 115, ..Dbp::new(m, 80.0, 1, 1.0) };
        assert_eq!(dbp.step_km(), forward.step_km);
        let z = dbp.run(&y, 64e9).unwrap();
        assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-12));
        // another number of steps: the span divided by the steps
        assert_eq!(Dbp { steps_per_span: 4, ..dbp }.step_km(), 20.0);
    }

    #[test]
    fn manakov_dbp_undoes_the_nonlinearity_of_two_polarisations() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.04e-23, gamma_w_km: 1.3 };
        let n = 1024;
        let w = |k: usize, f: f64, ph: f64| (2.0 * std::f64::consts::PI * f * k as f64 / n as f64 + ph).sin();
        let x0: Vec<C64> = (0..n).map(|k| C64::new(w(k, 31.0, 0.2), w(k, 17.0, 1.0)) * 0.1).collect();
        let y0: Vec<C64> = (0..n).map(|k| C64::new(w(k, 23.0, 2.0), w(k, 41.0, 0.5)) * 0.1).collect();
        let (mut x, mut y) = (x0.clone(), y0.clone());
        let fwd = ManakovSplitStep { rule: StepRule::Fixed(0.1), max_iterations: 20, tolerance: 1e-12, medium: m };
        let fs = 128e9;
        let g = (m.alpha_np_km / 2.0 * 80.0).exp();
        for _ in 0..2 {
            fwd.run(&mut x, &mut y, 80.0, fs).unwrap();
            for v in x.iter_mut().chain(y.iter_mut()) {
                *v *= g;
            }
        }
        let err = |xi: f64, steps: usize| {
            let (mut u, mut v) = (x.clone(), y.clone());
            Dbp { steps_per_span: steps, xi, ..Dbp::new(m, 80.0, 2, g) }.run_dual(&mut u, &mut v, fs).unwrap();
            let e: f64 = u.iter().zip(&x0).chain(v.iter().zip(&y0)).map(|(a, b)| (a - b).norm_sqr()).sum();
            let p: f64 = x0.iter().chain(&y0).map(C64::norm_sqr).sum();
            (e / p).sqrt()
        };
        let (linear, full) = (err(0.0, 800), err(1.0, 800));
        assert!(full < 1e-3 && full < linear / 50.0, "{full} vs {linear}");
    }
}

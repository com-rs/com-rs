//! Dual-polarisation nonlinear propagation over a multi-span link: the Manakov equation
//! solved by the split-step method of OptiCommPy's `manakovSSF`
//! ([`com_math::nlse::ManakovSplitStep`]), with an amplifier after every span.
//!
//! Every span of length $L_{\mathrm{span}}$ is followed by an EDFA of gain $\alpha
//! L_{\mathrm{span}}$ that adds ASE noise of the same variance to both polarisations
//! (see [`super::edfa::Edfa`]), a noiseless gain, or nothing. With the adaptive step of the
//! nonlinear phase rotation, the steps are short at the beginning of the spans and
//! long at their end, where the power has decayed.
//!
//! # Step size in WDM
//!
//! The nonlinear phase of a step is the trapezoidal mean of the power at the two ends of
//! the step. When channels walk off from each other within a step, the intensity pattern
//! that drives the cross-phase modulation moves between the two ends and the mean
//! smooths it, so the cross-channel interference comes out too small: five channels of
//! 32 GBd spaced by 50 GHz over 500 km of lossless SMF give 0.7 dB too little NLI with
//! steps of 0.5 km, 0.04 dB with 0.1 km, against the scalar split-step method of
//! [`super::fiber::Ssfm`] (the power at the middle of the step), which is converged at
//! 0.5 km (`validation/tests/dar2014.rs`). The step must keep the walk-off of the outer
//! channels small against the symbol period; the adaptive rule of the nonlinear phase
//! lengthens the steps at low power, where this matters most.
//!
//! [`ManakovSsfm::dbp`] is its digital back-propagation. [`ManakovSsfm`] neglects
//! polarisation-mode dispersion; `pmd::ManakovPmd` adds it with a
//! coarser scheme.
//!
//! # Examples
//! ```
//! use com_channel::optical::fiber::{Amplification, FiberParams};
//! use com_channel::optical::manakov::ManakovSsfm;
//! use com_core::signal::DualPol;
//! use com_math::C64;
//! let fiber = FiberParams { n_spans: 2, ..FiberParams::default() };
//! let link = ManakovSsfm { amplification: Amplification::Ideal, ..ManakovSsfm::new(fiber) };
//! let x: Vec<C64> = (0..256).map(|k| C64::new((k as f64 * 0.1).cos(), 0.0) * 0.03).collect();
//! let mut u = DualPol::new(x.clone(), x)?;
//! let e0 = u.x.iter().chain(&u.y).map(|v| v.norm_sqr()).sum::<f64>();
//! link.propagate(&mut u, 64e9, None)?;
//! let e1 = u.x.iter().chain(&u.y).map(|v| v.norm_sqr()).sum::<f64>();
//! assert!((e1 / e0 - 1.0).abs() < 1e-9);   // lossless with ideal amplifiers
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. D. Marcuse, C. R. Menyuk and P. K. A. Wai, "Application of the Manakov-PMD
//!    equation to studies of signal propagation in optical fibers with randomly varying
//!    birefringence," *J. Lightw. Technol.*, 15(9), 1735–1746, 1997.
//! 2. P. Serena, C. Lasagni, S. Musetti and A. Bononi, "On numerical simulations of
//!    ultra-wideband long-haul optical communication systems," *J. Lightw. Technol.*,
//!    38(5), 1019–1031, 2020.

use super::fiber::{Amplification, FiberParams, amplified_spans};
use super::raman::Raman;
use com_core::ensure;
use com_core::prelude::*;
use com_core::signal::DualPol;
use com_dsp::compensation::dbp::Dbp;
use com_math::nlse::{ManakovSplitStep, Medium, StepRule};

/// A dual-polarisation multi-span link (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct ManakovSsfm {
    /// The link.
    pub fiber: FiberParams,
    /// The step size.
    pub rule: StepRule,
    /// Largest number of trapezoidal iterations per step.
    pub max_iterations: usize,
    /// Convergence tolerance of the iterations.
    pub tolerance: f64,
    /// The amplifier after every span.
    pub amplification: Amplification,
    /// Distributed Raman amplification and ISRS in every span (see [`super::raman`]):
    /// the amplifier after the span then restores every frequency to its launch power.
    pub raman: Option<Raman>,
}

impl ManakovSsfm {
    /// The link with OptiCommPy's defaults: steps of 0.02 rad of nonlinear phase, 10
    /// iterations at most, tolerance $10^{-5}$, EDFAs.
    pub fn new(fiber: FiberParams) -> ManakovSsfm {
        ManakovSsfm {
            fiber,
            rule: StepRule::NonlinearPhase(2e-2),
            max_iterations: 10,
            tolerance: 1e-5,
            amplification: Amplification::Edfa,
            raman: None,
        }
    }

    /// Propagate both polarisations (√W) sampled at `fs`; the ASE of the EDFAs is drawn
    /// from `rng` (noiseless amplifiers without one). Returns the number of steps.
    ///
    /// # Errors
    /// [`Error::Config`] for a non-positive fixed step or rotation.
    pub fn propagate(&self, u: &mut DualPol, fs: f64, rng: Option<&mut Rng>) -> Result<usize> {
        match self.rule {
            StepRule::Fixed(h) | StepRule::NonlinearPhase(h) => {
                ensure(h > 0.0, || format!("the step must be positive, got {h}"))?;
            }
            _ => {}
        }
        let f = &self.fiber;
        let solver = ManakovSplitStep {
            medium: f.medium(),
            rule: self.rule,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
        };
        let solver = match self.raman {
            // the loss is in the gain profile of the span
            Some(_) => ManakovSplitStep { medium: Medium { alpha_np_km: 0.0, ..solver.medium }, ..solver },
            None => solver,
        };
        let DualPol { x, y } = u;
        amplified_spans(
            f,
            f.span_km,
            &self.amplification,
            self.raman.as_ref(),
            &mut [x, y],
            fs,
            rng,
            |_, pol, raman| {
                let (x, y) = pol.split_at_mut(1);
                match raman {
                    Some(span) => {
                        solver.run_with_gain(x[0], y[0], f.span_km, fs, &|z0, z1, out| span.field_gain(z0, z1, out))
                    }
                    None => solver.run(x[0], y[0], f.span_km, fs),
                }
            },
        )
    }

    /// The dual-polarisation digital back-propagation of this link with
    /// `steps_per_span` steps per span and the nonlinear scale `xi` (see
    /// `com_dsp::compensation::dbp`, [`Dbp::run_dual`]): the gain of the amplifiers is removed before each
    /// span. It back-propagates the power profile of the lumped amplifiers:
    /// [`ManakovSsfm::raman`] is ignored.
    pub fn dbp(&self, steps_per_span: usize, xi: f64) -> Dbp {
        let f = &self.fiber;
        Dbp {
            steps_per_span,
            xi,
            ..Dbp::new(f.medium(), f.span_km, f.n_spans, self.amplification.field_gain(f, f.span_km))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical::fiber::tests::signal;
    use crate::optical::raman::{Direction, RamanPump};

    /// The span loop of `ManakovSsfm::propagate` as it was written before the span driver.
    fn former(link: &ManakovSsfm, u: &mut DualPol, fs: f64, mut rng: Option<&mut Rng>) -> usize {
        let f = &link.fiber;
        let solver = ManakovSplitStep {
            medium: f.medium(),
            rule: link.rule,
            max_iterations: link.max_iterations,
            tolerance: link.tolerance,
        };
        let mut steps = 0;
        if let Some(raman) = &link.raman {
            let solver = ManakovSplitStep { medium: Medium { alpha_np_km: 0.0, ..solver.medium }, ..solver };
            for _ in 0..f.n_spans {
                let span = raman.field_span(f, &[&u.x, &u.y], fs).unwrap();
                steps += solver
                    .run_with_gain(&mut u.x, &mut u.y, f.span_km, fs, &|z0, z1, out| span.field_gain(z0, z1, out))
                    .unwrap();
                span.amplify(&mut u.x, fs, &link.amplification, f.nf_db, rng.as_deref_mut());
                span.amplify(&mut u.y, fs, &link.amplification, f.nf_db, rng.as_deref_mut());
            }
            return steps;
        }
        for _ in 0..f.n_spans {
            steps += solver.run(&mut u.x, &mut u.y, f.span_km, fs).unwrap();
            link.amplification.apply(&mut u.x, f, f.span_km, fs, rng.as_deref_mut()).unwrap();
            link.amplification.apply(&mut u.y, f, f.span_km, fs, rng.as_deref_mut()).unwrap();
        }
        steps
    }

    /// The span driver gives the fields and the step counts of the former loop bit for
    /// bit, with lumped and Raman amplification, with and without ASE.
    #[test]
    fn the_span_driver_is_the_former_span_loop() {
        let fiber = FiberParams { n_spans: 2, span_km: 30.0, ..FiberParams::default() };
        let u0 = DualPol::new(signal(256, 1, 3e-3), signal(256, 2, 3e-3)).unwrap();
        let pump = RamanPump::at_wavelength(1450.0, 0.3, Direction::Counter);
        let raman = Raman { pumps: vec![pump], step_km: 1.0, ..Raman::default() };
        for (amplification, raman) in [
            (Amplification::Edfa, None),
            (Amplification::Ideal, None),
            (Amplification::None, None),
            (Amplification::Edfa, Some(raman)),
        ] {
            let link = ManakovSsfm { amplification, raman, ..ManakovSsfm::new(fiber.clone()) };
            for seed in [None, Some(8)] {
                let (mut a, mut b) = (u0.clone(), u0.clone());
                let (mut ra, mut rb) = (seed.map(Rng::new), seed.map(Rng::new));
                let steps = link.propagate(&mut a, 64e9, ra.as_mut()).unwrap();
                assert_eq!(steps, former(&link, &mut b, 64e9, rb.as_mut()));
                assert_eq!(a, b, "{:?}", link.amplification);
            }
        }
    }
}

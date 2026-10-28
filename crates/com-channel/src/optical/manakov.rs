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
//! [`ManakovSsfm::dbp`] is its digital back-propagation. [`ManakovSsfm`] neglects
//! polarisation-mode dispersion; `pmd::Manakov` adds it with a
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

use super::fiber::{Amplification, FiberParams};
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
    pub fn propagate(&self, u: &mut DualPol, fs: f64, mut rng: Option<&mut Rng>) -> Result<usize> {
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
        if let Some(raman) = &self.raman {
            let solver = ManakovSplitStep { medium: Medium { alpha_np_km: 0.0, ..solver.medium }, ..solver };
            let mut steps = 0;
            for _ in 0..f.n_spans {
                let span = raman.field_span(f, &[&u.x, &u.y], fs)?;
                steps += solver
                    .run_with_gain(&mut u.x, &mut u.y, f.span_km, fs, &|z0, z1, out| span.field_gain(z0, z1, out));
                span.amplify(&mut u.x, fs, &self.amplification, f.nf_db, rng.as_deref_mut());
                span.amplify(&mut u.y, fs, &self.amplification, f.nf_db, rng.as_deref_mut());
            }
            return Ok(steps);
        }
        let mut steps = 0;
        for _ in 0..f.n_spans {
            steps += solver.run(&mut u.x, &mut u.y, f.span_km, fs);
            self.amplification.apply(&mut u.x, f, f.span_km, fs, rng.as_deref_mut())?;
            self.amplification.apply(&mut u.y, f, f.span_km, fs, rng.as_deref_mut())?;
        }
        Ok(steps)
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

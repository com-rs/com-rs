//! Dual-polarisation propagation: Jones matrices, polarisation-mode dispersion and the
//! Manakov-PMD equation.
//!
//! - [`Jones`]: `2×2` complex matrices, uniform SU(2) rotations, differential group delay;
//! - [`PmdLink`]: the coarse-step PMD of a link (random birefringent sections);
//! - [`Manakov`]: two polarisations with PMD by split steps, with its exact inverse and
//!   the lumped PMD equaliser (the upper bounds of a receiver that knows the PMD).
//!
//! # References
//!
//! 1. D. Marcuse, C. R. Menyuk and P. K. A. Wai, "Application of the Manakov-PMD
//!    equation to studies of signal propagation in optical fibers with randomly
//!    varying birefringence," *J. Lightw. Technol.*, 15(9), 1997.

use super::fiber::{Amplification, FiberParams};
use com_core::ensure;
use com_core::ensure_shape;
use com_core::prelude::*;
use com_math::fft;
use com_math::nlse::manakov_rotation;
use com_math::{cis, effective_length};

/// A `2×2` Jones matrix `[[a, b], [c, d]]` (row-major).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jones(pub [C64; 4]);

impl Jones {
    /// The identity.
    pub const IDENTITY: Jones = Jones([C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)]);

    /// `J (x, y)`.
    #[inline]
    pub fn apply(&self, x: C64, y: C64) -> (C64, C64) {
        let m = &self.0;
        (m[0] * x + m[1] * y, m[2] * x + m[3] * y)
    }

    /// The product `self · other`.
    pub fn mul(&self, o: &Jones) -> Jones {
        let (a, b) = (&self.0, &o.0);
        Jones([
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        ])
    }

    /// The conjugate transpose (the inverse of a unitary matrix).
    pub fn adjoint(&self) -> Jones {
        let m = &self.0;
        Jones([m[0].conj(), m[2].conj(), m[1].conj(), m[3].conj()])
    }

    /// A rotation drawn uniformly on SU(2) (a uniform unit quaternion).
    pub fn random_su2(rng: &mut Rng) -> Jones {
        let q = [rng.gaussian(), rng.gaussian(), rng.gaussian(), rng.gaussian()];
        let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (a, b, c, d) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
        Jones([C64::new(a, b), C64::new(c, d), C64::new(-c, d), C64::new(a, -b)])
    }

    /// The real rotation by `theta` (rad), `[[cos θ, -sin θ], [sin θ, cos θ]]`.
    pub fn rotation(theta: f64) -> Jones {
        let (s, c) = theta.sin_cos();
        Jones([C64::new(c, 0.0), C64::new(-s, 0.0), C64::new(s, 0.0), C64::new(c, 0.0)])
    }

    /// The differential group delay `diag(e^{-jωτ/2}, e^{+jωτ/2})` at the angular frequency `ω`.
    #[inline]
    pub fn dgd(omega: f64, tau_s: f64) -> Jones {
        let p = cis(-0.5 * omega * tau_s);
        Jones([p, C64::new(0.0, 0.0), C64::new(0.0, 0.0), p.conj()])
    }
}

/// One birefringent section: a rotation then a differential group delay.
#[derive(Clone, Debug, PartialEq)]
pub struct PmdSection {
    /// Rotation onto the principal states of the section.
    pub rotation: Jones,
    /// Differential group delay (s).
    pub dgd_s: f64,
}

/// The polarisation-mode dispersion of a link, as a concatenation of sections.
///
/// [`PmdLink::random`] draws, for a PMD parameter `τ` (ps/√km), sections of length
/// `L_c` with a uniform SU(2) rotation and a DGD `τ_k ~ N(τ̄, (τ̄/5)²)`,
/// `τ̄ = τ √(3π/8 · L_c)`, so that the mean DGD of the link is `τ √L` (the coarse-step
/// model of Marcuse, Menyuk and Wai, JLT 15(9), 1997).
#[derive(Clone, Debug, PartialEq)]
pub struct PmdLink {
    /// The sections, span after span.
    pub sections: Vec<PmdSection>,
    /// Sections per span.
    pub sections_per_span: usize,
    /// Length of a section (km).
    pub section_km: f64,
}

impl PmdLink {
    /// No PMD (identity sections), `sections_per_span` steps per span.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for zero sections per span.
    pub fn none(fiber: &FiberParams, sections_per_span: usize) -> Result<PmdLink> {
        ensure(sections_per_span >= 1, || "a PMD link needs at least one section per span".into())?;
        let n = fiber.n_spans * sections_per_span;
        Ok(PmdLink {
            sections: vec![PmdSection { rotation: Jones::IDENTITY, dgd_s: 0.0 }; n],
            sections_per_span,
            section_km: fiber.span_km / sections_per_span as f64,
        })
    }

    /// Random PMD of parameter `pmd_ps_sqrt_km` with `sections_per_span` sections per span.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for zero sections per span or a negative parameter.
    pub fn random(
        fiber: &FiberParams,
        pmd_ps_sqrt_km: f64,
        sections_per_span: usize,
        rng: &mut Rng,
    ) -> Result<PmdLink> {
        ensure(sections_per_span >= 1, || "a PMD link needs at least one section per span".into())?;
        ensure(pmd_ps_sqrt_km >= 0.0, || format!("the PMD parameter must be non-negative, got {pmd_ps_sqrt_km}"))?;
        let sps = sections_per_span;
        let lc = fiber.span_km / sps as f64;
        let tau_bar = pmd_ps_sqrt_km * 1e-12 * (3.0 * std::f64::consts::PI / 8.0 * lc).sqrt();
        let sections = (0..fiber.n_spans * sps)
            .map(|_| PmdSection { rotation: Jones::random_su2(rng), dgd_s: tau_bar * (1.0 + 0.2 * rng.gaussian()) })
            .collect();
        Ok(PmdLink { sections, sections_per_span: sps, section_km: lc })
    }

    /// `sections` sections of unit length with a uniform SU(2) rotation and a DGD
    /// `τ_k ~ N(τ̄, (τ̄/5)²)`, `τ̄ = ⟨DGD⟩ √(3π/(8N))`, so that the mean DGD of the
    /// concatenation is `mean_dgd_s` (the model of [`PmdLink::random`] for a given mean
    /// DGD rather than a PMD parameter).
    pub fn with_mean_dgd(mean_dgd_s: f64, sections: usize, rng: &mut Rng) -> PmdLink {
        let n = sections.max(1);
        let tau = mean_dgd_s * (3.0 * std::f64::consts::PI / (8.0 * n as f64)).sqrt();
        PmdLink {
            sections: (0..n)
                .map(|_| PmdSection { rotation: Jones::random_su2(rng), dgd_s: tau * (1.0 + 0.2 * rng.gaussian()) })
                .collect(),
            sections_per_span: n,
            section_km: 1.0,
        }
    }

    /// Apply the lumped response `J(ω)` to the two polarisations sampled at `fs`
    /// (periodic blocks, in the frequency domain).
    pub fn apply(&self, x: &mut Vec<C64>, y: &mut Vec<C64>, fs: f64) {
        let omega = fft::angular_frequencies(x.len(), fs);
        let j = self.lumped_response(&omega);
        let (mut fx, mut fy) = (fft::fft(x), fft::fft(y));
        for ((u, v), m) in fx.iter_mut().zip(fy.iter_mut()).zip(&j) {
            (*u, *v) = m.apply(*u, *v);
        }
        *x = fft::ifft(&fx);
        *y = fft::ifft(&fy);
    }

    /// RMS DGD of the concatenation `√Σ τ_k²` (s).
    pub fn rms_dgd_s(&self) -> f64 {
        self.sections.iter().map(|s| s.dgd_s * s.dgd_s).sum::<f64>().sqrt()
    }

    /// The lumped Jones matrix `J(ω) = Π_k D_k(ω) R_k` at every angular frequency.
    pub fn lumped_response(&self, omega: &[f64]) -> Vec<Jones> {
        omega
            .iter()
            .map(|&w| {
                self.sections.iter().fold(Jones::IDENTITY, |j, s| Jones::dgd(w, s.dgd_s).mul(&s.rotation).mul(&j))
            })
            .collect()
    }
}

/// Dual-polarisation propagation: the Manakov-PMD equation by split steps.
///
/// `∂u/∂z = (-α/2 - β₁(z) ∂/∂t - j β₂/2 ∂²/∂t²) u + j (8/9) γ ‖u‖² u`, `u = (u_x, u_y)`.
///
/// Each PMD section is one asymmetric split step: in the frequency domain the scalar
/// loss and dispersion and the section `D_k(ω) R_k`, then the nonlinear rotation
/// `u ← u e^{j(8/9)γ L_eff ‖u‖²}` with `L_eff = (1 - e^{-αL_c})/α`; after every span the
/// field is amplified (the EDFA of the fibre parameters, with ASE when a generator is
/// given). [`Manakov::backpropagate_with_pmd`] is the exact inverse (the upper bound of
/// a receiver that knows the PMD), [`Manakov::equalize_pmd`] the lumped ideal MIMO
/// equaliser `J(ω)⁻¹`.
///
/// The solvers of `com_math::nlse` do not apply here: their linear step is a scalar
/// response, the same for both polarisations and for every step, while the
/// birefringence of a section is a `2×2` Jones matrix per frequency that changes from
/// section to section; and they are symmetric (half linear steps around the nonlinear
/// one), where this coarse-step model takes one whole section then its nonlinear
/// rotation, so that a section is one realisation of the PMD. The nonlinear rotation
/// itself is [`com_math::nlse::manakov_rotation`], the step of
/// `com_math::nlse::SplitStep::run_dual`.
#[derive(Clone, Debug)]
pub struct Manakov {
    /// The link.
    pub fiber: FiberParams,
    /// Its polarisation-mode dispersion (one split step per section).
    pub pmd: PmdLink,
}

impl Manakov {
    fn linear(&self, u: &mut DualPol, omega: &[f64], s: &PmdSection, inverse: bool) {
        let f = &self.fiber;
        let dz = self.pmd.section_km;
        let sign = if inverse { -1.0 } else { 1.0 };
        let g = (-sign * f.alpha_np_km() / 2.0 * dz).exp();
        let c = sign * 0.5 * f.beta2_s2_km() * dz;
        let r = if inverse { s.rotation.adjoint() } else { s.rotation };
        for (i, &w) in omega.iter().enumerate() {
            let h = cis(c * w * w) * g;
            let (mut a, mut b) = (u.x[i] * h, u.y[i] * h);
            if inverse {
                let (p, q) = Jones::dgd(w, -s.dgd_s).apply(a, b);
                (a, b) = r.apply(p, q);
            } else {
                let (p, q) = r.apply(a, b);
                (a, b) = Jones::dgd(w, s.dgd_s).apply(p, q);
            }
            u.x[i] = a;
            u.y[i] = b;
        }
    }

    fn nonlinear(&self, u: &mut DualPol, sign: f64) {
        let f = &self.fiber;
        let a = f.alpha_np_km();
        let dz = self.pmd.section_km;
        let l_eff = effective_length(a, dz);
        manakov_rotation(&mut u.x, &mut u.y, sign * 8.0 / 9.0 * f.gamma_w_km * l_eff);
    }

    /// Propagate a dual-polarisation field sampled at `fs` (ASE from `rng` if given).
    ///
    /// # Errors
    /// [`Error::Shape`] if the PMD link does not match the number of spans.
    pub fn propagate(&self, u: &mut DualPol, fs: f64, mut rng: Option<&mut Rng>) -> Result<()> {
        ensure_shape(self.pmd.sections.len() == self.fiber.n_spans * self.pmd.sections_per_span, || {
            format!(
                "the PMD link has {} sections for {} spans of {} sections",
                self.pmd.sections.len(),
                self.fiber.n_spans,
                self.pmd.sections_per_span
            )
        })?;
        let n = u.len();
        let p = fft::plan(n);
        let omega = fft::angular_frequencies(n, fs);
        let span_km = self.fiber.span_km;
        for (k, s) in self.pmd.sections.iter().enumerate() {
            p.forward(&mut u.x);
            p.forward(&mut u.y);
            self.linear(u, &omega, s, false);
            p.inverse(&mut u.x);
            p.inverse(&mut u.y);
            self.nonlinear(u, 1.0);
            if (k + 1) % self.pmd.sections_per_span == 0 {
                Amplification::Edfa.apply(&mut u.x, &self.fiber, span_km, fs, rng.as_deref_mut())?;
                Amplification::Edfa.apply(&mut u.y, &self.fiber, span_km, fs, rng.as_deref_mut())?;
            }
        }
        Ok(())
    }

    /// The exact inverse of [`Manakov::propagate`] without noise.
    pub fn backpropagate_with_pmd(&self, u: &mut DualPol, fs: f64) {
        let n = u.len();
        let p = fft::plan(n);
        let omega = fft::angular_frequencies(n, fs);
        let gain = 1.0 / self.fiber.span_gain().sqrt();
        for (i, s) in self.pmd.sections.iter().enumerate().rev() {
            if (i + 1) % self.pmd.sections_per_span == 0 {
                u.scale(gain);
            }
            self.nonlinear(u, -1.0);
            p.forward(&mut u.x);
            p.forward(&mut u.y);
            self.linear(u, &omega, s, true);
            p.inverse(&mut u.x);
            p.inverse(&mut u.y);
        }
    }

    /// The lumped ideal PMD equaliser `J(ω)⁻¹ = J(ω)ᴴ` applied in the frequency domain.
    pub fn equalize_pmd(&self, u: &mut DualPol, fs: f64) {
        let n = u.len();
        let p = fft::plan(n);
        let omega = fft::angular_frequencies(n, fs);
        let j = self.pmd.lumped_response(&omega);
        p.forward(&mut u.x);
        p.forward(&mut u.y);
        for i in 0..n {
            let (a, b) = j[i].adjoint().apply(u.x[i], u.y[i]);
            u.x[i] = a;
            u.y[i] = b;
        }
        p.inverse(&mut u.x);
        p.inverse(&mut u.y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optical::fiber::linear_fiber_channel;
    use crate::optical::fiber::tests::signal;

    #[test]
    fn manakov_round_trip_and_pmd_equaliser() {
        let fiber = FiberParams { n_spans: 2, span_km: 20.0, ..FiberParams::default() };
        let mut r = Rng::new(4);
        let pmd = PmdLink::random(&fiber, 0.5, 10, &mut r).unwrap();
        let m = Manakov { fiber: fiber.clone(), pmd };
        let u0 = DualPol::new(signal(128, 2, 1e-3), signal(128, 3, 1e-3)).unwrap();
        let mut u = u0.clone();
        m.propagate(&mut u, 64e9, None).unwrap();
        m.backpropagate_with_pmd(&mut u, 64e9);
        assert!(u.x.iter().zip(&u0.x).all(|(a, b)| (a - b).norm() < 1e-9));
        // linear lossless link: dispersion compensation + J⁻¹ is exact
        let lin = FiberParams { gamma_w_km: 0.0, alpha_db_km: 0.0, ..fiber };
        let m = Manakov { fiber: lin.clone(), pmd: PmdLink::random(&lin, 0.5, 10, &mut r).unwrap() };
        let mut u = u0.clone();
        m.propagate(&mut u, 64e9, None).unwrap();
        let mut v = DualPol::new(
            linear_fiber_channel(
                &u.x,
                &FiberParams { d_ps_nm_km: -lin.d_ps_nm_km, ..lin.clone() },
                lin.total_km(),
                64e9,
            ),
            linear_fiber_channel(
                &u.y,
                &FiberParams { d_ps_nm_km: -lin.d_ps_nm_km, ..lin.clone() },
                lin.total_km(),
                64e9,
            ),
        )
        .unwrap();
        m.equalize_pmd(&mut v, 64e9);
        assert!(v.x.iter().zip(&u0.x).all(|(a, b)| (a - b).norm() < 1e-9));
    }

    #[test]
    fn a_pmd_realisation_is_unitary_and_rotations_invert() {
        let mut rng = Rng::new(4);
        let x0: Vec<C64> = (0..256).map(|_| rng.complex_gaussian()).collect();
        let y0: Vec<C64> = (0..256).map(|_| rng.complex_gaussian()).collect();
        let (mut x, mut y) = (x0.clone(), y0.clone());
        let link = PmdLink::with_mean_dgd(20e-12, 30, &mut rng);
        link.apply(&mut x, &mut y, 64e9);
        let e = |a: &[C64], b: &[C64]| a.iter().chain(b).map(C64::norm_sqr).sum::<f64>();
        assert!((e(&x, &y) / e(&x0, &y0) - 1.0).abs() < 1e-12);
        let (r, back) = (Jones::rotation(0.7), Jones::rotation(-0.7));
        let (u, v) = r.apply(x0[0], y0[0]);
        let (u, v) = back.apply(u, v);
        assert!((u - x0[0]).norm() < 1e-12 && (v - y0[0]).norm() < 1e-12);
    }
}

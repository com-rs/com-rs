//! The symmetric split-step Fourier method with a constant step (`com_math::nlse::SplitStep`,
//! the solver of `com_channel::optical::fiber::Ssfm`) against O. V. Sinkin, R. Holzlöhner,
//! J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling
//! optical-fiber communications systems," *J. Lightw. Technol.* 21(1), 61–68, 2003: the
//! collision of two first-order solitons of their Sec. III-B, propagated over 400 km of
//! lossless fibre ($\beta'' = -0.1$ ps²/km, $\gamma = 2.2$ W⁻¹km⁻¹, $\eta = 0.44$ ps⁻¹,
//! pulses 100 ps apart and 800 GHz apart in frequency, 3072 Fourier modes over 400 ps).
//! The test measures the global relative error of their eq. (12),
//! $\varepsilon = \lVert u_{n} - u_{a} \rVert / \lVert u_{a} \rVert$, against a solution
//! $u_{a}$ with steps of 1 m, as a function of the number of FFTs; the points of the
//! constant-step curve of their Fig. 2, the tolerances and the differences found are in
//! the card `validation/cards/sinkin2003_fig2.yaml`. The local-error, nonlinear-phase,
//! walk-off and logarithmic step rules of their Sec. II are not compared.
//!
//! The same collision has an exact solution, the two-soliton solution of the nonlinear
//! Schrödinger equation $j u_{\xi} + \frac{1}{2} u_{\tau\tau} + |u|^{2} u = 0$ (R. Hirota,
//! "Exact envelope-soliton solutions of a nonlinear wave equation," *J. Math. Phys.*,
//! 1973), with $\xi = z / L_{D}$, $\tau = \eta t$, $L_{D} = 1/(\eta^{2} |\beta''|)$:
//!
//! $$
//! u = \frac{e^{\theta_{1}} + e^{\theta_{2}} + a_{1} e^{\theta_{1} + \theta_{1}^{\ast} + \theta_{2}} +
//!   a_{2} e^{\theta_{1} + \theta_{2} + \theta_{2}^{\ast}}}{1 + \sum_{i, j} b_{ij}
//!   e^{\theta_{i} + \theta_{j}^{\ast}} + c\, e^{\theta_{1} + \theta_{1}^{\ast} + \theta_{2} + \theta_{2}^{\ast}}},
//! \qquad \theta_{i} = k_{i} \tau + \tfrac{j}{2} k_{i}^{2} \xi + \theta_{i}^{0},
//! $$
//!
//! with $b_{ij} = (k_{i} + k_{j}^{\ast})^{-2}$, $a_{1} = (k_{1} - k_{2})^{2} b_{11} b_{21}$,
//! $a_{2} = (k_{1} - k_{2})^{2} b_{12} b_{22}$ and $c = |k_{1} - k_{2}|^{4} b_{11} b_{22} b_{12}
//! b_{21}$; here $k_{1, 2} = 1 \pm j\, 2\pi \cdot 400\ \text{GHz} / \eta$ and $\theta_{i}^{0} =
//! \ln 2 - \tau_{i}$ put the pulses at $\tau_{i} = \mp 50$ ps $\cdot\, \eta$. The formula is
//! checked twice: it satisfies the equation to the truncation of fourth-order finite
//! differences, and the split step with steps of 1 m reaches it within $10^{-8}$ after
//! 400 km. Both solvers are then measured against it (card
//! `validation/cards/sinkin2003_exact.yaml`): the scalar split step of eq. (3) and the
//! Manakov solver (`ManakovSplitStep`, the trapezoidal phase of OptiCommPy, and the phase
//! at the middle of the step) with the second polarisation empty and $\gamma$ scaled by
//! 9/8.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::C64;
use comrs::math::nlse::{ManakovSplitStep, Medium, NonlinearStep, SplitStep, StepRule, Steps};
use comrs::metrics::evm_data_aided;
use serde_json::json;

/// The fibre of Sec. III-A: lossless, $\beta'' = -0.1$ ps²/km, $\gamma = 2.2$ W⁻¹km⁻¹.
const FIBER: Medium = Medium { alpha_np_km: 0.0, beta2_s2_km: -0.1e-24, gamma_w_km: 2.2 };
const LENGTH_KM: f64 = 400.0;
const MODES: usize = 3072;
const WINDOW_S: f64 = 400e-12;

/// Two first-order solitons $\eta \sqrt{|\beta''|/\gamma}\, \mathrm{sech}(\eta t)$ (√W) at
/// ∓50 ps, at ±400 GHz: with the convention of the solver, a positive frequency offset
/// moves later in time, so the pulses meet at 200 km.
fn launch() -> Vec<C64> {
    let eta = 0.44e12;
    let amplitude = eta * (FIBER.beta2_s2_km.abs() / FIBER.gamma_w_km).sqrt();
    let dt = WINDOW_S / MODES as f64;
    (0..MODES)
        .map(|k| {
            let t = (k as f64 - MODES as f64 / 2.0) * dt;
            [(-50e-12, 400e9), (50e-12, -400e9)]
                .iter()
                .map(|&(t0, f0): &(f64, f64)| {
                    let s = 1.0 / (eta * (t - t0)).cosh();
                    C64::from_polar(amplitude * s, 2.0 * std::f64::consts::PI * f0 * t)
                })
                .sum()
        })
        .collect()
}

/// The field after 400 km with `n_steps` constant steps, and the number of FFTs.
fn propagate(n_steps: usize) -> (Vec<C64>, usize) {
    let fs = MODES as f64 / WINDOW_S;
    let h = LENGTH_KM / n_steps as f64;
    let mut e = launch();
    SplitStep::new(&FIBER, Steps::new(n_steps, h), MODES, fs).run(&mut e);
    // a forward and an inverse transform per step, and the two at the ends
    (e, 2 * n_steps + 2)
}

/// The global relative error of eq. (12).
fn global_error(u: &[C64], reference: &[C64]) -> f64 {
    evm_data_aided(u, reference).unwrap().sqrt()
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn sinkin_2003_fig_2_constant_steps() {
    let mut card = Card::load("sinkin2003_fig2").unwrap();
    let t = std::time::Instant::now();
    let (reference, _) = propagate(400_000);
    println!("reference (steps of 1 m): {:.1} s", t.elapsed().as_secs_f64());
    // the pulses have collided and passed through each other: they end at ±50 ps (they
    // would be at ±150 ps had they moved apart)
    let inner: f64 = reference[MODES / 4..3 * MODES / 4].iter().map(C64::norm_sqr).sum();
    let total: f64 = reference.iter().map(C64::norm_sqr).sum();
    assert!(inner > 0.99 * total, "the pulses did not collide: {:.3} of the energy within ±100 ps", inner / total);
    let mut points = Vec::new();
    for ffts in [70_000usize, 30_000, 9_500, 3_200, 1_600, 750, 440, 320] {
        let n_steps = (ffts - 2) / 2;
        let (u, n_fft) = propagate(n_steps);
        let eps = global_error(&u, &reference);
        println!("{n_steps} steps of {:.1} m, {n_fft} FFTs: error {eps:.3e}", LENGTH_KM / n_steps as f64 * 1e3);
        card.check(&json!({"ffts": ffts}), eps);
        points.push((n_fft as f64, eps));
    }
    // the order of the scheme: the slope of log(error) against log(steps) where the
    // error is small (Sec. III-E: "approximately 2")
    let (a, b) = (points[0], points[3]);
    card.check(&json!({"claim": "order"}), -(b.1 / a.1).ln() / (b.0 / a.0).ln());
    card.finish().unwrap();
}

/// The inverse duration $\eta$ of the solitons (1/s).
const ETA: f64 = 0.44e12;

/// The exact two-soliton solution $u(\tau, \xi)$ of the module (normalised units).
fn two_solitons(tau: f64, xi: f64) -> C64 {
    let j = C64::new(0.0, 1.0);
    let b = 2.0 * std::f64::consts::PI * 400e9 / ETA;
    let (k1, k2) = (C64::new(1.0, b), C64::new(1.0, -b));
    let theta = |k: C64, tau_i: f64| (k * tau + j * k * k * xi / 2.0 + 2f64.ln() - tau_i).exp();
    let (e1, e2) = (theta(k1, -50e-12 * ETA), theta(k2, 50e-12 * ETA));
    let (c1, c2) = (e1.conj(), e2.conj());
    let bij = |x: C64, y: C64| 1.0 / ((x + y.conj()) * (x + y.conj()));
    let (b11, b12, b21, b22) = (bij(k1, k1), bij(k1, k2), bij(k2, k1), bij(k2, k2));
    let d = (k1 - k2) * (k1 - k2);
    let (a1, a2) = (d * b11 * b21, d * b12 * b22);
    let c = d.norm_sqr() * b11 * b22 * b12 * b21;
    let num = e1 + e2 + a1 * e1 * c1 * e2 + a2 * e1 * e2 * c2;
    let den = 1.0 + b11 * e1 * c1 + b12 * e1 * c2 + b21 * e2 * c1 + b22 * e2 * c2 + c * e1 * c1 * e2 * c2;
    num / den
}

/// The exact field (√W) at `z_km` on the grid of the simulation.
fn exact_field(z_km: f64) -> Vec<C64> {
    let ld = 1.0 / (ETA * ETA * FIBER.beta2_s2_km.abs());
    let p0 = ETA * ETA * FIBER.beta2_s2_km.abs() / FIBER.gamma_w_km;
    let dt = WINDOW_S / MODES as f64;
    (0..MODES).map(|k| two_solitons((k as f64 - MODES as f64 / 2.0) * dt * ETA, z_km / ld) * p0.sqrt()).collect()
}

#[test]
fn the_two_soliton_solution_satisfies_the_nlse() {
    // fourth-order central differences, near the collision (xi = 200 km / L_D) and before
    // it: the residual is their truncation, which falls as h^4
    let ld = 1.0 / (ETA * ETA * FIBER.beta2_s2_km.abs());
    let residual = |tau: f64, xi: f64, h: f64| {
        let u = |dt: f64, dx: f64| two_solitons(tau + dt, xi + dx);
        let u_xi = (-u(0.0, 2.0 * h) + 8.0 * u(0.0, h) - 8.0 * u(0.0, -h) + u(0.0, -2.0 * h)) / (12.0 * h);
        let u_tt = (-u(2.0 * h, 0.0) + 16.0 * u(h, 0.0) - 30.0 * u(0.0, 0.0) + 16.0 * u(-h, 0.0) - u(-2.0 * h, 0.0))
            / (12.0 * h * h);
        let v = u(0.0, 0.0);
        ((C64::new(0.0, 1.0) * u_xi + u_tt / 2.0 + v * v.norm_sqr()).norm(), u_tt.norm())
    };
    for (tau, xi) in [(0.1, 200.0 / ld), (0.5, 200.0 / ld + 0.01), (-21.0, 0.5)] {
        let ((r1, scale), (r2, _)) = (residual(tau, xi, 4e-3), residual(tau, xi, 2e-3));
        println!("({tau}, {xi:.3}): residual {r1:.2e} then {r2:.2e}, |u_tt| {scale:.2e}");
        assert!(r2 < 1e-6 * scale, "residual {r2:.2e} at ({tau}, {xi}), |u_tt| {scale:.2e}");
        assert!(r1 / r2 > 12.0 || r2 < 1e-9 * scale, "the residual does not fall as h^4: {r1:.2e}, {r2:.2e}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn both_solvers_converge_to_the_exact_collision() {
    let fs = MODES as f64 / WINDOW_S;
    let (launch, target) = (exact_field(0.0), exact_field(LENGTH_KM));
    let scalar = |n_steps: usize| {
        let mut e = launch.clone();
        SplitStep::new(&FIBER, Steps::new(n_steps, LENGTH_KM / n_steps as f64), MODES, fs).run(&mut e);
        global_error(&e, &target)
    };
    let manakov = |n_steps: usize, nonlinear: NonlinearStep| {
        let solver = ManakovSplitStep {
            rule: StepRule::Fixed(LENGTH_KM / n_steps as f64),
            nonlinear,
            ..ManakovSplitStep::new(Medium { gamma_w_km: 9.0 / 8.0 * FIBER.gamma_w_km, ..FIBER })
        };
        let (mut x, mut y) = (launch.clone(), vec![C64::new(0.0, 0.0); MODES]);
        solver.run(&mut x, &mut y, LENGTH_KM, fs).unwrap();
        global_error(&x, &target)
    };
    let mut card = Card::load("sinkin2003_exact").unwrap();
    card.check(&json!({"solver": "scalar", "steps": 400_000}), scalar(400_000));
    let (s1, s2) = (scalar(1600), scalar(800));
    let (m1, m2) = (manakov(1600, NonlinearStep::Trapezoidal), manakov(800, NonlinearStep::Trapezoidal));
    let mid = manakov(1600, NonlinearStep::Midpoint);
    println!(
        "1600 steps: scalar {s1:.3e}, Manakov {m1:.3e} (midpoint {mid:.3e}); 800 steps: scalar {s2:.3e}, Manakov {m2:.3e}"
    );
    card.check(&json!({"solver": "manakov_midpoint", "claim": "error_over_scalar"}), mid / s1);
    card.check(&json!({"solver": "scalar", "claim": "order"}), (s2 / s1).log2());
    card.check(&json!({"solver": "manakov", "claim": "order"}), (m2 / m1).log2());
    card.check(&json!({"solver": "manakov", "claim": "error_over_scalar"}), m1 / s1);
    // the error against the exact solution is the error against the solution of 1 m
    // steps, the reference of the Fig. 2 test
    let mut reference = launch.clone();
    SplitStep::new(&FIBER, Steps::new(400_000, 1e-3), MODES, fs).run(&mut reference);
    let mut e = launch.clone();
    SplitStep::new(&FIBER, Steps::new(1600, LENGTH_KM / 1600.0), MODES, fs).run(&mut e);
    card.check(&json!({"claim": "exact_over_numerical_reference"}), s1 / global_error(&e, &reference));
    card.finish().unwrap();
}

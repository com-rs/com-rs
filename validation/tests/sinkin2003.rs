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
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::C64;
use comrs::math::nlse::{Medium, SplitStep, Steps};
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
    SplitStep::new(&FIBER, Steps::forward(n_steps, h), MODES, fs).run(&mut e);
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

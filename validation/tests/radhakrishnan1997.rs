//! The Manakov solver (`com_math::nlse::ManakovSplitStep`) against the exact bright
//! two-soliton solution of the Manakov system of R. Radhakrishnan, M. Lakshmanan and
//! J. Hietarinta, "Inelastic collision and switching of coupled bright solitons in
//! optical fibers," *Phys. Rev. E* 56(2), 2213–2216, 1997 (arXiv solv-int/9703008): the
//! only exact check of the coupling of the two polarisations, which a solver with one
//! polarisation empty does not see.
//!
//! The system is $j q_{i,x} + q_{i,tt} + 2\mu (|q_{1}|^{2} + |q_{2}|^{2}) q_{i} = 0$
//! (eq. (1)), here with $\mu = 1$; its two-soliton solution is eqs. (7)–(14), with
//! $\eta_{j} = k_{j} (t + j k_{j} x)$. The case is that of Fig. 1, $k_{1} = 1 + j$,
//! $k_{2} = 2 - j$, $\beta_{1} = \beta_{2} = \alpha_{1} = 1$, $\alpha_{2} = (39 + 80 j)/89$:
//! a head-on collision in which soliton 1 leaves almost all its energy in component 2,
//! with the transition amplitudes of eqs. (16) and (19).
//!
//! In physical units ($T_{0}$ = 1 ps, $\beta_{2} = -20$ ps²/km, $\gamma$ = 1.3 /(W km)),
//! the Manakov equation $\partial_{z} A = -j \frac{\beta_{2}}{2} \partial_{T}^{2} A + j
//! \frac{8}{9} \gamma \lVert A \rVert^{2} A$ is the system with $T = T_{0} t$, $z = L x$,
//! $L = 2 T_{0}^{2} / |\beta_{2}|$ (0.1 km) and $A = \sqrt{P_{0}}\, q$, $P_{0} = 2\mu / (\frac{8}{9}
//! \gamma L)$. The field is launched with the exact solution at $x = -3$ (the solitons 12
//! $T_{0}$ apart) and compared with it at $x = +3$, 0.6 km later, on 2048 samples over 60 ps.
//! The values and tolerances are in the card `validation/cards/radhakrishnan1997_fig1.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::C64;
use comrs::math::nlse::{ManakovSplitStep, Medium, NonlinearStep, StepRule};
use serde_json::json;

const T0: f64 = 1e-12;
const BETA2: f64 = -20e-27;
const GAMMA: f64 = 1.3;
const SAMPLES: usize = 2048;
const WINDOW: f64 = 60.0;
const X0: f64 = -3.0;
const X1: f64 = 3.0;

/// The parameters of Fig. 1: $k_{1}, k_{2}$ and $(\alpha_{j}, \beta_{j})$.
fn parameters() -> ([C64; 2], [C64; 2], [C64; 2]) {
    let k = [C64::new(1.0, 1.0), C64::new(2.0, -1.0)];
    let alpha = [C64::new(1.0, 0.0), C64::new(39.0, 80.0) / 89.0];
    let beta = [C64::new(1.0, 0.0), C64::new(1.0, 0.0)];
    (k, alpha, beta)
}

/// The two-soliton solution $(q_{1}, q_{2})$ of eqs. (7)–(14) with $\mu = 1$.
fn two_solitons(t: f64, x: f64) -> (C64, C64) {
    let (k, a, b) = parameters();
    let j = C64::new(0.0, 1.0);
    let e: Vec<C64> = k.iter().map(|&kk| (kk * (t + j * kk * x)).exp()).collect();
    let (e1, e2, c1, c2) = (e[0], e[1], e[0].conj(), e[1].conj());
    let kappa = |i: usize, m: usize| (a[i] * a[m].conj() + b[i] * b[m].conj()) / (k[i] + k[m].conj());
    let (k11, k12, k21, k22) = (kappa(0, 0), kappa(0, 1), kappa(1, 0), kappa(1, 1));
    let (k1, k2) = (k[0], k[1]);
    let d0 = k12 / (k1 + k2.conj());
    let r1 = k11 / (k1 + k1.conj());
    let r2 = k22 / (k2 + k2.conj());
    let pre1 = (k1 - k2) / ((k1 + k1.conj()) * (k1.conj() + k2));
    let pre2 = (k2 - k1) / ((k2 + k2.conj()) * (k1 + k2.conj()));
    let r3 = (k1 - k2).norm_sqr() / ((k1 + k1.conj()) * (k2 + k2.conj()) * (k1 + k2.conj()).norm_sqr())
        * (k11 * k22 - k12 * k21);
    let den = 1.0 + e1 * c1 * r1 + e1 * c2 * d0 + c1 * e2 * d0.conj() + e2 * c2 * r2 + e1 * c1 * e2 * c2 * r3;
    let component = |w: [C64; 2]| {
        let d1 = pre1 * (w[0] * k21 - w[1] * k11);
        let d2 = pre2 * (w[1] * k12 - w[0] * k22);
        (w[0] * e1 + w[1] * e2 + e1 * c1 * e2 * d1 + e1 * e2 * c2 * d2) / den
    };
    (component(a), component(b))
}

/// $L$ (km) and $P_{0}$ (W) of the module.
fn scales() -> (f64, f64) {
    let l = 2.0 * T0 * T0 / BETA2.abs();
    (l, 2.0 / (8.0 / 9.0 * GAMMA * l))
}

/// The exact field (√W) of both polarisations at the normalised distance `x`.
fn exact_field(x: f64) -> (Vec<C64>, Vec<C64>) {
    let p0 = scales().1;
    (0..SAMPLES)
        .map(|n| {
            let t = (n as f64 - SAMPLES as f64 / 2.0) * WINDOW / SAMPLES as f64;
            let (q1, q2) = two_solitons(t, x);
            (q1 * p0.sqrt(), q2 * p0.sqrt())
        })
        .unzip()
}

/// The relative error $\lVert u - u_{a} \rVert / \lVert u_{a} \rVert$ over both polarisations.
fn error(x: &[C64], y: &[C64], xa: &[C64], ya: &[C64]) -> f64 {
    let d: f64 = x.iter().zip(xa).chain(y.iter().zip(ya)).map(|(u, v)| (u - v).norm_sqr()).sum();
    let n: f64 = xa.iter().chain(ya).map(C64::norm_sqr).sum();
    (d / n).sqrt()
}

/// The field after the collision with `n_steps` constant steps and the nonlinear step
/// `nonlinear`.
fn propagate(n_steps: usize, nonlinear: NonlinearStep) -> (Vec<C64>, Vec<C64>) {
    let (l, _) = scales();
    let length = (X1 - X0) * l;
    let solver = ManakovSplitStep {
        rule: StepRule::Fixed(length / n_steps as f64),
        nonlinear,
        max_iterations: 50,
        tolerance: 1e-14,
        ..ManakovSplitStep::new(Medium { alpha_np_km: 0.0, beta2_s2_km: BETA2, gamma_w_km: GAMMA })
    };
    let (mut x, mut y) = exact_field(X0);
    let fs = SAMPLES as f64 / (WINDOW * T0);
    let steps = solver.run(&mut x, &mut y, length, fs).unwrap();
    // the last step may be split by the rounding of length / n_steps
    assert!(steps == n_steps || steps == n_steps + 1, "{steps} steps");
    (x, y)
}

#[test]
fn the_manakov_two_soliton_solution_satisfies_eq_1() {
    // fourth-order differences: the residual is their truncation, which falls as h^4
    let residual = |t: f64, x: f64, h: f64| {
        let q = |dt: f64, dx: f64| two_solitons(t + dt, x + dx);
        let d_x = |c: fn((C64, C64)) -> C64| {
            (-c(q(0.0, 2.0 * h)) + 8.0 * c(q(0.0, h)) - 8.0 * c(q(0.0, -h)) + c(q(0.0, -2.0 * h))) / (12.0 * h)
        };
        let d_tt = |c: fn((C64, C64)) -> C64| {
            (-c(q(2.0 * h, 0.0)) + 16.0 * c(q(h, 0.0)) - 30.0 * c(q(0.0, 0.0)) + 16.0 * c(q(-h, 0.0))
                - c(q(-2.0 * h, 0.0)))
                / (12.0 * h * h)
        };
        let (q1, q2) = q(0.0, 0.0);
        let power = q1.norm_sqr() + q2.norm_sqr();
        let j = C64::new(0.0, 1.0);
        let r1 = j * d_x(|v| v.0) + d_tt(|v| v.0) + 2.0 * power * q1;
        let r2 = j * d_x(|v| v.1) + d_tt(|v| v.1) + 2.0 * power * q2;
        ((r1.norm_sqr() + r2.norm_sqr()).sqrt(), (d_tt(|v| v.0).norm_sqr() + d_tt(|v| v.1).norm_sqr()).sqrt())
    };
    for (t, x) in [(0.1, 0.05), (-0.4, -0.1), (6.0, X0), (0.3, 0.0)] {
        let ((r1, scale), (r2, _)) = (residual(t, x, 4e-3), residual(t, x, 2e-3));
        assert!(r2 < 1e-6 * scale.max(1e-3), "residual {r2:.2e} at ({t}, {x}), |q_tt| {scale:.2e}");
        assert!(r1 / r2 > 12.0 || r2 < 1e-9 * scale.max(1e-3), "the residual does not fall as h^4: {r1:.2e}, {r2:.2e}");
    }
}

/// The energy of each polarisation in the half of the window $t > 0$ (`right`) or
/// $t < 0$: after the collision soliton 1 (velocity $2 k_{1I} = 2$) is at $t = +6$.
fn energies(x: &[C64], y: &[C64], right: bool) -> (f64, f64) {
    let half = |u: &[C64]| -> f64 {
        let (a, b) = if right { (SAMPLES / 2, SAMPLES) } else { (0, SAMPLES / 2) };
        u[a..b].iter().map(C64::norm_sqr).sum()
    };
    (half(x), half(y))
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_manakov_solver_follows_the_inelastic_collision_of_fig_1() {
    let mut card = Card::load("radhakrishnan1997_fig1").unwrap();
    let (xa, ya) = exact_field(X1);
    // the paper's transition amplitudes (eqs. (15), (16), (21)): the share of soliton 1
    // in component 1 before and after the collision
    let (k, a, b) = parameters();
    let kappa = |i: usize, m: usize| (a[i] * a[m].conj() + b[i] * b[m].conj()) / (k[i] + k[m].conj());
    let (l1, l2) = (kappa(1, 0) / kappa(0, 0), kappa(0, 1) / kappa(1, 1));
    let t11 = (1.0 - l2 * a[1] / a[0]).norm() / (1.0 - l1 * l2).norm().sqrt();
    let t12 = (1.0 - l2 * b[1] / b[0]).norm() / (1.0 - l1 * l2).norm().sqrt();
    let share_before = a[0].norm_sqr() / (a[0].norm_sqr() + b[0].norm_sqr());
    let (p1, p2) = (a[0].norm_sqr() * t11 * t11, b[0].norm_sqr() * t12 * t12);
    let share_after = p1 / (p1 + p2);
    card.check(&json!({"quantity": "share_in_component_1_after", "source": "eq_16"}), share_after);
    let (e1, e2) = energies(&xa, &ya, true);
    card.check(&json!({"quantity": "share_in_component_1_after", "source": "exact_solution"}), e1 / (e1 + e2));
    println!("soliton 1 in component 1: {share_before:.3} before, {share_after:.4} after (eq. (16))");
    for nonlinear in [NonlinearStep::Midpoint, NonlinearStep::Trapezoidal] {
        let name = format!("{nonlinear:?}").to_lowercase();
        let start = std::time::Instant::now();
        let (x, y) = propagate(600, nonlinear);
        let seconds = start.elapsed().as_secs_f64();
        let (e1, e2) = energies(&x, &y, true);
        let fine = error(&x, &y, &xa, &ya);
        let (x2, y2) = propagate(300, nonlinear);
        let coarse = error(&x2, &y2, &xa, &ya);
        println!(
            "{name}: error {fine:.3e} (600 steps, {seconds:.3} s), {coarse:.3e} (300 steps); soliton 1 in component 1: {:.4}",
            e1 / (e1 + e2)
        );
        card.check(&json!({"solver": name, "quantity": "share_in_component_1_after"}), e1 / (e1 + e2));
        card.check(&json!({"solver": name, "quantity": "order"}), (coarse / fine).log2());
        card.check(&json!({"solver": name, "steps": 600, "quantity": "error"}), fine);
    }
    // no floor: the error keeps falling as h^2 over a factor 100 in the step
    let (x, y) = propagate(600, NonlinearStep::Midpoint);
    let e600 = error(&x, &y, &xa, &ya);
    let (x, y) = propagate(60_000, NonlinearStep::Midpoint);
    let e60k = error(&x, &y, &xa, &ya);
    println!("midpoint: {e600:.3e} (600 steps), {e60k:.3e} (60 000 steps)");
    card.check(&json!({"solver": "midpoint", "quantity": "order_600_to_60000_steps"}), (e600 / e60k).log10() / 2.0);
    card.finish().unwrap();
}

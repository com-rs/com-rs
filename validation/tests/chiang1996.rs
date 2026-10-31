//! The cross-phase modulation of the split-step solvers against the small-signal theory
//! of T.-K. Chiang, N. Kagi, M. E. Marhic and L. G. Kazovsky, "Cross-phase modulation in
//! fiber links with multiple optical amplifiers and dispersion compensators," *J. Lightw.
//! Technol.* 14(3), 249–260, 1996: a continuous probe and a pump whose power is
//! $P_{20} + P_{2m} \cos \omega t$, copolarised, in one span; the probe leaves with the
//! phase modulation of amplitude (eqs. (8)–(9))
//!
//! $$
//! \Delta\phi_{1} = 2 \gamma P_{2m} \sqrt{\eta_{\mathrm{XPM}}}\, L_{\mathrm{eff}}, \qquad
//! \eta_{\mathrm{XPM}} = \frac{\alpha^{2}}{\omega^{2} d_{12}^{2} + \alpha^{2}} \left[1 +
//! \frac{4 \sin^{2}(\omega d_{12} L / 2)\, e^{-\alpha L}}{(1 - e^{-\alpha L})^{2}}\right],
//! $$
//!
//! with $d_{12} = \beta_{2} \Delta\omega$ the walk-off between the channels: the filter
//! that the walk-off applies to the intensity of the pump. The theory neglects the change
//! of the shapes by the dispersion; here $\beta_{2} \omega^{2} L / 2 \le 0.14$ rad up to
//! 2 GHz while $\omega d_{12} L$ reaches 17 rad. The index of the phase modulation of the
//! probe is read on its spectrum, $(|X(\omega)| + |X(-\omega)|) / |X(0)|$, which the
//! dispersion (a phase per frequency) leaves unchanged.
//!
//! The link: 80 km of SMF (0.2 dB/km, 17 ps/(nm km), $\gamma$ = 1.3 /(W km)), the pump
//! 125 GHz above the probe (about 1 nm), $P_{10}$ = 0.1 mW, $P_{20}$ = 1 mW, $P_{2m}$ =
//! 0.5 mW, 2^15 samples at 1.024 THz (a window of 32 ns), steps of 0.5 km (0.02 km for the
//! trapezoidal phase of OptiCommPy, which averages the beat of the two waves at the ends
//! of a step). The Manakov solver, with both waves in one polarisation, has the
//! coefficient $\frac{8}{9}\gamma$.
//! The values and tolerances are in the card `validation/cards/chiang1996_eq9.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::C64;
use comrs::channel::optical::fiber::{Amplification, FiberParams, Ssfm};
use comrs::channel::optical::manakov::ManakovSsfm;
use comrs::core::signal::DualPol;
use comrs::math::fft::fft;
use comrs::math::nlse::{NonlinearStep, StepRule};
use serde_json::json;
use std::f64::consts::PI;
use std::fmt::Write as _;

const N: usize = 1 << 15;
const FS: f64 = 1.024e12;
const PUMP_HZ: f64 = 125e9;
const P10: f64 = 1e-4;
const P20: f64 = 1e-3;
const P2M: f64 = 5e-4;
const STEP_KM: f64 = 0.5;

fn fiber() -> FiberParams {
    FiberParams { n_spans: 1, span_km: 80.0, ..FiberParams::default() }
}

/// The probe at 0 Hz and the pump at `PUMP_HZ` modulated at `fm` (Hz).
fn launch(fm: f64) -> Vec<C64> {
    (0..N)
        .map(|n| {
            let t = n as f64 / FS;
            let pump = (P20 + P2M * (2.0 * PI * fm * t).cos()).sqrt();
            C64::new(P10.sqrt(), 0.0) + C64::from_polar(pump, 2.0 * PI * PUMP_HZ * t)
        })
        .collect()
}

/// The index of the phase modulation of the probe at `fm`, read on its spectrum.
fn index(x: &[C64], fm: f64) -> f64 {
    let s = fft(x);
    let bin = (fm * N as f64 / FS).round() as usize;
    (s[bin].norm() + s[N - bin].norm()) / s[0].norm()
}

/// Eqs. (8)–(9) with the nonlinear coefficient `gamma` (1/(W km)).
fn theory(fm: f64, gamma: f64) -> f64 {
    let f = fiber();
    let (alpha, l) = (f.alpha_np_km(), f.span_km);
    let d12 = f.beta2_s2_km() * 2.0 * PI * PUMP_HZ;
    let w = 2.0 * PI * fm;
    let e = (-alpha * l).exp();
    let eta = alpha * alpha / (w * w * d12 * d12 + alpha * alpha)
        * (1.0 + 4.0 * (w * d12 * l / 2.0).sin().powi(2) * e / ((1.0 - e) * (1.0 - e)));
    2.0 * gamma * P2M * eta.sqrt() * f.effective_length_km()
}

const FREQUENCIES: [f64; 6] = [0.125e9, 0.25e9, 0.5e9, 1.0e9, 1.5e9, 2.0e9];

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_xpm_of_the_solvers_follows_eq_9_of_chiang_1996() {
    let mut card = Card::load("chiang1996_eq9").unwrap();
    let ssfm = Ssfm { amplification: Amplification::None, ..Ssfm::new(fiber(), STEP_KM) };
    for fm in FREQUENCIES {
        let mut x = launch(fm);
        ssfm.propagate(&mut x, FS, None).unwrap();
        let ratio = index(&x, fm) / theory(fm, fiber().gamma_w_km);
        let mut row = format!("{:.3} GHz: theory {:.3e} rad; scalar {ratio:.4}", fm / 1e9, theory(fm, 1.3));
        card.check(&json!({"solver": "scalar", "fm_ghz": fm / 1e9}), ratio);
        // OptiCommPy's trapezoidal phase needs steps 25 times shorter (see the card)
        for (nonlinear, step) in [(NonlinearStep::Midpoint, STEP_KM), (NonlinearStep::Trapezoidal, 0.02)] {
            let link = ManakovSsfm {
                rule: StepRule::Fixed(step),
                nonlinear,
                amplification: Amplification::None,
                ..ManakovSsfm::new(fiber())
            };
            let mut u = DualPol { x: launch(fm), y: vec![C64::new(0.0, 0.0); N] };
            link.propagate(&mut u, FS, None).unwrap();
            let ratio = index(&u.x, fm) / theory(fm, 8.0 / 9.0 * fiber().gamma_w_km);
            let name = format!("{nonlinear:?}").to_lowercase();
            write!(row, ", Manakov {name} {ratio:.4}").unwrap();
            card.check(&json!({"solver": name, "fm_ghz": fm / 1e9}), ratio);
        }
        println!("{row}");
    }
    card.finish().unwrap();
}

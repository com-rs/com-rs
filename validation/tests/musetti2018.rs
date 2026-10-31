//! The step rule `StepRule::Fwm` of the Manakov solver against S. Musetti, P. Serena and
//! A. Bononi, "On the accuracy of split-step Fourier simulations for wideband nonlinear
//! optical communications," *J. Lightw. Technol.* 36(23), 5669–5677, 2018.
//!
//! The rule sets the first step of every span by the four-wave-mixing phase across the
//! signal bandwidth, $h_{1} = \Phi_{\mathrm{FWM}} / (|\beta_{2}| (2\pi B)^{2})$ (eq. (13)),
//! and grows the next ones with the loss, by the constant local error (eq. (10), $q = 3$,
//! "FWM-CLE") or by the update of the nonlinear-phase criterion (eq. (9), "FWM-NLP").
//! The accuracy is the ratio $a_{\mathrm{SSFM}} / a_{\mathrm{NL}}$ of the variance of the
//! numerical error to that of the nonlinear interference (eq. (7)), both measured on the
//! received centre channel: the error against a reference with $\Phi_{\mathrm{FWM}}$ =
//! 1 rad, the interference as the reference less its projection on the transmitted
//! symbols.
//!
//! The case of Fig. 7: seven channels of 49 GBd spaced by 50 GHz ($B$ = 350 GHz),
//! dual-polarisation 16-QAM, 0 dBm per channel, one span of 100 km of SMF (0.2 dB/km,
//! 17 ps/(nm km), $\gamma$ = 1.3 /(W km)), the symmetric step (the phase at the middle of
//! the step). Here Nyquist pulses (the paper: roll-off 0.01), no third-order dispersion
//! (the paper: 0.057 ps/(nm² km)), 4116 symbols per channel and 24 samples per symbol
//! (1.176 THz, three times the bandwidth: the first-order four-wave mixing without
//! aliasing, Appendix B). The values read on the figure and the tolerances are in the
//! card `validation/cards/musetti2018_fig7.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::wdm::NyquistGrid;
use comrs::C64;
use comrs::channel::optical::fiber::FiberParams;
use comrs::core::rng::Rng;
use comrs::mapping::constellation::Constellation;
use comrs::math::nlse::{ManakovSplitStep, Medium, StepRule, StepUpdate};
use serde_json::json;

const RATE: f64 = 49e9;
/// 4116 = 49 × 84 symbols: 50 GHz is 4200 bins.
const GRID: NyquistGrid = NyquistGrid { symbols: 4116, sps: 24, slot_bins: 4200 };
const CHANNELS: i64 = 7;
const BANDWIDTH: f64 = 350e9;

fn medium() -> Medium {
    FiberParams { span_km: 100.0, ..FiberParams::default() }.medium()
}

/// The transmitted symbols of the centre channel and the launched field (both
/// polarisations), 0 dBm per channel.
fn launch() -> ([Vec<C64>; 2], [Vec<C64>; 2]) {
    let points = Constellation::qam(16).unwrap().normalized().points().to_vec();
    let mut rng = Rng::new(2018);
    let p = 1e-3 / 2.0;
    let mut fields = [vec![C64::new(0.0, 0.0); GRID.len()], vec![C64::new(0.0, 0.0); GRID.len()]];
    let mut centre = [Vec::new(), Vec::new()];
    for slot in -(CHANNELS / 2)..=CHANNELS / 2 {
        for (pol, field) in fields.iter_mut().enumerate() {
            let a: Vec<C64> = (0..GRID.symbols).map(|_| points[rng.below(points.len())]).collect();
            for (v, w) in field.iter_mut().zip(GRID.channel(&a, slot, p)) {
                *v += w;
            }
            if slot == 0 {
                centre[pol] = a;
            }
        }
    }
    (centre, fields)
}

/// The received symbols of the centre channel (both polarisations) after the span with
/// the rule of $\Phi_{\mathrm{FWM}}$ = `phase` and the update `update`, and the number
/// of steps.
fn received(launched: &[Vec<C64>; 2], phase: f64, update: StepUpdate) -> ([Vec<C64>; 2], usize) {
    let fs = RATE * GRID.sps as f64;
    let m = medium();
    let solver = ManakovSplitStep {
        rule: StepRule::Fwm { phase_rad: phase, bandwidth_hz: Some(BANDWIDTH), update, max_nl_phase_rad: None },
        ..ManakovSplitStep::new(m)
    };
    let (mut x, mut y) = (launched[0].clone(), launched[1].clone());
    let steps = solver.run(&mut x, &mut y, 100.0, fs).unwrap();
    // the dispersion undone and the loss restored before the matched filter
    let undo = Medium { alpha_np_km: -m.alpha_np_km, beta2_s2_km: -m.beta2_s2_km, gamma_w_km: 0.0 };
    let linear = ManakovSplitStep { rule: StepRule::Fixed(100.0), ..ManakovSplitStep::new(undo) };
    linear.run(&mut x, &mut y, 100.0, fs).unwrap();
    ([GRID.matched_filter(&x, 1e-3 / 2.0), GRID.matched_filter(&y, 1e-3 / 2.0)], steps)
}

/// The variance of the interference: the reference less its projection on the
/// transmitted symbols, per polarisation.
fn interference(reference: &[Vec<C64>; 2], sent: &[Vec<C64>; 2]) -> f64 {
    reference
        .iter()
        .zip(sent)
        .map(|(r, a)| {
            let c: C64 =
                r.iter().zip(a).map(|(u, v)| u * v.conj()).sum::<C64>() / a.iter().map(C64::norm_sqr).sum::<f64>();
            r.iter().zip(a).map(|(u, v)| (u - c * v).norm_sqr()).sum::<f64>()
        })
        .sum()
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_accuracy_of_the_fwm_rule_follows_fig_7_of_musetti_2018() {
    let (sent, launched) = launch();
    let (reference, n_ref) = received(&launched, 1.0, StepUpdate::ConstantLocalError);
    let nli = interference(&reference, &sent);
    println!("reference: {n_ref} steps");
    let mut card = Card::load("musetti2018_fig7").unwrap();
    for (update, name, phases) in [
        (StepUpdate::ConstantLocalError, "fwm_cle", [4.5, 9.0, 18.0, 36.0].as_slice()),
        (StepUpdate::NonlinearPhase, "fwm_nlp", [1.0, 4.0, 10.0].as_slice()),
    ] {
        for &phase in phases {
            let (r, steps) = received(&launched, phase, update);
            let error: f64 =
                r.iter().zip(&reference).flat_map(|(u, v)| u.iter().zip(v)).map(|(a, b)| (a - b).norm_sqr()).sum();
            let ratio = comrs::math::lin2db(error / nli);
            println!("{name} {phase} rad: {steps} steps, a_SSFM / a_NL = {ratio:.1} dB");
            card.check(&json!({"update": name, "phase_rad": phase}), ratio);
        }
    }
    card.finish().unwrap();
}

//! Blind equalisation and decision-directed carrier recovery of 16-QAM
//! (`com_dsp::equalizer::AdaptiveEqualizer`, `com_dsp::sync::carrier::ddpll`) against
//! I. Fatadin, D. Ives and S. J. Savory, "Blind equalization and carrier phase recovery in
//! a 16-QAM optical coherent system," *J. Lightw. Technol.* 27(15), 3042–3049, 2009, at
//! 14 GBd with the OSNR of the paper (signal power over the noise in 0.1 nm, all of it in
//! the detected polarisation: $E_{s}/N_{0} = \mathrm{OSNR} \cdot 12.5/14$, the theory of
//! their Fig. 2).
//!
//! - Fig. 3: the OSNR penalty at a BER of $10^{-3}$ after 0 to 1000 ps/nm of chromatic
//!   dispersion, with 7 taps at $T/2$ adapted by the CMA, the RDE and the DD-LMS ($\mu =
//!   10^{-3}$), and without equaliser; root-raised-cosine pulses of roll-off 1 at the
//!   transmitter and the receiver, a 6-bit ADC (`fatadin2009_fig3`).
//! - Figs. 7 and 8: the decision-directed carrier recovery of their eq. (11), a
//!   first-order loop $\hat\phi(k+1) = \hat\phi(k) - \mu_{\phi} \Im[z(k) e^{*}(k)]$, on
//!   unit-power 16-QAM with Wiener phase noise of $\Delta\nu T_{s}$ (`fatadin2009_fig7_8`).
//!   The same loop, with $\mu_{\phi} = 0.01$, follows the equalisers of Fig. 3.
//!   `ddpll` is the second-order loop of OptiCommPy: with $\tau_{1} = 10^{9} T_{s}$ and
//!   $\tau_{2} = \mu_{\phi} \tau_{1}$ its proportional gain is $\mu_{\phi}$ and its
//!   integral gain $10^{-9}$, the first-order loop of the paper.
//!
//! The tests measure; the values read on the figures, the tolerances and the differences
//! are in the cards.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::crossing;
use comrs::C64;
use comrs::channel::awgn::awgn;
use comrs::channel::optical::fiber::{FiberParams, linear_fiber_channel};
use comrs::core::rng::Rng;
use comrs::dsp::equalizer::{AdaptiveEqualizer, Algorithm};
use comrs::dsp::filter::{PulseShape, fir_filter, pulse_shape};
use comrs::dsp::resample::{downsample, upsample};
use comrs::dsp::signal::pnorm;
use comrs::dsp::sync::carrier::{ddpll, derotate};
use comrs::dsp::sync::frame::{align, frame_sync};
use comrs::hardware::converter::{Quantizer, optimal_loading, rms_per_dimension};
use comrs::hardware::oscillator::{PhaseOrigin, wiener_phase};
use comrs::mapping::constellation::{Constellation, ConstellationKind};
use comrs::math::lin2db;
use comrs::math::search::bisect;
use comrs::metrics::{ber, theory_ber};
use serde_json::json;

const SYMBOL_RATE: f64 = 14e9;
/// The reference bandwidth of the OSNR, 0.1 nm at 1550 nm.
const B_REF: f64 = 12.5e9;

/// The OSNR (dB) at which the theory of Fig. 2, Gray 16-QAM at $E_{s}/N_{0} = \mathrm{OSNR}
/// \cdot B_{\mathrm{ref}}/R_{s}$, reaches the BER `target`.
fn theory_osnr(target: f64) -> f64 {
    let ebn0 = |osnr: f64| osnr + lin2db(B_REF / SYMBOL_RATE) - lin2db(4.0);
    bisect(|osnr| theory_ber(ConstellationKind::Qam, 16, ebn0(osnr)).unwrap() > target, 10.0, 25.0, 60)
}

fn qam16() -> Constellation {
    Constellation::qam(16).unwrap().normalized()
}

fn random_bits(n: usize, rng: &mut Rng) -> Vec<u8> {
    (0..n).map(|_| rng.below(2) as u8).collect()
}

/// The BER of `z` (one sample per symbol, aligned) against `bits`, after `skip` symbols.
fn bit_error_rate(c: &Constellation, z: &[C64], bits: &[u8], skip: usize) -> f64 {
    let b = c.bits_per_symbol();
    ber(&c.demodulate(&z[skip..]), &bits[skip * b..z.len() * b]).unwrap()
}

/// The 16-QAM link of Sec. III after `cd_ps_nm` of dispersion at `osnr_db`, equalised by
/// `alg` (sampled at the symbol instants without equaliser): the BER after the first
/// `skip` symbols (the convergence).
fn link_ber(alg: Option<Algorithm>, cd_ps_nm: f64, osnr_db: f64, n_sym: usize, rng: &mut Rng) -> f64 {
    let (c, sps) = (qam16(), 2usize);
    let fs = SYMBOL_RATE * sps as f64;
    let bits = random_bits(n_sym * 4, rng);
    let symbols = c.modulate(&bits).unwrap();
    let rrc = pulse_shape(PulseShape::Rrc { rolloff: 1.0 }, sps, 32 * sps + 1).unwrap();
    let tx = fir_filter(&rrc, &upsample(&symbols, sps));
    // a lossless fibre of 17 ps/(nm km): the dispersion D L of the abscissa of Fig. 3
    let fiber = FiberParams { alpha_db_km: 0.0, d_ps_nm_km: 17.0, ..FiberParams::default() };
    let rx = linear_fiber_channel(&tx, &fiber, cd_ps_nm / fiber.d_ps_nm_km, fs);
    let rx = awgn(&rx, osnr_db, fs, B_REF, rng);
    let mut rx = fir_filter(&rrc, &rx);
    Quantizer::with_loading(6, rms_per_dimension(&rx), optimal_loading(6)).unwrap().apply(&mut rx);
    let z = match alg {
        Some(a) => {
            let eq = AdaptiveEqualizer { n_taps: 7, sps, mu: 1e-3, ..AdaptiveEqualizer::new(a) };
            eq.run(&[pnorm(&rx)], None, c.points()).unwrap().symbols.swap_remove(0)
        }
        // the power of the symbol instants, not of the half-symbol samples
        None => pnorm(&downsample(&rx, sps, 0)),
    };
    // the carrier tracking of Fig. 6 (eq. (11), mu_phi = 0.01): the dispersion turns the
    // field by a constant phase (0.39 rad at 500 ps/nm) that the blind equalisers keep,
    // and their common phase wanders with the noise of the gradient; then the delay and
    // the ambiguity of pi/2 against the transmitted symbols
    let z = derotate(&z, &decision_directed_loop(&z, 0.01), 1.0).unwrap();
    let skip = n_sym / 2;
    let sync = frame_sync(&z, &symbols, skip..skip + 2000, 20, Some(4)).unwrap();
    let z = align(&z, sync.delay, sync.rotation, n_sym - 20);
    bit_error_rate(&c, &z, &bits, skip)
}

/// The phase corrections of the first-order loop of eq. (11) of step `mu` on unit-power
/// 16-QAM: `ddpll` with $\tau_{1} = 10^{9} T_{s}$ and $\tau_{2} = \mu \tau_{1}$ (see the
/// module).
fn decision_directed_loop(x: &[C64], mu: f64) -> Vec<f64> {
    let ts = 1.0 / SYMBOL_RATE;
    let tau1 = 1e9 * ts;
    ddpll(x, ts, 1.0, tau1, mu * tau1, qam16().points(), None).unwrap()
}

/// The OSNR (dB) at a BER of 10⁻³ of `alg` after `cd_ps_nm`, from a sweep by steps of
/// 0.5 dB up to the first point below 10⁻³ (records of 400 000 symbols, the BER counted
/// on the second half).
fn required_osnr(alg: Option<Algorithm>, cd_ps_nm: f64, rng: &mut Rng) -> f64 {
    let mut points = Vec::new();
    for osnr in (0..20).map(|i| 15.0 + 0.5 * f64::from(i)) {
        let b = link_ber(alg, cd_ps_nm, osnr, 400_000, rng);
        points.push((osnr, b));
        if b < 1e-3 {
            break;
        }
    }
    crossing(&points, 1e-3).unwrap_or(f64::INFINITY)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn fatadin_2009_fig_3_cd_compensation() {
    let mut card = Card::load("fatadin2009_fig3").unwrap();
    let mut rng = Rng::new(2009);
    // the theory of Fig. 2: 16-QAM at Es/N0 = OSNR 12.5/14
    let theory = theory_osnr(1e-3);
    println!("theory: 1e-3 at {theory:.3} dB");
    // every point is measured; the card holds those of the figure that the chain
    // reproduces, the others (the CMA at 1000 ps/nm, the DD-LMS beyond 500 ps/nm, the
    // link without equaliser beyond 100 ps/nm) are in its findings
    let all = [0.0, 250.0, 500.0, 750.0, 1000.0];
    let mut at_1000 = [0.0; 2];
    for (name, alg, cds, checked) in [
        ("cma", Some(Algorithm::Cma), &all[..], &all[..4]),
        ("rde", Some(Algorithm::Rde), &all[..], &all[..]),
        ("dd-lms", Some(Algorithm::DdLms), &all[..], &all[..3]),
        ("none", None, &[0.0, 100.0, 200.0, 300.0][..], &[0.0, 100.0][..]),
    ] {
        for &cd in cds {
            let t = std::time::Instant::now();
            let penalty = required_osnr(alg, cd, &mut rng) - theory;
            println!("{name}, {cd} ps/nm: penalty {penalty:.2} dB ({:.1} s)", t.elapsed().as_secs_f64());
            if checked.contains(&cd) {
                card.check(&json!({"equalizer": name, "cd_ps_nm": cd}), penalty);
            }
            if cd == 1000.0 && name != "dd-lms" {
                at_1000[usize::from(name == "rde")] = penalty;
            }
        }
    }
    // Sec. III: the RDE outperforms the CMA
    card.check(&json!({"claim": "cma_minus_rde_at_1000"}), at_1000[0] - at_1000[1]);
    card.finish().unwrap();
}

/// The carrier recovery of one record of unit-power 16-QAM at the symbol rate.
struct Recovery {
    /// The standard deviation of the phase error (rad), after the first 1000 symbols.
    phase_error_sd: f64,
    /// Whether the loop slipped: a phase error beyond π/4 (rad).
    slipped: bool,
    /// The BER with the ambiguity of π/2 resolved on every block of 1000 symbols against
    /// the transmitted ones (the errors of the cycle slips are not counted).
    ber: f64,
}

/// `n_sym` symbols with Wiener phase noise of `dnu_ts` and noise at `osnr_db`, recovered
/// by the loop of eq. (11) of step `mu`.
fn carrier_recovery(dnu_ts: f64, mu: f64, osnr_db: f64, n_sym: usize, rng: &mut Rng) -> Recovery {
    let c = qam16();
    let bits = random_bits(n_sym * 4, rng);
    let symbols = c.modulate(&bits).unwrap();
    let phase = wiener_phase((2.0 * std::f64::consts::PI * dnu_ts).sqrt(), n_sym, PhaseOrigin::Zero, rng);
    let x = derotate(&symbols, &phase, 1.0).unwrap();
    let x = awgn(&x, osnr_db, SYMBOL_RATE, B_REF, rng);
    let theta = decision_directed_loop(&x, mu);
    let z = derotate(&x, &theta, 1.0).unwrap();
    let skip = 1000;
    let err: Vec<f64> = phase.iter().zip(&theta).skip(skip).map(|(p, t)| p + t).collect();
    let mean = err.iter().sum::<f64>() / err.len() as f64;
    let sd = (err.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / err.len() as f64).sqrt();
    let slipped = err.iter().any(|e| e.abs() > std::f64::consts::FRAC_PI_4);
    let block = 1000;
    let mut aligned = Vec::with_capacity(n_sym);
    for start in (0..n_sym).step_by(block) {
        let end = (start + block).min(n_sym);
        let sync = frame_sync(&z[start..end], &symbols[start..end], 0..end - start, 0, Some(4)).unwrap();
        aligned.extend(align(&z[start..end], 0, sync.rotation, end - start));
    }
    Recovery { phase_error_sd: sd, slipped, ber: bit_error_rate(&c, &aligned, &bits, skip) }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn fatadin_2009_figs_7_8_carrier_recovery() {
    let mut card = Card::load("fatadin2009_fig7_8").unwrap();
    let mut rng = Rng::new(27);
    // Fig. 7: the phase error at an OSNR of 18 dB over 14 µs (196 000 symbols, Fig. 9);
    // the median of 20 records, and the share of the records where the loop slipped. The
    // points that slip (mu = 0.1 at 1.5e-4, mu = 0.01) are measured for the findings of
    // the card, not checked
    for (mu, dnu_ts, checked) in [
        (0.1, 0.25e-4, true),
        (0.1, 0.5e-4, true),
        (0.1, 0.75e-4, true),
        (0.1, 1e-4, true),
        (0.1, 1.5e-4, false),
        (0.01, 0.25e-4, false),
        (0.001, 1e-4, true),
    ] {
        let runs: Vec<Recovery> = (0..20).map(|_| carrier_recovery(dnu_ts, mu, 18.0, 196_000, &mut rng)).collect();
        let mut sds: Vec<f64> = runs.iter().map(|r| r.phase_error_sd).collect();
        sds.sort_by(f64::total_cmp);
        let median = (sds[9] + sds[10]) / 2.0;
        let slips = runs.iter().filter(|r| r.slipped).count();
        println!("mu {mu}, dnu Ts {dnu_ts:.2e}: median phase error {median:.3} rad, {slips}/20 records slipped");
        if checked {
            card.check(&json!({"figure": 7, "mu": mu, "dnu_ts": dnu_ts}), median);
        }
    }
    // Fig. 8: the BER with mu = 0.1, the ambiguity resolved block by block; the curves of
    // 1e-4 and 2e-4, where the loop slips, are measured for the findings
    for dnu_ts in [1e-5, 1e-4, 2e-4] {
        let mut points = Vec::new();
        for osnr in [16.0, 17.0, 18.0, 19.0, 20.0] {
            let r = carrier_recovery(dnu_ts, 0.1, osnr, 1_000_000, &mut rng);
            println!("dnu Ts {dnu_ts:.0e}, OSNR {osnr} dB: BER {:.3e} (slipped: {})", r.ber, r.slipped);
            if dnu_ts == 1e-5 {
                card.check(&json!({"figure": 8, "dnu_ts": dnu_ts, "osnr_db": osnr}), r.ber);
            }
            points.push((osnr, r.ber));
        }
        if let Some(osnr) = crossing(&points, 1e-3) {
            println!("dnu Ts {dnu_ts:.0e}: penalty at 1e-3 {:.2} dB", osnr - theory_osnr(1e-3));
        }
    }
    card.finish().unwrap();
}

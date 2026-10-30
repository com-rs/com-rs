//! End-to-end integration tests: Monte Carlo error rates against the theory, the
//! consistency of the information metrics, and the optical link models against their
//! closed forms.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::optical::edfa::Edfa;
use comrs::channel::optical::fiber::{Amplification, Ssfm};
use comrs::dsp::signal;
use comrs::hardware::optical::{Iqm, Photodiode, coherent_receiver};
use comrs::mapping::constellation::gray_code;
use comrs::math::{cis, power};
use comrs::prelude::*;
use comrs::sim::scenario::from_fn;

/// BER and SER of a constellation over AWGN at a given `E_b/N_0`.
fn error_rates() -> impl Scenario {
    from_fn("awgn-error-rates", |cond, rng, rec| {
        let kind = cond.str("kind")?.parse::<ConstellationKind>()?;
        let c = Constellation::new(kind, cond.usize("m")?)?.normalized();
        let (idx, x) = source::symbols(&c, 2000, None, rng)?;
        let y = awgn::awgn_ebn0(&x, cond.f64("ebn0_db")?, c.bits_per_symbol() as f64, 1.0, rng);
        let bits = comrs::mapping::constellation::indices_to_bits(&idx, c.bits_per_symbol());
        rec.errors("ber", metrics::bit_errors(&c.demodulate(&y), &bits)? as u64, bits.len() as u64);
        rec.errors("ser", metrics::symbol_errors(&c.decide(&y), &idx)? as u64, idx.len() as u64);
        Ok(())
    })
}

#[test]
fn monte_carlo_error_rates_match_the_theory() {
    let grid = Grid::from_conditions(vec![
        Condition::new().with("kind", "qam").with("m", 4usize).with("ebn0_db", 6.0),
        Condition::new().with("kind", "qam").with("m", 16usize).with("ebn0_db", 10.0),
        Condition::new().with("kind", "qam").with("m", 64usize).with("ebn0_db", 14.0),
        Condition::new().with("kind", "psk").with("m", 2usize).with("ebn0_db", 6.0),
        Condition::new().with("kind", "psk").with("m", 8usize).with("ebn0_db", 12.0),
        Condition::new().with("kind", "pam").with("m", 4usize).with("ebn0_db", 12.0),
    ]);
    let rule = Stop::min_errors("ber", 2000).or(Stop::max_trials(4000));
    let results = Runner::new().seed(21).batch(64).stop(rule).run(&error_rates(), &grid).unwrap();
    for r in &results {
        let kind = r.condition.str("kind").unwrap().parse::<ConstellationKind>().unwrap();
        let (m, ebn0) = (r.condition.usize("m").unwrap(), r.condition.f64("ebn0_db").unwrap());
        // the Gray BER formula is a nearest-neighbour approximation: 3 % at these SNRs
        let ber = r.metric("ber").unwrap().value();
        let theory = metrics::theory_ber(kind, m, ebn0).unwrap();
        assert!((ber / theory - 1.0).abs() < 0.08, "{}: BER {ber:.3e} against {theory:.3e}", r.condition.label());
        // the SER of QAM and PAM is exact, the PSK one a tight bound
        let ser = r.metric("ser").unwrap().value();
        let theory = metrics::theory_ser(kind, m, ebn0).unwrap();
        let tol = if kind == ConstellationKind::Psk { 0.1 } else { 0.06 };
        assert!((ser / theory - 1.0).abs() < tol, "{}: SER {ser:.3e} against {theory:.3e}", r.condition.label());
    }
}

#[test]
fn information_rates_are_ordered_and_bounded() {
    let mut rng = Rng::new(8);
    let c = Constellation::qam(16).unwrap().normalized();
    let mut previous = 0.0;
    for snr_db in [0.0, 6.0, 12.0, 18.0] {
        let (_, x) = source::symbols(&c, 40_000, None, &mut rng).unwrap();
        let y = awgn::awgn(&x, snr_db, 1.0, 1.0, &mut rng);
        let mi = metrics::monte_carlo_mi(&y, &x, &c, None).unwrap();
        let gmi = metrics::monte_carlo_gmi(&y, &x, &c, None).unwrap().gmi;
        let capacity = (1.0 + db2lin(snr_db)).log2();
        assert!(gmi <= mi + 0.02, "GMI {gmi} above MI {mi} at {snr_db} dB");
        assert!(mi <= capacity + 0.02 && mi <= 4.0 + 1e-9, "MI {mi} above the bounds at {snr_db} dB");
        assert!(mi > previous);
        previous = mi;
    }
    assert!(previous > 3.95);
}

#[test]
fn llr_signs_are_the_hard_decisions() {
    let mut rng = Rng::new(3);
    for c in [Constellation::qam(64).unwrap(), Constellation::psk(8).unwrap(), Constellation::pam(4).unwrap()] {
        let c = c.normalized();
        let (_, x) = source::symbols(&c, 500, None, &mut rng).unwrap();
        let y = awgn::awgn(&x, 15.0, 1.0, 1.0, &mut rng);
        let hard = c.demodulate(&y);
        for max_log in [true, false] {
            let llr = c.llr(&y, 0.05, None, max_log).unwrap();
            // LLR = ln p(b = 0) - ln p(b = 1): positive for a 0
            let from_llr: Vec<u8> = comrs::core::signal::hard_decisions(&llr);
            if max_log {
                assert_eq!(from_llr, hard, "max-log LLRs are the nearest-point decisions");
            } else {
                let disagree = from_llr.iter().zip(&hard).filter(|(a, b)| a != b).count();
                assert!(disagree * 200 < hard.len(), "{disagree} bits differ");
            }
        }
    }
}

#[test]
fn gray_labels_differ_by_one_bit_between_neighbours() {
    for (c, d_min) in [(Constellation::qam(16).unwrap(), 2.0), (Constellation::pam(8).unwrap(), 2.0)] {
        let p = c.points();
        let labels = gray_code(c.bits_per_symbol() as u32).unwrap();
        assert_eq!(labels.len(), c.order());
        for i in 0..p.len() {
            for j in 0..i {
                if ((p[i] - p[j]).norm() - d_min).abs() < 1e-9 {
                    assert_eq!((i ^ j).count_ones(), 1, "points {i} and {j} are neighbours");
                }
            }
        }
    }
}

#[test]
fn linear_split_step_is_the_linear_channel_with_ideal_gain() {
    let fiber = FiberParams { gamma_w_km: 0.0, n_spans: 3, span_km: 50.0, ..FiberParams::default() };
    let mut rng = Rng::new(2);
    let x = rng.complex_noise(1024, 1e-3);
    let fs = 64e9;
    let ssfm = Ssfm { fiber: fiber.clone(), step_km: 5.0, amplification: Amplification::Ideal, raman: None };
    let y = ssfm.propagate(&x, fs, None).unwrap();
    let reference = fiber::linear_fiber_channel(&x, &fiber, fiber.total_km(), fs);
    let gain = (fiber.alpha_np_km() / 2.0 * fiber.total_km()).exp();
    let err: f64 = y.iter().zip(&reference).map(|(a, b)| (a - b * gain).norm_sqr()).sum::<f64>();
    assert!(err / (power(&x) * x.len() as f64) < 1e-20);
}

#[test]
fn ase_of_a_chain_of_amplifiers_adds_up() {
    let fiber = FiberParams { gamma_w_km: 0.0, n_spans: 5, ..FiberParams::default() };
    let fs = 64e9;
    let ssfm = Ssfm::new(fiber.clone(), 10.0);
    let mut rng = Rng::new(17);
    let silence = vec![C64::new(0.0, 0.0); 1 << 16];
    let y = ssfm.propagate(&silence, fs, Some(&mut rng)).unwrap();
    // every EDFA compensates the loss of the following span: the noises add up
    let amp = Edfa { gain_db: fiber.span_gain_db(), nf_db: fiber.nf_db, fc_hz: fiber.fc_hz };
    let expected = fiber.n_spans as f64 * amp.noise_variance(fs);
    assert!((power(&y) / expected - 1.0).abs() < 0.03, "{} against {expected}", power(&y));
}

#[test]
fn nonlinear_phase_of_a_constant_field_is_gamma_p_leff() {
    // a constant field is not dispersed: the split step only accumulates the phase
    // γ P L_eff per span (the step discretises L_eff)
    let fiber = FiberParams { n_spans: 2, ..FiberParams::default() };
    let p: f64 = 5e-3;
    let x = vec![C64::new(p.sqrt(), 0.0); 256];
    let ssfm = Ssfm { fiber: fiber.clone(), step_km: 0.1, amplification: Amplification::Ideal, raman: None };
    let y = ssfm.propagate(&x, 32e9, None).unwrap();
    let phase = y[0].arg();
    let expected = fiber.n_spans as f64 * fiber.gamma_w_km * p * fiber.effective_length_km();
    assert!((phase - expected).abs() < 1e-3 * expected, "{phase} against {expected}");
    assert!((y[0].norm_sqr() / p - 1.0).abs() < 1e-9);
}

#[test]
fn optical_transmitter_and_coherent_receiver_recover_the_symbols() {
    // an IQ modulator biased at the null is linear for small drives: E ≈ E_in sin(πu/2Vπ)/√2
    let c = Constellation::qam(16).unwrap().normalized();
    let mut rng = Rng::new(6);
    let (idx, x) = source::symbols(&c, 2000, None, &mut rng).unwrap();
    let iqm = Iqm::default();
    let drive: Vec<C64> = x.iter().map(|v| v * 0.05 * iqm.vpi).collect();
    let laser = vec![C64::new(0.1, 0.0); x.len()];
    let e = iqm.modulate(&laser, &drive).unwrap();
    let lo = vec![C64::new(1.0, 0.0); x.len()];
    let r = coherent_receiver(&e, &lo, &Photodiode::ideal(), 32e9, &mut rng).unwrap();
    assert_eq!(c.decide(&signal::pnorm(&r)), idx);
    // and the residual distortion of the sine is small but present
    let snr = metrics::snr_db(&signal::pnorm(&r), &x).unwrap();
    assert!(snr > 25.0 && snr < 80.0, "{snr}");
}

#[test]
fn frequency_shift_and_derotation_cancel() {
    let mut rng = Rng::new(1);
    let x = rng.complex_noise(100, 1.0);
    let y = signal::freq_shift(&x, 1e9, 32e9);
    let theta: Vec<f64> = (0..100).map(|n| -2.0 * std::f64::consts::PI * 1e9 * n as f64 / 32e9).collect();
    let z = comrs::dsp::sync::carrier::derotate(&y, &theta, 1.0).unwrap();
    assert!(z.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
    assert!((cis(0.5) * cis(-0.5) - C64::new(1.0, 0.0)).norm() < 1e-15);
}

//! The reference points of D. J. Costello and G. D. Forney, "Channel coding: the road to
//! channel capacity," *Proc. IEEE* 95(6), 2007 (arXiv cs/0611112), against the limits and
//! closed forms of the library.
//!
//! - Sec. II: the ultimate Shannon limit $\ln 2$ ($-1.59$ dB, `gaussian_limit_ebn0_db`),
//!   1.76 dB at $\eta = 2$, uncoded binary PAM at $P_{b} = 10^{-5}$ for $E_{b}/N_{0} =
//!   9.6$ dB (`ebn0_for_ber`), hence a real coding gain of at most 11.2 dB, a gap of 7.8
//!   dB at $\eta = 2$ and 3.35 dB more as $\eta \to 0$; binary codes cost less than
//!   0.2 dB below 1 b/s/Hz (`shannon_limit_ebn0_db` at rate 1/2 against $\eta = 1$); the
//!   (32, 6, 16) biorthogonal code, $P_{b} \approx (62/6) Q(\sqrt{6 E_{b}/N_{0}})$, needs
//!   5.8 dB at $10^{-5}$, a real coding gain of 3.8 dB.
//! - Sec. V-A: uncoded square QAM reaches $P_{s} = 10^{-5}$ at $\mathrm{SNR}_{\mathrm{norm}}
//!   \approx 8.5$ dB (`ser_esn0` of 1024-QAM); shaping gains at most $\pi e/6$ (1.53 dB),
//!   which Maxwell–Boltzmann probabilities on a large QAM grid approach (`maxwell_boltzmann`).
//! - Sec. IV-B: a 64-state convolutional code decoded by the Viterbi algorithm obtains a
//!   real coding gain "on the order of 6 dB" (`ConvolutionalCode::k7`, soft decisions).
//!
//! The values and tolerances are in the cards `validation/cards/costello2007_limits.yaml`
//! and `validation/cards/costello2007_viterbi.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::{coded_frame, sweep_crossing, until_errors};
use comrs::fec::block::BlockCode;
use comrs::fec::bounds::{bpsk_noise_variance, gaussian_limit_ebn0_db, shannon_limit_ebn0_db};
use comrs::fec::conv::ConvolutionalCode;
use comrs::mapping::constellation::{Constellation, ConstellationKind};
use comrs::mapping::source::maxwell_boltzmann;
use comrs::mapping::theory::{ebn0_for_ber, ser_esn0};
use comrs::math::{bisect, db2lin, entropy, lin2db, qfunc};
use serde_json::json;

/// The SNR per two dimensions (linear) at which square `m`-QAM reaches the SER `target`,
/// normalised by $2^{\eta} - 1 = m - 1$.
fn qam_snr_norm_db(m: usize, target: f64) -> f64 {
    let esn0 = bisect(|e| ser_esn0(ConstellationKind::Qam, m, e).unwrap() > target, 0.0, 60.0, 100);
    esn0 - lin2db(m as f64 - 1.0)
}

/// The gain (dB) of Maxwell–Boltzmann probabilities of entropy `bits` on square `m`-QAM
/// over the uniform square QAM of $2^{\text{bits}}$ points on the same grid.
fn shaping_gain_db(m: usize, bits: usize) -> f64 {
    let c = Constellation::qam(m).unwrap();
    let lambda = bisect(|l| entropy(&maxwell_boltzmann(&c, l)) > bits as f64, 0.0, 1.0, 200);
    let shaped = c.energy_with(&maxwell_boltzmann(&c, lambda));
    let u = Constellation::qam(1 << bits).unwrap();
    let uniform = u.energy_with(&vec![1.0 / u.points().len() as f64; u.points().len()]);
    lin2db(uniform / shaped)
}

#[test]
fn the_limits_of_secs_ii_and_v_of_costello_2007() {
    let mut card = Card::load("costello2007_limits").unwrap();
    let ultimate = gaussian_limit_ebn0_db(0.0);
    let eta2 = gaussian_limit_ebn0_db(2.0);
    let uncoded = ebn0_for_ber(ConstellationKind::Psk, 2, 1e-5).unwrap();
    let biorthogonal = bisect(|e| 62.0 / 6.0 * qfunc((6.0 * db2lin(e)).sqrt()) > 1e-5, 0.0, 15.0, 100);
    for (quantity, value) in [
        ("ultimate_shannon_limit", ultimate),
        ("shannon_limit_eta_2", eta2),
        ("uncoded_binary_pam_1e-5", uncoded),
        ("largest_real_coding_gain_1e-5", uncoded - ultimate),
        ("gap_to_capacity_eta_2", uncoded - eta2),
        ("gain_eta_2_to_0", eta2 - ultimate),
        ("binary_input_cost_eta_1", shannon_limit_ebn0_db(0.5) - gaussian_limit_ebn0_db(1.0)),
        ("biorthogonal_32_6_16_1e-5", biorthogonal),
        ("biorthogonal_real_coding_gain", uncoded - biorthogonal),
        ("uncoded_qam_snr_norm_1e-5", qam_snr_norm_db(1024, 1e-5)),
        ("ultimate_shaping_gain", lin2db(std::f64::consts::PI * std::f64::consts::E / 6.0)),
        ("maxwell_boltzmann_gain_10_bits", shaping_gain_db(4096, 10)),
    ] {
        card.check(&json!({ "quantity": quantity }), value);
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_64_state_code_gains_about_6_db_as_in_sec_iv_of_costello_2007() {
    let code = ConvolutionalCode::k7(1000).unwrap();
    let uncoded = ebn0_for_ber(ConstellationKind::Psk, 2, 1e-5).unwrap();
    let mut seed = 0;
    let crossing = sweep_crossing(3.5, 0.25, 12, 1e-5, |_, ebn0| {
        seed += 1;
        let s2 = bpsk_noise_variance(ebn0, code.k() as f64 / code.n() as f64);
        let c = until_errors(seed, 200, 200_000, 100, |rng| {
            let f = coded_frame(&code, s2, rng)?;
            Ok((f.bit_errors, code.k() as u64))
        })
        .unwrap();
        c.errors as f64 / c.total as f64
    })
    .unwrap();
    println!("K = 7, soft Viterbi: 1e-5 at {crossing:.2} dB, real coding gain {:.2} dB", uncoded - crossing);
    let mut card = Card::load("costello2007_viterbi").unwrap();
    card.check(&json!({"quantity": "real_coding_gain_1e-5"}), uncoded - crossing);
    card.finish().unwrap();
}

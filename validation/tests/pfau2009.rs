//! The blind phase search (`com_dsp::sync::carrier::bps`) with the differential quadrant
//! code of square QAM (`com_mapping::differential::QuadrantQam`) against T. Pfau,
//! S. Hoffmann and R. Noé, "Hardware-efficient coherent digital receiver concept with
//! feedforward carrier recovery for M-QAM constellations," *J. Lightw. Technol.* 27(8),
//! 2009, Sec. V.
//!
//! The link of the paper: unit-energy square QAM whose quadrant is differentially
//! encoded, Wiener phase noise of increments of variance $2\pi \Delta f T_{s}$ (eq. (2)),
//! white Gaussian noise, the blind phase search over $2N + 1 = 19$ symbols with $B = 32$
//! test phases for 4- and 16-QAM and $B = 64$ for 64- and 256-QAM, the phase unwrapped
//! with a period of $\pi/2$ (the quadrant jumps of eq. (12)), hard decisions and the
//! differential decoding of the quadrants. The penalty is the $E_{s}/N_{0}$ at a BER of
//! $10^{-3}$ above that of Gray QAM given by eq. (17) of the paper; it includes the cost
//! of the differential code.
//!
//! - Fig. 6: the penalty without phase noise, 0.7, 0.6, 0.5 and 0.55 dB.
//! - Table I: the linewidth times symbol duration of a 1 dB penalty, $4.1 \cdot 10^{-4}$,
//!   $1.4 \cdot 10^{-4}$, $4.0 \cdot 10^{-5}$ and $8.0 \cdot 10^{-6}$: the penalty measured
//!   there.
//!
//! The paper simulates 200 000 symbols per point; here 10⁶. The values read and the
//! tolerances are in the card `validation/cards/pfau2009_table1.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::sweep_crossing;
use comrs::C64;
use comrs::channel::awgn::add_noise;
use comrs::core::rng::Rng;
use comrs::dsp::sync::carrier::{bps, derotate, unwrap};
use comrs::hardware::oscillator::{PhaseOrigin, wiener_phase};
use comrs::mapping::differential::QuadrantQam;
use comrs::mapping::source;
use comrs::math::{cis, db2lin, lin2db, qfuncinv};
use serde_json::json;
use std::f64::consts::{FRAC_PI_2, PI};

const SYMBOLS: usize = 1_000_000;
const HALF_WIDTH: usize = 9;

/// The $E_{s}/N_{0}$ (dB) at which Gray square QAM reaches `ber`, eq. (17) of the paper.
fn theory_esn0_db(m: usize, ber: f64) -> f64 {
    let (mf, k) = (m as f64, (m as f64).log2());
    let q = qfuncinv(k / 2.0 / (1.0 - 1.0 / mf.sqrt()) * (1.0 - (1.0 - ber).sqrt()));
    lin2db((mf - 1.0) / 3.0 * q * q)
}

/// The number of test phases of the paper.
fn test_phases(m: usize) -> usize {
    if m <= 16 { 32 } else { 64 }
}

/// The BER of the link at `esn0_db` with the linewidth times symbol duration `dfts`.
fn link_ber(m: usize, dfts: f64, esn0_db: f64, rng: &mut Rng) -> f64 {
    let qq = QuadrantQam::new(m).unwrap();
    let bits = source::bits(SYMBOLS * qq.bits_per_symbol(), rng);
    let x = qq.map_bits(&bits).unwrap();
    let phase = wiener_phase((2.0 * PI * dfts).sqrt(), x.len(), PhaseOrigin::Zero, rng);
    let mut r: Vec<C64> = x.iter().zip(&phase).map(|(v, p)| v * cis(*p)).collect();
    add_noise(&mut r, 1.0 / db2lin(esn0_db), rng);
    let theta = unwrap(&bps(&r, HALF_WIDTH, &qq.points(), test_phases(m)).unwrap(), FRAC_PI_2);
    let decided = qq.demap_bits(&derotate(&r, &theta, 1.0).unwrap()).unwrap();
    let errors = decided.iter().zip(&bits).filter(|(a, b)| a != b).count();
    errors as f64 / bits.len() as f64
}

/// The penalty (dB) at a BER of $10^{-3}$.
fn penalty(m: usize, dfts: f64, rng: &mut Rng) -> f64 {
    let theory = theory_esn0_db(m, 1e-3);
    let crossing = sweep_crossing(theory, 0.2, 20, 1e-3, |_, esn0| link_ber(m, dfts, esn0, rng)).unwrap();
    crossing - theory
}

#[test]
fn eq_17_is_the_gray_ber_of_square_qam() {
    // QPSK: BER = Q(√(Es/N0)) up to 2(1 - √(1 - BER)) ≈ BER, so 10⁻³ at Q⁻¹(10⁻³)²
    assert!((theory_esn0_db(4, 1e-3) - lin2db(qfuncinv(1e-3).powi(2))).abs() < 1e-3);
    assert!((theory_esn0_db(16, 1e-3) - 16.54).abs() < 0.01);
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_linewidth_tolerance_follows_table_i_of_pfau_2009() {
    let mut card = Card::load("pfau2009_table1").unwrap();
    let mut rng = Rng::new(2009);
    for (m, table) in [(4, 4.1e-4), (16, 1.4e-4), (64, 4.0e-5), (256, 8.0e-6)] {
        let (p0, p1) = (penalty(m, 0.0, &mut rng), penalty(m, table, &mut rng));
        println!("{m}-QAM: penalty {p0:.2} dB without phase noise, {p1:.2} dB at {table:e}");
        card.check(&json!({"m": m, "quantity": "penalty_without_phase_noise"}), p0);
        card.check(&json!({"m": m, "quantity": "penalty_at_table_i"}), p1);
    }
    card.finish().unwrap();
}

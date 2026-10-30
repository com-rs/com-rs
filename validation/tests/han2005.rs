//! The PAPR of OFDM and its reduction (`comrs::waveform::papr`, `comrs::waveform::ofdm`)
//! against S. H. Han and J. H. Lee, "An overview of peak-to-average power ratio
//! reduction techniques for multicarrier transmission," *IEEE Wireless Commun.* 12(2),
//! 56–65, 2005: the PAPR of eq. (4) on the signal oversampled $L = 4$ times, for the
//! sixteen BPSK blocks of four subcarriers of Table 1, the PTS example of Fig. 3 (eight
//! subcarriers, four adjacent subblocks, factors $\pm 1$) and the SLM example (eight
//! subcarriers, the four phase sequences of the text); and the 0.1 % PAPR of QPSK OFDM of
//! 256 and 1024 subcarriers, unmodified and with SLM of 16 candidates (Fig. 1 and the
//! text, Monte Carlo). The tests measure; the values of the paper, their tolerances and
//! the differences found are in the cards `validation/cards/han2005_examples.yaml` and
//! `validation/cards/han2005_fig1.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::math::papr_db;
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::papr::{Partition, Pts, Slm};
use serde_json::json;

fn block(v: &[f64]) -> Vec<C64> {
    v.iter().map(|&x| C64::new(x, 0.0)).collect()
}

fn label(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
}

/// The PAPR (dB) of one block of BPSK symbols, oversampled four times (eq. (4)).
fn papr_of(v: &[f64]) -> f64 {
    let ofdm = Ofdm::new(v.len(), v.len()).unwrap();
    papr_db(&ofdm.modulate(&block(v)).unwrap())
}

/// Table 1, the PTS example of Fig. 3 and the SLM example.
#[test]
fn han_lee_2005_examples() {
    let mut card = Card::load("han2005_examples").unwrap();
    // Table 1: the sixteen blocks of four BPSK subcarriers
    for i in 0..16usize {
        let v: Vec<f64> = (0..4).map(|b| if (i >> (3 - b)) & 1 == 0 { 1.0 } else { -1.0 }).collect();
        card.check(&json!({"table": 1, "block": label(&v)}), papr_of(&v));
    }
    // PTS (Fig. 3): four adjacent subblocks of two subcarriers, factors ±1
    let x = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0];
    let ofdm = Ofdm::new(8, 8).unwrap();
    let pts = Pts { subblocks: 4, phases: 2, partition: Partition::Adjacent };
    let (reduced, side) = ofdm.modulate_with(&block(&x), &pts).unwrap();
    let b: Vec<f64> = pts.factors(side[0]).iter().map(|f| f.re.round()).collect();
    println!("PTS: factors {b:?}");
    card.check(&json!({"example": "pts", "metric": "papr_original_db"}), papr_of(&x));
    card.check(&json!({"example": "pts", "metric": "papr_reduced_db"}), papr_db(&reduced));
    card.check(
        &json!({"example": "pts", "metric": "factors_match"}),
        f64::from(u8::from(b == [1.0, -1.0, -1.0, -1.0])),
    );
    let assign = pts.assignment(8);
    let modified: Vec<f64> = x.iter().zip(&assign).map(|(v, &m)| v * b[m]).collect();
    card.check(
        &json!({"example": "pts", "metric": "block_matches"}),
        f64::from(u8::from(modified == [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])),
    );
    // SLM: the four phase sequences of the text applied to X
    let x = [1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0];
    let sequences = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
    ];
    let paprs: Vec<f64> =
        sequences.iter().map(|s| papr_of(&x.iter().zip(s).map(|(a, b)| a * b).collect::<Vec<_>>())).collect();
    println!("SLM: PAPR of the four candidates {paprs:.2?} dB");
    for (u, p) in paprs.iter().enumerate() {
        card.check(&json!({"example": "slm", "candidate": u + 1}), *p);
    }
    card.finish().unwrap();
}

/// The PAPR exceeded by a fraction `p` of the values.
fn exceeded_by(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((1.0 - p) * v.len() as f64) as usize]
}

/// Fig. 1: the 0.1 % PAPR of QPSK OFDM, unmodified and with SLM of 16 candidates.
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn han_lee_2005_fig_1() {
    let mut card = Card::load("han2005_fig1").unwrap();
    let qpsk = Constellation::psk(4).unwrap().normalized();
    for n in [256, 1024] {
        let t = std::time::Instant::now();
        let ofdm = Ofdm::new(n, n).unwrap();
        let slm = Slm { candidates: 16, seed: 2005 };
        let mut rng = Rng::new(n as u64);
        let (mut plain, mut reduced) = (Vec::new(), Vec::new());
        for _ in 0..100_000 {
            let data: Vec<C64> = (0..n).map(|_| qpsk.points()[rng.below(4)]).collect();
            plain.push(papr_db(&ofdm.modulate(&data).unwrap()));
            reduced.push(papr_db(&ofdm.modulate_with(&data, &slm).unwrap().0));
        }
        let (a, b) = (exceeded_by(plain, 1e-3), exceeded_by(reduced, 1e-3));
        println!("N = {n}: 0.1 % PAPR {a:.2} dB, {b:.2} dB with SLM ({:.0} s)", t.elapsed().as_secs_f64());
        card.check(&json!({"n": n, "candidates": 1}), a);
        card.check(&json!({"n": n, "candidates": 16}), b);
        card.check(&json!({"n": n, "metric": "reduction_db"}), a - b);
    }
    card.finish().unwrap();
}

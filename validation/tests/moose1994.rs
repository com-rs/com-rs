//! The frequency offset of OFDM against P. H. Moose, "A technique for orthogonal
//! frequency division multiplexing frequency offset correction," *IEEE Trans. Commun.*
//! 42(10), 2908–2914, 1994: the interference of eq. (10), the bound of eq. (15), the
//! estimate of eq. (21) and its spread (eq. (26), Fig. 3), the example of Sec. IV and the
//! experiment of Fig. 1 (`comrs::waveform::ofdm::sync`). The tests measure; the values of
//! the paper, their tolerances and the differences found (the experiment of Fig. 1 above
//! the interference of eq. (10) at large offsets) are in the card
//! `validation/cards/moose1994.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::awgn::add_noise;
use comrs::hardware::oscillator::frequency_offset;
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::ofdm::sync::{cfo_snr_bound, ici_power, moose, moose_std, repeated_preamble};
use serde_json::json;
use std::f64::consts::TAU;

/// N = 256 and 193 subcarriers (K = 96), at one sample per subcarrier spacing.
fn grid() -> Ofdm {
    Ofdm { oversampling: 1, ..Ofdm::new(256, 193).unwrap() }
}

fn part(name: &str) -> Card {
    Card::load("moose1994").unwrap().part(&json!({ "part": name })).unwrap()
}

#[test]
fn moose_1994_example_interference_and_bound() {
    // Sec. IV: Es/N0 = 28 dB on the shortened symbols, four times the carrier spacing
    let mut card = part("example");
    let s0 = moose_std(db2lin(28.0));
    let s = 4.0 * s0;
    // the signal-to-interference ratio of eq. (15) without noise
    let sir = cfo_snr_bound(f64::INFINITY, s);
    card.check(&json!({"part": "example", "quantity": "sigma_initial"}), s0);
    card.check(&json!({"part": "example", "quantity": "sigma_data"}), s);
    card.check(&json!({"part": "example", "quantity": "sir_db"}), lin2db(sir));
    let ec = db2lin(11.0);
    card.check(&json!({"part": "example", "quantity": "snr_loss_db"}), lin2db(ec) - lin2db(cfo_snr_bound(ec, s)));
    card.finish().unwrap();
    // eq. (13): the worst subcarrier at half a spacing
    let ofdm = grid();
    let mut card = part("interference");
    let worst = (0..ofdm.n_active).map(|i| ici_power(&ofdm, i, 0.5)).fold(0.0, f64::max);
    card.check(&json!({"part": "interference", "quantity": "worst_sum_at_half_spacing"}), worst);
    card.finish().unwrap();
    // the solid lines of Fig. 1
    let mut card = part("bound");
    for (ec, eps) in [(29, 0.1), (29, 0.5), (11, 0.25)] {
        let v = lin2db(cfo_snr_bound(db2lin(f64::from(ec)), eps));
        card.check(&json!({"part": "bound", "ec_n0_db": ec, "eps": eps}), v);
    }
    card.finish().unwrap();
}

/// The SNR after the DFT (dB) of 8PSK on every subcarrier under the offset `eps`, at
/// `ec_n0_db` per subcarrier: the common gain fitted, the rest counted as interference
/// and noise (Moose, eq. (14)).
fn snr_after_dft(eps: f64, ec_n0_db: f64, rng: &mut Rng) -> f64 {
    let ofdm = grid();
    let psk = Constellation::psk(8).unwrap();
    let n = ofdm.fft_len();
    // noise per sample such that a subcarrier of unit power has Ec/N0 after the DFT
    let noise_var = n as f64 / (ofdm.n_active as f64 * db2lin(ec_n0_db));
    let (mut xy, mut xx, mut sent, mut got) = (C64::new(0.0, 0.0), 0.0, Vec::new(), Vec::new());
    for _ in 0..200 {
        let x: Vec<C64> = (0..ofdm.n_active).map(|_| psk.points()[rng.below(8)]).collect();
        let mut s = ofdm.to_time(&x).unwrap();
        frequency_offset(&mut s, TAU * eps / n as f64, false);
        add_noise(&mut s, noise_var, rng);
        let y = ofdm.to_active(&s).unwrap();
        for (a, b) in x.iter().zip(&y) {
            xy += b * a.conj();
            xx += a.norm_sqr();
        }
        sent.extend(x);
        got.extend(y);
    }
    let g = xy / xx;
    let err: f64 = sent.iter().zip(&got).map(|(a, b)| (b - g * a).norm_sqr()).sum();
    lin2db(g.norm_sqr() * xx / err)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn moose_1994_fig_1_experiment() {
    let mut rng = Rng::new(1994);
    let mut card = part("experiment");
    for (ec, eps) in [(29, 0.05), (29, 0.1), (29, 0.15), (11, 0.1)] {
        let v = snr_after_dft(eps, f64::from(ec), &mut rng);
        card.check(&json!({"part": "experiment", "ec_n0_db": ec, "eps": eps}), v);
    }
    card.finish().unwrap();
    // the finding of the card: the experiment points of Fig. 1 at large offsets
    for eps in [0.25, 0.35, 0.5] {
        println!("eps {eps}: {:.2} dB (Fig. 1: 7.7, 4.5, 1.0 dB)", snr_after_dft(eps, 29.0, &mut rng));
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn moose_1994_fig_3_estimate() {
    let ofdm = grid();
    let n = ofdm.fft_len();
    let mut rng = Rng::new(3);
    let mut card = part("estimate");
    for (es_n0_db, eps) in [(22, 0.0), (30, 0.0), (40, 0.0), (50, 0.0), (40, 0.45), (50, 0.45)] {
        let mut sq = 0.0;
        let trials: u32 = 2000;
        for t in 0..trials {
            let p = repeated_preamble(&ofdm, u64::from(t)).unwrap();
            let cp = p.len() - 2 * n;
            // Es: the energy of one block; the noise per sample gives Es/N0
            let es: f64 = p[cp..cp + n].iter().map(C64::norm_sqr).sum();
            let mut r = p.clone();
            frequency_offset(&mut r, TAU * eps / n as f64, false);
            add_noise(&mut r, es / db2lin(f64::from(es_n0_db)), &mut rng);
            sq += (moose(&r, cp, &ofdm).unwrap() - eps).powi(2);
        }
        let std = (sq / f64::from(trials)).sqrt();
        card.check(&json!({"part": "estimate", "es_n0_db": es_n0_db, "eps": eps}), std);
    }
    card.finish().unwrap();
}

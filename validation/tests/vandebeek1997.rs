//! The joint maximum-likelihood timing and frequency offset of OFDM from the cyclic
//! prefix (`comrs::waveform::ofdm::sync::cp_ml`) against J.-J. van de Beek, M. Sandell
//! and P. O. Börjesson, "ML estimation of time and frequency offset in OFDM systems,"
//! *IEEE Trans. Signal Process.* 45(7), 1800–1805, 1997: the offset of eq. (2) and the
//! mean squared errors of Fig. 6 on the AWGN channel. The tests measure; the values read
//! on the figure, their tolerance and the convention of the SNR (the figure is
//! reproduced 3 dB to the left of its axis) are in the card
//! `validation/cards/vandebeek1997.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::awgn::add_noise;
use comrs::hardware::oscillator::frequency_offset;
use comrs::math::{bisect, power};
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::ofdm::sync::{cfo_snr_bound, cp_ml};
use serde_json::json;
use std::f64::consts::TAU;

#[test]
fn van_de_beek_1997_eq_2() {
    let mut card = Card::load("vandebeek1997").unwrap().part(&json!({"part": "bound"})).unwrap();
    // without noise, the offset at which the SNR after the DFT falls to 30 dB
    let eps = bisect(|e| lin2db(cfo_snr_bound(f64::INFINITY, e)) > 30.0, 1e-4, 0.4, 100);
    card.check(&json!({"part": "bound", "quantity": "offset_for_30_db"}), eps);
    card.finish().unwrap();
}

/// The mean squared errors (timing in samples², offset in subcarrier spacings²) of
/// `trials` windows of 2N + L samples, each holding one whole symbol of a QPSK stream.
fn mse(cp: usize, snr_db: f64, trials: usize, rng: &mut Rng) -> (f64, f64) {
    let ofdm = Ofdm { oversampling: 1, n_cp: cp, ..Ofdm::new(256, 256).unwrap() };
    let (n, eps) = (ofdm.fft_len(), 0.25);
    let block = ofdm.block_len();
    let (mut et, mut ef) = (0.0, 0.0);
    for _ in 0..trials {
        // three symbols; the window starts theta samples before the prefix of the second
        let data: Vec<C64> = (0..3 * ofdm.n_active).map(|_| comrs::mapping::source::qpsk(rng)).collect();
        let mut s = ofdm.modulate(&data).unwrap();
        frequency_offset(&mut s, TAU * eps / n as f64, false);
        // the SNR of the figure plus 3.01 dB: the convention that reproduces it (card)
        let snr = 2.0 * db2lin(snr_db);
        let noise_var = power(&s) / snr;
        add_noise(&mut s, noise_var, rng);
        let theta = rng.below(n + 1);
        let start = block - theta;
        let e = cp_ml(&s[start..start + 2 * n + cp], &ofdm, snr).unwrap();
        et += (e.timing as f64 - theta as f64).powi(2);
        ef += (e.cfo - eps).powi(2);
    }
    (et / trials as f64, ef / trials as f64)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn van_de_beek_1997_fig_6() {
    let mut rng = Rng::new(1997);
    let mut measured = std::collections::BTreeMap::new();
    for key in [(15, 10), (15, 15), (15, 20), (8, 10), (8, 20)] {
        let (t, f) = mse(key.0, f64::from(key.1), 4000, &mut rng);
        println!("L = {}, {} dB: timing {t:.3e}, frequency {f:.3e}", key.0, key.1);
        measured.insert(key, (t, f));
    }
    let load = |part: &str| Card::load("vandebeek1997").unwrap().part(&json!({ "part": part })).unwrap();
    let mut card = load("timing");
    for (cp, snr) in [(15, 10), (15, 15), (15, 20)] {
        card.check(&json!({"part": "timing", "cp": cp, "snr_db": snr}), measured[&(cp, snr)].0);
    }
    card.finish().unwrap();
    let mut card = load("frequency");
    for (cp, snr) in [(15, 10), (15, 20), (8, 10), (8, 20)] {
        card.check(&json!({"part": "frequency", "cp": cp, "snr_db": snr}), measured[&(cp, snr)].1);
    }
    card.finish().unwrap();
}

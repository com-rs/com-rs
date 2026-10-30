//! The timing from the cyclic prefix (`comrs::waveform::ofdm::sync::cp_ml`) against J.-J.
//! van de Beek, M. Sandell, M. Isaksson and P. O. Börjesson, "Low-complex frame
//! synchronization in OFDM systems," *Proc. IEEE ICUPC*, 982–986, 1995, Fig. 7 (AWGN,
//! N = 1024, a prefix of 128 samples), whose SNR is stated as E|s|²/E|n|²: the
//! convention that settles the 3 dB of the journal version (card `vandebeek1997`). The
//! test measures; the values read on the figure are in the card
//! `validation/cards/vandebeek1995.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::awgn::add_noise;
use comrs::math::power;
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::ofdm::sync::cp_ml;
use serde_json::json;

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn van_de_beek_1995_fig_7() {
    let ofdm = Ofdm { oversampling: 1, n_cp: 128, ..Ofdm::new(1024, 1024).unwrap() };
    let (n, block) = (ofdm.fft_len(), ofdm.block_len());
    let mut rng = Rng::new(1995);
    let mut card = Card::load("vandebeek1995").unwrap();
    for snr_db in [5, 10, 15, 20] {
        let snr = db2lin(f64::from(snr_db));
        let trials = 4000;
        let mut sq = 0.0;
        for _ in 0..trials {
            let data: Vec<C64> = (0..3 * ofdm.n_active).map(|_| rng.qpsk()).collect();
            let mut s = ofdm.modulate(&data).unwrap();
            let noise_var = power(&s) / snr;
            add_noise(&mut s, noise_var, &mut rng);
            let theta = rng.below(n + 1);
            let start = block - theta;
            let e = cp_ml(&s[start..start + 2 * n + 128], &ofdm, snr).unwrap();
            sq += (e.timing as f64 - theta as f64).powi(2);
        }
        card.check(&json!({ "snr_db": snr_db }), sq / f64::from(trials));
    }
    card.finish().unwrap();
}

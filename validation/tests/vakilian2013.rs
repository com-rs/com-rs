//! The spectrum of UFMC (`comrs::waveform::ufmc`) against V. Vakilian, T. Wild, F.
//! Schaich, S. ten Brink and J.-F. Frigon, "Universal-filtered multi-carrier technique for
//! wireless systems beyond LTE," *Proc. IEEE Globecom Workshops*, 223–228, 2013, Fig. 3(b):
//! one resource block of 12 QPSK subcarriers on a grid of $N = 128$, the Dolph–Chebyshev
//! filter of $L = 16$ taps and 120 dB of attenuation (Sec. IV), against the same block in
//! OFDM without prefix. The spectra are averaged over 2000 symbols, with a resolution of
//! 1/16 subcarrier, normalised to the band, and their envelope is read past the last
//! subcarrier. The test measures; the values read on the figure, their tolerances and the
//! difference found (the skirt of UFMC 5 to 7 dB above the figure) are in the card
//! `validation/cards/vakilian2013_fig3b.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::dsp::spectrum::periodogram;
use comrs::mapping::source;
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::ufmc::Ufmc;
use serde_json::json;

const N: usize = 128;
const Q: usize = 12;
const PAD: usize = 16;

/// The averaged spectrum (dB, normalised to the band) against the frequency in subcarriers:
/// the periodogram of the symbols zero-padded to `N · PAD` samples, at the rate of `N`
/// samples per symbol (one subcarrier per unit of frequency).
fn spectrum(modulate: impl Fn(&[C64]) -> Vec<C64>, rng: &mut Rng) -> Vec<(f64, f64)> {
    let qpsk = Constellation::psk(4).unwrap().normalized();
    let mut padded = Vec::with_capacity(2000 * N * PAD);
    for _ in 0..2000 {
        let (_, data) = source::symbols(&qpsk, Q, None, rng).unwrap();
        let mut s = modulate(&data);
        s.resize(N * PAD, C64::new(0.0, 0.0));
        padded.extend(s);
    }
    let (freqs, density) = periodogram(&padded, N as f64, N * PAD).unwrap();
    let band: Vec<f64> = freqs.iter().zip(&density).filter(|(f, _)| f.abs() < 5.0).map(|(_, p)| *p).collect();
    let mean = band.iter().sum::<f64>() / band.len() as f64;
    freqs.iter().zip(&density).map(|(f, p)| (*f, lin2db(p / mean))).collect()
}

/// The envelope `d` subcarriers past the band (the last subcarrier at ±5.5).
fn envelope(sp: &[(f64, f64)], d: f64) -> f64 {
    sp.iter()
        .filter(|(f, _)| f.abs() > 5.5 + d - 0.5 && f.abs() <= 5.5 + d + 0.5)
        .map(|(_, p)| *p)
        .fold(f64::MIN, f64::max)
}

#[test]
fn vakilian_2013_fig_3b() {
    let mut rng = Rng::new(2013);
    let mut card = Card::load("vakilian2013_fig3b").unwrap();
    let ufmc = Ufmc::new(N, 1, Q, 16, 120.0).unwrap();
    let ofdm = Ofdm { oversampling: 1, n_cp: 0, ..Ofdm::new(N, Q).unwrap() };
    let u = spectrum(|d| ufmc.modulate(d).unwrap(), &mut rng);
    let o = spectrum(|d| ofdm.modulate(d).unwrap(), &mut rng);
    for d in [20, 40] {
        card.check(
            &json!({"waveform": "ofdm", "metric": "envelope_db", "subcarriers_away": d}),
            envelope(&o, f64::from(d)),
        );
    }
    let first_below = (1..40).find(|d| envelope(&u, f64::from(*d)) < -60.0).unwrap();
    card.check(&json!({"waveform": "ufmc", "metric": "first_below_minus_60_db"}), f64::from(first_below));
    card.check(
        &json!({"waveform": "ufmc", "metric": "below_ofdm_db", "subcarriers_away": 20}),
        envelope(&o, 20.0) - envelope(&u, 20.0),
    );
    card.finish().unwrap();
}

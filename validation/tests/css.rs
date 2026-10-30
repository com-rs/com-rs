//! Chirp spread spectrum (`comrs::waveform::css`, the modulation of LoRa) against the
//! literature, by simulation:
//!
//! - L. Vangelista, "Frequency shift chirp modulation: the LoRa modulation," *IEEE
//!   Signal Process. Lett.* 24(12), 2017
//!   (`references/modulation/2017_vangelista_frequency-shift-chirp-modulation-lora_spl.pdf`), Fig.
//!   1, SF 7 over the channel √0.8 δ[n] + √0.2 δ[n - 1]: the FSK points of the figure
//!   (read 1.2e-2 at -5 dB, 1.5e-3 at -3 dB) are reproduced within 30 %; the simulated
//!   FSCM is 1.4 to 2.2 times above the points read (2.4e-2, 4e-3, 6.5e-4, 5e-5 at -9,
//!   -7, -5, -3 dB), about 0.5 dB, with or without the tail of the previous symbol and
//!   with the waveform of the letter or that of LoRa, while its AWGN curves agree with
//!   the theory (`css::theory::literature`); FSCM beats FSK by more than 2 dB, as in
//!   the figure;
//! - O. Afisiadis, M. Cotting, A. Burg and A. Balatsoukas-Stimming, "On the error rate of
//!   the LoRa modulation with interference," *IEEE Trans. Wireless Commun.* 19(2), 2020
//!   (`references/modulation/2020_afisiadis_lora-error-rate-interference_twc_arxiv.pdf`): Fig. 2
//!   (the simulated SER for SF 7 to 12 against the exact eq. (16)) and Fig. 7 (an
//!   interferer of -3 dB, neither chip- nor phase-aligned: the approximation (63) with
//!   the strongest bin at $[-\lfloor\tau\rceil]_{M}$ against the simulation; the
//!   chip- and phase-aligned model is pessimistic, as in the figure);
//! - M. Chiani and A. Elzanaty, "On the LoRa modulation for IoT: waveform properties and
//!   spectral analysis," *IEEE Internet Things J.* 6(5), 2019
//!   (`references/modulation/2019_chiani_lora-waveform-spectral-analysis_iotj_arxiv.pdf`), Table I
//!   for M = 2^10 and 2^12 (the 99 % band 0.990 B and 0.986 B, the maximum
//!   cross-correlation 0.0075 for 2^12).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use comrs::channel::awgn::add_noise;
use comrs::dsp::filter::convolve_full;
use comrs::metrics::errors::symbol_errors;
use comrs::prelude::*;
use comrs::waveform::css::{Css, Waveform, theory};

/// SER of `css` through the taps `h` (applied to the stream) at the SNR per sample `snr_db`.
fn ser_channel(css: &Css, h: &[f64], snr_db: f64, n_sym: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let m = css.m();
    let s: Vec<usize> = (0..n_sym).map(|_| rng.below(m)).collect();
    let x = css.modulate(&s).unwrap();
    // the causal filter, then the noise
    let hc: Vec<C64> = h.iter().map(|&v| C64::new(v, 0.0)).collect();
    let mut y = convolve_full(&hc, &x);
    y.truncate(x.len());
    add_noise(&mut y, comrs::math::db2lin(-snr_db), &mut rng);
    symbol_errors(&css.demodulate(&y).unwrap(), &s).unwrap() as f64 / n_sym as f64
}

/// SER under a same-SF interferer of power `pi_db`, random offsets (integer and phase-aligned if `aligned`).
fn ser_interferer(sf: u32, snr_db: f64, pi_db: f64, aligned: bool, n_sym: usize, seed: u64) -> f64 {
    let css = Css::lora(sf);
    let m = css.m();
    let mut rng = Rng::new(seed);
    let noise_var = 1.0 / comrs::math::db2lin(snr_db);
    let a = comrs::math::db2lin(pi_db).sqrt();
    let mut errors = 0;
    for _ in 0..n_sym {
        let s = rng.below(m);
        let (s1, s2) = (rng.below(m), rng.below(m));
        let (tau, w) = if aligned {
            (rng.below(m) as f64, 0.0)
        } else {
            (rng.uniform() * m as f64, rng.uniform() * std::f64::consts::TAU)
        };
        let x = css.symbol(s).unwrap();
        let xi = css.interferer(s1, s2, tau).unwrap();
        let ph = comrs::math::cis(rng.uniform() * std::f64::consts::TAU);
        let mut y: Vec<C64> = x.iter().zip(&xi).map(|(v, vi)| ph * (v + vi * comrs::math::cis(w) * a)).collect();
        add_noise(&mut y, noise_var, &mut rng);
        if css.demodulate(&y).unwrap()[0] != s {
            errors += 1;
        }
    }
    errors as f64 / n_sym as f64
}

/// The BER of the symbols of `css` (all wrong symbols equally likely).
fn ber_of(ser: f64, m: usize) -> f64 {
    ser * m as f64 / (2.0 * (m as f64 - 1.0))
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn vangelista2017_fig1_frequency_selective() {
    let h = [0.8f64.sqrt(), 0.2f64.sqrt()];
    let fsk = Css { sf: 7, oversampling: 1, waveform: Waveform::Fsk };
    for (snr, read) in [(-5.0, 1.2e-2), (-3.0, 1.5e-3)] {
        let b = ber_of(ser_channel(&fsk, &h, snr, 40_000, 1), 128);
        assert!((b / read - 1.0).abs() < 0.3, "FSK {snr} dB: {b} vs {read}");
    }
    for waveform in [Waveform::Vangelista, Waveform::LoRa] {
        let css = Css { sf: 7, oversampling: 1, waveform };
        for (snr, read) in [(-9.0, 2.4e-2), (-7.0, 4e-3), (-5.0, 6.5e-4)] {
            let b = ber_of(ser_channel(&css, &h, snr, 60_000, 2), 128);
            assert!((1.2..2.5).contains(&(b / read)), "{waveform:?} {snr} dB: {b} vs {read}");
            // FSCM ahead of FSK by more than 2 dB
            let f = ber_of(ser_channel(&fsk, &h, snr + 2.0, 60_000, 3), 128);
            assert!(b < f, "{waveform:?} {snr} dB: {b} vs FSK {f} at +2 dB");
        }
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn afisiadis2020_fig2_monte_carlo() {
    for sf in 7u32..=12 {
        // the SNR of a SER of 1e-2: 200 errors expected
        let snr = comrs::math::bisect(|s| theory::ser(sf, s).unwrap() > 1e-2, -35.0, 0.0, 60);
        let css = Css::lora(sf);
        let sim = ser_channel(&css, &[1.0], snr, 20_000, u64::from(sf));
        let want = theory::ser(sf, snr).unwrap();
        assert!((sim / want - 1.0).abs() < 0.15, "SF {sf} {snr} dB: {sim} vs {want}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn afisiadis2020_fig7_interference() {
    for (sf, snr) in [(9u32, -14.0), (9, -12.0), (9, -10.0), (10, -16.0), (10, -14.0), (11, -17.0)] {
        let n = 40_000;
        let sim = ser_interferer(sf, snr, -3.0, false, n, 2);
        let approx = theory::ser_interference(sf, snr, -3.0, 0.2).unwrap();
        assert!(sim * n as f64 > 40.0, "SF {sf} {snr} dB: {sim}");
        assert!((approx / sim - 1.0).abs() < 0.35, "SF {sf} {snr} dB: approximation {approx} vs simulation {sim}");
        // the chip- and phase-aligned model overestimates the SER
        let aligned = ser_interferer(sf, snr, -3.0, true, n, 3);
        assert!(aligned > 1.3 * sim, "SF {sf} {snr} dB: aligned {aligned} vs {sim}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn chiani2019_table1_large_spreading_factors() {
    for (sf, b99) in [(10u32, 0.990), (12, 0.986)] {
        let b = theory::occupied_bandwidth(sf, 0.99, 0.6).unwrap();
        assert!((b / b99 - 1.0).abs() < 0.005, "SF {sf}: {b}");
    }
    let c = theory::max_cross_correlation(12).unwrap();
    assert!((c - 0.0075).abs() < 2e-4, "{c}");
}

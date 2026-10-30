//! The spectrum of UFMC (`comrs::waveform::ufmc`) against V. Vakilian, T. Wild, F.
//! Schaich, S. ten Brink and J.-F. Frigon, "Universal-filtered multi-carrier technique for
//! wireless systems beyond LTE," *Proc. IEEE Globecom Workshops*, 223–228, 2013, Fig. 3(b):
//! one resource block of 12 QPSK subcarriers on a grid of $N = 128$, the Dolph–Chebyshev
//! filter of $L = 16$ taps and 120 dB of attenuation (Sec. IV), against the same block in
//! OFDM without prefix. The spectra are averaged over 2000 symbols, with a resolution of
//! 1/16 subcarrier, normalised to the band, and their envelope is read past the last
//! subcarrier:
//!
//! - OFDM stays near −26 dB 20 subcarriers away and −30 dB at 40 and beyond, as on the
//!   figure (here −26.3, −30.3; within 2 dB);
//! - UFMC reaches −60 dB 17 to 18 subcarriers away on the figure, here 19 (between 17 and
//!   20 checked), and lies more than 35 dB below OFDM 20 subcarriers away;
//! - between, the skirt of UFMC is 5 to 7 dB above the figure (−33.7 dB against about
//!   −40 at 10 subcarriers): the figure shows one realisation, and the filter it used is
//!   not given beyond its length and attenuation.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::math::fft::fft;
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::ufmc::Ufmc;

const N: usize = 128;
const Q: usize = 12;
const PAD: usize = 16;

/// The averaged spectrum (dB, normalised to the band) against the frequency in subcarriers.
fn spectrum(modulate: impl Fn(&[C64]) -> Vec<C64>, rng: &mut Rng) -> Vec<(f64, f64)> {
    let qpsk = Constellation::psk(4).unwrap().normalized();
    let mut acc = vec![0.0; N * PAD];
    for _ in 0..2000 {
        let data: Vec<C64> = (0..Q).map(|_| qpsk.points()[rng.below(4)]).collect();
        let mut s = modulate(&data);
        s.resize(N * PAD, C64::new(0.0, 0.0));
        for (a, v) in acc.iter_mut().zip(fft(&s)) {
            *a += v.norm_sqr();
        }
    }
    let freq = |k: usize| {
        let k = k as f64 / PAD as f64;
        if k >= N as f64 / 2.0 { k - N as f64 } else { k }
    };
    let band: Vec<f64> = (0..N * PAD).filter(|k| freq(*k).abs() < 5.0).map(|k| acc[k]).collect();
    let mean = band.iter().sum::<f64>() / band.len() as f64;
    (0..N * PAD).map(|k| (freq(k), lin2db(acc[k] / mean))).collect()
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
    let ufmc = Ufmc::new(N, 1, Q, 16, 120.0).unwrap();
    let ofdm = Ofdm { oversampling: 1, n_cp: 0, ..Ofdm::new(N, Q).unwrap() };
    let u = spectrum(|d| ufmc.modulate(d).unwrap(), &mut rng);
    let o = spectrum(|d| ofdm.modulate(d).unwrap(), &mut rng);
    assert!((envelope(&o, 20.0) + 26.0).abs() < 2.0 && (envelope(&o, 40.0) + 30.0).abs() < 2.0);
    let first_below = (1..40).find(|d| envelope(&u, f64::from(*d)) < -60.0).unwrap();
    assert!((17..=20).contains(&first_below), "UFMC below -60 dB from {first_below} subcarriers");
    assert!(envelope(&u, 20.0) < envelope(&o, 20.0) - 35.0);
}

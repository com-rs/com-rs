//! The comb-type pilot interpolations of S. Coleri, M. Ergen, A. Puri and A. Bahai,
//! "Channel estimation techniques based on pilot arrangement in OFDM systems," *IEEE
//! Trans. Broadcast.* 48(3), 2002
//! (`references/2002_coleri_ofdm-pilot-channel-estimation_tbc.pdf`), Sec. V and VI
//! (`comrs::waveform::ofdm::estimation::FrequencyInterpolation`): LS estimates on a comb
//! of one pilot in $L$ over 128 subcarriers of a 1024-point FFT, then linear,
//! second-order (eq. (19)), cubic spline, low-pass (MATLAB `interp`) and time-domain
//! (eqs. (20)–(22)) interpolation, with Rayleigh taps on the profiles of channel 1 (eq.
//! (23), the ATTC ensemble E) and channel 2 (Table II, a simplified DVB-T channel).
//!
//! The paper ranks them, best first: low-pass, spline, time domain, second order,
//! linear. Its simulation parameters are not consistent (Table I: 128 active carriers
//! of a 1024-point FFT at 44.1 kHz cannot span 17.5 kHz, and 70 Hz of Doppler over a
//! symbol of 1280 samples, 29 ms, decorrelates the channel within the symbol), and with
//! $L = 8$ a comb samples delays below $1024/8 = 128$ samples only, while channel 1 has
//! a path at 137 (and the band of MATLAB `interp`, `alpha` = 0.5, ends at 32). The mean
//! square errors computed here (2 000 channels, the data subcarriers) show:
//!
//! - with the parameters of the paper ($L = 8$, 40 dB), linear interpolation is the
//!   best on both channels and low-pass the worst (channel 1: linear -15.3 dB, second
//!   order -13.8, time domain -13.7, spline -11.0, low-pass -5.8): the ranking of the
//!   paper is not reproduced;
//! - it is when the comb oversamples the channel: channel 1 with $L = 2$ at 40 dB gives
//!   low-pass -36.3 dB, spline -34.2, second order -31.1, linear -29.3;
//! - at 20 dB the order reverses again (linear -22.1 dB, low-pass -16.7 with $L = 2$):
//!   the higher-order interpolators do not average the noise of the pilots;
//! - time-domain interpolation treats the pilot band as periodic: over the whole
//!   1024-subcarrier band, with the delays of channel 2 inside the window of the comb,
//!   it is exact up to the noise (-40 dB), the best by 20 dB; on a 128-subcarrier
//!   part of the band the edges make it the second worst.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use comrs::prelude::*;
use comrs::waveform::ofdm::estimation::{FrequencyInterpolation, estimate};
use comrs::waveform::ofdm::grid::{PilotPattern, ResourceGrid};

/// Eq. (23): delays (samples of the 1024-point FFT) and amplitudes.
const CHANNEL_1: [(usize, f64); 6] = [(0, 1.0), (2, 0.3162), (17, 0.1995), (36, 0.1296), (75, 0.1), (137, 0.1)];
/// Table II (the phases are those of one realisation; the taps are drawn here).
const CHANNEL_2: [(usize, f64); 12] = [
    (0, 0.2478),
    (1, 0.1287),
    (3, 0.3088),
    (4, 0.4252),
    (5, 0.49),
    (7, 0.0365),
    (8, 0.1197),
    (12, 0.1948),
    (17, 0.4187),
    (24, 0.317),
    (29, 0.2055),
    (49, 0.1846),
];

/// Linear, second order, spline, low-pass, time domain.
const METHODS: [FrequencyInterpolation; 5] = [
    FrequencyInterpolation::Linear,
    FrequencyInterpolation::SecondOrder,
    FrequencyInterpolation::Spline,
    FrequencyInterpolation::LowPass { half: 4, alpha: 0.5 },
    FrequencyInterpolation::TimeDomain,
];

/// The MSE (dB) of every method over the data subcarriers of the first `band`
/// subcarriers of a 1024-point FFT, a comb of one pilot in `spacing`, Rayleigh taps.
fn mse_db(profile: &[(usize, f64)], band: usize, spacing: usize, snr_db: f64, trials: usize) -> Vec<f64> {
    let grid = ResourceGrid { n_subcarriers: band, n_symbols: 1, pattern: PilotPattern::Comb { spacing }, seed: 1 };
    let total: f64 = profile.iter().map(|p| p.1 * p.1).sum();
    let sigma2 = 10f64.powf(-snr_db / 10.0);
    let tx = grid.map(&vec![C64::new(1.0, 0.0); grid.n_data()]).unwrap();
    let mut rng = Rng::new(5);
    let mut acc = [0.0; METHODS.len()];
    for _ in 0..trials {
        let taps: Vec<C64> = profile.iter().map(|p| rng.complex_gaussian() * (p.1 / total.sqrt())).collect();
        let h: Vec<C64> = (0..band)
            .map(|k| {
                let phase = |d: usize| comrs::math::cis(-std::f64::consts::TAU * (k * d) as f64 / 1024.0);
                profile.iter().zip(&taps).map(|(p, a)| a * phase(p.0)).sum()
            })
            .collect();
        let rx: Vec<C64> = tx.iter().zip(&h).map(|(x, g)| x * g + rng.complex_gaussian() * sigma2.sqrt()).collect();
        for (a, m) in acc.iter_mut().zip(&METHODS) {
            let e = estimate(&grid, &rx, sigma2, m).unwrap();
            let data = (0..band).filter(|k| k % spacing != 0);
            *a += data.map(|k| (e[k] - h[k]).norm_sqr()).sum::<f64>() / (band - band / spacing) as f64;
        }
    }
    acc.iter().map(|a| 10.0 * (a / trials as f64).log10()).collect()
}

fn argmin(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] < v[b] { i } else { b })
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn coleri_2002_interpolation_ranking() {
    // the parameters of the paper: linear first, low-pass last
    for ch in [&CHANNEL_1[..], &CHANNEL_2[..]] {
        let m = mse_db(ch, 128, 8, 40.0, 2000);
        assert_eq!(argmin(&m), 0, "{m:?}");
        assert!(m[3] > m[0] + 5.0 && m.iter().all(|&v| v <= m[3]), "{m:?}");
    }
    // an oversampled comb: low-pass < spline < second order < linear, as in the paper
    let m = mse_db(&CHANNEL_1, 128, 2, 40.0, 2000);
    assert!(m[3] < m[2] && m[2] < m[1] && m[1] < m[0], "{m:?}");
    // at 20 dB the noise of the pilots dominates: linear first again
    let m = mse_db(&CHANNEL_1, 128, 2, 20.0, 2000);
    assert_eq!(argmin(&m), 0, "{m:?}");
    // time domain is exact over the whole periodic band when the delays fit the comb
    let m = mse_db(&CHANNEL_2, 1024, 8, 40.0, 200);
    assert_eq!(argmin(&m), 4, "{m:?}");
    assert!((m[4] + 40.0).abs() < 1.0 && m[4] < m[0] - 15.0, "{m:?}");
}

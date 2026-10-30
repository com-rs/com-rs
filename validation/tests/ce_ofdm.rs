//! Constant-envelope OFDM (`comrs::waveform::ce_ofdm`) against the literature, by
//! simulation:
//!
//! - J. Guerreiro, R. Dinis and P. Montezuma, "On the detection of CE-OFDM signals,"
//!   *IEEE Commun. Lett.* 20(11), 2165–2168, 2016
//!   (`references/multicarrier/2016_guerreiro_ce-ofdm-detection_comml.pdf`), Sec. III-A and Fig. 1:
//!   the phase detector (no unwrapping) with QPSK, N = 512 and M = 4 needs 10.8 dB at a
//!   BER of 1e-3 for 2πh = 0.75 and 11.6 dB for 2πh = 1.0, and floors near 1.4e-2 at
//!   2πh = 1.25. The simulation needs 0.2 and 0.4 dB more and floors at 1.5e-2;
//! - M. J. López Morales, R. Dinis and A. García Armada, "Near-optimal detection of
//!   CE-OFDM signals with high power efficiency via GAMP-based receivers," *Proc. IEEE
//!   Globecom Workshops*, 2022 (`references/multicarrier/2022_lopez_ce-ofdm-gamp-receiver_globecomw.pdf`),
//!   Fig. 3: at 2πb = 1.4 GAMP approaches the optimum BER $Q(\sqrt{(2\pi b)^{2}
//!   2E_{b}/N_{0}})$ of Guerreiro et al. (eq. (8)) while the phase detector stays above
//!   4e-2. The paper's GAMP is 2.4 dB from that optimum at 1e-3; the library's, which
//!   estimates the real coordinates of the symbols, 0.5 dB. The $E_{b}/N_{0}$ axis of
//!   Fig. 3 is not that of Guerreiro et al.: its phase detector at 2πb = 0.5 sits 6 dB
//!   (the oversampling factor M = 4) left of Fig. 1 of the 2016 letter and of the linear
//!   theory, and its optimum curves are shifted by 6 dB at 2πb = 0.5 and 2.5 dB at 1.4,
//!   so only the gaps between curves are compared;
//! - R. A. Pacheco and D. Hatzinakos, "Error rate analysis of phase-modulated OFDM
//!   (OFDM-PM) in AWGN channels," *Proc. IEEE ICASSP*, 2006
//!   (`references/multicarrier/2006_pacheco_ofdm-pm-error-rate_icassp.pdf`), Fig. 1: the 99 %
//!   bandwidth of simulated OFDM-PM (104 subcarriers) against the analysis, here
//!   `theory::occupied_bandwidth`;
//! - A. U. Ahmed and J. R. Zeidler, "Novel low-complexity receivers for constant envelope
//!   OFDM," *IEEE Trans. Signal Process.* 63(17), 4572–4582, 2015
//!   (`references/multicarrier/2015_ahmed_ce-ofdm-low-complexity-receivers_tsp.pdf`), Sec. V and Fig. 6
//!   (N = 64, 2X sampling): the basic linear receiver is within 0.5 dB of the arctangent
//!   receiver at 2πh = 0.3 (about 0.4 dB in the figure); the enhanced receiver beats the basic one at 2πh = 0.5 and 0.7; at 0.7 the
//!   arctangent receiver is better at low BERs (down to 1e-4; the wraps of the phase floor
//!   it near 1e-6, below the figure) and the enhanced one at high BERs.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::mapping::constellation::Constellation;
use comrs::prelude::*;
use comrs::waveform::ce_ofdm::{CeOfdm, Receiver, Unwrap, gamp::Gamp, theory};
use std::f64::consts::TAU;

/// A detector: the link, the received samples and the noise variance to the estimates.
type Detector<'a> = &'a dyn Fn(&CeOfdm, &[C64], f64) -> Vec<C64>;

/// The BER of QPSK over CE-OFDM with `blocks` blocks at `E_b/N_0` (dB), for each of the
/// detectors (given the link, the received samples and the noise variance).
fn ber(ce: &CeOfdm, ebn0_db: f64, blocks: usize, seed: u64, detectors: &[Detector]) -> Vec<f64> {
    let c = Constellation::qam(4).unwrap().normalized();
    let mut rng = Rng::new(seed);
    let (_, d) = comrs::mapping::source::symbols(&c, ce.n_data * blocks, None, &mut rng).unwrap();
    let sigma2 = ce.noise_variance(ebn0_db + comrs::math::lin2db(2.0));
    let r: Vec<C64> =
        ce.modulate(&d).unwrap().into_iter().map(|v| v + rng.complex_gaussian() * sigma2.sqrt()).collect();
    let truth = c.decide(&d);
    detectors
        .iter()
        .map(|f| {
            let e = f(ce, &r, sigma2);
            let wrong: usize = c.decide(&e).iter().zip(&truth).map(|(a, b)| (a ^ b).count_ones() as usize).sum();
            wrong as f64 / (2.0 * d.len() as f64)
        })
        .collect()
}

fn phase_detector(ce: &CeOfdm, r: &[C64], _: f64) -> Vec<C64> {
    ce.demodulate(r, Unwrap::None).unwrap()
}

/// The `E_b/N_0` (dB) where the BER crosses `target`, interpolated in log BER between two
/// simulated points.
fn crossing(points: &[(f64, f64)], target: f64) -> f64 {
    let w = points.windows(2).find(|w| w[0].1 >= target && w[1].1 < target).expect("the points bracket the target");
    let (a, b) = (w[0], w[1]);
    a.0 + (b.0 - a.0) * (a.1.ln() - target.ln()) / (a.1.ln() - b.1.ln())
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn guerreiro2016_fig1_phase_detector() {
    // N = 512 subcarriers with Hermitian symmetry: K = 255 data subcarriers, M = J = 4
    let link = |two_pi_h: f64| CeOfdm { n_data: 255, oversampling: 4, h: two_pi_h / TAU, n_cp: 0 };
    for (two_pi_h, paper, grid) in [(0.75, 10.8, [10.0, 10.8, 11.6, 12.4]), (1.0, 11.6, [11.0, 11.8, 12.6, 13.4])] {
        let ce = link(two_pi_h);
        let points: Vec<(f64, f64)> = grid.iter().map(|&e| (e, ber(&ce, e, 2000, 11, &[&phase_detector])[0])).collect();
        let need = crossing(&points, 1e-3);
        assert!((need - paper).abs() < 0.5, "2πh = {two_pi_h}: {need:.2} dB vs {paper} dB ({points:?})");
        // the phase detector loses against the optimum asymptotic BER (blue curves)
        let opt = comrs::math::bisect(|e| theory::ber_optimum(two_pi_h, e) > 1e-3, 0.0, 20.0, 60);
        assert!(need - opt > 1.0, "2πh = {two_pi_h}: {need} vs the optimum {opt}");
    }
    // the floor at 2πh = 1.25, read near 1.4e-2 from 16 to 38 dB
    let ce = link(1.25);
    for e in [20.0, 30.0] {
        let b = ber(&ce, e, 400, 12, &[&phase_detector])[0];
        assert!((1.0e-2..2.0e-2).contains(&b), "{e} dB: {b}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn lopez2022_fig3_gamp_reaches_the_optimum() {
    let ce = CeOfdm { n_data: 255, oversampling: 4, h: 1.4 / TAU, n_cp: 0 };
    let gamp = |ce: &CeOfdm, r: &[C64], s2: f64| {
        Gamp::default().detect(ce, r, &Constellation::qam(4).unwrap().normalized(), s2).unwrap()
    };
    let mut points = Vec::new();
    for e in [2.0, 3.0, 4.0, 5.0] {
        let b = ber(&ce, e, 200, 5, &[&gamp, &phase_detector]);
        // the phase detector cannot follow the phase beyond ±π
        assert!(b[1] > 4e-2, "{e} dB: phase detector {}", b[1]);
        points.push((e, b[0]));
    }
    let need = crossing(&points, 1e-3);
    let opt = comrs::math::bisect(|e| theory::ber_optimum(1.4, e) > 1e-3, 0.0, 20.0, 60);
    assert!(need - opt < 1.0 && need - opt > -0.2, "GAMP {need:.2} dB vs the optimum {opt:.2} dB ({points:?})");
    // the floor of the phase detector at a high SNR (above 4e-2 up to 14 dB in Fig. 3)
    let b = ber(&ce, 14.0, 100, 6, &[&phase_detector])[0];
    assert!(b > 4e-2, "14 dB: {b}");
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn pacheco2006_fig1_simulated_bandwidth() {
    // 104 subcarriers as the paper, oversampled 8 times; the mean periodogram of whole
    // blocks (no prefix: every block is periodic), the carrier line removed
    let k = 104;
    let c = Constellation::qam(4).unwrap().normalized();
    for sigma in [0.5, 1.1, 2.0] {
        let ce = CeOfdm { n_data: k, oversampling: 8, h: sigma / TAU, n_cp: 0 };
        let n = ce.n_fft();
        let blocks = 1000;
        let (_, d) = comrs::mapping::source::symbols(&c, k * blocks, None, &mut Rng::new(3)).unwrap();
        let s = ce.modulate(&d).unwrap();
        let mut p = vec![0.0; n];
        for block in s.chunks(n) {
            for (acc, v) in p.iter_mut().zip(comrs::math::fft::fft(block)) {
                *acc += v.norm_sqr();
            }
        }
        // the power within ±z bins, z = 1 … n/2, without the line
        let total: f64 = p[1..].iter().sum();
        let mut inside = 0.0;
        let mut z = 0;
        while inside < 0.99 * total {
            z += 1;
            inside += p[z] + p[n - z];
        }
        let simulated = z as f64 / k as f64;
        let analysis = theory::occupied_bandwidth(sigma, 0.99).unwrap();
        assert!((simulated / analysis - 1.0).abs() < 0.04, "σ = {sigma}: {simulated} vs {analysis}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn ahmed2015_fig6_low_complexity_receivers() {
    let link = |two_pi_h: f64| CeOfdm { n_data: 32, oversampling: 2, h: two_pi_h / TAU, n_cp: 0 };
    let with = |r: Receiver| move |ce: &CeOfdm, x: &[C64], _: f64| ce.demodulate_with(x, r).unwrap();
    let (lin, enh) = (with(Receiver::Linear), with(Receiver::Enhanced));
    let detectors: [Detector; 3] = [&phase_detector, &lin, &enh];
    // 2πh = 0.3: the basic linear receiver is close to the arctangent one (eq. (20)):
    // within 0.5 dB (about 0.4 dB in Fig. 6, Bussgang's gain e^{-(2πh)²} = -0.39 dB)
    let b = ber(&link(0.3), 16.0, 20_000, 1, &detectors);
    let shifted = ber(&link(0.3), 16.5, 20_000, 1, &detectors);
    assert!(b[1] > b[0] && shifted[1] < b[0], "2πh = 0.3: {b:?} at 16 dB, {shifted:?} at 16.5 dB");
    // 2πh = 0.5 and 0.7: the enhanced receiver beats the basic one
    for (two_pi_h, e) in [(0.5, 12.0), (0.7, 11.0)] {
        let b = ber(&link(two_pi_h), e, 20_000, 2, &detectors);
        assert!(b[2] < 0.9 * b[1], "2πh = {two_pi_h}: {b:?}");
    }
    // 2πh = 0.7: the enhanced receiver is better at a high BER, the arctangent one at a
    // low BER (1e-4 at 12 dB; beyond 14 dB the wraps of the phase, not unwrapped, floor
    // the arctangent receiver near 1e-6 and the enhanced one is better again, below the
    // range of Fig. 6)
    let high = ber(&link(0.7), 4.0, 5000, 3, &detectors);
    let low = ber(&link(0.7), 12.0, 20_000, 4, &detectors);
    assert!(high[2] < high[0] && low[0] < low[2], "high BER {high:?}, low BER {low:?}");
}

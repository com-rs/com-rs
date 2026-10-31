//! The EGN model against Dar, Feder, Mecozzi and Shtaif (2014): the fourth-order noise
//! (FON) of the interfering channels, the term that makes the nonlinear interference
//! depend on the modulation format.
//!
//! Sec. 3 of the paper: five channels of Nyquist pulses (square spectrum) at 32 GBd,
//! spaced by 50 GHz, over 500 km of SMF ($D$ = 17 ps/(nm·km), $\gamma$ = 1.3 /(W·km),
//! 0.2 dB/km), one polarisation, the self-channel interference removed by ideal
//! back-propagation and the channel of interest isolated by a matched filter. After
//! 500 km the error of the GN model with QPSK is about 10, 8.6, 5.8 and 3.7 dB, and the
//! difference between QPSK and 16-QAM about 6, 4.8, 2.8 and 1.5 dB, for distributed
//! amplification and spans of 25, 50 and 100 km.
//!
//! The model: the cross-channel interference of the centre channel, the NLI of the five
//! channels less that of the centre channel alone (`GnModel::nli_psd` integrated over
//! the band by Simpson's rule), by the GN model and by the EGN model with the moments of
//! each format. `GnModel` is dual-polarisation: going from one polarisation to two
//! scales the second-order noise by 16/27 and the FON by 40/81 (Dar et al. 2013,
//! Sec. 5), so the excess $r - 1$ of the ratio EGN/GN of one polarisation is 6/5 of
//! that of two. Distributed amplification is a lossless fibre, here 50 spans of 10 km
//! without loss accumulated coherently, which is exactly 500 km and keeps the link
//! function of a span smooth on the grid. The values read and their tolerances are in
//! the card `validation/cards/dar2014_sec3.yaml`.
//!
//! The split-step simulation of Sec. 3 (`Ssfm`, one polarisation, distributed
//! amplification), the reference that decides between the paper and the EGN model: 4096
//! symbols per channel and run, 16 samples per symbol, Nyquist pulses built in the
//! frequency domain, no noise, steps of 0.5 km (0.1 km changes the result by 0.01 dB),
//! 64 runs per format. The cross-channel interference of the centre channel is the
//! difference between the received channel with and without its neighbours, after the
//! dispersion is undone and the channel isolated by its matched filter, less its
//! projection on the channel alone: the average rotation induced by the neighbours turns
//! the channel with its self-channel interference, which the projection removes with it
//! (at $-10$ dBm, subtracting the channel alone without the rotation $\theta$ leaves
//! $(e^{j\theta} - 1)$ times the self-channel interference, 3.5 dB of error on the QPSK
//! result). The power, $-16$ dBm per channel, is in the first-order regime ($-22$ and
//! $-28$ dBm give the same values). The card is `validation/cards/dar2014_ssfm.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::C64;
use comrs::channel::optical::fiber::{Amplification, FiberParams, Ssfm};
use comrs::channel::optical::gn::{FormatMoments, GnModel, WdmSpectrum};
use comrs::channel::optical::manakov::ManakovSsfm;
use comrs::core::rng::Rng;
use comrs::core::signal::DualPol;
use comrs::mapping::constellation::Constellation;
use comrs::math::fft::{fft, ifft};
use comrs::math::nlse::{NonlinearStep, StepRule};
use comrs::math::quadrature::try_simpson;
use serde_json::json;

const RS: f64 = 32e9;

fn smf(span_km: f64, n_spans: usize, alpha_db_km: f64) -> FiberParams {
    FiberParams { n_spans, span_km, d_ps_nm_km: 17.0, gamma_w_km: 1.3, alpha_db_km, ..FiberParams::default() }
}

/// $P^{-3}$ times the NLI power in the band of the centre channel of `n_channels`
/// channels spaced by 50 GHz, by Simpson's rule on 16 intervals.
fn eta(fiber: &FiberParams, n_channels: usize, egn: Option<FormatMoments>) -> f64 {
    let half = (n_channels / 2) as f64;
    let freqs = (0..n_channels).map(|k| (k as f64 - half) * 50e9).collect();
    let s = WdmSpectrum::uniform(freqs, 1.0, RS, 0.0);
    let m = GnModel { egn, ..GnModel::new(fiber.clone()) };
    try_simpson(|f| m.nli_psd(&s, f), -RS / 2.0, RS / 2.0, 16).unwrap()
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn fon_of_qpsk_and_16qam_follows_sec_3_of_dar_2014() {
    let qpsk = FormatMoments { phi: -1.0, psi: 4.0 };
    let qam16 = FormatMoments::of(Constellation::qam(16).unwrap().points(), None).unwrap();
    let mut card = Card::load("dar2014_sec3").unwrap();
    for (amplification, fiber) in [
        ("distributed", smf(10.0, 50, 0.0)),
        ("25 km", smf(25.0, 20, 0.2)),
        ("50 km", smf(50.0, 10, 0.2)),
        ("100 km", smf(100.0, 5, 0.2)),
    ] {
        let xci = |egn| eta(&fiber, 5, egn) - eta(&fiber, 1, egn);
        let gn = xci(None);
        // the ratio EGN/GN of one polarisation
        let ratio = |fm| 1.0 + 6.0 / 5.0 * (xci(Some(fm)) / gn - 1.0);
        let (r_qpsk, r_16) = (ratio(qpsk), ratio(qam16));
        card.check(&json!({"amplification": amplification, "quantity": "gn_error_qpsk"}), -comrs::math::lin2db(r_qpsk));
        card.check(
            &json!({"amplification": amplification, "quantity": "qpsk_vs_16qam"}),
            comrs::math::lin2db(r_16 / r_qpsk),
        );
    }
    card.finish().unwrap();
}

/// Symbols per channel and run of the split-step simulation.
const SYMBOLS: usize = 4096;
/// Samples per symbol.
const SPS: usize = 16;

/// The field (√W) of a channel of Nyquist pulses carrying `a` at `slot` × 50 GHz with
/// the power `p` (W): the spectrum of the symbols placed in the band of the channel.
fn nyquist_channel(a: &[C64], slot: i64, p: f64) -> Vec<C64> {
    let (n, l) = (SYMBOLS as i64, (SYMBOLS * SPS) as i64);
    let spectrum = fft(a);
    // 50 GHz is 50/32 of the symbol rate, an integer number of bins
    let offset = slot * n * 50 / 32;
    let mut x = vec![C64::new(0.0, 0.0); l as usize];
    for j in -n / 2..n / 2 {
        x[(j + offset).rem_euclid(l) as usize] = spectrum[j.rem_euclid(n) as usize];
    }
    let scale = p.sqrt() * SPS as f64;
    ifft(&x).into_iter().map(|v| v * scale).collect()
}

/// The symbols of the centre channel of the field `y`, normalised by the amplitude of a
/// channel of power `p`: the matched filter of a Nyquist pulse keeps the band of the
/// channel, sampled at the symbol rate.
fn matched_filter(y: &[C64], p: f64) -> Vec<C64> {
    let (n, l) = (SYMBOLS as i64, y.len() as i64);
    let spectrum = fft(y);
    let band: Vec<C64> =
        (0..n).map(|i| if i < n / 2 { i } else { i - n }).map(|j| spectrum[j.rem_euclid(l) as usize]).collect();
    let scale = 1.0 / (p.sqrt() * SPS as f64);
    ifft(&band).into_iter().map(|v| v * scale).collect()
}

/// The symbols of a format: `None` for Gaussian symbols, else the points of the
/// constellation, drawn uniformly.
fn draw(points: Option<&[C64]>, rng: &mut Rng) -> Vec<C64> {
    (0..SYMBOLS)
        .map(|_| match points {
            None => rng.complex_gaussian(),
            Some(pts) => pts[rng.below(pts.len())],
        })
        .collect()
}

/// $P^{-3}$ times the power of the cross-channel interference of the centre channel of
/// five channels in one run of the split-step simulation of `fiber` (see the module).
fn xci_run(fiber: &FiberParams, points: Option<&[C64]>, p: f64, seed: u64) -> f64 {
    let fs = RS * SPS as f64;
    let step = 0.5;
    let ssfm = Ssfm { amplification: Amplification::Ideal, ..Ssfm::new(fiber.clone(), step) };
    let steps = (fiber.span_km / step).round() as usize;
    let mut rng = Rng::new(seed);
    let symbols: Vec<Vec<C64>> = (0..5).map(|_| draw(points, &mut rng)).collect();
    let alone = nyquist_channel(&symbols[2], 0, p);
    let mut all = alone.clone();
    for (slot, s) in [(-2, &symbols[0]), (-1, &symbols[1]), (1, &symbols[3]), (2, &symbols[4])] {
        for (v, w) in all.iter_mut().zip(nyquist_channel(s, slot, p)) {
            *v += w;
        }
    }
    let received = |mut x: Vec<C64>| {
        ssfm.propagate(&mut x, fs, None).unwrap();
        // the dispersion undone by a linear back-propagation
        matched_filter(&ssfm.backpropagate_scaled(&x, fs, steps, 0.0).unwrap(), p)
    };
    let (r5, r1) = (received(all), received(alone));
    let c: C64 = r5.iter().zip(&r1).map(|(a, b)| a * b.conj()).sum::<C64>() / r1.iter().map(C64::norm_sqr).sum::<f64>();
    let var = r5.iter().zip(&r1).map(|(a, b)| (a - c * b).norm_sqr()).sum::<f64>() / SYMBOLS as f64;
    var * p / p.powi(3)
}

/// [`xci_run`] for two polarisations of power `p/2` each, propagated by the Manakov
/// solver with the nonlinear step `nonlinear`: the interference of both polarisations.
fn xci_run_dual(fiber: &FiberParams, points: Option<&[C64]>, p: f64, seed: u64, nonlinear: NonlinearStep) -> f64 {
    let fs = RS * SPS as f64;
    let step = 0.5;
    let link = ManakovSsfm {
        rule: StepRule::Fixed(step),
        nonlinear,
        amplification: Amplification::Ideal,
        ..ManakovSsfm::new(fiber.clone())
    };
    let linear = Ssfm::new(fiber.clone(), step);
    let steps = (fiber.span_km / step).round() as usize;
    let mut rng = Rng::new(seed);
    let symbols: Vec<[Vec<C64>; 2]> = (0..5).map(|_| [draw(points, &mut rng), draw(points, &mut rng)]).collect();
    let field = |slots: &[usize]| {
        let zero = vec![C64::new(0.0, 0.0); SYMBOLS * SPS];
        let mut u = DualPol { x: zero.clone(), y: zero };
        for &i in slots {
            for (pol, out) in [&mut u.x, &mut u.y].into_iter().enumerate() {
                for (v, w) in out.iter_mut().zip(nyquist_channel(&symbols[i][pol], i as i64 - 2, p / 2.0)) {
                    *v += w;
                }
            }
        }
        u
    };
    let (mut u5, mut u1) = (field(&[0, 1, 2, 3, 4]), field(&[2]));
    link.propagate(&mut u5, fs, None).unwrap();
    link.propagate(&mut u1, fs, None).unwrap();
    let mut nli = 0.0;
    for (a5, a1) in [(&u5.x, &u1.x), (&u5.y, &u1.y)] {
        let received = |x: &[C64]| matched_filter(&linear.backpropagate_scaled(x, fs, steps, 0.0).unwrap(), p / 2.0);
        let (r5, r1) = (received(a5), received(a1));
        let c: C64 =
            r5.iter().zip(&r1).map(|(a, b)| a * b.conj()).sum::<C64>() / r1.iter().map(C64::norm_sqr).sum::<f64>();
        nli += r5.iter().zip(&r1).map(|(a, b)| (a - c * b).norm_sqr()).sum::<f64>() / SYMBOLS as f64 * p / 2.0;
    }
    nli / p.powi(3)
}

/// The mean of `run(seed)` over the seeds `first..first + runs`, the runs spread over the
/// threads and summed in order, and its relative standard error.
fn mean_over_runs(first: u64, runs: u64, run: impl Fn(u64) -> f64 + Sync) -> (f64, f64) {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let seeds: Vec<u64> = (first..first + runs).collect();
    let mut values = vec![0.0; seeds.len()];
    let chunk = seeds.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_seeds, chunk_values) in seeds.chunks(chunk).zip(values.chunks_mut(chunk)) {
            let run = &run;
            scope.spawn(move || {
                for (s, v) in chunk_seeds.iter().zip(chunk_values.iter_mut()) {
                    *v = run(*s);
                }
            });
        }
    });
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sem = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
    (mean, sem / mean)
}

/// The mean of [`xci_run`] over the runs of seeds `first..first + runs`, the runs spread
/// over the threads and summed in order.
fn xci(fiber: &FiberParams, points: Option<&[C64]>, p: f64, first: u64, runs: u64) -> (f64, f64) {
    mean_over_runs(first, runs, |seed| xci_run(fiber, points, p, seed))
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_split_step_simulation_confirms_sec_3_of_dar_2014() {
    let fiber = smf(500.0, 1, 0.0);
    let p = comrs::math::dbm2w(-16.0);
    let runs = 64;
    let qpsk = Constellation::qam(4).unwrap().normalized();
    let qam16 = Constellation::qam(16).unwrap().normalized();
    let (g, eg) = xci(&fiber, None, p, 1000, runs);
    let (q, eq) = xci(&fiber, Some(qpsk.points()), p, 1000, runs);
    let (m, em) = xci(&fiber, Some(qam16.points()), p, 1000, runs);
    // the GN model is dual-polarisation: one polarisation of the same power has 27/8
    // times its NLI (the Gaussian pairings, 2 against 16/27)
    let gn = eta(&smf(10.0, 50, 0.0), 5, None) - eta(&smf(10.0, 50, 0.0), 1, None);
    let db = comrs::math::lin2db;
    println!(
        "split step: Gaussian {g:.4e} ({:.1} %), QPSK {q:.4e} ({:.1} %), 16-QAM {m:.4e} ({:.1} %); GN model {:.4e}",
        100.0 * eg,
        100.0 * eq,
        100.0 * em,
        27.0 / 8.0 * gn
    );
    let mut card = Card::load("dar2014_ssfm").unwrap();
    card.check(&json!({"quantity": "gn_error_qpsk"}), db(g / q));
    card.check(&json!({"quantity": "qpsk_vs_16qam"}), db(m / q));
    card.check(&json!({"quantity": "gaussian_against_gn_model"}), db(g / (27.0 / 8.0 * gn)));
    card.finish().unwrap();
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_manakov_simulation_follows_the_gn_and_egn_models() {
    // the link of Sec. 3 with two polarisations: the Manakov solver against GnModel, which
    // is dual-polarisation, directly
    let fiber = smf(500.0, 1, 0.0);
    let p = comrs::math::dbm2w(-16.0);
    let runs = 64;
    let qpsk = Constellation::qam(4).unwrap().normalized();
    let gn = eta(&smf(10.0, 50, 0.0), 5, None) - eta(&smf(10.0, 50, 0.0), 1, None);
    let qpsk_moments = FormatMoments { phi: -1.0, psi: 4.0 };
    let egn = eta(&smf(10.0, 50, 0.0), 5, Some(qpsk_moments)) - eta(&smf(10.0, 50, 0.0), 1, Some(qpsk_moments));
    let db = comrs::math::lin2db;
    let mut card = Card::load("dar2014_manakov").unwrap();
    for (name, nonlinear) in [("midpoint", NonlinearStep::Midpoint), ("trapezoidal", NonlinearStep::Trapezoidal)] {
        let (g, eg) = mean_over_runs(2000, runs, |seed| xci_run_dual(&fiber, None, p, seed, nonlinear));
        let (q, eq) = mean_over_runs(2000, runs, |seed| xci_run_dual(&fiber, Some(qpsk.points()), p, seed, nonlinear));
        println!(
            "{name}: Gaussian {g:.4e} ({:.1} %) against GN {gn:.4e}; QPSK {q:.4e} ({:.1} %) against EGN {egn:.4e}",
            100.0 * eg,
            100.0 * eq
        );
        card.check(&json!({"nonlinear_step": name, "quantity": "gaussian_against_gn_model"}), db(g / gn));
        card.check(&json!({"nonlinear_step": name, "quantity": "qpsk_against_egn_model"}), db(q / egn));
    }
    card.finish().unwrap();
}

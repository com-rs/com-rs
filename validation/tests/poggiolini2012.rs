//! The GN integral of `com_channel::optical::gn` (P. Poggiolini, "The GN model of
//! non-linear propagation in uncompensated coherent optical systems," *J. Lightw.
//! Technol.* 30(24), 2012) against the Manakov solver, the self-channel and the
//! cross-channel interference separately: dual-polarisation Gaussian symbols on Nyquist
//! channels of 32 GBd spaced by 50 GHz, 5 × 80 km of SMF (0.2 dB/km, 16 ps/(nm km),
//! $\gamma$ = 1.3 /(W km)) with ideal amplifiers and no noise, 4096 symbols and 16
//! samples per symbol (512 GHz: more than the $\frac{3}{2} B_{\mathrm{WDM}} + \frac{1}{2}
//! B_{\mathrm{CUT}}$ of Musetti et al. 2018, Table I, so that no four-wave mixing aliases
//! onto the centre channel), 32 runs per case, at $-4$ and 2 dBm per channel. The
//! simulated interference of the centre channel after its matched filter, less its
//! projection on the transmitted symbols, is compared with the PSD of the GN model
//! integrated over the band of the channel (what the matched filter of a Nyquist pulse
//! sees) and with its value at the centre times the symbol rate (the approximation of
//! `GnModel::snr`, locally white). The card is
//! `validation/cards/poggiolini2012_band.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use com_validation::card::Card;
use com_validation::wdm::NyquistGrid;
use comrs::C64;
use comrs::channel::optical::fiber::{Amplification, FiberParams, Ssfm};
use comrs::channel::optical::gn::{GnModel, WdmSpectrum};
use comrs::channel::optical::manakov::ManakovSsfm;
use comrs::core::rng::Rng;
use comrs::core::signal::DualPol;
use comrs::math::quadrature::try_simpson;
use serde_json::json;

const RS: f64 = 32e9;
const GRID: NyquistGrid = NyquistGrid { symbols: 4096, sps: 16, slot_bins: 4096 * 50 / 32 };

fn fiber() -> FiberParams {
    FiberParams {
        n_spans: 5,
        span_km: 80.0,
        d_ps_nm_km: 16.0,
        gamma_w_km: 1.3,
        alpha_db_km: 0.2,
        ..FiberParams::default()
    }
}

/// NLI variance / P^3 of the centre channel, `n` channels, one run.
fn run(n: usize, p: f64, seed: u64) -> f64 {
    let fs = RS * GRID.sps as f64;
    let link = ManakovSsfm { amplification: Amplification::Ideal, ..ManakovSsfm::new(fiber()) };
    let mut rng = Rng::new(seed);
    let half = (n / 2) as i64;
    let mut u = DualPol { x: vec![C64::new(0.0, 0.0); GRID.len()], y: vec![C64::new(0.0, 0.0); GRID.len()] };
    let mut centre = [Vec::new(), Vec::new()];
    for slot in -half..=half {
        for (pol, f) in [&mut u.x, &mut u.y].into_iter().enumerate() {
            let a: Vec<C64> = (0..GRID.symbols).map(|_| rng.complex_gaussian()).collect();
            for (v, w) in f.iter_mut().zip(GRID.channel(&a, slot, p / 2.0)) {
                *v += w;
            }
            if slot == 0 {
                centre[pol] = a;
            }
        }
    }
    link.propagate(&mut u, fs, None).unwrap();
    // undo the dispersion of the whole link (the ideal amplifiers restore the power)
    let lin = Ssfm::new(FiberParams { n_spans: 1, span_km: 400.0, alpha_db_km: 0.0, ..fiber() }, 400.0);
    let mut nli = 0.0;
    for (pol, x) in [&u.x, &u.y].into_iter().enumerate() {
        let r = GRID.matched_filter(&lin.backpropagate_scaled(x, fs, 1, 0.0).unwrap(), p / 2.0);
        let a = &centre[pol];
        let c: C64 = r.iter().zip(a).map(|(u, v)| u * v.conj()).sum::<C64>() / a.iter().map(C64::norm_sqr).sum::<f64>();
        nli += r.iter().zip(a).map(|(u, v)| (u - c * v).norm_sqr()).sum::<f64>() / GRID.symbols as f64 * p / 2.0;
    }
    nli / p.powi(3)
}

fn mean(n: usize, p: f64, runs: u64) -> (f64, f64) {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let seeds: Vec<u64> = (0..runs).collect();
    let mut v = vec![0.0; seeds.len()];
    let chunk = seeds.len().div_ceil(threads);
    std::thread::scope(|s| {
        for (cs, cv) in seeds.chunks(chunk).zip(v.chunks_mut(chunk)) {
            s.spawn(move || {
                for (seed, out) in cs.iter().zip(cv.iter_mut()) {
                    *out = run(n, p, 100 + seed);
                }
            });
        }
    });
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0) / v.len() as f64).sqrt();
    (m, sd / m)
}

/// GN: centre PSD × Rs and the band integral, (W/Hz)/W^3 × Hz = 1/W^2.
fn gn(n: usize) -> (f64, f64) {
    let half = (n / 2) as f64;
    let freqs = (0..n).map(|k| (k as f64 - half) * 50e9).collect();
    let s = WdmSpectrum::uniform(freqs, 1.0, RS, 0.0);
    let m = GnModel::new(fiber());
    let centre = m.nli_psd(&s, 0.0).unwrap() * RS;
    let band = try_simpson(|f| m.nli_psd(&s, f), -RS / 2.0, RS / 2.0, 32).unwrap();
    (centre, band)
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_simulated_interference_follows_the_band_integral_of_the_gn_model() {
    let db = |x: f64| 10.0 * x.log10();
    let (g1c, g1b) = gn(1);
    let (g5c, g5b) = gn(5);
    let mut card = Card::load("poggiolini2012_band").unwrap();
    card.check(&json!({"quantity": "centre_over_band", "channels": 1}), db(g1c / g1b));
    card.check(&json!({"quantity": "centre_over_band", "channels": 5}), db(g5c / g5b));
    for pdbm in [-4.0, 2.0] {
        let p = comrs::math::dbm2w(pdbm);
        let (s1, e1) = mean(1, p, 32);
        let (s5, e5) = mean(5, p, 32);
        println!(
            "{pdbm} dBm: SCI {:+.2} dB ({:.1} %), total {:+.2} dB ({:.1} %), XCI + MCI {:+.2} dB against the band integral",
            db(s1 / g1b),
            100.0 * e1,
            db(s5 / g5b),
            100.0 * e5,
            db((s5 - s1) / (g5b - g1b))
        );
        card.check(&json!({"quantity": "sci", "power_dbm": pdbm}), db(s1 / g1b));
        card.check(&json!({"quantity": "total", "power_dbm": pdbm}), db(s5 / g5b));
        card.check(&json!({"quantity": "xci_mci", "power_dbm": pdbm}), db((s5 - s1) / (g5b - g1b)));
    }
    card.finish().unwrap();
}

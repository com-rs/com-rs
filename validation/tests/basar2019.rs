//! The reconfigurable intelligent surface (`comrs::channel::ris::Ris`, the co-phasing of
//! `comrs::mimo::precoding::ris_co_phasing`, the SER by the MGF of
//! `comrs::mapping::theory::ser_mgf`) against E. Basar, M. Di Renzo, J. de Rosny, M.
//! Debbah, M.-S. Alouini and R. Zhang, "Wireless communications through reconfigurable
//! intelligent surfaces," *IEEE Access*, 2019 (arXiv 1906.09490), Sec. IV.
//!
//! - **Fig. 6**: the SER of BPSK in the Gaussian approximation of the co-phased
//!   amplitude, eq. (17), lies below its bound (18), and below the SER of BPSK over
//!   AWGN at low SNR, for $N$ = 16 and 32.
//! - **Fig. 7**: the SER of BPSK simulated through $N$ = 4 to 256 co-phased Rayleigh
//!   elements against eq. (17): "accurate for $N \ge 32$", and doubling $N$ improves
//!   the required SNR by about 6 dB in the waterfall region. Compared here as the
//!   $E_{s}/N_{0}$ at which the SER crosses $10^{-3}$.
//!
//! The values and their tolerances are in the card `validation/cards/basar2019_fig7.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::{sweep_crossing, until_errors};
use comrs::channel::ris::Ris;
use comrs::mapping::constellation::ConstellationKind;
use comrs::mapping::theory::ser_mgf;
use comrs::math::{C64, db2lin, qfunc};
use comrs::mimo::precoding::ris_co_phasing;
use serde_json::json;

const TARGET: f64 = 1e-3;

/// Eq. (17): the SER of BPSK in the Gaussian approximation of the co-phased amplitude.
fn theory(ris: Ris, esn0_db: f64) -> f64 {
    let e = db2lin(esn0_db);
    ser_mgf(ConstellationKind::Psk, 2, |s| ris.clt_snr_mgf(e, s)).unwrap()
}

/// The SER of BPSK through `ris` with co-phased elements, one channel per symbol.
fn simulated(ris: Ris, esn0_db: f64, seed: u64) -> f64 {
    let sigma = (1.0 / db2lin(esn0_db) / 2.0).sqrt();
    let c = until_errors(seed, 400, 20_000_000, 10_000, |rng| {
        let ch = ris.draw(rng);
        let gain = ch.cascade_gain(&ris_co_phasing(&ch.h, &ch.g)?)?;
        let x = if rng.bit() == 1 { -1.0 } else { 1.0 };
        let r = gain * x + C64::new(rng.gaussian(), rng.gaussian()) * sigma;
        Ok((u64::from((r.re < 0.0) != (x < 0.0)), 1))
    })
    .unwrap();
    c.rate()
}

/// The $E_{s}/N_{0}$ (dB) at which eq. (17) crosses the target, on a grid of 0.05 dB.
fn theory_crossing(ris: Ris) -> f64 {
    sweep_crossing(-50.0, 0.05, 1400, TARGET, |_, x| theory(ris, x)).unwrap()
}

#[test]
fn eq_17_lies_below_its_bound_and_awgn_as_in_fig_6_of_basar_2019() {
    for n in [16, 32] {
        let ris = Ris::new(n).unwrap();
        for esn0_db in [-30.0, -20.0, -10.0, 0.0, 10.0] {
            // eq. (18): the integrand of eq. (17) at η = π/2, times π/2 / π
            let bound = 0.5 * ris.clt_snr_mgf(db2lin(esn0_db), -1.0);
            let p = theory(ris, esn0_db);
            assert!(p <= bound, "N = {n} at {esn0_db} dB: {p} above the bound {bound}");
        }
        // at -10 dB the RIS of 16 elements is far better than AWGN at the same SNR
        let awgn = qfunc((2.0 * db2lin(-10.0)).sqrt());
        assert!(theory(ris, -10.0) < awgn / 100.0);
    }
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_simulated_ser_follows_eq_17_as_in_fig_7_of_basar_2019() {
    let mut card = Card::load("basar2019_fig7").unwrap();
    let mut crossings = Vec::new();
    for n in [4usize, 8, 16, 32, 64, 128, 256] {
        let ris = Ris::new(n).unwrap();
        let th = theory_crossing(ris);
        let sim =
            sweep_crossing(th - 3.0, 0.5, 13, TARGET, |k, x| simulated(ris, x, 100 * n as u64 + k as u64)).unwrap();
        println!("N = {n}: eq. (17) {th:.2} dB, simulation {sim:.2} dB");
        if n >= 32 {
            card.check(&json!({"elements": n, "quantity": "simulation_minus_eq17_db"}), sim - th);
        }
        crossings.push((n, th));
    }
    for w in crossings.windows(2).filter(|w| w[0].0 >= 32) {
        card.check(&json!({"elements": w[0].0, "quantity": "doubling_gain_db"}), w[0].1 - w[1].1);
    }
    card.finish().unwrap();
}

//! The uplink of multi-user massive MIMO (`comrs::mimo::massive::Uplink`, the MR and ZF
//! receivers of `comrs::mimo::detection::Linear`) against H. Q. Ngo, E. G. Larsson and
//! T. L. Marzetta, "Energy and spectral efficiency of very large multiuser MIMO
//! systems," *IEEE Trans. Commun.*, 2013 (arXiv 1112.3810).
//!
//! The users are dropped as in Sec. V-A: uniformly in a hexagonal cell of radius
//! 1000 m, at least 100 m from the base station, $\beta_{k} = z_{k} / (r_{k} / 100)^{3.8}$
//! with $z_{k}$ log-normal of 8 dB; one drop of $K = 10$ users, drawn once (the drop of
//! the paper is not published, so its curves are not reproduced point by point).
//!
//! - **Fig. 1**: at $p_{u}$ = 10 dB with $\tau = K$ pilots, the closed-form lower bounds
//!   of Propositions 2, 3, 6 and 7 lie below the achievable sum rates (eqs. (13), (37)),
//!   measured by Monte Carlo with the estimates of `Linear`, and the gap closes as $M$
//!   grows ("all bounds are very tight, especially at large $M$").
//! - **Fig. 4**: the transmit power that reaches 1 bit/s/Hz per user, from the bounds,
//!   falls by about 3 dB when $M$ doubles with perfect CSI and by about 1.5 dB with
//!   estimated CSI.
//!
//! The values and their tolerances are in the cards `validation/cards/ngo2013_fig1.yaml`
//! and `validation/cards/ngo2013_fig4.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::fading::Mimo;
use comrs::mapping::constellation::Constellation;
use comrs::math::C64;
use comrs::mimo::detection::{Criterion, Linear};
use comrs::mimo::massive::Uplink;
use comrs::prelude::*;
use serde_json::json;

/// The large-scale fading of `k` users dropped uniformly in a hexagonal cell of radius
/// 1000 m, at least 100 m from the base station: β = z / (r / 100)^3.8, z log-normal of
/// 8 dB (Sec. V-A).
fn drop_users(k: usize, rng: &mut Rng) -> Vec<f64> {
    let (radius, rh) = (1000.0f64, 100.0f64);
    let s3 = 3f64.sqrt();
    let mut beta = Vec::with_capacity(k);
    while beta.len() < k {
        let (x, y) = (radius * (2.0 * rng.uniform() - 1.0), radius * (2.0 * rng.uniform() - 1.0));
        let inside = y.abs() <= s3 / 2.0 * radius && s3 * x.abs() + y.abs() <= s3 * radius;
        let r = x.hypot(y);
        if inside && r >= rh {
            let z = 10f64.powf(8.0 * rng.gaussian() / 10.0);
            beta.push(z / (r / rh).powf(3.8));
        }
    }
    beta
}

/// The achievable sum rate (eqs. (13), (37)) by Monte Carlo over `trials` channels,
/// and its standard error.
fn sum_rate(up: &Uplink, receiver: Criterion, trials: usize, rng: &mut Rng) -> (f64, f64) {
    let lin = Linear { constellation: Constellation::psk(4).unwrap().normalized(), criterion: receiver };
    let (m, k) = (up.antennas, up.large_scale.len());
    let scale: Vec<f64> = up.estimate_variance().iter().map(|v| (up.snr_lin * v).sqrt()).collect();
    let noise = up.effective_noise_variance();
    let y = vec![C64::new(0.0, 0.0); m];
    let rates: Vec<f64> = (0..trials)
        .map(|_| {
            let mut h = Mimo::rayleigh(m, k).draw(rng).unwrap();
            for (j, s) in scale.iter().enumerate() {
                h.column_mut(j).mapv_inplace(|v| v * *s);
            }
            let e = lin.estimate(&h.view(), &y, noise).unwrap();
            e.noise_var.iter().map(|v| (1.0 + 1.0 / v).log2()).sum::<f64>()
        })
        .collect();
    let mean = rates.iter().sum::<f64>() / trials as f64;
    let var = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (trials - 1) as f64;
    (mean, (var / trials as f64).sqrt())
}

const USERS: usize = 10;

fn uplink(antennas: usize, beta: &[f64], snr_lin: f64, csi: &str) -> Uplink {
    let up = Uplink::new(antennas, beta.to_vec(), snr_lin).unwrap();
    if csi == "estimated" { up.with_pilots(USERS).unwrap() } else { up }
}

fn name(r: Criterion) -> &'static str {
    if r == Criterion::Mr { "mr" } else { "zf" }
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_bounds_are_tight_lower_bounds_as_in_fig_1_of_ngo_2013() {
    let mut rng = Rng::new(2013);
    let beta = drop_users(USERS, &mut rng);
    let mut card = Card::load("ngo2013_fig1").unwrap();
    for csi in ["perfect", "estimated"] {
        for r in [Criterion::Mr, Criterion::Zf] {
            let mut previous = f64::INFINITY;
            for m in [20, 50, 100, 200, 500] {
                let up = uplink(m, &beta, 10.0, csi);
                let (rate, se) = sum_rate(&up, r, 400, &mut rng);
                let bound = up.sum_rate_bound(r).unwrap();
                // a lower bound (Jensen's inequality), to the Monte Carlo error
                assert!(bound <= rate + 3.0 * se, "{csi} {r:?} M = {m}: bound {bound} above {rate} ± {se}");
                let gap = 100.0 * (rate - bound) / rate;
                assert!(gap < previous + 3.0 * 100.0 * se / rate, "{csi} {r:?} M = {m}: the gap grows");
                previous = gap;
                if m == 500 {
                    card.check(&json!({"receiver": name(r), "csi": csi}), gap);
                }
            }
        }
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn doubling_the_antennas_saves_the_power_of_fig_4_of_ngo_2013() {
    let mut rng = Rng::new(2013);
    let beta = drop_users(USERS, &mut rng);
    // the transmit power (dB) at which the bound of the sum rate reaches 1 bit/s/Hz per user
    let power_db = |m: usize, csi: &str, r: Criterion| -> f64 {
        comrs::math::search::bisect(
            |db| uplink(m, &beta, comrs::math::db2lin(db), csi).sum_rate_bound(r).unwrap() < USERS as f64,
            -60.0,
            80.0,
            80,
        )
    };
    let mut card = Card::load("ngo2013_fig4").unwrap();
    for csi in ["perfect", "estimated"] {
        for r in [Criterion::Mr, Criterion::Zf] {
            for m in [400, 102_400] {
                // with estimated CSI 1.5 dB is the limit of large M (the card's findings)
                if csi == "estimated" && m == 400 {
                    continue;
                }
                let saving = power_db(m, csi, r) - power_db(2 * m, csi, r);
                card.check(&json!({"receiver": name(r), "csi": csi, "antennas": m}), saving);
            }
        }
    }
    card.finish().unwrap();
}

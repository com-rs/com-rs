//! The degradation of OFDM by a carrier frequency offset against T. Pollet, M. Van
//! Bladel and M. Moeneclaey, "BER sensitivity of OFDM systems to carrier frequency
//! offset and Wiener phase noise," *IEEE Trans. Commun.* 43(2/3/4), 191–193, 1995: the
//! power of the received bin and of the interference (eq. (6) and its appendix), and the
//! degradation of eq. (4) against eq. (7). The tests measure; the expected values and
//! what they settle in Moose (1994) are in the card `validation/cards/pollet1995.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::prelude::*;
use comrs::waveform::ofdm::Ofdm;
use comrs::waveform::ofdm::sync::{cfo_coefficient, ici_power};
use serde_json::json;

/// A full grid of `n` bins at one sample per subcarrier spacing.
fn full(n: usize) -> Ofdm {
    Ofdm { oversampling: 1, ..Ofdm::new(n, n).unwrap() }
}

#[test]
fn pollet_1995_eqs_4_6_7() {
    let mut card = Card::load("pollet1995").unwrap();
    for (n, eps) in [(64, 0.1), (64, 0.5), (256, 0.3)] {
        let total = cfo_coefficient(n, 0, eps).norm_sqr() + ici_power(&full(n), n / 2, eps);
        card.check(&json!({"part": "power", "n": n, "eps": eps}), total);
    }
    for (n, eps, es_n0_db) in [(256, 0.01, 20), (256, 0.003, 30)] {
        // eq. (4): D = -10 log E0² + 10 log(1 + V0 Es/N0)
        let e0 = cfo_coefficient(n, 0, eps).norm_sqr();
        let v0 = ici_power(&full(n), n / 2, eps);
        let d = -lin2db(e0) + lin2db(1.0 + v0 * db2lin(f64::from(es_n0_db)));
        card.check(&json!({"part": "degradation", "n": n, "eps": eps, "es_n0_db": es_n0_db}), d);
    }
    card.finish().unwrap();
}

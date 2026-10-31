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
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::optical::fiber::FiberParams;
use comrs::channel::optical::gn::{FormatMoments, GnModel, WdmSpectrum};
use comrs::mapping::constellation::Constellation;
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
    let k = 16;
    let acc: f64 = (0..=k)
        .map(|i| {
            let w = if i == 0 || i == k {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            w * m.nli_psd(&s, -RS / 2.0 + RS * f64::from(i) / f64::from(k)).unwrap()
        })
        .sum();
    acc * RS / f64::from(k) / 3.0
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

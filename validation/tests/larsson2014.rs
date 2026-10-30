//! Favourable propagation of the i.i.d. Rayleigh channel (`comrs::channel::fading::Mimo`,
//! `comrs::math::linalg::svd`) against E. G. Larsson, O. Edfors, F. Tufvesson and T. L.
//! Marzetta, "Massive MIMO for next generation wireless systems," *IEEE Commun. Mag.*
//! 52(2), 186–195, 2014, Fig. 6: the distribution of the singular value spread
//! $20 \log_{10}(\sigma_{\max} / \sigma_{\min})$ of the channel of 4 single-antenna
//! terminals and 4, 32 or 128 base-station antennas, i.i.d. curves (the measured linear
//! and cylindrical arrays are not reproducible). The test measures its quartiles; the
//! values read on the figure, their tolerance and the reason for it are in the card
//! `validation/cards/larsson2014_fig6.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::fading::Mimo;
use comrs::math::linalg::svd;
use comrs::prelude::*;
use serde_json::json;

#[test]
fn larsson_2014_fig_6() {
    let mut card = Card::load("larsson2014_fig6").unwrap();
    let mut rng = Rng::new(2014);
    for antennas in [4, 32, 128] {
        let mimo = Mimo::rayleigh(antennas, 4);
        let mut spread: Vec<f64> = (0..5000)
            .map(|_| {
                let (_, s, _) = svd(&mimo.draw(&mut rng).unwrap().view());
                20.0 * (s[0] / s[s.len() - 1]).log10()
            })
            .collect();
        spread.sort_by(f64::total_cmp);
        for cdf in [0.25, 0.5, 0.75] {
            let q = spread[(cdf * spread.len() as f64) as usize];
            println!("{antennas} antennas, CDF {cdf}: {q:.2} dB");
            card.check(&json!({"antennas": antennas, "cdf": cdf}), q);
        }
    }
    card.finish().unwrap();
}

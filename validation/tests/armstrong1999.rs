//! The weights of the intercarrier interference (`comrs::waveform::ofdm::sync::
//! cfo_coefficient`) against J. Armstrong, "Analysis of new and existing methods of
//! reducing intercarrier interference due to carrier frequency offset in OFDM," *IEEE
//! Trans. Commun.* 47(3), 365–369, 1999, eq. (8) and Figs. 2–3. The test measures; the
//! values read on the figures are in the card `validation/cards/armstrong1999.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::waveform::ofdm::sync::cfo_coefficient;
use serde_json::json;

#[test]
fn armstrong_1999_figs_2_3() {
    let mut card = Card::load("armstrong1999").unwrap();
    for (fig, n, eps, d) in [(2, 16, 0.2, 0), (2, 16, 0.2, 1), (2, 16, 0.2, -1), (3, 2, 0.1, 0), (3, 1024, 0.1, 0)] {
        let c = cfo_coefficient(n, d, eps);
        for (part, v) in [("real", c.re), ("imag", c.im), ("power", c.norm_sqr())] {
            let inputs = json!({"fig": fig, "n": n, "eps": eps, "d": d, "part": part});
            if card.expected(&inputs).is_some() {
                card.check(&inputs, v);
            }
        }
    }
    card.finish().unwrap();
}

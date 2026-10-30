//! The truncated impulse response of the chromatic-dispersion compensation
//! (`comrs::dsp::compensation::cd::{cd_savory_length, cd_savory_taps}`) against S. J.
//! Savory, "Digital coherent optical receivers: algorithms and subsystems," *IEEE J. Sel.
//! Topics Quantum Electron.* 16(5), 1164–1179, 2010, eq. (11):
//!
//! $$
//! h_{\mathrm{cd}}[k] = \frac{1}{\sqrt{\rho}} \exp\left(-j \frac{\pi}{\rho}
//! \left(k - \frac{N - 1}{2}\right)^{2}\right), \qquad N = \lfloor |\rho| \rfloor, \qquad
//! \rho = \frac{2\pi\beta_{2}L}{T_{\mathrm{ADC}}^{2}},
//! $$
//!
//! for $\beta_{2} = -21$ ps²/km (the value of Sec. VI-A): the number of taps, the modulus
//! of the taps and their phase relative to the centre tap. The paper takes the electrical
//! engineering convention of the field, $e^{j(\omega t - \beta z)}$ (its footnote 11), and
//! com-rs that of physics, so the phases of the library are those of eq. (11) conjugated;
//! the test conjugates them. The values of eq. (11), the tolerances and the differences
//! are in the card `validation/cards/savory2010_eq11.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::dsp::compensation::cd::{cd_savory_length, cd_savory_taps};
use serde_json::json;

/// $\beta_{2}$ (s²/km) of Sec. VI-A.
const BETA2: f64 = -21e-24;

#[test]
fn savory_2010_eq_11() {
    let mut card = Card::load("savory2010_eq11").unwrap();
    for (length_km, fs) in [(100.0, 56e9), (500.0, 56e9), (1000.0, 56e9), (80.0, 64e9)] {
        let n = cd_savory_length(BETA2, length_km, fs);
        card.check(&json!({"quantity": "taps", "length_km": length_km, "fs": fs}), n as f64);
    }
    // the taps of 1000 km at 56 GSa/s, about the centre tap c
    let (length_km, fs) = (1000.0, 56e9);
    let h = cd_savory_taps(BETA2, length_km, fs, cd_savory_length(BETA2, length_km, fs)).unwrap();
    let c = h.len() / 2;
    for m in [0, 1, 5, c] {
        let inputs = |q: &str| json!({"quantity": q, "length_km": length_km, "fs": fs, "offset": m});
        card.check(&inputs("modulus"), h[c + m].norm());
        if m > 0 {
            card.check(&inputs("phase"), (h[c + m] / h[c]).conj().arg());
        }
    }
    card.finish().unwrap();
}

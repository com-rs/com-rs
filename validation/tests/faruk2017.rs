//! The least-squares FIR compensation of chromatic dispersion over the whole band
//! (`comrs::dsp::compensation::cd::cd_ls_taps` with $\Omega = \pi$ and no penalty) against
//! M. S. Faruk and S. J. Savory, "Digital signal processing for coherent transceivers
//! employing multilevel formats," *J. Lightw. Technol.* 35(5), 1125–1141, 2017, eq. (12):
//!
//! $$
//! h_{c_{LS}}(n) = \frac{1}{2j}\left[\mathrm{erfi}\left(\sqrt{\frac{j\pi}{\rho}}
//! \left(n - \frac{N_{c} - 1}{2} + \frac{\rho}{2}\right)\right) - \mathrm{erfi}\left(
//! \sqrt{\frac{j\pi}{\rho}}\left(n - \frac{N_{c} - 1}{2} - \frac{\rho}{2}\right)\right)
//! \right] \frac{1}{\sqrt{\rho}} \exp\left(-\frac{j\pi}{\rho}\left(n - \frac{N_{c} -
//! 1}{2}\right)^{2}\right),
//! $$
//!
//! $\rho = 2\pi\beta_{2}z/T_{s}^{2}$, $N_{c} = \lfloor |\rho| \rfloor$, for $\beta_{2} =
//! -21$ ps²/km: the modulus of the taps and their phase relative to the centre tap
//! (conjugated: the paper takes the field as $e^{j(\omega t - \beta z)}$, com-rs as
//! $e^{-j(\omega t - \beta z)}$). The values of eq. (12), computed with the complex `erfi`
//! of `SciPy`, the tolerances and the differences are in the card
//! `validation/cards/faruk2017_eq12.yaml`. The truncated impulse response of their
//! eq. (11) is that of Savory (2010), checked by `validation/tests/savory2010.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::dsp::compensation::cd::cd_ls_taps;
use serde_json::json;

/// $\beta_{2}$ (s²/km).
const BETA2: f64 = -21e-24;

#[test]
fn faruk_2017_eq_12() {
    let mut card = Card::load("faruk2017_eq12").unwrap();
    for (length_km, fs, n_taps) in [(100.0, 56e9, 41usize), (1000.0, 56e9, 413)] {
        // the whole band, |f| <= fs/2, without penalty: eq. (12)
        let h = cd_ls_taps(BETA2, length_km, fs, n_taps, fs / 2.0, 0.0).unwrap();
        let c = n_taps / 2;
        for m in [0, 1, 5, c] {
            let inputs = |q: &str| json!({"quantity": q, "length_km": length_km, "offset": m});
            card.check(&inputs("modulus"), h[c + m].norm());
            if m > 0 {
                card.check(&inputs("phase"), (h[c + m] / h[c]).conj().arg());
            }
        }
    }
    card.finish().unwrap();
}

//! Predistortion by indirect learning (`comrs::dsp::compensation::dpd`) against C. Eun
//! and E. J. Powers, "A new Volterra predistorter based on the indirect learning
//! architecture," *IEEE Trans. Signal Process.* 45(1), 223–227, 1997, Sec. V: the
//! baseband satellite channel of their Fig. 2 — the transmit filter $h = [0.8, 0.1]$ of
//! eq. (19), the travelling-wave tube of eqs. (15)–(16) with $\alpha_{a} = 2$,
//! $\beta_{a} = 1$, $\alpha_{\phi} = \pi/3$, $\beta_{\phi} = 1$
//! (`comrs::hardware::amplifier::Saleh`) and the receive filter $g = [0.9, 0.2, 0.1]$ of
//! eq. (20), both causal — driven by 16-PSK and 16-QAM symbols (one sample per symbol)
//! of peak amplitude 0.64, without noise, and compensated to the unit gain. The NMSE of
//! eq. (18) is measured on 500 fresh symbols, as in Figs. 5(c) and 6(c). The paper's
//! predistorter is a third-order Volterra series of memory 3 trained by RLS; here it is
//! the memory polynomial of com-rs (the diagonal of that series) of the same order and
//! memory, fitted by batch least squares. The values read in the paper, their tolerance
//! and the reason for it are in the card `validation/cards/eun1997_figs5_6.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::dsp::compensation::dpd::MemoryPolynomialDpd;
use comrs::dsp::filter::convolve_full;
use comrs::hardware::amplifier::Saleh;
use comrs::metrics::evm_data_aided;
use comrs::prelude::*;
use serde_json::json;
use std::f64::consts::PI;

/// Peak amplitude of the input of the predistorter (Sec. V).
const PEAK: f64 = 0.64;

/// A causal FIR filter of real taps: the first `x.len()` samples of the convolution.
fn causal(taps: &[f64], x: &[C64]) -> Vec<C64> {
    let h: Vec<C64> = taps.iter().map(|&t| C64::new(t, 0.0)).collect();
    convolve_full(&h, x).into_iter().take(x.len()).collect()
}

/// The channel of Fig. 2: transmit filter, TWT, receive filter.
fn channel(z: &[C64]) -> Vec<C64> {
    let twt = Saleh { alpha_a: 2.0, beta_a: 1.0, alpha_p: PI / 3.0, beta_p: 1.0 };
    causal(&[0.9, 0.2, 0.1], &twt.amplify(&causal(&[0.8, 0.1], z)))
}

/// `n` random symbols of `c` scaled to the peak amplitude [`PEAK`].
fn symbols(c: &Constellation, n: usize, rng: &mut Rng) -> Vec<C64> {
    let peak = c.points().iter().map(|p| p.norm()).fold(0.0, f64::max);
    (0..n).map(|_| c.points()[rng.below(c.points().len())] * (PEAK / peak)).collect()
}

#[test]
fn eun_powers_1997_figs_5_6() {
    let mut card = Card::load("eun1997_figs5_6").unwrap();
    let mut rng = Rng::new(1997);
    for (name, c) in [("16psk", Constellation::psk(16).unwrap()), ("16qam", Constellation::qam(16).unwrap())] {
        let train = symbols(&c, 4000, &mut rng);
        let test = symbols(&c, 500, &mut rng);
        let mut dpd = MemoryPolynomialDpd::new(vec![1, 3], 3);
        dpd.train_indirect(&train, |z| Ok(channel(z)), C64::new(1.0, 0.0), 6, 1e-9).unwrap();
        let without = evm_data_aided(&channel(&test), &test).unwrap();
        let with = evm_data_aided(&channel(&dpd.apply(&test)), &test).unwrap();
        println!("{name}: NMSE {without:.3e} without, {with:.3e} with the predistorter");
        card.check(&json!({"modulation": name}), with);
    }
    card.finish().unwrap();
}

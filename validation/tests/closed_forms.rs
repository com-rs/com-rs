//! BPSK over AWGN and over flat Rayleigh fading against their closed forms.
//!
//! $$
//! P_{b}^{\mathrm{AWGN}} = Q\left(\sqrt{2 E_{b} / N_{0}}\right), \qquad
//! P_{b}^{\mathrm{Rayleigh}} = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1 + \bar{\gamma}}}\right)
//! $$
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::{MIN_ERRORS, within_three_sigma};
use comrs::channel::fading::Rayleigh;
use comrs::mapping::theory::ber_rayleigh_mrc;
use comrs::math::qfunc;
use comrs::prelude::*;

const BLOCK: usize = 10_000;

/// BPSK symbols `1 - 2b` of a block of random bits.
fn bpsk(rng: &mut Rng) -> (Vec<u8>, Vec<C64>) {
    let bits = source::bits(BLOCK, rng);
    let x = bits.iter().map(|&b| C64::new(1.0 - 2.0 * f64::from(b), 0.0)).collect();
    (bits, x)
}

/// Count bit errors over blocks until `MIN_ERRORS`, with `link` returning the decision
/// statistic of every bit.
fn simulate(seed: u64, mut link: impl FnMut(&[C64], &mut Rng) -> Vec<f64>) -> (u64, u64) {
    let (mut errors, mut total) = (0u64, 0u64);
    let mut batch = 0;
    while errors < MIN_ERRORS {
        let mut rng = batch_rng(seed, batch);
        let (bits, x) = bpsk(&mut rng);
        let z = link(&x, &mut rng);
        errors += bits.iter().zip(&z).filter(|(b, z)| (**z < 0.0) != (**b == 1)).count() as u64;
        total += BLOCK as u64;
        batch += 1;
    }
    (errors, total)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn bpsk_over_awgn_follows_q_of_sqrt_2_ebn0() {
    for ebn0_db in [0.0, 2.0, 4.0, 6.0, 8.0] {
        let (errors, total) =
            simulate(1, |x, rng| awgn::awgn_ebn0(x, ebn0_db, 1, 1.0, rng).iter().map(|v| v.re).collect());
        let theory = qfunc((2.0 * db2lin(ebn0_db)).sqrt());
        assert!(within_three_sigma(errors, total, theory), "{ebn0_db} dB: {errors}/{total} vs {theory}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn bpsk_over_flat_rayleigh_follows_the_closed_form() {
    for ebn0_db in [0.0, 5.0, 10.0, 15.0] {
        let (errors, total) = simulate(2, |x, rng| {
            let s = Signal::from_stream(x.to_vec(), Hertz(1.0)).expect("one stream");
            let (y, h) = Rayleigh { coherence: 1 }.fade(&s, rng).expect("valid coherence");
            // the noise of the average SNR: E|h|² = 1
            let sigma = (1.0 / db2lin(ebn0_db)).sqrt();
            y.stream(0).iter().zip(&h[0]).map(|(v, g)| ((v + rng.complex_gaussian() * sigma) * g.conj()).re).collect()
        });
        let g = db2lin(ebn0_db);
        let theory = ber_rayleigh_mrc(g, 1);
        assert!(within_three_sigma(errors, total, theory), "{ebn0_db} dB: {errors}/{total} vs {theory}");
    }
}

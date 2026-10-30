//! BPSK over AWGN and over flat Rayleigh fading against their closed forms.
//!
//! $$
//! P_{b}^{\mathrm{AWGN}} = Q\left(\sqrt{2 E_{b} / N_{0}}\right), \qquad
//! P_{b}^{\mathrm{Rayleigh}} = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1 + \bar{\gamma}}}\right)
//! $$
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::{MIN_ERRORS, until_errors, within_three_sigma};
use comrs::channel::awgn::add_noise;
use comrs::channel::fading::Rayleigh;
use comrs::fec::block::bpsk;
use comrs::mapping::theory::ber_rayleigh_mrc;
use comrs::math::qfunc;
use comrs::metrics::errors::bit_errors;
use comrs::prelude::*;

const BLOCK: usize = 10_000;

/// Count bit errors over blocks of BPSK symbols `1 - 2b` until `MIN_ERRORS`, block `b`
/// drawn from `batch_rng(seed, b)`, with `link` returning the decision statistic of
/// every bit.
fn simulate(seed: u64, link: impl Fn(&[C64], &mut Rng) -> Vec<f64> + Sync) -> (u64, u64) {
    let count = until_errors(seed, MIN_ERRORS, u64::MAX, 1, |rng| {
        let bits = source::bits(BLOCK, rng);
        let x: Vec<C64> = bpsk(&bits).iter().map(|&v| C64::new(v, 0.0)).collect();
        let decided: Vec<u8> = link(&x, rng).iter().map(|&z| u8::from(z < 0.0)).collect();
        Ok((bit_errors(&decided, &bits)? as u64, BLOCK as u64))
    })
    .unwrap();
    (count.errors, count.total)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn bpsk_over_awgn_follows_q_of_sqrt_2_ebn0() {
    for ebn0_db in [0.0, 2.0, 4.0, 6.0, 8.0] {
        let (errors, total) =
            simulate(1, |x, rng| awgn::awgn_ebn0(x, ebn0_db, 1.0, 1.0, rng).iter().map(|v| v.re).collect());
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
            let mut y = y.stream(0).to_vec();
            add_noise(&mut y, 1.0 / db2lin(ebn0_db), rng);
            y.iter().zip(&h[0]).map(|(v, g)| (v * g.conj()).re).collect()
        });
        let g = db2lin(ebn0_db);
        let theory = ber_rayleigh_mrc(g, 1).unwrap();
        assert!(within_three_sigma(errors, total, theory), "{ebn0_db} dB: {errors}/{total} vs {theory}");
    }
}

/// The capacity of the binary-input AWGN channel of `com_fec::bounds` (the moments of
/// the information density) is the mutual information of 2-PAM of
/// `com_mapping::theory` (a real channel of variance $\sigma^{2} = 1/\mathrm{SNR}$): two
/// quadratures of the same integral.
#[test]
fn the_biawgn_capacity_is_the_mutual_information_of_bpsk() {
    use comrs::fec::bounds::biawgn_capacity;
    use comrs::mapping::constellation::ConstellationKind;
    use comrs::mapping::theory;
    for snr_db in [-6.0, -2.0, 0.0, 3.0, 8.0] {
        let c = biawgn_capacity(comrs::math::db2lin(-snr_db));
        let mi = theory::mi(ConstellationKind::Pam, 2, snr_db, None, 60).unwrap();
        assert!((c - mi).abs() < 1e-6, "{snr_db} dB: {c} vs {mi}");
    }
}

//! The channel codes of 5G NR (`comrs::fec::nr`) against the normal approximation of
//! the block error probability (Y. Polyanskiy, H. V. Poor, S. Verdú, "Channel coding
//! rate in the finite blocklength regime," *IEEE Trans. Inf. Theory* 56(5), 2010), over
//! BPSK and AWGN: the codes of TS 38.212 with belief propagation (LDPC) and CRC-aided
//! SC list (polar) reach a block error rate of 1e-2 within a fraction of a dB of the
//! approximation of the best code of the same length and rate — never better than it.
//! The encoders themselves are compared bit for bit with the reference vectors of Sionna
//! in `tests/test_sionna_vectors.py`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::bpsk_awgn_llr;
use comrs::fec::block::BlockCode;
use comrs::fec::bounds::normal_approximation;
use comrs::fec::nr::{NrLdpc, NrPolar};
use comrs::math::bisect;
use comrs::prelude::*;

/// The block error rate of `code` over BPSK and AWGN at `ebn0_db`.
fn bler(code: &dyn BlockCode, ebn0_db: f64, blocks: usize, rng: &mut Rng) -> f64 {
    let (k, n) = (code.k(), code.n());
    let s2 = 1.0 / (2.0 * k as f64 / n as f64 * db2lin(ebn0_db));
    let mut errors = 0;
    for _ in 0..blocks {
        let b: Vec<u8> = (0..k).map(|_| rng.below(2) as u8).collect();
        let c = code.encode_block(&b).unwrap();
        let llr = bpsk_awgn_llr(&c, s2, rng);
        errors += usize::from(code.decode_block(&llr).unwrap() != b);
    }
    errors as f64 / blocks as f64
}

/// The Eb/N0 (dB) at which the normal approximation reaches 1e-2.
fn na_point(n: usize, k: usize) -> f64 {
    bisect(|x| normal_approximation(n, k, x) > 1e-2, -2.0, 10.0, 60)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn nr_ldpc_within_a_db_of_the_normal_approximation() {
    let mut rng = Rng::new(11);
    let code = NrLdpc::new(500, 1000).unwrap();
    let x = na_point(1000, 500);
    // measured: 0.63 dB from the approximation at 1e-2 (50 iterations of sum-product)
    assert!(bler(&code, x + 0.2, 400, &mut rng) > 1e-2);
    assert!(bler(&code, x + 1.0, 2000, &mut rng) < 1e-2);
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn nr_polar_within_two_db_of_the_normal_approximation() {
    let mut rng = Rng::new(12);
    let code = NrPolar::uplink(40, 128).unwrap().with_list(8).unwrap();
    let x = na_point(128, 40);
    // measured: 1.1 dB at 1e-2 with a list of 8 and the 11 bits of the CRC
    assert!(bler(&code, x + 0.5, 1000, &mut rng) > 1e-2);
    assert!(bler(&code, x + 2.0, 3000, &mut rng) < 1e-2);
}

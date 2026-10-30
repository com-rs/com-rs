//! Successive-cancellation list decoding of polar codes (`comrs::fec::polar`) against I. Tal
//! and A. Vardy, "List decoding of polar codes," *IEEE Trans. Inf. Theory* 61(5),
//! 2213–2226, 2015, Fig. 1: the word error rate of a polar code of length 2048 and rate
//! 1/2 on the BPSK AWGN channel, designed at $E_{b}/N_{0} = 2$ dB, with successive
//! cancellation and with a list of 8. Read on the figure: SC 0.26, 0.0385 and 0.0032 at
//! 1.5, 2 and 2.5 dB; $L = 8$ 0.018 and 0.0028 at 1.5 and 2 dB, on the maximum-likelihood
//! bound. The construction differs (the Gaussian approximation here, the method of Tal &
//! Vardy, *IEEE Trans. Inf. Theory* 59(10), 2013, there): the rates are compared within a
//! factor 1.6, and the list must gain at least a factor 10 over SC at 2 dB as on the figure
//! (14). Here SC gives 0.30, 0.050 and 0.0049 (up to 1.5 times the figure, about 0.07 dB
//! to the right), the list of 8 0.0215 and 0.0033 (15 times below SC at 2 dB); the list of
//! 32 at 1.5 dB 0.018 against 0.014. The list decoder of the
//! library computes every leaf from the channel, $O(L n^{2} \log n)$ per frame, not the
//! $O(L n \log n)$ of their Sec. IV: the list of 8 runs at about 15 frames per second for
//! $n = 2048$, hence at most 10 000 frames per point.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::{MIN_ERRORS, bpsk_awgn_llr};
use comrs::fec::block::BlockCode;
use comrs::fec::polar::PolarCode;
use comrs::prelude::*;

fn wer(list: usize, ebn0_db: f64, max_frames: usize, rng: &mut Rng) -> f64 {
    let code = PolarCode::new(2048, 1024, 2.0).unwrap().with_list(list);
    let noise_var = 1.0 / (2.0 * 0.5 * db2lin(ebn0_db));
    let (mut errors, mut frames) = (0u64, 0usize);
    while errors < MIN_ERRORS && frames < max_frames {
        if frames > 0 && frames % 2000 == 0 {
            println!("  L = {list}, {ebn0_db} dB: {errors} errors in {frames} frames");
        }
        let info: Vec<u8> = (0..1024).map(|_| rng.below(2) as u8).collect();
        let c = code.encode_block(&info).unwrap();
        if code.decode_block(&bpsk_awgn_llr(&c, noise_var, rng)).unwrap() != info {
            errors += 1;
        }
        frames += 1;
    }
    errors as f64 / frames as f64
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn tal_vardy_2015_fig_1() {
    let mut rng = Rng::new(2015);
    let read = [(1, 1.5, 0.26), (1, 2.0, 0.0385), (1, 2.5, 0.0032), (8, 1.5, 0.018), (8, 2.0, 0.0028)];
    let mut at_2 = [0.0; 2];
    for (list, ebn0, want) in read {
        // the list decoder is slow: 10 000 frames (about 28 errors at 2 dB)
        let cap = if list == 1 { 200_000 } else { 10_000 };
        let t = std::time::Instant::now();
        let got = wer(list, ebn0, cap, &mut rng);
        println!("  ({:.0} s)", t.elapsed().as_secs_f64());
        println!("L = {list}, {ebn0} dB: {got:.3e} (figure {want})");
        assert!((got / want).ln().abs() < 1.6f64.ln(), "L = {list}, {ebn0} dB: {got} vs {want}");
        if ebn0 == 2.0 {
            at_2[usize::from(list > 1)] = got;
        }
    }
    assert!(at_2[0] / at_2[1] > 10.0, "list gain at 2 dB: {at_2:?}");
}

//! Successive-cancellation list decoding of polar codes (`comrs::fec::polar`) against I. Tal
//! and A. Vardy, "List decoding of polar codes," *IEEE Trans. Inf. Theory* 61(5),
//! 2213–2226, 2015, Fig. 1: the word error rate of a polar code of length 2048 and rate
//! 1/2 on the BPSK AWGN channel, designed at $E_{b}/N_{0} = 2$ dB, with successive
//! cancellation and with lists of 8 and 32. The test measures; the values read on the
//! figure, their tolerance and the reason for it are in the card
//! `validation/cards/tal2015_fig1.yaml`. The construction differs (the Gaussian
//! approximation here, the method of Tal & Vardy, *IEEE Trans. Inf. Theory* 59(10), 2013,
//! there), which puts SC about 0.07 dB to the right of the figure. The decoders run on the
//! arrays of their Sec. IV, $O(L n \log n)$ per frame: every point reaches 100 errors, in
//! about 2 minutes.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::{MIN_ERRORS, bpsk_awgn_llr};
use comrs::fec::block::BlockCode;
use comrs::fec::polar::PolarCode;
use comrs::prelude::*;
use serde_json::json;

fn wer(list: usize, ebn0_db: f64, max_frames: usize, rng: &mut Rng) -> f64 {
    let code = PolarCode::new(2048, 1024, 2.0).unwrap().with_list(list).unwrap();
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
    let mut card = Card::load("tal2015_fig1").unwrap();
    let mut wers = Vec::new();
    for (list, ebn0) in [(1, 1.5), (1, 2.0), (1, 2.5), (8, 1.5), (8, 2.0), (32, 1.5), (32, 2.0)] {
        let t = std::time::Instant::now();
        let got = wer(list, ebn0, 200_000, &mut rng);
        println!("L = {list}, {ebn0} dB: {got:.3e} ({:.0} s)", t.elapsed().as_secs_f64());
        card.check(&json!({"list": list, "ebn0_db": ebn0}), got);
        wers.push(got);
    }
    // SC over the list of 8 at 2 dB; the list of 32 over the list of 8 at 1.5 dB
    card.check(&json!({"claim": "list_gain_at_2db"}), wers[1] / wers[4]);
    card.check(&json!({"claim": "list_32_over_list_8_at_1.5db"}), wers[5] / wers[3]);
    card.finish().unwrap();
}

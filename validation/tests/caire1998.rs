//! Coded modulation (CM) and bit-interleaved coded modulation (BICM) with square QAM
//! against G. Caire, G. Taricco and E. Biglieri, "Bit-interleaved coded modulation,"
//! *IEEE Trans. Inf. Theory* 44(3), 1998
//! (`references/coding/1998_caire_bit-interleaved-coded-modulation_it.pdf`), Sec. III-C and
//! Figs. 6–7 (`comrs::mapping::bicm`; the capacities of Figs. 4–5 are checked by
//! `crates/com-mapping/src/bicm.rs::literature`):
//!
//! - over AWGN, CM beats BICM at every rate, by less than 0.5 dB at $R_{0} = 7$
//!   bit/dim (0.34 dB here); at $R_{0} = 3$ the text says "more than 3 dB", while eqs.
//!   (17)–(21) give 2.55 dB (CM 11.04 dB, BICM 13.60 dB) and Fig. 6 itself reads about
//!   2.7 dB; a Monte Carlo of eq. (19) with $2 \cdot 10^{5}$ samples at 13.6 dB gives
//!   $R_{0} = 2.999 \pm 0.004$ against 3.003 by quadrature;
//! - over Rayleigh fading with perfect CSI, BICM "generally" beats CM above $R_{0} = 1$
//!   (here from below 1.25 bit/dim with 16QAM, from 1.5–1.75 with 64QAM) and gains
//!   "about 4 dB" at $R_{0} = 7$ with 256QAM (4.0 dB here: CM 33.45 dB, BICM 29.4 dB).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use comrs::mapping::bicm::{bicm_cutoff_rate, cm_cutoff_rate};
use comrs::mapping::theory::RateChannel;
use comrs::math::bisect;
use comrs::prelude::*;

/// The SNR (dB) at which the cutoff rate `r0` reaches `target`.
fn snr_for(r0: impl Fn(f64) -> f64, target: f64, iters: usize) -> f64 {
    bisect(|s| r0(s) < target, -10.0, 60.0, iters)
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn caire_1998_fig_6_awgn_cutoff_rate() {
    let c = Constellation::qam(256).unwrap();
    let gap = |target: f64| {
        let cm = snr_for(|s| cm_cutoff_rate(&c, s, RateChannel::Awgn), target, 50);
        let bicm = snr_for(|s| bicm_cutoff_rate(&c, s, RateChannel::Awgn, 12).unwrap(), target, 16);
        bicm - cm
    };
    let (g3, g7) = (gap(3.0), gap(7.0));
    assert!((2.4..2.8).contains(&g3), "R0 = 3: {g3} dB (text: more than 3 dB; Fig. 6: about 2.7 dB)");
    assert!((0.0..0.5).contains(&g7), "R0 = 7: {g7} dB");
    // CM beats BICM at every SNR
    for s in [-5.0, 0.0, 10.0, 20.0, 30.0] {
        assert!(cm_cutoff_rate(&c, s, RateChannel::Awgn) > bicm_cutoff_rate(&c, s, RateChannel::Awgn, 12).unwrap());
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn caire_1998_fig_7_rayleigh_cutoff_rate() {
    let c = Constellation::qam(256).unwrap();
    let cm7 = snr_for(|s| cm_cutoff_rate(&c, s, RateChannel::Rayleigh), 7.0, 50);
    assert!((cm7 - 33.45).abs() < 0.05, "{cm7}");
    // BICM reaches R0 = 7 between 3.5 and 4.5 dB earlier
    assert!(bicm_cutoff_rate(&c, cm7 - 4.5, RateChannel::Rayleigh, 6).unwrap() < 7.0);
    assert!(bicm_cutoff_rate(&c, cm7 - 3.5, RateChannel::Rayleigh, 6).unwrap() > 7.0);
    // BICM needs less SNR than CM above a crossing near R0 = 1 (below 1.25 for 16QAM,
    // between 1.5 and 1.75 for 64QAM) and more below it
    for m in [16, 64] {
        let c = Constellation::qam(m).unwrap();
        for (target, bicm_wins) in [(0.5, false), (1.25, m == 16), (1.5, m == 16), (1.75, true), (3.0, true)] {
            let cm = snr_for(|s| cm_cutoff_rate(&c, s, RateChannel::Rayleigh), target, 40);
            let bicm = snr_for(|s| bicm_cutoff_rate(&c, s, RateChannel::Rayleigh, 8).unwrap(), target, 14);
            assert_eq!(bicm < cm, bicm_wins, "{m}QAM, R0 = {target}: CM {cm} dB, BICM {bicm} dB");
        }
    }
}

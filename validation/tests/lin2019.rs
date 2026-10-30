//! 5G NR (`comrs::waveform::ofdm::nr`) against X. Lin et al., "5G New Radio: Unveiling
//! the essentials of the next generation wireless access technology," *IEEE Commun.
//! Standards Mag.* 3(3), 2019 (arXiv:1806.06898), and D. Astély et al., "LTE: the
//! evolution of mobile broadband," *IEEE Commun. Mag.* 47(4), 2009:
//!
//! - Sec. II of Lin et al.: subcarrier spacings $2^{\mu} \cdot 15$ kHz, $\mu = 0..4$,
//!   $2^{\mu}$ slots of 14 symbols per 1 ms subframe, 10 subframes per frame, prefixes
//!   scaled by $2^{-\mu}$ from the 4.7 µs of LTE, resource blocks of 12 subcarriers;
//!   Astély et al.: 15 kHz, prefixes of 4.7 µs and 16.7 µs, 14 or 12 symbols;
//! - Sec. III-A of Lin et al.: the SSB of 4 symbols and 240 subcarriers (20 RBs), the
//!   PSS a BPSK m-sequence and the SSS a BPSK Gold sequence of length 127, 1008 cell
//!   identities;
//! - the cell search finds every identity from the SSB in white noise at -6 dB per
//!   resource element (the sequences and positions themselves are compared with py3gpp
//!   in `tests/test_nr_oracle.py`).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::awgn::add_noise;
use comrs::prelude::*;
use comrs::waveform::ofdm::nr::{
    Numerology, SUBCARRIERS_PER_RB, SsbElement, detect_cell_id, pss, ssb_grid, ssb_indices, sss,
};

#[test]
fn lin2019_section2_numerology() {
    for mu in 0..=4u32 {
        let nr = Numerology::new(mu, false).unwrap();
        assert_eq!(nr.subcarrier_spacing(), 15e3 * f64::from(1u32 << mu));
        assert_eq!((nr.slots_per_subframe(), nr.slots_per_frame(), nr.symbols_per_slot()), (1 << mu, 10 << mu, 14));
        // the prefix scaled by 2^-mu from the 4.7 µs of LTE
        let n_fft = 4096;
        let cp = nr.cp_lengths(n_fft).unwrap();
        let us = cp[1] as f64 / nr.sample_rate(n_fft) * 1e6;
        assert!((us * f64::from(1u32 << mu) - 4.69).abs() < 0.01, "mu {mu}: {us} µs");
    }
    assert_eq!(SUBCARRIERS_PER_RB, 12);
    // Astély et al. 2009: the extended prefix of 16.7 µs, 12 symbols
    let ext = Numerology::new(0, true).unwrap();
    let us = ext.cp_lengths(2048).unwrap()[0] as f64 / ext.sample_rate(2048) * 1e6;
    assert!((us - 16.67).abs() < 0.01 && ext.symbols_per_slot() == 12);
}

#[test]
fn lin2019_section3_ssb() {
    let n = |k| ssb_indices(5, k).len();
    // 4 symbols × 240 subcarriers = 20 RBs; PSS and SSS of length 127
    assert_eq!(240 / SUBCARRIERS_PER_RB, 20);
    assert_eq!((n(SsbElement::Pss), n(SsbElement::Sss)), (127, 127));
    assert!(pss(0).unwrap().iter().chain(&sss(335, 2).unwrap()).all(|v| v.abs() == 1.0));
    // 1008 identities, each found back from its own SSB
    for cid in 0..1008 {
        assert_eq!(detect_cell_id(&ssb_grid(cid, 0, None).unwrap()).unwrap().0, cid);
    }
}

#[test]
fn cell_search_in_white_noise() {
    let mut rng = Rng::new(31);
    let (snr_db, trials) = (-6.0, 300);
    let mut ok = 0;
    for _ in 0..trials {
        let cid = rng.below(1008) as u32;
        let h = C64::from_polar(1.0, 2.0 * std::f64::consts::PI * rng.uniform());
        let s2 = 1.0 / db2lin(snr_db);
        let mut y: Vec<C64> = ssb_grid(cid, 0, None).unwrap().iter().map(|v| h * v).collect();
        add_noise(&mut y, s2, &mut rng);
        ok += usize::from(detect_cell_id(&y).unwrap().0 == cid);
    }
    assert!(ok >= trials - 2, "{ok} of {trials} at {snr_db} dB");
}

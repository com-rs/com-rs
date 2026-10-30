//! Locating a sync word in binary data (`comrs::dsp::sync::frame::locate_sync_word`)
//! against J. L. Massey, "Optimum frame synchronization," *IEEE Trans. Commun.* 20(2),
//! 115–119, 1972: frames of $N$ equiprobable $\pm 1$ symbols carrying a sync word of $L$
//! symbols at a random position, BPSK on AWGN at $E/N_{0} = \frac{1}{2}$, 1 and 2, the
//! detector outputs quantised to the 16 levels $(2j - 1)\sqrt{E}/6$ of eq. (12), for the
//! Barker sequences of 13 and 7 symbols ($N = 91$ and 28) and the Neuman–Hofman sequence
//! of 13. Tables I (standard) and II (with the sign ambiguity of PSK) give the fraction
//! of frames wrongly synchronised by the optimum (OPT) and correlation (COR) rules over
//! 100 trials each; here 4000 trials, compared within three standard deviations of a
//! proportion over 100 trials. The 3 dB advantage of the text holds: OPT at $E/N_{0} = 1$
//! is within the same margin of COR at $E/N_{0} = 2$ (Barker 13: 0.060 against 0.081,
//! standard case). The Neuman–Hofman sequence beats the Barker sequence of the same length
//! by less than the tables suggest (COR at $E/N_{0} = \frac{1}{2}$: 0.396 against 0.402,
//! where Table I gives 0.32 against 0.42, 100 trials).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::dsp::sync::frame::{SyncRule, locate_sync_word};
use comrs::prelude::*;

const TRIALS: usize = 4000;

/// A sync word, its frame length and the rates of OPT and COR at E/N0 = 1/2, 1 and 2.
type Row<'a> = (&'a [f64], usize, [[f64; 2]; 3]);

/// The 16 quantisation levels $(2j - 1)/6$, $-8 < j \le 8$, of a unit amplitude.
fn quantise(x: f64) -> f64 {
    let j = ((x * 3.0).floor() + 1.0).clamp(-7.0, 8.0);
    (2.0 * j - 1.0) / 6.0
}

fn error_rate(sync: &[f64], n: usize, snr: f64, rule: SyncRule, psk: bool, rng: &mut Rng) -> f64 {
    let sigma = (0.5 / snr).sqrt();
    let mut errors = 0;
    for _ in 0..TRIALS {
        let m = rng.below(n);
        let mut x: Vec<f64> = (0..n).map(|_| if rng.below(2) == 0 { 1.0 } else { -1.0 }).collect();
        for (i, s) in sync.iter().enumerate() {
            x[(m + i) % n] = *s;
        }
        let sign = if psk && rng.below(2) == 1 { -1.0 } else { 1.0 };
        let r: Vec<f64> = x.iter().map(|v| quantise(sign * (v + sigma * rng.gaussian()))).collect();
        if locate_sync_word(&r, sync, rule, psk).unwrap() != m {
            errors += 1;
        }
    }
    errors as f64 / TRIALS as f64
}

fn check(table: &[Row<'_>], psk: bool) {
    let mut rng = Rng::new(1972);
    for (sync, n, rows) in table {
        let mut got = [[0.0; 2]; 3];
        for (k, snr) in [0.5, 1.0, 2.0].iter().enumerate() {
            for (j, rule) in [SyncRule::Optimum { snr: *snr }, SyncRule::Correlation].iter().enumerate() {
                let p = error_rate(sync, *n, *snr, *rule, psk, &mut rng);
                let want = rows[k][j];
                let sd = (want.max(0.01) * (1.0 - want) / 100.0).sqrt();
                assert!(
                    (p - want).abs() < 3.0 * sd + 0.01,
                    "L = {}, E/N0 = {snr}, rule {j}: {p} vs {want}",
                    sync.len()
                );
                got[k][j] = p;
            }
        }
        // 3 dB: OPT at E/N0 = 1 performs as COR at E/N0 = 2
        let sd = (got[2][1] * (1.0 - got[2][1]) / 100.0).sqrt();
        assert!((got[1][0] - got[2][1]).abs() < 3.0 * sd + 0.02, "L = {}: {:?}", sync.len(), got);
    }
}

const BARKER13: [f64; 13] = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
const NEUMAN_HOFMAN13: [f64; 13] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
const BARKER7: [f64; 7] = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0];

#[test]
fn massey_1972_table_i() {
    // rows: E/N0 = 1/2, 1, 2; columns OPT, COR
    let table: [Row<'_>; 3] = [
        (&BARKER13, 91, [[0.31, 0.42], [0.09, 0.19], [0.00, 0.08]]),
        (&NEUMAN_HOFMAN13, 91, [[0.28, 0.32], [0.07, 0.18], [0.00, 0.07]]),
        (&BARKER7, 28, [[0.40, 0.45], [0.21, 0.32], [0.09, 0.22]]),
    ];
    check(&table, false);
}

#[test]
fn massey_1972_table_ii_psk() {
    let table: [Row<'_>; 3] = [
        (&BARKER13, 91, [[0.39, 0.47], [0.14, 0.27], [0.00, 0.12]]),
        (&NEUMAN_HOFMAN13, 91, [[0.39, 0.49], [0.14, 0.24], [0.00, 0.13]]),
        (&BARKER7, 28, [[0.63, 0.63], [0.37, 0.46], [0.21, 0.40]]),
    ];
    check(&table, true);
}

//! Probabilistic amplitude shaping (`comrs::systems::pas`) and the information rates of
//! shaped ASK against G. Böcherer, F. Steiner and P. Schulte, "Bandwidth efficient and
//! rate-matched low-density parity-check coded modulation," *IEEE Trans. Commun.*,
//! 63(12), 4651–4665, 2015 (arXiv:1502.02733v2, the numbering of its tables):
//!
//! - Tables I and III: the SNRs at which the optimised Maxwell–Boltzmann, the uniform
//!   input and bit-metric decoding reach $m - 1$ bits per channel use (deterministic,
//!   the helpers of `com_validation::shaping`);
//! - Tables VI–X: the frame error rates of PAS on the DVB-S2 LDPC codes, `#[ignore]`d
//!   (`cargo test --release -p com-validation --test bocherer2015 -- --ignored
//!   --nocapture`).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::shaping::{bmd_ask, mi_ask, snr_for, snr_mb};
use com_validation::until_errors;
use comrs::fec::block::BlockCode;
use comrs::fec::ccdm::Ccdm;
use comrs::mapping::source;
use comrs::math::{db2lin, lin2db};
use comrs::systems::dvbs2::fec::{CodeRate, Dvbs2Fec, FrameSize};
use comrs::systems::pas::Pas;
use std::time::Instant;

/// Table I of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: the SNR (dB) at
/// which the mutual information of the optimised Maxwell–Boltzmann input $X^{\clubsuit}$
/// (eqs. 8–10), of the uniform input and the capacity $\frac{1}{2}\log_{2}(1 + P)$
/// (eq. 1) reach $m - 1$ bits per channel use for $2^{m}$-ASK, and the shaping gaps
/// (printed to 4 decimals). Table III: the SNR at which the rate of bit-metric decoding
/// with the BRGC labelling (eq. 50, the labels of Table II) of $X^{\clubsuit}$ reaches
/// the same rates.
#[test]
fn bocherer_2015_table_i_and_iii_shaping_gaps_of_ask() {
    // (M, rate, SNR(X♣), SNR(uniform), gap, SNR(capacity), gap, SNR(R_BMD) of Table III)
    let table = [
        (4, 1.0, 4.8180, 5.1181, -0.3001, 4.7712, 0.0468, 4.8313),
        (8, 2.0, 11.8425, 12.6187, -0.7761, 11.7609, 0.0816, 11.8481),
        (16, 3.0, 18.0910, 19.1681, -1.0772, 17.9934, 0.0975, 18.0951),
        (32, 4.0, 24.1706, 25.4140, -1.2434, 24.0654, 0.1052, 24.1742),
        (64, 5.0, 30.2078, 31.5384, -1.3307, 30.0988, 0.1090, 30.2110),
    ];
    for (m, r, s_mb, s_unif, gap_unif, s_cap, gap_cap, s_bmd) in table {
        let (snr, nu) = snr_mb(m, r, &|nu, s| mi_ask(m, nu, s));
        let unif = snr_for(&|s| mi_ask(m, 0.0, s), r).unwrap();
        let cap = lin2db(2f64.powf(2.0 * r) - 1.0);
        let bmd = snr_for(&|s| bmd_ask(m, nu, s), r).unwrap();
        println!(
            "{m}-ASK, {r} bit: X♣ {snr:.4} dB (paper {s_mb}), uniform {unif:.4} ({s_unif}), capacity {cap:.4} \
             ({s_cap}), BMD {bmd:.4} ({s_bmd}), nu = {nu:.6}"
        );
        assert!((cap - s_cap).abs() < 6e-5, "{m}-ASK: capacity {cap} vs {s_cap}");
        assert!((snr - s_mb).abs() < 5e-4, "{m}-ASK: X♣ {snr} vs {s_mb}");
        assert!((unif - s_unif).abs() < 5e-4, "{m}-ASK: uniform {unif} vs {s_unif}");
        assert!((snr - unif - gap_unif).abs() < 1e-3, "{m}-ASK: shaping gap {} vs {gap_unif}", snr - unif);
        assert!((snr - cap - gap_cap).abs() < 1e-3, "{m}-ASK: gap to capacity {} vs {gap_cap}", snr - cap);
        assert!((bmd - s_bmd).abs() < 5e-4, "{m}-ASK: BMD {bmd} vs {s_bmd}");
    }
}

/// One operating point of Tables VI–X of Böcherer, Steiner & Schulte (2015),
/// arXiv:1502.02733v2: $2^{m}$-ASK, the DVB-S2 code (64 800 bits), the bit mapper $\pi$
/// without its final 1 (the signs), the spectral efficiency (bits per channel use), the
/// SNR (dB) and the frame error rate with its 95 % confidence interval; the seed of the
/// simulation.
struct Point {
    table: &'static str,
    m: usize,
    code: CodeRate,
    levels: &'static [usize],
    rate: f64,
    snr_db: f64,
    fer: f64,
    ci: f64,
    seed: u64,
}

/// The Maxwell–Boltzmann amplitude distribution $\propto e^{-\nu a^{2}}$ of $2^{m}$-ASK,
/// $a = 1, 3, \dots, 2^{m} - 1$ (eqs. 8 and 57).
fn amplitudes(m: usize, nu: f64) -> Vec<f64> {
    (0..1usize << (m - 1)).map(|a| (-nu * ((2 * a + 1) as f64).powi(2)).exp()).collect()
}

/// The PAS transceiver of a point: the DVB-S2 LDPC code of 64 800 bits (sum-product,
/// 100 iterations, Sec. IX), the bit mapper of the table, and the MB amplitudes whose
/// matcher gives the spectral efficiency of the table, $R = k_{c}/n_{c}$ (the
/// effective rate of Sec. IX-3; the rate adaption of eqs. 58–60 keeps the amplitudes
/// in the MB family).
fn pas_of(p: &Point) -> Pas {
    let mut code = Dvbs2Fec::new(FrameSize::Normal, p.code).unwrap().ldpc().clone();
    code.max_iterations = 100;
    let n_c = code.n() / p.m;
    let extra = code.k() - n_c * (p.m - 1);
    let rate = |nu: f64| (Ccdm::from_distribution(&amplitudes(p.m, nu), n_c).unwrap().k() + extra) as f64 / n_c as f64;
    // the rate decreases with nu
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if rate(mid) > p.rate { lo = mid } else { hi = mid }
    }
    let nu = if (rate(lo) - p.rate).abs() <= (rate(hi) - p.rate).abs() { lo } else { hi };
    Pas::new(code, p.m, &amplitudes(p.m, nu), 1.0).unwrap().with_bit_mapper(p.levels).unwrap()
}

/// Frames per batch of the runner in [`frame_errors`]: the stopping rule is checked after
/// every batch (48 frames, the rounds of the former loop on twelve threads).
const FRAMES_PER_BATCH: u64 = 48;

/// The frame errors of PAS over frames $0, 1, \dots$ at an SNR $E[X^{2}]/\sigma^{2}$ (dB,
/// one real dimension), until `min_errors` errors or `max_frames` frames, by batches of
/// [`FRAMES_PER_BATCH`] frames run on the threads of `com_sim`'s runner
/// ([`until_errors`]: the count does not depend on the number of threads). A frame is in
/// error when the decoded data bits (the dematcher's and the signs') differ from the
/// sent ones; when the LDPC decoder returns the sent information bits the dematcher
/// returns the sent bits, and it runs only on the other frames. Frame `f` draws its bits
/// and its noise from `batch_rng(seed, f)`: the SNRs share their random numbers.
fn frame_errors(pas: &Pas, snr_db: f64, seed: u64, min_errors: u64, max_frames: u64) -> (u64, u64) {
    let half = pas.matcher().composition().len();
    let levels: Vec<f64> = (0..2 * half).map(|i| (2 * i) as f64 - (2 * half - 1) as f64).collect();
    let es: f64 = pas.level_distribution().iter().zip(&levels).map(|(p, x)| p * x * x).sum();
    let noise_var = es / db2lin(snr_db);
    let sigma = noise_var.sqrt();
    let count = until_errors(seed, min_errors, max_frames, FRAMES_PER_BATCH, |rng| {
        let bits = source::bits(pas.info_bits(), rng);
        let x = pas.encode(&bits)?;
        let y: Vec<f64> = x.iter().map(|v| v + sigma * rng.gaussian()).collect();
        let sent = pas.encode_codeword(&bits)?;
        let (post, _) = pas.code().decode_llr(&pas.llr(&y, noise_var)?)?;
        let info_ok = pas.code().info_positions().iter().all(|&i| (post[i] < 0.0) == (sent[i] == 1));
        Ok((u64::from(!info_ok && pas.decode(&y, noise_var)? != bits), 1))
    })
    .unwrap();
    (count.errors, count.trials)
}

/// The SNR (dB) at which the line through the points `(snr, ln FER)` reaches `target`
/// (least squares; the points without errors are left out).
fn snr_at_fer(points: &[(f64, u64, u64)], target: f64) -> Option<f64> {
    let pts: Vec<(f64, f64)> =
        points.iter().filter(|(_, e, _)| *e > 0).map(|&(s, e, n)| (s, (e as f64 / n as f64).ln())).collect();
    if pts.len() < 2 {
        return None;
    }
    let k = pts.len() as f64;
    let (ms, mf) = (pts.iter().map(|p| p.0).sum::<f64>() / k, pts.iter().map(|p| p.1).sum::<f64>() / k);
    let sxy: f64 = pts.iter().map(|p| (p.0 - ms) * (p.1 - mf)).sum();
    let sxx: f64 = pts.iter().map(|p| (p.0 - ms) * (p.0 - ms)).sum();
    let slope = sxy / sxx;
    (slope < 0.0).then(|| ms + (target.ln() - mf) / slope)
}

/// PAS on the DVB-S2 LDPC codes against one point of Tables VI–X of Böcherer, Steiner &
/// Schulte (2015): the FER of com-rs at the SNR of the table and at ±0.05 dB (and
/// between them while fewer than two points have 5 errors), and the SNR at which it
/// reaches the FER of the table (log-linear fit). Returns the difference in dB from the
/// SNR of the table.
///
/// The system is that of the paper: the DVB-S2 LDPC code alone (no BCH), sum-product
/// decoding with 100 iterations (layered here, Sec. IX does not give the schedule),
/// bit-metric decoding with the priors of the amplitudes, the bit mapper $\pi$ of the
/// table (Sec. VII-B) and the BRGC labelling of Table II. The labels of com-rs differ
/// from those of the paper by the first amplitude bit (BRGC of the amplitude index
/// instead of the descending order of Table II) and the sign bit ($0$: $+$, the
/// paper's $b(1) = 1$): both give the same code up to a fixed codeword and a reflection
/// of the channel, which leave the FER of a sum-product decoder unchanged. The
/// compositions of the matcher are those of `Ccdm::from_distribution` (largest
/// remainders), which can differ from eq. (10) of Schulte & Böcherer (2016) by one
/// amplitude.
fn check_pas_point(p: &Point) -> f64 {
    let start = Instant::now();
    let pas = pas_of(p);
    let rate = pas.rate();
    assert!((rate - p.rate).abs() < 0.005, "Table {}: rate {rate} vs {}", p.table, p.rate);
    let label = format!("Table {} ({}-ASK, {}, R = {rate:.4})", p.table, 1 << p.m, p.code.name());
    let mut points: Vec<(f64, u64, u64)> = Vec::new();
    let run = |s: f64, points: &mut Vec<(f64, u64, u64)>| {
        let (e, n) = frame_errors(&pas, s, p.seed, 30, 6000);
        println!("{label}: {s:.3} dB, FER {e}/{n} = {:.2e}", e as f64 / n as f64);
        points.push((s, e, n));
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
    };
    for ds in [-0.05, 0.0, 0.05] {
        run(p.snr_db + ds, &mut points);
    }
    // a waterfall steeper than a decade per 0.05 dB: halve the step after the last point
    // with 5 errors
    while points.iter().filter(|q| q.1 >= 5).count() < 2 && points.len() < 6 {
        let i = points.iter().rposition(|q| q.1 >= 5).unwrap_or(0);
        if i + 1 >= points.len() {
            break;
        }
        run((points[i].0 + points[i + 1].0) / 2.0, &mut points);
    }
    let at = points.iter().find(|q| (q.0 - p.snr_db).abs() < 1e-9).unwrap();
    let snr = snr_at_fer(&points, p.fer);
    let delta = snr.map_or(f64::NAN, |s| s - p.snr_db);
    println!(
        "{label}: paper FER {:.1e} ± {:.1e} at {} dB; com-rs {:.2e} there, FER {:.1e} at {:.3} dB: {delta:+.3} dB \
         ({:.0} s)",
        p.fer,
        p.ci,
        p.snr_db,
        at.1 as f64 / at.2 as f64,
        p.fer,
        snr.unwrap_or(f64::NAN),
        start.elapsed().as_secs_f64()
    );
    delta
}

/// Table X of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 4-ASK, the
/// DVB-S2 code of rate 2/3, $\pi = (2, 1)$, the practical operating point (first row):
/// 1.13 bits per channel use, FER $5.2 \cdot 10^{-3} \pm 2.6 \cdot 10^{-3}$ at 6.70 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 5e-3: minutes in release, run with --ignored"]
fn bocherer_2015_table_x_pas_dvbs2() {
    let p = Point {
        table: "X",
        m: 2,
        code: CodeRate::R2_3,
        levels: &[2],
        rate: 1.13,
        snr_db: 6.70,
        fer: 5.2e-3,
        ci: 2.6e-3,
        seed: 100,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}

/// Table IX of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 8-ASK, the
/// DVB-S2 code of rate 3/4, $\pi = (3, 2, 1)$, the practical operating point (first
/// row): 1.85 bits per channel use, FER $1.5 \cdot 10^{-3} \pm 8 \cdot 10^{-4}$ at
/// 11.45 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 1.5e-3: minutes in release, run with --ignored"]
fn bocherer_2015_table_ix_pas_dvbs2() {
    let p = Point {
        table: "IX",
        m: 3,
        code: CodeRate::R3_4,
        levels: &[3, 2],
        rate: 1.85,
        snr_db: 11.45,
        fer: 1.5e-3,
        ci: 8.0e-4,
        seed: 101,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}

/// Table VIII of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 16-ASK, the
/// DVB-S2 code of rate 5/6, $\pi = (4, 3, 2, 1)$: the practical operating point (first
/// row), 2.96 bits per channel use, FER $2.0 \cdot 10^{-2} \pm 9.8 \cdot 10^{-3}$ at
/// 18.40 dB, and the adapted point of 2.50 bits per channel use, FER $7.2 \cdot
/// 10^{-3} \pm 3.6 \cdot 10^{-3}$ at 15.51 dB (rate adaption, eq. 62).
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at FERs of 1e-2: minutes in release, run with --ignored"]
fn bocherer_2015_table_viii_pas_dvbs2() {
    let first = Point {
        table: "VIII",
        m: 4,
        code: CodeRate::R5_6,
        levels: &[4, 3, 2],
        rate: 2.96,
        snr_db: 18.40,
        fer: 2.0e-2,
        ci: 9.8e-3,
        seed: 102,
    };
    let adapted = Point { rate: 2.50, snr_db: 15.51, fer: 7.2e-3, ci: 3.6e-3, seed: 103, ..first };
    for p in [first, adapted] {
        let delta = check_pas_point(&p);
        assert!(delta.abs() < 0.2, "{} bits: {delta} dB", p.rate);
    }
}

/// Table VII of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 32-ASK, the
/// DVB-S2 code of rate 5/6, $\pi = (4, 5, 2, 3, 1)$, the practical operating point
/// (first row): 3.62 bits per channel use, FER $2.1 \cdot 10^{-3} \pm 1.1 \cdot
/// 10^{-3}$ at 22.60 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 2e-3: a quarter of an hour in release, run with --ignored"]
fn bocherer_2015_table_vii_pas_dvbs2() {
    let p = Point {
        table: "VII",
        m: 5,
        code: CodeRate::R5_6,
        levels: &[4, 5, 2, 3],
        rate: 3.62,
        snr_db: 22.60,
        fer: 2.1e-3,
        ci: 1.1e-3,
        seed: 104,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}

/// Table VI of Böcherer, Steiner & Schulte (2015), arXiv:1502.02733v2: 64-ASK, the
/// DVB-S2 code of rate 9/10, $\pi = (4, 2, 5, 3, 6, 1)$, the practical operating point
/// (first row): 5.09 bits per channel use, FER $4.1 \cdot 10^{-3} \pm 2.1 \cdot
/// 10^{-3}$ at 31.80 dB.
#[test]
#[ignore = "DVB-S2 frames of 64 800 bits at a FER of 4e-3: a quarter of an hour in release, run with --ignored"]
fn bocherer_2015_table_vi_pas_dvbs2() {
    let p = Point {
        table: "VI",
        m: 6,
        code: CodeRate::R9_10,
        levels: &[4, 2, 5, 3, 6],
        rate: 5.09,
        snr_db: 31.80,
        fer: 4.1e-3,
        ci: 2.1e-3,
        seed: 105,
    };
    let delta = check_pas_point(&p);
    assert!(delta.abs() < 0.2, "{delta} dB");
}

//! The channel codes of com-rs against AFF3CT.
//!
//! AFF3CT (A. Cassagne et al., "AFF3CT: A Fast Forward Error Correction Toolbox!,"
//! *SoftwareX* 10, 100345, 2019) is an independent implementation of the same codes and
//! decoders. `tools/record_aff3ct.py` ran it on every configuration below (BPSK over
//! AWGN, the command line in each fixture) and recorded, for every $E_{b}/N_{0}$, its
//! frames, bit and frame errors in `validation/aff3ct/<name>.json`. These tests simulate
//! the same configuration with com-rs, at the same noise variance
//!
//! $$
//! \sigma^{2} = \frac{1}{2 R\, E_{b}/N_{0}},
//! $$
//!
//! with the rate $R$ AFF3CT converts $E_{b}/N_{0}$ with (`bit_rate`, the information
//! bits over the transmitted ones), and compare the two curves point by point.
//!
//! # Acceptance rule
//!
//! The frames of a point are independent trials: a point passes when the Wilson score
//! intervals at 99 % of the two frame error rates overlap. The bit errors come in
//! bursts (a decoding failure flips many bits), so the binomial interval of the bit
//! error rate would be far too narrow: its relative uncertainty is that of the number
//! of erroneous frames inflated by the spread of the bit errors per erroneous frame,
//! $\delta \approx z \sqrt{(1 + c_{v}^{2})/N_{\mathrm{fe}}}$, taken with $c_{v} = 1$; the
//! bit error rates pass when $|\ln(\mathrm{BER}_{\mathrm{com}} / \mathrm{BER}_{\mathrm{aff3ct}})|
//! \le \delta_{\mathrm{com}} + \delta_{\mathrm{aff3ct}}$ (two such intervals overlap).
//! Overlapping intervals at 99 % reject two equal rates with a probability of about
//! $3 \cdot 10^{-4}$ per point.
//!
//! A point of com-rs runs until the frame errors of AFF3CT (200) or a number of frames
//! set per configuration, so that the curves keep their statistics down to a frame
//! error rate of about $10^{-3}$ (below, the interval of com-rs widens and the test
//! stays valid). The configurations of a few seconds always run, with at most
//! [`QUICK`] frames and errors per point (`COMRS_VALIDATION_FULL=1` lifts the limit);
//! the others are `#[ignore]`d:
//!
//! ```text
//! cargo test --release -p com-validation --test aff3ct -- --ignored --nocapture
//! ```
//!
//! The LDPC matrices and the polar frozen sets that AFF3CT reads
//! (`validation/aff3ct/matrices/`, `validation/aff3ct/frozen/`) are written by com-rs;
//! [`inputs_given_to_aff3ct_are_those_of_com_rs`] checks them, and rewrites them with
//! `COMRS_RECORD_AFF3CT=1`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::bpsk_awgn_llr;
use comrs::core::signal::hard_decisions;
use comrs::fec::bch::Bch;
use comrs::fec::block::BlockCode;
use comrs::fec::conv::ConvolutionalCode;
use comrs::fec::crc::Crc;
use comrs::fec::interleaving::Interleaver;
use comrs::fec::ldpc::{CheckRule, LdpcCode};
use comrs::fec::polar::PolarCode;
use comrs::fec::rs::ReedSolomon;
use comrs::fec::turbo::TurboCode;
use comrs::prelude::*;
use comrs::sim::scenario::from_fn;
use comrs::sim::stats::z_score;
use comrs::systems::dvbs2::{CodeRate, Dvbs2Fec, FrameSize};
use serde_json::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const CONFIDENCE: f64 = 0.99;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("aff3ct")
}

// ------------------------------------------------------------------------- fixtures

/// One point of a recorded AFF3CT curve.
struct Point {
    ebn0_db: f64,
    frames: u64,
    bit_errors: u64,
    frame_errors: u64,
}

/// A recorded AFF3CT curve.
struct Fixture {
    command: String,
    version: String,
    rate: f64,
    bits_per_frame: u64,
    points: Vec<Point>,
}

fn fixture(name: &str) -> Fixture {
    let path = dir().join(format!("{name}.json"));
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e}; record it with tools/record_aff3ct.py", path.display())),
    )
    .unwrap();
    assert_eq!(doc["tool"], "AFF3CT");
    assert_eq!(doc["name"], name);
    let rate = &doc["bit_rate"];
    let u = |v: &Value| v.as_u64().unwrap();
    Fixture {
        command: doc["command"].as_str().unwrap().to_string(),
        version: doc["version"].as_str().unwrap().to_string(),
        rate: u(&rate[0]) as f64 / u(&rate[1]) as f64,
        bits_per_frame: u(&doc["bits_per_frame"]),
        points: doc["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| Point {
                ebn0_db: p["ebn0_db"].as_f64().unwrap(),
                frames: u(&p["frames"]),
                bit_errors: u(&p["bit_errors"]),
                frame_errors: u(&p["frame_errors"]),
            })
            .collect(),
    }
}

// ------------------------------------------------------------------------- links

/// One frame over BPSK and AWGN of variance `s2` per dimension: the bit errors among the
/// monitored bits, and the number of monitored bits.
type Link = Box<dyn Fn(f64, &mut Rng) -> Result<(u64, u64)> + Sync>;

fn errors(a: &[u8], b: &[u8]) -> u64 {
    a.iter().zip(b).filter(|(x, y)| x != y).count() as u64
}

/// A block code monitored on its information bits.
fn block<C: BlockCode + Sync + 'static>(code: C) -> Link {
    Box::new(move |s2, rng| {
        let info: Vec<u8> = (0..code.k()).map(|_| rng.bit()).collect();
        let c = code.encode_block(&info)?;
        let decided = code.decode_block(&bpsk_awgn_llr(&c, s2, rng))?;
        Ok((errors(&decided, &info), info.len() as u64))
    })
}

/// A binary BCH code, decoded from the hard decisions, on its information bits.
fn bch(code: Bch) -> Link {
    Box::new(move |s2, rng| {
        let info: Vec<u8> = (0..code.k()).map(|_| rng.bit()).collect();
        let c = code.encode(&info)?;
        let (decided, _) = code.decode(&hard_decisions(&bpsk_awgn_llr(&c, s2, rng)))?;
        Ok((errors(&decided, &info), info.len() as u64))
    })
}

/// An LDPC code monitored on all its coded bits (AFF3CT's `--sim-coded`).
fn ldpc(code: LdpcCode) -> Link {
    Box::new(move |s2, rng| {
        let info: Vec<u8> = (0..code.k()).map(|_| rng.bit()).collect();
        let c = code.encode_block(&info)?;
        let (post, _) = code.decode_llr(&bpsk_awgn_llr(&c, s2, rng))?;
        Ok((errors(&hard_decisions(&post), &c), c.len() as u64))
    })
}

fn decoder(mut code: LdpcCode, rule: CheckRule, layered: bool, iterations: usize) -> LdpcCode {
    code.decoder.rule = rule;
    code.decoder.layered = layered;
    code.decoder.max_iterations = iterations;
    code
}

/// The PEG constructions whose matrices AFF3CT reads, by file name.
const MATRICES: [(&str, usize, usize); 2] = [("peg_n576_k288.alist", 576, 288), ("peg_n648_k540.alist", 648, 540)];

fn matrix(file: &str) -> LdpcCode {
    LdpcCode::from_alist_file(dir().join("matrices").join(file)).unwrap()
}

/// The polar codes whose frozen sets AFF3CT reads: length, information bits, CRC, design
/// $E_{b}/N_{0}$ (dB).
fn polar_codes() -> Vec<PolarCode> {
    vec![
        PolarCode::new(256, 128, 2.0).unwrap(),
        PolarCode::new(1024, 512, 2.0).unwrap(),
        PolarCode::new(256, 128, 2.0).unwrap().with_crc(Crc::crc11()).unwrap(),
    ]
}

/// The frozen-set file of AFF3CT (`--enc-fb-gen-method FILE`): $N$, the noise type and
/// the design $\sigma$, then the $N$ positions from the most reliable, of which AFF3CT
/// takes the first $K$ (with the CRC bits) as information positions.
fn frozen_file(code: &PolarCode, design_ebn0_db: f64) -> (String, String) {
    let n = code.n();
    let info = code.info_positions();
    let sigma = (1.0 / (2.0 * code.code_rate() * db2lin(design_ebn0_db))).sqrt();
    let mut text = format!("{n}\nawgn\n{sigma:.6}\n");
    for p in info.iter().copied().chain((0..n).filter(|p| !info.contains(p))) {
        writeln!(text, "{p}").unwrap();
    }
    (format!("polar_n{n}_k{}.txt", info.len()), text)
}

/// Every configuration: its link, whether it runs by default, and the frames per point
/// at most (the simulation of a point stops at the frame errors of AFF3CT or there).
struct Case {
    name: &'static str,
    link: Link,
    max_frames: u64,
}

fn case(name: &'static str) -> Case {
    let rs = |m, poly, t, n| block(ReedSolomon::with_field(m, poly, 1, 1, t, n).unwrap());
    let (link, max_frames): (Link, u64) = match name {
        // AFF3CT's polynomials: x⁴ + x + 1, x⁶ + x + 1, x⁸ + x⁶ + x⁵ + x⁴ + 1 (com-rs
        // takes x⁸ + x⁴ + x³ + x² + 1 by default), roots α … α²ᵗ
        "rs_n15_k11_t2" => (rs(4, 0x13, 2, 15), 300_000),
        "rs_n63_k55_t4" => (rs(6, 0x43, 4, 63), 300_000),
        "rs_n255_k239_t8" => (rs(8, 0x171, 8, 255), 100_000),
        "bch_n63_k45_t3" => (bch(Bch::new(6, 0x43, 3, 63).unwrap()), 300_000),
        // x⁷ + x + 1 (com-rs: x⁷ + x³ + 1 by default)
        "bch_n127_k92_t5" => (bch(Bch::new(7, 0x83, 5, 127).unwrap()), 300_000),
        "bch_n255_k223_t4" => (bch(Bch::new(8, 0x171, 4, 255).unwrap()), 200_000),
        "conv_k7_k64_viterbi" => (block(ConvolutionalCode::k7(64).unwrap()), 300_000),
        "ldpc_peg_n576_k288_flooding_spa_i50" => {
            (ldpc(decoder(matrix("peg_n576_k288.alist"), CheckRule::SumProduct, false, 50)), 200_000)
        }
        "ldpc_peg_n576_k288_layered_nms075_i20" => {
            (ldpc(decoder(matrix("peg_n576_k288.alist"), CheckRule::NormalizedMinSum(0.75), true, 20)), 200_000)
        }
        "ldpc_peg_n576_k288_flooding_oms015_i20" => {
            (ldpc(decoder(matrix("peg_n576_k288.alist"), CheckRule::OffsetMinSum(0.15), false, 20)), 200_000)
        }
        "ldpc_peg_n648_k540_layered_spa_i20" => {
            (ldpc(decoder(matrix("peg_n648_k540.alist"), CheckRule::SumProduct, true, 20)), 200_000)
        }
        "ldpc_dvbs2_short_r1_2_layered_spa_i50" => {
            let code = Dvbs2Fec::new(FrameSize::Short, CodeRate::R1_2).unwrap().ldpc().clone();
            (ldpc(decoder(code, CheckRule::SumProduct, true, 50)), 120_000)
        }
        "polar_n256_k128_sc" => (block(polar_codes().swap_remove(0)), 200_000),
        "polar_n1024_k512_sc" => (block(polar_codes().swap_remove(1)), 50_000),
        "polar_n256_k128_cascl_l8_crc11" => (block(polar_codes().swap_remove(2).with_list(8).unwrap()), 50_000),
        "turbo_lte_k40_logmap_i8" => (block(TurboCode::lte(40).unwrap()), 200_000),
        "turbo_lte_k1024_maxlogmap_i6" => {
            // the QPP of K = 1024: f₁ = 31, f₂ = 64 (TS 36.212, Table 5.1.3-3)
            let mut code = TurboCode::with_interleaver(Interleaver::qpp(1024, 31, 64).unwrap()).unwrap();
            code.decoder.iterations = 6;
            code.decoder.exact = false;
            (block(code), 30_000)
        }
        other => panic!("no com-rs configuration for the fixture {other}"),
    };
    Case { name, link, max_frames }
}

// ------------------------------------------------------------------------- comparison

/// The relative half-width at `CONFIDENCE` of a bit error rate made of `frame_errors`
/// erroneous frames (see the module).
fn ber_spread(frame_errors: u64) -> f64 {
    z_score(CONFIDENCE) * (2.0 / frame_errors as f64).sqrt()
}

/// Frames per point at most, and frame errors to reach, of the configurations that
/// always run (`COMRS_VALIDATION_FULL=1` gives them the statistics of the others).
const QUICK: (u64, u64) = (10_000, 100);

/// Simulate the configuration `name` at every point of its fixture and compare; a
/// `quick` configuration stops its points at [`QUICK`].
fn compare(name: &'static str, quick: bool) {
    let fx = fixture(name);
    let Case { link, max_frames, .. } = case(name);
    let full = std::env::var("COMRS_VALIDATION_FULL").is_ok_and(|v| !v.is_empty() && v != "0");
    let grid = Grid::new().axis("ebn0_db", fx.points.iter().map(|p| p.ebn0_db).collect::<Vec<_>>());
    let rate = fx.rate;
    let bits = fx.bits_per_frame;
    let scenario = from_fn(name, move |cond, rng, rec| {
        let s2 = 1.0 / (2.0 * rate * db2lin(cond.f64("ebn0_db")?));
        let (be, monitored) = link(s2, rng)?;
        assert_eq!(monitored, bits, "com-rs monitors {monitored} bits per frame, AFF3CT {bits}");
        rec.errors("ber", be, monitored);
        rec.errors("fer", u64::from(be > 0), 1);
        Ok(())
    });
    let mut target = fx.points.iter().map(|p| p.frame_errors).max().unwrap_or(100);
    let mut max_frames = max_frames;
    if quick && !full {
        max_frames = max_frames.min(QUICK.0);
        target = target.min(QUICK.1);
    }
    let results = Runner::new()
        .seed(comrs::core::rng::fnv1a64(name.as_bytes()))
        .batch(512)
        .stop(Stop::min_errors("fer", target).or(Stop::max_trials(max_frames)))
        .run(&scenario, &grid)
        .unwrap();
    println!("{name} (AFF3CT {}): {}", fx.version, fx.command);
    println!("  Eb/N0 |      FER AFF3CT [99 %]              |      FER com-rs [99 %]              | ratio | BER ratio");
    let mut failures = Vec::new();
    let mut worst: f64 = 1.0;
    for (p, r) in fx.points.iter().zip(&results) {
        let ours = match r.metric("fer") {
            Some(Estimate::Rate(e)) => *e,
            other => panic!("{other:?}"),
        };
        let ber = r.metric("ber").unwrap().value();
        let theirs = ErrorRate { errors: p.frame_errors, total: p.frames };
        let (a_lo, a_hi) = theirs.wilson(CONFIDENCE);
        let (c_lo, c_hi) = ours.wilson(CONFIDENCE);
        let ratio = ours.rate() / theirs.rate();
        let their_ber = p.bit_errors as f64 / (p.frames * bits) as f64;
        let ber_ratio = ber / their_ber;
        if ours.errors >= 20 && p.frame_errors >= 20 {
            worst = worst.max(ratio.max(1.0 / ratio));
        }
        println!(
            "  {:5.2} | {:.3e} [{:.3e}, {:.3e}] {:>5} | {:.3e} [{:.3e}, {:.3e}] {:>5} | {ratio:5.2} | {ber_ratio:5.2}",
            p.ebn0_db,
            theirs.rate(),
            a_lo,
            a_hi,
            p.frame_errors,
            ours.rate(),
            c_lo,
            c_hi,
            ours.errors
        );
        if a_hi < c_lo || c_hi < a_lo {
            failures.push(format!("{} dB: FER {:.3e} vs AFF3CT {:.3e}", p.ebn0_db, ours.rate(), theirs.rate()));
        }
        if ours.errors > 0 && p.frame_errors > 0 {
            let allowed = ber_spread(ours.errors) + ber_spread(p.frame_errors);
            if ber_ratio.ln().abs() > allowed {
                failures.push(format!(
                    "{} dB: BER {ber:.3e} vs AFF3CT {their_ber:.3e} (ratio {ber_ratio:.2}, allowed e^±{allowed:.2})",
                    p.ebn0_db
                ));
            }
        }
    }
    println!("  largest FER ratio (points with 20 errors or more): {worst:.2}");
    assert!(failures.is_empty(), "{name} disagrees with AFF3CT:\n{}", failures.join("\n"));
}

// ------------------------------------------------------------------------- inputs

/// The LDPC matrices and the frozen sets of `validation/aff3ct/` are the constructions
/// of com-rs; `COMRS_RECORD_AFF3CT=1` writes them.
#[test]
fn inputs_given_to_aff3ct_are_those_of_com_rs() {
    let record = std::env::var("COMRS_RECORD_AFF3CT").is_ok_and(|v| !v.is_empty() && v != "0");
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    for (file, n, k) in MATRICES {
        let code = LdpcCode::peg(n, k, 3, 1).unwrap();
        assert_eq!(code.k(), k, "{file}: H has full rank");
        files.push((dir().join("matrices").join(file), code.to_alist()));
    }
    for code in polar_codes() {
        let (file, text) = frozen_file(&code, 2.0);
        files.push((dir().join("frozen").join(file), text));
    }
    for (path, text) in &files {
        if record {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        // a checkout with CRLF line endings (Git on Windows) holds the same file
        let on_disk = std::fs::read_to_string(path).unwrap_or_default().replace("\r\n", "\n");
        assert!(on_disk == *text, "{} is not the construction of com-rs (COMRS_RECORD_AFF3CT=1)", path.display());
    }
    // the reader gives back the constructions
    for (file, n, k) in MATRICES {
        assert_eq!(matrix(file).rows(), LdpcCode::peg(n, k, 3, 1).unwrap().rows(), "{file}");
    }
}

/// Every fixture has a configuration here, and every configuration a fixture.
#[test]
fn every_fixture_has_a_configuration() {
    let mut names: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().is_some_and(|x| x == "json")).then(|| p.file_stem().unwrap().to_string_lossy().into())
        })
        .collect();
    names.sort();
    let mut known: Vec<String> = CASES.iter().map(|c| (*c).to_string()).collect();
    known.sort();
    assert_eq!(names, known, "the fixtures of {} and the configurations of this file", Path::new("aff3ct").display());
    for name in CASES {
        let fx = fixture(name);
        assert!(fx.points.iter().all(|p| p.frames > 0 && p.frame_errors <= p.frames), "{name}");
        assert_eq!(case(name).name, name);
    }
}

/// The configurations, in the order of `tools/record_aff3ct.py`.
const CASES: [&str; 17] = [
    "rs_n15_k11_t2",
    "rs_n63_k55_t4",
    "rs_n255_k239_t8",
    "bch_n63_k45_t3",
    "bch_n127_k92_t5",
    "bch_n255_k223_t4",
    "conv_k7_k64_viterbi",
    "ldpc_peg_n576_k288_flooding_spa_i50",
    "ldpc_peg_n576_k288_layered_nms075_i20",
    "ldpc_peg_n576_k288_flooding_oms015_i20",
    "ldpc_peg_n648_k540_layered_spa_i20",
    "ldpc_dvbs2_short_r1_2_layered_spa_i50",
    "polar_n256_k128_sc",
    "polar_n1024_k512_sc",
    "polar_n256_k128_cascl_l8_crc11",
    "turbo_lte_k40_logmap_i8",
    "turbo_lte_k1024_maxlogmap_i6",
];

// ------------------------------------------------------------------------- the curves
//
// Every test cites AFF3CT (A. Cassagne et al., SoftwareX 10, 100345, 2019) and the
// reference of the code; the fixture holds the exact AFF3CT command line.

/// Reed–Solomon (15, 11), $t = 2$, over GF(16) ($x^{4} + x + 1$), Berlekamp–Massey
/// decoding of the hard decisions, against AFF3CT (Cassagne et al. 2019). Code: I. S.
/// Reed and G. Solomon, *J. SIAM* 8(2), 1960; Lin & Costello, *Error Control Coding*,
/// 2nd ed., 2004, ch. 7.
#[test]
fn rs_15_11_matches_aff3ct() {
    compare("rs_n15_k11_t2", true);
}

/// Reed–Solomon (63, 55), $t = 4$, over GF(64) ($x^{6} + x + 1$), against AFF3CT
/// (Cassagne et al. 2019). Code: Lin & Costello 2004, ch. 7.
#[test]
fn rs_63_55_matches_aff3ct() {
    compare("rs_n63_k55_t4", true);
}

/// Reed–Solomon (255, 239), $t = 8$, over GF(256) with AFF3CT's polynomial
/// $x^{8} + x^{6} + x^{5} + x^{4} + 1$ (not the default of com-rs), against AFF3CT
/// (Cassagne et al. 2019). Code: Lin & Costello 2004, ch. 7.
#[test]
fn rs_255_239_matches_aff3ct() {
    compare("rs_n255_k239_t8", true);
}

/// Binary BCH (63, 45), $t = 3$, Berlekamp–Massey decoding, against AFF3CT (Cassagne et
/// al. 2019). Code: Lin & Costello 2004, ch. 6.
#[test]
fn bch_63_45_matches_aff3ct() {
    compare("bch_n63_k45_t3", true);
}

/// Binary BCH (127, 92), $t = 5$, over GF(128) with $x^{7} + x + 1$, against AFF3CT
/// (Cassagne et al. 2019). Code: Lin & Costello 2004, ch. 6.
#[test]
fn bch_127_92_matches_aff3ct() {
    compare("bch_n127_k92_t5", true);
}

/// Binary BCH (255, 223), $t = 4$, over GF(256) with $x^{8} + x^{6} + x^{5} + x^{4} + 1$,
/// against AFF3CT (Cassagne et al. 2019). Code: Lin & Costello 2004, ch. 6.
#[test]
fn bch_255_223_matches_aff3ct() {
    compare("bch_n255_k223_t4", true);
}

/// The $(133, 171)_{8}$ convolutional code of constraint length 7, 64 information bits,
/// terminated, soft-decision Viterbi decoding, against AFF3CT (Cassagne et al. 2019).
/// Code: A. J. Viterbi, *IEEE Trans. Inf. Theory* 13(2), 1967.
#[test]
fn conv_k7_viterbi_matches_aff3ct() {
    compare("conv_k7_k64_viterbi", true);
}

/// A PEG (576, 288) LDPC code (Hu, Eleftheriou & Arnold, *IEEE Trans. Inf. Theory*
/// 51(1), 2005), flooding sum-product, 50 iterations, against AFF3CT (Cassagne et al.
/// 2019) decoding the same matrix.
#[test]
fn ldpc_flooding_sum_product_matches_aff3ct() {
    compare("ldpc_peg_n576_k288_flooding_spa_i50", true);
}

/// The PEG (576, 288) LDPC code, layered normalised min-sum ($\alpha = 0.75$, Chen &
/// Fossorier, *IEEE Trans. Commun.* 50(3), 2002), 20 iterations, against AFF3CT
/// (Cassagne et al. 2019).
#[test]
fn ldpc_layered_normalized_min_sum_matches_aff3ct() {
    compare("ldpc_peg_n576_k288_layered_nms075_i20", true);
}

/// The PEG (576, 288) LDPC code, flooding offset min-sum ($\beta = 0.15$), 20
/// iterations, against AFF3CT (Cassagne et al. 2019).
#[test]
fn ldpc_flooding_offset_min_sum_matches_aff3ct() {
    compare("ldpc_peg_n576_k288_flooding_oms015_i20", true);
}

/// A PEG (648, 540) LDPC code of rate 5/6, layered sum-product (Hocevar, *SiPS* 2004),
/// 20 iterations, against AFF3CT (Cassagne et al. 2019).
#[test]
fn ldpc_layered_sum_product_matches_aff3ct() {
    compare("ldpc_peg_n648_k540_layered_spa_i20", true);
}

/// The LDPC code of DVB-S2, short frame of rate 1/2 ($n = 16200$, $k = 7200$, ETSI EN
/// 302 307-1 V1.4.1, Sec. 5.3.2 and Table B.1), built by com-rs and by AFF3CT from the
/// tables of the standard, layered sum-product, 50 iterations, against AFF3CT (Cassagne
/// et al. 2019).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn ldpc_dvbs2_short_matches_aff3ct() {
    compare("ldpc_dvbs2_short_r1_2_layered_spa_i50", false);
}

/// The polar (256, 128) code (Arıkan, *IEEE Trans. Inf. Theory* 55(7), 2009) with the
/// frozen set of com-rs, successive cancellation, against AFF3CT (Cassagne et al. 2019)
/// reading the same frozen set.
#[test]
fn polar_256_sc_matches_aff3ct() {
    compare("polar_n256_k128_sc", true);
}

/// The polar (1024, 512) code with the frozen set of com-rs, successive cancellation,
/// against AFF3CT (Cassagne et al. 2019).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn polar_1024_sc_matches_aff3ct() {
    compare("polar_n1024_k512_sc", false);
}

/// The polar (256, 128) code with the CRC11 of 5G NR, CRC-aided SC list decoding with
/// $L = 8$ (Tal & Vardy, *IEEE Trans. Inf. Theory* 61(5), 2015), against AFF3CT
/// (Cassagne et al. 2019).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn polar_256_ca_scl_matches_aff3ct() {
    compare("polar_n256_k128_cascl_l8_crc11", false);
}

/// The LTE turbo code with $K = 40$ (QPP $f_{1} = 3$, $f_{2} = 10$, 3GPP TS 36.212,
/// Table 5.1.3-3), 8 iterations of log-MAP, against AFF3CT (Cassagne et al. 2019).
/// Code: Berrou, Glavieux & Thitimajshima, *ICC* 1993.
#[test]
fn turbo_lte_40_log_map_matches_aff3ct() {
    compare("turbo_lte_k40_logmap_i8", true);
}

/// The LTE turbo code with $K = 1024$ (QPP $f_{1} = 31$, $f_{2} = 64$, 3GPP TS 36.212,
/// Table 5.1.3-3), 6 iterations of max-log-MAP, against AFF3CT (Cassagne et al. 2019).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn turbo_lte_1024_max_log_map_matches_aff3ct() {
    compare("turbo_lte_k1024_maxlogmap_i6", false);
}

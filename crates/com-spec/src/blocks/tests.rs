//! Tests of the registry and of links of several families.

use super::*;
use com_math::{C64, cis, power};
use serde_json::Map;
use serde_json::json;

fn link(blocks: &[Json]) -> Vec<Box<dyn Block>> {
    blocks.iter().map(|b| build(b, &BuildContext::default()).unwrap()).collect()
}

/// The frame after the blocks of `link` with the generator seeded by `seed` (for the
/// tests of the block modules).
pub(super) fn run(link: &[Json], seed: u64) -> Frame {
    let mut rng = Rng::new(seed);
    let mut s = Frame::default();
    for b in link {
        build(b, &BuildContext::default()).unwrap().apply(&mut s, &mut rng).unwrap();
    }
    s
}

#[test]
fn ce_ofdm_blocks_round_trip_and_record_the_theory() {
    let blocks = link(&[
        json!({"type": "source", "constellation": "16-QAM", "n_symbols": 640}),
        json!({"type": "ce_ofdm_tx", "n_data": 64, "oversampling": 2, "h": 0.1, "n_cp": 4}),
        json!({"type": "ce_ofdm_rx", "n_data": 64, "oversampling": 2, "h": 0.1, "n_cp": 4, "esn0_db": 20.0}),
    ]);
    let mut s = Frame::default();
    let mut rng = Rng::new(1);
    blocks[0].apply(&mut s, &mut rng).unwrap();
    blocks[1].apply(&mut s, &mut rng).unwrap();
    assert_eq!(s.sps, (2.0 * 2.0 * 65.0 + 4.0) / 64.0);
    assert!(s.samples.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
    blocks[2].apply(&mut s, &mut rng).unwrap();
    assert_eq!((s.sps, s.samples.len()), (1.0, 640));
    assert!(s.samples.iter().zip(&s.symbols).all(|(a, b)| (a - b).norm() < 1e-9));
    let names: Vec<&str> = s.values.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"ce_ofdm_rx.ser_theory") && names.contains(&"ce_ofdm_rx.ber_theory"));
}

#[test]
fn every_accepted_parameter_is_documented() {
    // serde names the accepted fields when it refuses an unknown one
    for info in registry() {
        let v = json!({"type": info.name, "zz_unknown": 1});
        let msg = match build(&v, &BuildContext::default()) {
            Ok(_) => panic!("block {} accepted an unknown parameter", info.name),
            Err(e) => e.to_string(),
        };
        let Some(rest) = msg.split("expected one of ").nth(1).or_else(|| msg.split("expected ").nth(1)) else {
            panic!("block {}: {msg}", info.name);
        };
        let accepted: Vec<&str> = rest.split('`').skip(1).step_by(2).collect();
        let documented: Vec<&str> = info.params.iter().map(|p| p.name).collect();
        let mut a = accepted.clone();
        let mut d = documented.clone();
        a.sort_unstable();
        d.sort_unstable();
        assert_eq!(a, d, "block {}: accepted {accepted:?}, documented {documented:?}", info.name);
    }
}

/// A value of the right type for a documented parameter (`None`: leave it out).
#[allow(clippy::match_same_arms, reason = "one arm per family of blocks")]
fn example(b: &str, p: &str) -> Option<Json> {
    Some(match (b, p) {
        ("capture", _) => return None,
        ("dump", "dir") => json!("results/signals"),
        ("dump", "datatype") => json!("cf64"),
        ("dump", "trials") => json!(2),
        ("cpm_tx" | "cpm_rx", "h") => json!("1/2"),
        ("cpm_tx" | "cpm_rx", "pulse") => json!("gmsk"),
        ("cpm_tx" | "cpm_rx", "precode") => json!(false),
        ("cpm_rx", "detector") => json!("ml"),
        ("pas_encode" | "pas_decode", "constellation") => json!("16-QAM"),
        ("dvbs2_tx" | "dvbs2_rx", "modulation") => json!("8PSK"),
        ("dvbs2_tx" | "dvbs2_rx", "rate") => json!("3/5"),
        ("dvbs2_tx" | "dvbs2_rx", "frame") => json!("short"),
        ("dvbs2_tx" | "dvbs2_rx", "pilots") => json!(false),
        ("dvbs2_tx" | "dvbs2_rx", "scrambling_code" | "frames" | "iterations") => json!(2),
        (_, "constellation") => json!("QPSK"),
        (_, "precision") => json!("f32"),
        (_, "fiber") => json!({"n_spans": 2}),
        (_, "kind") => json!("rrc"),
        (_, "reference") => json!("signal"),
        (_, "origin") => json!("zero"),
        (_, "alignment") => json!("causal"),
        (_, "amplification") => json!("edfa"),
        ("classify", "method") => json!("cumulants"),
        (_, "method") => json!("max_variance"),
        ("noise_estimate", "mode") => json!("decisions"),
        (_, "algorithm") => json!("cma"),
        ("mimo_tx", "scheme") => json!("spatial"),
        ("wdm_tx", "channel") => json!(1),
        ("tdl", "profile") => json!("EVA"),
        ("cdl", "profile") => json!("CDL-C"),
        ("ofdm_tx", "pilots") => json!({"pattern": "comb", "spacing": 4}),
        ("ofdm_tx", "n_fft") => json!(64),
        ("ofdm_tx", "n_subcarriers") => json!(48),
        ("ofdm_rx", "estimation") => json!("ls"),
        (_, "detector") => json!("mmse"),
        ("probe", "metrics") => json!(["power_dbm", "papr_db"]),
        ("classify", "candidates") => json!(["QPSK", "16-QAM"]),
        ("pas_encode" | "pas_decode", "n") => json!(96),
        ("pas_encode" | "pas_decode", "k") => return None, // n (m-1)/m by default
        ("pas_encode" | "pas_decode", "rule") => json!("sum_product"),
        ("pas_encode" | "pas_decode", "layered") => json!(true),
        ("pas_encode" | "pas_decode", "var_degree" | "iterations") => json!(3),
        ("probe", "scale") => json!("power"),
        ("pmd", "theta_deg") => return None, // exclusive with the random realisation
        ("fec_encode" | "fec_decode", "code") => json!({"type": "ldpc", "n": 64, "k": 32}),
        (_, "rx_correlation" | "tx_correlation") => json!(0.3),
        ("ce_ofdm_tx" | "ce_ofdm_rx", "h") => json!(0.1),
        ("ce_ofdm_rx", "unwrap") => json!("stream"),
        ("ce_ofdm_rx", "receiver") => json!("enhanced"),
        (_, "h") => json!([[1.0, 0.0]]),
        (_, "t") => json!([1.0, 0.0, 0.0, 1.0]),
        (_, "beta") => json!([0.1, 0.0]),
        ("phase_noise", "sigma_rad") | ("cfo", "omega_rad") => return None, // exclusive with the others
        (
            _,
            "centred" | "compensate_loss" | "noise" | "nyquist" | "refined" | "exclude" | "nonlinear" | "dc_null"
            | "llr" | "interleave" | "max_log",
        ) => {
            json!(true)
        }
        (
            _,
            "n_symbols" | "sps" | "n_taps" | "sps_out" | "offset" | "passes" | "n" | "b" | "m" | "steps_per_span"
            | "start" | "length" | "max_delay" | "symmetry" | "window" | "n_data" | "oversampling" | "n_cp" | "n_tx"
            | "n_rx" | "coherence_samples" | "k" | "seed" | "blocks" | "skip_start" | "skip_end" | "n_channels"
            | "max_iterations" | "iterations" | "pols" | "kikuchi" | "sections" | "bits",
        ) => json!(2),
        _ => json!(1.0),
    })
}

#[test]
fn every_documented_parameter_is_accepted() {
    for info in registry() {
        let mut m = Map::new();
        m.insert("type".into(), json!(info.name));
        for p in &info.params {
            if let Some(v) = example(info.name, p.name) {
                m.insert(p.name.into(), v);
            }
        }
        if info.name == "capture" {
            continue;
        }
        if let Err(e) = build(&Json::Object(m), &BuildContext::default()) {
            panic!("block {}: {e}", info.name);
        }
    }
}

#[test]
fn every_default_is_a_valid_value() {
    // the required parameters only: every other one takes its declared default
    for info in registry() {
        if info.name == "capture" {
            continue;
        }
        let mut m = Map::new();
        m.insert("type".into(), json!(info.name));
        for p in info.params.iter().filter(|p| matches!(p.default, ParamDefault::Required)) {
            m.insert(p.name.into(), example(info.name, p.name).unwrap_or(json!(1.0)));
        }
        // one of two exclusive parameters
        match info.name {
            "phase_noise" => m.insert("linewidth_hz".into(), json!(1e5)),
            "cfo" => m.insert("offset_hz".into(), json!(1e3)),
            _ => None,
        };
        if let Err(e) = build(&Json::Object(m), &BuildContext::default()) {
            panic!("block {}: {e}", info.name);
        }
        for p in &info.params {
            if let ParamDefault::Value(v, _) = &p.default {
                assert!(p.check(info.name, v).is_ok(), "{}.{}: default {v} out of its own range", info.name, p.name);
            }
        }
    }
}

#[test]
fn accepted_values_and_bounds_are_checked() {
    let ctx = BuildContext::default();
    let e = build(&json!({"type": "awgn", "snr_db": 3, "reference": "peak"}), &ctx).err().unwrap().to_string();
    assert!(e.contains("reference") && e.contains("one of"), "{e}");
    // accepted values are compared without case
    assert!(build(&json!({"type": "dvbs2_tx", "modulation": "8psk", "rate": "3/5"}), &ctx).is_ok());
    let e = build(&json!({"type": "source", "pols": 3}), &ctx).err().unwrap().to_string();
    assert!(e.contains("pols must be >= 1 and <= 2"), "{e}");
    let e = build(&json!({"type": "adaptive_equalizer", "mu": [1e-3, 0.0]}), &ctx).err().unwrap().to_string();
    assert!(e.contains("mu must be > 0"), "{e}");
    let e = build(&json!({"type": "awgn"}), &ctx).err().unwrap().to_string();
    assert!(e.contains("missing field `snr_db`"), "{e}");
    let e = build(&json!({"type": "awgn", "snr_db": "high"}), &ctx).err().unwrap().to_string();
    assert!(e.contains("snr_db") && e.contains("invalid type"), "{e}");
}

#[test]
fn accepted_values_are_parsed_without_case_as_they_are_checked() {
    let ctx = BuildContext::default();
    // a typed value (serde, lowercase), a string matched by the block, a list of them
    for b in [
        json!({"type": "awgn", "snr_db": 3, "reference": "Unit"}),
        json!({"type": "pulse_shape", "kind": "RRC", "sps": 2}),
        json!({"type": "decimate", "method": "Max_Variance"}),
        json!({"type": "matched_filter", "kind": "Rc"}),
    ] {
        assert!(build(&b, &ctx).is_ok(), "{b}: {:?}", build(&b, &ctx).err().map(|e| e.to_string()));
    }
    // the same value as written in the declaration gives the same block
    let lower = run(
        &[json!({"type": "source", "n_symbols": 64}), json!({"type": "awgn", "snr_db": 3, "reference": "unit"})],
        1,
    );
    let upper = run(
        &[json!({"type": "source", "n_symbols": 64}), json!({"type": "awgn", "snr_db": 3, "reference": "UNIT"})],
        1,
    );
    assert_eq!(lower.samples, upper.samples);
}

#[test]
fn the_documentation_shows_types_units_and_defaults() {
    let info = |n: &str| registry().into_iter().find(|b| b.name == n).unwrap();
    let awgn = info("awgn");
    let snr = &awgn.params[0];
    assert_eq!((snr.name, snr.unit, snr.default_text().as_str()), ("snr_db", "dB", "required"));
    assert_eq!(snr.schema, json!({"type": "number"}));
    assert_eq!(awgn.params[2].default_text(), "symbol rate");
    assert_eq!(info("ddpll").params[1].default_text(), "1/(2π 10 MHz)");
    assert_eq!(info("pulse_shape").params[0].default_text(), "rrc");
    assert_eq!(info("rx_iq").params[0].schema["maxItems"], json!(4));
    let text = awgn.describe();
    assert!(text.contains("needs: samples; gives: noise") && text.contains("[dB]"), "{text}");
    assert_eq!(awgn.to_json()["required"], json!(["snr_db"]));
}

#[test]
fn links_are_checked_before_they_run() {
    assert!(check_link(&["source", "awgn", "noise_estimate", "demap"]).is_ok());
    assert!(check_link(&["fec_encode", "map", "awgn_ebn0", "demap", "fec_decode"]).is_ok());
    assert!(check_link(&["source", "mimo_tx", "ofdm_tx", "tdl", "awgn", "ofdm_rx"]).is_ok());
    let e = check_link(&["fec_encode", "map", "demap"]).unwrap_err().to_string();
    assert!(e.contains("link block 3 (demap) needs noise") && e.contains("awgn"), "{e}");
    let e = check_link(&["source", "awgn", "ofdm_rx"]).unwrap_err().to_string();
    assert!(e.contains("needs ofdm") && e.contains("ofdm_tx"), "{e}");
    let e = check_link(&["source", "awgn", "fec_decode"]).unwrap_err().to_string();
    assert!(e.contains("needs llr"), "{e}");
}

#[test]
fn the_json_schema_describes_every_block() {
    let schema = json_schema();
    let alts = schema["oneOf"].as_array().unwrap();
    for info in registry() {
        let alt = alts.iter().find(|a| a["title"] == json!(info.name)).unwrap();
        assert_eq!(alt["properties"].as_object().unwrap().len(), info.params.len() + 2, "{}", info.name);
    }
    let awgn = alts.iter().find(|a| a["title"] == json!("awgn")).unwrap();
    let snr = &awgn["properties"]["snr_db"]["anyOf"][0];
    assert_eq!((snr["type"].as_str(), snr["x-unit"].as_str()), (Some("number"), Some("dB")));
    assert_eq!(awgn["properties"]["reference"]["anyOf"][0]["enum"], json!(["signal", "unit"]));
    let rx = alts.iter().find(|a| a["title"] == json!("ce_ofdm_rx")).unwrap();
    assert_eq!(rx["properties"]["unwrap"]["anyOf"][0]["enum"], json!(["none", "stream"]));
    let probe = alts.iter().find(|a| a["title"] == json!("probe")).unwrap();
    assert_eq!(probe["properties"]["metrics"]["anyOf"][0]["items"]["type"], json!("string"));
    // a block without a required parameter may be written as its name alone
    assert!(alts.iter().any(|a| a["const"] == json!("decimate")));
}

#[test]
fn unknown_blocks_and_parameters_are_errors() {
    let ctx = BuildContext::default();
    let e = build(&json!({"type": "awgnn"}), &ctx).err().unwrap().to_string();
    assert!(e.contains("unknown block") && e.contains("awgn_ebn0"), "{e}");
    let e = build(&json!({"type": "awgn", "snr": 3}), &ctx).err().unwrap().to_string();
    assert!(e.contains("awgn") && e.contains("snr"), "{e}");
    let e = build(&json!({"type": "awgn"}), &ctx).err().unwrap().to_string();
    assert!(e.contains("snr_db"), "{e}");
}

#[test]
fn shared_parameter_groups_keep_unknown_names_rejected() {
    let ctx = BuildContext::default();
    // the detector of mimo_rx and ofdm_rx, the LDPC code of fec_encode and pas_encode:
    // one declaration each, flattened into the blocks
    for (block, extra) in [
        (json!({"type": "ofdm_rx", "detector": "zf"}), "rule"),
        (json!({"type": "mimo_rx", "detector": "zf"}), "estimation"),
        (json!({"type": "pas_encode"}), "detector"),
        (json!({"type": "fec_encode", "code": {"type": "ldpc", "n": 128, "k": 64}}), "list"),
    ] {
        let mut wrong = block.clone();
        let target = if block["type"] == "fec_encode" { &mut wrong["code"] } else { &mut wrong };
        target[extra] = json!(1);
        let e = build(&wrong, &ctx).err().unwrap().to_string();
        assert!(e.contains(&format!("unknown field `{extra}`")), "{e}");
    }
    let names = |b: &str| -> Vec<&str> {
        registry().into_iter().find(|i| i.name == b).unwrap().params.iter().map(|p| p.name).collect()
    };
    assert_eq!(names("ofdm_rx"), ["estimation", "detector", "k", "iterations", "llr"]);
    assert_eq!(names("mimo_rx"), ["detector", "k", "iterations", "llr"]);
    let pas = names("pas_encode");
    assert_eq!(&pas[..9], ["constellation", "mb_lambda", "n", "k", "var_degree", "seed", "rule", "alpha", "layered"]);
}

#[test]
fn the_detectors_of_the_blocks_are_those_of_com_mimo() {
    use com_mimo::receiver::DetectorKind;
    for b in ["mimo_rx", "ofdm_rx"] {
        let info = registry().into_iter().find(|i| i.name == b).unwrap();
        let p = info.params.iter().find(|p| p.name == "detector").unwrap();
        assert_eq!(p.choices, DetectorKind::NAMES, "{b}");
    }
    // zf_sic was missing from the hand-written list
    let s = run(
        &[
            json!({"type": "source", "constellation": "QPSK", "n_symbols": 64}),
            json!({"type": "mimo_tx", "n_tx": 2}),
            json!({"type": "rayleigh", "n_rx": 2}),
            json!({"type": "awgn", "snr_db": 30}),
            json!({"type": "mimo_rx", "detector": "zf_sic"}),
        ],
        1,
    );
    assert_eq!(s.samples.len(), 64);
}

#[test]
fn the_single_polarisation_blocks_are_declared_so() {
    let mut per_pol: Vec<&str> = registry().iter().filter(|b| b.per_polarization).map(|b| b.name).collect();
    per_pol.sort_unstable();
    let mut expected = [
        "pulse_shape",
        "fir",
        "rx_iq",
        "tx_iq",
        "fiber",
        "ssfm",
        "edfa",
        "cdc",
        "dbp",
        "resample",
        "matched_filter",
        "decimate",
        "gardner",
        "foe",
        "bps",
        "viterbi",
        "ddpll",
        "adc",
        "dac",
    ];
    expected.sort_unstable();
    assert_eq!(per_pol, expected);
    // none of them starts a link
    assert!(registry().iter().all(|b| !(b.starts && b.per_polarization)));
}

#[test]
fn the_blocks_that_need_symbols_say_so_with_one_message() {
    let ctx = BuildContext::default();
    for b in [json!({"type": "mimo_tx", "n_tx": 2}), json!("ofdm_tx"), json!({"type": "cpm_tx", "h": "1/2"})] {
        let mut s = Frame::default();
        let mut rng = Rng::new(1);
        build(&json!({"type": "source", "constellation": "QPSK", "n_symbols": 64}), &ctx)
            .unwrap()
            .apply(&mut s, &mut rng)
            .unwrap();
        build(&json!({"type": "pulse_shape", "sps": 2}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        let e = build(&b, &ctx).unwrap().apply(&mut s, &mut rng).unwrap_err().to_string();
        assert!(e.contains("needs the signal at 1 sample per symbol, it has 2 samples per symbol"), "{e}");
    }
}

#[test]
fn constellations_by_name() {
    for (s, kind, m) in [
        ("16-QAM", ConstellationKind::Qam, 16),
        ("qpsk", ConstellationKind::Qam, 4),
        ("8-PSK", ConstellationKind::Psk, 8),
        ("bpsk", ConstellationKind::Psk, 2),
        ("4-pam", ConstellationKind::Pam, 4),
    ] {
        let c = parse_constellation(&json!(s)).unwrap();
        assert_eq!((c.kind(), c.order()), (kind, m), "{s}");
        assert!((c.energy() - 1.0).abs() < 1e-12);
    }
    assert!(parse_constellation(&json!({"kind": "psk", "m": 8})).is_ok());
    assert!(parse_constellation(&json!("sixteen")).is_err());
}

#[test]
fn shaping_and_matched_filtering_recover_the_symbols() {
    let blocks = link(&[
        json!({"type": "source", "constellation": "16-QAM", "n_symbols": 2000}),
        json!({"type": "pulse_shape", "sps": 4, "rolloff": 0.2}),
        json!({"type": "matched_filter", "rolloff": 0.2}),
        json!("decimate"),
        json!({"type": "normalize"}),
    ]);
    let mut s = Frame::default();
    let mut rng = Rng::new(1);
    for b in &blocks {
        b.apply(&mut s, &mut rng).unwrap();
    }
    assert_eq!((s.sps, s.samples.len()), (1.0, 2000));
    let err: f64 = s.samples[100..1900].iter().zip(&s.symbols[100..1900]).map(|(a, b)| (a - b).norm_sqr()).sum();
    assert!(err / 1800.0 < 1e-3, "{}", err / 1800.0);
}

#[test]
fn the_impairments_follow_the_step_centred_unit_and_causal_conventions() {
    // phase noise of origin step: the first sample is rotated too
    for (origin, rotated) in [("zero", false), ("step", true)] {
        let mut p = Frame { samples: vec![C64::new(1.0, 0.0); 4], sps: 1.0, symbol_rate: 1.0, ..Frame::default() };
        build(&json!({"type": "phase_noise", "sigma_rad": 0.1, "origin": origin}), &BuildContext::default())
            .unwrap()
            .apply(&mut p, &mut Rng::new(1))
            .unwrap();
        assert_eq!(p.samples[0] != C64::new(1.0, 0.0), rotated, "{origin}");
    }
    // causal FIR, centred CFO, unit-energy AWGN reference
    let mut s = Frame {
        samples: vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        sps: 1.0,
        symbol_rate: 1.0,
        ..Frame::default()
    };
    let mut rng = Rng::new(0);
    build(&json!({"type": "fir", "h": [[0.5, 0.0], [0.0, 0.25]]}), &BuildContext::default())
        .unwrap()
        .apply(&mut s, &mut rng)
        .unwrap();
    assert_eq!(&s.samples[..2], &[C64::new(0.5, 0.0), C64::new(0.0, 0.25)]);
    build(&json!({"type": "cfo", "omega_rad": 0.1, "centred": true}), &BuildContext::default())
        .unwrap()
        .apply(&mut s, &mut rng)
        .unwrap();
    assert!((s.samples[0] - C64::new(0.5, 0.0) * cis(-0.2)).norm() < 1e-15);
    let mut q = Frame { samples: vec![C64::new(0.0, 0.0); 200_000], sps: 1.0, symbol_rate: 1.0, ..Frame::default() };
    build(&json!({"type": "awgn", "snr_db": 10.0, "reference": "unit"}), &BuildContext::default())
        .unwrap()
        .apply(&mut q, &mut rng)
        .unwrap();
    assert!((power(&q.samples) / 0.1 - 1.0).abs() < 0.02);
}

#[test]
fn frame_sync_aligns_and_excludes_the_preamble() {
    let blocks = link(&[
        json!({"type": "source", "constellation": "QPSK", "n_symbols": 1000}),
        json!({"type": "frame_sync", "start": 10, "length": 100}),
    ]);
    let mut s = Frame::default();
    let mut rng = Rng::new(3);
    blocks[0].apply(&mut s, &mut rng).unwrap();
    s.samples.rotate_right(5);
    s.samples.iter_mut().for_each(|v| *v *= C64::new(0.0, -1.0));
    blocks[1].apply(&mut s, &mut rng).unwrap();
    assert_eq!(s.values[0], ("frame_sync.delay".to_string(), 5.0));
    assert!((s.samples[500] - s.symbols[500]).norm() < 1e-12);
    assert!(s.excluded[10] && s.excluded[109] && !s.excluded[110]);
}

#[test]
fn a_classic_receiver_declares_its_posterior_with_noise_estimate() {
    use crate::{RunOptions, Spec, run};
    let spec = |mode: &str, report: &str| {
        Spec::from_yaml(&format!(
            "name: post\nseed: 2\nlink: [{{type: source, constellation: 16-QAM, n_symbols: 4000}}, \
             {{type: awgn, snr_db: 14}}, {{type: noise_estimate, mode: {mode}}}]\n\
             metrics: {{report: [{report}], scale: none}}\nstop: {{max_trials: 8}}"
        ))
        .unwrap()
    };
    // with the true variance (known symbols), the posterior is the channel's
    let r = run(&spec("known", "mi, air, air_opt, posterior_calibration"), &RunOptions::default()).unwrap();
    let v = |k: &str| r[0].metric(k).unwrap().value();
    assert!((v("air") - v("mi")).abs() < 0.01, "air {} mi {}", v("air"), v("mi"));
    assert!((v("posterior_calibration") - 1.0).abs() < 1e-9 && v("air_opt") >= v("air"));
    assert!((v("noise_estimate.sigma2") / 10f64.powf(-1.4) - 1.0).abs() < 0.05);
    // decisions underestimate the variance: an over-confident posterior
    let r = run(&spec("decisions", "air, air_opt, posterior_calibration"), &RunOptions::default()).unwrap();
    let v = |k: &str| r[0].metric(k).unwrap().value();
    assert!(v("posterior_calibration") < 1.0 && v("air_opt") > v("air"));
    // without a declared posterior the air metrics are a configuration error
    let bare = Spec::from_yaml(
        "name: bare\nlink: [source, {type: awgn, snr_db: 14}]\nmetrics: {report: [air]}\nstop: {max_trials: 1}",
    )
    .unwrap();
    let e = run(&bare, &RunOptions::default()).unwrap_err();
    assert!(matches!(e, Error::Config(_)) && e.to_string().contains("posterior"), "{e}");
}

#[test]
fn fibre_parameters_merge_the_specification_and_the_block() {
    let ctx = BuildContext { fiber: Some(json!({"n_spans": 10, "span_km": 100.0})), ..BuildContext::default() };
    let f = ctx.fiber(Some(&json!({"span_km": 50.0}))).unwrap();
    assert_eq!((f.n_spans, f.span_km, f.d_ps_nm_km), (10, 50.0, 17.0));
    assert!(ctx.fiber(Some(&json!({"spans": 3}))).is_err());
}

#[test]
fn the_keys_of_the_nested_mappings_are_documented_and_checked() {
    let ctx = BuildContext::default();
    let info = |name: &str| registry().into_iter().find(|b| b.name == name).unwrap();
    // the fibre of the optical blocks, its Raman pumps, the pilots, the exponential profile
    let manakov = info("manakov");
    let fiber = manakov.params.iter().find(|p| p.name == "fiber").unwrap();
    let keys: Vec<&str> = fiber.fields.iter().map(|p| p.name).collect();
    assert_eq!(keys, ["alpha_db_km", "d_ps_nm_km", "gamma_w_km", "span_km", "n_spans", "fc_hz", "nf_db", "raman"]);
    let text = manakov.describe();
    assert!(text.contains("d_ps_nm_km") && text.contains("raman"), "{text}");
    let schema = fiber.json_schema();
    assert_eq!(schema["anyOf"][0]["properties"]["span_km"]["anyOf"][0]["default"], json!(80.0));
    assert_eq!(schema["anyOf"][0]["additionalProperties"], json!(false));
    let raman = fiber.fields.iter().find(|p| p.name == "raman").unwrap();
    let pumps = raman.fields.iter().find(|p| p.name == "pumps").unwrap();
    assert!(pumps.fields.iter().any(|p| p.name == "wavelength_nm"));
    let pilots = info("ofdm_tx").params.into_iter().find(|p| p.name == "pilots").unwrap();
    assert!(pilots.fields.iter().any(|p| p.name == "pattern" && p.choices.contains(&"comb")));
    let tdl = info("tdl").params.into_iter().find(|p| p.name == "profile").unwrap();
    assert_eq!(tdl.fields.iter().map(|p| p.name).collect::<Vec<_>>(), ["rms_delay_s", "resolution_s"]);
    // an unknown key is refused, the accepted values are read without case
    let e = build(&json!({"type": "ssfm", "fiber": {"span": 80}}), &ctx).err().unwrap().to_string();
    assert!(e.contains("unknown field `span`") && e.contains("span_km"), "{e}");
    let pump = json!({"raman": {"pumps": [{"wavelength_nm": 1450, "power_mw": 100, "direction": "Counter"}]}});
    assert!(build(&json!({"type": "manakov", "fiber": pump}), &ctx).is_ok());
    let pump = json!({"raman": {"pumps": [{"wavelength_nm": 1450, "power_mw": 100, "side": "co"}]}});
    assert!(build(&json!({"type": "manakov", "fiber": pump}), &ctx).is_err());
    for pilots in [json!("NONE"), json!({"pattern": "Comb", "spacing": 2})] {
        let b = json!({"type": "ofdm_tx", "n_fft": 64, "n_subcarriers": 48, "pilots": pilots});
        assert!(build(&b, &ctx).is_ok(), "{pilots}");
    }
    let e = build(&json!({"type": "tdl", "profile": {"rms_delay_s": 0.0, "resolution_s": 1e-7}}), &ctx);
    assert!(e.err().unwrap().to_string().contains("rms_delay_s must be > 0"));
    let e = build(&json!({"type": "source", "constellation": {"kind": "qam", "m": 16, "gray": true}}), &ctx);
    assert!(e.err().unwrap().to_string().contains("unknown field `gray`"));
}

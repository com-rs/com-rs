use super::*;
use crate::blocks::tests::run;
use com_fec::ldpc;
use serde_json::Value as Json;
use serde_json::json;

#[test]
fn the_parameters_of_the_codes_are_documented_and_checked() {
    use crate::blocks::registry;
    let enc = registry().into_iter().find(|b| b.name == "fec_encode").unwrap();
    let code = enc.params.iter().find(|p| p.name == "code").unwrap();
    let kinds: Vec<&str> = code.variants.iter().map(|(k, _)| *k).collect();
    assert_eq!(kinds, ["conv", "rs", "turbo", "ldpc", "polar"]);
    let text = enc.describe();
    assert!(text.contains("{type: ldpc}") && text.contains("var_degree") && text.contains("min_sum"), "{text}");
    // the JSON Schema lists every code with its defaults and accepted values
    let schema = code.json_schema();
    let ldpc = schema["anyOf"][0]["oneOf"].as_array().unwrap().iter().find(|c| c["title"] == json!("ldpc")).unwrap();
    assert_eq!(ldpc["required"], json!(["type", "n", "k"]));
    assert_eq!(ldpc["properties"]["iterations"]["anyOf"][0]["default"], json!(ldpc::DEFAULT_MAX_ITERATIONS));
    assert_eq!(ldpc["properties"]["rule"]["anyOf"][0]["enum"][1], json!("min_sum"));
    // every code accepts the parameters it documents, and refuses the others
    for (kind, params) in &code.variants {
        let mut m = serde_json::Map::new();
        m.insert("type".into(), json!(kind));
        for p in params.iter().filter(|p| matches!(p.default, crate::blocks::ParamDefault::Required)) {
            let v = match (*kind, p.name) {
                ("rs", "m") => json!(4),
                ("rs", "t") => json!(2),
                ("polar", "n") => json!(64),
                ("turbo", "k") => json!(40),
                (_, "n") => json!(128),
                _ => json!(32),
            };
            m.insert(p.name.into(), v);
        }
        assert!(build_code(&Json::Object(m.clone())).is_ok(), "{kind}");
        m.insert("zz".into(), json!(1));
        let e = build_code(&Json::Object(m)).err().unwrap().to_string();
        assert!(e.contains("unknown field `zz`") && e.contains(&format!("{kind} code")), "{e}");
    }
    let e = build_code(&json!({"type": "bch"})).err().unwrap().to_string();
    assert!(e.contains("unknown code type") && e.contains("polar"), "{e}");
    // the type is read without case, the bounds are declared
    assert!(build_code(&json!({"type": "LDPC", "n": 128, "k": 64})).is_ok());
    let e = build_code(&json!({"type": "rs", "m": 20, "t": 2})).err().unwrap().to_string();
    assert!(e.contains("m must be >= 2 and <= 16"), "{e}");
    let e = build_code(&json!({"type": "ldpc", "n": 128, "k": 64, "rule": "bp"})).err().unwrap().to_string();
    assert!(e.contains("rule must be one of"), "{e}");
}

#[test]
fn every_code_decodes_a_clean_channel() {
    for code in [
        json!({"type": "conv", "k": 100}),
        json!({"type": "turbo", "k": 40}),
        json!({"type": "ldpc", "n": 128, "k": 64}),
        json!({"type": "polar", "n": 64, "k": 32, "crc": "crc6", "list": 4}),
    ] {
        let s = run(
            &[
                json!({"type": "fec_encode", "code": code, "blocks": 3, "interleave": true}),
                json!({"type": "map", "constellation": "16-QAM"}),
                json!({"type": "awgn_ebn0", "ebn0_db": 12.0}),
                json!({"type": "demap"}),
                json!({"type": "fec_decode", "code": code, "interleave": true}),
            ],
            1,
        );
        assert_eq!(s.coding.decoded, s.coding.payload, "{code}");
        assert_eq!(s.coding.n_blocks, 3);
    }
}

#[test]
fn coded_ebn0_counts_the_code_rate() {
    // rate 1/2 on QPSK: E_s/N_0 = E_b/N_0, so σ² = 10^(-Eb/N0/10)
    let s = run(
        &[
            json!({"type": "fec_encode", "code": {"type": "ldpc", "n": 256, "k": 128}, "blocks": 40}),
            json!({"type": "map"}),
            json!({"type": "awgn_ebn0", "ebn0_db": 3.0}),
        ],
        2,
    );
    assert!((s.channel.noise_var.unwrap() / 10f64.powf(-0.3) - 1.0).abs() < 0.05);
}

#[test]
fn probes_measure_in_the_middle_of_the_link() {
    let s = run(
        &[
            json!({"type": "source", "constellation": "QPSK", "n_symbols": 20_000}),
            json!({"type": "probe", "label": "tx", "metrics": ["power_dbm", "snr_db"]}),
            json!({"type": "awgn", "snr_db": 10.0}),
            json!({"type": "probe", "label": "rx", "metrics": ["snr_db", "ser"], "scale": "none"}),
        ],
        3,
    );
    let get = |k: &str| s.probes.get(k).map(com_sim::stats::Estimate::value).unwrap();
    assert!((get("tx.power_dbm") - 30.0).abs() < 1e-9);
    assert!((get("rx.snr_db") - 10.0).abs() < 0.1);
    assert!(get("rx.ser") > 0.0);
}

#[test]
fn pas_decodes_a_clean_channel_and_declares_its_distribution() {
    let p = json!({"constellation": "16-QAM", "mb_lambda": 1.0, "n": 192, "blocks": 3});
    let mut enc = p.clone();
    enc["type"] = json!("pas_encode");
    let mut dec = p;
    dec["type"] = json!("pas_decode");
    let s = run(&[enc, json!({"type": "awgn", "snr_db": 30.0}), dec], 4);
    assert_eq!(s.coding.decoded, s.coding.payload);
    assert_eq!(s.coding.n_blocks, 3);
    let pmf = s.pmf.clone().unwrap();
    assert!((pmf.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // the inner points are the most likely
    let c = s.constellation.clone().unwrap();
    let inner = c.points().iter().zip(&pmf).filter(|(z, _)| z.norm() < 0.5).map(|(_, p)| *p).fold(0.0, f64::max);
    let outer = c.points().iter().zip(&pmf).filter(|(z, _)| z.norm() > 1.2).map(|(_, p)| *p).fold(1.0, f64::min);
    assert!(inner > 2.0 * outer, "{inner} {outer}");
}

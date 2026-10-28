//! The quick validation specifications of `specs/validation/` reproduce the literature
//! (the slow ones run with `comrs verify`, see the justfile).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
#![cfg(feature = "spec")]

use comrs::spec::{RunOptions, Spec, run, search, verify};
use std::path::PathBuf;

fn check(name: &str) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../specs/validation").join(name);
    let spec = Spec::from_path(&path).unwrap();
    let checks = if spec.search.is_some() {
        verify::verify_search(&spec.expect, &search::search(&spec, &RunOptions::default()).unwrap()).unwrap()
    } else {
        let results = run(&spec, &RunOptions::default()).unwrap();
        verify::verify(&spec.expect, &results, spec.confidence).unwrap()
    };
    assert!(!checks.is_empty(), "{name}: nothing checked");
    let failed: Vec<String> = checks
        .iter()
        .filter(|c| !c.passed)
        .map(|c| format!("{} | {}: expected {}, measured {:?}", c.name, c.condition, c.expected, c.measured))
        .collect();
    assert!(failed.is_empty(), "{name}:\n{}", failed.join("\n"));
}

#[test]
fn error_rates_follow_the_closed_forms() {
    check("proakis_awgn.yaml");
}

#[test]
fn information_rates_follow_the_quadratures() {
    check("information_rates.yaml");
}

#[test]
fn operating_points_follow_the_inverted_closed_forms() {
    check("required_snr.yaml");
}

#[test]
fn ce_ofdm_follows_the_linear_theory_of_thompson() {
    check("thompson2008_eq15.yaml");
}

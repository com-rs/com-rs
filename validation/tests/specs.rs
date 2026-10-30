//! The quick validation specifications of `specs/validation/` that CI does not verify
//! with `comrs verify` (the list `ci_specs` of the justfile, `VALIDATION_SPECS` of
//! `.github/workflows/ci.yml`), run by `com_spec::verify::verify_spec` as `comrs verify`
//! runs them; and the check that those two lists are equal.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
#![cfg(feature = "spec")]

use comrs::spec::{RunOptions, Spec, verify};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn check(name: &str) {
    let spec = Spec::from_path(root().join("specs/validation").join(name)).unwrap();
    let (_, checks) = verify::verify_spec(&spec, &RunOptions::default()).unwrap();
    assert!(!checks.is_empty(), "{name}: nothing checked");
    let failed: Vec<String> = checks
        .iter()
        .filter(|c| !c.passed)
        .map(|c| format!("{} | {}: expected {}, measured {:?}", c.name, c.condition, c.expected, c.measured))
        .collect();
    assert!(failed.is_empty(), "{name}:\n{}", failed.join("\n"));
}

/// The specifications that CI verifies: `ci_specs` of the justfile and
/// `VALIDATION_SPECS` of the workflow.
fn ci_lists() -> (Vec<String>, Vec<String>) {
    let justfile = std::fs::read_to_string(root().join("justfile")).unwrap();
    let line = justfile.lines().find(|l| l.starts_with("ci_specs :=")).expect("ci_specs in the justfile");
    let just: Vec<String> = line.split('"').nth(1).unwrap().split_whitespace().map(String::from).collect();
    let ci = std::fs::read_to_string(root().join(".github/workflows/ci.yml")).unwrap();
    let lines = ci.lines().skip_while(|l| !l.trim_start().starts_with("VALIDATION_SPECS:")).skip(1);
    let workflow: Vec<String> =
        lines.map(str::trim).take_while(|l| l.starts_with("specs/")).map(String::from).collect();
    (just, workflow)
}

#[test]
fn the_justfile_and_the_workflow_verify_the_same_specifications() {
    let (just, workflow) = ci_lists();
    assert!(!just.is_empty(), "no specification in ci_specs");
    assert_eq!(just, workflow, "ci_specs of the justfile and VALIDATION_SPECS of ci.yml differ");
    for s in &just {
        assert!(root().join(s).is_file(), "{s} does not exist");
    }
    // the specifications of this file are not verified by CI a second time
    for name in QUICK {
        assert!(!just.iter().any(|s| s.ends_with(name)), "{name} is verified by CI already");
    }
}

/// The quick specifications that CI does not verify.
const QUICK: [&str; 1] = ["required_snr.yaml"];

#[test]
fn operating_points_follow_the_inverted_closed_forms() {
    check(QUICK[0]);
}

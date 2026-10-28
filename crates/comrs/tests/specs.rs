//! Every specification of `specs/` builds and runs one trial per condition.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
#![cfg(feature = "spec")]

use comrs::spec::{RunOptions, Spec, run};
use std::path::PathBuf;

fn specs() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../specs");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
        .collect();
    v.sort();
    v
}

#[test]
fn every_specification_runs_one_trial_per_condition() {
    let mut checked = 0;
    for path in specs() {
        let spec = Spec::from_path(&path).unwrap_or_else(|e| panic!("{e}"));
        // the WDM campaigns propagate hundreds of thousands of samples by the Manakov
        // equation: run in release only (`cargo test --release`), and checked against the
        // GN model by tests/test_performance.py
        if spec.link.iter().any(|b| b.to_string().contains("\"manakov\"")) && cfg!(debug_assertions) {
            continue;
        }
        let opts = RunOptions { fast_dev_run: true, batch: Some(1), ..RunOptions::default() };
        let r = run(&spec, &opts).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(r.len(), spec.conditions().len());
        // a condition is run once, or skipped by the error-floor callback of the spec
        let ok = |c: &comrs::sim::runner::ConditionResult| {
            (c.trials == 1 && !c.record.is_empty()) || c.status == comrs::sim::runner::Status::Skipped
        };
        assert!(r.iter().all(ok), "{}", path.display());
        assert!(r.iter().any(|c| c.trials == 1), "{}", path.display());
        checked += 1;
    }
    assert!(checked >= 4);
}

#[test]
fn every_specification_passes_the_check_of_its_ports() {
    // every specification of specs/ and its subdirectories, without running it
    let mut dirs = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../specs")];
    let mut checked = 0;
    while let Some(dir) = dirs.pop() {
        for p in std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()) {
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|e| e == "yaml") {
                let spec = Spec::from_path(&p).unwrap_or_else(|e| panic!("{e}"));
                let kinds: Vec<String> = spec
                    .link
                    .iter()
                    .map(|b| b.as_str().map_or_else(|| b["type"].as_str().unwrap().to_string(), str::to_string))
                    .collect();
                let kinds: Vec<&str> = kinds.iter().map(String::as_str).collect();
                comrs::spec::blocks::check_link(&kinds).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
                checked += 1;
            }
        }
    }
    assert!(checked >= 20, "{checked}");
}

#[test]
fn a_specification_reproduces_its_rust_counterpart() {
    // the qam_awgn specification draws exactly what the example scenario draws
    use comrs::prelude::*;
    let spec = Spec::from_yaml(
        "name: same\nseed: 5\ngrid: {ebn0_db: [6]}\nlink: [{type: source, constellation: 16-QAM, n_symbols: 1000}, {type: awgn_ebn0, ebn0_db: $ebn0_db}]\nmetrics: {report: [ber], scale: none}\nstop: {max_trials: 64}",
    )
    .unwrap();
    let declarative = run(&spec, &RunOptions::default()).unwrap();
    let s = comrs::sim::scenario::from_fn("same", |cond, rng, rec| {
        let c = Constellation::qam(16)?.normalized();
        let (idx, x) = source::symbols(&c, 1000, None, rng)?;
        let bits = comrs::mapping::constellation::indices_to_bits(&idx, 4);
        let y = awgn::awgn_ebn0(&x, cond.f64("ebn0_db")?, 4, 1.0, rng);
        rec.errors("ber", metrics::bit_errors(&c.demodulate(&y), &bits) as u64, bits.len() as u64);
        Ok(())
    });
    let coded = Runner::new().seed(5).stop(Stop::max_trials(64)).run(&s, &Grid::new().axis("ebn0_db", [6i64])).unwrap();
    assert_eq!(declarative[0].metric("ber"), coded[0].metric("ber"));
}

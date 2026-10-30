//! Integration tests of the Monte Carlo framework: reproducibility, stopping rules,
//! callbacks, loggers and error handling, through the public API only.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::prelude::*;
use comrs::sim::callbacks::Action;
use comrs::sim::runner::Status;
use comrs::sim::scenario::from_fn;
use comrs::sim::stopping::Progress;
use std::sync::{Arc, Mutex};

/// A Bernoulli trial of probability `p` (the condition) on 100 draws, and a mean.
fn bernoulli() -> impl Scenario {
    from_fn("bernoulli", |cond, rng, rec| {
        let p = cond.f64("p")?;
        let errors = (0..100).filter(|_| rng.uniform() < p).count() as u64;
        rec.errors("rate", errors, 100);
        rec.value("draw", rng.gaussian());
        Ok(())
    })
}

#[test]
fn results_do_not_depend_on_threads_or_batch_size_at_fixed_trials() {
    let grid = Grid::new().axis("p", [0.01, 0.3]);
    let run = |threads, batch| {
        Runner::new()
            .seed(9)
            .threads(threads)
            .batch(batch)
            .stop(Stop::max_trials(240))
            .run(&bernoulli(), &grid)
            .unwrap()
    };
    let reference = run(1, 240);
    for (threads, batch) in [(2, 16), (4, 40), (8, 1)] {
        let r = run(threads, batch);
        for (a, b) in reference.iter().zip(&r) {
            assert_eq!(a.trials, b.trials);
            assert_eq!(a.metric("rate"), b.metric("rate"));
            // the Welford merges happen in trial order: equal to rounding
            let (ma, mb) = (a.metric("draw").unwrap().value(), b.metric("draw").unwrap().value());
            assert!((ma - mb).abs() < 1e-12);
        }
    }
}

#[test]
fn seeds_and_scenario_names_give_independent_streams() {
    let grid = Grid::new().axis("p", [0.5]);
    let a = Runner::new().seed(1).stop(Stop::max_trials(64)).run(&bernoulli(), &grid).unwrap();
    let b = Runner::new().seed(2).stop(Stop::max_trials(64)).run(&bernoulli(), &grid).unwrap();
    assert_ne!(a[0].metric("rate"), b[0].metric("rate"));
    let renamed = from_fn("other", |cond, rng, rec| {
        let p = cond.f64("p")?;
        rec.errors("rate", (0..100).filter(|_| rng.uniform() < p).count() as u64, 100);
        Ok(())
    });
    let c = Runner::new().seed(1).stop(Stop::max_trials(64)).run(&renamed, &grid).unwrap();
    assert_ne!(a[0].metric("rate"), c[0].metric("rate"));
}

#[test]
fn the_estimates_are_consistent_with_the_truth() {
    let grid = Grid::new().axis("p", [0.02, 0.2]);
    let r = Runner::new().seed(4).stop(Stop::max_trials(2000)).run(&bernoulli(), &grid).unwrap();
    for c in &r {
        let p = c.condition.f64("p").unwrap();
        let (lo, hi) = c.metric("rate").unwrap().interval(0.999);
        assert!(lo < p && p < hi, "{p} outside [{lo}, {hi}]");
        let (lo, hi) = c.metric("draw").unwrap().interval(0.999);
        assert!(lo < 0.0 && 0.0 < hi);
    }
}

#[test]
fn min_errors_stops_after_the_batch_that_reaches_the_target() {
    let grid = Grid::new().axis("p", [0.1, 1e-6]);
    let rule = Stop::min_errors("rate", 50).or(Stop::max_trials(300));
    let r = Runner::new().seed(3).batch(2).stop(rule).run(&bernoulli(), &grid).unwrap();
    // p = 0.1: about 10 errors per trial, 50 errors after 5 or 6 trials, checked every 2
    let e = r[0].metric("rate").unwrap();
    assert!(e.count() >= 50 && e.count() < 50 + 2 * 100);
    assert_eq!(r[0].trials % 2, 0);
    assert_eq!(r[0].status, Status::Completed);
    // p = 1e-6: the error target is out of reach, the trial cap stops it
    assert_eq!(r[1].trials, 300);
}

#[test]
fn precision_rule_reaches_the_requested_relative_half_width() {
    let grid = Grid::new().axis("p", [0.05]);
    let rule = Stop::precision("rate", 0.05).or(Stop::max_trials(100_000));
    let r = Runner::new().seed(5).batch(16).stop(rule).run(&bernoulli(), &grid).unwrap();
    let e = r[0].metric("rate").unwrap();
    assert!(e.relative_precision(0.95) <= 0.05);
    assert!(r[0].trials < 100_000);
}

#[test]
fn unbounded_rules_are_rejected() {
    let r = Runner::new().stop(Stop::min_errors("rate", 10)).run(&bernoulli(), &Grid::new().axis("p", [0.1]));
    assert!(matches!(r, Err(Error::Config(_))));
}

#[test]
fn fast_dev_run_runs_one_batch_per_condition() {
    let grid = Grid::new().axis("p", [0.1, 0.2, 0.3]);
    let r =
        Runner::new().batch(4).fast_dev_run(true).stop(Stop::min_errors("rate", 1)).run(&bernoulli(), &grid).unwrap();
    assert!(r.iter().all(|c| c.trials == 4 && c.status == Status::DevRun));
}

#[test]
fn trial_and_setup_errors_name_the_condition() {
    let failing =
        from_fn(
            "failing",
            |cond, _, _| {
                if cond.f64("p")? > 0.5 { Err(Error::Numerical("diverged".into())) } else { Ok(()) }
            },
        );
    let err = Runner::new().stop(Stop::max_trials(4)).run(&failing, &Grid::new().axis("p", [0.1, 0.9])).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("p=0.9") && msg.contains("diverged"), "{msg}");
    // a missing key is a configuration error of the trial
    let err = Runner::new().stop(Stop::max_trials(4)).run(&bernoulli(), &Grid::new().axis("q", [0.1])).unwrap_err();
    assert!(err.to_string().contains("q=0.1"));
}

/// Records the hooks it receives, and stops the run after the second condition.
struct Spy {
    log: Arc<Mutex<Vec<String>>>,
}

impl Callback for Spy {
    fn on_run_start(&mut self, info: &comrs::sim::runner::RunInfo) {
        self.log.lock().unwrap().push(format!("start {}", info.n_conditions));
    }
    fn on_condition_start(&mut self, index: usize, _: &Condition) -> Action {
        self.log.lock().unwrap().push(format!("condition {index}"));
        Action::Continue
    }
    fn on_batch_end(&mut self, p: &Progress<'_>) -> Action {
        self.log.lock().unwrap().push(format!("batch {}", p.trials));
        if p.trials >= 20 { Action::Skip } else { Action::Continue }
    }
    fn on_condition_end(&mut self, r: &ConditionResult) -> Action {
        self.log.lock().unwrap().push(format!("end {} {}", r.index, r.status.as_str()));
        if r.index == 1 { Action::StopRun } else { Action::Continue }
    }
    fn on_run_end(&mut self, results: &[ConditionResult]) {
        self.log.lock().unwrap().push(format!("done {}", results.len()));
    }
}

#[test]
fn callbacks_see_every_hook_and_can_stop_conditions_and_the_run() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let grid = Grid::new().axis("p", [0.1, 0.2, 0.3]);
    let r = Runner::new()
        .batch(10)
        .stop(Stop::max_trials(30))
        .callback(Spy { log: log.clone() })
        .run(&bernoulli(), &grid)
        .unwrap();
    assert_eq!(r.len(), 3);
    assert_eq!(r[0].status, Status::Stopped);
    assert_eq!(r[0].trials, 20);
    assert_eq!(r[2].status, Status::Skipped);
    assert_eq!(r[2].trials, 0);
    let log = log.lock().unwrap().join("; ");
    assert_eq!(
        log,
        "start 3; condition 0; batch 10; batch 20; end 0 stopped; condition 1; batch 10; batch 20; \
         end 1 stopped; end 2 skipped; done 3"
    );
}

#[test]
fn skip_after_works_along_one_axis_for_every_value_of_the_others() {
    let s = from_fn("decay", |cond, rng, rec| {
        let p = cond.f64("scale")? * 10f64.powf(-cond.f64("snr_db")? / 5.0);
        rec.errors("ber", (rng.uniform() < p) as u64, 1);
        Ok(())
    });
    // scale 1: 1, 1e-2, 1e-4, 1e-6; scale 100 saturates the first points
    let grid = Grid::new().axis("scale", [1.0, 100.0]).axis("snr_db", [0.0, 10.0, 20.0, 30.0]);
    let r = Runner::new()
        .stop(Stop::max_trials(5000))
        .callback(SkipAfter::below("ber", 1e-3, "snr_db"))
        .run(&s, &grid)
        .unwrap();
    let status: Vec<&str> = r.iter().map(|c| c.status.as_str()).collect();
    assert_eq!(
        status,
        ["completed", "completed", "completed", "skipped", "completed", "completed", "completed", "completed"]
    );
}

#[test]
fn csv_and_json_loggers_write_every_condition() {
    let dir = std::env::temp_dir().join(format!("comrs-it-loggers-{}", std::process::id()));
    let grid = Grid::new().axis("p", [0.5, 0.1]).fixed("label", "a,b");
    Runner::new()
        .stop(Stop::max_trials(32))
        .callback(SkipAfter::below("rate", 0.2, "p"))
        .logger(CsvLogger::new(dir.join("nested/out.csv")))
        .logger(JsonLogger::new(dir.join("out.json")))
        .run(&bernoulli(), &grid)
        .unwrap();
    let csv = std::fs::read_to_string(dir.join("nested/out.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(
        lines[0],
        "index,p,label,status,trials,wall_s,draw,draw_lo,draw_hi,draw_count,draw_total,rate,rate_lo,rate_hi,rate_count,rate_total"
    );
    assert_eq!(lines.len(), 3);
    assert!(lines[1].starts_with("0,0.5,\"a,b\",completed,32,"));
    let json = std::fs::read_to_string(dir.join("out.json")).unwrap();
    for key in [
        "\"scenario\": \"bernoulli\"",
        "\"kind\": \"rate\"",
        "\"kind\": \"mean\"",
        "\"status\": \"completed\"",
        "\"version\"",
        "\"commit\"",
        "\"config\"",
    ] {
        assert!(json.contains(key), "{key} missing");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn grids_compose_axes_zips_fixed_values_and_explicit_lists() {
    let g = Grid::new()
        .zip(&["m", "kind"], vec![vec![4usize.into(), "qam".into()], vec![8usize.into(), "psk".into()]])
        .unwrap()
        .axis("snr_db", [1.0, 2.0])
        .fixed("n", 10usize);
    let c = g.conditions();
    assert_eq!(c.len(), 4);
    assert_eq!(c[2].label(), "m=8, kind=psk, snr_db=1, n=10");
    let explicit = Grid::from_conditions(vec![Condition::new().with("a", 1usize), Condition::new().with("a", 2usize)])
        .axis("b", [true, false]);
    let c = explicit.conditions();
    assert_eq!(c.len(), 4);
    assert!(c[3].bool("b").is_ok_and(|b| !b) && c[3].usize("a").is_ok_and(|a| a == 2));
}

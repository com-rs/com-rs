//! Search of the operating point: the value of a variable at which a metric reaches a
//! target, with its confidence interval.
//!
//! A sweep answers "what is the BER at 12 dB?"; a link budget asks the converse, "which
//! SNR gives a BER of 10⁻³?". The `search` section of a specification answers it
//! directly:
//!
//! ```yaml
//! search:
//!   along: ebn0_db          # the variable searched (a grid axis or not)
//!   range: [4, 16]          # the bracket of the search
//!   metric: ber
//!   target: 1.0e-3
//!   tolerance: 0.02         # stop when the bracket is narrower
//!   log: true               # interpolate log10 of the metric (default)
//! ```
//!
//! For every combination of the other grid axes, the metric is evaluated at both ends of
//! the range (the target must lie between the two values), then the bracket is halved
//! until it is narrower than `tolerance` (at most `max_evaluations` evaluations). Every
//! evaluation is a run of one condition with the stopping rule of the specification;
//! all of them use the same random streams (common random numbers), so the comparison
//! of two values of the variable is not blurred by the draws, and the estimate is
//! monotone wherever the link is.
//!
//! The crossing is interpolated between the two ends of the final bracket (linearly in
//! `log10` of the metric, or in the metric). Its confidence interval comes from the
//! confidence intervals of these two estimates: the crossings of the lower and of the
//! upper bounds of the metric, interpolated the same way (and extrapolated when a bound
//! does not cross the target inside the bracket).
//!
//! # Examples
//! ```
//! use com_spec::{search::search, RunOptions, Spec};
//! let spec = Spec::from_yaml(r#"
//! name: qpsk_required
//! link:
//!   - {type: source, constellation: QPSK, n_symbols: 4000}
//!   - {type: awgn_ebn0, ebn0_db: $ebn0_db}
//! metrics: {report: [ber], scale: none}
//! stop: {min_errors: {ber: 400}, max_trials: 400}
//! search: {along: ebn0_db, range: [2, 10], metric: ber, target: 1.0e-3, tolerance: 0.05}
//! "#).unwrap();
//! let r = search(&spec, &RunOptions::default()).unwrap();
//! // Q(√(2 Eb/N0)) = 1e-3 at 6.79 dB
//! assert!((r[0].value - 6.79).abs() < 0.15, "{}", r[0].value);
//! assert!(r[0].lo <= r[0].value && r[0].value <= r[0].hi);
//! ```

use crate::params::parse;
use crate::{RunOptions, Spec, SpecScenario};
use com_core::{Error, Result, ensure};
use com_sim::grid::{Condition, Grid, Value};
use com_sim::json;
use com_sim::loggers::{create_parent_dir, csv_field};
use serde::Deserialize;
use serde_json::Value as Json;
use std::path::Path;

/// The `search` section of a specification.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSpec {
    /// The variable searched.
    pub along: String,
    /// The bracket `[lo, hi]` of the search.
    pub range: [f64; 2],
    /// The metric.
    pub metric: String,
    /// The value of the metric to reach.
    pub target: f64,
    /// Stop when the bracket is narrower (in the unit of the variable).
    #[serde(default = "d_tolerance")]
    pub tolerance: f64,
    /// Interpolate `log10` of the metric (rates spanning decades).
    #[serde(default = "d_true")]
    pub log: bool,
    /// Largest number of evaluations per combination of the other axes.
    #[serde(default = "d_max_eval")]
    pub max_evaluations: usize,
}

fn d_tolerance() -> f64 {
    0.05
}
fn d_true() -> bool {
    true
}
fn d_max_eval() -> usize {
    30
}

impl SearchSpec {
    /// Parse and validate the `search` mapping.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing field, an empty range or a non-positive tolerance.
    pub fn from_json(v: &Json) -> Result<SearchSpec> {
        let s: SearchSpec = parse("search", v.clone())?;
        ensure(s.range[0] < s.range[1], || format!("search: the range {:?} must increase", s.range))?;
        ensure(s.tolerance > 0.0, || "search: the tolerance must be positive".into())?;
        ensure(s.max_evaluations >= 3, || "search: at least 3 evaluations".into())?;
        ensure(!s.log || s.target > 0.0, || "search: a log search needs a positive target".into())?;
        Ok(s)
    }
}

/// One evaluation of the metric during a search.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Value of the variable.
    pub x: f64,
    /// Estimate of the metric and its confidence interval.
    pub value: f64,
    /// Lower bound of the interval.
    pub lo: f64,
    /// Upper bound of the interval.
    pub hi: f64,
    /// Trials run.
    pub trials: u64,
}

/// The operating point of one combination of the other axes.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// The values of the other axes.
    pub condition: Condition,
    /// The variable searched.
    pub along: String,
    /// The metric.
    pub metric: String,
    /// Its target.
    pub target: f64,
    /// The value of the variable at which the metric reaches the target.
    pub value: f64,
    /// Lower bound of its confidence interval.
    pub lo: f64,
    /// Upper bound of its confidence interval.
    pub hi: f64,
    /// Whether the final bracket is narrower than the tolerance.
    pub converged: bool,
    /// Every evaluation, in the order of the search.
    pub evaluations: Vec<Evaluation>,
}

/// The value of the variable at which `g` crosses `g(target)` between `(a, ga)` and
/// `(b, gb)` (extrapolated outside; the midpoint for a flat segment).
fn crossing(a: f64, ga: f64, b: f64, gb: f64, gt: f64) -> f64 {
    if (gb - ga).abs() < 1e-300 || !ga.is_finite() || !gb.is_finite() {
        return 0.5 * (a + b);
    }
    a + (b - a) * (gt - ga) / (gb - ga)
}

/// The conditions of the other axes: the grid without `along`, in order, unique.
fn combinations(grid: &Grid, along: &str) -> Vec<Condition> {
    let mut out: Vec<Condition> = Vec::new();
    for c in grid.conditions() {
        let mut d = Condition::new();
        for (k, v) in c.entries() {
            if k != along {
                d.set(k, v.clone());
            }
        }
        if !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

/// The sweep of the first points of the search of `spec`: every combination of the other
/// axes, with the searched variable at the lower end of the range (what `comrs check`
/// runs); `None` without a `search` section.
pub(crate) fn first_points(spec: &Spec) -> Option<Spec> {
    let s = spec.search.as_ref()?;
    let conditions = combinations(&spec.grid, &s.along).into_iter().map(|c| c.with(&s.along, s.range[0])).collect();
    Some(Spec { grid: Grid::from_conditions(conditions), search: None, ..spec.clone() })
}

/// Run the search of a specification.
///
/// # Errors
/// [`Error::Config`] without a `search` section, or when the target is not between the
/// values of the metric at the ends of the range; the errors of the runs.
pub fn search(spec: &Spec, opts: &RunOptions) -> Result<Vec<SearchResult>> {
    let s = spec.search.as_ref().ok_or_else(|| Error::Config(format!("{}: no search section", spec.name)))?;
    let g = |v: f64| if s.log { v.max(1e-300).log10() } else { v };
    let mut results = Vec::new();
    for base in combinations(&spec.grid, &s.along) {
        let eval = |x: f64| -> Result<Evaluation> {
            let cond = base.clone().with(&s.along, x);
            // a single condition: index 0, so every value of the variable sees the same
            // random streams (common random numbers)
            let grid = Grid::from_conditions(vec![cond.clone()]);
            let r = spec.bare_runner(opts).run(&SpecScenario::new(spec), &grid)?.remove(0);
            let est = r
                .metric(&s.metric)
                .ok_or_else(|| Error::Config(format!("search: no metric {:?} in ({})", s.metric, cond.label())))?;
            let (lo, hi) = est.interval(spec.confidence);
            if opts.progress.unwrap_or(spec.progress) {
                eprintln!(
                    "[{}] ({}): {} = {:.4e} [{lo:.3e}, {hi:.3e}]",
                    spec.name,
                    cond.label(),
                    s.metric,
                    est.value()
                );
            }
            Ok(Evaluation { x, value: est.value(), lo, hi, trials: r.trials })
        };
        let (mut a, mut b) = (eval(s.range[0])?, eval(s.range[1])?);
        let mut evaluations = vec![a.clone(), b.clone()];
        let side = |e: &Evaluation| (e.value - s.target).signum();
        if side(&a) * side(&b) >= 0.0 {
            return Err(Error::Config(format!(
                "search ({}): the target {} of {} is not between {:.4e} at {} = {} and {:.4e} at {} = {}",
                base.label(),
                s.target,
                s.metric,
                a.value,
                s.along,
                a.x,
                b.value,
                s.along,
                b.x
            )));
        }
        while b.x - a.x > s.tolerance && evaluations.len() < s.max_evaluations {
            let m = eval(0.5 * (a.x + b.x))?;
            evaluations.push(m.clone());
            if side(&m) == side(&a) {
                a = m;
            } else {
                b = m;
            }
        }
        let gt = g(s.target);
        let value = crossing(a.x, g(a.value), b.x, g(b.value), gt);
        let x1 = crossing(a.x, g(a.lo), b.x, g(b.lo), gt);
        let x2 = crossing(a.x, g(a.hi), b.x, g(b.hi), gt);
        results.push(SearchResult {
            condition: base,
            along: s.along.clone(),
            metric: s.metric.clone(),
            target: s.target,
            value,
            lo: x1.min(x2).min(value),
            hi: x1.max(x2).max(value),
            converged: b.x - a.x <= s.tolerance,
            evaluations,
        });
    }
    if let Some(p) = opts.csv.as_ref().or(spec.csv.as_ref()) {
        write_csv(p, &results)?;
    }
    if let Some(p) = opts.json.as_ref().or(spec.json.as_ref()) {
        write_json(p, spec, &results)?;
    }
    Ok(results)
}

/// One row per combination: its keys, then the variable at the target, its interval,
/// the convergence and the number of evaluations.
fn write_csv(path: &Path, results: &[SearchResult]) -> Result<()> {
    create_parent_dir(path)?;
    let mut out = String::new();
    if let Some(first) = results.first() {
        let keys: Vec<&str> = first.condition.keys().collect();
        let mut header: Vec<String> = keys.iter().map(|k| csv_field(k)).collect();
        let a = &first.along;
        header.extend([
            "metric".into(),
            "target".into(),
            a.clone(),
            format!("{a}_lo"),
            format!("{a}_hi"),
            "converged".into(),
            "evaluations".into(),
        ]);
        out.push_str(&(header.join(",") + "\n"));
        for r in results {
            let mut row: Vec<String> = keys
                .iter()
                .map(|k| r.condition.get(k).map(|v| csv_field(&v.to_string())).unwrap_or_default())
                .collect();
            row.extend([
                r.metric.clone(),
                format!("{:e}", r.target),
                format!("{}", r.value),
                format!("{}", r.lo),
                format!("{}", r.hi),
                r.converged.to_string(),
                r.evaluations.len().to_string(),
            ]);
            out.push_str(&(row.join(",") + "\n"));
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn write_json(path: &Path, spec: &Spec, results: &[SearchResult]) -> Result<()> {
    create_parent_dir(path)?;
    let items: Vec<String> = results
        .iter()
        .map(|r| {
            let cond: Vec<(&str, String)> = r
                .condition
                .entries()
                .iter()
                .map(|(k, v)| {
                    (
                        k.as_str(),
                        match v {
                            Value::F64(x) => json::number(*x),
                            Value::I64(x) => x.to_string(),
                            Value::Str(s) => json::string(s),
                            Value::Bool(b) => b.to_string(),
                        },
                    )
                })
                .collect();
            let evals: Vec<String> = r
                .evaluations
                .iter()
                .map(|e| {
                    json::object(&[
                        ("x", json::number(e.x)),
                        ("value", json::number(e.value)),
                        ("lo", json::number(e.lo)),
                        ("hi", json::number(e.hi)),
                        ("trials", e.trials.to_string()),
                    ])
                })
                .collect();
            json::object(&[
                ("condition", json::object(&cond)),
                ("along", json::string(&r.along)),
                ("metric", json::string(&r.metric)),
                ("target", json::number(r.target)),
                ("value", json::number(r.value)),
                ("lo", json::number(r.lo)),
                ("hi", json::number(r.hi)),
                ("converged", r.converged.to_string()),
                ("evaluations", json::array(&evals)),
            ])
        })
        .collect();
    let doc = json::object(&[
        ("scenario", json::string(&spec.name)),
        ("seed", spec.seed.to_string()),
        ("confidence", json::number(spec.confidence)),
        ("version", json::string(com_sim::VERSION)),
        ("commit", json::string(com_sim::COMMIT)),
        ("config", spec.source.to_string()),
        ("search", json::array(&items)),
    ]);
    std::fs::write(path, doc + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(extra: &str) -> Spec {
        Spec::from_yaml(&format!(
            "name: s\nseed: 4\ngrid: {{m: [4, 16]}}\nlink: [{{type: source, constellation: {{kind: qam, m: $m}}, n_symbols: 4000}}, \
             {{type: awgn_ebn0, ebn0_db: $ebn0_db}}]\nmetrics: {{report: [ber], scale: none}}\n\
             stop: {{min_errors: {{ber: 300}}, max_trials: 300}}\n{extra}"
        ))
        .unwrap()
    }

    #[test]
    fn the_operating_points_follow_the_theory() {
        let s = spec("search: {along: ebn0_db, range: [2, 16], metric: ber, target: 1.0e-3, tolerance: 0.05}");
        let r = search(&s, &RunOptions::default()).unwrap();
        assert_eq!(r.len(), 2);
        for (res, m) in r.iter().zip([4usize, 16]) {
            let theory =
                com_mapping::theory::ebn0_for_ber(com_mapping::constellation::ConstellationKind::Qam, m, 1e-3).unwrap();
            assert!(res.converged && res.evaluations.len() <= 12);
            assert!(res.lo - 0.1 <= theory && theory <= res.hi + 0.1, "M = {m}: {res:?} against {theory}");
            assert!(res.hi - res.lo < 0.5);
        }
    }

    #[test]
    fn an_unbracketed_target_is_an_error() {
        let s = spec("search: {along: ebn0_db, range: [12, 16], metric: ber, target: 1.0e-1}");
        let e = search(&s, &RunOptions::default()).unwrap_err().to_string();
        assert!(e.contains("not between"), "{e}");
        assert!(
            Spec::from_yaml("name: x\nlink: [source]\nsearch: {along: x, range: [2, 1], metric: ber, target: 1}")
                .is_err()
        );
    }

    #[test]
    fn the_first_points_of_a_search_are_a_sweep() {
        let s = spec("search: {along: ebn0_db, range: [4, 12], metric: ber, target: 1.0e-2}");
        // the searched variable is not in the grid: the link cannot be built as such
        assert!(s.link_for(&s.conditions()[0]).is_err());
        let sweep = first_points(&s).unwrap();
        let labels: Vec<String> = sweep.conditions().iter().map(Condition::label).collect();
        assert_eq!(labels, ["m=4, ebn0_db=4", "m=16, ebn0_db=4"]);
        assert!(sweep.search.is_none() && sweep.link_for(&sweep.conditions()[1]).is_ok());
        assert!(first_points(&sweep).is_none());
    }

    #[test]
    fn reports_are_written() {
        let dir = std::env::temp_dir().join(format!("comrs-search-{}", std::process::id()));
        let s = spec("search: {along: ebn0_db, range: [4, 12], metric: ber, target: 1.0e-2, tolerance: 0.5}");
        let opts = RunOptions { csv: Some(dir.join("s.csv")), json: Some(dir.join("s.json")), ..RunOptions::default() };
        search(&s, &opts).unwrap();
        let csv = std::fs::read_to_string(dir.join("s.csv")).unwrap();
        assert!(csv.starts_with("m,metric,target,ebn0_db,ebn0_db_lo,ebn0_db_hi,converged,evaluations\n"), "{csv}");
        assert_eq!(csv.lines().count(), 3);
        assert!(std::fs::read_to_string(dir.join("s.json")).unwrap().contains("\"evaluations\""));
        std::fs::remove_dir_all(&dir).ok();
    }
}

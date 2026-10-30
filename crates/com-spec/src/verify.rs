//! Expected results: a specification that checks itself against the literature.
//!
//! The `expect` list of a specification states what the results must be, where they
//! come from, and how close they must be. `comrs verify` (or [`verify`]) runs the
//! specification and checks every expectation on every condition it applies to:
//!
//! ```yaml
//! expect:
//!   - name: BER of Gray 16-QAM over AWGN
//!     source: Proakis & Salehi, Digital Communications, 5th ed., eq. (4.3-30)
//!     where: {m: 16}                     # the conditions it applies to (default: all)
//!     metric: ber
//!     theory: {function: ber, kind: qam, m: $m, ebn0_db: $ebn0_db}
//!     interval: 0.999                    # the expected value is inside the 99.9 % interval
//!     rtol: 0.05                         # … or within 5 % of the estimate
//!     min_count: 1000                    # skip the conditions with fewer errors
//!   - name: Effective SNR of linear equalisation, Fig. 5 of Häger & Pfister (2021)
//!     where: {receiver: cdc}
//!     metric: snr_eff_db
//!     curve: {along: power_dbm, x: [-8, -6, -4], y: [17.40, 16.96, 14.50]}
//!     atol: 0.5
//! ```
//!
//! The expected value is one of:
//!
//! - `value`: a number;
//! - `curve`: a published curve `y(x)` along a variable of the grid, interpolated
//!   linearly; the conditions outside the curve, or on a missing point (`null`), are
//!   not checked;
//! - `reference`: another metric of the same condition (a theoretical value that a block
//!   records, for instance `ce_ofdm_rx.ser_impulsive`);
//! - `theory`: a closed form evaluated on the condition (`$name` substitutes a grid
//!   variable): `ber`, `ser` (with `kind`, `m`, and `ebn0_db` or `snr_db`), `mi`, `gmi`
//!   (with `kind`, `m`, `snr_db`, and `mb_lambda` for the Maxwell-Boltzmann priors of a
//!   shaped source), and for a search the inverses `ebn0_for_ber`,
//!   `ebn0_for_ser`, `snr_for_ber`, `snr_for_ser` (with `kind`, `m`, `target`; see
//!   `com_mapping::theory::ebn0_for_ber`), and
//!   `gn_snr_db`, the SNR of the centre channel of a WDM link by the GN model (with
//!   `fiber`, merged over the `fiber` of the specification as for the fibre blocks,
//!   `n_channels`, `spacing_hz`, `symbol_rate`, `rolloff`, `power_dbm` and
//!   `coherent`, see `com_channel::optical::gn`; a `raman` mapping in `fiber` gives the
//!   GN model with the Raman power profiles; `format`, a constellation with
//!   `mb_lambda` for shaped symbols, gives the EGN model), and `ber_rayleigh`, the BER of BPSK
//!   or QPSK with maximal-ratio combining of `branches` Rayleigh branches whose mean
//!   $E_{b}/N_{0}$ is `ebn0_db` plus `offset_db` (with `m`, `kind`; see
//!   `com_mapping::theory::ber_rayleigh_mrc`), and `dvbs2_esn0_db`,
//!   the ideal $E_{s}/N_{0}$ of a DVB-S2 MODCOD at a PER of $10^{-7}$ (Table 13 of EN
//!   302 307-1, with `modulation`, `rate`) plus `offset_db`.
//!
//! On a specification with a `search`, the expectations are about the searched
//! variable (`metric: ebn0_db` for a search along `ebn0_db`) and are checked against its
//! value and confidence interval ([`Expectation::check_search`]).
//!
//! Every `function` takes its own parameters, and an unknown one is an error (`branches`
//! on `ber`, say); `kind` is read without case.
//!
//! An expectation passes when the measured value $v$ satisfies
//! $|v - e| \le a + r |e|$ (`atol: a`, `rtol: r`), or, with `factor: f`, when
//! $e/f \le v \le f e$ (error rates over several decades), or, with `interval: c`, when
//! $e$ lies in the confidence interval of level $c$ of the estimate; at least one of
//! `atol`, `rtol`, `factor` and `interval` is required. `min_count` skips the conditions whose estimate counts fewer errors (or
//! values), and `measured_range: [lo, hi]` those whose measured value is outside the range
//! (the domain of validity of an approximation, or the values precise enough).
//!
//! An expectation that checks no condition is an error, not a pass: a `where` that
//! matches no condition (a misspelt variable or value), a curve outside the grid, or
//! conditions all skipped by `min_count` or `measured_range` verify nothing, and the
//! message names the expectation and the step that left no condition.
//!
//! [`report`] is the verification report that `comrs verify` writes to
//! `DIR/<name>.verify.json` and that the Python `verify_spec` returns: the header of
//! the reports of the runs (the seed, the confidence, the version and commit of the
//! library, the specification) and every check.

mod theory;

use crate::params::{parse, substitute};
use crate::{RunOptions, Spec};
use com_core::{Error, Result, ensure};
use com_sim::grid::{Condition, Value};
use com_sim::json;
use com_sim::loggers::{run_header, write_json};
use com_sim::runner::ConditionResult;
use serde::Deserialize;
use serde_json::{Map, Value as Json};
use theory::Theory;

/// The acceptance rule of a measured value $v$ against an expected value $e$:
/// $|v - e| \le a + r |e|$ (with `atol` $a$ or `rtol` $r$), or $e/f \le v \le f e$ (with
/// `factor` $f$, for error rates over several decades); either rule suffices.
#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    /// Absolute tolerance.
    #[serde(default)]
    pub atol: Option<f64>,
    /// Tolerance relative to the expected value.
    #[serde(default)]
    pub rtol: Option<f64>,
    /// Factor, at least 1.
    #[serde(default)]
    pub factor: Option<f64>,
}

impl Tolerance {
    /// Whether a rule is given.
    pub fn is_set(&self) -> bool {
        self.atol.is_some() || self.rtol.is_some() || self.factor.is_some()
    }

    /// Whether `v` is accepted for `e`.
    ///
    /// # Examples
    /// ```
    /// use com_spec::verify::Tolerance;
    /// let t = Tolerance { factor: Some(2.0), ..Tolerance::default() };
    /// assert!(t.accepts(1.9e-3, 1e-3) && !t.accepts(2.1e-3, 1e-3));
    /// ```
    pub fn accepts(&self, v: f64, e: f64) -> bool {
        let by_tol = (self.atol.is_some() || self.rtol.is_some())
            && (v - e).abs() <= self.atol.unwrap_or(0.0) + self.rtol.unwrap_or(0.0) * e.abs();
        let by_factor = self.factor.is_some_and(|f| e > 0.0 && v >= e / f && v <= e * f);
        by_tol || by_factor
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct ExpectSpec {
    #[serde(default)]
    name: String,
    #[serde(default)]
    source: String,
    #[serde(default, rename = "where")]
    filter: Map<String, Json>,
    metric: String,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    curve: Option<CurveSpec>,
    #[serde(default)]
    theory: Option<Json>,
    #[serde(default)]
    reference: Option<String>,
    #[serde(default)]
    atol: Option<f64>,
    #[serde(default)]
    factor: Option<f64>,
    #[serde(default)]
    rtol: Option<f64>,
    #[serde(default)]
    interval: Option<f64>,
    #[serde(default)]
    min_count: u64,
    #[serde(default)]
    measured_range: Option<[f64; 2]>,
    /// Why the tolerance is what it is (the precision of the published value, the
    /// sampling error), for the reader.
    #[serde(default)]
    #[allow(dead_code, reason = "documentation of the specification, not used to check")]
    why: String,
    /// A comment, for the reader.
    #[serde(default)]
    #[allow(dead_code, reason = "documentation of the specification, not used to check")]
    note: String,
}

impl ExpectSpec {
    fn tolerance(&self) -> Tolerance {
        Tolerance { atol: self.atol, rtol: self.rtol, factor: self.factor }
    }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct CurveSpec {
    along: String,
    x: Vec<f64>,
    y: Vec<Option<f64>>,
}

impl CurveSpec {
    /// The curve at `x`, interpolated linearly; `None` outside the curve or next to a
    /// missing point.
    fn at(&self, x: f64) -> Result<Option<f64>> {
        let n = self.x.len();
        if x < self.x[0] - 1e-12 || x > self.x[n - 1] + 1e-12 {
            return Ok(None);
        }
        let i = self.x.iter().position(|&xi| xi >= x - 1e-12).unwrap_or(n - 1);
        if (self.x[i] - x).abs() <= 1e-12 {
            return Ok(self.y[i]);
        }
        Ok(match (self.y[i - 1], self.y[i]) {
            (Some(a), Some(b)) => Some(a + (b - a) * (x - self.x[i - 1]) / (self.x[i] - self.x[i - 1])),
            _ => None,
        })
    }
}

/// An expectation of a specification, validated.
#[derive(Clone, Debug)]
pub struct Expectation {
    spec: ExpectSpec,
}

/// The outcome of one expectation on one condition.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// Name of the expectation.
    pub name: String,
    /// Where the expected value comes from.
    pub source: String,
    /// The condition.
    pub condition: String,
    /// The metric.
    pub metric: String,
    /// Expected value.
    pub expected: f64,
    /// Measured value and its interval at the confidence of the specification.
    pub measured: Interval,
    /// Whether the check passed.
    pub passed: bool,
}

/// A measured value and its confidence interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// The estimate.
    pub value: f64,
    /// Lower bound of its confidence interval.
    pub lo: f64,
    /// Upper bound of its confidence interval.
    pub hi: f64,
}

impl Expectation {
    /// Parse and validate an expectation.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing expected value or tolerance.
    pub fn from_json(v: &Json) -> Result<Expectation> {
        let spec: ExpectSpec = parse("expect", v.clone())?;
        let sources = [spec.value.is_some(), spec.curve.is_some(), spec.theory.is_some(), spec.reference.is_some()];
        ensure(sources.iter().filter(|&&b| b).count() == 1, || {
            format!("expectation {:?}: give exactly one of value, curve, theory and reference", spec.name)
        })?;
        ensure(spec.tolerance().is_set() || spec.interval.is_some(), || {
            format!("expectation {:?}: give atol, rtol, factor or interval", spec.name)
        })?;
        ensure(!matches!(spec.factor, Some(f) if f < 1.0), || {
            format!("expectation {:?}: factor must be at least 1", spec.name)
        })?;
        if let Some(c) = &spec.curve {
            ensure(c.x.len() == c.y.len() && !c.x.is_empty(), || {
                format!("expectation {:?}: the curve needs as many x as y", spec.name)
            })?;
            ensure(c.x.windows(2).all(|w| w[0] < w[1]), || {
                format!("expectation {:?}: the x of the curve must increase", spec.name)
            })?;
        }
        Ok(Expectation { spec })
    }

    fn applies(&self, cond: &Condition) -> bool {
        self.spec.filter.iter().all(|(k, want)| match (cond.get(k), want) {
            (Some(v), Json::Number(n)) => {
                v.as_f64().zip(n.as_f64()).is_some_and(|(a, b)| (a - b).abs() <= 1e-9 * b.abs().max(1.0))
            }
            (Some(Value::Str(s)), Json::String(w)) => s == w,
            (Some(Value::Bool(b)), Json::Bool(w)) => b == w,
            _ => false,
        })
    }

    fn expected(&self, cond: &Condition) -> Result<Option<f64>> {
        let s = &self.spec;
        ensure(s.reference.is_none(), || {
            format!("expectation {:?}: a reference metric needs the results of a run", s.name)
        })?;
        if let Some(v) = s.value {
            return Ok(Some(v));
        }
        if let Some(c) = &s.curve {
            return c.at(cond.f64(&c.along)?);
        }
        let t: Theory = parse("expect theory", substitute(s.theory.as_ref().unwrap_or(&Json::Null), cond)?)?;
        t.value(&s.name).map(Some)
    }

    /// The name of the expectation in a message: its name, or its metric.
    fn label(&self) -> String {
        if self.spec.name.is_empty() { format!("on {}", self.spec.metric) } else { format!("{:?}", self.spec.name) }
    }

    /// The error of an expectation that checked no condition: `funnel` counts the
    /// conditions left after each step, with the step's name.
    fn checked_nothing(&self, funnel: &[(&str, usize)]) -> Error {
        let steps: Vec<String> = funnel.iter().map(|(what, n)| format!("{n} {what}")).collect();
        let filter = if self.spec.filter.is_empty() {
            String::new()
        } else {
            format!(" (where: {})", Json::Object(self.spec.filter.clone()))
        };
        Error::Config(format!("expectation {} checks no condition{filter}: {}", self.label(), steps.join(", ")))
    }

    /// Check the expectation on the results of a run.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing metric or variable, or an expectation that checks
    /// no condition (see the module documentation).
    pub fn check(&self, results: &[ConditionResult], confidence: f64) -> Result<Vec<Check>> {
        let s = &self.spec;
        let mut out = Vec::new();
        let run = results.iter().filter(|r| r.trials > 0).count();
        let (mut matched, mut expected, mut counted) = (0, 0, 0);
        for r in results.iter().filter(|r| r.trials > 0 && self.applies(&r.condition)) {
            matched += 1;
            let e = match &s.reference {
                Some(m) => Some(
                    r.metric(m)
                        .ok_or_else(|| {
                            Error::Config(format!(
                                "expectation {:?}: no reference metric {m:?} in ({})",
                                s.name,
                                r.condition.label()
                            ))
                        })?
                        .value(),
                ),
                None => self.expected(&r.condition)?,
            };
            let Some(e) = e else { continue };
            expected += 1;
            let est = r.metric(&s.metric).ok_or_else(|| {
                Error::Config(format!(
                    "expectation {:?}: no metric {:?} in ({})",
                    s.name,
                    s.metric,
                    r.condition.label()
                ))
            })?;
            if est.count() < s.min_count {
                continue;
            }
            counted += 1;
            let v = est.value();
            if s.measured_range.is_some_and(|[a, b]| !(a <= v && v <= b)) {
                continue;
            }
            let (lo, hi) = est.interval(confidence);
            let by_interval = s.interval.is_some_and(|c| {
                let (a, b) = est.interval(c);
                a <= e && e <= b
            });
            out.push(Check {
                name: s.name.clone(),
                source: s.source.clone(),
                condition: r.condition.label(),
                metric: s.metric.clone(),
                expected: e,
                measured: Interval { value: v, lo, hi },
                passed: s.tolerance().accepts(v, e) || by_interval,
            });
        }
        if out.is_empty() {
            let mut funnel =
                vec![("conditions run", run), ("match `where`", matched), ("have an expected value", expected)];
            if s.min_count > 0 {
                funnel.push(("reach min_count", counted));
            }
            if s.measured_range.is_some() {
                funnel.push(("are within measured_range", 0));
            }
            return Err(self.checked_nothing(&funnel));
        }
        Ok(out)
    }
}

impl Expectation {
    /// Check the expectation on the operating points of a search: the expected value of
    /// the searched variable must lie within `atol + rtol |e|` of the estimate, or (with
    /// `interval`) inside the confidence interval of the search widened by `atol`.
    ///
    /// # Errors
    /// [`Error::Config`] if the expectation is not about the searched variable, or checks
    /// no operating point.
    pub fn check_search(&self, results: &[crate::search::SearchResult]) -> Result<Vec<Check>> {
        let s = &self.spec;
        let mut out = Vec::new();
        let mut matched = 0;
        for r in results.iter().filter(|r| self.applies(&r.condition)) {
            matched += 1;
            ensure(s.metric == r.along, || {
                format!("expectation {:?}: a search checks its variable {:?}, not {:?}", s.name, r.along, s.metric)
            })?;
            let Some(e) = self.expected(&r.condition)? else { continue };
            let atol = s.atol.unwrap_or(0.0);
            let by_interval = s.interval.is_some() && r.lo - atol <= e && e <= r.hi + atol;
            out.push(Check {
                name: s.name.clone(),
                source: s.source.clone(),
                condition: r.condition.label(),
                metric: s.metric.clone(),
                expected: e,
                measured: Interval { value: r.value, lo: r.lo, hi: r.hi },
                passed: s.tolerance().accepts(r.value, e) || by_interval,
            });
        }
        if out.is_empty() {
            return Err(self.checked_nothing(&[
                ("operating points", results.len()),
                ("match `where`", matched),
                ("have an expected value", 0),
            ]));
        }
        Ok(out)
    }
}

/// Check every expectation of a list on the operating points of a search.
///
/// # Errors
/// [`Error::Config`] for an empty list, an invalid expectation, one that is not about
/// the searched variable or one that checks no operating point.
pub fn verify_search(expect: &[Json], results: &[crate::search::SearchResult]) -> Result<Vec<Check>> {
    ensure(!expect.is_empty(), || "no expectation to verify".to_string())?;
    let mut out = Vec::new();
    for e in expect {
        out.extend(Expectation::from_json(e)?.check_search(results)?);
    }
    Ok(out)
}

/// Check every expectation of a list on the results of a run.
///
/// # Errors
/// [`Error::Config`] for an empty list, an invalid expectation, a missing metric or
/// variable, or an expectation that checks no condition: a verification never passes
/// with nothing checked.
pub fn verify(expect: &[Json], results: &[ConditionResult], confidence: f64) -> Result<Vec<Check>> {
    ensure(!expect.is_empty(), || "no expectation to verify".to_string())?;
    let mut out = Vec::new();
    for e in expect {
        out.extend(Expectation::from_json(e)?.check(results, confidence)?);
    }
    Ok(out)
}

/// An expectation with the fibre of its specification: the `fiber` of a `gn_snr_db`
/// theory merged over `fiber` (the `fiber` of the specification), as the fibre blocks
/// merge theirs ([`BuildContext::fiber`](crate::blocks::BuildContext::fiber));
/// [`Spec::from_json`] resolves its expectations so.
///
/// # Errors
/// [`Error::Config`] for a `fiber` that is not a mapping.
pub(crate) fn resolve_fiber(mut e: Json, fiber: Option<&Json>) -> Result<Json> {
    let Some(fiber) = fiber.filter(|f| !f.is_null()) else { return Ok(e) };
    let Some(t) = e.get_mut("theory").and_then(Json::as_object_mut) else { return Ok(e) };
    if t.get("function").and_then(Json::as_str) != Some("gn_snr_db") {
        return Ok(e);
    }
    let own = t.remove("fiber");
    if own.as_ref().is_some_and(|f| !(f.is_object() || f.is_null())) {
        // a variable of the grid: substituted later, not merged
        t.insert("fiber".into(), own.unwrap_or(Json::Null));
        return Ok(e);
    }
    let merged = crate::blocks::merge_fiber(Some(fiber), own.as_ref())?;
    t.insert("fiber".into(), Json::Object(merged));
    Ok(e)
}

/// What [`verify_spec`] ran: the conditions of the grid, or the search.
#[derive(Clone, Debug)]
pub enum Verified {
    /// The result of every condition.
    Run(Vec<ConditionResult>),
    /// The operating points of the search.
    Search(Vec<crate::search::SearchResult>),
}

/// Run a specification (its search, when it has one) and check its expectations: what
/// `comrs verify` and the Python `verify_spec` do.
///
/// # Errors
/// [`Error::Config`] for a specification without expectations (before anything runs);
/// the errors of the run or the search and those of [`verify`] or [`verify_search`].
pub fn verify_spec(spec: &crate::Spec, opts: &crate::RunOptions) -> Result<(Verified, Vec<Check>)> {
    ensure(!spec.expect.is_empty(), || format!("{}: no expectation to verify", spec.name))?;
    Ok(if spec.search.is_some() {
        let found = crate::search::search(spec, opts)?;
        let checks = verify_search(&spec.expect, &found)?;
        (Verified::Search(found), checks)
    } else {
        let results = crate::run(spec, opts)?;
        let checks = verify(&spec.expect, &results, spec.confidence)?;
        (Verified::Run(results), checks)
    })
}

/// The verification report of a specification: the header of the reports of its runs
/// ([`com_sim::loggers::run_header`]: the name, the seed, the batch, the threads, the
/// stopping rule and the confidence of `spec` with the options `opts`, the version and
/// commit of the library, the specification), `passed` (every check passed) and
/// `checks` (one object per check: `name`, `source`, `condition`, `metric`,
/// `expected`, `value`, `lo`, `hi`, `passed`), as JSON text.
pub fn report(spec: &Spec, opts: &RunOptions, checks: &[Check]) -> String {
    let items: Vec<String> = checks
        .iter()
        .map(|c| {
            json::object(&[
                ("name", json::string(&c.name)),
                ("source", json::string(&c.source)),
                ("condition", json::string(&c.condition)),
                ("metric", json::string(&c.metric)),
                ("expected", json::number(c.expected)),
                ("value", json::number(c.measured.value)),
                ("lo", json::number(c.measured.lo)),
                ("hi", json::number(c.measured.hi)),
                ("passed", c.passed.to_string()),
            ])
        })
        .collect();
    let mut fields = run_header(&spec.bare_runner(opts).run_info(&spec.name, spec.grid.len()));
    fields.push(("passed", checks.iter().all(|c| c.passed).to_string()));
    fields.push(("checks", json::array(&items)));
    json::object(&fields)
}

/// Write the verification report of a specification ([`report`]) to `path`, its
/// parent directories created.
///
/// # Errors
/// [`Error::Io`] if the file cannot be written.
pub fn write_report(path: &std::path::Path, spec: &Spec, opts: &RunOptions, checks: &[Check]) -> Result<()> {
    write_json(path, &report(spec, opts, checks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_sim::scenario::Record;
    use serde_json::json;

    fn result(cond: Condition, errors: u64, total: u64) -> ConditionResult {
        let mut record = Record::new();
        record.errors("ber", errors, total);
        record.value("snr", 10.0);
        record.value("snr", 10.2);
        ConditionResult {
            index: 0,
            condition: cond,
            status: com_sim::runner::Status::Completed,
            trials: 1,
            wall_s: 0.0,
            record,
        }
    }

    #[test]
    fn values_curves_and_theory() {
        let r = vec![
            result(Condition::new().with("m", 4usize).with("ebn0_db", 6.0), 2388, 1_000_000),
            result(Condition::new().with("m", 4usize).with("ebn0_db", 7.0), 100, 1_000_000),
        ];
        let theory = json!({"name": "qpsk", "metric": "ber", "theory": {"function": "ber", "m": "$m", "ebn0_db": "$ebn0_db"}, "rtol": 0.05});
        let c = verify(&[theory], &r, 0.95).unwrap();
        assert_eq!((c.len(), c[0].passed, c[1].passed), (2, true, false));
        let curve = json!({"metric": "snr", "curve": {"along": "ebn0_db", "x": [5.0, 6.5, 8.0], "y": [9.8, 10.2, null]}, "atol": 0.05});
        let c = verify(&[curve], &r, 0.95).unwrap();
        // 6 dB interpolates 9.8 → 10.2 to 10.067; 7 dB falls next to a missing point
        assert_eq!(c.len(), 1);
        assert!((c[0].expected - 10.0667).abs() < 1e-3 && c[0].passed);
        let filtered = json!({"where": {"ebn0_db": 7}, "metric": "ber", "value": 1e-4, "interval": 0.95});
        let c = verify(&[filtered], &r, 0.95).unwrap();
        assert_eq!((c.len(), c[0].passed), (1, true));
        let few = json!({"metric": "ber", "value": 1e-4, "rtol": 0.1, "min_count": 1000});
        assert_eq!(verify(&[few], &r, 0.95).unwrap().len(), 1);
    }

    #[test]
    fn an_expectation_that_checks_no_condition_is_an_error() {
        let r = vec![
            result(Condition::new().with("m", 4usize).with("ebn0_db", 6.0), 2388, 1_000_000),
            result(Condition::new().with("m", 4usize).with("ebn0_db", 7.0), 100, 1_000_000),
        ];
        // a misspelt variable, a value that is not in the grid
        for filter in [json!({"M": 4}), json!({"m": 16})] {
            let e = json!({"name": "typo", "where": filter, "metric": "ber", "value": 1e-3, "factor": 2});
            let msg = verify(&[e], &r, 0.95).unwrap_err().to_string();
            assert!(msg.contains("\"typo\" checks no condition") && msg.contains("0 match `where`"), "{msg}");
        }
        // a curve outside the grid
        let e = json!({"metric": "snr", "curve": {"along": "ebn0_db", "x": [10.0, 12.0], "y": [1.0, 2.0]}, "atol": 1});
        let msg = verify(&[e], &r, 0.95).unwrap_err().to_string();
        assert!(msg.contains("on snr checks no condition") && msg.contains("0 have an expected value"), "{msg}");
        // every condition skipped
        let e = json!({"metric": "ber", "value": 1e-3, "factor": 2, "min_count": 1_000_000});
        let msg = verify(&[e], &r, 0.95).unwrap_err().to_string();
        assert!(msg.contains("0 reach min_count"), "{msg}");
        // one empty expectation fails the whole verification, and so does an empty list
        let good = json!({"metric": "ber", "value": 1e-3, "factor": 10});
        let bad = json!({"where": {"m": 64}, "metric": "ber", "value": 1e-3, "factor": 10});
        assert_eq!(verify(std::slice::from_ref(&good), &r, 0.95).unwrap().len(), 2);
        assert!(verify(&[good, bad], &r, 0.95).is_err());
        assert!(verify(&[], &r, 0.95).is_err());
    }

    #[test]
    fn invalid_expectations_are_errors() {
        for bad in [
            json!({"metric": "ber", "value": 1.0}),
            json!({"metric": "ber", "rtol": 0.1}),
            json!({"metric": "ber", "value": 1.0, "theory": {"function": "ber", "m": 4}, "rtol": 0.1}),
            json!({"metric": "ber", "curve": {"along": "x", "x": [2, 1], "y": [0, 0]}, "rtol": 0.1}),
        ] {
            assert!(Expectation::from_json(&bad).is_err(), "{bad}");
        }
        let r = vec![result(Condition::new().with("x", 1.0), 1, 10)];
        let e = verify(&[json!({"metric": "evm", "value": 1.0, "atol": 1})], &r, 0.95).unwrap_err();
        assert!(e.to_string().contains("evm"));
    }

    #[test]
    fn every_theory_takes_its_own_parameters() {
        let r = vec![result(Condition::new().with("ebn0_db", 6.0), 2388, 1_000_000)];
        let check = |theory: Json| {
            verify(&[json!({"metric": "ber", "theory": theory, "rtol": 0.05})], &r, 0.95).map(|c| c[0].passed)
        };
        assert!(check(json!({"function": "ber", "kind": "QAM", "m": 4, "ebn0_db": "$ebn0_db"})).unwrap());
        // a parameter of another function, an unknown function, the kind without case
        let e = check(json!({"function": "ber", "m": 4, "ebn0_db": 6, "branches": 2})).unwrap_err().to_string();
        assert!(e.contains("unknown field `branches`"), "{e}");
        let e = check(json!({"function": "bler", "m": 4})).unwrap_err().to_string();
        assert!(e.contains("unknown variant `bler`") && e.contains("gn_snr_db"), "{e}");
        let rayleigh = |kind: &str| check(json!({"function": "ber_rayleigh", "kind": kind, "ebn0_db": 6, "branches": 1}));
        assert_eq!(rayleigh("QAM").unwrap(), rayleigh("qam").unwrap());
        assert!(rayleigh("pam").is_err());
    }

    #[test]
    fn the_gn_theory_sees_the_fibre_of_the_specification() {
        // the same link, its fibre given once by the specification, once by the theory
        let theory = |fiber: &str| {
            format!(
                "{{metric: snr_db, theory: {{function: gn_snr_db, {fiber}n_channels: 3, spacing_hz: 50e9, \
                 symbol_rate: 32e9, rolloff: 0.1, power_dbm: 0}}, atol: 0.1}}"
            )
        };
        let spec = |top: &str, fiber: &str| {
            Spec::from_yaml(&format!("name: g\n{top}link: [source]\nexpect: [{}]", theory(fiber))).unwrap()
        };
        let expected = |s: &Spec| {
            Expectation::from_json(&s.expect[0]).unwrap().expected(&Condition::new()).unwrap().unwrap()
        };
        let by_spec = expected(&spec("fiber: {n_spans: 10, span_km: 100}\n", ""));
        let by_theory = expected(&spec("", "fiber: {n_spans: 10, span_km: 100}, "));
        let merged = expected(&spec("fiber: {n_spans: 10}\n", "fiber: {span_km: 100}, "));
        let default = expected(&spec("", ""));
        assert_eq!(by_spec, by_theory);
        assert_eq!(merged, by_theory);
        assert!((default - by_theory).abs() > 1.0, "{default} {by_theory}");
    }
}

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
//!   `fiber`, `n_channels`, `spacing_hz`, `symbol_rate`, `rolloff`, `power_dbm` and
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
//! An expectation passes when the measured value `v` satisfies
//! `|v - e| ≤ atol + rtol |e|`, or, with `factor: f`, when `e/f ≤ v ≤ f e` (error rates
//! over several decades), or, with `interval: c`, when `e` lies in the confidence
//! interval of level `c` of the estimate; at least one of `atol`, `rtol`, `factor` and
//! `interval` is required. `min_count` skips the conditions whose estimate counts fewer errors (or
//! values), and `measured_range: [lo, hi]` those whose measured value is outside the range
//! (the domain of validity of an approximation, or the values precise enough).

use crate::params::{parse, substitute};
use com_channel::optical::gn::{FormatMoments, GnModel, WdmSpectrum};
use com_core::{Error, Result, ensure};
use com_mapping::constellation::ConstellationKind;
use com_mapping::theory;
use com_math::{db2lin, lin2db};
use com_metrics as metrics;
use com_sim::grid::{Condition, Value};
use com_sim::runner::ConditionResult;
use serde::Deserialize;
use serde_json::{Map, Value as Json};

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
}

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct CurveSpec {
    along: String,
    x: Vec<f64>,
    y: Vec<Option<f64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TheorySpec {
    function: String,
    #[serde(default = "d_qam")]
    kind: String,
    #[serde(default = "d_m")]
    m: usize,
    #[serde(default)]
    ebn0_db: Option<f64>,
    #[serde(default)]
    snr_db: Option<f64>,
    #[serde(default = "d_points")]
    points: usize,
    #[serde(default)]
    target: Option<f64>,
    #[serde(default)]
    fiber: Option<Json>,
    #[serde(default)]
    n_channels: Option<usize>,
    #[serde(default)]
    spacing_hz: Option<f64>,
    #[serde(default)]
    symbol_rate: Option<f64>,
    #[serde(default)]
    rolloff: f64,
    #[serde(default)]
    power_dbm: Option<f64>,
    #[serde(default = "d_true")]
    coherent: bool,
    #[serde(default = "d_one")]
    branches: usize,
    #[serde(default)]
    mb_lambda: Option<f64>,
    #[serde(default)]
    offset_db: f64,
    #[serde(default)]
    format: Option<Json>,
    #[serde(default)]
    modulation: Option<String>,
    #[serde(default)]
    rate: Option<String>,
}

fn d_one() -> usize {
    1
}

fn d_qam() -> String {
    "qam".into()
}
fn d_m() -> usize {
    4
}
fn d_true() -> bool {
    true
}

/// The SNR (dB) of the centre channel of a WDM link by the GN model.
fn gn_snr_db(t: &TheorySpec, name: &str) -> Result<f64> {
    let need = |what: &str| Error::Config(format!("expectation {name:?}: gn_snr_db needs {what}"));
    let ctx = crate::blocks::BuildContext::default();
    let fiber = ctx.fiber(t.fiber.as_ref())?;
    let n = t.n_channels.ok_or_else(|| need("n_channels"))?;
    let freqs = com_hardware::optical::wdm_grid(n, t.spacing_hz.ok_or_else(|| need("spacing_hz"))?);
    let rs = t.symbol_rate.ok_or_else(|| need("symbol_rate"))?;
    let p = com_math::dbm2w(t.power_dbm.ok_or_else(|| need("power_dbm"))?);
    let egn = match &t.format {
        None => None,
        Some(Json::String(s)) if s.eq_ignore_ascii_case("gaussian") => Some(FormatMoments::GAUSSIAN),
        Some(v) => {
            // the priors of the source with mb_lambda: on the unit-energy constellation
            let c = crate::blocks::parse_constellation(v)?.normalized();
            let px = t.mb_lambda.map(|l| com_mapping::source::maxwell_boltzmann(&c, l));
            Some(FormatMoments::of(c.points(), px.as_deref())?)
        }
    };
    let model = GnModel { coherent: t.coherent, raman: ctx.raman(t.fiber.as_ref())?, egn, ..GnModel::new(fiber) };
    let snr = model.snr(&WdmSpectrum::uniform(freqs, p, rs, t.rolloff), n / 2)?;
    Ok(lin2db(snr))
}
fn d_points() -> usize {
    40
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
    pub measured: (f64, f64, f64),
    /// Whether the check passed.
    pub passed: bool,
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
        ensure(spec.atol.is_some() || spec.rtol.is_some() || spec.interval.is_some() || spec.factor.is_some(), || {
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
            let x = cond.f64(&c.along)?;
            let n = c.x.len();
            if x < c.x[0] - 1e-12 || x > c.x[n - 1] + 1e-12 {
                return Ok(None);
            }
            let i = c.x.iter().position(|&xi| xi >= x - 1e-12).unwrap_or(n - 1);
            if (c.x[i] - x).abs() <= 1e-12 {
                return Ok(c.y[i]);
            }
            return Ok(match (c.y[i - 1], c.y[i]) {
                (Some(a), Some(b)) => Some(a + (b - a) * (x - c.x[i - 1]) / (c.x[i] - c.x[i - 1])),
                _ => None,
            });
        }
        let t: TheorySpec = parse("expect theory", substitute(s.theory.as_ref().unwrap_or(&Json::Null), cond)?)?;
        let kind: ConstellationKind = t.kind.parse()?;
        let bits = (t.m as f64).log2();
        let ebn0 = |t: &TheorySpec| -> Result<f64> {
            t.ebn0_db
                .or(t.snr_db.map(|s| s - lin2db(bits)))
                .ok_or_else(|| Error::Config(format!("expectation {:?}: the theory needs ebn0_db or snr_db", s.name)))
        };
        let snr = |t: &TheorySpec| -> Result<f64> {
            t.snr_db
                .or(t.ebn0_db.map(|e| e + lin2db(bits)))
                .ok_or_else(|| Error::Config(format!("expectation {:?}: the theory needs snr_db or ebn0_db", s.name)))
        };
        let target =
            || t.target.ok_or_else(|| Error::Config(format!("expectation {:?}: the theory needs a target", s.name)));
        Ok(Some(match t.function.as_str() {
            "ebn0_for_ber" => theory::ebn0_for_ber(kind, t.m, target()?)?,
            "ebn0_for_ser" => theory::ebn0_for_ser(kind, t.m, target()?)?,
            "snr_for_ber" => theory::ebn0_for_ber(kind, t.m, target()?)? + lin2db(bits),
            "snr_for_ser" => theory::ebn0_for_ser(kind, t.m, target()?)? + lin2db(bits),
            "ber" => metrics::theory_ber(kind, t.m, ebn0(&t)?)?,
            "ser" => metrics::theory_ser(kind, t.m, ebn0(&t)?)?,
            "mi" | "gmi" => {
                // Maxwell-Boltzmann priors on the unit-energy uniform constellation, as the
                // source with mb_lambda
                let px = match t.mb_lambda {
                    Some(l) => Some(com_mapping::source::maxwell_boltzmann(
                        &com_mapping::constellation::Constellation::new(kind, t.m)?.normalized(),
                        l,
                    )),
                    None => None,
                };
                let f = if t.function == "mi" { metrics::theory_mi } else { metrics::theory_gmi };
                f(kind, t.m, snr(&t)?, px.as_deref(), t.points)?
            }
            "gn_snr_db" => gn_snr_db(&t, &s.name)?,
            "dvbs2_esn0_db" => {
                let need = |w: &str| Error::Config(format!("expectation {:?}: dvbs2_esn0_db needs {w}", s.name));
                let m = t.modulation.as_deref().ok_or_else(|| need("modulation"))?;
                let r = t.rate.as_deref().ok_or_else(|| need("rate"))?;
                com_systems::dvbs2::Modcod::from_name(m, r)?.threshold_db() + t.offset_db
            }
            "ber_rayleigh" => {
                ensure(t.m <= 4 && (t.kind == "qam" || t.kind == "psk"), || {
                    format!("expectation {:?}: ber_rayleigh is for BPSK and QPSK", s.name)
                })?;
                theory::ber_rayleigh_mrc(db2lin(ebn0(&t)? + t.offset_db), t.branches)?
            }
            other => {
                return Err(Error::Config(format!(
                    "unknown theory {other:?} (ber, ser, mi, gmi, ebn0_for_ber, ebn0_for_ser, snr_for_ber, snr_for_ser, gn_snr_db, ber_rayleigh, dvbs2_esn0_db)"
                )));
            }
        }))
    }

    /// Check the expectation on the results of a run.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing metric or variable.
    pub fn check(&self, results: &[ConditionResult], confidence: f64) -> Result<Vec<Check>> {
        let s = &self.spec;
        let mut out = Vec::new();
        for r in results.iter().filter(|r| r.trials > 0 && self.applies(&r.condition)) {
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
            let v = est.value();
            if s.measured_range.is_some_and(|[a, b]| !(a <= v && v <= b)) {
                continue;
            }
            let (lo, hi) = est.interval(confidence);
            let tol = s.atol.unwrap_or(0.0) + s.rtol.unwrap_or(0.0) * e.abs();
            let by_tol = (s.atol.is_some() || s.rtol.is_some()) && (v - e).abs() <= tol;
            let by_interval = s.interval.is_some_and(|c| {
                let (a, b) = est.interval(c);
                a <= e && e <= b
            });
            let by_factor = s.factor.is_some_and(|f| e > 0.0 && v >= e / f && v <= e * f);
            out.push(Check {
                name: s.name.clone(),
                source: s.source.clone(),
                condition: r.condition.label(),
                metric: s.metric.clone(),
                expected: e,
                measured: (v, lo, hi),
                passed: by_tol || by_interval || by_factor,
            });
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
    /// [`Error::Config`] if the expectation is not about the searched variable.
    pub fn check_search(&self, results: &[crate::search::SearchResult]) -> Result<Vec<Check>> {
        let s = &self.spec;
        let mut out = Vec::new();
        for r in results.iter().filter(|r| self.applies(&r.condition)) {
            ensure(s.metric == r.along, || {
                format!("expectation {:?}: a search checks its variable {:?}, not {:?}", s.name, r.along, s.metric)
            })?;
            let Some(e) = self.expected(&r.condition)? else { continue };
            let atol = s.atol.unwrap_or(0.0);
            let by_tol =
                (s.atol.is_some() || s.rtol.is_some()) && (r.value - e).abs() <= atol + s.rtol.unwrap_or(0.0) * e.abs();
            let by_interval = s.interval.is_some() && r.lo - atol <= e && e <= r.hi + atol;
            let by_factor = s.factor.is_some_and(|f| e > 0.0 && r.value >= e / f && r.value <= e * f);
            out.push(Check {
                name: s.name.clone(),
                source: s.source.clone(),
                condition: r.condition.label(),
                metric: s.metric.clone(),
                expected: e,
                measured: (r.value, r.lo, r.hi),
                passed: by_tol || by_interval || by_factor,
            });
        }
        Ok(out)
    }
}

/// Check every expectation of a list on the operating points of a search.
///
/// # Errors
/// [`Error::Config`] for an invalid expectation or one that is not about the searched
/// variable.
pub fn verify_search(expect: &[Json], results: &[crate::search::SearchResult]) -> Result<Vec<Check>> {
    let mut out = Vec::new();
    for e in expect {
        out.extend(Expectation::from_json(e)?.check_search(results)?);
    }
    Ok(out)
}

/// Check every expectation of a list on the results of a run.
///
/// # Errors
/// [`Error::Config`] for an invalid expectation, a missing metric or variable.
pub fn verify(expect: &[Json], results: &[ConditionResult], confidence: f64) -> Result<Vec<Check>> {
    let mut out = Vec::new();
    for e in expect {
        out.extend(Expectation::from_json(e)?.check(results, confidence)?);
    }
    Ok(out)
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
}

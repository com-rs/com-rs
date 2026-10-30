//! The specification: its parsing ([`Spec`]), its grid and its stopping rule.

use crate::{params, search, verify};
use com_core::{Error, Result};
use com_sim::callbacks::SkipAfter;
use com_sim::grid::{Condition, Grid, Value};
use com_sim::stopping::{Stop, check_confidence};
use serde::Deserialize;
use serde_json::{Map, Value as Json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// A parsed specification (see the module documentation for the format).
#[derive(Clone, Debug)]
pub struct Spec {
    /// Name of the campaign (it enters the seeds of the trials).
    pub name: String,
    /// One-line description.
    pub description: String,
    /// Base seed.
    pub seed: u64,
    /// Trials per batch.
    pub batch: u64,
    /// Worker threads (0: all cores).
    pub threads: usize,
    /// Confidence level of the reported intervals and of the precision targets of the
    /// stopping rule, in $(0, 1)$.
    pub confidence: f64,
    /// The conditions.
    pub grid: Grid,
    /// Defaults of the fibre blocks.
    pub fiber: Option<Json>,
    /// The blocks of the link, before substitution.
    pub link: Vec<Json>,
    /// The metrics, before substitution.
    pub metrics: Json,
    /// The stopping rule.
    pub stop: Stop,
    /// Print one line per condition.
    pub progress: bool,
    /// Skip the rest of a sweep below an error floor (`callbacks: {skip_below: …}`).
    pub skip_below: Option<SkipAfter>,
    /// Time every block of every trial (see [`Link::trial`](crate::Link::trial)).
    pub profile: bool,
    /// CSV output.
    pub csv: Option<PathBuf>,
    /// JSON output.
    pub json: Option<PathBuf>,
    /// Directory of the specification file (relative paths of the blocks).
    pub base_dir: Option<PathBuf>,
    /// The expected results (see [`verify`]), the fibre of their `gn_snr_db` theories
    /// merged over the `fiber` of the specification.
    pub expect: Vec<Json>,
    /// The search of an operating point (see [`search`]), instead of the sweep.
    pub search: Option<search::SearchSpec>,
    /// The specification as parsed, recorded in the JSON report of every run.
    pub source: Json,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSpec {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    seed: u64,
    #[serde(default = "d_batch")]
    batch: u64,
    #[serde(default)]
    threads: usize,
    #[serde(default = "d_confidence")]
    confidence: f64,
    #[serde(default)]
    grid: Map<String, Json>,
    #[serde(default)]
    cases: Vec<Map<String, Json>>,
    #[serde(default)]
    fiber: Option<Json>,
    link: Vec<Json>,
    #[serde(default)]
    metrics: Json,
    #[serde(default)]
    stop: StopSpec,
    #[serde(default)]
    callbacks: CallbacksSpec,
    #[serde(default)]
    profile: bool,
    #[serde(default)]
    output: OutputSpec,
    #[serde(default)]
    expect: Vec<Json>,
    #[serde(default)]
    search: Option<Json>,
}

fn d_batch() -> u64 {
    32
}
fn d_confidence() -> f64 {
    0.95
}

/// The `stop` mapping.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct StopSpec {
    max_trials: Option<u64>,
    min_trials: Option<u64>,
    #[serde(default)]
    min_errors: BTreeMap<String, u64>,
    #[serde(default)]
    precision: BTreeMap<String, f64>,
    max_time_s: Option<f64>,
}

impl StopSpec {
    /// The rule, its precision targets at the confidence level of the specification.
    fn rule(&self, confidence: f64) -> Result<Stop> {
        for (m, r) in &self.precision {
            if !(r.is_finite() && *r > 0.0) {
                return Err(Error::Config(format!("stop: precision of {m:?} must be a positive number, got {r}")));
            }
        }
        let mut targets: Vec<Stop> = self.min_errors.iter().map(|(m, k)| Stop::min_errors(m, *k)).collect();
        targets.extend(self.precision.iter().map(|(m, r)| Stop::precision_at(m, *r, confidence)));
        let mut caps: Vec<Stop> = Vec::new();
        if let Some(n) = self.max_trials {
            caps.push(Stop::max_trials(n));
        }
        if let Some(t) = self.max_time_s {
            let d = Duration::try_from_secs_f64(t).ok().filter(|_| t.is_finite()).ok_or_else(|| {
                Error::Config(format!("stop: max_time_s must be a finite number of seconds >= 0, got {t}"))
            })?;
            caps.push(Stop::max_time(d));
        }
        if targets.is_empty() && caps.is_empty() {
            caps.push(Stop::max_trials(self.min_trials.unwrap_or(100).max(1)));
        }
        let mut any = Vec::new();
        if !targets.is_empty() {
            if let Some(n) = self.min_trials {
                targets.push(Stop::min_trials(n));
            }
            any.push(if targets.len() == 1 { targets.remove(0) } else { Stop::All(targets) });
        }
        any.extend(caps);
        Ok(if any.len() == 1 { any.remove(0) } else { Stop::Any(any) })
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CallbacksSpec {
    #[serde(default)]
    progress: bool,
    #[serde(default)]
    skip_below: Option<SkipBelowSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SkipBelowSpec {
    metric: String,
    threshold: f64,
    along: String,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct OutputSpec {
    csv: Option<PathBuf>,
    json: Option<PathBuf>,
}

/// The format of a specification text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// YAML (a superset of JSON).
    Yaml,
    /// JSON.
    Json,
    /// TOML.
    Toml,
}

impl Format {
    /// Read a document in this format as JSON (the specifications, and the validation
    /// cards of `com-validation`).
    ///
    /// A key repeated in a mapping is an error (the readers of YAML and JSON would keep
    /// the last value without a word; TOML refuses it itself).
    ///
    /// # Errors
    /// [`Error::Config`] for a syntax error or a repeated key.
    ///
    /// # Examples
    /// ```
    /// use com_spec::Format;
    /// assert_eq!(Format::Yaml.read("a: [1, 2]")?["a"][1], 2);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn read(self, text: &str) -> Result<Json> {
        match self {
            Format::Yaml => serde_norway::from_str::<Unique>(text)
                .map(|u| u.0)
                .map_err(|e| Error::Config(format!("YAML: {e}"))),
            Format::Json => {
                serde_json::from_str::<Unique>(text).map(|u| u.0).map_err(|e| Error::Config(format!("JSON: {e}")))
            }
            Format::Toml => toml::from_str(text).map_err(|e| Error::Config(format!("TOML: {e}"))),
        }
    }
}

/// A JSON tree read from a document, refusing a key repeated in a mapping.
struct Unique(Json);

impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Unique, D::Error> {
        d.deserialize_any(UniqueVisitor).map(Unique)
    }
}

/// The visitor of [`Unique`]: the values of `serde_json::Value`, the mappings checked.
struct UniqueVisitor;

impl<'de> serde::de::Visitor<'de> for UniqueVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> std::result::Result<Json, E> {
        Ok(Json::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> std::result::Result<Json, E> {
        Ok(Json::from(v))
    }

    fn visit_u64<E>(self, v: u64) -> std::result::Result<Json, E> {
        Ok(Json::from(v))
    }

    /// A non-finite number becomes `null`, as in `serde_json::Value`.
    fn visit_f64<E>(self, v: f64) -> std::result::Result<Json, E> {
        Ok(Json::from(v))
    }

    fn visit_str<E>(self, v: &str) -> std::result::Result<Json, E> {
        Ok(Json::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> std::result::Result<Json, E> {
        Ok(Json::String(v))
    }

    fn visit_unit<E>(self) -> std::result::Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_none<E>(self) -> std::result::Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> std::result::Result<Json, D::Error> {
        Unique::deserialize(d).map(|u| u.0)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Json, A::Error> {
        let mut out = Vec::new();
        while let Some(Unique(v)) = seq.next_element()? {
            out.push(v);
        }
        Ok(Json::Array(out))
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> std::result::Result<Json, A::Error> {
        let mut out = Map::new();
        while let Some(k) = map.next_key::<String>()? {
            let Unique(v) = map.next_value()?;
            if out.insert(k.clone(), v).is_some() {
                return Err(serde::de::Error::custom(format!("the key {k:?} is given twice in a mapping")));
            }
        }
        Ok(Json::Object(out))
    }
}

fn grid_axis(name: &str, v: &Json) -> Result<Vec<Value>> {
    match v {
        Json::Array(a) => a.iter().map(|x| params::from_json(name, x)).collect(),
        Json::Object(o) => {
            if let Some(Json::Array(l)) = o.get("linspace") {
                let f: Vec<f64> = l.iter().filter_map(Json::as_f64).collect();
                if f.len() != 3 || f[2] < 1.0 {
                    return Err(Error::Config(format!("grid {name:?}: linspace is [start, stop, count]")));
                }
                let n = f[2] as usize;
                return Ok((0..n)
                    .map(|i| Value::F64(if n == 1 { f[0] } else { f[0] + (f[1] - f[0]) * i as f64 / (n - 1) as f64 }))
                    .collect());
            }
            let get = |k: &str| o.get(k).and_then(Json::as_f64);
            match (get("from"), get("to"), get("step")) {
                (Some(a), Some(b), Some(st)) if st != 0.0 && (b - a) / st >= 0.0 => {
                    let n = ((b - a) / st + 1e-9).floor() as usize + 1;
                    let ints =
                        [a, b, st].iter().all(|v| v.fract() == 0.0) && o.values().all(|v| v.is_i64() || v.is_u64());
                    Ok((0..n)
                        .map(|i| {
                            let x = a + st * i as f64;
                            if ints { Value::I64(x as i64) } else { Value::F64(x) }
                        })
                        .collect())
                }
                _ => Err(Error::Config(format!(
                    "grid {name:?}: an axis is a list, {{from, to, step}} or {{linspace: [start, stop, count]}}"
                ))),
            }
        }
        scalar => Ok(vec![params::from_json(name, scalar)?]),
    }
}

impl Spec {
    /// Parse a specification from a JSON tree.
    ///
    /// # Errors
    /// [`Error::Config`] for a missing or unknown key, an invalid grid (a key given by
    /// a case and by an axis, see [`Grid::check`]) or rule.
    pub fn from_json(v: Json) -> Result<Spec> {
        let source = v.clone();
        // a non-finite number of YAML or TOML (`.inf`, `nan`) becomes a JSON null, which
        // an optional field would take for an absent one
        if let Some((k, _)) = v.get("stop").and_then(Json::as_object).and_then(|m| m.iter().find(|(_, x)| x.is_null()))
        {
            return Err(Error::Config(format!("stop: {k} must be a finite number, got null or a non-finite value")));
        }
        let raw: RawSpec = serde_json::from_value(v).map_err(|e| Error::Config(format!("specification: {e}")))?;
        check_confidence(raw.confidence, "confidence")?;
        let mut grid = if raw.cases.is_empty() {
            Grid::new()
        } else {
            let mut conds = Vec::new();
            for case in &raw.cases {
                let mut c = Condition::new();
                for (k, v) in case {
                    c.set(k, params::from_json(k, v)?);
                }
                conds.push(c);
            }
            Grid::from_conditions(conds)
        };
        for (k, v) in &raw.grid {
            grid = grid.axis(k, grid_axis(k, v)?);
        }
        grid.check()?;
        if raw.link.is_empty() {
            return Err(Error::Config("the link has no block".into()));
        }
        let expect = raw
            .expect
            .into_iter()
            .map(|e| verify::resolve_fiber(e, raw.fiber.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        for e in &expect {
            verify::Expectation::from_json(e)?;
        }
        Ok(Spec {
            source,
            name: raw.name,
            description: raw.description,
            seed: raw.seed,
            batch: raw.batch.max(1),
            threads: raw.threads,
            confidence: raw.confidence,
            grid,
            fiber: raw.fiber,
            link: raw.link,
            metrics: raw.metrics,
            stop: raw.stop.rule(raw.confidence)?,
            progress: raw.callbacks.progress,
            skip_below: raw.callbacks.skip_below.map(|s| SkipAfter::below(&s.metric, s.threshold, &s.along)),
            profile: raw.profile,
            csv: raw.output.csv,
            json: raw.output.json,
            base_dir: None,
            expect,
            search: raw.search.as_ref().map(search::SearchSpec::from_json).transpose()?,
        })
    }

    /// Parse a specification text.
    ///
    /// # Errors
    /// [`Error::Config`] for a syntax error or an invalid specification.
    ///
    /// # Examples
    /// ```
    /// use com_spec::{Format, Spec};
    /// let spec = Spec::parse(r#"{"name": "t", "grid": {"x": [1, 2]}, "link": ["source"]}"#, Format::Json)?;
    /// assert_eq!(spec.conditions().len(), 2);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn parse(text: &str, format: Format) -> Result<Spec> {
        Spec::from_json(format.read(text)?)
    }

    /// Parse a YAML specification.
    ///
    /// # Errors
    /// See [`Spec::parse`].
    pub fn from_yaml(text: &str) -> Result<Spec> {
        Spec::parse(text, Format::Yaml)
    }

    /// Read a specification file (`.yaml`, `.yml`, `.json` or `.toml`).
    ///
    /// # Errors
    /// [`Error::Io`] if the file cannot be read, [`Error::Config`] otherwise.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Spec> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)?;
        let format = match path.extension().and_then(|e| e.to_str()) {
            Some("json") => Format::Json,
            Some("toml") => Format::Toml,
            _ => Format::Yaml,
        };
        let mut spec = Spec::parse(&text, format).map_err(|e| e.context(path.display()))?;
        spec.base_dir = path.parent().map(Path::to_path_buf);
        Ok(spec)
    }

    /// The conditions of the grid.
    pub fn conditions(&self) -> Vec<Condition> {
        self.grid.conditions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopping_rules() {
        let s = |y: &str| Spec::from_yaml(&format!("name: t\nlink: [source]\nstop: {y}")).unwrap().stop.to_string();
        assert_eq!(s("{}"), "max_trials(100)");
        assert_eq!(s("{max_trials: 5}"), "max_trials(5)");
        assert_eq!(s("{min_errors: {ber: 10}, max_trials: 5}"), "(errors(ber) ≥ 10 or max_trials(5))");
        assert_eq!(
            s("{min_errors: {ber: 10}, precision: {snr_db: 0.1}, min_trials: 2, max_trials: 5}"),
            "((errors(ber) ≥ 10 and precision(snr_db) ≤ 0.1 at 0.95 and min_trials(2)) or max_trials(5))"
        );
    }

    #[test]
    fn non_finite_or_negative_stopping_values_are_errors() {
        let s = |y: &str| Spec::from_yaml(&format!("name: t\nlink: [source]\nstop: {y}"));
        for bad in ["{max_time_s: .inf}", "{max_time_s: -1}", "{max_time_s: .nan}", "{max_time_s: 1e300}"] {
            let e = s(bad).unwrap_err().to_string();
            assert!(e.contains("max_time_s"), "{bad}: {e}");
        }
        for bad in ["{precision: {snr_db: 0}, max_trials: 5}", "{precision: {snr_db: -0.1}, max_trials: 5}"] {
            let e = s(bad).unwrap_err().to_string();
            assert!(e.contains("precision of \"snr_db\""), "{bad}: {e}");
        }
        assert!(s("{precision: {snr_db: .inf}, max_trials: 5}").is_err());
        assert_eq!(s("{max_time_s: 0.5}").unwrap().stop.to_string(), "time ≥ 0.5 s");
    }

    #[test]
    fn the_confidence_is_a_probability_and_reaches_the_precision_targets() {
        let s = |c: &str| Spec::from_yaml(&format!("name: t\nlink: [source]\nconfidence: {c}"));
        for bad in ["95", "1", "0", "-0.5", "1.5"] {
            let e = s(bad).unwrap_err().to_string();
            assert!(e.contains("confidence") && e.contains("between 0 and 1"), "{bad}: {e}");
        }
        let spec = Spec::from_yaml(
            "name: t\nlink: [source]\nconfidence: 0.99\nstop: {precision: {snr_db: 0.1}, max_trials: 5}",
        )
        .unwrap();
        assert_eq!(spec.stop.to_string(), "(precision(snr_db) ≤ 0.1 at 0.99 or max_trials(5))");
        // the runner refuses it too, whoever builds it
        let spec = Spec::from_yaml("name: t\nlink: [source]\nstop: {max_trials: 2}").unwrap();
        let e = spec
            .bare_runner(&crate::RunOptions::default())
            .confidence(95.0)
            .run(&crate::SpecScenario::new(&spec), &spec.grid)
            .unwrap_err();
        assert!(e.to_string().contains("between 0 and 1"), "{e}");
    }

    #[test]
    fn grids_from_lists_ranges_and_cases() {
        let spec = Spec::from_yaml(
            "name: g\nlink: [source]\ncases: [{m: 4}, {m: 16}]\ngrid:\n  snr_db: {from: 0, to: 4, step: 2}\n  rate: {linspace: [1.0, 2.0, 3]}\n  kind: qam\n",
        )
        .unwrap();
        let c = spec.conditions();
        assert_eq!(c.len(), 2 * 3 * 3);
        assert_eq!(c[0].label(), "m=4, snr_db=0, rate=1, kind=qam");
        assert_eq!(c[17].label(), "m=16, snr_db=4, rate=2, kind=qam");
        assert!(matches!(c[1].get("rate"), Some(Value::F64(_))));
        assert!(matches!(c[3].get("snr_db"), Some(Value::I64(2))));
    }

    #[test]
    fn a_key_given_twice_is_an_error() {
        let e = Spec::from_yaml("name: g\nlink: [source]\ncases: [{m: 4}, {m: 16}]\ngrid: {m: [64]}").unwrap_err();
        assert!(e.to_string().contains("\"m\" twice"), "{e}");
        // a key repeated in a mapping of the document, of the grid or of a block
        for text in [
            "name: g\nlink: [source]\ngrid: {m: [4], m: [16]}",
            "name: g\nlink: [{type: awgn, snr_db: 3, snr_db: 5}]",
        ] {
            let e = Spec::from_yaml(text).unwrap_err().to_string();
            assert!(e.contains("\"m\" is given twice") || e.contains("\"snr_db\" is given twice"), "{e}");
        }
        let e = Spec::parse(r#"{"name": "g", "grid": {"m": [4], "m": [16]}, "link": ["source"]}"#, Format::Json);
        assert!(e.unwrap_err().to_string().contains("given twice"));
    }

    #[test]
    fn the_three_formats_give_the_same_specification() {
        let y = Spec::parse("name: t\ngrid: {x: [1, 2]}\nlink: [source]", Format::Yaml).unwrap();
        let j = Spec::parse(r#"{"name": "t", "grid": {"x": [1, 2]}, "link": ["source"]}"#, Format::Json).unwrap();
        let t = Spec::parse("name = \"t\"\nlink = [\"source\"]\n[grid]\nx = [1, 2]\n", Format::Toml).unwrap();
        for s in [&j, &t] {
            assert_eq!(s.conditions(), y.conditions());
            assert_eq!(s.link, y.link);
        }
    }
}

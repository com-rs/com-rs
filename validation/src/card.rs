//! The validation cards of `validation/cards/` checked from a Rust test.
//!
//! A card whose runner is a Rust test (`run: {test: validation/tests/<file>.rs::<name>}`,
//! or the whole file when several tests measure parts of it, see [`Card::part`])
//! holds the expected values, their tolerances and the reasons for them; the test only
//! measures. It loads its card, checks every measured value against the case of the
//! same inputs, and finishes, which prints the table of the comparison and fails if a
//! case failed or was never measured:
//!
//! ```no_run
//! use com_validation::card::Card;
//! use serde_json::json;
//! let mut card = Card::load("tal2015_fig1").unwrap();
//! card.check(&json!({"list": 1, "ebn0_db": 1.5}), 0.30);
//! card.finish().unwrap();
//! ```
//!
//! An expected value is a number, accepted by the tolerance of the case (or of the
//! card): `|v - e| ≤ atol + rtol |e|`, or `e/f ≤ v ≤ f e` with `factor: f`
//! ([`Tolerance`]); or a range `{min: a, max: b}` (either bound may be left out), for the
//! claims of a paper ("more than 35 dB below", "beats the list of 8").

use comrs::core::{Error, Result};
use comrs::spec::Format;
use comrs::spec::verify::Tolerance;
use serde_json::Value as Json;
use std::fmt::Write as _;
use std::path::PathBuf;

/// The folder of the cards.
pub fn folder() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cards")
}

/// What a case expects.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Expected {
    Value(f64, Tolerance),
    Range(Option<f64>, Option<f64>),
}

impl Expected {
    fn accepts(&self, v: f64) -> bool {
        match *self {
            Expected::Value(e, t) => t.accepts(v, e),
            Expected::Range(lo, hi) => lo.is_none_or(|a| v >= a) && hi.is_none_or(|b| v <= b),
        }
    }

    fn describe(&self) -> String {
        match *self {
            Expected::Value(e, t) => {
                let mut s = format!("{e:.4e}");
                if let Some(a) = t.atol {
                    let _ = write!(s, " ± {a}");
                }
                if let Some(r) = t.rtol {
                    let _ = write!(s, " ± {} %", 100.0 * r);
                }
                if let Some(f) = t.factor {
                    let _ = write!(s, " ×÷ {f}");
                }
                s
            }
            Expected::Range(lo, hi) => match (lo, hi) {
                (Some(a), Some(b)) => format!("in [{a}, {b}]"),
                (Some(a), None) => format!("≥ {a}"),
                (None, Some(b)) => format!("≤ {b}"),
                (None, None) => "any".to_string(),
            },
        }
    }
}

#[derive(Clone, Debug)]
struct Case {
    inputs: serde_json::Map<String, Json>,
    expected: Expected,
    measured: Option<f64>,
}

/// A validation card loaded for a Rust test.
#[derive(Debug)]
pub struct Card {
    id: String,
    locator: String,
    cases: Vec<Case>,
}

/// The tolerance of a case, or the one it inherits; `exact: true` is `atol: 0`.
fn tolerance(v: Option<&Json>, inherited: Tolerance) -> Result<Tolerance> {
    match v {
        None => Ok(inherited),
        Some(t) if t.get("exact") == Some(&Json::Bool(true)) => {
            Ok(Tolerance { atol: Some(0.0), ..Tolerance::default() })
        }
        Some(t) => serde_json::from_value(t.clone()).map_err(|e| Error::Config(format!("tolerance: {e}"))),
    }
}

fn expected(v: &Json, tol: Tolerance, what: &str) -> Result<Expected> {
    match v {
        Json::Number(n) => {
            let e = n.as_f64().unwrap_or(f64::NAN);
            if !tol.is_set() {
                return Err(Error::Config(format!("{what}: no tolerance (atol, rtol or factor)")));
            }
            Ok(Expected::Value(e, tol))
        }
        Json::Object(m) if m.keys().all(|k| k == "min" || k == "max") && !m.is_empty() => {
            Ok(Expected::Range(m.get("min").and_then(Json::as_f64), m.get("max").and_then(Json::as_f64)))
        }
        _ => Err(Error::Config(format!("{what}: expected a number or {{min, max}}, got {v}"))),
    }
}

/// Whether two input values are the same (numbers to 1e-12 relative).
fn same(a: &Json, b: &Json) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-12 * x.abs().max(y.abs()),
        _ => a == b,
    }
}

impl Card {
    /// Load the card `validation/cards/<id>.yaml`.
    ///
    /// # Errors
    /// [`Error::Io`] if it cannot be read, [`Error::Config`] if it is not a card whose
    /// runner is a Rust test, or a case has no tolerance.
    pub fn load(id: &str) -> Result<Card> {
        let path = folder().join(format!("{id}.yaml"));
        let v = Format::Yaml.read(&std::fs::read_to_string(&path)?)?;
        if v["id"] != id {
            return Err(Error::Config(format!("{}: the id is not {id:?}", path.display())));
        }
        if v["run"]["test"].as_str().is_none() {
            return Err(Error::Config(format!("card {id}: its runner is not a Rust test (run: {{test: …}})")));
        }
        let card_tol = tolerance(v.get("tolerance"), Tolerance::default())?;
        let cases = v["cases"]
            .as_array()
            .ok_or_else(|| Error::Config(format!("card {id}: no cases")))?
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let tol = tolerance(c.get("tolerance"), card_tol)?;
                Ok(Case {
                    inputs: c["inputs"].as_object().cloned().unwrap_or_default(),
                    expected: expected(&c["expected"], tol, &format!("card {id}, case {i}"))?,
                    measured: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let locator = v["source"]["locator"].as_str().unwrap_or_default().to_string();
        Ok(Card { id: id.to_string(), locator, cases })
    }

    /// Keep only the cases whose inputs contain `part` (a JSON object), for a card
    /// measured by several tests (`run: {test: validation/tests/<file>.rs}`), each
    /// finishing its part.
    ///
    /// # Errors
    /// [`Error::Config`] if no case matches.
    pub fn part(mut self, part: &Json) -> Result<Card> {
        let keys = part.as_object().cloned().unwrap_or_default();
        self.cases.retain(|c| keys.iter().all(|(k, v)| c.inputs.get(k).is_some_and(|w| same(v, w))));
        if self.cases.is_empty() {
            return Err(Error::Config(format!("card {}: no case has the inputs {part}", self.id)));
        }
        Ok(self)
    }

    /// Record the value measured for the case of `inputs` (a JSON object, equal to the
    /// `inputs` of one case of the card).
    ///
    /// # Panics
    /// If no case or several cases have these inputs, or the case was already measured:
    /// the test and its card disagree.
    pub fn check(&mut self, inputs: &Json, measured: f64) {
        let keys = inputs.as_object().cloned().unwrap_or_default();
        let matches: Vec<usize> = (0..self.cases.len())
            .filter(|&i| {
                let c = &self.cases[i].inputs;
                c.len() == keys.len() && keys.iter().all(|(k, v)| c.get(k).is_some_and(|w| same(v, w)))
            })
            .collect();
        assert!(matches.len() == 1, "card {}: {} cases have the inputs {inputs}", self.id, matches.len());
        let case = &mut self.cases[matches[0]];
        assert!(case.measured.is_none(), "card {}: the case {inputs} is measured twice", self.id);
        case.measured = Some(measured);
    }

    /// The expected value of the case of `inputs`, for a test that needs it (to stop a
    /// simulation, to print it).
    pub fn expected(&self, inputs: &Json) -> Option<f64> {
        let keys = inputs.as_object()?;
        self.cases
            .iter()
            .find(|c| {
                c.inputs.len() == keys.len() && keys.iter().all(|(k, v)| c.inputs.get(k).is_some_and(|w| same(v, w)))
            })
            .and_then(|c| match c.expected {
                Expected::Value(e, _) => Some(e),
                Expected::Range(..) => None,
            })
    }

    /// Print the table of the comparison and check it.
    ///
    /// # Errors
    /// [`Error::Numerical`] listing the cases that failed or were never measured.
    pub fn finish(self) -> Result<()> {
        println!("card {} ({})", self.id, self.locator);
        let mut bad = Vec::new();
        for c in &self.cases {
            let inputs = Json::Object(c.inputs.clone());
            let (status, measured) = match c.measured {
                None => ("not measured", "-".to_string()),
                Some(v) if c.expected.accepts(v) => ("pass", format!("{v:.4e}")),
                Some(v) => ("FAIL", format!("{v:.4e}")),
            };
            println!("  {inputs}: measured {measured}, expected {} — {status}", c.expected.describe());
            if status != "pass" {
                bad.push(format!("{inputs} {status}"));
            }
        }
        if bad.is_empty() { Ok(()) } else { Err(Error::Numerical(format!("card {}: {}", self.id, bad.join("; ")))) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn values_and_ranges_follow_their_rule() {
        let t = Tolerance { factor: Some(1.6), ..Tolerance::default() };
        assert!(Expected::Value(0.26, t).accepts(0.30) && !Expected::Value(0.26, t).accepts(0.5));
        assert!(Expected::Range(Some(10.0), None).accepts(14.0) && !Expected::Range(None, Some(1.0)).accepts(1.2));
        assert!(expected(&json!(1.0), Tolerance::default(), "x").is_err());
        assert!(expected(&json!({"low": 1}), t, "x").is_err());
        assert!(same(&json!(1), &json!(1.0)) && !same(&json!("a"), &json!("b")));
    }
}

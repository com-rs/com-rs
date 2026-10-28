//! Grids of simulation conditions.
//!
//! A [`Condition`] is an ordered set of named [`Value`]s (numbers, integers, strings,
//! booleans) read by the scenario with typed getters ([`Condition::f64`],
//! [`Condition::usize`], [`Condition::str`], …). A [`Grid`] produces the conditions of a
//! campaign:
//!
//! - [`Grid::axis`] adds an axis; the conditions are the **Cartesian product** of the
//!   axes, enumerated like nested loops in declaration order (the **last axis varies
//!   fastest**: an SNR declared last gives one sweep per value of the other axes);
//! - [`Grid::fixed`] adds a constant to every condition;
//! - [`Grid::zip`] adds several axes that vary together (one value of each per step);
//! - [`Grid::from_conditions`] takes an explicit list.
//!
//! # Examples
//! ```
//! use com_sim::grid::{Grid, Value};
//! let g = Grid::new()
//!     .axis("modulation", ["qpsk", "16qam"])
//!     .axis("snr_db", [10.0, 15.0, 20.0])
//!     .fixed("n_symbols", 4096usize);
//! let c = g.conditions();
//! assert_eq!(c.len(), 6);
//! assert_eq!(c[1].str("modulation").unwrap(), "qpsk");
//! assert_eq!(c[1].f64("snr_db").unwrap(), 15.0);           // the last axis varies fastest
//! assert_eq!(c[5].usize("n_symbols").unwrap(), 4096);
//! assert_eq!(c[0].label(), "modulation=qpsk, snr_db=10, n_symbols=4096");
//! assert_eq!(Value::from(3usize).as_f64(), Some(3.0));
//! ```

use com_core::{Error, Result};
use std::fmt;

/// The value of a condition entry.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A real number.
    F64(f64),
    /// An integer.
    I64(i64),
    /// A string.
    Str(String),
    /// A boolean.
    Bool(bool),
}

impl Value {
    /// The value as a real number (integers are converted).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            Value::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::F64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::Str(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}
impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::F64(v as f64)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::I64(v)
    }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::I64(v as i64)
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::I64(v as i64)
    }
}
impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::I64(v as i64)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

/// One simulation condition: named values in a fixed order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Condition {
    entries: Vec<(String, Value)>,
}

impl Condition {
    /// An empty condition.
    pub fn new() -> Condition {
        Condition::default()
    }

    /// The same condition with `key` set to `value` (replaced if present).
    pub fn with(mut self, key: &str, value: impl Into<Value>) -> Condition {
        self.set(key, value);
        self
    }

    /// Set `key` to `value` (replaced if present, appended otherwise).
    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        let v = value.into();
        match self.entries.iter_mut().find(|(k, _)| k == key) {
            Some(e) => e.1 = v,
            None => self.entries.push((key.to_string(), v)),
        }
    }

    /// The entries in order.
    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    /// The keys in order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }

    /// The value of `key`, if present.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn need(&self, key: &str) -> Result<&Value> {
        self.get(key).ok_or_else(|| Error::Condition(format!("no key {key:?} in ({})", self.label())))
    }

    /// The real number at `key` (integers are converted).
    ///
    /// # Errors
    /// [`Error::Condition`] if the key is missing or is not a number.
    pub fn f64(&self, key: &str) -> Result<f64> {
        self.need(key)?.as_f64().ok_or_else(|| Error::Condition(format!("{key:?} is not a number")))
    }

    /// The integer at `key`.
    ///
    /// # Errors
    /// [`Error::Condition`] if the key is missing or is not an integer.
    pub fn i64(&self, key: &str) -> Result<i64> {
        match self.need(key)? {
            Value::I64(v) => Ok(*v),
            Value::F64(v) if v.fract() == 0.0 => Ok(*v as i64),
            _ => Err(Error::Condition(format!("{key:?} is not an integer"))),
        }
    }

    /// The non-negative integer at `key`.
    ///
    /// # Errors
    /// [`Error::Condition`] if the key is missing or is not a non-negative integer.
    pub fn usize(&self, key: &str) -> Result<usize> {
        let v = self.i64(key)?;
        usize::try_from(v).map_err(|_| Error::Condition(format!("{key:?} = {v} is negative")))
    }

    /// The string at `key`.
    ///
    /// # Errors
    /// [`Error::Condition`] if the key is missing or is not a string.
    pub fn str(&self, key: &str) -> Result<&str> {
        match self.need(key)? {
            Value::Str(s) => Ok(s),
            _ => Err(Error::Condition(format!("{key:?} is not a string"))),
        }
    }

    /// The boolean at `key`.
    ///
    /// # Errors
    /// [`Error::Condition`] if the key is missing or is not a boolean.
    pub fn bool(&self, key: &str) -> Result<bool> {
        match self.need(key)? {
            Value::Bool(b) => Ok(*b),
            _ => Err(Error::Condition(format!("{key:?} is not a boolean"))),
        }
    }

    /// `key=value, …` in order.
    pub fn label(&self) -> String {
        self.entries.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(", ")
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Part {
    Axis(String, Vec<Value>),
    Zip(Vec<String>, Vec<Vec<Value>>),
    Fixed(String, Value),
}

/// A grid of conditions (see the module documentation).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Grid {
    parts: Vec<Part>,
    explicit: Option<Vec<Condition>>,
}

impl Grid {
    /// An empty grid (one empty condition).
    pub fn new() -> Grid {
        Grid::default()
    }

    /// An explicit list of conditions.
    pub fn from_conditions(conditions: Vec<Condition>) -> Grid {
        Grid { parts: Vec::new(), explicit: Some(conditions) }
    }

    /// Add an axis of values.
    pub fn axis<V: Into<Value>>(mut self, name: &str, values: impl IntoIterator<Item = V>) -> Grid {
        self.parts.push(Part::Axis(name.to_string(), values.into_iter().map(Into::into).collect()));
        self
    }

    /// Add axes that vary together: `rows[i]` gives one value per name.
    ///
    /// # Panics
    /// If a row does not have one value per name.
    pub fn zip(mut self, names: &[&str], rows: Vec<Vec<Value>>) -> Grid {
        assert!(rows.iter().all(|r| r.len() == names.len()), "every row needs one value per name");
        self.parts.push(Part::Zip(names.iter().map(std::string::ToString::to_string).collect(), rows));
        self
    }

    /// Add a constant to every condition.
    pub fn fixed(mut self, name: &str, value: impl Into<Value>) -> Grid {
        self.parts.push(Part::Fixed(name.to_string(), value.into()));
        self
    }

    /// The conditions, in order.
    pub fn conditions(&self) -> Vec<Condition> {
        let mut out = vec![Condition::new()];
        for p in &self.parts {
            let mut next = Vec::new();
            for c in &out {
                match p {
                    Part::Axis(name, values) => {
                        for v in values {
                            next.push(c.clone().with(name, v.clone()));
                        }
                    }
                    Part::Zip(names, rows) => {
                        for row in rows {
                            let mut c2 = c.clone();
                            for (n, v) in names.iter().zip(row) {
                                c2.set(n, v.clone());
                            }
                            next.push(c2);
                        }
                    }
                    Part::Fixed(name, v) => next.push(c.clone().with(name, v.clone())),
                }
            }
            out = next;
        }
        match &self.explicit {
            Some(list) => list
                .iter()
                .flat_map(|e| {
                    out.iter().map(move |c| {
                        let mut m = e.clone();
                        for (k, v) in c.entries() {
                            m.set(k, v.clone());
                        }
                        m
                    })
                })
                .collect(),
            None => out,
        }
    }

    /// Number of conditions.
    pub fn len(&self) -> usize {
        self.conditions().len()
    }

    /// Whether the grid has no condition.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

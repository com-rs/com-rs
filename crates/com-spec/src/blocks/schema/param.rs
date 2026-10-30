//! The parameters of a block: their documentation ([`ParamInfo`]), the JSON Schema of
//! their types ([`ParamType`]), the values given as a mapping ([`Mapping`], [`NameOr`])
//! and the parser of a mapping of parameters ([`Params`]).

use super::object_schema;
use com_core::{Error, Result};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as Json, json};
use std::fmt::Write;

/// The default of a parameter.
#[derive(Clone, Debug)]
pub enum ParamDefault {
    /// The parameter must be given.
    Required,
    /// An optional parameter left out; the text says what the block then does.
    Absent(&'static str),
    /// A value, with the text that shows it.
    Value(Json, String),
}

/// One parameter of a block.
#[derive(Clone, Debug)]
pub struct ParamInfo {
    /// Name.
    pub name: &'static str,
    /// Default.
    pub default: ParamDefault,
    /// Unit (`dB`, `Hz`, `ps/nm/km`…; empty for a number without unit or a non-number).
    pub unit: &'static str,
    /// The accepted values of a string (empty: any value of the type).
    pub choices: &'static [&'static str],
    /// Exclusive lower bound of a number.
    pub gt: Option<f64>,
    /// Inclusive lower bound of a number.
    pub ge: Option<f64>,
    /// Inclusive upper bound of a number.
    pub le: Option<f64>,
    /// Exclusive upper bound of a number.
    pub lt: Option<f64>,
    /// Meaning.
    pub doc: &'static str,
    /// The JSON Schema of the type of the value (`{"type": "number"}`…).
    pub schema: Json,
    /// The mappings the value may be, by their `type`, each with its parameters (the
    /// codes of `fec_encode`, see [`ParamType::variants`]); empty for any other value.
    pub variants: Vec<(&'static str, Vec<ParamInfo>)>,
    /// The keys of a value that is a mapping of parameters ([`Mapping`], [`NameOr`],
    /// see [`ParamType::fields`]); empty for any other value.
    pub fields: Vec<ParamInfo>,
}

impl ParamInfo {
    /// The bounds of a number, `> 0`, `<= 1`…
    fn bounds(&self) -> Vec<String> {
        [(">", self.gt), (">=", self.ge), ("<=", self.le), ("<", self.lt)]
            .iter()
            .filter_map(|(op, b)| b.map(|b| format!("{op} {b}")))
            .collect()
    }

    /// The parameter as one line of text (name, default, meaning, unit, accepted values,
    /// bounds), then one line per key of its mapping and per parameter of every variant
    /// of its value, indented.
    pub fn describe(&self, indent: usize) -> String {
        let mut doc = self.doc.to_string();
        if !self.unit.is_empty() {
            let _ = write!(doc, " [{}]", self.unit);
        }
        if !self.choices.is_empty() {
            let _ = write!(doc, " ({})", self.choices.join(", "));
        }
        let bounds = self.bounds();
        if !bounds.is_empty() {
            let _ = write!(doc, " ({})", bounds.join(", "));
        }
        let pad = " ".repeat(indent);
        let mut s = format!("{pad}{:<20} {:<14} {doc}\n", self.name, self.default_text());
        for p in &self.fields {
            s.push_str(&p.describe(indent + 4));
        }
        for (kind, params) in &self.variants {
            let _ = writeln!(s, "{pad}  {{type: {kind}}}");
            for p in params {
                s.push_str(&p.describe(indent + 4));
            }
        }
        s
    }

    /// The default as text: `required`, the value, or what an absent value means.
    pub fn default_text(&self) -> String {
        match &self.default {
            ParamDefault::Required => "required".into(),
            ParamDefault::Absent(t) => (*t).into(),
            ParamDefault::Value(_, t) => t.clone(),
        }
    }

    /// The JSON Schema of the parameter: its type, unit, accepted values, bounds,
    /// default and meaning (a string `$name` of a grid variable is accepted too).
    pub fn json_schema(&self) -> Json {
        let mut s = match &self.schema {
            Json::Object(o) => o.clone(),
            _ => Map::new(),
        };
        let mut doc = self.doc.to_string();
        if !self.unit.is_empty() {
            s.insert("x-unit".into(), json!(self.unit));
            let _ = write!(doc, " [{}]", self.unit);
        }
        s.insert("description".into(), json!(doc));
        if !self.choices.is_empty() {
            // the accepted values of a list are those of its items
            match s.get_mut("items") {
                Some(Json::Object(items)) => {
                    items.insert("enum".into(), json!(self.choices));
                }
                _ => {
                    s.insert("enum".into(), json!(self.choices));
                }
            }
        }
        let bound = |b: f64| Json::from(b);
        if let Some(b) = self.gt {
            s.insert("exclusiveMinimum".into(), bound(b));
        }
        if let Some(b) = self.ge {
            s.insert("minimum".into(), bound(b));
        }
        if let Some(b) = self.le {
            s.insert("maximum".into(), bound(b));
        }
        if let Some(b) = self.lt {
            s.insert("exclusiveMaximum".into(), bound(b));
        }
        match &self.default {
            ParamDefault::Value(v, _) => {
                s.insert("default".into(), v.clone());
            }
            ParamDefault::Absent(t) => {
                s.insert("x-default".into(), json!(t));
            }
            ParamDefault::Required => {}
        }
        json!({"anyOf": [Json::Object(s), {"type": "string", "pattern": "^\\$", "description": "a grid variable"}]})
    }

    /// Check a given value against the accepted values and the bounds.
    #[cfg(test)]
    pub(crate) fn check(&self, kind: &str, v: &Json) -> Result<()> {
        self.accept(kind, v.clone()).map(drop)
    }

    /// Check a given value against the accepted values and the bounds, and return it
    /// with every string among the accepted values in its declared spelling: the
    /// accepted values are compared without case ([`choice`]), and the parser of the
    /// block then sees the spelling it declared.
    pub(crate) fn accept(&self, kind: &str, mut v: Json) -> Result<Json> {
        if !self.choices.is_empty() {
            let mut all = true;
            let mut canonical = |x: &mut Json| {
                if let Json::String(s) = x {
                    match choice(self.choices, s) {
                        Some(c) => *s = c.to_string(),
                        None => all = false,
                    }
                }
            };
            match &mut v {
                Json::Array(a) => a.iter_mut().for_each(&mut canonical),
                x => canonical(x),
            }
            if !all {
                return Err(Error::Config(format!(
                    "block {kind:?}: {} must be one of {:?}, got {v}",
                    self.name, self.choices
                )));
            }
        }
        let numbers: Vec<f64> = match &v {
            Json::Number(n) => n.as_f64().into_iter().collect(),
            Json::Array(a) => a.iter().filter_map(Json::as_f64).collect(),
            _ => Vec::new(),
        };
        for x in numbers {
            let fails = self.gt.is_some_and(|b| x <= b)
                || self.ge.is_some_and(|b| x < b)
                || self.le.is_some_and(|b| x > b)
                || self.lt.is_some_and(|b| x >= b);
            if fails {
                return Err(Error::Config(format!(
                    "block {kind:?}: {} must be {}, got {x}",
                    self.name,
                    self.bounds().join(" and ")
                )));
            }
        }
        Ok(v)
    }
}

/// The accepted value that `s` names, compared without ASCII case: the one rule of the
/// accepted values of the blocks, for the check of a parameter and its parsing alike.
///
/// # Examples
/// ```
/// use com_spec::blocks::schema::choice;
/// assert_eq!(choice(&["signal", "unit"], "Unit"), Some("unit"));
/// assert_eq!(choice(&["signal", "unit"], "peak"), None);
/// ```
pub fn choice(choices: &[&'static str], s: &str) -> Option<&'static str> {
    choices.iter().copied().find(|c| c.eq_ignore_ascii_case(s))
}

/// The JSON Schema of the type of a parameter.
pub trait ParamType {
    /// The schema (`{"type": "number"}`, `{"type": "array", "items": …}`…).
    fn schema() -> Json;
    /// The mappings a value of the type may be, by their `type`, each with its
    /// parameters ([`ParamInfo::variants`]): none by default.
    fn variants() -> Vec<(&'static str, Vec<ParamInfo>)> {
        Vec::new()
    }
    /// The keys of a value of the type that is a mapping of parameters
    /// ([`ParamInfo::fields`]): none by default.
    fn fields() -> Vec<ParamInfo> {
        Vec::new()
    }
}

/// The value of a parameter that is a mapping of the parameters `T` (declared with
/// [`block_params!`](crate::block_params)), kept as given until the block parses it
/// ([`Mapping::parse`]; the fibre blocks merge it over the fibre of the specification
/// first): its keys are documented with those of the block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mapping<T> {
    /// The mapping as given.
    pub value: Json,
    _params: std::marker::PhantomData<fn() -> T>,
}

impl<'de, T> serde::Deserialize<'de> for Mapping<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Ok(Mapping { value: Json::deserialize(d)?, _params: std::marker::PhantomData })
    }
}

impl<T: Params> Mapping<T> {
    /// Read the parameters of the mapping (`kind` names it in the errors).
    ///
    /// # Errors
    /// As [`Params::parse`].
    pub fn parse(&self, kind: &str) -> Result<T> {
        T::parse(kind, self.value.clone())
    }
}

impl<T: Params> ParamType for Mapping<T> {
    fn schema() -> Json {
        Json::Object(object_schema(&T::params(), &[]))
    }

    fn fields() -> Vec<ParamInfo> {
        T::params()
    }
}

/// The value of a parameter that is a name or a mapping of the parameters `T` (a
/// constellation `16-QAM` or `{kind: qam, m: 16}`, the pilots `none` or `{pattern: comb,
/// …}`), kept as given until the block reads it ([`NameOr::get`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NameOr<T>(pub Mapping<T>);

/// What a [`NameOr`] is.
#[derive(Clone, Debug, PartialEq)]
pub enum Named<'a, T> {
    /// A name.
    Name(&'a str),
    /// A mapping, its parameters read.
    Params(T),
}

impl<'de, T> serde::Deserialize<'de> for NameOr<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Mapping::deserialize(d).map(NameOr)
    }
}

impl<T: Params> NameOr<T> {
    /// The name, or the parameters of the mapping (`kind` names it in the errors).
    ///
    /// # Errors
    /// [`Error::Config`] for a value that is neither a string nor a valid mapping.
    pub fn get(&self, kind: &str) -> Result<Named<'_, T>> {
        match &self.0.value {
            Json::String(s) => Ok(Named::Name(s)),
            Json::Object(_) => self.0.parse(kind).map(Named::Params),
            other => Err(Error::Config(format!("{kind}: a name or a mapping, got {other}"))),
        }
    }

    /// The value as given.
    pub fn json(&self) -> &Json {
        &self.0.value
    }
}

impl<T: Params> ParamType for NameOr<T> {
    fn schema() -> Json {
        json!({"anyOf": [{"type": "string"}, Mapping::<T>::schema()]})
    }

    fn fields() -> Vec<ParamInfo> {
        T::params()
    }
}

impl ParamType for f64 {
    fn schema() -> Json {
        json!({"type": "number"})
    }
}

macro_rules! unsigned {
    ($($t:ty),*) => {$(
        impl ParamType for $t {
            fn schema() -> Json {
                json!({"type": "integer", "minimum": 0})
            }
        }
    )*};
}
unsigned!(usize, u64, u32, u8);

impl ParamType for i64 {
    fn schema() -> Json {
        json!({"type": "integer"})
    }
}

impl ParamType for bool {
    fn schema() -> Json {
        json!({"type": "boolean"})
    }
}

impl ParamType for String {
    fn schema() -> Json {
        json!({"type": "string"})
    }
}

/// Any JSON value: a constellation name or mapping, a code, a list of taps…
impl ParamType for Json {
    fn schema() -> Json {
        json!({})
    }
}

impl<T: ParamType> ParamType for Option<T> {
    fn schema() -> Json {
        T::schema()
    }

    fn variants() -> Vec<(&'static str, Vec<ParamInfo>)> {
        T::variants()
    }

    fn fields() -> Vec<ParamInfo> {
        T::fields()
    }
}

impl<T: ParamType, const N: usize> ParamType for [T; N] {
    fn schema() -> Json {
        json!({"type": "array", "items": T::schema(), "minItems": N, "maxItems": N})
    }
}

impl<T: ParamType> ParamType for Vec<T> {
    fn schema() -> Json {
        json!({"type": "array", "items": T::schema()})
    }

    fn fields() -> Vec<ParamInfo> {
        T::fields()
    }
}

json_param! {
    /// A number, or a list of numbers (one per stage, per channel…).
    NumberOrList,
    json!({"anyOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}}]})
}

/// The parameters of a block, declared with `block_params!`.
pub trait Params: Sized {
    /// The documentation of every parameter.
    fn params() -> Vec<ParamInfo>;
    /// Read the parameters of a block of type `kind` (a JSON mapping).
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown parameter, a missing one, a value of the wrong
    /// type, out of its bounds or not among its accepted values.
    fn parse(kind: &str, v: Json) -> Result<Self> {
        let params = Self::params();
        let mut m = mapping(kind, v, &params)?;
        Self::take_from(kind, &mut m, &params)
    }
    /// Take the parameters from the mapping `m` of a block whose parameters are `params`
    /// (these among them): the parser of a group of parameters shared by several blocks
    /// (a `[flatten]` field of `block_params!`), whose unknown names
    /// [`mapping`] rejected against the parameters of the whole block.
    ///
    /// # Errors
    /// As [`Params::parse`].
    fn take_from(kind: &str, m: &mut Map<String, Json>, params: &[ParamInfo]) -> Result<Self>;
}

/// The mapping of the parameters of a block, without unknown names.
///
/// # Errors
/// [`Error::Config`] if `v` is not a mapping or has a name that `params` does not
/// declare (the message lists the accepted ones, as serde does).
pub fn mapping(kind: &str, v: Json, params: &[ParamInfo]) -> Result<Map<String, Json>> {
    let m = match v {
        Json::Object(m) => m,
        Json::Null => Map::new(),
        other => return Err(Error::Config(format!("block {kind:?}: the parameters must be a mapping, got {other}"))),
    };
    if let Some(unknown) = m.keys().find(|k| !params.iter().any(|p| p.name == k.as_str())) {
        let names: Vec<String> = params.iter().map(|p| format!("`{}`", p.name)).collect();
        let expected = match names.len() {
            0 => "there are no fields".to_string(),
            1 => format!("expected {}", names[0]),
            _ => format!("expected one of {}", names.join(", ")),
        };
        return Err(Error::Config(format!("block {kind:?}: unknown field `{unknown}`, {expected}")));
    }
    Ok(m)
}

/// Take the value of the parameter `name` from the mapping, or its default.
///
/// # Errors
/// [`Error::Config`] for a missing required parameter, a value of the wrong type, out of
/// its bounds or not among its accepted values.
pub fn take<T: DeserializeOwned>(kind: &str, m: &mut Map<String, Json>, params: &[ParamInfo], name: &str) -> Result<T> {
    let Some(p) = params.iter().find(|p| p.name == name) else {
        return Err(Error::Config(format!("block {kind:?}: undeclared parameter {name}")));
    };
    let v = match m.remove(name) {
        Some(v) => p.accept(kind, v)?,
        None => match &p.default {
            ParamDefault::Required => return Err(Error::Config(format!("block {kind:?}: missing field `{name}`"))),
            ParamDefault::Absent(_) => Json::Null,
            ParamDefault::Value(v, _) => v.clone(),
        },
    };
    serde_json::from_value(v).map_err(|e| Error::Config(format!("block {kind:?}: {name}: {e}")))
}

/// The text of a default written in `block_params!`: a string literal without its
/// quotes, anything else as written.
pub fn shown(tokens: &str) -> String {
    tokens.strip_prefix('"').and_then(|t| t.strip_suffix('"')).unwrap_or(tokens).to_string()
}

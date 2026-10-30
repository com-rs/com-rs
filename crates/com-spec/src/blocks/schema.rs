//! The single description of a block: its parameters and its ports.
//!
//! The parameters of a block are declared once, with
//! [`block_params!`](crate::block_params): the name and the Rust type of every
//! parameter, its default, its unit, its accepted values, its bounds and its meaning.
//! The macro writes the structure, its parser ([`Params`]) and its documentation
//! ([`ParamInfo`]), so that what a block accepts, what `comrs blocks` prints, what the
//! Python classes of `comrs.blocks` take and the JSON Schema of the links
//! ([`json_schema`]) cannot differ. A block of another crate declares its parameters the
//! same way (see [`register`](super::register)).
//!
//! ```
//! use com_spec::blocks::Params;
//! com_spec::block_params! {
//!     struct AwgnSpec {
//!         snr_db: f64 [required] "dB", "signal-to-noise ratio";
//!         reference: String ["signal"] one_of ["signal", "unit"], "power reference of the SNR";
//!         bandwidth_hz: Option<f64> [absent "symbol rate"] "Hz" > 0, "noise bandwidth";
//!     }
//! }
//! let p = AwgnSpec::parse("awgn", serde_json::json!({"snr_db": 10})).unwrap();
//! assert_eq!((p.snr_db, p.reference.as_str(), p.bandwidth_hz), (10.0, "signal", None));
//! assert_eq!(AwgnSpec::params()[2].default_text(), "symbol rate");
//! assert!(AwgnSpec::parse("awgn", serde_json::json!({"snr_db": 10, "bandwidth_hz": 0})).is_err());
//! ```
//!
//! A default is `required`, `absent` (an `Option` left to `None`, with the text that says
//! what the block then does) or a JSON value, read as the value of a specification
//! would be (`VALUE => "TEXT"` shows it as `TEXT`). Accepted values (`one_of`) are
//! compared without case; bounds (`>`, `>=`, `<=`) apply to a number and to every number
//! of a list.
//!
//! # Ports
//!
//! A block also declares what it reads from the [`Frame`] of a trial and what it may
//! add to it, as [`Fact`]s: `awgn` needs `samples` and gives `noise`, `demap` needs
//! `samples`, `symbols`, `constellation` and `noise` and gives `llr`. [`check_link`]
//! checks a link before it runs, so that a missing block is named at once (`demap
//! needs noise: add awgn, awgn_ebn0, noise_estimate… before it`), and a graphical editor
//! can tell which blocks may follow which. In debug builds every block checks its
//! declaration while it runs: a fact it gives must be declared, and a fact it needs must
//! have been there.

use super::Frame;
use com_core::{Error, Result};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as Json, json};
use std::fmt::Write;

/// What the [`Frame`] of a trial holds at some point of a link.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Fact {
    /// Transmitted bits (the labels of the symbols, or the coded bits before `map`).
    Bits,
    /// Transmitted symbols.
    Symbols,
    /// The constellation of the transmitted symbols.
    Constellation,
    /// Samples.
    Samples,
    /// A noise variance: of the channel, declared by the receiver or per symbol.
    Noise,
    /// The fading of the trial (channel state information).
    Csi,
    /// The information bits of a coded link.
    Payload,
    /// Bit LLRs.
    Llr,
    /// Decoded information bits.
    Decoded,
    /// An OFDM frame between `ofdm_tx` and `ofdm_rx`.
    Ofdm,
    /// A WDM grid between `wdm_tx` and `wdm_rx`.
    Wdm,
    /// The reference sequence of a measured capture, not yet aligned.
    Capture,
}

impl Fact {
    /// Every fact.
    pub const ALL: [Fact; 12] = [
        Fact::Bits,
        Fact::Symbols,
        Fact::Constellation,
        Fact::Samples,
        Fact::Noise,
        Fact::Csi,
        Fact::Payload,
        Fact::Llr,
        Fact::Decoded,
        Fact::Ofdm,
        Fact::Wdm,
        Fact::Capture,
    ];

    /// The name of the fact (`noise`, `llr`…).
    pub fn name(self) -> &'static str {
        match self {
            Fact::Bits => "bits",
            Fact::Symbols => "symbols",
            Fact::Constellation => "constellation",
            Fact::Samples => "samples",
            Fact::Noise => "noise",
            Fact::Csi => "csi",
            Fact::Payload => "payload",
            Fact::Llr => "llr",
            Fact::Decoded => "decoded",
            Fact::Ofdm => "ofdm",
            Fact::Wdm => "wdm",
            Fact::Capture => "capture",
        }
    }

    /// Whether the frame holds the fact.
    pub fn holds(self, s: &Frame) -> bool {
        match self {
            Fact::Bits => !s.bits.is_empty(),
            Fact::Symbols => !s.symbols.is_empty(),
            Fact::Constellation => s.constellation.is_some(),
            Fact::Samples => !s.samples.is_empty(),
            Fact::Noise => {
                s.receiver.sigma2.is_some() || s.channel.noise_var.is_some() || !s.receiver.symbol_noise.is_empty()
            }
            Fact::Csi => s.channel.csi.is_some(),
            Fact::Payload => !s.coding.payload.is_empty(),
            Fact::Llr => !s.coding.llr.is_empty(),
            Fact::Decoded => !s.coding.decoded.is_empty(),
            Fact::Ofdm => s.in_flight.ofdm.is_some(),
            Fact::Wdm => s.in_flight.wdm.is_some(),
            Fact::Capture => !s.capture.reference.is_empty(),
        }
    }
}

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
    /// Meaning.
    pub doc: &'static str,
    /// The JSON Schema of the type of the value (`{"type": "number"}`…).
    pub schema: Json,
    /// The mappings the value may be, by their `type`, each with its parameters (the
    /// codes of `fec_encode`, see [`ParamType::variants`]); empty for any other value.
    pub variants: Vec<(&'static str, Vec<ParamInfo>)>,
}

impl ParamInfo {
    /// A parameter documented by text only (its type is not declared).
    pub fn text(name: &'static str, default: &'static str, doc: &'static str) -> ParamInfo {
        let default = match default {
            "required" => ParamDefault::Required,
            other => ParamDefault::Absent(other),
        };
        ParamInfo {
            name,
            default,
            unit: "",
            choices: &[],
            gt: None,
            ge: None,
            le: None,
            doc,
            schema: json!({}),
            variants: Vec::new(),
        }
    }

    /// The parameter as one line of text (name, default, meaning, unit, accepted values,
    /// bounds), then one line per parameter of every variant of its value, indented.
    pub fn describe(&self, indent: usize) -> String {
        let mut doc = self.doc.to_string();
        if !self.unit.is_empty() {
            let _ = write!(doc, " [{}]", self.unit);
        }
        if !self.choices.is_empty() {
            let _ = write!(doc, " ({})", self.choices.join(", "));
        }
        let bounds: Vec<String> = [(">", self.gt), ("≥", self.ge), ("≤", self.le)]
            .iter()
            .filter_map(|(op, b)| b.map(|b| format!("{op} {b}")))
            .collect();
        if !bounds.is_empty() {
            let _ = write!(doc, " ({})", bounds.join(", "));
        }
        let pad = " ".repeat(indent);
        let mut s = format!("{pad}{:<20} {:<14} {doc}\n", self.name, self.default_text());
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
            let fails =
                self.gt.is_some_and(|b| x <= b) || self.ge.is_some_and(|b| x < b) || self.le.is_some_and(|b| x > b);
            if fails {
                let mut range = Vec::new();
                if let Some(b) = self.gt {
                    range.push(format!("> {b}"));
                }
                if let Some(b) = self.ge {
                    range.push(format!(">= {b}"));
                }
                if let Some(b) = self.le {
                    range.push(format!("<= {b}"));
                }
                return Err(Error::Config(format!(
                    "block {kind:?}: {} must be {}, got {x}",
                    self.name,
                    range.join(" and ")
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
}

/// The JSON Schema of a mapping with a `type` among `variants`, each with its parameters
/// (their types, defaults, units, accepted values and bounds), for
/// [`ParamType::schema`].
pub fn variants_schema(variants: &[(&'static str, Vec<ParamInfo>)]) -> Json {
    let one_of: Vec<Json> = variants
        .iter()
        .map(|(kind, params)| {
            let mut props = Map::new();
            props.insert("type".into(), json!({"const": kind}));
            let mut required = vec![json!("type")];
            for p in params {
                props.insert(p.name.into(), p.json_schema());
                if matches!(p.default, ParamDefault::Required) {
                    required.push(json!(p.name));
                }
            }
            json!({"type": "object", "title": kind, "properties": props, "required": required, "additionalProperties": false})
        })
        .collect();
    json!({"type": "object", "oneOf": one_of})
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
    fn parse(kind: &str, v: Json) -> Result<Self>;
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

/// Declare the parameters of a block: the structure, its [`Params`] parser and its
/// documentation (see the [module documentation](self)).
///
/// ```text
/// block_params! {
///     [pub] struct NAME {
///         FIELD: TYPE [DEFAULT] ["UNIT"] [one_of [VALUE, …]] [> B] [>= B] [<= B], "MEANING";
///         …
///     }
/// }
/// ```
///
/// `DEFAULT` is `required`, `absent`, `absent "TEXT"`, a value or `VALUE => "TEXT"`; the
/// type of a field implements [`ParamType`] (its JSON Schema) and `Deserialize`.
#[macro_export]
macro_rules! block_params {
    (@default required) => { $crate::blocks::schema::ParamDefault::Required };
    (@default absent) => { $crate::blocks::schema::ParamDefault::Absent("none") };
    (@default absent $text:literal) => { $crate::blocks::schema::ParamDefault::Absent($text) };
    (@default $v:expr => $text:literal) => {
        $crate::blocks::schema::ParamDefault::Value($crate::__macro::json!($v), $text.to_string())
    };
    (@default $v:expr) => {
        $crate::blocks::schema::ParamDefault::Value(
            $crate::__macro::json!($v),
            $crate::blocks::schema::shown(stringify!($v)),
        )
    };
    (@or $d:literal) => { $d };
    (@or $d:literal $v:literal) => { $v };
    (@bound) => { None };
    (@bound $b:literal) => { Some(f64::from($b)) };
    (
        $(#[$sattr:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$fattr:meta])*
                $field:ident : $ty:ty [ $($def:tt)+ ]
                $($unit:literal)?
                $(one_of [$($choice:literal),+ $(,)?])?
                $(> $gt:literal)?
                $(>= $ge:literal)?
                $(<= $le:literal)?
                , $doc:expr ;
            )*
        }
    ) => {
        $(#[$sattr])*
        $vis struct $name {
            $( $(#[$fattr])* $field: $ty, )*
        }

        impl $crate::blocks::schema::Params for $name {
            fn params() -> Vec<$crate::blocks::schema::ParamInfo> {
                vec![$(
                    $crate::blocks::schema::ParamInfo {
                        name: stringify!($field),
                        default: $crate::block_params!(@default $($def)+),
                        unit: $crate::block_params!(@or "" $($unit)?),
                        choices: &[$($($choice),+)?],
                        gt: $crate::block_params!(@bound $($gt)?),
                        ge: $crate::block_params!(@bound $($ge)?),
                        le: $crate::block_params!(@bound $($le)?),
                        doc: $doc,
                        schema: <$ty as $crate::blocks::schema::ParamType>::schema(),
                        variants: <$ty as $crate::blocks::schema::ParamType>::variants(),
                    }
                ),*]
            }

            #[allow(unused_mut, unused_variables, reason = "a block without parameters")]
            fn parse(kind: &str, v: $crate::__macro::Json) -> $crate::__macro::Result<Self> {
                let params = Self::params();
                let mut m = $crate::blocks::schema::mapping(kind, v, &params)?;
                Ok($name {
                    $( $field: $crate::blocks::schema::take(kind, &mut m, &params, stringify!($field))?, )*
                })
            }
        }
    };
}

/// Check the ports of a link: every block finds the facts it needs among those that
/// the blocks before it give.
///
/// # Errors
/// [`Error::Config`] naming the first block that misses a fact, and the blocks that give
/// it.
pub fn check_link(kinds: &[&str], registry: &[super::BlockInfo]) -> Result<()> {
    let mut have: Vec<Fact> = Vec::new();
    for (i, k) in kinds.iter().enumerate() {
        let Some(b) = registry.iter().find(|b| b.name == *k) else { continue };
        if let Some(f) = b.needs.iter().find(|f| !have.contains(f)) {
            let givers: Vec<&str> = registry.iter().filter(|g| g.gives.contains(f)).map(|g| g.name).collect();
            return Err(Error::Config(format!(
                "link block {} ({k}) needs {}, which no block before it gives: add one of {givers:?} before it",
                i + 1,
                f.name()
            )));
        }
        have.extend(b.gives.iter().copied());
    }
    Ok(())
}

/// The JSON Schema of a block of a link: one alternative per block of the registry (a
/// mapping with its `type`, an optional `label` and its parameters, or its name alone
/// when it needs no parameter).
pub fn json_schema(registry: &[super::BlockInfo]) -> Json {
    let mut alternatives = Vec::new();
    for b in registry {
        let mut props = Map::new();
        props.insert("type".into(), json!({"const": b.name}));
        props.insert("label".into(), json!({"type": "string", "description": "prefix of the recorded values"}));
        let mut required = vec![json!("type")];
        for p in &b.params {
            props.insert(p.name.into(), p.json_schema());
            if matches!(p.default, ParamDefault::Required) {
                required.push(json!(p.name));
            }
        }
        alternatives.push(json!({
            "type": "object",
            "title": b.name,
            "description": b.summary,
            "properties": props,
            "required": required,
            "additionalProperties": false,
            "x-needs": b.needs.iter().map(|f| f.name()).collect::<Vec<_>>(),
            "x-gives": b.gives.iter().map(|f| f.name()).collect::<Vec<_>>(),
            "x-starts": b.starts,
        }));
        if required.len() == 1 {
            alternatives.push(json!({"const": b.name, "description": b.summary}));
        }
    }
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "com-rs link block",
        "oneOf": alternatives,
    })
}

/// A block that checks its declaration while it runs (debug builds): the facts it
/// gains must be declared in `gives` (what it needs is checked before the link runs,
/// by [`check_link`]).
#[cfg_attr(not(debug_assertions), allow(dead_code, reason = "built in debug builds only"))]
pub(super) struct Checked {
    pub(super) inner: Box<dyn super::Block>,
    pub(super) kind: String,
    pub(super) gives: &'static [Fact],
}

impl super::Block for Checked {
    fn apply(&self, s: &mut Frame, rng: &mut com_core::rng::Rng) -> Result<()> {
        let before: Vec<Fact> = Fact::ALL.into_iter().filter(|f| f.holds(s)).collect();
        self.inner.apply(s, rng)?;
        if let Some(f) = Fact::ALL.into_iter().find(|f| f.holds(s) && !before.contains(f) && !self.gives.contains(f)) {
            panic!("block {:?} gave {}, which it does not declare in gives", self.kind, f.name());
        }
        Ok(())
    }

    fn needs_training(&self) -> bool {
        self.inner.needs_training()
    }

    fn train(&mut self, s: &Frame, rng: &mut com_core::rng::Rng) -> Result<()> {
        self.inner.train(s, rng)
    }
}

//! Substitution of the grid variables and conversions between the condition values and
//! the JSON tree of a specification.

use com_core::{Error, Result};
use com_sim::grid::{Condition, Value};
use serde_json::{Map, Number, Value as Json};

/// A condition value as JSON.
pub(crate) fn to_json(v: &Value) -> Result<Json> {
    Ok(match v {
        Value::F64(x) => Json::Number(
            Number::from_f64(*x).ok_or_else(|| Error::Config(format!("the value {x} cannot be substituted")))?,
        ),
        Value::I64(x) => Json::Number((*x).into()),
        Value::Str(s) => Json::String(s.clone()),
        Value::Bool(b) => Json::Bool(*b),
    })
}

/// A JSON scalar as a condition value (integers stay integers).
pub(crate) fn from_json(key: &str, v: &Json) -> Result<Value> {
    match v {
        Json::Number(n) => Ok(match n.as_i64() {
            Some(i) => Value::I64(i),
            None => Value::F64(n.as_f64().unwrap_or(f64::NAN)),
        }),
        Json::String(s) => Ok(Value::Str(s.clone())),
        Json::Bool(b) => Ok(Value::Bool(*b)),
        other => {
            Err(Error::Config(format!("grid value of {key:?} must be a number, a string or a boolean, got {other}")))
        }
    }
}

/// Whether `s` is a variable name (letters, digits, `_`, `.`, not starting with a digit).
fn is_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

fn lookup<'a>(cond: &'a Condition, name: &str) -> Result<&'a Value> {
    cond.get(name).ok_or_else(|| {
        let keys: Vec<&str> = cond.keys().collect();
        Error::Config(format!("unknown variable ${name}: the grid defines {keys:?}"))
    })
}

/// The variable that a string of a specification is exactly, `name` for `$name` or
/// `${name}` (letters, digits, `_` and `.`, not starting with a digit), `None` for any
/// other string.
///
/// # Examples
/// ```
/// use com_spec::variable;
/// assert_eq!(variable("$ebn0_db"), Some("ebn0_db"));
/// assert_eq!(variable("${fiber.span_km}"), Some("fiber.span_km"));
/// assert_eq!(variable("${a}_${b}"), None);
/// ```
pub fn variable(s: &str) -> Option<&str> {
    let name = s.strip_prefix('$')?;
    let name = name.strip_prefix('{').and_then(|n| n.strip_suffix('}')).unwrap_or(name);
    is_name(name).then_some(name)
}

/// Replace the variables of the condition in a JSON tree.
///
/// - a string that is exactly `$name` or `${name}` becomes the value of `name`, with its
///   type (number, string or boolean);
/// - `${name}` inside a longer string is replaced by the text of the value;
/// - `$$` escapes a dollar sign.
///
/// # Errors
/// [`Error::Config`] for a variable that the condition does not define.
///
/// # Examples
/// ```
/// use com_sim::grid::{Condition, Value};
/// use serde_json::json;
/// let mut cond = Condition::new();
/// cond.set("a", Value::I64(4));
/// cond.set("b", Value::Str("QAM".into()));
/// let v = com_spec::substitute(&json!({"m": "$a", "name": "${a}_${b}", "cost": "$$5"}), &cond)?;
/// assert_eq!(v, json!({"m": 4, "name": "4_QAM", "cost": "$5"}));
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn substitute(tree: &Json, cond: &Condition) -> Result<Json> {
    Ok(match tree {
        Json::String(s) => substitute_str(s, cond)?,
        Json::Array(a) => Json::Array(a.iter().map(|v| substitute(v, cond)).collect::<Result<_>>()?),
        Json::Object(o) => {
            let mut m = Map::new();
            for (k, v) in o {
                m.insert(k.clone(), substitute(v, cond)?);
            }
            Json::Object(m)
        }
        other => other.clone(),
    })
}

fn substitute_str(s: &str, cond: &Condition) -> Result<Json> {
    if let Some(name) = variable(s) {
        return to_json(lookup(cond, name)?);
    }
    if !s.contains('$') {
        return Ok(Json::String(s.to_string()));
    }
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('$') {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 1..];
        if let Some(t) = tail.strip_prefix('$') {
            out.push('$');
            rest = t;
        } else if let Some(t) = tail.strip_prefix('{') {
            let end = t.find('}').ok_or_else(|| Error::Config(format!("unclosed ${{ in {s:?}")))?;
            out.push_str(&lookup(cond, &t[..end])?.to_string());
            rest = &t[end + 1..];
        } else {
            out.push('$');
            rest = tail;
        }
    }
    out.push_str(rest);
    Ok(Json::String(out))
}

/// Remove the keys `type` and `label` of a block and return `(type, label, params)`.
pub(crate) fn split_block(v: &Json) -> Result<(String, Option<String>, Json)> {
    match v {
        Json::String(s) => Ok((s.clone(), None, Json::Object(Map::new()))),
        Json::Object(o) => {
            let mut o = o.clone();
            let kind = match o.remove("type") {
                Some(Json::String(t)) => t,
                Some(other) => return Err(Error::Config(format!("block type must be a string, got {other}"))),
                None => return Err(Error::Config(format!("block without a type: {v}"))),
            };
            let label = match o.remove("label") {
                Some(Json::String(l)) => Some(l),
                None => None,
                Some(other) => return Err(Error::Config(format!("block label must be a string, got {other}"))),
            };
            Ok((kind, label, Json::Object(o)))
        }
        other => Err(Error::Config(format!("a block is a name or a mapping with a type, got {other}"))),
    }
}

/// Deserialize the parameters of a block, with the block type in the error message.
///
/// # Errors
/// [`Error::Config`] naming the block and the field serde refuses.
pub fn parse<T: serde::de::DeserializeOwned>(kind: &str, v: Json) -> Result<T> {
    serde_json::from_value(v).map_err(|e| Error::Config(format!("block {kind:?}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn variables_keep_their_type_or_interpolate() {
        let c = Condition::new().with("snr_db", 12.5).with("m", 16usize).with("kind", "qam");
        let t = json!({"a": "$snr_db", "b": ["${m}", "x"], "c": "out/${kind}_${m}.csv", "d": "$$5", "e": 3});
        let r = substitute(&t, &c).unwrap();
        assert_eq!(r, json!({"a": 12.5, "b": [16, "x"], "c": "out/qam_16.csv", "d": "$5", "e": 3}));
        assert!(substitute(&json!("$missing"), &c).unwrap_err().to_string().contains("missing"));
    }

    #[test]
    fn blocks_split_into_type_label_and_parameters() {
        let (k, l, p) = split_block(&json!({"type": "awgn", "label": "rx", "snr_db": 3})).unwrap();
        assert_eq!((k.as_str(), l.as_deref(), p), ("awgn", Some("rx"), json!({"snr_db": 3})));
        assert_eq!(split_block(&json!("bps")).unwrap().0, "bps");
        assert!(split_block(&json!({"snr_db": 3})).is_err());
    }
}

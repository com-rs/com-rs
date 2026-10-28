//! A minimal JSON writer for the reports (no dependency).

use std::fmt::Write;

/// A JSON string literal.
pub fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON number (`null` for a non-finite value).
pub fn number(v: f64) -> String {
    if v.is_finite() { format!("{v}") } else { "null".to_string() }
}

/// A JSON object from already-encoded values.
pub fn object(fields: &[(&str, String)]) -> String {
    let body: Vec<String> = fields.iter().map(|(k, v)| format!("{}: {}", string(k), v)).collect();
    format!("{{{}}}", body.join(", "))
}

/// A JSON array from already-encoded values.
pub fn array(items: &[String]) -> String {
    format!("[{}]", items.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaping() {
        assert_eq!(string("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
        assert_eq!(number(f64::NAN), "null");
        assert_eq!(object(&[("x", number(1.5))]), "{\"x\": 1.5}");
    }
}

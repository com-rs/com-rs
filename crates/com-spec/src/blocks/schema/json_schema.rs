//! The JSON Schema of a mapping of parameters and of a block of a link.

use super::{ParamDefault, ParamInfo};
use crate::blocks::BlockInfo;
use serde_json::{Map, Value as Json, json};
use std::borrow::Borrow;

/// The JSON Schema of a mapping of parameters: its `properties` (the schema of every
/// parameter), the `required` ones, no other key; `fixed` adds properties first (the
/// `type` of a block or of a variant) and makes them required.
pub fn object_schema(params: &[ParamInfo], fixed: &[(&str, Json)]) -> Map<String, Json> {
    let mut props = Map::new();
    let mut required: Vec<Json> = Vec::new();
    for (k, v) in fixed {
        props.insert((*k).into(), v.clone());
        required.push(json!(k));
    }
    for p in params {
        props.insert(p.name.into(), p.json_schema());
        if matches!(p.default, ParamDefault::Required) {
            required.push(json!(p.name));
        }
    }
    let mut s = Map::new();
    s.insert("type".into(), json!("object"));
    s.insert("properties".into(), Json::Object(props));
    s.insert("required".into(), Json::Array(required));
    s.insert("additionalProperties".into(), json!(false));
    s
}

/// The JSON Schema of a mapping with a `type` among `variants`, each with its parameters
/// (their types, defaults, units, accepted values and bounds), for
/// [`ParamType::schema`](super::ParamType::schema).
pub fn variants_schema(variants: &[(&'static str, Vec<ParamInfo>)]) -> Json {
    let one_of: Vec<Json> = variants
        .iter()
        .map(|(kind, params)| {
            let mut s = object_schema(params, &[("type", json!({"const": kind}))]);
            s.insert("title".into(), json!(kind));
            Json::Object(s)
        })
        .collect();
    json!({"type": "object", "oneOf": one_of})
}

/// The JSON Schema of a block of a link: one alternative per block of the registry (a
/// mapping with its `type`, an optional `label` and its parameters, or its name alone
/// when it needs no parameter).
pub fn json_schema<B: Borrow<BlockInfo>>(registry: &[B]) -> Json {
    let mut alternatives = Vec::new();
    for b in registry.iter().map(Borrow::borrow) {
        let mut s = object_schema(&b.params, &[("type", json!({"const": b.name}))]);
        if let Some(Json::Object(props)) = s.get_mut("properties") {
            props.insert("label".into(), json!({"type": "string", "description": "prefix of the recorded values"}));
        }
        let only_type = s.get("required").and_then(Json::as_array).is_some_and(|r| r.len() == 1);
        s.insert("title".into(), json!(b.name));
        s.insert("description".into(), json!(b.summary));
        s.insert("x-needs".into(), json!(b.needs.iter().map(|f| f.name()).collect::<Vec<_>>()));
        s.insert("x-gives".into(), json!(b.gives.iter().map(|f| f.name()).collect::<Vec<_>>()));
        s.insert("x-starts".into(), json!(b.starts));
        alternatives.push(Json::Object(s));
        if only_type {
            alternatives.push(json!({"const": b.name, "description": b.summary}));
        }
    }
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "com-rs link block",
        "oneOf": alternatives,
    })
}

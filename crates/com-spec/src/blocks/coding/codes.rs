//! The codes of `fec_encode` and `fec_decode`: their parameters, the LDPC construction
//! built once, and [`build_code`].

use crate::blocks::schema::{ParamType, variants_schema};
use crate::blocks::{ParamInfo, Params};
use com_core::{Error, Result};
use com_fec::block::BlockCode;
use com_fec::conv::ConvolutionalCode;
use com_fec::crc::Crc;
use com_fec::ldpc::{self, CheckRule, LdpcCode};
use com_fec::polar::PolarCode;
use com_fec::puncture::{Punctured, Puncturer};
use com_fec::rs::ReedSolomon;
use com_fec::turbo::{TurboCode, TurboDecoder};
use com_math::Jacobian;
use serde::Deserialize;
use serde_json::Value as Json;

// ------------------------------------------------------------------------- codes

block_params! {
    struct ConvSpec {
        k: usize [required] "bits" >= 1, "information bits per codeword";
        hard: bool [false], "hard-decision Viterbi decoding";
        puncture: Option<String> [absent "unpunctured"],
            "puncturing pattern of com_fec::puncture::Puncturer::from_name (ieee80211_2_3, dvbs_3_4…)";
    }
}

block_params! {
    struct RsSpec {
        m: u32 [required] "bits" >= 2 <= 16, "bits per symbol: the field GF(2^m)";
        t: usize [required] "symbols", "symbol errors corrected";
        n: Option<usize> [absent "2^m - 1"] "symbols", "length of the shortened code";
        first_root: usize [1], "first root b of the generator α^b, …";
        root_step: usize [1], "step c between the roots of the generator";
    }
}

block_params! {
    struct TurboSpec {
        k: usize [required] "bits" >= 8, "information bits: an interleaver length of LTE";
        iterations: usize [8], "decoding iterations";
        max_log: bool [false], "max-log-MAP instead of log-MAP";
    }
}

block_params! {
    /// The construction and the decoder of a PEG LDPC code: the parameters shared by
    /// the `ldpc` code of `fec_encode` and `fec_decode` and by `pas_encode` and
    /// `pas_decode`.
    pub(super) struct LdpcSettings {
        var_degree: usize [ldpc::DEFAULT_VAR_DEGREE => "3"] >= 1, "variable-node degree of the PEG code";
        seed: u64 [ldpc::DEFAULT_PEG_SEED => "1"], "seed of the PEG construction";
        rule: String [ldpc::DEFAULT_RULE.name() => "sum_product"]
            one_of ["sum_product", "min_sum", "normalized_min_sum", "offset_min_sum"],
            "check-node rule";
        alpha: f64 [ldpc::DEFAULT_ALPHA => "0.8"], "scale or offset of min-sum";
        layered: bool [ldpc::DEFAULT_LAYERED => "true"], "layered schedule";
        iterations: usize [ldpc::DEFAULT_MAX_ITERATIONS => "50"] >= 1, "decoding iterations at most";
    }
}

/// The regular LDPC code of length `n` with `k` information bits and variable-node
/// degree `var_degree` built by progressive edge growth from `seed`, built once:
/// `fec_encode` and `fec_decode` (and `pas_encode` and `pas_decode`) build the same
/// code from one mapping, and the construction is deterministic.
fn peg(n: usize, k: usize, var_degree: usize, seed: u64) -> Result<LdpcCode> {
    /// The codes built, by `(n, k, var_degree, seed)`, the last ones kept.
    type Built = Vec<((usize, usize, usize, u64), LdpcCode)>;
    static BUILT: std::sync::Mutex<Built> = std::sync::Mutex::new(Vec::new());
    const KEPT: usize = 8;
    let key = (n, k, var_degree, seed);
    let found = BUILT.lock().ok().and_then(|b| b.iter().find(|(k, _)| *k == key).map(|(_, c)| c.clone()));
    if let Some(c) = found {
        return Ok(c);
    }
    let c = LdpcCode::peg(n, k, var_degree, seed)?;
    if let Ok(mut b) = BUILT.lock() {
        if b.len() == KEPT {
            b.remove(0);
        }
        b.push((key, c.clone()));
    }
    Ok(c)
}

impl LdpcSettings {
    /// The regular LDPC code of length `n` with `k` information bits built by
    /// progressive edge growth, with these decoder settings.
    pub(super) fn code(&self, n: usize, k: usize) -> Result<LdpcCode> {
        let mut c = peg(n, k, self.var_degree, self.seed)?;
        c.decoder.rule = CheckRule::from_name(&self.rule, self.alpha)?;
        c.decoder.layered = self.layered;
        c.decoder.max_iterations = self.iterations;
        Ok(c)
    }
}

block_params! {
    struct LdpcSpec {
        n: usize [required] "bits", "length of the codeword";
        k: usize [required] "bits", "information bits";
        settings: LdpcSettings [flatten], "the construction and the decoder";
    }
}

block_params! {
    struct PolarSpec {
        n: usize [required] "bits", "length of the codeword, a power of 2";
        k: usize [required] "bits", "information bits (a CRC comes on top)";
        design_ebn0_db: f64 [2.0] "dB", "Eb/N0 of the Gaussian-approximation design";
        crc: Option<String> [absent "no CRC"], "CRC of CRC-aided SCL: crc6, crc11, crc16, crc24a, crc24b, crc24c";
        list: usize [1], "list size (1: successive cancellation)";
    }
}

/// The documentation of the parameters of a code.
type ParamsFn = fn() -> Vec<ParamInfo>;

/// The codes of `fec_encode` and `fec_decode`: the value of their `type` and their
/// parameters.
const CODES: [(&str, ParamsFn); 5] = [
    ("conv", ConvSpec::params),
    ("rs", RsSpec::params),
    ("turbo", TurboSpec::params),
    ("ldpc", LdpcSpec::params),
    ("polar", PolarSpec::params),
];

/// The value of the `code` parameter: a mapping with the `type` of the code and its
/// parameters, described in the documentation and the JSON Schema of the blocks.
#[derive(Clone, Deserialize)]
#[serde(transparent)]
pub(super) struct Code(pub(super) Json);

impl ParamType for Code {
    fn schema() -> Json {
        variants_schema(&Code::variants())
    }

    fn variants() -> Vec<(&'static str, Vec<ParamInfo>)> {
        CODES.iter().map(|&(kind, params)| (kind, params())).collect()
    }
}

/// Build a code from its specification (a mapping with a `type`).
///
/// # Errors
/// [`com_core::Error::Config`] for an unknown type or invalid parameters.
pub fn build_code(v: &Json) -> Result<Box<dyn BlockCode + Send + Sync>> {
    let mut p = match v {
        Json::Object(o) => o.clone(),
        other => return Err(Error::Config(format!("a code is a mapping with a type, got {other}"))),
    };
    let names = CODES.map(|(k, _)| k);
    let Some(Json::String(given)) = p.remove("type") else {
        return Err(Error::Config(format!("a code needs a type among {names:?}")));
    };
    let kind =
        crate::blocks::schema::choice(&names, &given).ok_or_else(|| crate::unknown("code type", &given, &names))?;
    let what = format!("{kind} code");
    let p = Json::Object(p);
    Ok(match kind {
        "conv" => {
            let ConvSpec { k, hard, puncture } = ConvSpec::parse(&what, p)?;
            let mut code = ConvolutionalCode::k7(k)?;
            code.hard_decision = hard;
            match puncture {
                None => Box::new(code),
                Some(name) => {
                    Box::new(Punctured::new(code, Puncturer::from_name(&name).map_err(|e| e.context("conv"))?)?)
                }
            }
        }
        "rs" => {
            let RsSpec { m, t, n, first_root, root_step } = RsSpec::parse(&what, p)?;
            let n = n.unwrap_or((1usize << m) - 1);
            let poly = com_math::gf2m::primitive_polynomial(m).unwrap_or(0);
            Box::new(ReedSolomon::with_field(m, poly, first_root, root_step, t, n)?)
        }
        "turbo" => {
            let TurboSpec { k, iterations, max_log } = TurboSpec::parse(&what, p)?;
            let mut code = TurboCode::lte(k)?;
            code.decoder = TurboDecoder { iterations, jacobian: Jacobian::from_max_log(max_log) };
            Box::new(code)
        }
        "ldpc" => {
            let q = LdpcSpec::parse(&what, p)?;
            Box::new(q.settings.code(q.n, q.k)?)
        }
        "polar" => {
            let PolarSpec { n, k, design_ebn0_db, crc, list } = PolarSpec::parse(&what, p)?;
            let mut c = PolarCode::new(n, k, design_ebn0_db)?;
            if let Some(name) = crc {
                c = c.with_crc(Crc::from_name(&name)?)?;
            }
            Box::new(c.with_list(list)?)
        }
        _ => unreachable!("a name of CODES"),
    })
}

#[cfg(test)]
mod tests {
    // the fields of the parameters are private to this module
    use super::super::pas::PasSpec;
    use super::*;
    use serde_json::json;

    #[test]
    fn the_ldpc_defaults_are_those_of_com_fec() {
        use crate::blocks::schema::ParamDefault;
        // the PAS blocks and the LDPC code of fec_encode document the same defaults
        for params in [PasSpec::params(), LdpcSpec::params()] {
            let shown: Vec<(&str, String, Json)> = params
                .into_iter()
                .filter_map(|p| match p.default {
                    ParamDefault::Value(v, text) => Some((p.name, text, v)),
                    _ => None,
                })
                .collect();
            let get = |name: &str| shown.iter().find(|(n, _, _)| *n == name).cloned().unwrap();
            for (name, value) in [
                ("var_degree", json!(ldpc::DEFAULT_VAR_DEGREE)),
                ("seed", json!(ldpc::DEFAULT_PEG_SEED)),
                ("rule", json!(ldpc::DEFAULT_RULE.name())),
                ("alpha", json!(ldpc::DEFAULT_ALPHA)),
                ("layered", json!(ldpc::DEFAULT_LAYERED)),
                ("iterations", json!(ldpc::DEFAULT_MAX_ITERATIONS)),
            ] {
                let (_, text, v) = get(name);
                assert_eq!(v, value, "{name}");
                // the documented text is the value
                assert_eq!(text, value.as_str().map_or_else(|| value.to_string(), str::to_string), "{name}");
            }
        }
        let q = LdpcSpec::parse("ldpc code", json!({"n": 128, "k": 64})).unwrap().settings;
        assert_eq!(
            (q.var_degree, q.seed, q.rule.as_str()),
            (ldpc::DEFAULT_VAR_DEGREE, ldpc::DEFAULT_PEG_SEED, "sum_product")
        );
        assert_eq!(
            (q.alpha, q.layered, q.iterations),
            (ldpc::DEFAULT_ALPHA, ldpc::DEFAULT_LAYERED, ldpc::DEFAULT_MAX_ITERATIONS)
        );
    }
}

//! Coding blocks: channel codes, mapping of the coded bits and soft demapping
//! (bit-interleaved coded modulation).
//!
//! | Block | Does |
//! |---|---|
//! | `fec_encode` | draws the information bits of `blocks` codewords and encodes them, optionally through a random bit interleaver |
//! | `map` | maps the coded bits on a constellation (the last symbol padded with random bits) |
//! | `demap` | the bit LLRs of the received symbols with their noise variances (exact or max-log) |
//! | `fec_decode` | deinterleaves and decodes the LLRs |
//!
//! A code is a mapping with a `type`:
//!
//! | `type` | Parameters | Code |
//! |---|---|---|
//! | `conv` | `k`, `hard`, `puncture` | the $(133, 171)_{8}$ code of constraint length 7, terminated, Viterbi decoding; `puncture` names a pattern of `com_fec::puncture::Puncturer::from_name` (`ieee80211_2_3`, `ieee80211_3_4`, `ieee80211_5_6`, `dvbs_2_3`, `dvbs_3_4`, `dvbs_5_6`, `dvbs_7_8`) |
//! | `rs` | `m`, `t`, `n`, `first_root`, `root_step` | a Reed–Solomon code over $\mathrm{GF}(2^{m})$ correcting $t$ symbols, shortened to $n$ symbols, hard decisions (`n = 2^m - 1`, $b = 1$, $c = 1$ by default; CCSDS: `m: 8, t: 16, first_root: 112, root_step: 11`) |
//! | `turbo` | `k`, `iterations`, `max_log` | the turbo code of LTE (log-MAP or max-log-MAP) |
//! | `ldpc` | `n`, `k`, `var_degree`, `seed`, `rule`, `alpha`, `layered`, `iterations` | a regular LDPC code built by progressive edge growth; the defaults are those of `com_fec::ldpc` (degree 3, seed 1, layered sum-product, 50 iterations, $\alpha = 0.8$) |
//! | `polar` | `n`, `k`, `design_ebn0_db`, `crc`, `list` | a polar code of the Gaussian approximation, SC or CRC-aided SCL decoding |
//!
//! The parameters of every code are declared as those of the blocks: `comrs blocks
//! fec_encode` lists them with their defaults, units and accepted values, and the JSON
//! Schema of `comrs schema` describes them.
//!
//! The encoder and the decoder are built from the same mapping (the decoder settings
//! are ignored by the encoder), so that a specification writes the code once with a
//! YAML anchor:
//!
//! ```yaml
//! link:
//!   - {type: fec_encode, code: &code {type: ldpc, n: 1024, k: 512}, blocks: 4, interleave: true}
//!   - {type: map, constellation: 16-QAM}
//!   - {type: rayleigh, coherence: 1}
//!   - {type: awgn_ebn0, ebn0_db: $ebn0_db}
//!   - mimo_rx
//!   - demap
//!   - {type: fec_decode, code: *code, interleave: true}
//! metrics: {report: [ber, coded_ber, bler]}
//! ```
//!
//! `awgn_ebn0` after `fec_encode` counts the energy per information bit: $E_{s}/N_{0} =
//! R \log_{2} M \cdot E_{b}/N_{0}$ with the code rate $R$.
//!
//! # References
//!
//! 1. G. Caire, G. Taricco and E. Biglieri, "Bit-interleaved coded modulation," *IEEE
//!    Trans. Inf. Theory*, 44(3), 927–946, 1998.

use super::schema::{ParamType, variants_schema};
use super::{Block, Coding, ConstellationParam, Extension, Frame, ParamInfo, Params, parse_constellation};
use com_core::rng::Rng;
use com_core::{Error, Result, ensure};
use com_fec::block::BlockCode;
use com_fec::conv::ConvolutionalCode;
use com_fec::crc::Crc;
use com_fec::interleaving::Interleaver;
use com_fec::ldpc::{self, CheckRule, LdpcCode};
use com_fec::polar::PolarCode;
use com_fec::puncture::{Punctured, Puncturer};
use com_fec::rs::ReedSolomon;
use com_fec::turbo::TurboCode;
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_systems::pas::Pas;
use serde::Deserialize;
use serde_json::Value as Json;

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            start "pas_encode",
            "Probabilistic amplitude shaping: CCDM-shaped amplitudes, an LDPC code whose parity bits are the signs, square QAM (the first block of a link).",
            PasSpec,
            needs [],
            gives [Bits, Symbols, Constellation, Samples, Payload],
            |q, _, _| Ok(Box::new(PasEncode(PasLink::new(&q)?))),
        ),
        builtin!(
            "pas_decode",
            "PAS receiver: bit LLRs with the amplitude priors, LDPC decoding, dematching (the parameters of pas_encode).",
            PasSpec,
            needs [Samples, Symbols, Noise],
            gives [Decoded],
            |q, _, _| Ok(Box::new(PasDecode(PasLink::new(&q)?))),
        ),
        builtin!(
            start "fec_encode",
            "Draw the information bits of the coded blocks and encode them (the first block of a coded link).",
            EncodeSpec,
            needs [],
            gives [Bits, Payload],
            |q, _, _| Ok(Box::new(Encode(Coder::new(&q.code, q.blocks, q.interleaver)?))),
        ),
        builtin!(
            "map",
            "Map the coded bits on a constellation (Gray labels, the last symbol padded with random bits).",
            MapSpec,
            needs [Bits],
            gives [Symbols, Constellation, Samples],
            |q, _, _| Ok(Box::new(Map::new(&q)?)),
        ),
        builtin!(
            "demap",
            "Bit LLRs of the received symbols, with the noise variance of every symbol (from the receiver, or the noise block).",
            Demap,
            needs [Samples, Symbols, Constellation, Noise],
            gives [Llr],
            |q, _, _| Ok(Box::new(q)),
        ),
        builtin!(
            "fec_decode",
            "Deinterleave and decode the LLRs into the information bits.",
            DecodeSpec,
            needs[Llr],
            gives[Decoded],
            |q, _, _| Ok(Box::new(Decode(Coder::new(&q.code, 1, q.interleaver)?))),
        ),
    ]
}

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
    struct LdpcSettings {
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
    fn code(&self, n: usize, k: usize) -> Result<LdpcCode> {
        let mut c = peg(n, k, self.var_degree, self.seed)?;
        c.rule = CheckRule::from_name(&self.rule, self.alpha)?;
        c.layered = self.layered;
        c.max_iterations = self.iterations;
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
struct Code(Json);

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
    let kind = super::schema::choice(&names, &given).ok_or_else(|| crate::unknown("code type", &given, &names))?;
    let what = format!("{kind} code");
    let p = Json::Object(p);
    Ok(match kind {
        "conv" => {
            let ConvSpec { k, hard, puncture } = ConvSpec::parse(&what, p)?;
            let code = ConvolutionalCode { hard_decision: hard, ..ConvolutionalCode::k7(k) };
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
            Box::new(TurboCode { iterations, exact: !max_log, ..TurboCode::lte(k)? })
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
            Box::new(c.with_list(list))
        }
        _ => unreachable!("a name of CODES"),
    })
}

block_params! {
    /// The bit interleaver of a coded link: the parameters shared by `fec_encode` and
    /// `fec_decode`, which must give the same.
    #[derive(Clone, Copy)]
    struct InterleaverSettings {
        interleave: bool [false], "random bit interleaver over the coded bits of the trial (BICM), the same in fec_encode and fec_decode";
        seed: u64 [0], "seed of the interleaver";
    }
}

block_params! {
    struct EncodeSpec {
        code: Code [required], "the code: {type: conv|rs|turbo|ldpc|polar, …} and its parameters";
        blocks: usize [1] >= 1, "codewords per trial";
        interleaver: InterleaverSettings [flatten], "the bit interleaver";
    }
}

block_params! {
    struct DecodeSpec {
        code: Code [required], "the code of fec_encode, with the decoder settings";
        interleaver: InterleaverSettings [flatten], "the bit interleaver of fec_encode";
    }
}

/// A code with its interleaver.
struct Coder {
    code: Box<dyn BlockCode + Send + Sync>,
    blocks: usize,
    interleaver: InterleaverSettings,
}

impl Coder {
    fn new(code: &Code, blocks: usize, interleaver: InterleaverSettings) -> Result<Coder> {
        Ok(Coder { code: build_code(&code.0)?, blocks, interleaver })
    }

    fn interleaver(&self, len: usize) -> Option<Interleaver> {
        self.interleaver.interleave.then(|| Interleaver::random(len, self.interleaver.seed))
    }
}

struct Encode(Coder);

impl Block for Encode {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let c = &self.0;
        let payload: Vec<u8> = (0..c.code.k() * c.blocks).map(|_| rng.bit()).collect();
        let mut bits = c.code.encode_stream(&payload)?;
        if let Some(il) = c.interleaver(bits.len()) {
            bits = il.interleave(&bits);
        }
        *s = Frame {
            trial: s.trial,
            bits,
            coding: Coding { payload, n_blocks: c.blocks, code_rate: Some(c.code.code_rate()), ..Coding::default() },
            sps: 1.0,
            symbol_rate: 1.0,
            ..Frame::default()
        };
        Ok(())
    }
}

struct Decode(Coder);

impl Block for Decode {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let c = &self.0;
        let n = c.code.n() * s.coding.n_blocks.max(1);
        ensure(s.coding.llr.len() >= n, || {
            format!("fec_decode needs {n} LLRs (a demap block before it), the signal has {}", s.coding.llr.len())
        })?;
        let llr = &s.coding.llr[..n];
        let llr = match c.interleaver(n) {
            Some(il) => il.deinterleave(llr),
            None => llr.to_vec(),
        };
        s.coding.decoded = c.code.decode_stream(&llr)?;
        Ok(())
    }
}

// --------------------------------------------------------------------- map, demap

block_params! {
    struct MapSpec {
        constellation: ConstellationParam ["QPSK"], "the constellation, as for source";
        symbol_rate: f64 [1.0] "Bd" > 0, "symbol rate";
    }
}

struct Map {
    c: Constellation,
    symbol_rate: f64,
}

impl Map {
    fn new(p: &MapSpec) -> Result<Map> {
        Ok(Map { c: parse_constellation(p.constellation.json())?, symbol_rate: p.symbol_rate })
    }
}

impl Block for Map {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        ensure(!s.bits.is_empty(), || "map needs coded bits: place it after fec_encode".into())?;
        let b = self.c.bits_per_symbol();
        let mut bits = std::mem::take(&mut s.bits);
        while bits.len() % b != 0 {
            bits.push(rng.bit());
        }
        let idx = self.c.indices(&bits)?;
        s.constellation = Some(self.c.clone());
        s.set_indices(idx);
        s.samples = s.symbols.clone();
        s.sps = 1.0;
        s.symbol_rate = self.symbol_rate;
        Ok(())
    }
}

block_params! {
    struct Demap {
        max_log: bool [false], "max-log approximation instead of the exact LLRs";
    }
}

impl Block for Demap {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let pooled = s.pooled();
        let p: &Frame = &pooled;
        p.require_symbols("demap")?;
        let c = p.constellation("demap")?;
        let var = p.symbol_noise("demap")?;
        let llr = if var.windows(2).all(|w| w[0] == w[1]) {
            c.llr(&p.samples, var.first().copied().unwrap_or(1.0), p.pmf.as_deref(), self.max_log)?
        } else {
            c.llr_with_variances(&p.samples, &var, p.pmf.as_deref(), self.max_log)?
        };
        drop(pooled);
        s.coding.llr = llr;
        Ok(())
    }
}

// --------------------------------------------------------------------------- PAS

block_params! {
    struct PasSpec {
        constellation: ConstellationParam ["64-QAM"], "a square QAM: 2^m-PAM per dimension";
        mb_lambda: f64 [1.0] >= 0,
            "Maxwell-Boltzmann amplitudes p(a) ∝ exp(-λ x²) on the unit-energy uniform QAM (0: uniform)";
        n: usize [1536] "bits", "length of the LDPC codeword, a multiple of 2m";
        k: Option<usize> [absent "n(m-1)/m"] "bits",
            "information bits of the code; above n(m-1)/m the signs also carry data (γ > 0)";
        ldpc: LdpcSettings [flatten], "the LDPC code, as the ldpc code of fec_encode";
        blocks: usize [1] >= 1, "codewords per trial (pas_encode)";
        symbol_rate: f64 [1.0] "Bd" > 0, "symbol rate (pas_encode)";
    }
}

/// A PAS link over a square QAM: one PAS transceiver per real dimension, the in-phase
/// and quadrature components of a QAM symbol being two consecutive PAM symbols.
struct PasLink {
    c: Constellation,
    pas: Pas,
    blocks: usize,
    symbol_rate: f64,
    delta: f64,
    levels: usize,
}

impl PasLink {
    fn new(p: &PasSpec) -> Result<PasLink> {
        let c = parse_constellation(p.constellation.json())?;
        let bits = c.bits_per_symbol();
        let side = 1usize << (bits / 2);
        ensure(bits % 2 == 0 && bits >= 4 && side * side == c.order(), || {
            "pas: the constellation must be a square QAM of at least 16 points".into()
        })?;
        let m = bits / 2;
        // the half spacing of the unit-energy QAM: its smallest coordinate
        let delta = c.points().iter().map(|z| z.re.abs()).fold(f64::INFINITY, f64::min);
        let amplitudes: Vec<f64> =
            (0..side / 2).map(|a| (-p.mb_lambda * ((2 * a + 1) as f64 * delta).powi(2)).exp()).collect();
        ensure(p.n % (2 * m) == 0, || {
            format!("pas: n must be a multiple of {} (two PAM symbols per QAM symbol)", 2 * m)
        })?;
        let k = p.k.unwrap_or(p.n / m * (m - 1));
        let code = p.ldpc.code(p.n, k)?;
        let pas = Pas::new(code, m, &amplitudes, delta)?;
        Ok(PasLink { c, pas, blocks: p.blocks, symbol_rate: p.symbol_rate, delta, levels: side })
    }

    /// The probability of every point of the constellation: the product of the level
    /// distributions of its coordinates.
    fn pmf(&self) -> Vec<f64> {
        let level = self.pas.level_distribution();
        let index = |v: f64| {
            let i = ((v / self.delta + (self.levels - 1) as f64) / 2.0).round();
            level[(i.max(0.0) as usize).min(self.levels - 1)]
        };
        self.c.points().iter().map(|z| index(z.re) * index(z.im)).collect()
    }
}

struct PasEncode(PasLink);

impl Block for PasEncode {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let l = &self.0;
        let k = l.pas.info_bits();
        let payload: Vec<u8> = (0..k * l.blocks).map(|_| rng.bit()).collect();
        let mut pam = Vec::with_capacity(l.pas.n_symbols() * l.blocks);
        for chunk in payload.chunks(k) {
            pam.extend(l.pas.encode(chunk)?);
        }
        let z: Vec<C64> = pam.chunks(2).map(|w| C64::new(w[0], w[1])).collect();
        let idx: Vec<usize> = z.iter().map(|v| l.c.nearest(*v)).collect();
        let n_qam = z.len();
        *s = Frame {
            trial: s.trial,
            constellation: Some(l.c.clone()),
            pmf: Some(l.pmf()),
            coding: Coding {
                payload,
                n_blocks: l.blocks,
                // information bits per QAM symbol, over its log2 M bits
                code_rate: Some((k * l.blocks) as f64 / (n_qam * l.c.bits_per_symbol()) as f64),
                ..Coding::default()
            },
            sps: 1.0,
            symbol_rate: l.symbol_rate,
            ..Frame::default()
        };
        s.set_indices(idx);
        s.samples = s.symbols.clone();
        Ok(())
    }
}

struct PasDecode(PasLink);

impl Block for PasDecode {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let l = &self.0;
        let pooled = s.pooled();
        let p: &Frame = &pooled;
        p.require_symbols("pas_decode")?;
        // the complex noise splits evenly between the two dimensions
        let var = p.symbol_noise("pas_decode")?;
        let y: Vec<f64> = p.samples.iter().flat_map(|z| [z.re, z.im]).collect();
        let v: Vec<f64> = var.iter().flat_map(|v| [v / 2.0, v / 2.0]).collect();
        let n_c = l.pas.n_symbols();
        // the codewords that pas_encode put in the frame
        let blocks = p.coding.n_blocks.max(1);
        ensure(y.len() >= n_c * blocks, || "pas_decode: fewer symbols than the codewords of pas_encode".into())?;
        let mut decoded = Vec::with_capacity(l.pas.info_bits() * blocks);
        for b in 0..blocks {
            decoded.extend(l.pas.decode_with(&y[b * n_c..(b + 1) * n_c], &v[b * n_c..(b + 1) * n_c])?);
        }
        drop(pooled);
        s.coding.decoded = decoded;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::tests::run;
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

    #[test]
    fn the_parameters_of_the_codes_are_documented_and_checked() {
        use crate::blocks::registry;
        let enc = registry().into_iter().find(|b| b.name == "fec_encode").unwrap();
        let code = enc.params.iter().find(|p| p.name == "code").unwrap();
        let kinds: Vec<&str> = code.variants.iter().map(|(k, _)| *k).collect();
        assert_eq!(kinds, ["conv", "rs", "turbo", "ldpc", "polar"]);
        let text = enc.describe();
        assert!(text.contains("{type: ldpc}") && text.contains("var_degree") && text.contains("min_sum"), "{text}");
        // the JSON Schema lists every code with its defaults and accepted values
        let schema = code.json_schema();
        let ldpc =
            schema["anyOf"][0]["oneOf"].as_array().unwrap().iter().find(|c| c["title"] == json!("ldpc")).unwrap();
        assert_eq!(ldpc["required"], json!(["type", "n", "k"]));
        assert_eq!(ldpc["properties"]["iterations"]["anyOf"][0]["default"], json!(ldpc::DEFAULT_MAX_ITERATIONS));
        assert_eq!(ldpc["properties"]["rule"]["anyOf"][0]["enum"][1], json!("min_sum"));
        // every code accepts the parameters it documents, and refuses the others
        for (kind, params) in &code.variants {
            let mut m = serde_json::Map::new();
            m.insert("type".into(), json!(kind));
            for p in params.iter().filter(|p| matches!(p.default, crate::blocks::ParamDefault::Required)) {
                let v = match (*kind, p.name) {
                    ("rs", "m") => json!(4),
                    ("rs", "t") => json!(2),
                    ("polar", "n") => json!(64),
                    ("turbo", "k") => json!(40),
                    (_, "n") => json!(128),
                    _ => json!(32),
                };
                m.insert(p.name.into(), v);
            }
            assert!(build_code(&Json::Object(m.clone())).is_ok(), "{kind}");
            m.insert("zz".into(), json!(1));
            let e = build_code(&Json::Object(m)).err().unwrap().to_string();
            assert!(e.contains("unknown field `zz`") && e.contains(&format!("{kind} code")), "{e}");
        }
        let e = build_code(&json!({"type": "bch"})).err().unwrap().to_string();
        assert!(e.contains("unknown code type") && e.contains("polar"), "{e}");
        // the type is read without case, the bounds are declared
        assert!(build_code(&json!({"type": "LDPC", "n": 128, "k": 64})).is_ok());
        let e = build_code(&json!({"type": "rs", "m": 20, "t": 2})).err().unwrap().to_string();
        assert!(e.contains("m must be >= 2 and <= 16"), "{e}");
        let e = build_code(&json!({"type": "ldpc", "n": 128, "k": 64, "rule": "bp"})).err().unwrap().to_string();
        assert!(e.contains("rule must be one of"), "{e}");
    }

    #[test]
    fn every_code_decodes_a_clean_channel() {
        for code in [
            json!({"type": "conv", "k": 100}),
            json!({"type": "turbo", "k": 40}),
            json!({"type": "ldpc", "n": 128, "k": 64}),
            json!({"type": "polar", "n": 64, "k": 32, "crc": "crc6", "list": 4}),
        ] {
            let s = run(
                &[
                    json!({"type": "fec_encode", "code": code, "blocks": 3, "interleave": true}),
                    json!({"type": "map", "constellation": "16-QAM"}),
                    json!({"type": "awgn_ebn0", "ebn0_db": 12.0}),
                    json!({"type": "demap"}),
                    json!({"type": "fec_decode", "code": code, "interleave": true}),
                ],
                1,
            );
            assert_eq!(s.coding.decoded, s.coding.payload, "{code}");
            assert_eq!(s.coding.n_blocks, 3);
        }
    }

    #[test]
    fn coded_ebn0_counts_the_code_rate() {
        // rate 1/2 on QPSK: E_s/N_0 = E_b/N_0, so σ² = 10^(-Eb/N0/10)
        let s = run(
            &[
                json!({"type": "fec_encode", "code": {"type": "ldpc", "n": 256, "k": 128}, "blocks": 40}),
                json!({"type": "map"}),
                json!({"type": "awgn_ebn0", "ebn0_db": 3.0}),
            ],
            2,
        );
        assert!((s.channel.noise_var.unwrap() / 10f64.powf(-0.3) - 1.0).abs() < 0.05);
    }

    #[test]
    fn probes_measure_in_the_middle_of_the_link() {
        let s = run(
            &[
                json!({"type": "source", "constellation": "QPSK", "n_symbols": 20_000}),
                json!({"type": "probe", "label": "tx", "metrics": ["power_dbm", "snr_db"]}),
                json!({"type": "awgn", "snr_db": 10.0}),
                json!({"type": "probe", "label": "rx", "metrics": ["snr_db", "ser"], "scale": "none"}),
            ],
            3,
        );
        let get = |k: &str| s.probes.get(k).map(com_sim::stats::Estimate::value).unwrap();
        assert!((get("tx.power_dbm") - 30.0).abs() < 1e-9);
        assert!((get("rx.snr_db") - 10.0).abs() < 0.1);
        assert!(get("rx.ser") > 0.0);
    }

    #[test]
    fn pas_decodes_a_clean_channel_and_declares_its_distribution() {
        let p = json!({"constellation": "16-QAM", "mb_lambda": 1.0, "n": 192, "blocks": 3});
        let mut enc = p.clone();
        enc["type"] = json!("pas_encode");
        let mut dec = p;
        dec["type"] = json!("pas_decode");
        let s = run(&[enc, json!({"type": "awgn", "snr_db": 30.0}), dec], 4);
        assert_eq!(s.coding.decoded, s.coding.payload);
        assert_eq!(s.coding.n_blocks, 3);
        let pmf = s.pmf.clone().unwrap();
        assert!((pmf.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // the inner points are the most likely
        let c = s.constellation.clone().unwrap();
        let inner = c.points().iter().zip(&pmf).filter(|(z, _)| z.norm() < 0.5).map(|(_, p)| *p).fold(0.0, f64::max);
        let outer = c.points().iter().zip(&pmf).filter(|(z, _)| z.norm() > 1.2).map(|(_, p)| *p).fold(1.0, f64::min);
        assert!(inner > 2.0 * outer, "{inner} {outer}");
    }
}

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
//!   - {type: rayleigh, coherence_samples: 1}
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

use super::Extension;

mod codes;
mod fec;
mod map;
mod pas;

pub use codes::build_code;
use fec::{Decode, DecodeSpec, Encode, EncodeSpec};
use map::{Demap, Map, MapSpec};
use pas::{PasDecode, PasEncode, PasLink, PasSpec};

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
            |q, _, _| Ok(Box::new(Encode::new(&q)?)),
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
            |q, _, _| Ok(Box::new(Decode::new(&q)?)),
        ),
    ]
}

#[cfg(test)]
mod tests;

//! The blocks `pas_encode` and `pas_decode`: probabilistic amplitude shaping over a
//! square QAM.

use super::codes::LdpcSettings;
use crate::blocks::{Block, Coding, ConstellationParam, Frame, parse_constellation};
use com_core::rng::Rng;
use com_core::{Result, ensure};
use com_mapping::constellation::Constellation;
use com_math::C64;
use com_systems::pas::Pas;

// --------------------------------------------------------------------------- PAS

block_params! {
    pub(super) struct PasSpec {
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
pub(super) struct PasLink {
    c: Constellation,
    pas: Pas,
    blocks: usize,
    symbol_rate: f64,
    delta: f64,
    levels: usize,
}

impl PasLink {
    pub(super) fn new(p: &PasSpec) -> Result<PasLink> {
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

pub(super) struct PasEncode(pub(super) PasLink);

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

pub(super) struct PasDecode(pub(super) PasLink);

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

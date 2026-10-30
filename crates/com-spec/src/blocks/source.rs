//! Sources: simulated symbols (`source`) and measured recordings (`capture`, `reference_sync`).

use super::{Block, Extension, BuildContext, CaptureState, ConstellationParam, Frame, Layout, parse_constellation};
use com_core::Error;
use com_core::Result;
use com_core::ensure;
use com_core::rng::Rng;
use com_dsp::signal;
use com_dsp::sync::frame;
use com_mapping::constellation::Constellation;
use com_mapping::constellation::indices_to_bits;
use com_mapping::source as sources;
use com_math::C64;
use serde_json::Value as Json;

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            start "source",
            "Draw the bits and symbols of a block (Gray labels of OptiCommPy).",
            SourceSpec,
            needs [],
            gives [Bits, Symbols, Constellation, Samples],
            |q, _, _| Ok(Box::new(Source::new(&q)?)),
        ),
        builtin!(
            start "capture",
            "A measured recording as the source: one segment per trial, the reference sequence not yet aligned.",
            CaptureSpec,
            needs [],
            gives [Bits, Symbols, Constellation, Samples, Capture],
            |q, _, ctx| Ok(Box::new(Capture::new(&q, ctx)?)),
        ),
        builtin!(
            "reference_sync",
            "Align the periodic reference of a capture on the received symbols (circular correlation); records offset and quality.",
            ReferenceSyncSpec,
            needs [Samples, Constellation, Capture],
            gives [Bits, Symbols],
            |q, label, _| Ok(Box::new(ReferenceSync { p: q, label })),
        ),
    ]
}

// ------------------------------------------------------------------- transmitter

block_params! {
    struct SourceSpec {
        constellation: ConstellationParam ["16-QAM"],
            "`16-QAM`, `QPSK`, `8-PSK`, `4-PAM`, `OOK`, {kind, m}, or `gaussian` (circular Gaussian symbols, no decisions); unit energy";
        n_symbols: usize [4096] >= 1, "symbols per trial";
        symbol_rate: f64 [1.0] "Bd" > 0, "symbol rate; 1 keeps a normalised time axis";
        mb_lambda: Option<f64> [absent], "Maxwell-Boltzmann shaping p(x) ∝ exp(-λ|x|²)";
        pols: usize [1] >= 1 <= 2, "polarisations: 2 draws independent symbols for x and y (dual-polarisation optics)";
    }
}

struct Source {
    c: Option<Constellation>,
    n: usize,
    symbol_rate: f64,
    px: Option<Vec<f64>>,
    pols: usize,
}

impl Source {
    fn new(p: &SourceSpec) -> Result<Source> {
        let c = match p.constellation.json() {
            Json::String(s) if s.eq_ignore_ascii_case("gaussian") => None,
            other => Some(parse_constellation(other)?),
        };
        ensure(c.is_some() || p.mb_lambda.is_none(), || "source: mb_lambda needs a constellation".into())?;
        let px = match (&c, p.mb_lambda) {
            (Some(c), Some(l)) => Some(sources::maxwell_boltzmann(c, l)),
            _ => None,
        };
        Ok(Source { c, n: p.n_symbols, symbol_rate: p.symbol_rate, px, pols: p.pols })
    }
}

impl Block for Source {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let total = self.n * self.pols;
        let (idx, x, bits) = match &self.c {
            Some(c) => {
                let (idx, x) = sources::symbols(c, total, self.px.as_deref(), rng)?;
                let bits = indices_to_bits(&idx, c.bits_per_symbol());
                (idx, x, bits)
            }
            None => (Vec::new(), (0..total).map(|_| rng.complex_gaussian()).collect(), Vec::new()),
        };
        *s = Frame {
            trial: s.trial,
            bits,
            indices: idx,
            samples: x[..self.n].to_vec(),
            antennas: if self.pols == 2 { vec![x[self.n..].to_vec()] } else { Vec::new() },
            symbols: x,
            sps: 1.0,
            symbol_rate: self.symbol_rate,
            excluded: vec![false; total],
            constellation: self.c.clone(),
            pmf: self.px.clone(),
            layout: Layout { polarisations: self.pols == 2, ..Layout::default() },
            ..Frame::default()
        };
        Ok(())
    }
}

// ------------------------------------------------------------------ measured data

block_params! {
    struct CaptureSpec {
        file: String [required],
            "SigMF (.sigmf-meta/.sigmf-data or a .sigmf archive), NPY (.npy) or raw samples, relative to the specification";
        channel: usize [0], "the channel of a multi-channel SigMF recording";
        datatype: Option<String> [absent], "data type of a raw file: cf32_le, ci16_le, cf64_le, ci8, cu8, …";
        fs: Option<f64> [absent "from file"] "Hz" > 0,
            "sampling rate; required when the file does not record it";
        symbol_rate: f64 [required] "Bd" > 0, "symbol rate";
        constellation: ConstellationParam ["16-QAM"], "the transmitted constellation";
        reference: String [required],
            "one period of the transmitted symbols (.npy complex) or indices (.npy or text integers)";
        segment: Option<usize> [absent "whole file"], "samples per trial";
        offset: Offset ["random"], "first sample of a segment: random (drawn per trial) or an integer";
    }
}

json_param! {
    /// The first sample of a segment of `capture`: `random` or an integer.
    Offset,
    serde_json::json!({"anyOf": [{"type": "string", "enum": ["random"]}, {"type": "integer", "minimum": 0}]})
}

struct Capture {
    samples: std::sync::Arc<Vec<C64>>,
    fs: f64,
    symbol_rate: f64,
    c: Constellation,
    reference: Vec<usize>,
    segment: usize,
    offset: Option<usize>,
}

/// One period of transmitted indices from a file: complex symbols (mapped to the nearest
/// points after normalisation) or integer indices, in NPY or as text integers.
fn read_reference(path: &std::path::Path, c: &Constellation) -> Result<Vec<usize>> {
    use crate::io::{NpyData, read_npy};
    let idx: Vec<i64> = if path.extension().is_some_and(|e| e == "npy") {
        match read_npy(path)?.data {
            NpyData::Int(v) => v,
            NpyData::Complex(v) => return Ok(c.decide(&signal::pnorm(&v))),
            NpyData::Real(v) => {
                return Ok(c.decide(&signal::pnorm(&v.iter().map(|&x| C64::new(x, 0.0)).collect::<Vec<_>>())));
            }
        }
    } else {
        std::fs::read_to_string(path)?
            .split(|ch: char| ch.is_whitespace() || ch == ',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse::<i64>().map_err(|_| Error::Config(format!("{}: {t:?} is not an index", path.display()))))
            .collect::<Result<_>>()?
    };
    idx.iter()
        .map(|&i| {
            usize::try_from(i)
                .ok()
                .filter(|&i| i < c.order())
                .ok_or_else(|| Error::Config(format!("{}: index {i} outside the {} points", path.display(), c.order())))
        })
        .collect()
}

impl Capture {
    fn new(p: &CaptureSpec, ctx: &BuildContext) -> Result<Capture> {
        let dt = p.datatype.as_deref().map(crate::io::Datatype::parse).transpose()?;
        let (samples, fs_file) = crate::io::read_samples(ctx.path(&p.file), dt, p.channel)?;
        let fs =
            p.fs.or(fs_file)
                .ok_or_else(|| Error::Config(format!("capture {}: the file has no sampling rate, give fs", p.file)))?;
        ensure(p.symbol_rate > 0.0 && fs >= p.symbol_rate, || "capture: need fs ≥ symbol_rate > 0".into())?;
        let c = parse_constellation(p.constellation.json())?;
        let reference = read_reference(&ctx.path(&p.reference), &c)?;
        ensure(!reference.is_empty(), || "capture: the reference is empty".into())?;
        let segment = p.segment.unwrap_or(samples.len()).min(samples.len());
        ensure(segment > 0, || format!("capture {}: no samples", p.file))?;
        let offset = match &p.offset.0 {
            Json::String(s) if s == "random" => None,
            Json::Number(n) => {
                let o =
                    n.as_u64().ok_or_else(|| Error::Config("capture: offset must be random or an integer".into()))?;
                ensure(o as usize + segment <= samples.len(), || "capture: the segment ends after the file".into())?;
                Some(o as usize)
            }
            other => return Err(Error::Config(format!("capture: offset must be random or an integer, got {other}"))),
        };
        Ok(Capture {
            samples: std::sync::Arc::new(samples),
            fs,
            symbol_rate: p.symbol_rate,
            c,
            reference,
            segment,
            offset,
        })
    }
}

impl Block for Capture {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        let start = self.offset.unwrap_or_else(|| rng.below(self.samples.len() - self.segment + 1));
        let sps = self.fs / self.symbol_rate;
        let n_sym = (self.segment as f64 / sps).floor() as usize;
        *s = Frame {
            trial: s.trial,
            samples: self.samples[start..start + self.segment].to_vec(),
            sps,
            symbol_rate: self.symbol_rate,
            constellation: Some(self.c.clone()),
            capture: CaptureState { reference: self.reference.clone(), unaligned: true },
            ..Frame::default()
        };
        let p = self.reference.len();
        s.set_indices((0..n_sym).map(|k| self.reference[k % p]).collect());
        Ok(())
    }
}

block_params! {
    struct ReferenceSyncSpec {
        window: usize [4096], "symbols of the correlation (at most one period)";
        start: Option<usize> [absent "middle"], "first symbol of the window, or middle";
        symmetry: u32 [4], "rotational symmetry of the constellation; 0 removes the exact phase";
    }
}

struct ReferenceSync {
    p: ReferenceSyncSpec,
    label: String,
}

impl Block for ReferenceSync {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        ensure(!s.capture.reference.is_empty(), || "reference_sync needs a capture as the source".into())?;
        s.require_one_sps("reference_sync")?;
        let c = s.constellation("reference_sync")?.clone();
        let refs = c.symbols(&s.capture.reference);
        let n = s.samples.len();
        let l = self.p.window.min(refs.len()).min(n);
        let start = self.p.start.unwrap_or((n - l) / 2);
        let symmetry = if self.p.symmetry == 0 { None } else { Some(self.p.symmetry) };
        let r = frame::periodic_offset(&signal::pnorm(&s.samples), &refs, start, l, symmetry)?;
        let p = s.capture.reference.len();
        s.samples.iter_mut().for_each(|v| *v *= r.rotation);
        let idx = (0..n).map(|k| s.capture.reference[(k + r.offset) % p]).collect();
        s.set_indices(idx);
        s.capture.unaligned = false;
        s.record(&self.label, "offset", r.offset as f64);
        s.record(&self.label, "quality", r.quality);
        Ok(())
    }
}

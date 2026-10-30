//! Measurements in the middle of a link: probes, recordings of the signal (`dump`) and
//! the declared posterior of a classic receiver.

use super::{Block, BlockInfo, BuildContext, Frame, Params};
use crate::io::Datatype;
use crate::metrics::{Metric, Metrics, MetricsWindow};
use com_core::Error;
use com_core::Result;
use com_core::ensure;
use com_core::rng::Rng;
use serde_json::Value as Json;
use std::path::PathBuf;

/// The documentation of the blocks of this module.
pub(super) fn infos() -> Vec<BlockInfo> {
    vec![
        block_info!(
            "probe",
            "Measure metrics at this point of the link; reported as <label>.<metric> (label: probe by default).",
            ProbeSpec,
            needs [Samples],
            gives [],
        ),
        block_info!(
            "dump",
            "Save the samples at this point of the link for the first trials (by index) of every condition, as SigMF recordings <dir>/<campaign>/<condition>/<label>_<trial>.sigmf-meta and -data (one per stream, suffixed _s<stream> after the first).",
            DumpSpec,
            needs [Samples],
            gives [],
        ),
        block_info!(
            "noise_estimate",
            "Declare the receiver's posterior: the noise variance of the aligned samples; records sigma2.",
            NoiseEstimateSpec,
            needs [Samples, Symbols, Constellation],
            gives [Noise],
        ),
        block_info!(
            "classify",
            "Blind modulation classification at one sample per symbol, by fourth-order cumulants or by likelihood; records class (the index of the decided candidate), correct (1 if it is the transmitted alphabet) and snr_m2m4_db (M2M4 for the decided alphabet).",
            ClassifySpec,
            needs [Samples],
            gives [],
        ),
    ]
}

/// Build a block of this module, or `None` for another type.
pub(super) fn build(k: &str, label: &str, p: &Json, ctx: &BuildContext) -> Option<Result<Box<dyn Block>>> {
    let (p, label) = (p.clone(), label.to_string());
    let block = move || -> Result<Option<Box<dyn Block>>> {
        Ok(Some(match k {
            "probe" => Box::new(Probe::new(ProbeSpec::parse(k, p)?, label)?),
            "dump" => Box::new(Dump::new(&DumpSpec::parse(k, p)?, label, ctx)?),
            "noise_estimate" => Box::new(NoiseEstimate::new(&NoiseEstimateSpec::parse(k, p)?, label)?),
            "classify" => Box::new(Classify::new(&ClassifySpec::parse(k, p)?, label)?),
            _ => return Ok(None),
        }))
    };
    block().transpose()
}

// ------------------------------------------------------------------------- probes

block_params! {
    struct ProbeSpec {
        metrics: Vec<String> [["snr_db"] => "[snr_db]"],
            "the metrics of the specifications (on symbols aligned at 1 sample per symbol), and power_dbm and papr_db anywhere";
        window: MetricsWindow [flatten], "the window and the scaling, as for the metrics of the specification";
    }
}

/// The metrics that a probe measures on the samples alone, aligned or not.
const SIGNAL_METRICS: [&str; 2] = ["power_dbm", "papr_db"];

/// Metrics measured in the middle of a link (`probe`).
struct Probe {
    label: String,
    signal: Vec<&'static str>,
    symbols: Option<Metrics>,
}

impl Probe {
    fn new(p: ProbeSpec, label: String) -> Result<Probe> {
        let mut signal = Vec::new();
        let mut symbols = Vec::new();
        for m in &p.metrics {
            match SIGNAL_METRICS.iter().find(|s| s.eq_ignore_ascii_case(m)) {
                Some(s) => signal.push(*s),
                None => symbols.push(m.parse::<Metric>().map_err(|e| e.context("probe"))?),
            }
        }
        let symbols = (!symbols.is_empty()).then(|| Metrics::with_window(symbols, p.window));
        Ok(Probe { label, signal, symbols })
    }
}

impl Block for Probe {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let mut rec = com_sim::scenario::Record::new();
        for &m in &self.signal {
            if m == "power_dbm" {
                rec.value(m, com_math::w2dbm(s.mean_power()));
            } else {
                let papr: Vec<f64> = s.streams().map(|x| com_math::papr_db(x)).collect();
                rec.value(m, papr.iter().sum::<f64>() / papr.len() as f64);
            }
        }
        if let Some(metrics) = &self.symbols {
            metrics.record(s, &mut rec)?;
        }
        for (k, e) in rec.iter() {
            let name = format!("{}.{k}", self.label);
            match e {
                com_sim::stats::Estimate::Rate(r) => s.probes.errors(&name, r.errors, r.total),
                com_sim::stats::Estimate::Mean(m) => s.probes.value(&name, m.mean()),
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------- dumps

block_params! {
    struct DumpSpec {
        dir: String ["results/signals"], "directory of the recordings (relative to the working directory)";
        trials: usize [1] >= 1, "trials saved per condition: those of index 0 to trials - 1, whatever the order the threads run them in";
        datatype: String ["cf64"] one_of ["cf32", "cf64", "ci16"], "sample format of the SigMF data file";
    }
}

/// The samples at a point of the link, saved for the first trials (`dump`).
///
/// Every stream of the frame goes to its own recording, with the samples per symbol,
/// the symbol rate, the block and the trial in the global metadata (`comrs:sps`,
/// `comrs:symbol_rate`, `comrs:block`, `comrs:trial`, `comrs:stream`), and
/// `core:sample_rate` is the symbol rate times the samples per symbol. The trials are
/// chosen by their index ([`Frame::trial`], set by the runner), so trial `t` of a
/// condition is the frame that the stream `(seed, name, condition, t)` gives, whatever
/// the order the threads run the trials in.
struct Dump {
    base: PathBuf,
    label: String,
    trials: u64,
    datatype: Datatype,
}

impl Dump {
    fn new(p: &DumpSpec, label: String, ctx: &BuildContext) -> Result<Dump> {
        let mut base = PathBuf::from(&p.dir);
        if let Some(name) = &ctx.name {
            base.push(name);
        }
        if let Some(cond) = &ctx.condition {
            if !cond.is_empty() {
                base.push(cond.replace(", ", "_").replace(' ', ""));
            }
        }
        Ok(Dump { base, label, trials: p.trials as u64, datatype: Datatype::parse(&p.datatype)? })
    }
}

impl Block for Dump {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let trial = s.trial;
        if trial >= self.trials {
            return Ok(());
        }
        std::fs::create_dir_all(&self.base)?;
        for (i, x) in s.streams().enumerate() {
            let name = if i == 0 { format!("{}_{trial}", self.label) } else { format!("{}_{trial}_s{i}", self.label) };
            let global = serde_json::json!({
                "comrs:block": self.label,
                "comrs:trial": trial,
                "comrs:stream": i,
                "comrs:sps": s.sps,
                "comrs:symbol_rate": s.symbol_rate,
            });
            crate::io::write_sigmf(self.base.join(name), x, self.datatype, s.symbol_rate * s.sps, Some(&global))?;
        }
        Ok(())
    }
}

block_params! {
    struct NoiseEstimateSpec {
        mode: String ["decisions"] one_of ["decisions", "known"],
            "decisions (blind, around the nearest points) or known (around the transmitted symbols)";
    }
}

/// The posterior of a classic receiver: `z` as it is and the mean squared distance to
/// the decided (or transmitted) points.
struct NoiseEstimate {
    known: bool,
    label: String,
}

impl NoiseEstimate {
    fn new(p: &NoiseEstimateSpec, label: String) -> Result<NoiseEstimate> {
        let known = match p.mode.as_str() {
            "decisions" => false,
            "known" => true,
            other => {
                return Err(Error::Config(format!("noise_estimate: mode must be decisions or known, got {other:?}")));
            }
        };
        Ok(NoiseEstimate { known, label })
    }
}

impl Block for NoiseEstimate {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.require_symbols("noise_estimate")?;
        let c = s.constellation("noise_estimate")?;
        let mut e = 0.0;
        let mut n = 0usize;
        for i in (0..s.samples.len()).filter(|&i| !s.excluded[i]) {
            let z = s.samples[i];
            let x = if self.known { s.symbols[i] } else { c.points()[c.nearest(z)] };
            e += (z - x).norm_sqr();
            n += 1;
        }
        ensure(n > 0, || "noise_estimate: every symbol is excluded".into())?;
        let sigma2 = (e / n as f64).max(f64::MIN_POSITIVE);
        s.receiver.sigma2 = Some(sigma2);
        s.record(&self.label, "sigma2", sigma2);
        Ok(())
    }
}

// ---------------------------------------------------------------------- classify

block_params! {
    struct ClassifySpec {
        candidates: Vec<String> [["BPSK", "QPSK", "8-PSK", "16-QAM", "64-QAM"] => "[BPSK, QPSK, 8-PSK, 16-QAM, 64-QAM]"],
            "the candidate alphabets";
        method: String ["ml"] one_of ["cumulants", "ml"],
            "cumulants (Swami & Sadler 2000, normalised by the received power: for high SNR) or ml (likelihood, the scale and the noise estimated by M2M4 under every candidate)";
    }
}

struct Classify {
    candidates: Vec<com_mapping::constellation::Constellation>,
    ml: bool,
    label: String,
}

impl Classify {
    fn new(p: &ClassifySpec, label: String) -> Result<Classify> {
        ensure(!p.candidates.is_empty(), || "classify: give at least one candidate".into())?;
        let candidates = p
            .candidates
            .iter()
            .map(|n| com_mapping::constellation::Constellation::from_name(n).map(|c| c.normalized()))
            .collect::<Result<Vec<_>>>()
            .map_err(|e| e.context("classify"))?;
        Ok(Classify { candidates, ml: p.method == "ml", label })
    }
}

impl Block for Classify {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.require_one_sps("classify")?;
        let r: Vec<_> =
            (0..s.samples.len()).filter(|&i| s.excluded.get(i).is_none_or(|e| !e)).map(|i| s.samples[i]).collect();
        let d = if self.ml {
            com_metrics::blind::classify_ml_blind(&r, &self.candidates)
        } else {
            com_metrics::blind::classify_cumulants(&r, &self.candidates, None)
        }
        .map_err(|e| e.context("classify"))?;
        let decided = &self.candidates[d.index];
        let snr = com_metrics::blind::snr_m2m4(&r, decided).map_err(|e| e.context("classify"))?;
        s.record(&self.label, "class", d.index as f64);
        if let Some(c) = &s.constellation {
            let correct = c.kind() == decided.kind() && c.order() == decided.order();
            s.record(&self.label, "correct", f64::from(u8::from(correct)));
        }
        s.record(&self.label, "snr_m2m4_db", com_math::lin2db(snr));
        Ok(())
    }
}

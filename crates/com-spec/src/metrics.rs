//! The metrics of a declarative link.
//!
//! The metrics compare the samples at the end of the link with the transmitted
//! symbols. The samples must be at one sample per symbol and aligned on the symbols
//! (after decimation and, when the receiver introduces a delay, frame
//! synchronisation). The comparison uses the symbols `n` with
//! `skip_start ≤ n < N - skip_end` that no block excluded (pilots, preambles).
//!
//! ```yaml
//! metrics:
//!   report: [ber, ser, snr_db, gmi]   # default [ber, ser, snr_db]
//!   skip_start: 64                    # edges of the filters
//!   skip_end: 64
//!   scale: power                      # power (default), ls or none
//! ```
//!
//! Before the decisions the samples are scaled:
//!
//! - `power`: to the power of the transmitted symbols on the window (blind);
//! - `ls`: by the least-squares complex gain `Σ x z* / Σ |z|²`, which also removes a
//!   constant phase (genie-aided);
//! - `none`: as they are.
//!
//! | Metric | Kind | Definition |
//! |---|---|---|
//! | `ber` | rate | hard-decision bit errors (Gray labels) over the window |
//! | `ser` | rate | symbol errors |
//! | `snr_db` | mean | `10 log10(Σ|x|² / Σ|z - x|²)` per trial, after the scaling |
//! | `snr_eff_db` | mean | `10 log10(ρ²/(1 - ρ²))`, `ρ² = |Σ z x*|² / (Σ|z|² Σ|x|²)` per trial: the effective SNR after the least-squares gain of `x` onto `z` (Häger & Pfister; independent of the scaling) |
//! | `evm` | mean | `Σ|z - x|² / Σ|x|²` per trial (power ratio) |
//! | `mse` | mean | `mean |z - x|²` per trial |
//! | `gmi` | mean | generalised mutual information (bits/symbol) per trial |
//! | `ngmi` | mean | GMI per bit of information, `GMI / H(X)` |
//! | `entropy` | mean | entropy `H(X)` of the source (bits/symbol): `log2 M`, or less with shaping |
//! | `mi` | mean | symbol-wise mutual information (bits/symbol) per trial |
//!
//! | `coded_ber` | rate | information bits decoded in error (`fec_decode`) |
//! | `bler` | rate | coded blocks with at least one information bit in error |
//!
//! | `air` | mean | rate of a decoder that uses the receiver's own posterior (bits/symbol) |
//! | `air_bmd` | mean | the same with the bit-wise LLRs of that posterior |
//! | `air_opt`, `air_bmd_opt` | mean | the same with the best scaling `s` of the posterior, per block |
//! | `posterior_calibration` | mean | `σ̂² / mean |z - x|²`: 1 for a calibrated receiver, below 1 for an over-confident one |
//!
//! The `air` metrics need a block that declares the receiver's posterior (a learned
//! receiver, or `noise_estimate` after a classic receiver) and use the samples as they are, without
//! scaling (see [`metrics::posterior_rates`]). Compared with `mi` and `gmi` (genie gain
//! and variance), they show what the receiver loses by its own estimates.
//!
//! With a shaped source (`source` with `mb_lambda`, or `pas_encode`), the information
//! rates, the soft demapper and the posterior use the probabilities of the points, and
//! `awgn_ebn0` counts the energy per bit of information (the entropy `H(X)`).
//!
//! The information rates use the least-squares channel gain of
//! [`metrics::information_rates`] (consistent at every SNR, independent of the
//! scaling), not the power normalisation of OptiCommPy's estimators.
//!
//! With two polarisations (`source` with `pols: 2`) the metrics pool both: the samples
//! of x then those of y, against the symbols of x then those of y, the window
//! (`skip_start`, `skip_end`) applying to each polarisation.
//!
//! The coded metrics compare the decoded information bits with the transmitted ones
//! and need no aligned symbols: a coded link may report them alone.
//!
//! The rates are counted over all the trials (Wilson intervals, usable by the
//! `min_errors` stopping rule); the other metrics are means of per-trial values.

use crate::blocks::Frame;
use crate::params::parse;
use com_core::{Error, Result, ensure, ensure_shape};
use com_math::C64;
use com_metrics as metrics;
use com_sim::scenario::Record;
use serde::Deserialize;
use serde_json::Value as Json;

/// The metrics a link can report.
pub const METRICS: [&str; 17] = [
    "ber",
    "ser",
    "snr_db",
    "snr_eff_db",
    "evm",
    "mse",
    "gmi",
    "ngmi",
    "mi",
    "entropy",
    "air",
    "air_bmd",
    "air_opt",
    "air_bmd_opt",
    "posterior_calibration",
    "coded_ber",
    "bler",
];

/// The metrics computed on the decoded information bits.
const CODED: [&str; 2] = ["coded_ber", "bler"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetricsSpec {
    #[serde(default = "d_report")]
    report: Vec<String>,
    #[serde(default)]
    skip_start: usize,
    #[serde(default)]
    skip_end: usize,
    #[serde(default = "d_power")]
    scale: String,
}

fn d_report() -> Vec<String> {
    vec!["ber".into(), "ser".into(), "snr_db".into()]
}
fn d_power() -> String {
    "power".into()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scale {
    Power,
    Ls,
    None,
}

/// The metrics of a link (see the module documentation).
#[derive(Clone, Debug)]
pub struct Metrics {
    report: Vec<String>,
    skip_start: usize,
    skip_end: usize,
    scale: Scale,
}

impl Metrics {
    /// Parse the `metrics` mapping of a specification (`null` gives the defaults).
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown metric, key or scaling.
    pub fn from_json(v: &Json) -> Result<Metrics> {
        let v = if v.is_null() { Json::Object(serde_json::Map::default()) } else { v.clone() };
        let m: MetricsSpec = parse("metrics", v)?;
        for r in &m.report {
            ensure(METRICS.contains(&r.as_str()), || format!("unknown metric {r:?}; the metrics are {METRICS:?}"))?;
        }
        let scale = match m.scale.as_str() {
            "power" => Scale::Power,
            "ls" => Scale::Ls,
            "none" => Scale::None,
            other => return Err(Error::Config(format!("metrics scale must be power, ls or none, got {other:?}"))),
        };
        Ok(Metrics { report: m.report, skip_start: m.skip_start, skip_end: m.skip_end, scale })
    }

    /// The reported metric names.
    pub fn names(&self) -> &[String] {
        &self.report
    }

    /// Evaluate the metrics of one trial and record them.
    ///
    /// # Errors
    /// [`Error::Config`] for a coded metric without a coded link, or a signal that is
    /// not at one sample per symbol aligned on the symbols; [`Error::Shape`] if the
    /// window is empty.
    pub fn record(&self, s: &Frame, rec: &mut Record) -> Result<()> {
        if self.report.is_empty() {
            return Ok(());
        }
        // both polarisations of a dual-polarisation channel, each with its own window
        let segments = if s.layout.polarisations { s.n_streams() } else { 1 };
        let pooled = s.pooled();
        let s: &Frame = &pooled;
        for m in self.report.iter().filter(|m| CODED.contains(&m.as_str())) {
            ensure(!s.coding.payload.is_empty() && s.coding.decoded.len() == s.coding.payload.len(), || {
                format!("metric {m:?} needs a coded link: fec_encode first and fec_decode before the metrics")
            })?;
            if m == "coded_ber" {
                let e = metrics::bit_errors(&s.coding.payload, &s.coding.decoded);
                rec.errors(m, e as u64, s.coding.payload.len() as u64);
            } else {
                let k = s.coding.payload.len() / s.coding.n_blocks.max(1);
                let e = metrics::block_errors(&s.coding.payload, &s.coding.decoded, k)?;
                rec.errors(m, e as u64, s.coding.n_blocks.max(1) as u64);
            }
        }
        if self.report.iter().all(|m| CODED.contains(&m.as_str())) {
            return Ok(());
        }
        s.require_symbols("metrics")?;
        let n = s.n_symbols();
        let per = n / segments;
        let idx: Vec<usize> = (0..segments)
            .flat_map(|p| {
                let hi = per.saturating_sub(self.skip_end);
                (self.skip_start.min(hi)..hi).map(move |i| p * per + i)
            })
            .filter(|&i| !s.excluded.get(i).copied().unwrap_or(false))
            .collect();
        ensure_shape(!idx.is_empty(), || format!("the metrics window of the {n} symbols is empty"))?;
        let x: Vec<C64> = idx.iter().map(|&i| s.symbols[i]).collect();
        let mut z: Vec<C64> = idx.iter().map(|&i| s.samples[i]).collect();
        // independent of the scaling: on the samples as they are
        let snr_eff_db = metrics::effective_snr_db(&z, &x);
        let g = match self.scale {
            Scale::None => C64::new(1.0, 0.0),
            Scale::Power => {
                let pz: f64 = z.iter().map(C64::norm_sqr).sum();
                let px: f64 = x.iter().map(C64::norm_sqr).sum();
                C64::new(if pz > 0.0 { (px / pz).sqrt() } else { 1.0 }, 0.0)
            }
            Scale::Ls => metrics::ls_gain(&z, &x),
        };
        for v in &mut z {
            *v *= g;
        }
        let err2: f64 = z.iter().zip(&x).map(|(a, b)| (a - b).norm_sqr()).sum();
        let sig2: f64 = x.iter().map(C64::norm_sqr).sum();
        let mut info: Option<metrics::InformationRates> = None;
        let mut post: Option<metrics::PosteriorRates> = None;
        for m in &self.report {
            match m.as_str() {
                "ber" => {
                    let c = s.constellation("metrics: ber")?;
                    let b = c.bits_per_symbol();
                    let tx: Vec<u8> = idx.iter().flat_map(|&i| s.bits[i * b..(i + 1) * b].iter().copied()).collect();
                    rec.errors("ber", metrics::bit_errors(&c.demodulate(&z), &tx) as u64, tx.len() as u64);
                }
                "ser" => {
                    let c = s.constellation("metrics: ser")?;
                    let tx: Vec<usize> = idx.iter().map(|&i| s.indices[i]).collect();
                    rec.errors("ser", metrics::symbol_errors(&c.decide(&z), &tx) as u64, tx.len() as u64);
                }
                "snr_db" => rec.value("snr_db", 10.0 * (sig2 / err2.max(f64::MIN_POSITIVE)).log10()),
                "snr_eff_db" => rec.value("snr_eff_db", snr_eff_db),
                "evm" => rec.value("evm", err2 / sig2),
                "mse" => rec.value("mse", err2 / z.len() as f64),
                "gmi" | "ngmi" | "mi" | "entropy" => {
                    let r = if let Some(v) = info {
                        v
                    } else {
                        let raw: Vec<C64> = idx.iter().map(|&i| s.samples[i]).collect();
                        let tx: Vec<usize> = idx.iter().map(|&i| s.indices[i]).collect();
                        let c = s.constellation("metrics: information rates")?;
                        let v = metrics::information_rates(&raw, &tx, c, s.pmf.as_deref())?;
                        info = Some(v);
                        v
                    };
                    rec.value(
                        m,
                        match m.as_str() {
                            "gmi" => r.gmi,
                            "ngmi" => r.ngmi,
                            "entropy" => r.entropy,
                            _ => r.mi,
                        },
                    );
                }
                m if m.starts_with("air") || m == "posterior_calibration" => {
                    let r = if let Some(v) = post {
                        v
                    } else {
                        let s2 = s.receiver.sigma2.ok_or_else(|| {
                            Error::Config(format!(
                                "metric {m:?} needs a block that declares the receiver's posterior \
                                 (a learned receiver, or noise_estimate after a classic receiver)"
                            ))
                        })?;
                        let raw: Vec<C64> = idx.iter().map(|&i| s.samples[i]).collect();
                        let tx: Vec<usize> = idx.iter().map(|&i| s.indices[i]).collect();
                        let v = metrics::posterior_rates(
                            &raw,
                            &tx,
                            s.constellation("metrics: air")?,
                            s2,
                            s.pmf.as_deref(),
                        )?;
                        post = Some(v);
                        v
                    };
                    rec.value(
                        m,
                        match m {
                            "air" => r.air,
                            "air_bmd" => r.air_bmd,
                            "air_opt" => r.air_opt,
                            "air_bmd_opt" => r.air_bmd_opt,
                            _ => r.calibration,
                        },
                    );
                }
                m if CODED.contains(&m) => {}
                _ => unreachable!("validated in from_json"),
            }
        }
        Ok(())
    }
}

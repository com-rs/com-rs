//! The metrics of a declarative link.
//!
//! The metrics compare the samples at the end of the link with the transmitted
//! symbols. The samples must be at one sample per symbol and aligned on the symbols
//! (after decimation and, when the receiver introduces a delay, frame
//! synchronisation). The comparison uses the symbols $n$ with
//! $s_{\mathrm{start}} \le n < N - s_{\mathrm{end}}$ (`skip_start`, `skip_end`) that no
//! block excluded (pilots, preambles).
//!
//! ```yaml
//! metrics:
//!   report: [ber, ser, snr_db, gmi]   # default [ber, ser, snr_db]
//!   skip_start: 64                    # edges of the filters
//!   skip_end: 64
//!   scale: power                      # power (default), ls or none
//! ```
//!
//! The names of the metrics and of the scalings are read without case. Before the
//! decisions the samples $z_{n}$ are scaled:
//!
//! - `power`: to the power of the transmitted symbols $x_{n}$ on the window (blind);
//! - `ls`: by the least-squares complex gain
//!   $\sum_{n} x_{n} z_{n}^{*} / \sum_{n} \lvert z_{n} \rvert^{2}$, which also removes a
//!   constant phase (genie-aided);
//! - `none`: as they are.
//!
//! | Metric | Kind | Definition |
//! |---|---|---|
//! | `ber` | rate | hard-decision bit errors (Gray labels) over the window |
//! | `ser` | rate | symbol errors |
//! | `snr_db` | mean | $10 \log_{10} \left( \sum \lvert x \rvert^{2} / \sum \lvert z - x \rvert^{2} \right)$ per trial, after the scaling ([`metrics::snr_db`]; a trial without error gives $+\infty$, which the record drops and counts) |
//! | `snr_eff_db` | mean | $10 \log_{10} \left( \rho^{2} / (1 - \rho^{2}) \right)$, $\rho^{2} = \lvert \sum z x^{*} \rvert^{2} / \left( \sum \lvert z \rvert^{2} \sum \lvert x \rvert^{2} \right)$ per trial: the effective SNR after the least-squares gain of $x$ onto $z$ (Häger & Pfister; independent of the scaling) |
//! | `evm` | mean | $\sum \lvert z - x \rvert^{2} / \sum \lvert x \rvert^{2}$ per trial (power ratio, [`metrics::evm_data_aided`]) |
//! | `mse` | mean | the mean of $\lvert z - x \rvert^{2}$ per trial ([`metrics::mse`]) |
//! | `gmi` | mean | generalised mutual information (bits/symbol) per trial |
//! | `ngmi` | mean | GMI per bit of information, $\mathrm{GMI} / H(X)$ |
//! | `entropy` | mean | entropy $H(X)$ of the source (bits/symbol): $\log_{2} M$, or less with shaping |
//! | `mi` | mean | symbol-wise mutual information (bits/symbol) per trial |
//! | `coded_ber` | rate | information bits decoded in error (`fec_decode`) |
//! | `bler` | rate | coded blocks with at least one information bit in error |
//! | `air` | mean | rate of a decoder that uses the receiver's own posterior (bits/symbol) |
//! | `air_bmd` | mean | the same with the bit-wise LLRs of that posterior |
//! | `air_opt`, `air_bmd_opt` | mean | the same with the best scaling $s$ of the posterior, per block |
//! | `posterior_calibration` | mean | $\hat{\sigma}^{2}$ over the mean of $\lvert z - x \rvert^{2}$: 1 for a calibrated receiver, below 1 for an over-confident one |
//!
//! The `air` metrics need a block that declares the receiver's posterior (a learned
//! receiver, or `noise_estimate` after a classic receiver) and use the samples as they are, without
//! scaling (see [`metrics::posterior_rates`]). Compared with `mi` and `gmi` (genie gain
//! and variance), they show what the receiver loses by its own estimates.
//!
//! With a shaped source (`source` with `mb_lambda`, or `pas_encode`), the information
//! rates, the soft demapper and the posterior use the probabilities of the points, and
//! `awgn_ebn0` counts the energy per bit of information (the entropy $H(X)$).
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

use crate::blocks::schema::ParamType;
use crate::blocks::{Frame, Params};
use com_core::{Error, Result, ensure_shape};
use com_math::{C64, energy};
use com_metrics as metrics;
use com_sim::scenario::Record;
use serde::Deserialize;
use serde_json::Value as Json;
use std::str::FromStr;

/// The metrics, declared once: the variant, its name, whether it is computed on the
/// decoded information bits.
macro_rules! metrics {
    ($($variant:ident = $name:literal $(, $coded:ident)?;)*) => {
        /// A metric of a declarative link (see the module documentation).
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Metric {
            $(
                #[doc = concat!("`", $name, "`.")]
                $variant,
            )*
        }

        /// The names of the metrics a link can report, in the order of [`Metric::ALL`].
        pub const METRICS: [&str; [$($name),*].len()] = [$($name),*];

        impl Metric {
            /// Every metric.
            pub const ALL: [Metric; METRICS.len()] = [$(Metric::$variant),*];

            /// The name of the metric in a specification and in the results.
            pub fn name(self) -> &'static str {
                match self {
                    $(Metric::$variant => $name,)*
                }
            }

            /// Whether the metric is computed on the decoded information bits.
            pub fn is_coded(self) -> bool {
                match self {
                    $(Metric::$variant => metrics!(@coded $($coded)?),)*
                }
            }
        }
    };
    (@coded coded) => { true };
    (@coded) => { false };
}

metrics! {
    Ber = "ber";
    Ser = "ser";
    SnrDb = "snr_db";
    SnrEffDb = "snr_eff_db";
    Evm = "evm";
    Mse = "mse";
    Gmi = "gmi";
    Ngmi = "ngmi";
    Mi = "mi";
    Entropy = "entropy";
    Air = "air";
    AirBmd = "air_bmd";
    AirOpt = "air_opt";
    AirBmdOpt = "air_bmd_opt";
    PosteriorCalibration = "posterior_calibration";
    CodedBer = "coded_ber", coded;
    Bler = "bler", coded;
}

impl FromStr for Metric {
    type Err = Error;

    /// The metric of a name, without case.
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown name (the message lists the metrics).
    ///
    /// # Examples
    /// ```
    /// use com_spec::metrics::Metric;
    /// assert_eq!("BER".parse::<Metric>()?, Metric::Ber);
    /// assert!("fer".parse::<Metric>().is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    fn from_str(s: &str) -> Result<Metric> {
        Metric::ALL
            .into_iter()
            .find(|m| m.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| crate::unknown("metric", s, &METRICS))
    }
}

/// The scaling of the samples before the decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Scale {
    /// To the power of the transmitted symbols.
    Power,
    /// By the least-squares complex gain.
    Ls,
    /// None.
    None,
}

impl ParamType for Scale {
    fn schema() -> Json {
        String::schema()
    }
}

crate::block_params! {
    /// The window and the scaling of the metrics, shared by the `metrics` of a
    /// specification and the `probe` blocks.
    #[derive(Clone, Debug)]
    pub(crate) struct MetricsWindow {
        skip_start: usize [0] "symbols", "symbols left out at the start of every polarisation (the edges of the filters)";
        skip_end: usize [0] "symbols", "symbols left out at the end of every polarisation";
        scale: Scale ["power"] one_of ["power", "ls", "none"],
            "scaling before the decisions: power (blind), ls (least-squares complex gain, genie-aided) or none";
    }
}

crate::block_params! {
    /// The `metrics` mapping of a specification.
    struct MetricsSpec {
        report: Vec<String> [["ber", "ser", "snr_db"] => "[ber, ser, snr_db]"] one_of (METRICS), "the metrics reported";
        window: MetricsWindow [flatten], "the window and the scaling";
    }
}

/// The metrics of a link (see the module documentation).
#[derive(Clone, Debug)]
pub struct Metrics {
    report: Vec<Metric>,
    window: MetricsWindow,
}

/// The values a trial computes once for several metrics.
#[derive(Default)]
struct Shared {
    info: Option<metrics::InformationRates>,
    posterior: Option<metrics::PosteriorRates>,
}

impl Metrics {
    /// Parse the `metrics` mapping of a specification (`null` gives the defaults).
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown metric, key or scaling.
    pub fn from_json(v: &Json) -> Result<Metrics> {
        let q = MetricsSpec::parse("metrics", v.clone())?;
        let report = q.report.iter().map(|m| m.parse()).collect::<Result<_>>()?;
        Ok(Metrics { report, window: q.window })
    }

    /// The metrics `report` on the window and with the scaling `window` (a probe).
    pub(crate) fn with_window(report: Vec<Metric>, window: MetricsWindow) -> Metrics {
        Metrics { report, window }
    }

    /// The reported metrics.
    pub fn names(&self) -> Vec<&'static str> {
        self.report.iter().map(|m| m.name()).collect()
    }

    /// Record what a trial of a link records after its blocks ([`crate::Link::trial`]):
    /// the metrics of the frame ([`Metrics::record`]), then the values and the probes
    /// that its blocks recorded. The Python chains record their frames with it.
    ///
    /// # Errors
    /// As [`Metrics::record`].
    pub fn record_frame(&self, s: &Frame, rec: &mut Record) -> Result<()> {
        self.record(s, rec)?;
        for (k, v) in &s.values {
            rec.value(k, *v);
        }
        rec.merge(&s.probes);
        Ok(())
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
        self.record_coded(s, rec)?;
        if self.report.iter().all(|m| m.is_coded()) {
            return Ok(());
        }
        s.require_symbols("metrics")?;
        let idx = self.window(s, segments)?;
        let x: Vec<C64> = idx.iter().map(|&i| s.symbols[i]).collect();
        let raw: Vec<C64> = idx.iter().map(|&i| s.samples[i]).collect();
        let tx: Vec<usize> = idx.iter().map(|&i| s.indices[i]).collect();
        let g = self.gain(&raw, &x)?;
        let z: Vec<C64> = raw.iter().map(|v| v * g).collect();
        let mut shared = Shared::default();
        for &m in &self.report {
            let name = m.name();
            match m {
                Metric::Ber => {
                    let c = s.constellation("metrics: ber")?;
                    let b = c.bits_per_symbol();
                    let bits: Vec<u8> = idx.iter().flat_map(|&i| s.bits[i * b..(i + 1) * b].iter().copied()).collect();
                    rec.errors(name, metrics::bit_errors(&c.demodulate(&z), &bits)? as u64, bits.len() as u64);
                }
                Metric::Ser => {
                    let c = s.constellation("metrics: ser")?;
                    rec.errors(name, metrics::symbol_errors(&c.decide(&z), &tx)? as u64, tx.len() as u64);
                }
                Metric::SnrDb => rec.value(name, metrics::snr_db(&z, &x)?),
                // independent of the scaling: on the samples as they are
                Metric::SnrEffDb => rec.value(name, metrics::effective_snr_db(&raw, &x)?),
                Metric::Evm => rec.value(name, metrics::evm_data_aided(&z, &x)?),
                Metric::Mse => rec.value(name, metrics::mse(&z, &x)?),
                Metric::Gmi | Metric::Ngmi | Metric::Mi | Metric::Entropy => {
                    let r = match shared.info {
                        Some(r) => r,
                        None => *shared.info.insert(metrics::information_rates(
                            &raw,
                            &tx,
                            s.constellation("metrics: information rates")?,
                            s.pmf.as_deref(),
                        )?),
                    };
                    let v = match m {
                        Metric::Gmi => r.gmi,
                        Metric::Ngmi => r.ngmi,
                        Metric::Entropy => r.entropy,
                        _ => r.mi,
                    };
                    rec.value(name, v);
                }
                Metric::Air | Metric::AirBmd | Metric::AirOpt | Metric::AirBmdOpt | Metric::PosteriorCalibration => {
                    let r = match shared.posterior {
                        Some(r) => r,
                        None => *shared.posterior.insert(Self::posterior(s, name, &raw, &tx)?),
                    };
                    let v = match m {
                        Metric::Air => r.air,
                        Metric::AirBmd => r.air_bmd,
                        Metric::AirOpt => r.air_opt,
                        Metric::AirBmdOpt => r.air_bmd_opt,
                        _ => r.calibration,
                    };
                    rec.value(name, v);
                }
                // recorded by record_coded
                Metric::CodedBer | Metric::Bler => {}
            }
        }
        Ok(())
    }

    /// The coded metrics, on the decoded information bits.
    fn record_coded(&self, s: &Frame, rec: &mut Record) -> Result<()> {
        for m in self.report.iter().filter(|m| m.is_coded()) {
            let c = &s.coding;
            if c.payload.is_empty() || c.decoded.len() != c.payload.len() {
                return Err(Error::Config(format!(
                    "metric {:?} needs a coded link: fec_encode first and fec_decode before the metrics",
                    m.name()
                )));
            }
            if *m == Metric::CodedBer {
                let e = metrics::bit_errors(&c.payload, &c.decoded)?;
                rec.errors(m.name(), e as u64, c.payload.len() as u64);
            } else {
                let k = c.payload.len() / c.n_blocks.max(1);
                let e = metrics::block_errors(&c.decoded, &c.payload, k)?;
                rec.errors(m.name(), e as u64, c.n_blocks.max(1) as u64);
            }
        }
        Ok(())
    }

    /// The indices of the symbols of the window: in every one of the `segments`
    /// polarisations, `skip_start` to `skip_end` from its end, the excluded symbols left
    /// out.
    fn window(&self, s: &Frame, segments: usize) -> Result<Vec<usize>> {
        let n = s.n_symbols();
        let per = n / segments;
        let (start, end) = (self.window.skip_start, self.window.skip_end);
        let idx: Vec<usize> = (0..segments)
            .flat_map(|p| {
                let hi = per.saturating_sub(end);
                (start.min(hi)..hi).map(move |i| p * per + i)
            })
            .filter(|&i| !s.excluded.get(i).copied().unwrap_or(false))
            .collect();
        ensure_shape(!idx.is_empty(), || format!("the metrics window of the {n} symbols is empty"))?;
        Ok(idx)
    }

    /// The gain of the scaling of the samples `z` against the symbols `x`.
    fn gain(&self, z: &[C64], x: &[C64]) -> Result<C64> {
        Ok(match self.window.scale {
            Scale::None => C64::new(1.0, 0.0),
            Scale::Power => {
                let (pz, px) = (energy(z), energy(x));
                C64::new(if pz > 0.0 { (px / pz).sqrt() } else { 1.0 }, 0.0)
            }
            Scale::Ls => metrics::ls_gain(z, x)?,
        })
    }

    /// The rates of the receiver's declared posterior, on the samples as they are.
    fn posterior(s: &Frame, name: &str, raw: &[C64], tx: &[usize]) -> Result<metrics::PosteriorRates> {
        let s2 = s.receiver.sigma2.ok_or_else(|| {
            Error::Config(format!(
                "metric {name:?} needs a block that declares the receiver's posterior \
                 (a learned receiver, or noise_estimate after a classic receiver)"
            ))
        })?;
        metrics::posterior_rates(raw, tx, s.constellation("metrics: air")?, s2, s.pmf.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_are_read_without_case() {
        let m = Metrics::from_json(&serde_json::json!({"report": ["BER", "Snr_dB"], "scale": "LS"})).unwrap();
        assert_eq!(m.names(), ["ber", "snr_db"]);
        assert_eq!(m.window.scale, Scale::Ls);
        let e = Metrics::from_json(&serde_json::json!({"report": ["fer"]})).unwrap_err().to_string();
        assert!(e.contains("report must be one of") && e.contains("coded_ber"), "{e}");
        assert_eq!(Metrics::from_json(&Json::Null).unwrap().names(), ["ber", "ser", "snr_db"]);
        assert_eq!(Metric::ALL.map(Metric::name), METRICS);
    }
}

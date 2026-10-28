//! What the builders need besides the parameters of a block: [`BuildContext`], with
//! the fibre and Raman parameters of the specification.

use crate::params::parse;
use com_channel::optical::fiber::FiberParams;
use com_channel::optical::raman::{Direction, Raman, RamanGain, RamanPump};
use com_core::{Error, Result};
use serde::Deserialize;
use serde_json::{Map, Value as Json};
use std::path::PathBuf;

/// The meaning of the `fiber` parameter of the optical blocks.
pub(super) const FIBER_DOC: &str = "fibre parameters, merged over the `fiber` of the specification (see FiberParams); its `raman` mapping adds Raman pumps and ISRS";

/// What the builders of the blocks need besides their parameters.
#[derive(Clone, Debug, Default)]
pub struct BuildContext {
    /// The `fiber` mapping of the specification (defaults of every fibre block).
    pub fiber: Option<Json>,
    /// The directory of the specification (relative paths of files).
    pub base_dir: Option<PathBuf>,
    /// The name of the campaign (the files a block writes).
    pub name: Option<String>,
    /// The label of the condition the link is built for (the files a block writes).
    pub condition: Option<String>,
}

impl BuildContext {
    /// The fibre parameters of a block: the defaults of [`FiberParams`], then the
    /// specification's `fiber`, then the block's `fiber`.
    ///
    /// # Errors
    /// [`Error::Config`] for parameters that are not a valid fibre mapping.
    pub fn fiber(&self, block: Option<&Json>) -> Result<FiberParams> {
        let f: FiberSpec = parse("fiber", Json::Object(self.fiber_map(block)?))?;
        Ok(FiberParams {
            alpha_db_km: f.alpha_db_km,
            d_ps_nm_km: f.d_ps_nm_km,
            gamma_w_km: f.gamma_w_km,
            span_km: f.span_km,
            n_spans: f.n_spans,
            fc_hz: f.fc_hz,
            nf_db: f.nf_db,
        })
    }

    /// The Raman amplification of a block: the `raman` mapping of its merged `fiber`
    /// (see the optical blocks), `None` without one.
    ///
    /// # Errors
    /// [`Error::Config`] for a `raman` mapping that is not valid.
    pub fn raman(&self, block: Option<&Json>) -> Result<Option<Raman>> {
        match self.fiber_map(block)?.remove("raman") {
            None | Some(Json::Null) => Ok(None),
            Some(v) => raman(v).map(Some),
        }
    }

    fn fiber_map(&self, block: Option<&Json>) -> Result<Map<String, Json>> {
        let mut m = Map::new();
        for src in [self.fiber.as_ref(), block].into_iter().flatten() {
            match src {
                Json::Object(o) => m.extend(o.clone()),
                Json::Null => {}
                other => return Err(Error::Config(format!("fiber must be a mapping, got {other}"))),
            }
        }
        Ok(m)
    }

    /// A path relative to the specification.
    pub fn path(&self, p: &str) -> PathBuf {
        let p = PathBuf::from(p);
        match &self.base_dir {
            Some(d) if p.is_relative() => d.join(p),
            _ => p,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, default)]
struct FiberSpec {
    alpha_db_km: f64,
    d_ps_nm_km: f64,
    gamma_w_km: f64,
    span_km: f64,
    n_spans: usize,
    fc_hz: f64,
    nf_db: f64,
    /// Read by [`BuildContext::raman`].
    #[allow(dead_code, reason = "parsed by BuildContext::raman")]
    raman: Option<Json>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RamanSpec {
    #[serde(default)]
    pumps: Vec<PumpSpec>,
    #[serde(default)]
    aeff_um2: Option<f64>,
    #[serde(default)]
    temperature_k: Option<f64>,
    #[serde(default)]
    isrs: Option<bool>,
    #[serde(default)]
    step_km: Option<f64>,
    #[serde(default)]
    slice_hz: Option<f64>,
    #[serde(default)]
    gain: Option<GainSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PumpSpec {
    #[serde(default)]
    wavelength_nm: Option<f64>,
    #[serde(default)]
    frequency_thz: Option<f64>,
    power_mw: f64,
    #[serde(default = "d_counter")]
    direction: String,
    #[serde(default)]
    alpha_db_km: Option<f64>,
}

fn d_counter() -> String {
    "counter".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GainSpec {
    offsets_thz: Vec<f64>,
    gain_m_w: Vec<f64>,
    reference_thz: f64,
}

/// The Raman amplification of a `raman` mapping.
fn raman(v: Json) -> Result<Raman> {
    let q: RamanSpec = parse("raman", v)?;
    let d = Raman::default();
    let pumps = q
        .pumps
        .iter()
        .map(|p| {
            let direction = match p.direction.as_str() {
                "co" | "forward" => Direction::Co,
                "counter" | "backward" => Direction::Counter,
                other => return Err(Error::Config(format!("raman: direction must be co or counter, got {other:?}"))),
            };
            let frequency_hz = match (p.wavelength_nm, p.frequency_thz) {
                (Some(w), None) => com_math::SPEED_OF_LIGHT / (w * 1e-9),
                (None, Some(f)) => f * 1e12,
                _ => return Err(Error::Config("raman: a pump needs wavelength_nm or frequency_thz".into())),
            };
            Ok(RamanPump { frequency_hz, power_w: p.power_mw * 1e-3, direction, alpha_db_km: p.alpha_db_km })
        })
        .collect::<Result<_>>()?;
    let gain = match q.gain {
        None => d.gain.clone(),
        Some(g) => {
            RamanGain::new(g.offsets_thz.iter().map(|f| f * 1e12).collect(), g.gain_m_w, g.reference_thz * 1e12)?
        }
    };
    Ok(Raman {
        pumps,
        gain,
        aeff_um2: q.aeff_um2.unwrap_or(d.aeff_um2),
        temperature_k: q.temperature_k.unwrap_or(d.temperature_k),
        isrs: q.isrs.unwrap_or(d.isrs),
        step_km: q.step_km.unwrap_or(d.step_km),
        slice_hz: q.slice_hz.unwrap_or(d.slice_hz),
        ..d
    })
}

impl Default for FiberSpec {
    fn default() -> Self {
        let f = FiberParams::default();
        FiberSpec {
            alpha_db_km: f.alpha_db_km,
            d_ps_nm_km: f.d_ps_nm_km,
            gamma_w_km: f.gamma_w_km,
            span_km: f.span_km,
            n_spans: f.n_spans,
            fc_hz: f.fc_hz,
            nf_db: f.nf_db,
            raman: None,
        }
    }
}

//! The blocks `map` and `demap`: the coded bits on a constellation and their LLRs.

use crate::blocks::{Block, ConstellationParam, Frame, parse_constellation};
use com_core::rng::Rng;
use com_core::signal::Symbols;
use com_core::traits::{Demap as _, Demapping};
use com_core::{Result, ensure};
use com_mapping::constellation::{Constellation, log_priors};
use com_math::Jacobian;

// --------------------------------------------------------------------- map, demap

block_params! {
    pub(super) struct MapSpec {
        constellation: ConstellationParam ["QPSK"], "the constellation, as for source";
        symbol_rate: f64 [1.0] "Bd" > 0, "symbol rate";
    }
}

pub(super) struct Map {
    c: Constellation,
    symbol_rate: f64,
}

impl Map {
    pub(super) fn new(p: &MapSpec) -> Result<Map> {
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
    pub(super) struct Demap {
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
        let jacobian = Jacobian::from_max_log(self.max_log);
        let llr = if var.windows(2).all(|w| w[0] == w[1]) {
            // one noise variance: the soft demapper of the trait, with the log-priors of
            // the probabilities of the source
            let lp = p.pmf.as_deref().map(|px| c.priors(Some(px)).map(|q| log_priors(&q))).transpose()?;
            let y = Symbols::from_stream(p.samples.clone())?;
            let demapping = Demapping { log_priors: lp.as_deref(), jacobian };
            c.demap_with(&y, var.first().copied().unwrap_or(1.0), demapping)?.as_slice().to_vec()
        } else {
            // one variance per symbol (a linear MIMO detector): the slice method
            c.llr_with_variances(&p.samples, &var, p.pmf.as_deref(), jacobian)?
        };
        drop(pooled);
        s.coding.llr = llr;
        Ok(())
    }
}

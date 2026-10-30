//! Sources, modulation and metrics.

mod closed_forms;
mod constellation;
mod generators;
mod measures;
mod shaping;

use crate::rng::PyRng;
use crate::{BArr, CArr, FArr, IArr, IntoPy, bvec, carr, count, cvec, farr, fvec, iarr, kind, map_real, uvec};
use comrs::fec::ccdm::Ccdm;
use comrs::mapping::constellation::Constellation;
use comrs::mapping::theory::Fading;
use comrs::mapping::{bicm, constellation as modulation, source as sources, theory};
use comrs::metrics;
pub(crate) use constellation::PyConstellation;
use numpy::IntoPyArray;
use pyo3::prelude::*;

fn fading(name: &str) -> PyResult<Fading> {
    name.parse().py()
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    constellation::register(m)?;
    generators::register(m)?;
    shaping::register(m)?;
    measures::register(m)?;
    closed_forms::register(m)?;
    Ok(())
}

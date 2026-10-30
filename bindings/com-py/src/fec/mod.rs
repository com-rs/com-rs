//! Channel coding: CRC, scrambling, interleaving, convolutional, turbo, LDPC and polar
//! codes, and the finite-length limits (the Reed–Solomon and BCH codes and the
//! puncturing patterns are in [`crate::fec_codes`]).

mod conv;
mod crc;
mod ldpc_code;
mod limits;
mod nr;
mod polar_code;

use crate::fec_codes::{pattern_rows, puncturer};
use crate::{BArr, FArr, IArr, IntoPy, bvec, count, extract_error, farr, fvec, iarr, map_real, py_bool};
use comrs::fec::block::BlockCode;
use comrs::fec::bounds;
use comrs::fec::conv::{ConvolutionalCode, Trellis};
use comrs::fec::crc::Crc;
use comrs::fec::interleaving::Interleaver;
use comrs::fec::ldpc::{self, CheckRule, LdpcCode};
use comrs::fec::polar::PolarCode;
use comrs::fec::puncture::{Punctured, PuncturedTurbo, Puncturer};
use comrs::fec::scrambling::Scrambler;
use comrs::fec::turbo::TurboCode;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Encode whole blocks of bits with a code, the GIL released.
fn encode<'py, C>(py: Python<'py>, code: &C, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>>
where
    C: BlockCode + Sync + ?Sized,
{
    let b = bvec(bits)?;
    Ok(py.detach(|| code.encode_stream(&b)).py()?.into_pyarray(py))
}

/// Decode whole blocks of LLRs with a code, the GIL released.
fn decode<'py, C>(py: Python<'py>, code: &C, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>>
where
    C: BlockCode + Sync + ?Sized,
{
    let l = fvec(llr)?;
    Ok(py.detach(|| code.decode_stream(&l)).py()?.into_pyarray(py))
}

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crc::register(m)?;
    conv::register(m)?;
    ldpc_code::register(m)?;
    polar_code::register(m)?;
    nr::register(m)?;
    limits::register(m)?;
    Ok(())
}

//! Raw interleaved samples of a [`Datatype`].

use super::Datatype;
use com_core::{Error, Result};
use com_math::C64;
use com_sim::loggers::create_parent_dir;
use std::path::Path;

pub(super) fn decode_all(bytes: &[u8], dt: Datatype) -> Result<Vec<C64>> {
    let k = dt.sample_bytes();
    if bytes.len() % k != 0 {
        return Err(Error::Shape(format!("{} bytes is not a whole number of {} samples", bytes.len(), dt.name())));
    }
    Ok(bytes.chunks_exact(k).map(|b| dt.decode(b)).collect())
}

/// Read raw interleaved samples.
///
/// # Errors
/// [`Error::Io`] if the file cannot be read, [`Error::Shape`] if its size is not a
/// whole number of samples.
pub fn read_raw(path: impl AsRef<Path>, datatype: Datatype) -> Result<Vec<C64>> {
    decode_all(&std::fs::read(path)?, datatype)
}

/// Write raw interleaved samples (integer types are rounded and saturated).
///
/// # Errors
/// [`Error::Io`].
pub fn write_raw(path: impl AsRef<Path>, x: &[C64], datatype: Datatype) -> Result<()> {
    write_creating_dirs(path.as_ref(), &encode_all(x, datatype))
}

pub(super) fn encode_all(x: &[C64], datatype: Datatype) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() * datatype.sample_bytes());
    for &v in x {
        datatype.encode(v, &mut out);
    }
    out
}

pub(super) fn write_creating_dirs(path: &Path, data: &[u8]) -> Result<()> {
    create_parent_dir(path)?;
    std::fs::write(path, data)?;
    Ok(())
}

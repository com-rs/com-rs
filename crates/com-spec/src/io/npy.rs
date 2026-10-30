//! NPY arrays of NumPy (complex, real or integer).

use super::Datatype;
use super::raw::{decode_all, write_creating_dirs};
use com_core::{Error, Result};
use com_math::C64;
use std::path::Path;

/// The content of an NPY file.
#[derive(Clone, Debug, PartialEq)]
pub enum NpyData {
    /// A complex array (`<c8`, `<c16`).
    Complex(Vec<C64>),
    /// A real array (`<f4`, `<f8`).
    Real(Vec<f64>),
    /// An integer or boolean array.
    Int(Vec<i64>),
}

/// An NPY array: its shape and its data in C order.
#[derive(Clone, Debug, PartialEq)]
pub struct Npy {
    /// The shape.
    pub shape: Vec<usize>,
    /// The data.
    pub data: NpyData,
}

impl Npy {
    /// The data as complex samples (real and integer arrays are converted).
    pub fn to_complex(&self) -> Vec<C64> {
        match &self.data {
            NpyData::Complex(v) => v.clone(),
            NpyData::Real(v) => v.iter().map(|&x| C64::new(x, 0.0)).collect(),
            NpyData::Int(v) => v.iter().map(|&x| C64::new(x as f64, 0.0)).collect(),
        }
    }
}

fn header_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let i = header.find(&format!("'{key}'"))? + key.len() + 2;
    let rest = header[i..].trim_start().strip_prefix(':')?.trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else if let Some(r) = rest.strip_prefix('\'') {
        r.find('\'')? + 2
    } else {
        rest.find([',', '}']).unwrap_or(rest.len())
    };
    Some(rest[..end].trim())
}

/// Read an NPY array (little-endian numbers or booleans, C order or one dimension).
///
/// # Errors
/// [`Error::Io`] for a missing file, [`Error::Config`] for an unsupported array.
pub fn read_npy(path: impl AsRef<Path>) -> Result<Npy> {
    let path = path.as_ref();
    let b = std::fs::read(path)?;
    let bad = |m: &str| Error::Config(format!("{}: {m}", path.display()));
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(bad("not an NPY file"));
    }
    let (hlen, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 if b.len() >= 12 => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
        _ => return Err(bad("unsupported NPY version")),
    };
    let header = std::str::from_utf8(b.get(start..start + hlen).ok_or_else(|| bad("truncated header"))?)
        .map_err(|_| bad("header is not text"))?;
    let descr = header_value(header, "descr").ok_or_else(|| bad("no descr"))?.trim_matches('\'');
    let fortran = header_value(header, "fortran_order").ok_or_else(|| bad("no fortran_order"))? == "True";
    let shape: Vec<usize> = header_value(header, "shape")
        .ok_or_else(|| bad("no shape"))?
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse::<usize>().map_err(|_| bad("invalid shape")))
        .collect::<Result<_>>()?;
    if fortran && shape.iter().filter(|&&d| d > 1).count() > 1 {
        return Err(bad("Fortran-ordered arrays of more than one dimension are not supported"));
    }
    let n = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| bad("invalid shape"))?;
    let data = &b[start + hlen..];
    let need = |k: usize| -> Result<&[u8]> {
        n.checked_mul(k).and_then(|len| data.get(..len)).ok_or_else(|| bad("truncated data"))
    };
    let le = |d: &str| d.starts_with('<') || d.starts_with('|') || d.starts_with('=');
    if !le(descr) {
        return Err(bad(&format!("big-endian or unknown dtype {descr:?}")));
    }
    let data = match &descr[1..] {
        "c16" => NpyData::Complex(decode_all(need(16)?, Datatype::CF64)?),
        "c8" => NpyData::Complex(decode_all(need(8)?, Datatype::CF32)?),
        "f8" => NpyData::Real(
            need(8)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect(),
        ),
        "f4" => NpyData::Real(
            need(4)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
        ),
        "i8" => {
            NpyData::Int(need(8)?.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect())
        }
        "i4" => NpyData::Int(
            need(4)?.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
        ),
        "i2" => NpyData::Int(need(2)?.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i64).collect()),
        "i1" => NpyData::Int(need(1)?.iter().map(|&c| c as i8 as i64).collect()),
        "u1" | "b1" => NpyData::Int(need(1)?.iter().map(|&c| c as i64).collect()),
        "u2" => NpyData::Int(need(2)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as i64).collect()),
        "u4" => NpyData::Int(
            need(4)?.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64).collect(),
        ),
        other => return Err(bad(&format!("unsupported dtype {other:?}"))),
    };
    Ok(Npy { shape, data })
}

fn write_npy_bytes(path: &Path, descr: &str, n: usize, payload: &[u8]) -> Result<()> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
    // magic (6) + version (2) + length (2) + header + newline, padded to 64 bytes
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.as_bytes());
    out.extend(payload);
    write_creating_dirs(path, &out)
}

/// Write a one-dimensional complex NPY array (`<c16`).
///
/// # Errors
/// [`Error::Io`].
pub fn write_npy(path: impl AsRef<Path>, x: &[C64]) -> Result<()> {
    let mut p = Vec::with_capacity(16 * x.len());
    for &v in x {
        Datatype::CF64.encode(v, &mut p);
    }
    write_npy_bytes(path.as_ref(), "<c16", x.len(), &p)
}

/// Write a one-dimensional integer NPY array (`<i8`), e.g. symbol indices.
///
/// # Errors
/// [`Error::Io`].
pub fn write_npy_i64(path: impl AsRef<Path>, x: &[i64]) -> Result<()> {
    let p: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy_bytes(path.as_ref(), "<i8", x.len(), &p)
}

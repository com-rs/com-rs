//! Reading and writing sampled signals: measured captures and reference sequences.
//!
//! Three formats cover the usual outputs of oscilloscopes, software-defined radios and
//! numerical tools:
//!
//! - **SigMF** (the Signal Metadata Format of GNU Radio and the SDR community): a JSON
//!   file `name.sigmf-meta` with the data type and the sampling rate, next to the raw
//!   samples `name.sigmf-data`; [`read_sigmf`], [`write_sigmf`];
//! - **raw** interleaved samples with a SigMF data type (`cf32_le` is the `.cfile` of
//!   GNU Radio, `ci16_le` the IQ format of most SDRs and many oscilloscopes);
//!   [`read_raw`], [`write_raw`];
//! - **NPY** arrays of NumPy (complex, real or integer, one dimension); [`read_npy`],
//!   [`write_npy`], [`write_npy_i64`].
//!
//! [`read_samples`] picks the format from the extension. Integer samples are converted
//! to floating point without scaling (the receivers normalise the power).
//!
//! # Examples
//! ```
//! use com_spec::io::{read_sigmf, write_sigmf, Datatype};
//! use com_math::C64;
//! let dir = std::env::temp_dir().join("comrs-doc-io");
//! let x: Vec<C64> = (0..8).map(|k| C64::new(k as f64, -(k as f64))).collect();
//! write_sigmf(dir.join("capture"), &x, Datatype::Ci16, 64e9, None).unwrap();
//! let c = read_sigmf(dir.join("capture.sigmf-meta")).unwrap();
//! assert_eq!(c.samples, x);
//! assert_eq!((c.sample_rate, c.datatype), (Some(64e9), Datatype::Ci16));
//! ```

use com_core::{Error, Result};
use com_math::C64;
use com_sim::loggers::create_parent_dir;
use serde_json::{Value as Json, json};
use std::path::{Path, PathBuf};

/// A sample format of SigMF (little endian).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Datatype {
    /// Complex, two `f32` (`cf32_le`, GNU Radio `.cfile`).
    Cf32,
    /// Complex, two `f64` (`cf64_le`).
    Cf64,
    /// Complex, two `i32` (`ci32_le`).
    Ci32,
    /// Complex, two `i16` (`ci16_le`).
    Ci16,
    /// Complex, two `i8` (`ci8`).
    Ci8,
    /// Complex, two `u8` offset by 127.5 (`cu8`, RTL-SDR).
    Cu8,
    /// Real `f32` (`rf32_le`).
    Rf32,
    /// Real `f64` (`rf64_le`).
    Rf64,
    /// Real `i16` (`ri16_le`).
    Ri16,
}

impl Datatype {
    /// Parse a SigMF data type (`cf32_le`, `ci16_le`, `cu8`, …).
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown or big-endian type.
    pub fn parse(s: &str) -> Result<Datatype> {
        Ok(match s {
            "cf32_le" | "cf32" => Datatype::Cf32,
            "cf64_le" | "cf64" => Datatype::Cf64,
            "ci32_le" | "ci32" => Datatype::Ci32,
            "ci16_le" | "ci16" => Datatype::Ci16,
            "ci8" => Datatype::Ci8,
            "cu8" => Datatype::Cu8,
            "rf32_le" | "rf32" => Datatype::Rf32,
            "rf64_le" | "rf64" => Datatype::Rf64,
            "ri16_le" | "ri16" => Datatype::Ri16,
            other => {
                return Err(Error::Config(format!(
                    "unsupported data type {other:?} (cf32_le, cf64_le, ci32_le, ci16_le, ci8, cu8, rf32_le, rf64_le, ri16_le)"
                )));
            }
        })
    }

    /// The SigMF name.
    pub fn name(self) -> &'static str {
        match self {
            Datatype::Cf32 => "cf32_le",
            Datatype::Cf64 => "cf64_le",
            Datatype::Ci32 => "ci32_le",
            Datatype::Ci16 => "ci16_le",
            Datatype::Ci8 => "ci8",
            Datatype::Cu8 => "cu8",
            Datatype::Rf32 => "rf32_le",
            Datatype::Rf64 => "rf64_le",
            Datatype::Ri16 => "ri16_le",
        }
    }

    /// Bytes per sample (both components for a complex type).
    pub fn sample_bytes(self) -> usize {
        match self {
            Datatype::Cf64 => 16,
            Datatype::Cf32 | Datatype::Ci32 | Datatype::Rf64 => 8,
            Datatype::Ci16 | Datatype::Rf32 => 4,
            Datatype::Ci8 | Datatype::Cu8 | Datatype::Ri16 => 2,
        }
    }

    fn decode(self, b: &[u8]) -> C64 {
        let f32_at = |i: usize| f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as f64;
        let f64_at = |i: usize| f64::from_le_bytes(b[i..i + 8].try_into().unwrap_or([0; 8]));
        let i16_at = |i: usize| i16::from_le_bytes([b[i], b[i + 1]]) as f64;
        let i32_at = |i: usize| i32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as f64;
        match self {
            Datatype::Cf32 => C64::new(f32_at(0), f32_at(4)),
            Datatype::Cf64 => C64::new(f64_at(0), f64_at(8)),
            Datatype::Ci32 => C64::new(i32_at(0), i32_at(4)),
            Datatype::Ci16 => C64::new(i16_at(0), i16_at(2)),
            Datatype::Ci8 => C64::new(b[0] as i8 as f64, b[1] as i8 as f64),
            Datatype::Cu8 => C64::new(b[0] as f64 - 127.5, b[1] as f64 - 127.5),
            Datatype::Rf32 => C64::new(f32_at(0), 0.0),
            Datatype::Rf64 => C64::new(f64_at(0), 0.0),
            Datatype::Ri16 => C64::new(i16_at(0), 0.0),
        }
    }

    fn encode(self, v: C64, out: &mut Vec<u8>) {
        let sat16 = |x: f64| x.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        let sat32 = |x: f64| x.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        let sat8 = |x: f64| x.round().clamp(i8::MIN as f64, i8::MAX as f64) as i8;
        match self {
            Datatype::Cf32 => {
                out.extend((v.re as f32).to_le_bytes());
                out.extend((v.im as f32).to_le_bytes());
            }
            Datatype::Cf64 => {
                out.extend(v.re.to_le_bytes());
                out.extend(v.im.to_le_bytes());
            }
            Datatype::Ci32 => {
                out.extend(sat32(v.re).to_le_bytes());
                out.extend(sat32(v.im).to_le_bytes());
            }
            Datatype::Ci16 => {
                out.extend(sat16(v.re).to_le_bytes());
                out.extend(sat16(v.im).to_le_bytes());
            }
            Datatype::Ci8 => out.extend([sat8(v.re) as u8, sat8(v.im) as u8]),
            Datatype::Cu8 => out.extend([
                (v.re + 127.5).round().clamp(0.0, 255.0) as u8,
                (v.im + 127.5).round().clamp(0.0, 255.0) as u8,
            ]),
            Datatype::Rf32 => out.extend((v.re as f32).to_le_bytes()),
            Datatype::Rf64 => out.extend(v.re.to_le_bytes()),
            Datatype::Ri16 => out.extend(sat16(v.re).to_le_bytes()),
        }
    }
}

fn decode_all(bytes: &[u8], dt: Datatype) -> Result<Vec<C64>> {
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
    let mut out = Vec::with_capacity(x.len() * datatype.sample_bytes());
    for &v in x {
        datatype.encode(v, &mut out);
    }
    write_creating_dirs(path.as_ref(), &out)
}

fn write_creating_dirs(path: &Path, data: &[u8]) -> Result<()> {
    create_parent_dir(path)?;
    std::fs::write(path, data)?;
    Ok(())
}

/// A SigMF recording.
#[derive(Clone, Debug)]
pub struct SigMF {
    /// The samples.
    pub samples: Vec<C64>,
    /// `core:sample_rate` (Hz).
    pub sample_rate: Option<f64>,
    /// `core:frequency` of the first capture (Hz).
    pub frequency: Option<f64>,
    /// `core:datatype`.
    pub datatype: Datatype,
    /// The whole metadata document.
    pub meta: Json,
}

/// The paths `(name.sigmf-meta, name.sigmf-data)` of a SigMF recording given either.
fn sigmf_paths(path: &Path) -> (PathBuf, PathBuf) {
    let s = path.to_string_lossy();
    let base = s.strip_suffix(".sigmf-meta").or_else(|| s.strip_suffix(".sigmf-data")).unwrap_or(&s).to_string();
    (PathBuf::from(format!("{base}.sigmf-meta")), PathBuf::from(format!("{base}.sigmf-data")))
}

/// Read a SigMF recording from its `.sigmf-meta` or `.sigmf-data` file (or their
/// common base name).
///
/// # Errors
/// [`Error::Io`] for a missing file, [`Error::Config`] for invalid metadata.
pub fn read_sigmf(path: impl AsRef<Path>) -> Result<SigMF> {
    let (meta_path, data_path) = sigmf_paths(path.as_ref());
    let meta: Json = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)
        .map_err(|e| Error::Config(format!("{}: {e}", meta_path.display())))?;
    let global = &meta["global"];
    let dt = global["core:datatype"]
        .as_str()
        .ok_or_else(|| Error::Config(format!("{}: no global core:datatype", meta_path.display())))?;
    let datatype = Datatype::parse(dt)?;
    let samples = read_raw(&data_path, datatype)?;
    Ok(SigMF {
        samples,
        sample_rate: global["core:sample_rate"].as_f64(),
        frequency: meta["captures"].get(0).and_then(|c| c["core:frequency"].as_f64()),
        datatype,
        meta,
    })
}

/// Write a SigMF recording `base.sigmf-meta` and `base.sigmf-data`.
///
/// `global` adds keys to the global object of the metadata (a description, the
/// hardware, the symbol rate of the transmitted signal, …).
///
/// # Errors
/// [`Error::Io`].
pub fn write_sigmf(
    base: impl AsRef<Path>,
    x: &[C64],
    datatype: Datatype,
    sample_rate: f64,
    global: Option<&Json>,
) -> Result<()> {
    let (meta_path, data_path) = sigmf_paths(base.as_ref());
    write_raw(&data_path, x, datatype)?;
    let mut g = json!({
        "core:datatype": datatype.name(),
        "core:sample_rate": sample_rate,
        "core:version": "1.0.0",
        "core:recorder": format!("comrs {}", env!("CARGO_PKG_VERSION")),
    });
    if let (Some(Json::Object(extra)), Json::Object(gm)) = (global, &mut g) {
        gm.extend(extra.clone());
    }
    let meta = json!({"global": g, "captures": [{"core:sample_start": 0}], "annotations": []});
    write_creating_dirs(&meta_path, (serde_json::to_string_pretty(&meta).unwrap_or_default() + "\n").as_bytes())
}

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
    let n: usize = shape.iter().product();
    let data = &b[start + hlen..];
    let need = |k: usize| -> Result<&[u8]> { data.get(..n * k).ok_or_else(|| bad("truncated data")) };
    let le = |d: &str| d.starts_with('<') || d.starts_with('|') || d.starts_with('=');
    if !le(descr) {
        return Err(bad(&format!("big-endian or unknown dtype {descr:?}")));
    }
    let data = match &descr[1..] {
        "c16" => NpyData::Complex(decode_all(need(16)?, Datatype::Cf64)?),
        "c8" => NpyData::Complex(decode_all(need(8)?, Datatype::Cf32)?),
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
        Datatype::Cf64.encode(v, &mut p);
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

/// Read samples from a SigMF recording (`.sigmf-meta`, `.sigmf-data`), an NPY array
/// (`.npy`) or a raw file of the given data type; returns the samples and the sampling
/// rate when the file records it.
///
/// # Errors
/// [`Error::Config`] for a raw file without a data type; the errors of the readers.
pub fn read_samples(path: impl AsRef<Path>, datatype: Option<Datatype>) -> Result<(Vec<C64>, Option<f64>)> {
    let path = path.as_ref();
    let name = path.to_string_lossy();
    if name.ends_with(".sigmf-meta") || name.ends_with(".sigmf-data") {
        let s = read_sigmf(path)?;
        return Ok((s.samples, s.sample_rate));
    }
    if name.ends_with(".npy") {
        return Ok((read_npy(path)?.to_complex(), None));
    }
    let dt = datatype.ok_or_else(|| {
        Error::Config(format!("{}: a raw file needs a datatype (cf32_le, ci16_le, …)", path.display()))
    })?;
    Ok((read_raw(path, dt)?, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("comrs-io-{}-{name}", std::process::id()))
    }

    #[test]
    fn raw_round_trips_for_every_type() {
        let x: Vec<C64> = (0..50).map(|k| C64::new(k as f64 - 25.0, (3.0 - k as f64 / 2.0).round())).collect();
        for dt in [Datatype::Cf32, Datatype::Cf64, Datatype::Ci32, Datatype::Ci16, Datatype::Ci8] {
            let p = tmp(dt.name());
            write_raw(&p, &x, dt).unwrap();
            assert_eq!(read_raw(&p, dt).unwrap(), x, "{}", dt.name());
            assert_eq!(std::fs::metadata(&p).unwrap().len() as usize, x.len() * dt.sample_bytes());
            std::fs::remove_file(p).ok();
        }
        let p = tmp("rf32");
        write_raw(&p, &x, Datatype::Rf32).unwrap();
        assert!(read_raw(&p, Datatype::Rf32).unwrap().iter().zip(&x).all(|(a, b)| a.re == b.re && a.im == 0.0));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn npy_round_trips_and_reads_numpy_headers() {
        let x: Vec<C64> = (0..7).map(|k| C64::new(k as f64 * 0.5, -1.0)).collect();
        let p = tmp("a.npy");
        write_npy(&p, &x).unwrap();
        let a = read_npy(&p).unwrap();
        assert_eq!((a.shape.clone(), a.data.clone()), (vec![7], NpyData::Complex(x)));
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!((10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize) % 64, 0);
        write_npy_i64(&p, &[3, -1, 7]).unwrap();
        assert_eq!(read_npy(&p).unwrap().data, NpyData::Int(vec![3, -1, 7]));
        std::fs::write(&p, b"not numpy").unwrap();
        assert!(read_npy(&p).is_err());
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn sigmf_keeps_the_metadata() {
        let x = vec![C64::new(0.25, -0.5); 16];
        let base = tmp("rec");
        write_sigmf(&base, &x, Datatype::Cf32, 80e9, Some(&json!({"comrs:symbol_rate": 32e9}))).unwrap();
        let r = read_sigmf(format!("{}.sigmf-data", base.display())).unwrap();
        assert_eq!(r.samples, x);
        assert_eq!(r.meta["global"]["comrs:symbol_rate"], json!(32e9));
        let (s, fs) = read_samples(format!("{}.sigmf-meta", base.display()), None).unwrap();
        assert_eq!((s.len(), fs), (16, Some(80e9)));
        assert!(read_samples(&base, None).is_err());
    }
}

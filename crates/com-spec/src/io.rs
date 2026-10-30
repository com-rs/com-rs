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
//! write_sigmf(dir.join("capture"), &x, Datatype::CI16, 64e9, None).unwrap();
//! let c = read_sigmf(dir.join("capture.sigmf-meta")).unwrap();
//! assert_eq!(c.samples, x);
//! assert_eq!((c.sample_rate, c.datatype), (Some(64e9), Datatype::CI16));
//! ```

use com_core::{Error, Result, ensure};
use com_math::C64;
use com_sim::loggers::create_parent_dir;
use serde_json::{Value as Json, json};
use std::path::{Path, PathBuf};

/// The scalar type of the components of a SigMF sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// `i32`.
    I32,
    /// `i16`.
    I16,
    /// `i8`.
    I8,
    /// `u32`.
    U32,
    /// `u16`.
    U16,
    /// `u8`.
    U8,
}

impl Scalar {
    /// Bytes per component.
    pub fn bytes(self) -> usize {
        match self {
            Scalar::F64 => 8,
            Scalar::F32 | Scalar::I32 | Scalar::U32 => 4,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I8 | Scalar::U8 => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::I32 => "i32",
            Scalar::I16 => "i16",
            Scalar::I8 => "i8",
            Scalar::U32 => "u32",
            Scalar::U16 => "u16",
            Scalar::U8 => "u8",
        }
    }

    fn is_integer(self) -> bool {
        !matches!(self, Scalar::F32 | Scalar::F64)
    }

    /// $2^{b-1}$ for an integer of $b$ bits: the full scale, and the offset of the
    /// unsigned types.
    fn half_range(self) -> f64 {
        2f64.powi(8 * self.bytes() as i32 - 1)
    }
}

/// A sample format of SigMF: `[c|r][f32|f64|i32|i16|i8|u32|u16|u8][_le|_be]` (the
/// 8-bit types without endianness).
///
/// Integer samples are converted to floating point without scaling; the unsigned types
/// are centred by $2^{b-1}$ (128 for `cu8`), as the reference implementation
/// `sigmf-python` does. [`SigMF::full_scale`] gives the scale of its `autoscale`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Datatype {
    /// Complex (interleaved I and Q) or real.
    pub complex: bool,
    /// The type of a component.
    pub scalar: Scalar,
    /// Big-endian components.
    pub big_endian: bool,
}

impl Datatype {
    /// `cf32_le`, the `.cfile` of GNU Radio.
    pub const CF32: Datatype = Datatype { complex: true, scalar: Scalar::F32, big_endian: false };
    /// `cf64_le`.
    pub const CF64: Datatype = Datatype { complex: true, scalar: Scalar::F64, big_endian: false };
    /// `ci32_le`.
    pub const CI32: Datatype = Datatype { complex: true, scalar: Scalar::I32, big_endian: false };
    /// `ci16_le`, the IQ format of most SDRs.
    pub const CI16: Datatype = Datatype { complex: true, scalar: Scalar::I16, big_endian: false };
    /// `ci8`.
    pub const CI8: Datatype = Datatype { complex: true, scalar: Scalar::I8, big_endian: false };
    /// `cu8` (RTL-SDR).
    pub const CU8: Datatype = Datatype { complex: true, scalar: Scalar::U8, big_endian: false };
    /// `rf32_le`.
    pub const RF32: Datatype = Datatype { complex: false, scalar: Scalar::F32, big_endian: false };
    /// `rf64_le`.
    pub const RF64: Datatype = Datatype { complex: false, scalar: Scalar::F64, big_endian: false };
    /// `ri16_le`.
    pub const RI16: Datatype = Datatype { complex: false, scalar: Scalar::I16, big_endian: false };

    /// Parse a SigMF data type (`cf32_le`, `ci16_be`, `cu8`, `ri8`, …; the endianness
    /// may be omitted, little endian then).
    ///
    /// # Errors
    /// [`Error::Config`] for an unknown type.
    pub fn parse(s: &str) -> Result<Datatype> {
        let bad = || {
            Error::Config(format!(
                "unsupported data type {s:?}: [c|r][f32|f64|i32|i16|i8|u32|u16|u8][_le|_be] (cf32_le, ci16_le, cu8, …)"
            ))
        };
        let (complex, rest) = match s.split_at_checked(1) {
            Some(("c", r)) => (true, r),
            Some(("r", r)) => (false, r),
            _ => return Err(bad()),
        };
        let (body, big_endian) = match rest.rsplit_once('_') {
            Some((b, "le")) => (b, false),
            Some((b, "be")) => (b, true),
            Some(_) => return Err(bad()),
            None => (rest, false),
        };
        let scalar = match body {
            "f32" => Scalar::F32,
            "f64" => Scalar::F64,
            "i32" => Scalar::I32,
            "i16" => Scalar::I16,
            "i8" => Scalar::I8,
            "u32" => Scalar::U32,
            "u16" => Scalar::U16,
            "u8" => Scalar::U8,
            _ => return Err(bad()),
        };
        Ok(Datatype { complex, scalar, big_endian: big_endian && scalar.bytes() > 1 })
    }

    /// The SigMF name.
    pub fn name(self) -> String {
        let end = match (self.scalar.bytes(), self.big_endian) {
            (1, _) => "",
            (_, false) => "_le",
            (_, true) => "_be",
        };
        format!("{}{}{end}", if self.complex { "c" } else { "r" }, self.scalar.name())
    }

    /// Bytes per sample (both components for a complex type).
    pub fn sample_bytes(self) -> usize {
        self.scalar.bytes() * if self.complex { 2 } else { 1 }
    }

    fn component(self, b: &[u8]) -> f64 {
        macro_rules! num {
            ($t:ty, $n:expr) => {{
                let a: [u8; $n] = b[..$n].try_into().unwrap_or([0; $n]);
                if self.big_endian { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) }
            }};
        }
        let h = self.scalar.half_range();
        match self.scalar {
            Scalar::F32 => f64::from(num!(f32, 4)),
            Scalar::F64 => num!(f64, 8),
            Scalar::I32 => f64::from(num!(i32, 4)),
            Scalar::I16 => f64::from(num!(i16, 2)),
            Scalar::I8 => f64::from(b[0] as i8),
            Scalar::U32 => f64::from(num!(u32, 4)) - h,
            Scalar::U16 => f64::from(num!(u16, 2)) - h,
            Scalar::U8 => f64::from(b[0]) - h,
        }
    }

    fn decode(self, b: &[u8]) -> C64 {
        let k = self.scalar.bytes();
        C64::new(self.component(b), if self.complex { self.component(&b[k..]) } else { 0.0 })
    }

    fn put(self, x: f64, out: &mut Vec<u8>) {
        macro_rules! put {
            ($v:expr) => {{
                let v = $v;
                if self.big_endian { out.extend(v.to_be_bytes()) } else { out.extend(v.to_le_bytes()) }
            }};
        }
        let h = self.scalar.half_range();
        // integers: rounded and saturated
        let sat = |lo: f64, hi: f64| x.round().clamp(lo, hi);
        match self.scalar {
            Scalar::F32 => put!(x as f32),
            Scalar::F64 => put!(x),
            Scalar::I32 => put!(sat(-h, h - 1.0) as i32),
            Scalar::I16 => put!(sat(-h, h - 1.0) as i16),
            Scalar::I8 => put!(sat(-h, h - 1.0) as i8),
            Scalar::U32 => put!((sat(-h, h - 1.0) + h) as u32),
            Scalar::U16 => put!((sat(-h, h - 1.0) + h) as u16),
            Scalar::U8 => put!((sat(-h, h - 1.0) + h) as u8),
        }
    }

    fn encode(self, v: C64, out: &mut Vec<u8>) {
        self.put(v.re, out);
        if self.complex {
            self.put(v.im, out);
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
    write_creating_dirs(path.as_ref(), &encode_all(x, datatype))
}

fn encode_all(x: &[C64], datatype: Datatype) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() * datatype.sample_bytes());
    for &v in x {
        datatype.encode(v, &mut out);
    }
    out
}

fn write_creating_dirs(path: &Path, data: &[u8]) -> Result<()> {
    create_parent_dir(path)?;
    std::fs::write(path, data)?;
    Ok(())
}

/// A capture segment of a SigMF recording.
#[derive(Clone, Debug, PartialEq)]
pub struct Capture {
    /// `core:sample_start`: the first sample of the segment.
    pub sample_start: usize,
    /// `core:frequency`: the centre frequency (Hz).
    pub frequency: Option<f64>,
    /// `core:datetime` (ISO 8601).
    pub datetime: Option<String>,
}

/// An annotation of a SigMF recording.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Annotation {
    /// `core:sample_start`.
    pub sample_start: usize,
    /// `core:sample_count` (to the end of the recording when absent).
    pub sample_count: Option<usize>,
    /// `core:label` (a short name: the modulation, the emitter, …).
    pub label: Option<String>,
    /// `core:comment`.
    pub comment: Option<String>,
    /// `core:freq_lower_edge` (Hz).
    pub freq_lower_edge: Option<f64>,
    /// `core:freq_upper_edge` (Hz).
    pub freq_upper_edge: Option<f64>,
}

impl Annotation {
    /// The keys of [`Annotation::from_mapping`].
    pub const KEYS: [&'static str; 6] =
        ["sample_start", "sample_count", "label", "comment", "freq_lower_edge", "freq_upper_edge"];

    /// An annotation from a mapping of the keys of [`Annotation::KEYS`] (the SigMF keys
    /// without their `core:` prefix), as the Python bindings take it: `sample_start` is
    /// required, the others are optional.
    ///
    /// # Errors
    /// [`Error::Config`] for another key, or a missing or negative `sample_start`.
    ///
    /// # Examples
    /// ```
    /// use com_spec::io::Annotation;
    /// let a = Annotation::from_mapping(&serde_json::json!({"sample_start": 3, "label": "QPSK"}))?;
    /// assert_eq!((a.sample_start, a.label.as_deref()), (3, Some("QPSK")));
    /// assert!(Annotation::from_mapping(&serde_json::json!({"sample_start": 3, "start": 1})).is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn from_mapping(a: &Json) -> Result<Annotation> {
        let keys = Annotation::KEYS;
        if let Some(k) = a.as_object().and_then(|o| o.keys().find(|k| !keys.contains(&k.as_str()))) {
            return Err(Error::Config(format!("unknown annotation key {k:?} (one of {keys:?})")));
        }
        let start = a["sample_start"]
            .as_u64()
            .ok_or_else(|| Error::Config("every annotation needs a non-negative integer sample_start".into()))?;
        Ok(Annotation {
            sample_start: start as usize,
            sample_count: a["sample_count"].as_u64().map(|v| v as usize),
            label: a["label"].as_str().map(str::to_string),
            comment: a["comment"].as_str().map(str::to_string),
            freq_lower_edge: a["freq_lower_edge"].as_f64(),
            freq_upper_edge: a["freq_upper_edge"].as_f64(),
        })
    }

    fn from_json(a: &Json) -> Annotation {
        let s = |k: &str| a[k].as_str().map(str::to_string);
        Annotation {
            sample_start: a["core:sample_start"].as_u64().unwrap_or(0) as usize,
            sample_count: a["core:sample_count"].as_u64().map(|v| v as usize),
            label: s("core:label"),
            comment: s("core:comment"),
            freq_lower_edge: a["core:freq_lower_edge"].as_f64(),
            freq_upper_edge: a["core:freq_upper_edge"].as_f64(),
        }
    }

    fn to_json(&self) -> Json {
        let mut m = serde_json::Map::new();
        m.insert("core:sample_start".into(), json!(self.sample_start));
        if let Some(v) = self.sample_count {
            m.insert("core:sample_count".into(), json!(v));
        }
        for (k, v) in [("core:label", &self.label), ("core:comment", &self.comment)] {
            if let Some(v) = v {
                m.insert(k.into(), json!(v));
            }
        }
        for (k, v) in [("core:freq_lower_edge", self.freq_lower_edge), ("core:freq_upper_edge", self.freq_upper_edge)] {
            if let Some(v) = v {
                m.insert(k.into(), json!(v));
            }
        }
        Json::Object(m)
    }
}

/// A SigMF recording.
#[derive(Clone, Debug)]
pub struct SigMF {
    /// The samples, the channels interleaved (`num_channels` per instant).
    pub samples: Vec<C64>,
    /// `core:num_channels` (1 when absent).
    pub num_channels: usize,
    /// `core:sample_rate` (Hz).
    pub sample_rate: Option<f64>,
    /// `core:frequency` of the first capture (Hz).
    pub frequency: Option<f64>,
    /// `core:datatype`.
    pub datatype: Datatype,
    /// The capture segments, in the order of their first sample.
    pub captures: Vec<Capture>,
    /// The annotations, in the order of their first sample.
    pub annotations: Vec<Annotation>,
    /// The whole metadata document.
    pub meta: Json,
}

impl SigMF {
    /// The samples of channel `k` (none beyond the last channel).
    pub fn channel(&self, k: usize) -> Vec<C64> {
        self.samples.iter().skip(k).step_by(self.num_channels.max(1)).copied().collect()
    }

    /// The samples of channel `k`, which must exist.
    ///
    /// # Errors
    /// [`Error::Config`] if the recording has no channel `k`.
    pub fn try_channel(&self, k: usize) -> Result<Vec<C64>> {
        ensure(k < self.num_channels, || format!("channel {k} of a recording of {}", self.num_channels))?;
        Ok(self.channel(k))
    }

    /// The full scale of an integer type, $2^{b-1}$ for $b$ bits (`None` for floating
    /// point): dividing by it gives the samples of `sigmf-python` with `autoscale`.
    pub fn full_scale(&self) -> Option<f64> {
        self.datatype.scalar.is_integer().then(|| self.datatype.scalar.half_range())
    }

    /// The samples of channel `k` covered by an annotation.
    pub fn annotated(&self, a: &Annotation, k: usize) -> Vec<C64> {
        let x = self.channel(k);
        let start = a.sample_start.min(x.len());
        let end = a.sample_count.map_or(x.len(), |c| (start + c).min(x.len()));
        x[start..end].to_vec()
    }
}

/// The paths `(name.sigmf-meta, name.sigmf-data)` of a SigMF recording given either.
fn sigmf_paths(path: &Path) -> (PathBuf, PathBuf) {
    let s = path.to_string_lossy();
    let base = s.strip_suffix(".sigmf-meta").or_else(|| s.strip_suffix(".sigmf-data")).unwrap_or(&s).to_string();
    (PathBuf::from(format!("{base}.sigmf-meta")), PathBuf::from(format!("{base}.sigmf-data")))
}

/// The files of a tar archive (POSIX ustar, as written by `sigmf-python` and GNU tar):
/// `(name, content)`.
fn untar(bytes: &[u8]) -> Result<Vec<(String, &[u8])>> {
    let mut files = Vec::new();
    let mut at = 0;
    let octal = |f: &[u8]| -> Result<usize> {
        let s: String = f.iter().take_while(|&&c| c != 0).map(|&c| c as char).collect();
        usize::from_str_radix(s.trim(), 8).map_err(|_| Error::Config(format!("tar: bad size field {s:?}")))
    };
    let text = |f: &[u8]| f.iter().take_while(|&&c| c != 0).map(|&c| c as char).collect::<String>();
    while at + 512 <= bytes.len() {
        let h = &bytes[at..at + 512];
        if h.iter().all(|&c| c == 0) {
            break;
        }
        let size = octal(&h[124..136])?;
        let (prefix, name) = (text(&h[345..500]), text(&h[..100]));
        let name = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
        let start = at + 512;
        // a corrupt size field may be anything: no arithmetic that can overflow
        let Some(end) = start.checked_add(size).filter(|&e| e <= bytes.len()) else {
            return Err(Error::Config(format!("tar: {name} ends after the archive")));
        };
        // regular files only ('0' or NUL)
        if matches!(h[156], b'0' | 0) {
            files.push((name, &bytes[start..end]));
        }
        at = end.div_ceil(512).saturating_mul(512);
    }
    Ok(files)
}

/// Read a SigMF recording from its `.sigmf-meta` or `.sigmf-data` file, their common
/// base name, or a `.sigmf` archive (the first recording of the archive).
///
/// The reader follows SigMF 1.x: every data type ([`Datatype`]), interleaved channels
/// (`core:num_channels`), the header bytes of the captures (`core:header_bytes`) and the
/// trailing bytes of the file (`core:trailing_bytes`) removed, a non-conforming dataset
/// (`core:dataset`, a data file of another name next to the metadata).
///
/// # Errors
/// [`Error::Io`] for a missing file, [`Error::Config`] for invalid metadata,
/// [`Error::Shape`] for a data file that is not a whole number of samples.
pub fn read_sigmf(path: impl AsRef<Path>) -> Result<SigMF> {
    let path = path.as_ref();
    if path.extension().is_some_and(|e| e == "sigmf") {
        let bytes = std::fs::read(path)?;
        let files = untar(&bytes)?;
        let (meta_name, meta) = files
            .iter()
            .find(|(n, _)| n.ends_with(".sigmf-meta"))
            .ok_or_else(|| Error::Config(format!("{}: no .sigmf-meta in the archive", path.display())))?;
        let data_name = format!("{}.sigmf-data", meta_name.trim_end_matches(".sigmf-meta"));
        let data = files.iter().find(|(n, _)| *n == data_name).map_or(&[][..], |(_, d)| *d);
        let meta: Json =
            serde_json::from_slice(meta).map_err(|e| Error::Config(format!("{}/{meta_name}: {e}", path.display())))?;
        return parse_sigmf(meta, data, &format!("{}/{meta_name}", path.display()));
    }
    let (meta_path, mut data_path) = sigmf_paths(path);
    let meta: Json = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)
        .map_err(|e| Error::Config(format!("{}: {e}", meta_path.display())))?;
    if let Some(d) = meta["global"]["core:dataset"].as_str() {
        data_path = meta_path.with_file_name(d);
    }
    let data = std::fs::read(&data_path)?;
    parse_sigmf(meta, &data, &meta_path.display().to_string())
}

fn parse_sigmf(meta: Json, data: &[u8], name: &str) -> Result<SigMF> {
    let global = &meta["global"];
    let dt =
        global["core:datatype"].as_str().ok_or_else(|| Error::Config(format!("{name}: no global core:datatype")))?;
    let datatype = Datatype::parse(dt)?;
    let num_channels = global["core:num_channels"].as_u64().unwrap_or(1).max(1) as usize;
    let trailing = global["core:trailing_bytes"].as_u64().unwrap_or(0) as usize;
    let data = &data[..data.len().saturating_sub(trailing)];
    let mut captures: Vec<(Capture, usize)> = meta["captures"]
        .as_array()
        .map(|cs| {
            cs.iter()
                .map(|c| {
                    let cap = Capture {
                        sample_start: c["core:sample_start"].as_u64().unwrap_or(0) as usize,
                        frequency: c["core:frequency"].as_f64(),
                        datetime: c["core:datetime"].as_str().map(str::to_string),
                    };
                    (cap, c["core:header_bytes"].as_u64().unwrap_or(0) as usize)
                })
                .collect()
        })
        .unwrap_or_default();
    captures.sort_by_key(|(c, _)| c.sample_start);
    // drop the header bytes in front of every capture
    let frame = datatype.sample_bytes() * num_channels;
    let mut body = Vec::with_capacity(data.len());
    let mut at = 0usize;
    for (i, (c, header)) in captures.iter().enumerate() {
        let skip_to = at.saturating_add(*header);
        ensure_le(skip_to, data.len(), name)?;
        // corrupt sample starts may be anything: saturate rather than overflow
        let samples = captures.get(i + 1).map(|(n, _)| (n.sample_start - c.sample_start).saturating_mul(frame));
        let end = samples.map_or(data.len(), |b| skip_to.saturating_add(b).min(data.len()));
        body.extend_from_slice(&data[skip_to..end]);
        at = end;
    }
    if captures.is_empty() {
        body.extend_from_slice(data);
    }
    let samples = decode_all(&body, datatype)?;
    if samples.len() % num_channels != 0 {
        return Err(Error::Shape(format!(
            "{name}: {} samples is not a whole number of {num_channels} channels",
            samples.len()
        )));
    }
    let mut annotations: Vec<Annotation> =
        meta["annotations"].as_array().map(|a| a.iter().map(Annotation::from_json).collect()).unwrap_or_default();
    annotations.sort_by_key(|a| a.sample_start);
    let captures: Vec<Capture> = captures.into_iter().map(|(c, _)| c).collect();
    Ok(SigMF {
        samples,
        num_channels,
        sample_rate: global["core:sample_rate"].as_f64(),
        frequency: captures.first().and_then(|c| c.frequency),
        datatype,
        captures,
        annotations,
        meta,
    })
}

fn ensure_le(a: usize, b: usize, name: &str) -> Result<()> {
    if a > b {
        return Err(Error::Config(format!("{name}: the header bytes of a capture go beyond the data file")));
    }
    Ok(())
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
    write_sigmf_annotated(base, x, datatype, sample_rate, None, global, &[])
}

/// Write a SigMF recording with the centre frequency of its capture (`core:frequency`)
/// and annotations (see [`write_sigmf`]).
///
/// # Errors
/// [`Error::Io`].
pub fn write_sigmf_annotated(
    base: impl AsRef<Path>,
    x: &[C64],
    datatype: Datatype,
    sample_rate: f64,
    frequency: Option<f64>,
    global: Option<&Json>,
    annotations: &[Annotation],
) -> Result<()> {
    let (meta_path, data_path) = sigmf_paths(base.as_ref());
    write_raw(&data_path, x, datatype)?;
    let mut g = json!({
        "core:datatype": datatype.name(),
        "core:sample_rate": sample_rate,
        "core:version": "1.2.0",
        "core:recorder": format!("comrs {}", env!("CARGO_PKG_VERSION")),
    });
    if let (Some(Json::Object(extra)), Json::Object(gm)) = (global, &mut g) {
        gm.extend(extra.clone());
    }
    let mut capture = json!({"core:sample_start": 0});
    if let (Some(f), Json::Object(c)) = (frequency, &mut capture) {
        c.insert("core:frequency".into(), json!(f));
    }
    let mut annotations = annotations.to_vec();
    annotations.sort_by_key(|a| a.sample_start);
    let annotations: Vec<Json> = annotations.iter().map(Annotation::to_json).collect();
    let meta = json!({"global": g, "captures": [capture], "annotations": annotations});
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

/// The samples of one channel of a recording ([`read_samples`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Samples {
    /// The complex samples.
    pub samples: Vec<C64>,
    /// The sampling rate (Hz), when the file records it.
    pub sample_rate: Option<f64>,
}

/// Read samples from a SigMF recording (`.sigmf-meta`, `.sigmf-data`, `.sigmf`), an NPY
/// array (`.npy`) or a raw file of the given data type: the samples of `channel` and the
/// sampling rate when the file records it.
///
/// # Errors
/// [`Error::Config`] for a raw file without a data type, or a channel that the recording
/// does not have; the errors of the readers.
pub fn read_samples(path: impl AsRef<Path>, datatype: Option<Datatype>, channel: usize) -> Result<Samples> {
    let path = path.as_ref();
    let name = path.to_string_lossy();
    let channels = |n: usize| {
        if channel < n {
            Ok(())
        } else {
            Err(Error::Config(format!("{}: channel {channel} of a recording of {n}", path.display())))
        }
    };
    if name.ends_with(".sigmf-meta") || name.ends_with(".sigmf-data") || name.ends_with(".sigmf") {
        let s = read_sigmf(path)?;
        channels(s.num_channels)?;
        return Ok(Samples { samples: s.channel(channel), sample_rate: s.sample_rate });
    }
    channels(1)?;
    if name.ends_with(".npy") {
        return Ok(Samples { samples: read_npy(path)?.to_complex(), sample_rate: None });
    }
    let dt = datatype.ok_or_else(|| {
        Error::Config(format!("{}: a raw file needs a datatype (cf32_le, ci16_le, …)", path.display()))
    })?;
    Ok(Samples { samples: read_raw(path, dt)?, sample_rate: None })
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
        for dt in [Datatype::CF32, Datatype::CF64, Datatype::CI32, Datatype::CI16, Datatype::CI8] {
            let p = tmp(&dt.name());
            write_raw(&p, &x, dt).unwrap();
            assert_eq!(read_raw(&p, dt).unwrap(), x, "{}", dt.name());
            assert_eq!(std::fs::metadata(&p).unwrap().len() as usize, x.len() * dt.sample_bytes());
            std::fs::remove_file(p).ok();
        }
        let p = tmp("rf32");
        write_raw(&p, &x, Datatype::RF32).unwrap();
        assert!(read_raw(&p, Datatype::RF32).unwrap().iter().zip(&x).all(|(a, b)| a.re == b.re && a.im == 0.0));
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn every_sigmf_type_round_trips() {
        let x: Vec<C64> = (0..40).map(|k| C64::new(k as f64 - 20.0, (5.0 - k as f64 / 3.0).round())).collect();
        for c in ["c", "r"] {
            for t in ["f32", "f64", "i32", "i16", "i8", "u32", "u16", "u8"] {
                for e in ["_le", "_be"] {
                    let name = format!("{c}{t}{e}");
                    let dt = Datatype::parse(&name).unwrap();
                    let p = tmp(&format!("all-{name}"));
                    write_raw(&p, &x, dt).unwrap();
                    let y = read_raw(&p, dt).unwrap();
                    let want: Vec<C64> = x.iter().map(|v| if dt.complex { *v } else { C64::new(v.re, 0.0) }).collect();
                    assert_eq!(y, want, "{name}");
                    std::fs::remove_file(p).ok();
                    // the 8-bit types have no endianness
                    assert_eq!(dt.name(), if t.ends_with('8') { format!("{c}{t}") } else { name });
                }
            }
        }
        // big endian: 1.0 as f32 is 3f 80 00 00
        let dt = Datatype::parse("rf32_be").unwrap();
        assert_eq!(dt.decode(&[0x3f, 0x80, 0, 0]), C64::new(1.0, 0.0));
        // unsigned: centred by 2^(b-1)
        assert_eq!(Datatype::CU8.decode(&[128, 0]), C64::new(0.0, -128.0));
        assert!(Datatype::parse("cx16").is_err() && Datatype::parse("ci16_xx").is_err());
    }

    #[test]
    fn sigmf_headers_channels_and_annotations() {
        // two channels of ci16, two captures each preceded by a header, trailing bytes
        let dir = tmp("ncd");
        std::fs::create_dir_all(&dir).unwrap();
        let x: Vec<C64> = (0..12).map(|k| C64::new(k as f64, -(k as f64))).collect();
        let raw = encode_all(&x, Datatype::CI16);
        // capture 0: samples 0..3 (6 values, 24 bytes) after 5 header bytes; capture 1:
        // samples 3..6 after 7 header bytes; then 3 trailing bytes
        let mut data = vec![9u8; 5];
        data.extend_from_slice(&raw[..24]);
        data.extend(vec![8u8; 7]);
        data.extend_from_slice(&raw[24..]);
        data.extend([1, 2, 3]);
        std::fs::write(dir.join("other.bin"), &data).unwrap();
        let meta = json!({
            "global": {"core:datatype": "ci16_le", "core:num_channels": 2, "core:sample_rate": 1e6,
                       "core:trailing_bytes": 3, "core:dataset": "other.bin", "core:version": "1.2.0"},
            "captures": [{"core:sample_start": 3, "core:header_bytes": 7, "core:frequency": 2.4e9},
                         {"core:sample_start": 0, "core:header_bytes": 5, "core:frequency": 2.4e9}],
            "annotations": [{"core:sample_start": 4, "core:sample_count": 2, "core:label": "QPSK"}]
        });
        std::fs::write(dir.join("rec.sigmf-meta"), meta.to_string()).unwrap();
        let r = read_sigmf(dir.join("rec.sigmf-meta")).unwrap();
        assert_eq!(r.samples, x);
        assert_eq!((r.num_channels, r.frequency, r.captures.len()), (2, Some(2.4e9), 2));
        assert_eq!(r.channel(1), x.iter().skip(1).step_by(2).copied().collect::<Vec<_>>());
        let a = &r.annotations[0];
        assert_eq!((a.label.as_deref(), r.annotated(a, 0)), (Some("QPSK"), vec![x[8], x[10]]));
        assert_eq!(r.full_scale(), Some(32768.0));
        // written annotations come back
        let base = dir.join("w");
        let ann = Annotation {
            sample_start: 2,
            sample_count: Some(3),
            label: Some("16-QAM".into()),
            ..Annotation::default()
        };
        write_sigmf_annotated(&base, &x, Datatype::CF32, 1e6, Some(915e6), None, std::slice::from_ref(&ann)).unwrap();
        let w = read_sigmf(&base).unwrap();
        assert_eq!((w.annotations, w.frequency), (vec![ann], Some(915e6)));
        std::fs::remove_dir_all(dir).ok();
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
    fn corrupt_archives_and_arrays_are_errors_not_panics() {
        // a tar header whose size field is out of range, or not a number
        for size in [&b"77777777777\0"[..], b"zz\0"] {
            let mut tar = vec![0u8; 1024];
            tar[..14].copy_from_slice(b"rec.sigmf-meta");
            tar[124..124 + size.len()].copy_from_slice(size);
            assert!(untar(&tar).is_err());
            let p = tmp("corrupt.sigmf");
            std::fs::write(&p, &tar).unwrap();
            assert!(read_sigmf(&p).is_err());
            std::fs::remove_file(p).ok();
        }
        // captures whose header bytes or sample starts overflow
        for (start, header) in [(0u64, u64::MAX), (u64::MAX, 0)] {
            let meta = json!({"global": {"core:datatype": "cf32_le"},
                "captures": [{"core:sample_start": 0, "core:header_bytes": header},
                             {"core:sample_start": start, "core:header_bytes": 0}]});
            let r = parse_sigmf(meta, &[0u8; 64], "corrupt");
            assert!(r.is_err() || r.is_ok_and(|s| s.samples.len() == 8), "{start} {header}");
        }
        // an NPY array whose shape overflows, or whose header is cut
        let npy = |header: &str| {
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            b.extend((header.len() as u16).to_le_bytes());
            b.extend(header.as_bytes());
            b.extend([0u8; 16]);
            b
        };
        let p = tmp("corrupt.npy");
        for header in [
            "{'descr': '<f8', 'fortran_order': False, 'shape': (4294967296, 4294967296, 4294967296), }",
            "{'descr': '<c16', 'fortran_order': False, 'shape': (1152921504606846976,), }",
            "{'descr': '<f8', 'fortran_order': False, 'shape': (-1,), }",
            "{'descr': '",
        ] {
            std::fs::write(&p, npy(header)).unwrap();
            assert!(read_npy(&p).is_err(), "{header}");
        }
        let mut cut = npy("{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }");
        cut[8] = 0xff;
        std::fs::write(&p, cut).unwrap();
        assert!(read_npy(&p).is_err());
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn sigmf_keeps_the_metadata() {
        let x = vec![C64::new(0.25, -0.5); 16];
        let base = tmp("rec");
        write_sigmf(&base, &x, Datatype::CF32, 80e9, Some(&json!({"comrs:symbol_rate": 32e9}))).unwrap();
        let r = read_sigmf(format!("{}.sigmf-data", base.display())).unwrap();
        assert_eq!(r.samples, x);
        assert_eq!(r.meta["global"]["comrs:symbol_rate"], json!(32e9));
        let Samples { samples: s, sample_rate: fs } =
            read_samples(format!("{}.sigmf-meta", base.display()), None, 0).unwrap();
        assert_eq!((s.len(), fs), (16, Some(80e9)));
        assert!(read_samples(&base, None, 0).is_err());
        assert!(read_samples(format!("{}.sigmf-meta", base.display()), None, 1).is_err());
    }
}

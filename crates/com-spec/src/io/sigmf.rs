//! SigMF recordings: the metadata, the captures, the annotations, the archives.

use super::raw::{decode_all, write_creating_dirs, write_raw};
use super::Datatype;
use com_core::{Error, Result, ensure};
use com_math::C64;
use serde_json::{Value as Json, json};
use std::path::{Path, PathBuf};

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
pub(super) fn untar(bytes: &[u8]) -> Result<Vec<(String, &[u8])>> {
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

pub(super) fn parse_sigmf(meta: Json, data: &[u8], name: &str) -> Result<SigMF> {
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

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

mod datatype;
mod npy;
mod raw;
mod sigmf;

pub use datatype::{Datatype, Scalar};
pub use npy::{Npy, NpyData, read_npy, write_npy, write_npy_i64};
pub use raw::{read_raw, write_raw};
pub use sigmf::{Annotation, Capture, SigMF, read_sigmf, write_sigmf, write_sigmf_annotated};

use com_core::{Error, Result};
use com_math::C64;
use std::path::Path;

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
    use super::raw::encode_all;
    use super::sigmf::{parse_sigmf, untar};
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;

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

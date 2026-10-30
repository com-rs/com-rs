//! The sample formats of SigMF ([`Datatype`]): the raw files and the SigMF recordings use them.

use com_core::{Error, Result};
use com_math::C64;

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

    pub(super) fn name(self) -> &'static str {
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

    pub(super) fn is_integer(self) -> bool {
        !matches!(self, Scalar::F32 | Scalar::F64)
    }

    /// $2^{b-1}$ for an integer of $b$ bits: the full scale, and the offset of the
    /// unsigned types.
    pub(super) fn half_range(self) -> f64 {
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

    pub(super) fn component(self, b: &[u8]) -> f64 {
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

    pub(super) fn decode(self, b: &[u8]) -> C64 {
        let k = self.scalar.bytes();
        C64::new(self.component(b), if self.complex { self.component(&b[k..]) } else { 0.0 })
    }

    pub(super) fn put(self, x: f64, out: &mut Vec<u8>) {
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

    pub(super) fn encode(self, v: C64, out: &mut Vec<u8>) {
        self.put(v.re, out);
        if self.complex {
            self.put(v.im, out);
        }
    }
}

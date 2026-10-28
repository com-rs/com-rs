//! The Raman gain spectrum of a fibre: the measured spectrum of standard single-mode
//! fibre (GNPy) or a table, scaled with the pump frequency.

use com_core::{Result, ensure, ensure_shape};

/// The Raman gain spectrum $g_{R}(\Delta f)$ of a fibre, as the product with the
/// effective area $g_{R} A_{\mathrm{eff}}$ (m/W), for a pump at a reference frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct RamanGain {
    offsets_hz: Vec<f64>,
    gain_m_w: Vec<f64>,
    reference_hz: f64,
}

/// Frequency offsets of the gain spectrum of standard single-mode fibre (THz).
const SSMF_OFFSETS_THZ: [f64; 90] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5,
    11.0, 11.5, 12.0, 12.5, 12.75, 13.0, 13.25, 13.5, 14.0, 14.5, 14.75, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0,
    18.25, 18.5, 18.75, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5,
    27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0,
    36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0,
];

/// $g_{R} A_{\mathrm{eff}}$ of standard single-mode fibre at the offsets above (m/W).
#[allow(clippy::unreadable_literal, reason = "the table of GNPy, digit for digit")]
const SSMF_GAIN_M_W: [f64; 90] = [
    0.0,
    8.524419934705497e-16,
    2.643567866245371e-15,
    4.410548410941305e-15,
    6.153422961291078e-15,
    7.484924703044943e-15,
    8.452060808349209e-15,
    9.101549322698156e-15,
    9.57837595158966e-15,
    1.0008642675474562e-14,
    1.0865773569905647e-14,
    1.1300776305865833e-14,
    1.2143238647099625e-14,
    1.3231065750676068e-14,
    1.4624900971525384e-14,
    1.6013330554840492e-14,
    1.7458119359310242e-14,
    1.9320241330434762e-14,
    2.1720395392873534e-14,
    2.4137337406734775e-14,
    2.628163218460466e-14,
    2.8041019963285974e-14,
    2.9723155447089933e-14,
    3.129353531005888e-14,
    3.251796163324624e-14,
    3.3198839487612773e-14,
    3.329527690685666e-14,
    3.313155691238456e-14,
    3.289013852154548e-14,
    3.2458917188506916e-14,
    3.060684277937575e-14,
    3.2660349473783173e-14,
    2.957419109657689e-14,
    2.518894321396672e-14,
    1.734560485857344e-14,
    9.902860761605233e-15,
    7.219176385099358e-15,
    6.079565990401311e-15,
    5.828373065963427e-15,
    7.20580801091692e-15,
    7.561924351387493e-15,
    7.621152352332206e-15,
    6.8859886780643254e-15,
    5.629181047471162e-15,
    3.679727598966185e-15,
    2.7555869742500355e-15,
    2.4810133942597675e-15,
    2.2160080532403624e-15,
    2.1440626024765557e-15,
    2.33873070799544e-15,
    2.557317929858713e-15,
    3.039839048226572e-15,
    4.8337165515610065e-15,
    5.4647431818257436e-15,
    5.229187813711269e-15,
    4.510768525811313e-15,
    3.3213473130607794e-15,
    2.2602577027996455e-15,
    1.969576495866441e-15,
    1.5179853954188527e-15,
    1.2953988551200156e-15,
    1.1304672156251838e-15,
    9.10004390675213e-16,
    8.432919922183503e-16,
    7.849224069008326e-16,
    7.827568196032024e-16,
    9.000514440646232e-16,
    1.3025926460013665e-15,
    1.5444108938497558e-15,
    1.8795594063060786e-15,
    1.7796130169921014e-15,
    1.5938159865046653e-15,
    1.1585522355108287e-15,
    8.507044444633358e-16,
    7.625404663756823e-16,
    8.14510750925789e-16,
    9.047944693473188e-16,
    9.636431901702084e-16,
    9.298633899602105e-16,
    8.349739503637023e-16,
    7.482901278066085e-16,
    6.240794767134268e-16,
    5.00652535687506e-16,
    3.553373263685851e-16,
    2.0344217706119682e-16,
    1.4267522642294203e-16,
    8.980016576743517e-17,
    2.9829068181832594e-17,
    1.4861959129014824e-17,
    7.404482113326137e-18,
];

impl RamanGain {
    /// The measured gain spectrum of standard single-mode fibre for a pump at 1454 nm
    /// (206.18 THz), the default of GNPy: peak $g_{R} A_{\mathrm{eff}} = 3.33 \cdot
    /// 10^{-14}$ m/W at 13.2 THz.
    pub fn ssmf() -> RamanGain {
        RamanGain {
            offsets_hz: SSMF_OFFSETS_THZ.iter().map(|f| f * 1e12).collect(),
            gain_m_w: SSMF_GAIN_M_W.to_vec(),
            reference_hz: 206.184_634_112_792e12,
        }
    }

    /// A gain spectrum: $g_{R} A_{\mathrm{eff}}$ (m/W) at increasing non-negative
    /// offsets (Hz), for a pump at `reference_hz`; linear between the offsets, zero
    /// beyond the last.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for tables of different lengths or fewer than two
    /// offsets, [`com_core::Error::Config`] for offsets that do not increase from a
    /// non-negative value, or a negative gain.
    pub fn new(offsets_hz: Vec<f64>, gain_m_w: Vec<f64>, reference_hz: f64) -> Result<RamanGain> {
        ensure_shape(offsets_hz.len() == gain_m_w.len() && offsets_hz.len() >= 2, || {
            format!(
                "the Raman gain needs at least two offsets and one gain per offset, got {} offsets and {} gains",
                offsets_hz.len(),
                gain_m_w.len()
            )
        })?;
        ensure(offsets_hz[0] >= 0.0 && offsets_hz.windows(2).all(|w| w[1] > w[0]), || {
            "the offsets of the Raman gain must increase from a non-negative value".into()
        })?;
        ensure(gain_m_w.iter().all(|g| *g >= 0.0), || "the Raman gain must be non-negative".into())?;
        ensure(reference_hz > 0.0, || "the reference frequency must be positive".into())?;
        Ok(RamanGain { offsets_hz, gain_m_w, reference_hz })
    }

    /// $g_{R} A_{\mathrm{eff}}$ (m/W) at the offset `df` (Hz) for a pump at the
    /// reference frequency.
    pub fn at(&self, df: f64) -> f64 {
        let (x, y) = (&self.offsets_hz, &self.gain_m_w);
        if df < x[0] || df > x[x.len() - 1] {
            return 0.0;
        }
        let i = x.partition_point(|v| *v <= df).clamp(1, x.len() - 1);
        let t = (df - x[i - 1]) / (x[i] - x[i - 1]);
        y[i - 1] + t * (y[i] - y[i - 1])
    }

    /// The gain efficiency $C$ (1/(W·km)) of a wave at `f_s` from a pump at `f_p >
    /// f_s`, for the effective area `aeff_m2`.
    pub fn efficiency(&self, f_s: f64, f_p: f64, aeff_m2: f64) -> f64 {
        self.at(f_p - f_s) * f_p / self.reference_hz / aeff_m2 * 1e3
    }

    /// The offset of the peak of the gain (Hz) and its $g_{R} A_{\mathrm{eff}}$ (m/W).
    pub fn peak(&self) -> (f64, f64) {
        self.offsets_hz
            .iter()
            .zip(&self.gain_m_w)
            .fold((0.0, 0.0), |(fo, go), (f, g)| if *g > go { (*f, *g) } else { (fo, go) })
    }
}

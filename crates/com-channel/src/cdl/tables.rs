//! The tables of 3GPP TR 38.901: the ray offsets of Table 7.5-3 and the CDL profiles
//! of Tables 7.7.1-1 to 7.7.1-5.

use com_core::prelude::*;
use com_math::{db2lin, lin2db};

/// Rays per cluster, $M$.
pub const RAYS_PER_CLUSTER: usize = 20;

/// Ray offset angles $\alpha_{m}$ of TR 38.901 Table 7.5-3 (degrees, for an RMS
/// angular spread of 1°), rays $m = 1, \ldots, 20$ in the order of the table.
pub const RAY_OFFSETS: [f64; RAYS_PER_CLUSTER] = [
    0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844,
    -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551,
];

/// One cluster of a CDL table: normalised delay, power and the four angles of its
/// centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cluster {
    /// Normalised delay (unit RMS delay spread).
    pub delay: f64,
    /// Power (dB), as printed in the table (the sum over the clusters is not 1).
    pub power_db: f64,
    /// Azimuth of departure (degrees).
    pub aod: f64,
    /// Azimuth of arrival (degrees).
    pub aoa: f64,
    /// Zenith of departure (degrees).
    pub zod: f64,
    /// Zenith of arrival (degrees).
    pub zoa: f64,
}

pub(super) const fn cl(delay: f64, power_db: f64, aod: f64, aoa: f64, zod: f64, zoa: f64) -> Cluster {
    Cluster { delay, power_db, aod, aoa, zod, zoa }
}

/// Per-cluster angular spreads of the rays and cross-polarisation power ratio of a
/// profile: the last rows of Tables 7.7.1-1 to 7.7.1-5.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spreads {
    /// $c_{\mathrm{ASD}}$ (degrees).
    pub asd: f64,
    /// $c_{\mathrm{ASA}}$ (degrees).
    pub asa: f64,
    /// $c_{\mathrm{ZSD}}$ (degrees).
    pub zsd: f64,
    /// $c_{\mathrm{ZSA}}$ (degrees).
    pub zsa: f64,
    /// XPR (dB).
    pub xpr_db: f64,
}

/// The clusters of a channel, its specular component and the spreads of its rays: a
/// profile as data ([`Profile`] gives those of the tables), for [`super::Cdl::custom`].
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterTable {
    /// The clusters (in the units of the tables, powers in dB of any normalisation).
    pub clusters: Vec<Cluster>,
    /// The specular component added to the first cluster, if any.
    pub los: Option<Cluster>,
    /// The per-cluster spreads.
    pub spreads: Spreads,
}

impl From<Profile> for ClusterTable {
    /// The table of a profile: [`Profile::table`], [`Profile::los`] and
    /// [`Profile::spreads`].
    fn from(p: Profile) -> ClusterTable {
        ClusterTable { clusters: p.table().to_vec(), los: p.los(), spreads: p.spreads() }
    }
}

/// The CDL profiles of TR 38.901 Section 7.7.1.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// CDL-A, Table 7.7.1-1 (NLOS, 23 clusters).
    A,
    /// CDL-B, Table 7.7.1-2 (NLOS, 23 clusters).
    B,
    /// CDL-C, Table 7.7.1-3 (NLOS, 24 clusters).
    C,
    /// CDL-D, Table 7.7.1-4 (LOS, 13 clusters, K-factor 13.3 dB).
    D,
    /// CDL-E, Table 7.7.1-5 (LOS, 14 clusters, K-factor 22 dB).
    E,
}

pub(super) static CDL_A: [Cluster; 23] = [
    cl(0.0000, -13.4, -178.1, 51.3, 50.2, 125.4),
    cl(0.3819, 0.0, -4.2, -152.7, 93.2, 91.3),
    cl(0.4025, -2.2, -4.2, -152.7, 93.2, 91.3),
    cl(0.5868, -4.0, -4.2, -152.7, 93.2, 91.3),
    cl(0.4610, -6.0, 90.2, 76.6, 122.0, 94.2),
    cl(0.5375, -8.2, 90.2, 76.6, 122.0, 94.2),
    cl(0.6708, -9.9, 90.2, 76.6, 122.0, 94.2),
    cl(0.5750, -10.5, 121.5, -1.8, 150.2, 47.1),
    cl(0.7618, -7.5, -81.7, -41.9, 55.2, 56.0),
    cl(1.5375, -15.9, 158.4, 94.2, 26.4, 30.1),
    cl(1.8978, -6.6, -83.0, 51.9, 126.4, 58.8),
    cl(2.2242, -16.7, 134.8, -115.9, 171.6, 26.0),
    cl(2.1718, -12.4, -153.0, 26.6, 151.4, 49.2),
    cl(2.4942, -15.2, -172.0, 76.6, 157.2, 143.1),
    cl(2.5119, -10.8, -129.9, -7.0, 47.2, 117.4),
    cl(3.0582, -11.3, -136.0, -23.0, 40.4, 122.7),
    cl(4.0810, -12.7, 165.4, -47.2, 43.3, 123.2),
    cl(4.4579, -16.2, 148.4, 110.4, 161.8, 32.6),
    cl(4.5695, -18.3, 132.7, 144.5, 10.8, 27.2),
    cl(4.7966, -18.9, -118.6, 155.3, 16.7, 15.2),
    cl(5.0066, -16.6, -154.1, 102.0, 171.7, 146.0),
    cl(5.3043, -19.9, 126.5, -151.8, 22.0, 150.7),
    cl(9.6586, -29.7, -56.2, 55.2, 144.9, 156.1),
];

pub(super) static CDL_B: [Cluster; 23] = [
    cl(0.0000, 0.0, 9.3, -173.3, 105.8, 78.9),
    cl(0.1072, -2.2, 9.3, -173.3, 105.8, 78.9),
    cl(0.2155, -4.0, 9.3, -173.3, 105.8, 78.9),
    cl(0.2095, -3.2, -34.1, 125.5, 115.3, 63.3),
    cl(0.2870, -9.8, -65.4, -88.0, 119.3, 59.9),
    cl(0.2986, -1.2, -11.4, 155.1, 103.2, 67.5),
    cl(0.3752, -3.4, -11.4, 155.1, 103.2, 67.5),
    cl(0.5055, -5.2, -11.4, 155.1, 103.2, 67.5),
    cl(0.3681, -7.6, -67.2, -89.8, 118.2, 82.6),
    cl(0.3697, -3.0, 52.5, 132.1, 102.0, 66.3),
    cl(0.5700, -8.9, -72.0, -83.6, 100.4, 61.6),
    cl(0.5283, -9.0, 74.3, 95.3, 98.3, 58.0),
    cl(1.1021, -4.8, -52.2, 103.7, 103.4, 78.2),
    cl(1.2756, -5.7, -50.5, -87.8, 102.5, 82.0),
    cl(1.5474, -7.5, 61.4, -92.5, 101.4, 62.4),
    cl(1.7842, -1.9, 30.6, -139.1, 103.0, 78.0),
    cl(2.0169, -7.6, -72.5, -90.6, 100.0, 60.9),
    cl(2.8294, -12.2, -90.6, 58.6, 115.2, 82.9),
    cl(3.0219, -9.8, -77.6, -79.0, 100.5, 60.8),
    cl(3.6187, -11.4, -82.6, 65.8, 119.6, 57.3),
    cl(4.1067, -14.9, -103.6, 52.7, 118.7, 59.9),
    cl(4.2790, -9.2, 75.6, 88.7, 117.8, 60.1),
    cl(4.7834, -11.3, -77.6, -60.4, 115.4, 62.3),
];

#[allow(clippy::approx_constant, reason = "the delay 0.6366 of cluster 6 is an entry of the table, not 2/π")]
pub(super) static CDL_C: [Cluster; 24] = [
    cl(0.0, -4.4, -46.6, -101.0, 97.2, 87.6),
    cl(0.2099, -1.2, -22.8, 120.0, 98.6, 72.1),
    cl(0.2219, -3.5, -22.8, 120.0, 98.6, 72.1),
    cl(0.2329, -5.2, -22.8, 120.0, 98.6, 72.1),
    cl(0.2176, -2.5, -40.7, -127.5, 100.6, 70.1),
    cl(0.6366, 0.0, 0.3, 170.4, 99.2, 75.3),
    cl(0.6448, -2.2, 0.3, 170.4, 99.2, 75.3),
    cl(0.6560, -3.9, 0.3, 170.4, 99.2, 75.3),
    cl(0.6584, -7.4, 73.1, 55.4, 105.2, 67.4),
    cl(0.7935, -7.1, -64.5, 66.5, 95.3, 63.8),
    cl(0.8213, -10.7, 80.2, -48.1, 106.1, 71.4),
    cl(0.9336, -11.1, -97.1, 46.9, 93.5, 60.5),
    cl(1.2285, -5.1, -55.3, 68.1, 103.7, 90.6),
    cl(1.3083, -6.8, -64.3, -68.7, 104.2, 60.1),
    cl(2.1704, -8.7, -78.5, 81.5, 93.0, 61.0),
    cl(2.7105, -13.2, 102.7, 30.7, 104.2, 100.7),
    cl(4.2589, -13.9, 99.2, -16.4, 94.9, 62.3),
    cl(4.6003, -13.9, 88.8, 3.8, 93.1, 66.7),
    cl(5.4902, -15.8, -101.9, -13.7, 92.2, 52.9),
    cl(5.6077, -17.1, 92.2, 9.7, 106.7, 61.8),
    cl(6.3065, -16.0, 93.3, 5.6, 93.0, 51.9),
    cl(6.6374, -15.7, 106.6, 0.7, 92.9, 61.7),
    cl(7.0427, -21.6, 119.5, -21.9, 105.2, 58.0),
    cl(8.6523, -22.8, -123.8, 33.6, 107.8, 57.0),
];

/// Cluster 1 of CDL-D: the specular (LOS) row, then the Laplacian row in `CDL_D`.
pub(super) static CDL_D_LOS: Cluster = cl(0.0, -0.2, 0.0, -180.0, 98.5, 81.5);

pub(super) static CDL_D: [Cluster; 13] = [
    cl(0.0, -13.5, 0.0, -180.0, 98.5, 81.5),
    cl(0.035, -18.8, 89.2, 89.2, 85.5, 86.9),
    cl(0.612, -21.0, 89.2, 89.2, 85.5, 86.9),
    cl(1.363, -22.8, 89.2, 89.2, 85.5, 86.9),
    cl(1.405, -17.9, 13.0, 163.0, 97.5, 79.4),
    cl(1.804, -20.1, 13.0, 163.0, 97.5, 79.4),
    cl(2.596, -21.9, 13.0, 163.0, 97.5, 79.4),
    cl(1.775, -22.9, 34.6, -137.0, 98.5, 78.3),
    cl(4.042, -27.8, -64.5, 74.5, 88.4, 82.0),
    cl(7.937, -23.6, -32.9, 45.3, 91.3, 79.2),
    cl(9.424, -24.8, 52.6, -70.0, 103.5, 66.0),
    cl(9.708, -30.0, -48.3, 72.8, 77.3, 80.7),
    cl(12.525, -27.7, -44.4, -78.7, 103.8, 71.3),
];

/// Cluster 1 of CDL-E: the specular (LOS) row, then the Laplacian row in `CDL_E`.
pub(super) static CDL_E_LOS: Cluster = cl(0.0, -0.03, 0.0, -180.0, 99.6, 80.4);

pub(super) static CDL_E: [Cluster; 14] = [
    cl(0.0, -22.03, 0.0, -180.0, 99.6, 80.4),
    cl(0.5133, -15.8, 57.5, 18.2, 104.2, 80.4),
    cl(0.5440, -18.1, 57.5, 18.2, 104.2, 80.4),
    cl(0.5630, -19.8, 57.5, 18.2, 104.2, 80.4),
    cl(0.5440, -22.9, -20.1, 101.8, 99.4, 80.8),
    cl(0.7112, -22.4, 16.2, 112.9, 100.8, 86.3),
    cl(1.9092, -18.6, 9.3, -155.5, 98.8, 82.7),
    cl(1.9293, -20.8, 9.3, -155.5, 98.8, 82.7),
    cl(1.9589, -22.6, 9.3, -155.5, 98.8, 82.7),
    cl(2.6426, -22.3, 19.0, -143.3, 100.8, 82.9),
    cl(3.7136, -25.6, 32.7, -94.7, 96.4, 88.0),
    cl(5.4524, -20.2, 0.5, 147.0, 98.9, 81.0),
    cl(12.0034, -29.8, 55.9, -56.2, 95.6, 88.6),
    cl(20.6419, -29.2, 57.6, -72.4, 97.1, 86.4),
];

impl std::str::FromStr for Profile {
    type Err = Error;

    /// The profile of a name: `cdl-a` to `cdl-e` (or `a` to `e`), case-insensitive.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    ///
    /// # Examples
    /// ```
    /// use com_channel::cdl::Profile;
    /// assert_eq!("CDL-D".parse::<Profile>()?, Profile::D);
    /// assert!("cdl-f".parse::<Profile>().is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    fn from_str(name: &str) -> Result<Profile> {
        let lower = name.to_ascii_lowercase();
        Ok(match lower.strip_prefix("cdl-").or_else(|| lower.strip_prefix("cdl")).unwrap_or(&lower) {
            "a" => Profile::A,
            "b" => Profile::B,
            "c" => Profile::C,
            "d" => Profile::D,
            "e" => Profile::E,
            _ => return Err(Error::Config(format!("unknown CDL profile {name:?}: cdl-a, cdl-b, cdl-c, cdl-d, cdl-e"))),
        })
    }
}

impl Profile {
    /// The clusters of the table, verbatim, in the order of the table (for CDL-D and
    /// CDL-E, cluster 1 is its Laplacian row; the specular row is [`Profile::los`]).
    pub fn table(self) -> &'static [Cluster] {
        match self {
            Profile::A => &CDL_A,
            Profile::B => &CDL_B,
            Profile::C => &CDL_C,
            Profile::D => &CDL_D,
            Profile::E => &CDL_E,
        }
    }

    /// The specular (LOS) row of cluster 1, for CDL-D and CDL-E.
    pub fn los(self) -> Option<Cluster> {
        match self {
            Profile::D => Some(CDL_D_LOS),
            Profile::E => Some(CDL_E_LOS),
            Profile::A | Profile::B | Profile::C => None,
        }
    }

    /// Number of clusters (23, 23, 24, 13, 14).
    pub fn n_clusters(self) -> usize {
        self.table().len()
    }

    /// The angular spreads $c_{\mathrm{ASD}}, c_{\mathrm{ASA}}, c_{\mathrm{ZSD}},
    /// c_{\mathrm{ZSA}}$ and the XPR of the profile (last rows of the table).
    pub fn spreads(self) -> Spreads {
        match self {
            Profile::A => Spreads { asd: 5.0, asa: 11.0, zsd: 3.0, zsa: 3.0, xpr_db: 10.0 },
            Profile::B => Spreads { asd: 10.0, asa: 22.0, zsd: 3.0, zsa: 7.0, xpr_db: 8.0 },
            Profile::C => Spreads { asd: 2.0, asa: 15.0, zsd: 3.0, zsa: 7.0, xpr_db: 7.0 },
            Profile::D => Spreads { asd: 5.0, asa: 8.0, zsd: 3.0, zsa: 3.0, xpr_db: 11.0 },
            Profile::E => Spreads { asd: 5.0, asa: 11.0, zsd: 3.0, zsa: 7.0, xpr_db: 8.0 },
        }
    }

    /// The Rician K-factor of the first cluster (dB), the specular row minus the
    /// Laplacian row: 13.3 dB for CDL-D, 22 dB for CDL-E, `None` without line of sight.
    pub fn k_factor_db(self) -> Option<f64> {
        self.los().map(|s| s.power_db - self.table()[0].power_db)
    }

    /// The power-delay profile for the desired RMS delay spread `delay_spread_s` (s):
    /// one tap per cluster, `(delay in s, power in dB)`, the powers normalised to a
    /// unit sum and the specular component of CDL-D/E added to its cluster. It is the
    /// tap list [`crate::multipath::Tdl::new`] takes, for a CDL profile with
    /// Clarke/Jakes fading instead of the ray geometry.
    ///
    /// # Examples
    /// ```
    /// use com_channel::cdl::Profile;
    /// use com_channel::multipath::rms_delay_spread;
    /// let pdp = Profile::A.pdp(100e-9);
    /// assert_eq!(pdp.len(), 23);
    /// assert!((rms_delay_spread(&pdp) - 100e-9).abs() < 2e-9);
    /// ```
    pub fn pdp(self, delay_spread_s: f64) -> Vec<(f64, f64)> {
        let powers = self.cluster_powers();
        self.table().iter().zip(&powers).map(|(c, &p)| (c.delay * delay_spread_s, lin2db(p))).collect()
    }

    /// The normalised cluster powers $P_{n}$ (unit sum, the specular component of
    /// CDL-D/E counted in its cluster).
    pub fn cluster_powers(self) -> Vec<f64> {
        let mut p: Vec<f64> = self.table().iter().map(|c| db2lin(c.power_db)).collect();
        if let Some(s) = self.los() {
            p[0] += db2lin(s.power_db);
        }
        let norm = self.normalisation();
        p.iter().map(|v| v * norm).collect()
    }

    /// The normalisation factor $1 / \sum_{n} 10^{P_{n, \mathrm{dB}} / 10}$ of the table
    /// (the specular row included).
    pub(super) fn normalisation(self) -> f64 {
        let mut tot: f64 = self.table().iter().map(|c| db2lin(c.power_db)).sum();
        if let Some(s) = self.los() {
            tot += db2lin(s.power_db);
        }
        1.0 / tot
    }
}

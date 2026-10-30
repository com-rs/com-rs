//! Achievable information rates (`comrs::math::gauss_hermite`,
//! `comrs::metrics::information`) against A. Alvarado, T. Fehenberger, B. Chen and F. M. J.
//! Willems, "Achievable information rates for fiber optics: applications and
//! computations," *J. Lightw. Technol.* 36(2), 424–439, 2018:
//!
//! - Table III: the nodes and weights of the 10-point Gauss–Hermite rule of the
//!   quadratures (44)–(45) (the nodes rounded to four decimals, the weights truncated to
//!   nine: 0.0013436457 is printed 0.001343645, 0.6108626337 is printed 0.610862633);
//! - Remarks 4–5 and eq. (35): a mismatched decoding metric, here the symbol posteriors of a
//!   receiver whose noise variance is four times too small (16-QAM, AWGN at 12 dB), loses
//!   rate unless it is scaled by the optimal $s$, which recovers the mutual information
//!   of the matched receiver (`posterior_rates`: `air` against `air_opt`), and likewise
//!   for the bit-wise metric, the GMI of eq. (35) (`air_bmd` against `air_bmd_opt`).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::math::{db2lin, gauss_hermite, power};
use comrs::metrics::information::{information_rates, posterior_rates};
use comrs::prelude::*;

#[test]
fn alvarado_2018_table_iii_gauss_hermite_nodes() {
    let nodes = [-3.4362, -2.5327, -1.7567, -1.0366, -0.3429, 0.3429, 1.0366, 1.7567, 2.5327, 3.4362];
    let weights = [
        0.000_007_640,
        0.001_343_645,
        0.033_874_394,
        0.240_138_611,
        0.610_862_633,
        0.610_862_633,
        0.240_138_611,
        0.033_874_394,
        0.001_343_645,
        0.000_007_640,
    ];
    let (t, w) = gauss_hermite(10);
    let mut pairs: Vec<(f64, f64)> = t.into_iter().zip(w).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    for ((x, a), (xt, at)) in pairs.iter().zip(nodes.iter().zip(&weights)) {
        assert!((x - xt).abs() < 5e-5 && (a - at).abs() < 1e-9, "{x} {a} vs {xt} {at}");
    }
}

#[test]
fn alvarado_2018_remark_5_mismatched_metric_needs_its_scaling() {
    let c = Constellation::qam(16).unwrap().normalized();
    let mut rng = Rng::new(2018);
    let (idx, x) = comrs::mapping::source::symbols(&c, 40_000, None, &mut rng).unwrap();
    let s2 = power(&x) / db2lin(12.0);
    let noise = rng.complex_noise(x.len(), s2);
    let z: Vec<C64> = x.iter().zip(&noise).map(|(a, b)| a + b).collect();
    let mi = information_rates(&z, &idx, &c, None).unwrap().mi;
    let matched = posterior_rates(&z, &idx, &c, s2, None).unwrap();
    let mismatched = posterior_rates(&z, &idx, &c, s2 / 4.0, None).unwrap();
    assert!((matched.air - mi).abs() < 0.01, "{} vs {mi}", matched.air);
    assert!(mismatched.air < mi - 0.05, "{} vs {mi}", mismatched.air);
    assert!((mismatched.air_opt - mi).abs() < 0.01, "{} vs {mi}", mismatched.air_opt);
    // the bit-wise metric: eq. (36) for the matched LLRs, eq. (35) for the mismatched ones
    assert!(mismatched.air_bmd < matched.air_bmd - 0.05, "{} vs {}", mismatched.air_bmd, matched.air_bmd);
    assert!(
        (mismatched.air_bmd_opt - matched.air_bmd).abs() < 0.01,
        "{} vs {}",
        mismatched.air_bmd_opt,
        matched.air_bmd
    );
}

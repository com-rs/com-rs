//! MIMO against the closed forms of maximal-ratio combining over i.i.d. Rayleigh
//! fading (Proakis & Salehi, *Digital Communications*, eq. 13.4-15,
//! `comrs::mapping::theory::ber_rayleigh_mrc`):
//!
//! $$
//! P_{b} = \left(\frac{1 - \mu}{2}\right)^{L} \sum_{k=0}^{L-1} \binom{L - 1 + k}{k}
//!         \left(\frac{1 + \mu}{2}\right)^{k}, \qquad \mu = \sqrt{\frac{\gamma}{1 + \gamma}}
//! $$
//!
//! - zero forcing over $n_{t} \times n_{r}$ has $L = n_{r} - n_{t} + 1$ branches of mean SNR
//!   $E_{b}/N_{0}$;
//! - Alamouti over $2 \times n_{r}$ has $L = 2 n_{r}$ branches of mean SNR $E_{b}/N_{0}/2$.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::{MIN_ERRORS, within_three_sigma};
use comrs::channel::fading::Mimo;
use comrs::mapping::theory::ber_rayleigh_mrc;
use comrs::math::bisect;
use comrs::math::linalg::{CMat, matvec};
use comrs::mimo::detection::{Linear, MimoDetector};
use comrs::mimo::stbc::Ostbc;
use comrs::prelude::*;
use ndarray::Array2;

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn zero_forcing_over_rayleigh_fading() {
    let c = Constellation::psk(2).unwrap();
    for (nr, nt, ebn0_db) in [(2usize, 2usize, 12.0), (4, 2, 6.0)] {
        let s2 = 1.0 / db2lin(ebn0_db);
        let ch = Mimo::rayleigh(nr, nt);
        let det = Linear::zf(c.clone());
        let (mut errors, mut total, mut batch) = (0u64, 0u64, 0u64);
        while errors < MIN_ERRORS {
            let mut rng = batch_rng(31, batch);
            for _ in 0..10_000 {
                let h: CMat = ch.draw(&mut rng).unwrap();
                let idx: Vec<usize> = (0..nt).map(|_| rng.below(2)).collect();
                let x: Vec<C64> = idx.iter().map(|&i| c.points()[i]).collect();
                let y: Vec<C64> =
                    matvec(&h.view(), &x).into_iter().map(|v| v + rng.complex_gaussian() * s2.sqrt()).collect();
                let d = det.detect(&h.view(), &y, s2).unwrap();
                errors += d.iter().zip(&idx).filter(|(a, b)| a != b).count() as u64;
                total += nt as u64;
            }
            batch += 1;
        }
        let theory = ber_rayleigh_mrc(db2lin(ebn0_db), nr - nt + 1).unwrap();
        assert!(within_three_sigma(errors, total, theory), "{nt}x{nr}: {errors}/{total} vs {theory}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn alamouti_over_rayleigh_fading() {
    let code = Ostbc::alamouti();
    for (nr, ebn0_db) in [(1usize, 12.0), (2, 6.0)] {
        let g = db2lin(ebn0_db);
        let ch = Mimo::rayleigh(nr, 2);
        let (mut errors, mut total, mut batch) = (0u64, 0u64, 0u64);
        while errors < MIN_ERRORS {
            let mut rng = batch_rng(32, batch);
            for _ in 0..5_000 {
                let bits = [rng.bit(), rng.bit()];
                let s: Vec<C64> = bits.iter().map(|&b| C64::new(1.0 - 2.0 * f64::from(b), 0.0)).collect();
                let x = code.encode(&s).unwrap();
                let h = ch.draw(&mut rng).unwrap();
                let y: Array2<C64> = h.dot(&x).mapv(|v| v + rng.complex_gaussian() * (1.0 / g).sqrt());
                let (est, _) = code.decode(&h.view(), &y.view(), 1.0 / g).unwrap();
                errors += est.iter().zip(bits).filter(|(e, b)| (e.re < 0.0) != (*b == 1)).count() as u64;
                total += 2;
            }
            batch += 1;
        }
        let theory = ber_rayleigh_mrc(g / 2.0, 2 * nr).unwrap();
        assert!(within_three_sigma(errors, total, theory), "2x{nr}: {errors}/{total} vs {theory}");
    }
}

/// The bit error rate of QPSK spatial multiplexing over $n_{t} \times n_{r}$ i.i.d.
/// Rayleigh fading with the hard-decision detector `det`, at every SNR $\rho = 1/N_{0}$
/// of `snrs_db` (the SNR of a stream), until `errors` bit errors.
fn ber_curve(
    det: &(dyn MimoDetector + Sync),
    nt: usize,
    nr: usize,
    snrs_db: &[f64],
    errors: u64,
    max_trials: u64,
) -> Vec<f64> {
    use comrs::sim::scenario::from_fn;
    let c = Constellation::qam(4).unwrap().normalized();
    let ch = Mimo::rayleigh(nr, nt);
    let scenario = from_fn("yang-hanzo-2015-table-x", |cond, rng, rec| {
        let n0 = 1.0 / db2lin(cond.f64("snr_db")?);
        let (mut wrong, mut bits) = (0u64, 0u64);
        for _ in 0..500 {
            let h: CMat = ch.draw(rng)?;
            let idx: Vec<usize> = (0..nt).map(|_| rng.below(4)).collect();
            let x: Vec<C64> = idx.iter().map(|&i| c.points()[i]).collect();
            let y: Vec<C64> =
                matvec(&h.view(), &x).into_iter().map(|v| v + rng.complex_gaussian() * n0.sqrt()).collect();
            let d = det.detect(&h.view(), &y, n0)?;
            // the labels are the Gray bits of the points
            wrong += d.iter().zip(&idx).map(|(a, b)| u64::from((a ^ b).count_ones())).sum::<u64>();
            bits += 2 * nt as u64;
        }
        rec.errors("ber", wrong, bits);
        Ok(())
    });
    let grid = Grid::new().axis("snr_db", snrs_db.to_vec());
    let res = Runner::new()
        .seed(33)
        .batch(48)
        .stop(Stop::min_errors("ber", errors).or(Stop::max_trials(max_trials)))
        .run(&scenario, &grid)
        .unwrap();
    res.iter().map(|r| r.metric("ber").unwrap().value()).collect()
}

/// The least-squares slope $-\mathrm{d} \log_{10} P_{b} / \mathrm{d} \log_{10} \rho$.
fn diversity_slope(snrs_db: &[f64], ber: &[f64]) -> f64 {
    let n = snrs_db.len() as f64;
    let xs: Vec<f64> = snrs_db.iter().map(|s| s / 10.0).collect();
    let ys: Vec<f64> = ber.iter().map(|b| b.log10()).collect();
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    -sxy / sxx
}

/// The slope that a curve of diversity order `order` has between the bit error rates
/// `ber`: the slope of the error rate of maximal-ratio combining over `order` branches
/// (`ber_rayleigh_mrc`, the curve of order `order` with a closed form) between the
/// mean SNRs at which it takes the same values.
fn reference_slope(order: usize, ber: &[f64]) -> f64 {
    // ber_rayleigh_mrc decreases with the SNR: bisection on the SNR in dB
    let snr_db_at = |p: f64| bisect(|s| ber_rayleigh_mrc(db2lin(s), order).unwrap() > p, -20.0, 80.0, 100);
    let snrs: Vec<f64> = ber.iter().map(|&p| snr_db_at(p)).collect();
    diversity_slope(&snrs, ber)
}

/// The receive diversity orders of the hard-decision detectors of uncoded spatial
/// multiplexing over i.i.d. Rayleigh fading, $n_{r} \ge n_{t}$: S. Yang and L. Hanzo,
/// "Fifty years of MIMO detection: the road to large-scale MIMOs," *IEEE Commun.
/// Surveys Tuts.* 17(4), 1941–1988, 2015 (arXiv:1507.05138), Table X: $n_{r}$ for ML
/// and sphere decoding, $n_{r} - n_{t} + 1$ for linear ZF and MMSE and for
/// ZF/MMSE-SIC and -OSIC, from $n_{r} - n_{t} + 1$ to $n_{r}$ for K-best depending on
/// $K$.
///
/// The slope $-\Delta \log_{10} P_{b} / \Delta \log_{10} \rho$ of the bit error rate of
/// QPSK on $2 \times 2$ and $2 \times 4$ links is fitted by least squares on three SNRs
/// (1000 bit errors at every point, a statistical standard error of about 0.03 on the
/// slope). A slope measured at a finite SNR is below the order it tends to, the more so
/// as the order is large (the $2 \times 4$ ML curve has a slope of 3.4 at $10^{-5}$), so
/// it is compared with the slope that a curve of the order of the table has between the
/// same error rates: the error rate of maximal-ratio combining over that many branches
/// (Proakis & Salehi, eq. 13.4-15), which has the exact order. A detector passes when
/// its slope is within ±0.3 of that of its order (of the range, for K-best); the
/// reference slopes of two consecutive orders differ by about 1, so the test tells the
/// orders apart.
///
/// One exception: on $2 \times 4$, the ordering of ZF- and MMSE-OSIC (V-BLAST) gives a
/// slope between those of orders 3 and 4 at every SNR within reach (3.25 and 3.38 from
/// $10^{-4}$ to $10^{-6}$, still 3.1 to 3.4 from $10^{-5}$ to $10^{-8}$), while Table X
/// gives the asymptotic order $n_{r} - n_{t} + 1 = 3$; the test only bounds these slopes
/// by the orders 3 to 4. On $2 \times 2$ the ordered SIC has the order 1 of the table.
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn diversity_orders_follow_yang_hanzo_2015_table_x() {
    use comrs::mimo::detection::{KBest, Ml, Sic, Sphere};
    /// A detector, the orders of Table X, the SNRs of the fit.
    type Entry = (&'static str, Box<dyn MimoDetector + Sync>, (usize, usize), [f64; 3]);
    const TOLERANCE: f64 = 0.3;
    let c = Constellation::qam(4).unwrap().normalized();
    let mut failed = Vec::new();
    for (nt, nr) in [(2usize, 2usize), (2, 4)] {
        let (full, low) = (nr, nr - nt + 1);
        // the ordered SIC on 2 × 4: see the documentation
        let osic = if nr == 2 { low } else { full };
        // three SNRs where the bit error rate falls from about 1e-3 to 1e-5
        let (high, linear) =
            if nr == 2 { ([15.0, 20.0, 25.0], [30.0, 35.0, 40.0]) } else { ([7.5, 10.0, 12.5], [10.0, 12.5, 15.0]) };
        let unordered_zf = Sic { ordered: false, ..Sic::zf(c.clone()) };
        let detectors: Vec<Entry> = vec![
            ("ML", Box::new(Ml::new(c.clone())), (full, full), high),
            ("sphere", Box::new(Sphere::new(c.clone())), (full, full), high),
            ("ZF", Box::new(Linear::zf(c.clone())), (low, low), linear),
            ("MMSE", Box::new(Linear::mmse(c.clone())), (low, low), linear),
            ("ZF-SIC", Box::new(unordered_zf), (low, low), linear),
            ("ZF-OSIC", Box::new(Sic::zf(c.clone())), (low, osic), linear),
            ("MMSE-OSIC", Box::new(Sic::mmse(c.clone())), (low, osic), linear),
            ("K-best K=1", Box::new(KBest::new(c.clone(), 1)), (low, low), linear),
            ("K-best K=2", Box::new(KBest::new(c.clone(), 2)), (low, full), linear),
            ("K-best K=4", Box::new(KBest::new(c.clone(), 4)), (full, full), high),
        ];
        for (name, det, (lo, hi), snrs) in detectors {
            let ber = ber_curve(det.as_ref(), nt, nr, &snrs, 1000, 1_000_000);
            let slope = diversity_slope(&snrs, &ber);
            let (ref_lo, ref_hi) = (reference_slope(lo, &ber), reference_slope(hi, &ber));
            let shown: Vec<String> = ber.iter().map(|b| format!("{b:.2e}")).collect();
            println!(
                "{nt}x{nr} {name:>10}: slope {slope:.2}, order {lo}..{hi} of Table X (slopes {ref_lo:.2}..{ref_hi:.2}); \
                 BER {} at {snrs:?} dB",
                shown.join(", ")
            );
            if slope < ref_lo - TOLERANCE || slope > ref_hi + TOLERANCE {
                failed.push(format!("{nt}x{nr} {name}: slope {slope:.2}, order {lo}..{hi} ({ref_lo:.2}..{ref_hi:.2})"));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

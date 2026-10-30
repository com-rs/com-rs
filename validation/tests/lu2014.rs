//! Linear detection in massive MIMO (`comrs::mimo::detection::Linear`,
//! `comrs::mimo::capacity`) against L. Lu, G. Y. Li, A. L. Swindlehurst, A. Ashikhmin and
//! R. Zhang, "An overview of massive MIMO: benefits and challenges," *IEEE J. Sel. Topics
//! Signal Process.* 8(5), 742–758, 2014, Sec. II-B: with favourable propagation, eq. (12),
//! the uplink sum capacity of $K$ users tends to $\sum_{k} \log_{2}(1 + N \rho_{u} d_{k})$,
//! eq. (13), and the linear detectors of Sec. III-B reach it as the number $N$ of
//! base-station antennas grows. Here $K = 4$ users of equal large-scale fading
//! ($d_{k} = 1$), $\rho_{u} = 1$, i.i.d. Rayleigh fading: the ergodic sum capacity
//! $\log_{2}\det(I + \rho_{u} H^{H} H)$ and the sum rates $\sum_{k} \log_{2}(1 +
//! \mathrm{SINR}_{k})$ of ZF and MMSE detection, the SINR being the inverse of the noise
//! variance of the unbiased estimates of [`Linear::estimate`]. The test measures; the
//! limits, their tolerance and the reason for it are in the card
//! `validation/cards/lu2014_eq13.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::fading::Mimo;
use comrs::math::linalg::hermitian;
use comrs::mimo::capacity::equal_power;
use comrs::mimo::detection::Linear;
use comrs::prelude::*;
use serde_json::json;

const USERS: usize = 4;
const RHO: f64 = 1.0;

#[test]
fn lu_2014_eq_13() {
    let mut card = Card::load("lu2014_eq13").unwrap();
    let mut rng = Rng::new(2014);
    let qpsk = Constellation::psk(4).unwrap().normalized();
    let (zf, mmse) = (Linear::zf(qpsk.clone()), Linear::mmse(qpsk));
    let trials = 2000;
    for antennas in [16, 64, 256] {
        let mimo = Mimo::rayleigh(antennas, USERS);
        let y = vec![C64::new(0.0, 0.0); antennas];
        let (mut capacity, mut rate_zf, mut rate_mmse) = (0.0, 0.0, 0.0);
        for _ in 0..trials {
            let h = mimo.draw(&mut rng).unwrap();
            // det(I + rho H H^H) = det(I + rho H^H H): the K x K form of Sylvester's identity
            capacity += equal_power(&hermitian(&h.view()).view(), antennas as f64 * RHO).unwrap();
            for (detector, rate) in [(&zf, &mut rate_zf), (&mmse, &mut rate_mmse)] {
                let (_, var) = detector.estimate(&h.view(), &y, 1.0 / RHO).unwrap();
                *rate += var.iter().map(|v| (1.0 + 1.0 / v).log2()).sum::<f64>();
            }
        }
        for (name, total) in [("capacity", capacity), ("zf", rate_zf), ("mmse", rate_mmse)] {
            let mean = total / trials as f64;
            println!("N = {antennas}, {name}: {mean:.3} bit/s/Hz");
            card.check(&json!({"antennas": antennas, "receiver": name}), mean);
        }
    }
    card.finish().unwrap();
}

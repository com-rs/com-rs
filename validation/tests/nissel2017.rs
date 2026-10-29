//! OFDM, CP-OFDM and FBMC/OQAM in doubly-selective channels against R. Nissel and M.
//! Rupp, "OFDM and FBMC-OQAM in doubly-selective channels: calculating the bit error
//! probability," *IEEE Commun. Lett.* 21(6), 2017
//! (`references/2017_nissel_ofdm-fbmc-oqam-doubly-selective-bep_comml.pdf`), with the
//! channels of `comrs::channel::multipath` (ITU-R M.1225 Vehicular A, Jakes Doppler) and
//! the one-tap equaliser on the exact diagonal of $Q^{H} H G$ (eq. (8)):
//!
//! - eqs. (16) and (17), the BEP of 4- and 16-QAM in doubly-flat fading, are those of
//!   `mapping::theory::ber_rayleigh_mrc` and `mapping::imperfect_csi::ber_16qam_rayleigh`
//!   (Chang & Su 2002) at $E_{b}/N_{0} = \mathrm{SNR}/\log_{2} m$, to $10^{-15}$; a
//!   simulation of FBMC/OQAM in doubly-flat fading reproduces (17) (Fig. 2: the same BEP as
//!   OFDM);
//! - Fig. 4 (64-QAM, no noise, 15 kHz, 2.5 GHz): the BER of OFDM without prefix and of
//!   CP-OFDM (a 4.76 µs prefix) at 50, 250 and 500 km/h within 30 % of the curves (OFDM
//!   3.2e-2, 4.3e-2, 6.8e-2; CP-OFDM below 1e-3, 1.45e-2, 4.5e-2). The FBMC curve of the
//!   paper uses the Hermite prototype of Haas & Belfiore (1997), not in com-rs: with the
//!   PHYDYAS prototype FBMC is between CP-OFDM and OFDM at 50 km/h and below CP-OFDM at
//!   500 km/h, as in the figure, but equal to CP-OFDM at 250 km/h (1.2e-2 against 8.5e-3
//!   for Hermite): PHYDYAS, four symbols long, suffers more from the Doppler spread.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
use comrs::channel::multipath::{Profile, Taps, Tdl};
use comrs::prelude::*;
use comrs::waveform::fbmc::{Fbmc, oqam_destagger, oqam_stagger};
use comrs::waveform::ofdm::Ofdm;

/// Sampling rate of Nissel & Rupp for Vehicular A, and the subcarrier spacing.
const FS: f64 = 2.94e6;
/// The subcarriers of 15 kHz at that rate.
const M: usize = 196;

fn doppler(kmh: f64) -> f64 {
    kmh / 3.6 * 2.5e9 / 299_792_458.0
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scheme {
    Ofdm,
    CpOfdm,
    Fbmc,
}

/// One tap of the realisation `taps` applied to `g` placed at `start`, projected on `q`
/// placed at `q_start`: q^H H g.
fn through(taps: &Taps, start: usize, g: &[C64], q_start: usize, q: &[C64]) -> C64 {
    let mut acc = C64::new(0.0, 0.0);
    for (j, qv) in q.iter().enumerate() {
        let n = q_start + j;
        let mut y = C64::new(0.0, 0.0);
        for (l, &d) in taps.delays().iter().enumerate() {
            if n >= d + start && n - d - start < g.len() {
                y += taps.coefficient(l, n, 0, 0) * g[n - d - start];
            }
        }
        acc += qv.conj() * y;
    }
    acc
}

/// BER of 64-QAM over Vehicular A at `kmh`, no noise, one-tap equalisation with the
/// exact diagonal of Q^H H G (Nissel & Rupp, eq. (8)), on the middle symbols.
fn ber(scheme: Scheme, kmh: f64, realisations: usize, seed: u64) -> f64 {
    let c = Constellation::qam(64).unwrap().normalized();
    let tdl = Tdl::new(Profile::VehicularA.taps(), doppler(kmh), FS, 1, 1).unwrap();
    let mut rng = Rng::new(seed);
    let (mut errors, mut total) = (0usize, 0usize);
    let n_cp = if scheme == Scheme::CpOfdm { 14 } else { 0 };
    let ofdm = Ofdm { oversampling: 1, n_cp, ..Ofdm::new(M, M).unwrap() };
    let fbmc = Fbmc::new(M, 4).unwrap();
    let k_c = 6; // complex symbols per subcarrier
    for _ in 0..realisations {
        let bits: Vec<u8> = (0..M * k_c * 6).map(|_| rng.below(2) as u8).collect();
        let x = c.modulate(&bits).unwrap();
        let s = match scheme {
            Scheme::Fbmc => fbmc.modulate(&oqam_stagger(&x, M)).unwrap(),
            _ => ofdm.modulate(&x).unwrap(),
        };
        let taps = tdl.draw(s.len(), &mut rng);
        let r = taps.apply(std::slice::from_ref(&s)).unwrap().remove(0);
        // the middle two complex symbols
        let keep = 2..4;
        let z: Vec<C64> = if scheme == Scheme::Fbmc {
            {
                let y = fbmc.demodulate(&r, 2 * k_c).unwrap();
                let mut est = vec![0.0; y.len()];
                for n in 2 * keep.start..2 * keep.end {
                    for m in 0..M {
                        let (start, g) = fbmc.pulse(m, n);
                        let h = through(&taps, start, &g, start, &g);
                        est[n * M + m] = (y[n * M + m] / h).re;
                    }
                }
                oqam_destagger(&est, M)
            }
        } else {
            {
                let y = ofdm.demodulate(&r).unwrap();
                let mut z = vec![C64::new(0.0, 0.0); y.len()];
                let n_tot = M + n_cp;
                for k in keep.clone() {
                    for i in 0..M {
                        let bin = ofdm.bin(i) as f64;
                        // transmit pulse with its prefix, receive pulse without
                        let g: Vec<C64> = (0..n_tot)
                            .map(|n| {
                                comrs::math::cis(std::f64::consts::TAU * bin * (n as f64 - n_cp as f64) / M as f64)
                                    / (M as f64).sqrt()
                            })
                            .collect();
                        let h = through(&taps, k * n_tot, &g, k * n_tot + n_cp, &g[n_cp..]);
                        z[k * M + i] = y[k * M + i] / h;
                    }
                }
                z
            }
        };
        let rx = c.demodulate(&z[keep.start * M..keep.end * M]);
        let tx = &bits[keep.start * M * 6..keep.end * M * 6];
        errors += rx.iter().zip(tx).filter(|(a, b)| a != b).count();
        total += tx.len();
    }
    errors as f64 / total as f64
}

#[test]
fn nissel_2017_eqs_16_17_are_the_rayleigh_closed_forms() {
    use comrs::mapping::imperfect_csi::ber_16qam_rayleigh;
    use comrs::mapping::theory::ber_rayleigh_mrc;
    for snr_db in [-5.0, 0.0, 10.0, 20.0, 35.0] {
        let snr = db2lin(snr_db);
        let eq16 = 0.5 - 0.5 / (1.0 + 2.0 / snr).sqrt();
        let eq17 = 0.5 - 3.0 / 8.0 / (1.0 + 10.0 / snr).sqrt() - 6.0 / 8.0 / (9.0 + 10.0 / snr).sqrt()
            + 5.0 / 8.0 / (25.0 + 10.0 / snr).sqrt();
        assert!((eq16 - ber_rayleigh_mrc(snr / 2.0, 1)).abs() < 1e-15);
        assert!((eq17 - ber_16qam_rayleigh(snr / 4.0)).abs() < 1e-15);
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn nissel_2017_fig_2_fbmc_in_doubly_flat_fading() {
    use comrs::mapping::imperfect_csi::ber_16qam_rayleigh;
    let c = Constellation::qam(16).unwrap().normalized();
    let fbmc = Fbmc::new(64, 4).unwrap();
    let mut rng = Rng::new(21);
    for snr_db in [10.0, 20.0] {
        let sigma2 = 1.0 / db2lin(snr_db);
        let (mut errors, mut total) = (0usize, 0usize);
        for _ in 0..3000 {
            let bits: Vec<u8> = (0..64 * 4 * 4).map(|_| rng.below(2) as u8).collect();
            let x = c.modulate(&bits).unwrap();
            let h = rng.complex_gaussian();
            let r: Vec<C64> = fbmc
                .modulate(&oqam_stagger(&x, 64))
                .unwrap()
                .iter()
                .map(|v| v * h + rng.complex_gaussian() * sigma2.sqrt())
                .collect();
            let y = fbmc.demodulate(&r, 8).unwrap();
            let est: Vec<f64> = y.iter().map(|v| (v / h).re).collect();
            let rx = c.demodulate(&oqam_destagger(&est, 64));
            errors += rx.iter().zip(&bits).filter(|(a, b)| a != b).count();
            total += bits.len();
        }
        let want = ber_16qam_rayleigh(db2lin(snr_db) / 4.0);
        let got = errors as f64 / total as f64;
        assert!((got / want - 1.0).abs() < 0.1, "{snr_db} dB: {got} against {want}");
    }
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn nissel_2017_fig_4_vehicular_a() {
    let read: [(f64, f64, f64); 3] = [(50.0, 3.2e-2, 1e-3), (250.0, 4.3e-2, 1.45e-2), (500.0, 6.8e-2, 4.5e-2)];
    for (kmh, ofdm_read, cp_read) in read {
        let ofdm = ber(Scheme::Ofdm, kmh, 300, 1);
        let cp = ber(Scheme::CpOfdm, kmh, 300, 2);
        let fbmc = ber(Scheme::Fbmc, kmh, 300, 3);
        assert!((ofdm / ofdm_read - 1.0).abs() < 0.3, "{kmh} km/h OFDM {ofdm}");
        if kmh < 100.0 {
            assert!(cp < cp_read && cp < fbmc && fbmc < ofdm, "{kmh} km/h: {cp} {fbmc} {ofdm}");
        } else {
            assert!((cp / cp_read - 1.0).abs() < 0.3, "{kmh} km/h CP-OFDM {cp}");
        }
        if kmh > 400.0 {
            assert!(fbmc < cp && cp < ofdm, "{kmh} km/h: {fbmc} {cp} {ofdm}");
        }
    }
}

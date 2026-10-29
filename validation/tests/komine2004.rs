//! The indoor visible-light channel (`comrs::channel::owc::indoor`) and the PIN-FET
//! receiver (`comrs::hardware::optical::PinFetReceiver`) against T. Komine and M.
//! Nakagawa, "Fundamental analysis for visible-light communication system using LED
//! lights," *IEEE Trans. Consum. Electron.* 50(1), 100–107, 2004: the room of their
//! Fig. 2 (5 × 5 × 3 m, four lights of 60 × 60 LEDs at 1 cm pitch, 2.5 m high, centred
//! at (1.25, 1.25), (3.75, 1.25), (1.25, 3.75) and (3.75, 3.75)), Table I (20 mW and
//! 0.73 cd per LED, semi-angle 70°, field of view 60°, 1 cm², concentrator of index 1.5)
//! and Table II, the desk 0.85 m high sampled every 0.1 m. The minimum, maximum and
//! average over the desk, here and in the captions:
//!
//! | quantity | here | paper |
//! |---|---|---|
//! | illuminance (Fig. 3, lx) | 313.5 / 1211.5 / 865.7 | 313.5 / 1211.5 / 865.7 |
//! | received power, line of sight (Fig. 4, dBm) | −2.83 / 3.98 / 2.11 | −2.8 / 4.0 / 2.0 |
//! | received power with reflections (Fig. 6, dBm) | −2.83 / 4.22 / 2.62 | −2.8 / 4.2 / 2.5 |
//! | SNR with ISI, line of sight (Fig. 8, dB) | 13.63 / 24.77 / 18.87 | 12.7 / 24.8 / 18.9 |
//! | SNR with ISI and reflections (Fig. 9, dB) | 12.95 / 20.03 / 15.43 | 11.6 / 20.1 / 15.4 |
//!
//! The paper leaves three choices open, made here as its figures require: the walls
//! reflect $\rho = 0.7$ and the ceiling and floor nothing (not stated; this reproduces
//! the maximum of Fig. 6 and the 0.5 dB that the reflections add on average, Sec. V), the bit of 10 ns starts at the first arrival
//! (as the time axis of Fig. 7), and the Lambertian mode is $-\ln 2/\ln\cos 70°$ (the
//! text prints it without its minus sign). The delay spread of the line of sight is a
//! few hundred picoseconds, so the bins are of 10 ps (0.8 dB lost on the maximum SNR with
//! bins of 0.1 ns). The minima of the SNR are 0.9 and 1.35 dB above the captions, at the
//! points between the lights; the maxima and averages agree within 0.15 dB. The noise of Table II (its 0.53 A/W, the text quoting
//! 0.54) is negligible against the interference, as the paper notes (Fig. 10).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::owc::indoor::{
    Detector, Emitter, Illumination, Reflectance, Resolution, Room, direction, illuminance_lx, lambertian_mode,
};
use comrs::hardware::optical::PinFetReceiver;
use comrs::math::lin2db;
use std::f64::consts::FRAC_PI_2;

const AREA: f64 = 1e-4;
const BIT: f64 = 10e-9;
const BANDWIDTH: f64 = 100e6;

fn lights() -> Vec<Emitter> {
    let mode = lambertian_mode(70f64.to_radians()).unwrap();
    let down = direction(-FRAC_PI_2, 0.0);
    let offsets: Vec<f64> = (0..60).map(|i| (f64::from(i) - 29.5) * 0.01).collect();
    let mut out = Vec::with_capacity(4 * 3600);
    for (cx, cy) in [(1.25, 1.25), (3.75, 1.25), (1.25, 3.75), (3.75, 3.75)] {
        for a in &offsets {
            for b in &offsets {
                out.push(Emitter::new([cx + a, cy + b, 2.5], down, mode, 20e-3));
            }
        }
    }
    out
}

/// The points of the desk, every 0.1 m from wall to wall.
fn desk() -> impl Iterator<Item = [f64; 3]> {
    (0..=50).flat_map(|i| (0..=50).map(move |j| [f64::from(i) * 0.1, f64::from(j) * 0.1, 0.85]))
}

fn detector(at: [f64; 3]) -> Detector {
    Detector { concentrator_index: Some(1.5), ..Detector::new(at, direction(FRAC_PI_2, 0.0), AREA, 60f64.to_radians()) }
}

fn min_max_mean(x: &[f64]) -> [f64; 3] {
    let min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    [min, max, x.iter().sum::<f64>() / x.len() as f64]
}

fn assert_close(what: &str, got: [f64; 3], want: [f64; 3], tol: [f64; 3]) {
    for (k, name) in ["min", "max", "mean"].iter().enumerate() {
        assert!((got[k] - want[k]).abs() <= tol[k], "{what} {name}: {:.2} vs {}", got[k], want[k]);
    }
}

/// The received power (dBm) and the SNR (dB) on the desk, without and with reflections.
fn desk_maps(lit: &Illumination) -> ([Vec<f64>; 2], [Vec<f64>; 2]) {
    let rx = PinFetReceiver::komine2004();
    let (mut power, mut snr) = ([vec![], vec![]], [vec![], vec![]]);
    for at in desk() {
        let h = lit.impulse_responses(&detector(at)).unwrap();
        let origin = h[0].first_arrival_s().unwrap();
        let all = h[0].add(&h[1]).unwrap();
        for (k, r) in [&h[0], &all].into_iter().enumerate() {
            power[k].push(lin2db(r.total_w() / 1e-3));
            let (s, isi) = r.bit_split(BIT, origin);
            snr[k].push(rx.ook_snr_db(s, isi, AREA, BANDWIDTH));
        }
    }
    (power, snr)
}

#[test]
fn komine_2004_illuminance_fig_3() {
    let lights = lights();
    let up = direction(FRAC_PI_2, 0.0);
    let lx: Vec<f64> = desk().map(|p| illuminance_lx(&lights, 0.73, p, up)).collect();
    assert_close("illuminance", min_max_mean(&lx), [313.5, 1211.5, 865.7], [0.5, 0.5, 0.5]);
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn komine_2004_received_power_and_snr_figs_4_6_8_9() {
    let rho = Reflectance { north: 0.7, south: 0.7, east: 0.7, west: 0.7, ceiling: 0.0, floor: 0.0 };
    let room = Room { size_m: [5.0, 5.0, 3.0], reflectance: rho };
    let lit = room.illuminate(&lights(), &Resolution::new(0.01e-9, vec![[50, 50, 30]])).unwrap();
    let (power, snr) = desk_maps(&lit);
    let [p0, p1] = power.map(|x| min_max_mean(&x));
    let [s0, s1] = snr.map(|x| min_max_mean(&x));
    println!("power {p0:.2?} {p1:.2?}, SNR {s0:.2?} {s1:.2?}");
    assert_close("power, line of sight", p0, [-2.8, 4.0, 2.0], [0.1, 0.1, 0.15]);
    assert_close("power with reflections", p1, [-2.8, 4.2, 2.5], [0.1, 0.1, 0.15]);
    assert_close("SNR, line of sight", s0, [12.7, 24.8, 18.9], [1.2, 0.2, 0.2]);
    assert_close("SNR with reflections", s1, [11.6, 20.1, 15.4], [1.5, 0.2, 0.2]);
}

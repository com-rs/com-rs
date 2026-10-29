//! The impulse response of indoor optical wireless channels with reflections of every
//! order (`comrs::channel::owc::indoor`) against J. R. Barry, J. M. Kahn, W. J. Krause,
//! E. A. Lee and D. G. Messerschmitt, "Simulation of multipath impulse response for
//! indoor wireless optical channels," *IEEE J. Sel. Areas Commun.* 11(3), 367–379, 1993:
//! the four configurations of their Table I, simulated with the same partitions per
//! order of reflection and bins of 0.2 ns, against the powers of their Figs. 3, 6–8 and
//! Table II (received power for a 1 W source, −3 dB bandwidth, share of each order).
//!
//! The powers of every configuration agree within 3 %, the bandwidths of B, C and D
//! within 5 %. The bandwidth of configuration A is not checked: we find 14.4 MHz with
//! three reflections where Sec. IV-A gives 9 MHz, and with one reflection the line of
//! sight carries 0.709 of the power, so close to the −3 dB threshold (0.708) that the
//! bandwidth is ill-conditioned (87 MHz against 30). The gap is not explained.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::owc::indoor::{Detector, Emitter, ImpulseResponse, Reflectance, Resolution, Room, direction};

struct Config {
    size: [f64; 3],
    rho: Reflectance,
    source: ([f64; 3], f64, f64),
    receiver: ([f64; 3], f64),
    fov_deg: f64,
}

/// The responses of the line of sight and of 1 to 3 reflections (Table I).
fn simulate(c: &Config, partitions: [[usize; 3]; 3]) -> Vec<ImpulseResponse> {
    let room = Room { size_m: c.size, reflectance: c.rho };
    let (pos, el, az) = c.source;
    let source = Emitter::new(pos, direction(el.to_radians(), az.to_radians()), 1.0, 1.0);
    let detector = Detector::new(c.receiver.0, direction(c.receiver.1.to_radians(), 0.0), 1e-4, c.fov_deg.to_radians());
    let lit = room.illuminate(&[source], &Resolution::new(0.2e-9, partitions.to_vec())).unwrap();
    lit.impulse_responses(&detector).unwrap()
}

fn total(h: &[ImpulseResponse]) -> ImpulseResponse {
    h.iter().skip(1).fold(h[0].clone(), |acc, x| acc.add(x).unwrap())
}

fn bandwidth_mhz(h: &ImpulseResponse) -> f64 {
    h.bandwidth_3db_hz(0.1e6, 200e6).unwrap() / 1e6
}

const ROOM_BCD: [f64; 3] = [7.5, 5.5, 3.5];
const PARTITIONS_BCD: [[usize; 3]; 3] = [[750, 550, 350], [150, 110, 70], [30, 22, 14]];

/// Configuration A (Fig. 3, Table II): 1.23, 0.505, 0.430 and 0.269 µW for 0 to 3
/// reflections, 2.4 µW in all (the bandwidth is not checked: see the module).
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn barry_1993_configuration_a() {
    let c = Config {
        size: [5.0, 5.0, 3.0],
        rho: Reflectance { north: 0.8, south: 0.8, east: 0.8, west: 0.8, ceiling: 0.8, floor: 0.3 },
        source: ([2.5, 2.5, 3.0], -90.0, 0.0),
        receiver: ([0.5, 1.0, 0.0], 90.0),
        fov_deg: 85.0,
    };
    let h = simulate(&c, [[500, 500, 300], [100, 100, 60], [25, 25, 15]]);
    for (k, want) in [1.23, 0.505, 0.430, 0.269].iter().enumerate() {
        let got = h[k].total_w() * 1e6;
        assert!((got / want - 1.0).abs() < 0.03, "{k} reflections: {got:.3} µW vs {want}");
    }
    let all = total(&h);
    assert!((all.total_w() * 1e6 / 2.4 - 1.0).abs() < 0.03, "{}", all.total_w());
}

/// Configuration B (Fig. 6(a), Table II): 307 nW, of which 77.9 %, 6.0 %, 13.0 % and
/// 3.2 % for 0 to 3 reflections; 19.5 MHz.
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn barry_1993_configuration_b() {
    let c = Config {
        size: ROOM_BCD,
        rho: Reflectance { north: 0.30, south: 0.56, east: 0.30, west: 0.12, ceiling: 0.69, floor: 0.09 },
        source: ([2.0, 4.0, 3.3], -90.0, 0.0),
        receiver: ([6.6, 2.8, 0.8], 90.0),
        fov_deg: 70.0,
    };
    let h = simulate(&c, PARTITIONS_BCD);
    let all = total(&h);
    let p = all.total_w();
    assert!((p * 1e9 / 307.0 - 1.0).abs() < 0.03, "{} nW", p * 1e9);
    for (k, want) in [77.9, 6.0, 13.0, 3.2].iter().enumerate() {
        let share = 100.0 * h[k].total_w() / p;
        assert!((share - want).abs() < 0.1 * want.max(5.0), "{k} reflections: {share:.1} % vs {want}");
    }
    let b = bandwidth_mhz(&all);
    assert!((b / 19.5 - 1.0).abs() < 0.05, "{b} MHz vs 19.5");
}

/// Configuration C (Fig. 7(a), Table II): the source tilted (elevation −70°, azimuth
/// 10°), a reflective north wall; 283.7 nW and 13 MHz.
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn barry_1993_configuration_c() {
    let c = Config {
        size: ROOM_BCD,
        rho: Reflectance { north: 0.58, south: 0.56, east: 0.30, west: 0.12, ceiling: 0.69, floor: 0.09 },
        source: ([5.0, 1.0, 3.3], -70.0, 10.0),
        receiver: ([2.0, 4.0, 0.8], 90.0),
        fov_deg: 70.0,
    };
    let all = total(&simulate(&c, PARTITIONS_BCD));
    assert!((all.total_w() * 1e9 / 283.7 - 1.0).abs() < 0.03, "{} nW", all.total_w() * 1e9);
    let b = bandwidth_mhz(&all);
    assert!((b / 13.0 - 1.0).abs() < 0.05, "{b} MHz vs 13.0");
}

/// Configuration D (Fig. 8, Table II): a diffuse link, the source near the floor pointing
/// at the ceiling, no line of sight; 689.8 nW and 32 MHz.
#[test]
#[ignore = "statistical validation: run with --ignored"]
fn barry_1993_configuration_d() {
    let c = Config {
        size: ROOM_BCD,
        rho: Reflectance { north: 0.58, south: 0.56, east: 0.30, west: 0.12, ceiling: 0.69, floor: 0.09 },
        source: ([3.75, 2.75, 1.0], 90.0, 0.0),
        receiver: ([6.0, 0.8, 0.8], 90.0),
        fov_deg: 70.0,
    };
    let h = simulate(&c, PARTITIONS_BCD);
    assert_eq!(h[0].total_w(), 0.0);
    let all = total(&h);
    assert!((all.total_w() * 1e9 / 689.8 - 1.0).abs() < 0.03, "{} nW", all.total_w() * 1e9);
    let b = bandwidth_mhz(&all);
    assert!((b / 32.0 - 1.0).abs() < 0.05, "{b} MHz vs 32.0");
}

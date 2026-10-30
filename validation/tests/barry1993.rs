//! The impulse response of indoor optical wireless channels with reflections of every
//! order (`comrs::channel::owc::indoor`) against J. R. Barry, J. M. Kahn, W. J. Krause,
//! E. A. Lee and D. G. Messerschmitt, "Simulation of multipath impulse response for
//! indoor wireless optical channels," *IEEE J. Sel. Areas Commun.* 11(3), 367–379, 1993:
//! the four configurations of their Table I, simulated with the same partitions per
//! order of reflection and bins of 0.2 ns. The tests measure; the powers of Figs. 3, 6–8
//! and Table II, their tolerances, and the unexplained gap of the bandwidth of
//! configuration A are in the card `validation/cards/barry1993_table2.yaml`, one part
//! per configuration.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::owc::indoor::{Detector, Emitter, ImpulseResponse, Reflectance, Resolution, Room, direction};
use serde_json::json;

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

fn part(config: &str) -> Card {
    Card::load("barry1993_table2").unwrap().part(&json!({ "config": config })).unwrap()
}

/// Configuration A (Fig. 3, Table II): the power of every order of reflection.
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
    let mut card = part("A");
    for (k, hk) in h.iter().enumerate() {
        card.check(&json!({"config": "A", "metric": "power_uw", "reflections": k}), hk.total_w() * 1e6);
    }
    card.check(&json!({"config": "A", "metric": "total_power_uw"}), total(&h).total_w() * 1e6);
    card.finish().unwrap();
}

/// Configuration B (Fig. 6(a), Table II): the power, its share per order, the bandwidth.
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
    let mut card = part("B");
    card.check(&json!({"config": "B", "metric": "total_power_nw"}), p * 1e9);
    for (k, hk) in h.iter().enumerate() {
        card.check(&json!({"config": "B", "metric": "share_percent", "reflections": k}), 100.0 * hk.total_w() / p);
    }
    card.check(&json!({"config": "B", "metric": "bandwidth_mhz"}), bandwidth_mhz(&all));
    card.finish().unwrap();
}

/// Configuration C (Fig. 7(a), Table II): the source tilted (elevation −70°, azimuth
/// 10°), a reflective north wall.
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
    let mut card = part("C");
    card.check(&json!({"config": "C", "metric": "total_power_nw"}), all.total_w() * 1e9);
    card.check(&json!({"config": "C", "metric": "bandwidth_mhz"}), bandwidth_mhz(&all));
    card.finish().unwrap();
}

/// Configuration D (Fig. 8, Table II): a diffuse link, the source near the floor pointing
/// at the ceiling, no line of sight.
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
    let all = total(&h);
    let mut card = part("D");
    card.check(&json!({"config": "D", "metric": "power_uw", "reflections": 0}), h[0].total_w() * 1e6);
    card.check(&json!({"config": "D", "metric": "total_power_nw"}), all.total_w() * 1e9);
    card.check(&json!({"config": "D", "metric": "bandwidth_mhz"}), bandwidth_mhz(&all));
    card.finish().unwrap();
}

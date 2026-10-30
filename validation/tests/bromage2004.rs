//! The Raman solver (`comrs::channel::optical::raman`) against J. Bromage, "Raman
//! amplification for fiber communications systems," *J. Lightw. Technol.* 22(1), 79–93,
//! 2004, and the gain identity of J. Hu, B. S. Marks and C. R. Menyuk, "Flat-gain fiber
//! Raman amplifiers using equally spaced pumps," *J. Lightw. Technol.* 22(6), 1519–1522,
//! 2004 (and of Perlin & Winful 2002):
//!
//! - Bromage, eqs. (3)–(5) and their examples (Sec. III-B): 10 km of SLA fibre
//!   ($C_{R} = 0.29$ W⁻¹km⁻¹, the peak of Fig. 5; 0.19 and 0.25 dB/km) need 1.24 W of pump
//!   for a small-signal net gain of 10 dB, 5 km of DCF ($C_{R} = 3.2$; 0.4 and 0.7 dB/km)
//!   250 mW; co- and counter-pumps add in the exponent, $C_{R} L_{\mathrm{eff}}[P^{+}(0) +
//!   P^{-}(L)]$ (here 9.97 and 9.93 dB, within 0.1 dB of 10; the closed form within
//!   10⁻³ dB);
//! - Bromage, Figs. 7–8: a span of 100 km and 20 dB of loss, counter-pumped at 1450 nm,
//!   followed by an amplifier of noise figure 5 dB; the noise figure of the span, eq. (9),
//!   and of the cascade, eq. (10), against the on-off gain. The pump loss is not given:
//!   0.22 dB/km fits both points read on Fig. 8 (NF of 20.15 and 17.50 dB at 10 and 20 dB
//!   of on-off gain against 20.1 and 17.5; of the span 19.14 and 17.33 against 19.0 and
//!   17.4; 25 dB without pump), within 0.3 dB; the signal reaches its minimum 9.7 dB below
//!   its launch power 68 km into the span at 20 dB of gain (Fig. 7: about 10.5 dB near 65
//!   km; within 1.5 dB and 8 km);
//! - Hu et al., eq. (2): whatever the depletion and the pump–pump interactions, the gain
//!   of every wave is $\ln G_{k} = -\alpha_{k} L + \sum_{j} g_{kj} I_{j}$ with the integrated
//!   powers $I_{j} = \int_{0}^{L} P_{j}\, dz$ of the solution, here four counter- and one
//!   co-propagating pumps (1.1 W) and 40 signals of 1 mW over 80 km, the integrals by the
//!   trapezoidal rule on the grid of the solver (within 10⁻³ dB; the largest gap is
//!   8.4 · 10⁻⁵ dB, the error of the trapezoids).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comrs::channel::optical::fiber::FiberParams;
use comrs::channel::optical::raman::{Direction, Raman, RamanGain, RamanPump};
use comrs::math::{PLANCK, bisect, db2lin, lin2db, nm2hz, np2db};

const SIGNAL_NM: f64 = 1550.0;
const PUMP_NM: f64 = 1450.0;

/// A gain spectrum of constant efficiency `c_r` (W⁻¹km⁻¹) for the pump at 1450 nm.
fn flat_gain(c_r: f64, aeff_um2: f64) -> RamanGain {
    let offsets: Vec<f64> = (0..=40).map(|k| f64::from(k) * 1e12).collect();
    let g = c_r * aeff_um2 * 1e-12 / 1e3;
    RamanGain::new(offsets, vec![g; 41], nm2hz(PUMP_NM)).unwrap()
}

/// The small-signal net gain (dB) of `length` km with the co- and counter-pump powers.
fn net_gain(c_r: f64, length: f64, alpha_s: f64, alpha_p: f64, co: f64, counter: f64) -> f64 {
    let fiber = FiberParams { span_km: length, alpha_db_km: alpha_s, ..FiberParams::default() };
    let pump = |power, direction| RamanPump {
        alpha_db_km: Some(alpha_p),
        ..RamanPump::at_wavelength(PUMP_NM, power, direction)
    };
    let pumps = [(co, Direction::Co), (counter, Direction::Counter)]
        .into_iter()
        .filter(|(p, _)| *p > 0.0)
        .map(|(p, d)| pump(p, d))
        .collect();
    let raman = Raman { pumps, gain: flat_gain(c_r, 80.0), aeff_um2: 80.0, ..Raman::default() };
    raman.solve(&fiber, &[(nm2hz(SIGNAL_NM), 1e-7)]).unwrap().net_gain_db(0)
}

/// Bromage's eq. (3) in dB.
fn eq_3(c_r: f64, length: f64, alpha_s: f64, alpha_p: f64, pumps: f64) -> f64 {
    let (a_s, a_p) = (alpha_s / np2db(1.0), alpha_p / np2db(1.0));
    let leff = (1.0 - (-a_p * length).exp()) / a_p;
    np2db(-a_s * length + c_r * leff * pumps)
}

#[test]
fn bromage_2004_eqs_3_5_examples() {
    // SLA: 10 km, 0.19 and 0.25 dB/km, 1.24 W; DCF: 5 km, 0.4 and 0.7 dB/km, 250 mW
    for (c_r, length, a_s, a_p, power) in [(0.29, 10.0, 0.19, 0.25, 1.24), (3.2, 5.0, 0.4, 0.7, 0.25)] {
        let closed = eq_3(c_r, length, a_s, a_p, power);
        assert!((closed - 10.0).abs() < 0.1, "eq. (3): {closed}");
        for (co, counter) in [(power, 0.0), (0.0, power), (0.4 * power, 0.6 * power)] {
            let g = net_gain(c_r, length, a_s, a_p, co, counter);
            assert!((g - closed).abs() < 1e-3, "C_R = {c_r}, {co} + {counter} W: {g} vs {closed}");
        }
    }
}

/// The noise figures (dB) of the span and of span and amplifier, and the depth (dB) and
/// position (km) of the minimum of the signal, at the on-off gain `target`.
fn figs_7_8(target: f64) -> (f64, f64, f64, f64) {
    let fiber = FiberParams { span_km: 100.0, alpha_db_km: 0.2, ..FiberParams::default() };
    let fs = nm2hz(SIGNAL_NM);
    let solve = |power: f64| {
        let pump =
            RamanPump { alpha_db_km: Some(0.22), ..RamanPump::at_wavelength(PUMP_NM, power, Direction::Counter) };
        Raman { pumps: vec![pump], temperature_k: 298.0, ..Raman::default() }.solve(&fiber, &[(fs, 1e-6)]).unwrap()
    };
    let p = solve(bisect(|w| solve(w).on_off_gain_db(0) < target, 1e-5, 3.0, 60));
    let g = db2lin(p.net_gain_db(0));
    // eq. (9): NF = 2 P_ASE / (h ν G B) + 1/G, P_ASE of one polarisation
    let span = p.ase_psd[0] / (PLANCK * fs * g) + 1.0 / g;
    // eq. (10): then an amplifier of 5 dB
    let total = span + (db2lin(5.0) - 1.0) / g;
    let power = p.power_w(0);
    let (i, min) = power.iter().enumerate().fold((0, f64::MAX), |a, (i, v)| if *v < a.1 { (i, *v) } else { a });
    (lin2db(span), lin2db(total), lin2db(power[0] / min), p.z_km[i])
}

#[test]
fn bromage_2004_figs_7_8_noise_figure() {
    let (span, total, _, _) = figs_7_8(1e-6);
    assert!((span - 20.0).abs() < 0.01 && (total - 25.0).abs() < 0.01);
    for (gain, span_read, total_read) in [(10.0, 19.0, 20.1), (20.0, 17.4, 17.5)] {
        let (span, total, _, _) = figs_7_8(gain);
        assert!((span - span_read).abs() < 0.3 && (total - total_read).abs() < 0.3, "{gain} dB: {span} {total}");
    }
    let (_, _, depth, at) = figs_7_8(20.0);
    assert!((depth - 10.5).abs() < 1.5 && (at - 65.0).abs() < 8.0, "Fig. 7: {depth} dB at {at} km");
}

#[test]
fn hu_2004_eq_2_gain_from_the_integrated_powers() {
    let fiber = FiberParams { span_km: 80.0, ..FiberParams::default() };
    let pumps = vec![
        RamanPump::at_wavelength(1420.0, 0.35, Direction::Counter),
        RamanPump::at_wavelength(1440.0, 0.25, Direction::Counter),
        RamanPump::at_wavelength(1460.0, 0.2, Direction::Counter),
        RamanPump::at_wavelength(1490.0, 0.15, Direction::Counter),
        RamanPump::at_wavelength(1430.0, 0.15, Direction::Co),
    ];
    let raman = Raman { pumps: pumps.clone(), ..Raman::default() };
    let signals: Vec<(f64, f64)> = (0..40).map(|k| (187e12 + f64::from(k) * 200e9, 1e-3)).collect();
    let p = raman.solve(&fiber, &signals).unwrap();
    let n = p.freqs_hz.len();
    let aeff = raman.aeff_um2 * 1e-12;
    // the coefficients of eq. (1), from the gain spectrum alone
    let coefficient = |k: usize, j: usize| {
        let (fk, fj) = (p.freqs_hz[k], p.freqs_hz[j]);
        if fj > fk {
            raman.gain.efficiency(fk, fj, aeff)
        } else if fj < fk {
            -(fk / fj) * raman.gain.efficiency(fj, fk, aeff)
        } else {
            0.0
        }
    };
    let integral = |j: usize| {
        let w = p.power_w(j);
        (1..w.len()).map(|i| 0.5 * (w[i] + w[i - 1]) * (p.z_km[i] - p.z_km[i - 1])).sum::<f64>()
    };
    let integrals: Vec<f64> = (0..n).map(integral).collect();
    for k in 0..n {
        let predicted: f64 =
            -p.alpha_np_km[k] * fiber.span_km + (0..n).map(|j| coefficient(k, j) * integrals[j]).sum::<f64>();
        let l = &p.log_power[k];
        let counter = k >= signals.len() && pumps[k - signals.len()].direction == Direction::Counter;
        // a counter-propagating wave grows from L to 0
        let solved = if counter { l[0] - l[l.len() - 1] } else { l[l.len() - 1] - l[0] };
        assert!(np2db((solved - predicted).abs()) < 1e-3, "wave {k}: {} vs {} dB", np2db(solved), np2db(predicted));
    }
}

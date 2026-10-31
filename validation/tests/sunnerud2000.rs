//! The pulse broadening by PMD of `com_channel::optical::pmd` against H. Sunnerud,
//! M. Karlsson and P. A. Andrekson, "Analytical theory for PMD-compensation," *IEEE
//! Photon. Technol. Lett.* 12(1), 50–52, 2000.
//!
//! Without chromatic dispersion, a pulse of rms width $\tau_{0}$ and input Stokes vector
//! $\hat{j}$ leaves a link of input PMD vector $\vec{\Omega}(\omega)$ with the rms width
//! (eq. (1))
//!
//! $$
//! \tau^{2} = \tau_{0}^{2} + \tfrac{1}{4} \left( \langle \Omega^{2} \rangle - (\hat{j} \cdot
//! \langle \vec{\Omega} \rangle)^{2} \right),
//! $$
//!
//! $\langle \cdot \rangle$ the mean over the normalised spectrum of the pulse; in the strong
//! coupling regime, for a Gaussian pulse and $x = E\{\Delta\tau^{2}\} / 4 \tau_{0}^{2}$, the
//! expected broadening $b^{2} = E\{\tau^{2} / \tau_{0}^{2}\}$ is $1 + x - \frac{1}{2} (\sqrt{1 +
//! 4x/3} - 1)$ for a random input polarisation (eq. (7)), $1 + x - \frac{3}{2} (\sqrt{1 +
//! 4x/3} - 1)$ at the principal state $\hat{j} = \langle \vec{\Omega} \rangle / |\langle
//! \vec{\Omega} \rangle|$ (eq. (8)) and $1 + x$ orthogonally to it (Sec. V): at a mean DGD
//! equal to $\tau_{0}$, 10 %, 1 % and 14 % of broadening.
//!
//! The case: Gaussian pulses of rms width 10 ps on 256 samples at 200 GHz through
//! `PmdLink::apply`, links of 200 sections of `PmdLink::with_mean_dgd` (mean DGD 0.5 to 2
//! times $\tau_{0}$; by construction $E\{\Delta\tau^{2}\} = \frac{3\pi}{8} \langle \Delta\tau
//! \rangle^{2}$), 2000 links per case; the PMD vectors of `com_validation::pmd` (Gordon &
//! Kogelnik 2000). The cards are `validation/cards/sunnerud2000_eq1.yaml` and
//! `validation/cards/sunnerud2000_fig1.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::pmd::{dot, jones_of, moments, pmd_vectors, stokes};
use comrs::C64;
use comrs::channel::optical::pmd::{Jones, PmdLink};
use comrs::core::rng::Rng;
use comrs::core::signal::DualPol;
use serde_json::json;
use std::f64::consts::PI;

const N: usize = 256;
const FS: f64 = 200e9;
const TAU0: f64 = 10e-12;
const SECTIONS: usize = 200;

/// The Gaussian pulse $e^{-t^{2} / 2T^{2}}$, $T = \sqrt{2} \tau_{0}$, centred on the block.
fn pulse() -> Vec<C64> {
    let t2 = 2.0 * TAU0 * TAU0;
    (0..N)
        .map(|k| {
            let t = (k as f64 - N as f64 / 2.0) / FS;
            C64::new((-t * t / (2.0 * t2)).exp(), 0.0)
        })
        .collect()
}

/// The spectral means $\langle \Omega^{2} \rangle$ and $\langle \vec{\Omega} \rangle$ of the
/// input PMD vector over the normalised spectrum of the pulse.
fn spectral_means(link: &PmdLink, spectrum: &[C64]) -> (f64, [f64; 3]) {
    let omega = comrs::math::fft::angular_frequencies(N, FS);
    let peak = spectrum.iter().map(C64::norm_sqr).fold(0.0, f64::max);
    let (mut w2, mut w1, mut den) = (0.0, [0.0; 3], 0.0);
    for (s, w) in spectrum.iter().zip(&omega) {
        let p = s.norm_sqr();
        if p > 1e-16 * peak {
            let v = pmd_vectors(link, *w).0;
            w2 += p * dot(&v, &v);
            for i in 0..3 {
                w1[i] += p * v[i];
            }
            den += p;
        }
    }
    (w2 / den, w1.map(|v| v / den))
}

/// The squared rms width of the pulse launched with the Jones vector `(x, y)`.
fn width_squared(link: &PmdLink, pulse: &[C64], x: C64, y: C64) -> f64 {
    let mut u = DualPol { x: pulse.iter().map(|v| v * x).collect(), y: pulse.iter().map(|v| v * y).collect() };
    link.apply(&mut u, FS).unwrap();
    moments(&u, FS).1.powi(2)
}

#[test]
fn eq_1_of_sunnerud_2000_holds_for_every_realisation() {
    let mut card = Card::load("sunnerud2000_eq1").unwrap();
    let mut rng = Rng::new(2000);
    let p = pulse();
    let spectrum = comrs::math::fft::fft(&p);
    let (_, tau0) = moments(&DualPol { x: p.clone(), y: vec![C64::new(0.0, 0.0); N] }, FS);
    card.check(&json!({"quantity": "initial_rms_width_over_tau0"}), tau0 / TAU0);
    let mut worst: f64 = 0.0;
    for _ in 0..4 {
        let link = PmdLink::with_mean_dgd(1.5 * TAU0, SECTIONS, &mut rng);
        let (w2, w1) = spectral_means(&link, &spectrum);
        for _ in 0..4 {
            let (x, y) = Jones::random_su2(&mut rng).apply(C64::new(1.0, 0.0), C64::new(0.0, 0.0));
            let j = stokes(x, y);
            let expected = tau0 * tau0 + 0.25 * (w2 - dot(&j, &w1).powi(2));
            worst = worst.max((width_squared(&link, &p, x, y) / expected - 1.0).abs());
        }
    }
    card.check(&json!({"quantity": "eq_1_relative_error"}), worst);
    card.finish().unwrap();
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn the_expected_broadening_follows_eqs_7_and_8_of_sunnerud_2000() {
    let mut card = Card::load("sunnerud2000_fig1").unwrap();
    let mut rng = Rng::new(2001);
    let p = pulse();
    let spectrum = comrs::math::fft::fft(&p);
    let tau0 = moments(&DualPol { x: p.clone(), y: vec![C64::new(0.0, 0.0); N] }, FS).1;
    let links = 2000;
    for ratio in [0.5, 1.0, 1.5, 2.0] {
        let mean_dgd = ratio * TAU0;
        let x = 3.0 * PI / 8.0 * mean_dgd * mean_dgd / (4.0 * TAU0 * TAU0);
        let root = (1.0 + 4.0 * x / 3.0).sqrt() - 1.0;
        let theory =
            [("average", 1.0 + x - 0.5 * root), ("principal_state", 1.0 + x - 1.5 * root), ("orthogonal", 1.0 + x)];
        let mut sums = [0.0; 3];
        let mut squares = [0.0; 3];
        for _ in 0..links {
            let link = PmdLink::with_mean_dgd(mean_dgd, SECTIONS, &mut rng);
            let (_, w1) = spectral_means(&link, &spectrum);
            let n1 = dot(&w1, &w1).sqrt();
            let psp = w1.map(|v| v / n1);
            // a unit vector orthogonal to the principal state
            let a = if psp[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let c = [psp[1] * a[2] - psp[2] * a[1], psp[2] * a[0] - psp[0] * a[2], psp[0] * a[1] - psp[1] * a[0]];
            let nc = dot(&c, &c).sqrt();
            let random = Jones::random_su2(&mut rng).apply(C64::new(1.0, 0.0), C64::new(0.0, 0.0));
            let inputs = [random, jones_of(psp), jones_of(c.map(|v| v / nc))];
            for (k, (jx, jy)) in inputs.into_iter().enumerate() {
                let b2 = width_squared(&link, &p, jx, jy) / (tau0 * tau0);
                sums[k] += b2;
                squares[k] += b2 * b2;
            }
        }
        for (k, (name, b2)) in theory.iter().enumerate() {
            let n = links as f64;
            let mean = sums[k] / n;
            let sem = ((squares[k] / n - mean * mean) / n).sqrt();
            println!(
                "<DGD>/tau0 = {ratio}: {name}: b = {:.4} (theory {:.4}), excess {:.4} ± {:.4} against {:.4}",
                mean.sqrt(),
                b2.sqrt(),
                mean - 1.0,
                sem,
                b2 - 1.0
            );
            card.check(&json!({"input": name, "mean_dgd_over_tau0": ratio}), (mean - 1.0) / (b2 - 1.0));
        }
    }
    card.finish().unwrap();
}

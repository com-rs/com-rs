//! The PMD of `com_channel::optical::pmd` against the relations of J. P. Gordon and
//! H. Kogelnik, "PMD fundamentals: Polarization mode dispersion in optical fibers,"
//! *Proc. Natl. Acad. Sci.* 97(9), 4541–4550, 2000.
//!
//! The Jones matrix $U(\omega)$ of a link (`PmdLink::lumped_response`, in SU(2), so no
//! common phase: $t_{0} = 0$) gives the input PMD vector by $j U^{\dagger} U_{\omega} =
//! \frac{1}{2} \vec{\tau}_{s} \cdot \vec{\sigma}$ (eq. (5.18)) and the output one by
//! $\vec{\tau}_{t} = R \vec{\tau}_{s}$ (eq. (5.19)), $R$ the Müller matrix isomorphic to $U$,
//! $R_{ik} = \frac{1}{2} \operatorname{tr}(\sigma_{i} U \sigma_{k} U^{\dagger})$, with the Pauli
//! matrices of the paper ($\sigma_{1} = \operatorname{diag}(1, -1)$, $\sigma_{2}$ the
//! exchange, $\sigma_{3}$ with $\mp j$). A signal of spectrum $e(\omega)$ and input Stokes
//! vector $\hat{s}$ is delayed, its centre of gravity, by (eq. (5.29))
//!
//! $$
//! \tau_{g} = \frac{1}{2} \left\langle \vec{\tau}_{s}(\omega) \cdot \hat{s} \right\rangle,
//! $$
//!
//! the mean weighted by $|e(\omega)|^{2}$, and by $\pm \Delta\tau / 2$ at the principal states
//! of a narrow band (eqs. (5.30)–(5.31)). The concatenation of sections with uniform
//! rotations adds the PMD vectors of the sections as a random walk: $\mathbb{E}
//! |\vec{\tau}|^{2} = \sum_{k} \tau_{k}^{2}$. The case: `PmdLink::with_mean_dgd` with a mean
//! DGD of 10 ps over 50 sections; Gaussian pulses of 200 ps (full width at half maximum)
//! on 4096 samples at 256 GHz, applied by `PmdLink::apply`. The card is
//! `validation/cards/gordon2000_pmd.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use com_validation::pmd::{dot, jones_of, moments, pauli, pmd_vectors, stokes, trace_product};
use comrs::C64;
use comrs::channel::optical::pmd::{Jones, PmdLink};
use comrs::core::rng::Rng;
use comrs::core::signal::DualPol;
use serde_json::json;

const MEAN_DGD: f64 = 10e-12;
const SECTIONS: usize = 50;

/// The Müller matrix $R_{ik} = \frac{1}{2} \operatorname{tr}(\sigma_{i} U \sigma_{k} U^{\dagger})$.
fn muller(u: &Jones) -> [[f64; 3]; 3] {
    let p = pauli();
    let mut r = [[0.0; 3]; 3];
    for (i, si) in p.iter().enumerate() {
        for (k, sk) in p.iter().enumerate() {
            let usu = u.mul(&Jones(*sk)).mul(&u.adjoint());
            r[i][k] = 0.5 * trace_product(si, &usu.0).re;
        }
    }
    r
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The centre of gravity (s) of a dual-polarisation field.
fn centre(u: &DualPol, fs: f64) -> f64 {
    moments(u, fs).0
}

#[test]
fn the_pmd_of_the_sections_follows_gordon_and_kogelnik_2000() {
    let mut card = Card::load("gordon2000_pmd").unwrap();
    let mut rng = Rng::new(2000);
    let link = PmdLink::with_mean_dgd(MEAN_DGD, SECTIONS, &mut rng);
    // eqs. (5.18)-(5.19): the PMD vectors, Hermitian and rotated by R
    let (tin, tout, defect) = pmd_vectors(&link, 0.0);
    let r = muller(&link.lumped_response(&[0.0])[0]);
    let rotated = [dot(&r[0], &tin), dot(&r[1], &tin), dot(&r[2], &tin)];
    let mismatch = (0..3).map(|i| (rotated[i] - tout[i]).abs()).fold(0.0_f64, f64::max) / norm(&tin);
    card.check(&json!({"relation": "eq_5_18_hermitian_traceless"}), defect / norm(&tin));
    card.check(&json!({"relation": "eq_5_19_output_vector"}), mismatch);
    // eq. (5.29) in the time domain: Gaussian pulses of random polarisation through apply
    let (n, fs) = (4096, 256e9);
    let sigma = 200e-12 / (2.0 * (2.0 * 2f64.ln()).sqrt());
    let omega = comrs::math::fft::angular_frequencies(n, fs);
    let mut worst: f64 = 0.0;
    for _ in 0..8 {
        let s = Jones::random_su2(&mut rng).apply(C64::new(1.0, 0.0), C64::new(0.0, 0.0));
        let shat = stokes(s.0, s.1);
        let pulse: Vec<C64> = (0..n)
            .map(|k| {
                let t = (k as f64 - n as f64 / 2.0) / fs;
                C64::new((-t * t / (2.0 * sigma * sigma)).exp(), 0.0)
            })
            .collect();
        let mut u = DualPol { x: pulse.iter().map(|v| v * s.0).collect(), y: pulse.iter().map(|v| v * s.1).collect() };
        let before = centre(&u, fs);
        link.apply(&mut u, fs).unwrap();
        let measured = centre(&u, fs) - before;
        // the spectral mean of ½ τ_s(ω)·ŝ weighted by |e(ω)|²
        let spectrum = comrs::math::fft::fft(&pulse);
        let (mut num, mut den) = (0.0, 0.0);
        for (k, w) in omega.iter().enumerate() {
            let p = spectrum[k].norm_sqr();
            if p > 1e-30 * spectrum[0].norm_sqr() {
                num += p * 0.5 * dot(&pmd_vectors(&link, *w).0, &shat);
                den += p;
            }
        }
        worst = worst.max((measured - num / den).abs() / MEAN_DGD);
    }
    card.check(&json!({"relation": "eq_5_29_mean_delay"}), worst);
    // eqs. (5.30)-(5.31): at the principal states the delay is ±Δτ/2
    let dgd = norm(&tin);
    let p = [tin[0] / dgd, tin[1] / dgd, tin[2] / dgd];
    for (sign, name) in [(1.0, "slow"), (-1.0, "fast")] {
        let (x, y) = jones_of([sign * p[0], sign * p[1], sign * p[2]]);
        let pulse: Vec<C64> = (0..n)
            .map(|k| {
                let t = (k as f64 - n as f64 / 2.0) / fs;
                C64::new((-t * t / (2.0 * (4.0 * sigma).powi(2))).exp(), 0.0)
            })
            .collect();
        let mut u = DualPol { x: pulse.iter().map(|v| v * x).collect(), y: pulse.iter().map(|v| v * y).collect() };
        let before = centre(&u, fs);
        link.apply(&mut u, fs).unwrap();
        card.check(
            &json!({"relation": "eq_5_31_principal_state", "state": name}),
            (centre(&u, fs) - before) / (sign * dgd / 2.0),
        );
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn the_concatenation_of_sections_is_a_random_walk() {
    let mut card = Card::load("gordon2000_random_walk").unwrap();
    let mut rng = Rng::new(2001);
    let links = 4000;
    let (mut sq, mut sum_tk2, mut mean) = (0.0, 0.0, 0.0);
    let mut values = Vec::with_capacity(links);
    for _ in 0..links {
        let link = PmdLink::with_mean_dgd(MEAN_DGD, SECTIONS, &mut rng);
        let t = norm(&pmd_vectors(&link, 0.0).0);
        sq += t * t;
        mean += t;
        sum_tk2 += link.rms_dgd_s().powi(2);
        values.push(t * t);
    }
    let n = links as f64;
    let (sq, sum_tk2, mean) = (sq / n, sum_tk2 / n, mean / n);
    let var = values.iter().map(|v| (v - sq).powi(2)).sum::<f64>() / (n - 1.0);
    let z = (sq - sum_tk2) / (var / n).sqrt();
    println!(
        "E|τ|² / E Σ τ_k² = {:.4} (z = {z:.2}); mean DGD {:.3} ps for a target of {:.1} ps",
        sq / sum_tk2,
        mean * 1e12,
        MEAN_DGD * 1e12
    );
    card.check(&json!({"quantity": "mean_square_over_sum_of_squares"}), sq / sum_tk2);
    card.check(&json!({"quantity": "z_score"}), z);
    card.check(&json!({"quantity": "mean_dgd_over_target"}), mean / MEAN_DGD);
    card.finish().unwrap();
}

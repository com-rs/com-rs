//! The gamma–gamma model of atmospheric turbulence
//! (`comrs::channel::owc::atmospheric::GammaGamma`) against M. A. Al-Habash, L. C.
//! Andrews and R. L. Phillips, "Mathematical model for the irradiance probability
//! density function of a laser beam propagating through turbulent media," *Opt. Eng.*
//! 40(8), 1554–1562, 2001, for a negligible inner scale: the parameters $\alpha$ and
//! $\beta$ printed with their figures, here and in the paper,
//!
//! | case | here | paper |
//! |---|---|---|
//! | plane wave, $\sigma_{1}^{2} = 25$ (Fig. 3) | 8.05, 1.03 | 8.05, 1.03 |
//! | spherical wave, $\beta_{0}^{2} = 0.06$ (Fig. 6) | 34.28, 32.82 | 34.27, 32.82 |
//! | spherical wave, $\beta_{0}^{2} = 2$ (Fig. 7) | 2.21, 1.70 | 2.20, 1.70 |
//! | spherical wave, $\beta_{0}^{2} = 5$ (Fig. 8) | 2.12, 1.24 | 2.11, 1.24 |
//!
//! and the scintillation index of 1.214 quoted for $\sigma_{1}^{2} = 25$ (Sec. 4). They
//! follow from the exponent $5/6$ of the small-scale variance; with the $7/6$ printed
//! in eqs. (19) and (26) the plane wave would give $\beta = 4.29$. The density
//! integrates to 1 with unit mean, and the draws (a product of gamma variables) have
//! the predicted scintillation index. The inner-scale cases (Figs. 1, 2, 4, 5, 9–14)
//! are not modelled. The printed values and their tolerances are in the card
//! `validation/cards/alhabash2001_figs3_6_8.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use com_validation::card::Card;
use comrs::channel::owc::atmospheric::{GammaGamma, Wave};
use comrs::prelude::*;
use serde_json::{Value, json};

/// The inputs of a case of the card: a plane wave by its Rytov variance $\sigma_{1}^{2}$,
/// a spherical wave by $\beta_{0}^{2}$, as the figures give them.
fn inputs(wave: Wave, parameter: f64, quantity: &str) -> Value {
    match wave {
        Wave::Plane => json!({"wave": "plane", "sigma1_sq": parameter, "quantity": quantity}),
        Wave::Spherical => json!({"wave": "spherical", "beta0_sq": parameter, "quantity": quantity}),
    }
}

#[test]
fn al_habash_2001_parameters_figs_3_and_6_to_8() {
    let mut card = Card::load("alhabash2001_figs3_6_8").unwrap();
    // (the parameter of the figure, the wave, the Rytov variance): spherical waves are
    // given by β₀² = 0.4 σ₁²
    let cases = [
        (25.0, Wave::Plane, 25.0),
        (0.06, Wave::Spherical, 0.06 / 0.4),
        (2.0, Wave::Spherical, 2.0 / 0.4),
        (5.0, Wave::Spherical, 5.0 / 0.4),
    ];
    for (parameter, wave, rytov) in cases {
        let g = GammaGamma::from_rytov(rytov, wave).unwrap();
        card.check(&inputs(wave, parameter, "alpha"), g.alpha);
        card.check(&inputs(wave, parameter, "beta"), g.beta);
    }
    let g = GammaGamma::from_rytov(25.0, Wave::Plane).unwrap();
    card.check(&inputs(Wave::Plane, 25.0, "scintillation_index"), g.scintillation_index());
    card.finish().unwrap();
}

#[test]
fn al_habash_2001_density_and_draws() {
    for (rytov, wave) in [(25.0, Wave::Plane), (5.0, Wave::Spherical), (0.15, Wave::Spherical)] {
        let g = GammaGamma::from_rytov(rytov, wave).unwrap();
        // midpoint rule on a fine grid: total mass and mean
        let (step, n) = (1e-4, 300_000);
        let (mut mass, mut mean) = (0.0, 0.0);
        for k in 0..n {
            let i = (f64::from(k) + 0.5) * step;
            let p = g.pdf(i) * step;
            mass += p;
            mean += i * p;
        }
        assert!((mass - 1.0).abs() < 2e-3 && (mean - 1.0).abs() < 5e-3, "{wave:?} {rytov}: {mass} {mean}");
        let mut rng = Rng::new(2001);
        let draws: Vec<f64> = (0..400_000).map(|_| g.draw(&mut rng)).collect();
        let m = draws.iter().sum::<f64>() / draws.len() as f64;
        let si = draws.iter().map(|x| x * x).sum::<f64>() / draws.len() as f64 / (m * m) - 1.0;
        assert!((m - 1.0).abs() < 0.01, "{wave:?} {rytov}: mean {m}");
        assert!(
            (si / g.scintillation_index() - 1.0).abs() < 0.03,
            "{wave:?} {rytov}: {si} vs {}",
            g.scintillation_index()
        );
    }
}

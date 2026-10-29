//! # com-validation — simulated curves against theory
//!
//! The statistical tests of `tests/` compare Monte Carlo estimates with closed forms,
//! published figures and tables, and, for the channel codes, the AFF3CT runs recorded in
//! `aff3ct/`. The quick ones run with `cargo test`; the slow ones are `#[ignore]`d and run
//! by hand, before a release (or by the workflow `validation.yml`, started by hand):
//!
//! ```text
//! cargo test --release -p com-validation -- --ignored
//! ```
//!
//! # Acceptance rule
//!
//! The rule of every comparison is stated in its test (overlapping Wilson intervals for
//! AFF3CT, slopes for the diversity orders, the tolerance of a published table). Against
//! a closed form, each simulated point runs until at least [`MIN_ERRORS`] errors. The relative
//! standard error of the estimate is then about $1/\sqrt{N_{e}}$ (10 % for 100 errors),
//! and a point passes when it lies within three standard errors of theory
//! ([`within_three_sigma`]).

/// Errors to collect at every simulated point.
pub const MIN_ERRORS: u64 = 100;

/// Whether `errors` out of `total` trials is within three standard errors of the
/// probability `theory` (binomial standard error $\sqrt{p(1-p)/n}$).
pub fn within_three_sigma(errors: u64, total: u64, theory: f64) -> bool {
    let n = total as f64;
    let p = errors as f64 / n;
    let sigma = (theory * (1.0 - theory) / n).sqrt();
    (p - theory).abs() <= 3.0 * sigma
}

/// The channel LLRs $2 y/\sigma^{2}$ of the bits `c` sent in BPSK ($0 \mapsto +1$) over a
/// real AWGN channel of variance `noise_var`, the link of the code tests.
pub fn bpsk_awgn_llr(c: &[u8], noise_var: f64, rng: &mut comrs::core::rng::Rng) -> Vec<f64> {
    let s = noise_var.sqrt();
    c.iter().map(|b| 2.0 * (1.0 - 2.0 * f64::from(*b) + s * rng.gaussian()) / noise_var).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_acceptance_rule_is_three_standard_errors() {
        // p = 1e-2 over 10⁴ trials: σ ≈ 9.95e-4
        assert!(within_three_sigma(100, 10_000, 1e-2));
        assert!(within_three_sigma(129, 10_000, 1e-2));
        assert!(!within_three_sigma(131, 10_000, 1e-2));
    }
}

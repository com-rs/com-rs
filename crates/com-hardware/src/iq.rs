//! IQ imbalance of a transmitter or a receiver.
//!
//! - [`TxIq`]: the imbalance of an IQ modulator, $y = x + \beta x^{*}$ (the image
//!   $\beta x^{*}$ of the signal);
//! - [`RxIq`]: the imbalance of an IQ demodulator, a real $2 \times 2$ matrix acting on
//!   the in-phase and quadrature components.
//!
//! Both are [`Impair`] blocks on every stream of a [`Signal`], and act in place on a
//! slice with `apply`.
//!
//! # Model
//!
//! $$
//! \begin{bmatrix} \Re y \\\\ \Im y \end{bmatrix}
//! = \begin{bmatrix} t_{00} & t_{01} \\\\ t_{10} & t_{11} \end{bmatrix}
//! \begin{bmatrix} \Re x \\\\ \Im x \end{bmatrix}
//! $$
//!
//! # Examples
//! ```
//! use com_hardware::iq::{RxIq, TxIq};
//! use com_math::C64;
//! let mut x = vec![C64::new(1.0, 2.0)];
//! TxIq { beta: C64::new(0.1, 0.0) }.apply(&mut x);
//! assert!((x[0] - C64::new(1.1, 1.8)).norm() < 1e-15);   // x + 0.1 x*
//! let mut y = vec![C64::new(1.0, 2.0)];
//! RxIq { t: [[1.0, 0.0], [0.0, 1.0]] }.apply(&mut y);
//! assert_eq!(y[0], C64::new(1.0, 2.0));                  // the identity
//! ```
//!
//! # References
//!
//! 1. T. Schenk, *RF Imperfections in High-Rate Wireless Systems: Impact and Digital
//!    Compensation*, Springer, 2008, ch. 4.

use com_core::prelude::*;

/// Transmitter IQ imbalance $y = x + \beta x^{*}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TxIq {
    /// The image coefficient $\beta$ (0: no imbalance).
    pub beta: C64,
}

impl TxIq {
    /// Distort `x` in place.
    pub fn apply(&self, x: &mut [C64]) {
        for v in x {
            *v += self.beta * v.conj();
        }
    }
}

impl Impair for TxIq {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        x.map_streams_in_place(out, |_, y| self.apply(y));
        Ok(())
    }
}

/// Receiver IQ imbalance $[\Re y; \Im y] = T [\Re x; \Im x]$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RxIq {
    /// The real matrix $T$, row after row (the identity: no imbalance).
    pub t: [[f64; 2]; 2],
}

impl RxIq {
    /// Distort `x` in place.
    pub fn apply(&self, x: &mut [C64]) {
        let t = self.t;
        for v in x {
            *v = C64::new(t[0][0] * v.re + t[0][1] * v.im, t[1][0] * v.re + t[1][1] * v.im);
        }
    }
}

impl Impair for RxIq {
    fn impair_into(&mut self, x: &Signal, _rng: &mut SimRng, out: &mut Signal) -> Result<()> {
        x.map_streams_in_place(out, |_, y| self.apply(y));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_imbalances_are_the_identity() {
        let x = Signal::from_streams(&[(0..16).map(|n| C64::new(f64::from(n), -1.0)).collect()], Hertz(1.0)).unwrap();
        let mut rng = Rng::new(0);
        assert_eq!(TxIq { beta: C64::new(0.0, 0.0) }.impair(&x, &mut rng).unwrap(), x);
        assert_eq!(RxIq { t: [[1.0, 0.0], [0.0, 1.0]] }.impair(&x, &mut rng).unwrap(), x);
        let swap = RxIq { t: [[0.0, 1.0], [1.0, 0.0]] }.impair(&x, &mut rng).unwrap();
        assert_eq!(swap.stream(0)[3], C64::new(-1.0, 3.0));
    }
}

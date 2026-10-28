//! The OFDM resource grid: data and pilot resource elements.
//!
//! A grid holds $K$ subcarriers × $L$ OFDM symbols; a [`PilotPattern`] marks the
//! resource elements (REs) that carry known pilots, the others carry data. Pilots are
//! unit-energy QPSK symbols drawn from a seed (known to the receiver).
//!
//! | Pattern | Pilots | Suited to |
//! |---|---|---|
//! | [`PilotPattern::Block`] | every subcarrier of one symbol in `period` | slow fading, frequency-selective channels |
//! | [`PilotPattern::Comb`] | one subcarrier in `spacing`, in every symbol | fast fading |
//! | [`PilotPattern::Lattice`] | one subcarrier in `freq`, one symbol in `time` | both, at a lower overhead |
//!
//! The grid is stored symbol after symbol: RE $(k, l)$ is at index $l K + k$.
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
//! use com_math::C64;
//! let g = ResourceGrid { n_subcarriers: 12, n_symbols: 7, pattern: PilotPattern::Lattice { freq: 3, time: 3 }, seed: 1 };
//! assert_eq!(g.pilot_positions().len(), 4 * 3);
//! let d = vec![C64::new(1.0, 0.0); g.n_data()];
//! let grid = g.map(&d)?;
//! assert_eq!(g.extract(&grid), d);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. S. Coleri, M. Ergen, A. Puri and A. Bahai, "Channel estimation techniques based on
//!    pilot arrangement in OFDM systems," *IEEE Trans. Broadcast.*, 48(3), 223–229, 2002.

use com_core::rng::Rng;
use com_core::{Error, Result, ensure_shape};
use com_math::{C64, cis};

/// The positions of the pilots in the grid.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PilotPattern {
    /// Every subcarrier of the symbols $l$ with $l \bmod$ `period` $= 0$.
    Block {
        /// Pilot symbol period.
        period: usize,
    },
    /// The subcarriers $k$ with $k \bmod$ `spacing` $= 0$ of every symbol.
    Comb {
        /// Pilot subcarrier spacing.
        spacing: usize,
    },
    /// The subcarriers $k \bmod$ `freq` $= 0$ of the symbols $l \bmod$ `time` $= 0$.
    Lattice {
        /// Pilot subcarrier spacing.
        freq: usize,
        /// Pilot symbol period.
        time: usize,
    },
    /// No pilots (a receiver that knows the channel).
    None,
}

impl PilotPattern {
    /// The pattern of a name, `block`, `comb`, `lattice` or `none` (case-insensitive),
    /// with the spacings it uses: `period` for `block`, `spacing` for `comb`, `freq` and
    /// `time` for `lattice` (the others are ignored).
    ///
    /// # Errors
    /// [`Error::Config`] for another name or a spacing of zero.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::ofdm::grid::PilotPattern;
    /// assert_eq!(PilotPattern::from_name("comb", 4, 6, 3, 4)?, PilotPattern::Comb { spacing: 6 });
    /// assert!(PilotPattern::from_name("diamond", 4, 4, 3, 4).is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn from_name(name: &str, period: usize, spacing: usize, freq: usize, time: usize) -> Result<PilotPattern> {
        let positive = |v: usize, what: &str| {
            if v >= 1 { Ok(v) } else { Err(Error::Config(format!("the pilot {what} must be at least 1, got {v}"))) }
        };
        Ok(match name.to_ascii_lowercase().as_str() {
            "block" => PilotPattern::Block { period: positive(period, "period")? },
            "comb" => PilotPattern::Comb { spacing: positive(spacing, "spacing")? },
            "lattice" => PilotPattern::Lattice { freq: positive(freq, "freq")?, time: positive(time, "time")? },
            "none" => PilotPattern::None,
            _ => {
                return Err(Error::Config(format!("unknown pilot pattern {name:?}: block, comb, lattice or none")));
            }
        })
    }
}

/// A grid of `n_subcarriers` × `n_symbols` resource elements with pilots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceGrid {
    /// Subcarriers $K$ (the active subcarriers of the OFDM modulator).
    pub n_subcarriers: usize,
    /// OFDM symbols $L$ per grid.
    pub n_symbols: usize,
    /// The pilot pattern.
    pub pattern: PilotPattern,
    /// Seed of the pilot symbols.
    pub seed: u64,
}

impl ResourceGrid {
    /// Whether RE $(k, l)$ carries a pilot.
    pub fn is_pilot(&self, k: usize, l: usize) -> bool {
        match self.pattern {
            PilotPattern::Block { period } => l % period.max(1) == 0,
            PilotPattern::Comb { spacing } => k % spacing.max(1) == 0,
            PilotPattern::Lattice { freq, time } => k % freq.max(1) == 0 && l % time.max(1) == 0,
            PilotPattern::None => false,
        }
    }

    /// The pilot subcarriers of symbol `l` that belong to port `port` out of `n_ports`:
    /// the pilots of a symbol are dealt to the ports in turn (frequency-division
    /// orthogonal pilots; the other ports are silent there).
    pub fn port_pilots(&self, l: usize, port: usize, n_ports: usize) -> Vec<usize> {
        (0..self.n_subcarriers)
            .filter(|&k| self.is_pilot(k, l))
            .enumerate()
            .filter(|(j, _)| j % n_ports.max(1) == port)
            .map(|(_, k)| k)
            .collect()
    }

    /// The grid of port `port` out of `n_ports`: [`ResourceGrid::map`] with the pilots
    /// of the other ports set to zero.
    ///
    /// # Errors
    /// As [`ResourceGrid::map`].
    pub fn map_port(&self, data: &[C64], port: usize, n_ports: usize) -> Result<Vec<C64>> {
        let mut g = self.map(data)?;
        if n_ports > 1 {
            for l in 0..self.n_symbols {
                let mine = self.port_pilots(l, port, n_ports);
                for k in (0..self.n_subcarriers).filter(|&k| self.is_pilot(k, l) && !mine.contains(&k)) {
                    g[l * self.n_subcarriers + k] = C64::new(0.0, 0.0);
                }
            }
        }
        Ok(g)
    }

    /// The pilot symbol of RE $(k, l)$ (unit-energy QPSK).
    pub fn pilot(&self, k: usize, l: usize) -> C64 {
        let mut r = Rng::new(self.seed ^ ((l * self.n_subcarriers + k) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        cis(std::f64::consts::FRAC_PI_4 * (2 * r.below(4) + 1) as f64)
    }

    /// The pilot REs $(k, l)$, symbol after symbol.
    pub fn pilot_positions(&self) -> Vec<(usize, usize)> {
        (0..self.n_symbols)
            .flat_map(|l| (0..self.n_subcarriers).map(move |k| (k, l)))
            .filter(|&(k, l)| self.is_pilot(k, l))
            .collect()
    }

    /// The symbols that carry at least one pilot.
    pub fn pilot_symbols(&self) -> Vec<usize> {
        (0..self.n_symbols).filter(|&l| (0..self.n_subcarriers).any(|k| self.is_pilot(k, l))).collect()
    }

    /// Number of data REs per grid.
    pub fn n_data(&self) -> usize {
        self.n_subcarriers * self.n_symbols - self.pilot_positions().len()
    }

    /// Place `data` (exactly [`ResourceGrid::n_data`] symbols) and the pilots: the grid,
    /// symbol after symbol.
    ///
    /// # Errors
    /// [`Error::Shape`] for another number of data symbols.
    pub fn map(&self, data: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(data.len() == self.n_data(), || {
            format!("{} data symbols for a grid of {} data REs", data.len(), self.n_data())
        })?;
        let mut it = data.iter();
        let mut out = Vec::with_capacity(self.n_subcarriers * self.n_symbols);
        for l in 0..self.n_symbols {
            for k in 0..self.n_subcarriers {
                out.push(if self.is_pilot(k, l) {
                    self.pilot(k, l)
                } else {
                    *it.next().unwrap_or(&C64::new(0.0, 0.0))
                });
            }
        }
        Ok(out)
    }

    /// The data REs of a grid, in the order of [`ResourceGrid::map`].
    pub fn extract<T: Copy>(&self, grid: &[T]) -> Vec<T> {
        (0..self.n_symbols)
            .flat_map(|l| (0..self.n_subcarriers).map(move |k| (k, l)))
            .filter(|&(k, l)| !self.is_pilot(k, l))
            .map(|(k, l)| grid[l * self.n_subcarriers + k])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_place_their_pilots() {
        let block =
            ResourceGrid { n_subcarriers: 8, n_symbols: 6, pattern: PilotPattern::Block { period: 3 }, seed: 0 };
        assert_eq!(block.pilot_symbols(), vec![0, 3]);
        assert_eq!(block.n_data(), 8 * 4);
        let comb = ResourceGrid { pattern: PilotPattern::Comb { spacing: 4 }, ..block.clone() };
        assert_eq!(comb.pilot_positions().len(), 2 * 6);
        assert!((comb.pilot(4, 2).norm() - 1.0).abs() < 1e-12);
        assert!(block.map(&[C64::new(0.0, 0.0); 3]).is_err());
    }
}

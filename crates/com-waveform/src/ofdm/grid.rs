//! The OFDM resource grid: data and pilot resource elements.
//!
//! A grid holds $K$ subcarriers × $L$ OFDM symbols; a [`PilotPattern`] marks the
//! resource elements (REs) that carry known pilots, the others carry data. Pilots are
//! unit-energy QPSK symbols drawn from a seed (known to the receiver), except those of a
//! [`PilotTable`], which carries its own values (the sequences and boosts of a standard).
//!
//! | Pattern | Pilots | Suited to |
//! |---|---|---|
//! | [`PilotPattern::Block`] | every subcarrier of one symbol in `period` | slow fading, frequency-selective channels |
//! | [`PilotPattern::Comb`] | one subcarrier in `spacing`, in every symbol | fast fading |
//! | [`PilotPattern::Lattice`] | one subcarrier in `freq`, one symbol in `time` | both, at a lower overhead |
//! | [`PilotPattern::Explicit`] | the positions and values of a [`PilotTable`] | the pilots of a standard (scattered, staggered, boosted) |
//!
//! The grid is stored symbol after symbol: RE $(k, l)$ is at index $l K + k$;
//! [`ResourceGrid::data_indices`] lists the indices of the data REs in that order, the
//! one walk of the grid that [`ResourceGrid::map`], [`ResourceGrid::extract`] and the
//! OFDM link share. [`ResourceGrid::check`] refuses a spacing of zero (which
//! [`ResourceGrid::is_pilot`] would otherwise read as one) and a table of another size.
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
//! use com_math::C64;
//! let g = ResourceGrid { n_subcarriers: 12, n_symbols: 7, pattern: PilotPattern::Lattice { freq: 3, time: 3 }, seed: 1 };
//! assert_eq!(g.pilot_positions().len(), 4 * 3);
//! let d = vec![C64::new(1.0, 0.0); g.n_data()];
//! let grid = g.map(&d)?;
//! assert_eq!(g.extract(&grid)?, d);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. S. Coleri, M. Ergen, A. Puri and A. Bahai, "Channel estimation techniques based on
//!    pilot arrangement in OFDM systems," *IEEE Trans. Broadcast.*, 48(3), 223–229, 2002.

use com_core::{Error, Result, ensure, ensure_shape};
use com_math::C64;
use std::sync::Arc;

/// The positions of the pilots in the grid.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
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
    /// The pilots of a table: any positions, with their values.
    Explicit(Arc<PilotTable>),
    /// No pilots (a receiver that knows the channel).
    None,
}

/// The pilots of a grid given one by one: their positions and their values (the
/// sequences, phases and power boosts that a standard prescribes).
#[derive(Clone, Debug, PartialEq)]
pub struct PilotTable {
    n_subcarriers: usize,
    n_symbols: usize,
    /// The value of every RE, symbol after symbol (`None` for a data RE).
    values: Vec<Option<C64>>,
}

impl PilotTable {
    /// A table of `n_subcarriers` × `n_symbols` REs with pilots `((k, l), value)`; a
    /// position given twice keeps its last value.
    ///
    /// # Errors
    /// [`Error::Config`] for an empty grid, a position outside it or a pilot of zero or
    /// non-finite value.
    pub fn new(
        n_subcarriers: usize,
        n_symbols: usize,
        pilots: impl IntoIterator<Item = ((usize, usize), C64)>,
    ) -> Result<PilotTable> {
        ensure(n_subcarriers > 0 && n_symbols > 0, || {
            format!("a pilot table needs at least one RE, got {n_subcarriers} × {n_symbols}")
        })?;
        let mut values = vec![None; n_subcarriers * n_symbols];
        for ((k, l), v) in pilots {
            ensure(k < n_subcarriers && l < n_symbols, || {
                format!("pilot ({k}, {l}) outside the grid of {n_subcarriers} × {n_symbols}")
            })?;
            ensure(v.norm() > 0.0 && v.re.is_finite() && v.im.is_finite(), || {
                format!("the pilot ({k}, {l}) must be finite and non-zero, got {v}")
            })?;
            values[l * n_subcarriers + k] = Some(v);
        }
        Ok(PilotTable { n_subcarriers, n_symbols, values })
    }

    /// Subcarriers of the table.
    pub fn n_subcarriers(&self) -> usize {
        self.n_subcarriers
    }

    /// OFDM symbols of the table.
    pub fn n_symbols(&self) -> usize {
        self.n_symbols
    }

    /// The pilot of RE $(k, l)$, `None` for a data RE or a position outside the table.
    pub fn get(&self, k: usize, l: usize) -> Option<C64> {
        if k < self.n_subcarriers && l < self.n_symbols { self.values[l * self.n_subcarriers + k] } else { None }
    }

    /// Number of pilots.
    pub fn len(&self) -> usize {
        self.values.iter().filter(|v| v.is_some()).count()
    }

    /// Whether the table has no pilot.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PilotPattern {
    /// The pattern of a name, `block`, `comb`, `lattice` or `none` (case-insensitive),
    /// with the spacings it uses: `period` for `block`, `spacing` for `comb`, `freq` and
    /// `time` for `lattice`. The arguments a pattern does not use are ignored, and not
    /// checked (the parsers of the specifications and of the bindings give them all).
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
#[derive(Clone, Debug, PartialEq)]
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
    /// Validate the grid.
    ///
    /// # Errors
    /// [`Error::Config`] for an empty grid, a pilot period or spacing of zero, or a pilot
    /// table of another size than the grid.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
    /// let g = ResourceGrid { n_subcarriers: 12, n_symbols: 7, pattern: PilotPattern::Comb { spacing: 0 }, seed: 1 };
    /// assert!(g.check().is_err() && g.map(&[]).is_err());
    /// ```
    pub fn check(&self) -> Result<()> {
        let (k, l) = (self.n_subcarriers, self.n_symbols);
        ensure(k >= 1 && l >= 1, || format!("a resource grid needs at least one RE, got {k} × {l}"))?;
        let positive =
            |v: usize, what: &str| ensure(v >= 1, || format!("the pilot {what} must be at least 1, got {v}"));
        match &self.pattern {
            PilotPattern::Block { period } => positive(*period, "period"),
            PilotPattern::Comb { spacing } => positive(*spacing, "spacing"),
            PilotPattern::Lattice { freq, time } => positive(*freq, "freq").and_then(|()| positive(*time, "time")),
            PilotPattern::Explicit(t) => ensure(t.n_subcarriers == k && t.n_symbols == l, || {
                format!("a pilot table of {} × {} for a grid of {k} × {l}", t.n_subcarriers, t.n_symbols)
            }),
            PilotPattern::None => Ok(()),
        }
    }

    /// The grid of a pilot table, of its size.
    pub fn from_table(table: PilotTable) -> ResourceGrid {
        ResourceGrid {
            n_subcarriers: table.n_subcarriers,
            n_symbols: table.n_symbols,
            pattern: PilotPattern::Explicit(Arc::new(table)),
            seed: 0,
        }
    }

    /// Whether RE $(k, l)$ carries a pilot (a spacing of zero, which
    /// [`ResourceGrid::check`] refuses, is read as one).
    pub fn is_pilot(&self, k: usize, l: usize) -> bool {
        match &self.pattern {
            PilotPattern::Block { period } => l % (*period).max(1) == 0,
            PilotPattern::Comb { spacing } => k % (*spacing).max(1) == 0,
            PilotPattern::Lattice { freq, time } => k % (*freq).max(1) == 0 && l % (*time).max(1) == 0,
            PilotPattern::Explicit(t) => t.get(k, l).is_some(),
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

    /// The pilot symbol of RE $(k, l)$: the value of the table of an explicit pattern,
    /// unit-energy QPSK drawn from the seed otherwise.
    pub fn pilot(&self, k: usize, l: usize) -> C64 {
        if let PilotPattern::Explicit(t) = &self.pattern {
            if let Some(v) = t.get(k, l) {
                return v;
            }
        }
        com_mapping::source::qpsk(&mut com_core::rng::member(self.seed, (l * self.n_subcarriers + k) as u64))
    }

    /// Every RE $(k, l)$, symbol after symbol (the order of the storage).
    fn res(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.n_symbols).flat_map(move |l| (0..self.n_subcarriers).map(move |k| (k, l)))
    }

    /// The pilot REs $(k, l)$, symbol after symbol.
    pub fn pilot_positions(&self) -> Vec<(usize, usize)> {
        self.res().filter(|&(k, l)| self.is_pilot(k, l)).collect()
    }

    /// The indices $l K + k$ of the data REs, symbol after symbol: the order in which
    /// [`ResourceGrid::map`] places the data and [`ResourceGrid::extract`] takes them.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
    /// let g = ResourceGrid { n_subcarriers: 4, n_symbols: 2, pattern: PilotPattern::Comb { spacing: 2 }, seed: 0 };
    /// assert_eq!(g.data_indices(), vec![1, 3, 5, 7]);
    /// ```
    pub fn data_indices(&self) -> Vec<usize> {
        self.res().filter(|&(k, l)| !self.is_pilot(k, l)).map(|(k, l)| l * self.n_subcarriers + k).collect()
    }

    /// The symbols that carry at least one pilot.
    pub fn pilot_symbols(&self) -> Vec<usize> {
        (0..self.n_symbols).filter(|&l| (0..self.n_subcarriers).any(|k| self.is_pilot(k, l))).collect()
    }

    /// Number of data REs per grid (counted, nothing allocated).
    pub fn n_data(&self) -> usize {
        self.res().filter(|&(k, l)| !self.is_pilot(k, l)).count()
    }

    /// Place `data` (exactly [`ResourceGrid::n_data`] symbols) and the pilots: the grid,
    /// symbol after symbol.
    ///
    /// # Errors
    /// The errors of [`ResourceGrid::check`]; [`Error::Shape`] for another number of data
    /// symbols.
    pub fn map(&self, data: &[C64]) -> Result<Vec<C64>> {
        self.check()?;
        let n_data = self.n_data();
        ensure_shape(data.len() == n_data, || format!("{} data symbols for a grid of {n_data} data REs", data.len()))?;
        let mut it = data.iter();
        Ok(self
            .res()
            .map(
                |(k, l)| if self.is_pilot(k, l) { self.pilot(k, l) } else { *it.next().unwrap_or(&C64::new(0.0, 0.0)) },
            )
            .collect())
    }

    /// The data REs of a grid, in the order of [`ResourceGrid::map`].
    ///
    /// # Errors
    /// [`Error::Shape`] unless `grid` holds the $K L$ REs of the grid.
    pub fn extract<T: Copy>(&self, grid: &[T]) -> Result<Vec<T>> {
        let n = self.n_subcarriers * self.n_symbols;
        ensure_shape(grid.len() == n, || format!("{} REs for a grid of {n}", grid.len()))?;
        Ok(self.data_indices().into_iter().map(|i| grid[i]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The walks of the grid were written in `map`, `extract`, `n_data` and the OFDM
    /// link before `data_indices`: the same indices, counts and grids for every pattern.
    #[test]
    fn the_data_indices_are_the_former_walks() {
        let table = scattered(49, 8);
        let patterns = [
            PilotPattern::Block { period: 3 },
            PilotPattern::Comb { spacing: 4 },
            PilotPattern::Lattice { freq: 3, time: 2 },
            PilotPattern::Explicit(Arc::new(table)),
            PilotPattern::None,
        ];
        for pattern in patterns {
            let g = ResourceGrid { n_subcarriers: 49, n_symbols: 8, pattern, seed: 5 };
            let former: Vec<usize> = (0..8)
                .flat_map(|l| (0..49).map(move |k| (k, l)))
                .filter(|&(k, l)| !g.is_pilot(k, l))
                .map(|(k, l)| l * 49 + k)
                .collect();
            assert_eq!(g.data_indices(), former);
            assert_eq!(g.n_data(), 49 * 8 - g.pilot_positions().len());
            let data: Vec<C64> = (0..g.n_data()).map(|i| C64::new(i as f64, -1.0)).collect();
            let mut it = data.iter();
            let mut old = Vec::new();
            for l in 0..8 {
                for k in 0..49 {
                    old.push(if g.is_pilot(k, l) { g.pilot(k, l) } else { *it.next().unwrap() });
                }
            }
            assert_eq!(g.map(&data).unwrap(), old);
        }
        let bad = ResourceGrid {
            n_subcarriers: 12,
            n_symbols: 7,
            pattern: PilotPattern::Lattice { freq: 3, time: 0 },
            seed: 0,
        };
        assert!(bad.check().is_err());
        let wrong_size = ResourceGrid { n_subcarriers: 48, ..ResourceGrid::from_table(scattered(49, 8)) };
        assert!(wrong_size.check().is_err());
    }

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

    /// Scattered pilots in the manner of DVB-T: subcarriers $k \equiv 3 (l \bmod 4)
    /// \pmod{12}$, BPSK boosted to the amplitude 4/3.
    fn scattered(k_n: usize, l_n: usize) -> PilotTable {
        let pilots = (0..l_n).flat_map(move |l| {
            (0..k_n).filter(move |k| k % 12 == 3 * (l % 4)).map(move |k| {
                let sign = if (k / 12) % 2 == 0 { 1.0 } else { -1.0 };
                ((k, l), C64::new(sign * 4.0 / 3.0, 0.0))
            })
        });
        PilotTable::new(k_n, l_n, pilots).unwrap()
    }

    #[test]
    fn explicit_tables_carry_their_positions_and_values() {
        let g = ResourceGrid::from_table(scattered(49, 8));
        // 5 + 4 + 4 + 4 pilots every four symbols
        assert_eq!(g.pilot_positions().len(), 34);
        assert_eq!(g.pilot_symbols(), (0..8).collect::<Vec<_>>());
        assert!(g.is_pilot(15, 1) && !g.is_pilot(15, 0));
        assert_eq!(g.pilot(12, 0), C64::new(-4.0 / 3.0, 0.0));
        let data: Vec<C64> = (0..g.n_data()).map(|i| C64::new(i as f64, 1.0)).collect();
        let grid = g.map(&data).unwrap();
        assert_eq!(grid[12], C64::new(-4.0 / 3.0, 0.0));
        assert_eq!(g.extract(&grid).unwrap(), data);
        assert!(g.extract(&grid[1..]).is_err());
        let one = C64::new(1.0, 0.0);
        assert!(PilotTable::new(4, 2, [((4, 0), one)]).is_err());
        assert!(PilotTable::new(4, 2, [((0, 2), one)]).is_err());
        assert!(PilotTable::new(4, 2, [((0, 0), C64::new(0.0, 0.0))]).is_err());
        assert!(PilotTable::new(0, 2, []).is_err());
    }
}

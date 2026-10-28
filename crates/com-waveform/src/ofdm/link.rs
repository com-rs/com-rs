//! An OFDM link of several antennas: the frame of resource grids, its transmission and
//! its reception.
//!
//! [`transmit`] maps the symbols of every transmit antenna (port) on resource grids of
//! `n_symbols` OFDM symbols, the pilots dealt to the ports in turn
//! ([`ResourceGrid::map_port`]), pads the last grid with random QPSK at the power of a
//! layer, and modulates every grid. The samples have the power $K/(L N)$ ($K$ active
//! subcarriers, IFFT of size $L N$): a resource element of unit energy spreads over
//! $L N / K$ samples, so that the noise variance per sample is the noise variance per
//! demodulated resource element.
//!
//! [`OfdmFrame`] keeps what the receiver needs: [`OfdmFrame::received_grid`]
//! demodulates the grids of every receive antenna, [`OfdmFrame::estimate`] estimates the
//! channel from the pilots of every port, and [`OfdmFrame::detect`] detects every data
//! resource element with a [`Receiver`], spatial multiplexing or Alamouti's code over
//! pairs of resource elements.
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::Ofdm;
//! use com_waveform::ofdm::grid::{PilotPattern, ResourceGrid};
//! use com_waveform::ofdm::link::transmit;
//! use com_mimo::receiver::{DetectorKind, Receiver};
//! use com_mapping::constellation::Constellation;
//! use com_core::rng::Rng;
//! use com_math::C64;
//! use ndarray::Array2;
//! let ofdm = Ofdm { n_cp: 4, oversampling: 1, ..Ofdm::new(32, 24)? };
//! let grid = ResourceGrid { n_subcarriers: 24, n_symbols: 4, pattern: PilotPattern::None, seed: 0 };
//! let c = Constellation::qam(4)?.normalized();
//! let x: Vec<C64> = (0..90).map(|i| c.points()[i % 4]).collect();
//! let (tx, frame) = transmit(&ofdm, &grid, vec![x.clone()], &mut Rng::new(1))?;
//! let rx = Receiver::new(DetectorKind::Zf, c, false)?;
//! let eye = |_: usize, _: &[Vec<C64>]| Ok(vec![C64::new(1.0, 0.0); 24 * 4]);
//! let out = frame.detect(&tx, eye, &rx, 1.0, false, 1e-6)?;
//! assert!(out.symbols[..90].iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-12));
//! # Ok::<(), com_core::Error>(())
//! ```

use super::Ofdm;
use super::estimation::{FrequencyInterpolation, estimate_port};
use super::grid::ResourceGrid;
use com_core::rng::Rng;
use com_core::{Result, ensure, ensure_shape};
use com_math::C64;
use com_mimo::receiver::{Receiver, SoftOutput};
use ndarray::Array2;

/// An OFDM frame: what the receiver needs to undo [`transmit`].
#[derive(Clone, Debug, PartialEq)]
pub struct OfdmFrame {
    /// The modulator.
    pub ofdm: Ofdm,
    /// The resource grid of every group of `n_symbols` OFDM symbols.
    pub grid: ResourceGrid,
    /// Grids in the frame.
    pub n_grids: usize,
    /// Transmit antennas (ports).
    pub n_ports: usize,
    /// Data symbols per antenna before the padding of the last grid.
    pub n_data: usize,
    /// Amplitude of the time samples, $\sqrt{K/(L N)}$.
    pub scale: f64,
}

/// Map the symbols of every antenna on resource grids and modulate them (see the
/// module): the samples of every antenna and the frame.
///
/// # Errors
/// [`com_core::Error::Shape`] for antennas of different lengths,
/// [`com_core::Error::Config`] for a grid without data resource element, and the errors
/// of the modulator.
pub fn transmit(
    ofdm: &Ofdm,
    grid: &ResourceGrid,
    streams: Vec<Vec<C64>>,
    rng: &mut Rng,
) -> Result<(Vec<Vec<C64>>, OfdmFrame)> {
    let n_ports = streams.len();
    ensure(n_ports >= 1, || "at least one antenna".into())?;
    let n_data = streams[0].len();
    ensure_shape(streams.iter().all(|x| x.len() == n_data), || {
        format!("the antennas carry different lengths: {:?}", streams.iter().map(Vec::len).collect::<Vec<_>>())
    })?;
    let per_grid = grid.n_data();
    ensure(per_grid > 0, || "the grid has no data resource element".into())?;
    let n_grids = n_data.div_ceil(per_grid).max(1);
    let (k, ln) = (ofdm.n_active, ofdm.fft_len());
    let scale = (k as f64 / ln as f64).sqrt();
    // the padding of the last grid: random QPSK at the power of a layer
    let pad_amp = std::f64::consts::FRAC_1_SQRT_2 / (n_ports as f64).sqrt();
    let mut out = Vec::with_capacity(n_ports);
    for (port, x) in streams.into_iter().enumerate() {
        let mut data = x;
        while data.len() < n_grids * per_grid {
            data.push(C64::new(
                if rng.bit() == 0 { pad_amp } else { -pad_amp },
                if rng.bit() == 0 { pad_amp } else { -pad_amp },
            ));
        }
        let mut time = Vec::with_capacity(n_grids * grid.n_symbols * ofdm.block_len());
        for chunk in data.chunks(per_grid) {
            let g = grid.map_port(chunk, port, n_ports)?;
            for sym in g.chunks(k) {
                time.extend(ofdm.with_prefix(&ofdm.to_time(sym))?.into_iter().map(|v| v * scale));
            }
        }
        out.push(time);
    }
    let frame = OfdmFrame { ofdm: ofdm.clone(), grid: grid.clone(), n_grids, n_ports, n_data, scale };
    Ok((out, frame))
}

impl OfdmFrame {
    /// The data resource elements of a grid, symbol after symbol (indices `l K + k`).
    pub fn data_res(&self) -> Vec<usize> {
        let (k, l_n) = (self.grid.n_subcarriers, self.grid.n_symbols);
        (0..l_n)
            .flat_map(|l| (0..k).map(move |kk| (kk, l)))
            .filter(|&(kk, l)| !self.grid.is_pilot(kk, l))
            .map(|(kk, l)| l * k + kk)
            .collect()
    }

    /// The sample at the middle of every OFDM symbol of grid `g` (after the prefix): the
    /// instant at which a genie-aided receiver takes the channel.
    pub fn symbol_middles(&self, g: usize) -> Vec<usize> {
        let (bl, cp, ln) = (self.ofdm.block_len(), self.ofdm.oversampling * self.ofdm.n_cp, self.ofdm.fft_len());
        (0..self.grid.n_symbols).map(|l| (g * self.grid.n_symbols + l) * bl + cp + ln / 2).collect()
    }

    /// The FFT bins of the active subcarriers.
    pub fn active_bins(&self) -> Vec<usize> {
        (0..self.ofdm.n_active).map(|i| self.ofdm.bin(i)).collect()
    }

    /// The demodulated grid `g` of every receive antenna, `[rx][l K + k]`.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if a stream is shorter than the frame.
    pub fn received_grid(&self, streams: &[Vec<C64>], g: usize) -> Result<Vec<Vec<C64>>> {
        let (k, l_n) = (self.grid.n_subcarriers, self.grid.n_symbols);
        let (bl, cp, ln) = (self.ofdm.block_len(), self.ofdm.oversampling * self.ofdm.n_cp, self.ofdm.fft_len());
        let mut rx = Vec::with_capacity(streams.len());
        for x in streams {
            let mut grid = Vec::with_capacity(k * l_n);
            for l in 0..l_n {
                let start = (g * l_n + l) * bl + cp;
                ensure_shape(start + ln <= x.len(), || {
                    format!("the signal of {} samples is shorter than the OFDM frame ({} samples)", x.len(), start + ln)
                })?;
                let block: Vec<C64> = x[start..start + ln].iter().map(|v| v / self.scale).collect();
                grid.extend(self.ofdm.to_active(&block));
            }
            rx.push(grid);
        }
        Ok(rx)
    }

    /// The channel of every resource element estimated from the pilots of every port,
    /// `[re][rx][tx]` flattened, from the received grids `rx` ([`OfdmFrame::received_grid`]).
    ///
    /// # Errors
    /// The errors of [`estimate_port`].
    pub fn estimate(&self, rx: &[Vec<C64>], noise_var: f64, method: &FrequencyInterpolation) -> Result<Vec<C64>> {
        let (nr, nt) = (rx.len(), self.n_ports);
        let mut h = vec![C64::new(0.0, 0.0); self.grid.n_subcarriers * self.grid.n_symbols * nr * nt];
        for (r, y) in rx.iter().enumerate() {
            for t in 0..nt {
                let est = estimate_port(&self.grid, y, noise_var, method, t, nt)?;
                for (re, v) in est.into_iter().enumerate() {
                    h[(re * nr + r) * nt + t] = v;
                }
            }
        }
        Ok(h)
    }

    /// Detect every data resource element of the frame received on `streams`: the
    /// channel of grid `g` is `channel(g, rx)` (`[re][rx][tx]` flattened, from the
    /// received grids `rx`), multiplied by `gain` (the amplitude of a layer); with
    /// `alamouti`, the pairs of consecutive data resource elements carry the blocks of
    /// Alamouti's code. The padding of the last grid is left out.
    ///
    /// # Errors
    /// The errors of `channel` and of [`Receiver::detect_uses`] (an odd number of data
    /// resource elements with Alamouti's code).
    pub fn detect<F>(
        &self,
        streams: &[Vec<C64>],
        mut channel: F,
        receiver: &Receiver,
        gain: f64,
        alamouti: bool,
        noise_var: f64,
    ) -> Result<SoftOutput>
    where
        F: FnMut(usize, &[Vec<C64>]) -> Result<Vec<C64>>,
    {
        let (nr, nt) = (streams.len(), self.n_ports);
        let data_res = self.data_res();
        let mut out = SoftOutput::default();
        for g in 0..self.n_grids {
            let rx = self.received_grid(streams, g)?;
            let h = channel(g, &rx)?;
            let used = self.n_data.saturating_sub(g * data_res.len()).min(data_res.len());
            receiver
                .detect_uses(
                    data_res[..used].iter().copied(),
                    alamouti,
                    noise_var,
                    |re| Ok(Array2::from_shape_fn((nr, nt), |(r, t)| h[(re * nr + r) * nt + t] * gain)),
                    |re| rx.iter().map(|v| v[re]).collect(),
                    &mut out,
                )
                .map_err(|e| e.context("the data resource elements"))?;
        }
        Ok(out)
    }
}

//! Nyquist WDM fields for the fibre tests: channels of sinc pulses (square spectra) built
//! in the frequency domain on a periodic block, and the matched filter of the centre
//! channel, which keeps its band and samples it at the symbol rate.

use comrs::C64;
use comrs::math::fft::{fft, ifft};

/// A block of `symbols` symbols per channel at `sps` samples per symbol, the channels
/// `slot_bins` frequency bins apart (the spacing over the symbol rate times `symbols`,
/// an integer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NyquistGrid {
    /// Symbols per channel.
    pub symbols: usize,
    /// Samples per symbol.
    pub sps: usize,
    /// Bins between two channels.
    pub slot_bins: i64,
}

impl NyquistGrid {
    /// The samples of the block.
    pub fn len(&self) -> usize {
        self.symbols * self.sps
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The field (√W) of a channel of Nyquist pulses carrying the symbols `a` (unit
    /// energy) in the slot `slot` with the power `p` (W): the spectrum of the symbols
    /// placed in the band of the channel.
    pub fn channel(&self, a: &[C64], slot: i64, p: f64) -> Vec<C64> {
        let (n, l) = (self.symbols as i64, self.len() as i64);
        let spectrum = fft(a);
        let mut x = vec![C64::new(0.0, 0.0); l as usize];
        for j in -n / 2..n / 2 {
            x[(j + slot * self.slot_bins).rem_euclid(l) as usize] = spectrum[j.rem_euclid(n) as usize];
        }
        let scale = p.sqrt() * self.sps as f64;
        ifft(&x).into_iter().map(|v| v * scale).collect()
    }

    /// The symbols of the centre channel of the field `y`, normalised by the amplitude of
    /// a channel of power `p`.
    pub fn matched_filter(&self, y: &[C64], p: f64) -> Vec<C64> {
        let (n, l) = (self.symbols as i64, y.len() as i64);
        let spectrum = fft(y);
        let band: Vec<C64> =
            (0..n).map(|i| if i < n / 2 { i } else { i - n }).map(|j| spectrum[j.rem_euclid(l) as usize]).collect();
        let scale = 1.0 / (p.sqrt() * self.sps as f64);
        ifft(&band).into_iter().map(|v| v * scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_matched_filter_returns_the_symbols_of_the_centre_channel() {
        let grid = NyquistGrid { symbols: 64, sps: 8, slot_bins: 100 };
        let a: Vec<C64> = (0..64).map(|k| C64::from_polar(1.0, 0.7 * k as f64)).collect();
        let b: Vec<C64> = (0..64).map(|k| C64::from_polar(1.0, -0.3 * k as f64)).collect();
        let field: Vec<C64> =
            grid.channel(&a, 0, 2e-3).iter().zip(grid.channel(&b, 1, 2e-3)).map(|(u, v)| u + v).collect();
        let power: f64 = grid.channel(&a, 0, 2e-3).iter().map(C64::norm_sqr).sum::<f64>() / grid.len() as f64;
        assert!((power - 2e-3).abs() < 1e-15);
        let r = grid.matched_filter(&field, 2e-3);
        assert!(r.iter().zip(&a).all(|(u, v)| (u - v).norm() < 1e-12));
    }
}

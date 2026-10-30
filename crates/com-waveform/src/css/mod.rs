//! Chirp spread spectrum: the frequency-shift chirp modulation of LoRa.
//!
//! # Model
//!
//! A symbol $a \in \{0, \dots, M - 1\}$, $M = 2^{\mathrm{SF}}$ with the spreading factor
//! $\mathrm{SF}$, is a chirp of $M$ chips over the bandwidth $B$ whose initial frequency
//! is shifted by $aB/M$ and folded at $B/2$. Its complex envelope over
//! $0 \le t < T_{s} = M/B$ is (M. Chiani and A. Elzanaty, *IEEE Internet Things J.*
//! 6(5), 2019, eq. (5))
//!
//! $$x(t; a) = \exp\left\\{j 2\pi B t \left(\frac{a}{M} - \frac{1}{2} + \frac{Bt}{2M} -
//! u\left(t - \frac{M - a}{B}\right)\right)\right\\},$$
//!
//! a memoryless continuous-phase modulation. At one sample per chip the fold vanishes
//! and $x_{a}\[n\] = e^{j2\pi(n^{2}/(2M) + (a/M - 1/2) n)}$ (their eq. (13); eq. (1) of
//! Tapparel et al. 2020, eq. (2) of Afisiadis et al. 2020). [`Css`] samples it
//! `oversampling` times per chip; [`Waveform`] also gives, for comparison, the discrete
//! chirp of L. Vangelista, *IEEE Signal Process. Lett.* 24(12), 2017, eq. (3),
//! $e^{j2\pi((a + n) \bmod M) n/M}$, whose quadratic phase lacks the factor $1/2$
//! (Chiani and Elzanaty, Sec. II-A), and the FSK of the same cardinality of his eq. (17).
//!
//! The modulator tabulates the chirp of symbol 0 and the tones $e^{j 2\pi k/(JM)}$ once
//! per call: the sample $n$ of symbol $a$ is the product of the base chirp, the tone
//! $(a n) \bmod JM$ and, after the fold, the conjugate of the tone $(M n) \bmod JM$,
//! within a few units of the last place of the direct evaluation of the phase.
//!
//! # Demodulation
//!
//! The receiver multiplies every symbol by the conjugate of the chirp of symbol 0 and
//! takes a DFT (Vangelista, eq. (16)): the dechirped symbol $a$ is a tone in bin $a$
//! (and, when oversampled, in bin $a - M$ after the fold), so the noncoherent decision
//! is the bin of largest energy, the maximum-likelihood detection of these orthogonal
//! signals with an unknown phase. [`theory`] gives its error rates, the spectrum and
//! the effect of a same-SF interferer.
//!
//! # SNR
//!
//! The SNR is that of a sample at one sample per chip (the signal power over the noise
//! power in the band $B$), the abscissa of the figures of Vangelista (2017), Afisiadis
//! et al. (2020) and Tapparel et al. (2020); the SNR per symbol is $M$ times larger.
//!
//! # Trademarks and patents
//!
//! LoRa is a trademark of Semtech Corporation, and the modulation is covered by patents
//! (F. Sforza, US 8,406,275, 2013; O. B. A. Seller and N. Sornin, US 9,252,834, 2016).
//! This module models the published descriptions above for simulation and research; it
//! is not the code of any vendor nor a certified implementation.
//!
//! # Examples
//! ```
//! use com_waveform::css::Css;
//! let css = Css::lora(7)?;
//! let s = css.modulate(&[0, 5, 127])?;
//! assert_eq!(s.len(), 3 * 128);
//! assert!(s.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));   // constant envelope
//! assert_eq!(css.demodulate(&s)?, vec![0, 5, 127]);
//! # Ok::<(), com_core::Error>(())
//! ```

pub mod theory;

use com_core::{Result, ensure, ensure_shape};
use com_math::{C64, cis, fft};
use std::f64::consts::TAU;

/// The family of waveforms (see the module).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Waveform {
    /// The chirp of LoRa (Chiani and Elzanaty 2019, eq. (5)), at any oversampling.
    LoRa,
    /// The discrete chirp of Vangelista (2017), eq. (3), one sample per chip.
    Vangelista,
    /// The FSK of the same cardinality, Vangelista (2017), eq. (17), one sample per chip.
    Fsk,
}

impl std::str::FromStr for Waveform {
    type Err = com_core::Error;

    /// `lora`, `vangelista` or `fsk` (case-insensitive).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for another name.
    fn from_str(s: &str) -> Result<Waveform> {
        match s.to_ascii_lowercase().as_str() {
            "lora" => Ok(Waveform::LoRa),
            "vangelista" => Ok(Waveform::Vangelista),
            "fsk" => Ok(Waveform::Fsk),
            _ => Err(com_core::Error::Config(format!("the waveform must be lora, vangelista or fsk, got {s:?}"))),
        }
    }
}

/// A chirp spread-spectrum link (see the module): a configuration with public fields,
/// built by [`Css::lora`] and changed by struct update, checked by [`Css::check`] when it
/// is used.
#[derive(Clone, Debug, PartialEq)]
pub struct Css {
    /// Spreading factor: `M = 2^SF` chips per symbol.
    pub sf: u32,
    /// Samples per chip (1: the chip rate `B`).
    pub oversampling: usize,
    /// The family of waveforms.
    pub waveform: Waveform,
}

impl Css {
    /// The LoRa waveform at one sample per chip.
    ///
    /// # Errors
    /// As [`Css::check`]: the spreading factor lies in 1..=16.
    ///
    /// # Examples
    /// ```
    /// use com_waveform::css::Css;
    /// let css = Css { oversampling: 4, ..Css::lora(7)? };
    /// assert_eq!(css.symbol_len(), 4 * 128);
    /// assert!(Css::lora(17).is_err());
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn lora(sf: u32) -> Result<Css> {
        let css = Css { sf, oversampling: 1, waveform: Waveform::LoRa };
        css.check()?;
        Ok(css)
    }

    /// The number of symbols (and of chips per symbol) `M = 2^SF`.
    pub fn m(&self) -> usize {
        1 << self.sf
    }

    /// Samples per symbol, `J M`.
    pub fn symbol_len(&self) -> usize {
        self.oversampling * self.m()
    }

    /// Check the configuration.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless the spreading factor lies in 1..=16 and the
    /// oversampling factor is at least 1 (exactly 1 for the FSK reference).
    pub fn check(&self) -> Result<()> {
        ensure((1..=16).contains(&self.sf), || format!("the spreading factor must lie in 1..=16, got {}", self.sf))?;
        ensure(self.oversampling >= 1, || "the oversampling factor must be at least 1".into())?;
        ensure(self.waveform == Waveform::LoRa || self.oversampling == 1, || {
            format!("the {:?} waveform is defined at one sample per chip", self.waveform)
        })
    }

    /// The phase (rad) of symbol `a` at the time `t` counted in chips, `0 ≤ t < M`.
    fn phase(&self, a: usize, t: f64) -> f64 {
        let m = self.m() as f64;
        let a = a as f64;
        match self.waveform {
            Waveform::LoRa => {
                let fold = if t >= m - a { 1.0 } else { 0.0 };
                TAU * t * (a / m - 0.5 + t / (2.0 * m) - fold)
            }
            // e^{j2π((a + n) mod M) n/M} = e^{j2π(n² + a n)/M} at the integer n
            Waveform::Vangelista => TAU * (t * t + a * t) / m,
            Waveform::Fsk => TAU * a * t / m,
        }
    }

    /// The `J M` samples of symbol `a`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters or `a ≥ M`.
    pub fn symbol(&self, a: usize) -> Result<Vec<C64>> {
        self.modulate(&[a])
    }

    /// The samples of a sequence of symbols (the chirps tabulated once, see the module).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters or a symbol `≥ M`.
    pub fn modulate(&self, symbols: &[usize]) -> Result<Vec<C64>> {
        self.check()?;
        let m = self.m();
        if let Some(a) = symbols.iter().find(|&&a| a >= m) {
            return Err(com_core::Error::Config(format!("symbol {a} does not lie below M = {m}")));
        }
        let chirps = Chirps::new(self);
        let mut out = Vec::with_capacity(symbols.len() * self.symbol_len());
        for &a in symbols {
            chirps.extend_symbol(&mut out, a);
        }
        Ok(out)
    }

    /// The energy of the `M` candidate symbols of every received symbol: the squared
    /// magnitudes of the DFT of the dechirped samples, bins `a` and, when oversampled,
    /// `a - M` (after the fold) added.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for invalid parameters, [`com_core::Error::Shape`] for
    /// a number of samples that is not a multiple of `J M`.
    pub fn energies(&self, r: &[C64]) -> Result<Vec<f64>> {
        self.check()?;
        let (m, l) = (self.m(), self.symbol_len());
        ensure_shape(r.len() % l == 0, || format!("{} samples is not a multiple of J M = {l}", r.len()))?;
        let down: Vec<C64> = Chirps::new(self).base.iter().map(C64::conj).collect();
        let mut out = Vec::with_capacity(r.len() / l * m);
        for block in r.chunks(l) {
            let y = fft::fft(&block.iter().zip(&down).map(|(a, b)| a * b).collect::<Vec<_>>());
            if self.oversampling == 1 {
                out.extend(y.iter().map(C64::norm_sqr));
            } else {
                out.extend((0..m).map(|k| y[k].norm_sqr() + y[(k + l - m) % l].norm_sqr()));
            }
        }
        Ok(out)
    }

    /// The noncoherent decisions: the candidate of largest energy in every symbol.
    ///
    /// # Errors
    /// As [`Css::energies`].
    pub fn demodulate(&self, r: &[C64]) -> Result<Vec<usize>> {
        let m = self.m();
        Ok(self.energies(r)?.chunks(m).map(|e| com_math::argmax(e.iter().copied()).map_or(0, |(k, _)| k)).collect())
    }

    /// One symbol of an interferer of the same spreading factor, not synchronised: the
    /// end of its symbol `s1` then the start of its symbol `s2`, which begins `tau` chips
    /// after the symbol of interest, `0 ≤ tau < M` (Afisiadis et al. 2020, eq. (21)), at
    /// one sample per chip.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for another waveform than LoRa at one sample per chip,
    /// symbols `≥ M` or `tau` outside `[0, M)`.
    pub fn interferer(&self, s1: usize, s2: usize, tau: f64) -> Result<Vec<C64>> {
        self.check()?;
        let m = self.m();
        ensure(self.waveform == Waveform::LoRa && self.oversampling == 1, || {
            "the interferer is a LoRa signal at one sample per chip".into()
        })?;
        ensure(s1 < m && s2 < m, || format!("the symbols must lie below M = {m}"))?;
        ensure((0.0..m as f64).contains(&tau), || format!("the offset must lie in [0, {m}), got {tau}"))?;
        let mf = m as f64;
        let x = |t: f64, s: usize| cis(TAU * (t * t / (2.0 * mf) + t * (s as f64 / mf - 0.5)));
        Ok((0..m)
            .map(|n| {
                let n = n as f64;
                if n < tau.ceil() { x(n + mf - tau, s1) } else { x(n - tau, s2) }
            })
            .collect())
    }
}

/// The tables of [`Css::modulate`]: the chirp of symbol 0 and the tones
/// $e^{j 2\pi k/(JM)}$, $k < JM$.
struct Chirps {
    /// The `J M` samples of symbol 0, $e^{j\varphi(0, n/J)}$.
    base: Vec<C64>,
    /// $e^{j 2\pi k/(JM)}$.
    tone: Vec<C64>,
    m: usize,
    j: usize,
    waveform: Waveform,
}

impl Chirps {
    /// The tables of a checked configuration.
    fn new(css: &Css) -> Chirps {
        let (m, j, l) = (css.m(), css.oversampling, css.symbol_len());
        let jf = j as f64;
        let base = (0..l).map(|n| cis(css.phase(0, n as f64 / jf))).collect();
        let tone = (0..l).map(|k| cis(TAU * k as f64 / l as f64)).collect();
        Chirps { base, tone, m, j, waveform: css.waveform }
    }

    /// Append the samples of symbol `a < M`: $x_{a}\[n\] = x_{0}\[n\]\, e^{j 2\pi a n/(JM)}$,
    /// times $e^{-j 2\pi n / J}$ after the fold of LoRa ($n \ge J(M - a)$); the FSK
    /// has the base 1.
    fn extend_symbol(&self, out: &mut Vec<C64>, a: usize) {
        let l = self.base.len();
        let fold = self.j * (self.m - a);
        out.extend(self.base.iter().enumerate().map(|(n, b)| {
            let v = b * self.tone[(a * n) % l];
            if self.waveform == Waveform::LoRa && n >= fold { v * self.tone[(self.m * n) % l].conj() } else { v }
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chirps were computed from their phase, sample after sample and symbol after
    /// symbol, before they were tabulated: the same samples within $10^{-11}$ up to
    /// SF 12, symbol 0 exactly. Against the phase reduced exactly modulo
    /// $2\pi$ (an integer numerator over $2M$, LoRa at one sample per chip), the
    /// tabulated chirps are the closer: the direct evaluation carries the rounding of a
    /// phase of up to $\pi M$ rad ($4.2 \cdot 10^{-12}$ at SF 12, $6.3 \cdot 10^{-13}$
    /// tabulated).
    #[test]
    fn the_tabulated_chirps_are_the_direct_ones() {
        for (sf, oversampling, waveform) in [
            (7u32, 1usize, Waveform::LoRa),
            (7, 4, Waveform::LoRa),
            (12, 1, Waveform::LoRa),
            (9, 8, Waveform::LoRa),
            (7, 1, Waveform::Vangelista),
            (10, 1, Waveform::Vangelista),
            (7, 1, Waveform::Fsk),
        ] {
            let css = Css { sf, oversampling, waveform };
            let m = css.m();
            let jf = oversampling as f64;
            let symbols: Vec<usize> = (0..m).step_by((m / 64).max(1)).chain([m - 1]).collect();
            let fast = css.modulate(&symbols).unwrap();
            for (i, &a) in symbols.iter().enumerate() {
                let direct: Vec<C64> = (0..css.symbol_len()).map(|n| cis(css.phase(a, n as f64 / jf))).collect();
                let got = &fast[i * css.symbol_len()..(i + 1) * css.symbol_len()];
                let err = got.iter().zip(&direct).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max);
                assert!(err < 1e-11, "SF {sf} J {oversampling} {waveform:?} symbol {a}: {err}");
                assert!(a != 0 || err == 0.0, "{waveform:?} symbol 0: {err}");
            }
        }
        assert!(Css::lora(7).unwrap().modulate(&[3, 128]).is_err());
        for sf in [7u32, 10, 12] {
            let css = Css::lora(sf).unwrap();
            let m = css.m();
            let (mut direct, mut tabulated) = (0.0f64, 0.0f64);
            for a in (0..m).step_by(m / 32) {
                let fast = css.symbol(a).unwrap();
                for (n, v) in fast.iter().enumerate() {
                    // 2π (n² + (2a - M) n - 2 M n [n ≥ M - a]) / (2M), the numerator reduced
                    let fold = usize::from(n >= m - a);
                    let num = (n * n + 2 * a * n + 4 * m * n - m * n - 2 * m * n * fold) % (2 * m);
                    let exact = cis(TAU * num as f64 / (2 * m) as f64);
                    direct = direct.max((cis(css.phase(a, n as f64)) - exact).norm());
                    tabulated = tabulated.max((v - exact).norm());
                }
            }
            assert!(tabulated < direct, "SF {sf}: tabulated {tabulated:e}, direct {direct:e}");
        }
    }

    #[test]
    fn the_waveforms_are_orthogonal_and_the_fft_receiver_is_the_correlator_bank() {
        // Vangelista 2017, eqs. (9) and (16); Chiani and Elzanaty 2019, Property 3
        for waveform in [Waveform::LoRa, Waveform::Vangelista, Waveform::Fsk] {
            let css = Css { sf: 5, oversampling: 1, waveform };
            let m = css.m();
            let x: Vec<Vec<C64>> = (0..m).map(|a| css.symbol(a).unwrap()).collect();
            for i in 0..m {
                for q in 0..m {
                    let c: C64 = x[i].iter().zip(&x[q]).map(|(a, b)| a * b.conj()).sum::<C64>() / m as f64;
                    let want = if i == q { 1.0 } else { 0.0 };
                    assert!((c - want).norm() < 1e-12, "{waveform:?}: <{i}, {q}> = {c}");
                }
            }
            // a received block: the DFT receiver gives the energies of the correlators
            let r: Vec<C64> = (0..m).map(|n| C64::new((0.3 * n as f64).sin(), (1.7 * n as f64).cos())).collect();
            let e = css.energies(&r).unwrap();
            for (q, xq) in x.iter().enumerate() {
                let c: C64 = r.iter().zip(xq).map(|(a, b)| a * b.conj()).sum();
                assert!((c.norm_sqr() - e[q]).abs() < 1e-9, "{waveform:?}: bin {q}");
            }
        }
    }

    #[test]
    fn lora_is_a_continuous_phase_chirp_sampled_at_any_rate() {
        // Chiani and Elzanaty 2019, Property 1: the phase starts and ends at 0 (mod 2π)
        let css = Css { sf: 6, oversampling: 8, waveform: Waveform::LoRa };
        let m = css.m() as f64;
        for a in [0usize, 1, 17, 63] {
            let end = css.phase(a, m - 1e-9).rem_euclid(TAU);
            assert!(end.min(TAU - end) < 1e-6, "symbol {a}: {end}");
            // the instantaneous frequency rises by B/M per chip and folds once
            let x = css.symbol(a).unwrap();
            let df: Vec<f64> = x.windows(2).map(|w| (w[1] * w[0].conj()).arg() * 8.0 / TAU).collect();
            assert!((df[0] - (a as f64 / m - 0.5)).abs() < 0.02, "{a}: {}", df[0]);
        }
        // oversampled, the receiver still decides right without noise
        let s = css.modulate(&[3, 0, 63, 40]).unwrap();
        assert_eq!(css.demodulate(&s).unwrap(), vec![3, 0, 63, 40]);
        // at one sample per chip it is eq. (13)
        let one = Css::lora(6).unwrap();
        let x = one.symbol(9).unwrap();
        for (n, v) in x.iter().enumerate() {
            let n = n as f64;
            let want = cis(TAU * n * (9.0 / 64.0 - 0.5 + n / 128.0));
            assert!((v - want).norm() < 1e-9);
        }
        assert!(Css { sf: 6, oversampling: 2, waveform: Waveform::Fsk }.symbol(0).is_err());
        assert!(one.symbol(64).is_err() && Css::lora(0).is_err());
    }

    #[test]
    fn the_interferer_is_the_lora_signal_shifted_by_an_integer_offset() {
        // for an integer τ, eq. (21) is the stream of the two symbols delayed by τ chips
        let css = Css::lora(5).unwrap();
        let stream = css.modulate(&[7, 20]).unwrap();
        for tau in [0usize, 1, 13, 31] {
            let y = css.interferer(7, 20, tau as f64).unwrap();
            for (n, v) in y.iter().enumerate() {
                assert!((v - stream[n + 32 - tau]).norm() < 1e-9, "τ = {tau}, n = {n}");
            }
        }
        assert!(css.interferer(0, 0, 32.0).is_err());
    }
}

//! The numerology, the synchronisation signal block and the demodulation reference
//! signals of 5G NR (and the numerology of LTE, its case $\mu = 0$).
//!
//! # Numerology
//!
//! NR scales the 15 kHz subcarrier spacing of LTE: $\Delta f = 2^{\mu} \cdot 15$ kHz,
//! $\mu = 0, \dots, 4$; a 10 ms frame holds ten 1 ms subframes of $2^{\mu}$ slots of 14
//! OFDM symbols (12 with the extended cyclic prefix, $\mu = 2$ only); a resource block
//! is 12 subcarriers (Lin et al. 2019, Sec. II; Astély et al. 2009 for LTE). At the
//! sampling rate $N_{\mathrm{FFT}}\Delta f$, the cyclic prefix of a symbol is
//! $144 N_{\mathrm{FFT}}/2048$ samples, and $(144 + 16 \cdot 2^{\mu}) N_{\mathrm{FFT}}/2048$
//! for the first symbol of every half subframe ($l = 0$ and $l = 7 \cdot 2^{\mu}$), so
//! that a half subframe lasts exactly 0.5 ms; the extended prefix is $512
//! N_{\mathrm{FFT}}/2048$ (TS 38.211, Sec. 5.3.1): 4.7 µs and 16.7 µs at $\mu = 0$, the
//! prefixes of LTE. [`modulate`] maps a grid of $K = 12 N_{\mathrm{RB}}$ subcarriers, the
//! subcarrier $k$ at the frequency $(k - K/2)\Delta f$, and prepends the prefixes.
//!
//! # Synchronisation signal block
//!
//! The SSB occupies 4 OFDM symbols and 240 subcarriers (20 resource blocks): the
//! primary synchronisation signal (PSS) on symbol 0, subcarriers 56 to 182; the
//! secondary (SSS) on symbol 2, same subcarriers; the broadcast channel (PBCH) on
//! symbols 1 and 3 and on subcarriers 0–47 and 192–239 of symbol 2, with its
//! demodulation reference signal (DMRS) on one subcarrier in four, offset by
//! $\nu = N_{\mathrm{ID}}^{\mathrm{cell}} \bmod 4$. The 1008 cell identities are
//! $N_{\mathrm{ID}}^{\mathrm{cell}} = 3 N_{\mathrm{ID}}^{(1)} + N_{\mathrm{ID}}^{(2)}$,
//! $N_{\mathrm{ID}}^{(1)} < 336$, $N_{\mathrm{ID}}^{(2)} < 3$ (TS 38.211, Sec. 7.4.2–7.4.3).
//!
//! - PSS: the BPSK m-sequence $d(n) = 1 - 2x((n + 43 N_{\mathrm{ID}}^{(2)}) \bmod 127)$,
//!   $x(i+7) = (x(i+4) + x(i)) \bmod 2$, $[x(6) \dots x(0)] = [1\,1\,1\,0\,1\,1\,0]$;
//! - SSS: the BPSK Gold sequence $d(n) = [1 - 2x_{0}((n + m_{0}) \bmod 127)][1 -
//!   2x_{1}((n + m_{1}) \bmod 127)]$, $x_{0}(i+7) = (x_{0}(i+4) + x_{0}(i)) \bmod 2$,
//!   $x_{1}(i+7) = (x_{1}(i+1) + x_{1}(i)) \bmod 2$, both started at $[0 \dots 0\,1]$,
//!   $m_{0} = 15\lfloor N_{\mathrm{ID}}^{(1)}/112 \rfloor + 5 N_{\mathrm{ID}}^{(2)}$,
//!   $m_{1} = N_{\mathrm{ID}}^{(1)} \bmod 112$;
//! - PBCH DMRS: the QPSK $r(m) = [(1 - 2c(2m)) + j(1 - 2c(2m+1))]/\sqrt{2}$, $m < 144$,
//!   of the Gold sequence of 3GPP ([`com_math::sequences::gold_3gpp`]) with
//!   $c_{\mathrm{init}} = 2^{11}(\bar{\imath}_{\mathrm{SSB}} + 1)(\lfloor N_{\mathrm{ID}}^{\mathrm{cell}}/4
//!   \rfloor + 1) + 2^{6}(\bar{\imath}_{\mathrm{SSB}} + 1) + (N_{\mathrm{ID}}^{\mathrm{cell}} \bmod 4)$,
//!   mapped frequency first.
//!
//! [`detect_cell_id`] finds $N_{\mathrm{ID}}^{(2)}$ by the largest correlation of the
//! PSS subcarriers with the three PSS, then $N_{\mathrm{ID}}^{(1)}$ by the largest real
//! part of $\hat{h}^{*} \langle y_{\mathrm{SSS}}, d_{\mathrm{SSS}} \rangle$ over the 336
//! SSS, with the channel $\hat{h}$ estimated on the PSS (assumed flat over the 127
//! subcarriers and the three symbols).
//!
//! # DMRS of the PDSCH
//!
//! [`PdschDmrs`] gives the positions and values of the DMRS of a PDSCH of mapping type A
//! over a whole slot (14 symbols, normal prefix), single-symbol, port 1000: the first
//! DMRS symbol $l_{0} = 2$ or 3 (`dmrs-TypeA-Position`) and the additional positions
//! $\{l_{0}\}$, $\{l_{0}, 11\}$, $\{l_{0}, 7, 11\}$, $\{l_{0}, 5, 8, 11\}$; the subcarriers
//! $k = 4n + 2k'$ (configuration type 1, one in two) or $k = 6n + k'$ (type 2, two in
//! six), $k' \in \{0, 1\}$, with the QPSK value $r(2n + k')$ of the Gold sequence of
//! $c_{\mathrm{init}} = (2^{17}(14 n_{s} + l + 1)(2N_{\mathrm{ID}} + 1) + 2N_{\mathrm{ID}})
//! \bmod 2^{31}$ (TS 38.211, Sec. 7.4.1.1). It gives a [`PilotTable`] for the channel
//! estimators of [`super::estimation`]. Shorter allocations, two-symbol DMRS and the
//! other ports need the tables of TS 38.211 (status needed in `references/README.md`).
//!
//! # Examples
//! ```
//! use com_waveform::ofdm::nr::{Numerology, pss, detect_cell_id, ssb_grid};
//! let nr = Numerology::new(1, false)?;                       // 30 kHz
//! assert_eq!((nr.subcarrier_spacing(), nr.slots_per_frame()), (30e3, 20));
//! assert_eq!(nr.cp_lengths(4096)?[..2], [352, 288]);         // (144 + 16·2)·2, 144·2
//! assert_eq!(pss(0)?.len(), 127);
//! let grid = ssb_grid(417, 0, None)?;                        // 4 × 240, no PBCH data
//! assert_eq!(detect_cell_id(&grid)?.0, 417);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. X. Lin et al., "5G New Radio: Unveiling the essentials of the next generation
//!    wireless access technology," *IEEE Commun. Standards Mag.* 3(3), 30–37, 2019
//!    (arXiv:1806.06898).
//! 2. D. Astély et al., "LTE: the evolution of mobile broadband," *IEEE Commun. Mag.*
//!    47(4), 44–51, 2009.
//! 3. 3GPP TS 38.211, *NR; Physical channels and modulation*, Sec. 5.3, 7.4.

use super::grid::PilotTable;
use com_core::{Result, ensure};
use com_math::C64;
use com_math::fft::ifft;
use com_math::sequences::{gold_3gpp, lfsr};
use std::f64::consts::FRAC_1_SQRT_2;

/// Subcarriers of a resource block.
pub const SUBCARRIERS_PER_RB: usize = 12;
/// Subcarriers of the SSB.
pub const SSB_SUBCARRIERS: usize = 240;
/// OFDM symbols of the SSB.
pub const SSB_SYMBOLS: usize = 4;
/// Length of the PSS and of the SSS.
pub const SYNC_LENGTH: usize = 127;
/// First subcarrier of the PSS and of the SSS in the SSB.
pub const SYNC_FIRST_SUBCARRIER: usize = 56;

/// A numerology of NR: $\Delta f = 2^{\mu} \cdot 15$ kHz (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Numerology {
    /// $\mu$, 0 to 4.
    pub mu: u32,
    /// The extended cyclic prefix (12 symbols per slot; $\mu = 2$ only in NR, and the
    /// extended prefix of LTE at $\mu = 0$).
    pub extended_cp: bool,
}

impl Numerology {
    /// The numerology $\mu$.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for $\mu > 4$, or an extended prefix with $\mu \notin \{0, 2\}$.
    pub fn new(mu: u32, extended_cp: bool) -> Result<Numerology> {
        ensure(mu <= 4, || format!("NR numerologies are mu = 0 to 4, got {mu}"))?;
        ensure(!extended_cp || mu == 0 || mu == 2, || {
            format!("the extended cyclic prefix exists for mu = 2 (and LTE, mu = 0), got {mu}")
        })?;
        Ok(Numerology { mu, extended_cp })
    }

    /// The subcarrier spacing $2^{\mu} \cdot 15$ kHz (Hz).
    pub fn subcarrier_spacing(&self) -> f64 {
        15e3 * f64::from(1u32 << self.mu)
    }

    /// Slots per 1 ms subframe, $2^{\mu}$.
    pub fn slots_per_subframe(&self) -> usize {
        1 << self.mu
    }

    /// Slots per 10 ms frame.
    pub fn slots_per_frame(&self) -> usize {
        10 * self.slots_per_subframe()
    }

    /// OFDM symbols per slot: 14, or 12 with the extended prefix.
    pub fn symbols_per_slot(&self) -> usize {
        if self.extended_cp { 12 } else { 14 }
    }

    /// The sampling rate $N_{\mathrm{FFT}}\Delta f$ (Hz).
    pub fn sample_rate(&self, n_fft: usize) -> f64 {
        n_fft as f64 * self.subcarrier_spacing()
    }

    /// The cyclic prefixes (samples) of the symbols of one subframe at the FFT size
    /// `n_fft` (see the module).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] when `n_fft` is not a multiple of 128 (the prefixes would not be
    /// whole samples).
    pub fn cp_lengths(&self, n_fft: usize) -> Result<Vec<usize>> {
        ensure(n_fft >= 128 && n_fft % 128 == 0, || {
            format!("the FFT size must be a multiple of 128 for whole prefixes, got {n_fft}")
        })?;
        let n = self.symbols_per_slot() * self.slots_per_subframe();
        if self.extended_cp {
            return Ok(vec![512 * n_fft / 2048; n]);
        }
        let half = 7 * self.slots_per_subframe();
        let long = (144 + 16 * self.slots_per_subframe()) * n_fft / 2048;
        Ok((0..n).map(|l| if l % half == 0 { long } else { 144 * n_fft / 2048 }).collect())
    }
}

/// OFDM-modulate `symbols` OFDM symbols of a grid of `k` subcarriers (`grid[l * k + i]`
/// for subcarrier `i` of symbol `l`), starting at the symbol `first_symbol` of a
/// subframe, at the FFT size `n_fft` (see the module). The IFFT is scaled by
/// $1/\sqrt{K}$: a grid of unit-energy symbols gives a unit mean power.
///
/// # Errors
/// [`com_core::Error::Config`] for an odd or too large `k`, a grid that is not a whole number of
/// symbols, or an FFT size that [`Numerology::cp_lengths`] refuses.
pub fn modulate(nr: &Numerology, grid: &[C64], k: usize, n_fft: usize, first_symbol: usize) -> Result<Vec<C64>> {
    ensure(k > 0 && k % 2 == 0 && k <= n_fft, || format!("the grid needs an even K ≤ N_FFT subcarriers, got {k}"))?;
    ensure(grid.len() % k == 0, || format!("{} values is not a whole number of symbols of {k}", grid.len()))?;
    let cp = nr.cp_lengths(n_fft)?;
    let scale = (n_fft as f64) / (k as f64).sqrt();
    let mut out = Vec::new();
    for (s, sym) in grid.chunks_exact(k).enumerate() {
        let mut bins = vec![C64::new(0.0, 0.0); n_fft];
        for (i, v) in sym.iter().enumerate() {
            bins[(i + n_fft - k / 2) % n_fft] = *v;
        }
        let t: Vec<C64> = ifft(&bins).into_iter().map(|v| v * scale).collect();
        let ncp = cp[(first_symbol + s) % cp.len()];
        out.extend_from_slice(&t[n_fft - ncp..]);
        out.extend(t);
    }
    Ok(out)
}

/// $1 - 2x$ as $\pm 1$.
fn bpsk(b: u8) -> f64 {
    1.0 - 2.0 * f64::from(b)
}

/// The PSS of $N_{\mathrm{ID}}^{(2)}$ (127 values $\pm 1$).
///
/// # Errors
/// [`com_core::Error::Config`] for $N_{\mathrm{ID}}^{(2)} > 2$.
pub fn pss(n_id2: u32) -> Result<Vec<f64>> {
    ensure(n_id2 < 3, || format!("N_ID2 is 0, 1 or 2, got {n_id2}"))?;
    let x = lfsr(&[0, 4], 7, &[0, 1, 1, 0, 1, 1, 1], SYNC_LENGTH);
    Ok((0..SYNC_LENGTH).map(|n| bpsk(x[(n + 43 * n_id2 as usize) % SYNC_LENGTH])).collect())
}

/// The SSS of $(N_{\mathrm{ID}}^{(1)}, N_{\mathrm{ID}}^{(2)})$ (127 values $\pm 1$).
///
/// # Errors
/// [`com_core::Error::Config`] for $N_{\mathrm{ID}}^{(1)} > 335$ or $N_{\mathrm{ID}}^{(2)} > 2$.
pub fn sss(n_id1: u32, n_id2: u32) -> Result<Vec<f64>> {
    ensure(n_id1 < 336 && n_id2 < 3, || format!("N_ID1 < 336 and N_ID2 < 3, got ({n_id1}, {n_id2})"))?;
    let x0 = lfsr(&[0, 4], 7, &[1], SYNC_LENGTH);
    let x1 = lfsr(&[0, 1], 7, &[1], SYNC_LENGTH);
    let m0 = (15 * (n_id1 / 112) + 5 * n_id2) as usize;
    let m1 = (n_id1 % 112) as usize;
    Ok((0..SYNC_LENGTH).map(|n| bpsk(x0[(n + m0) % SYNC_LENGTH]) * bpsk(x1[(n + m1) % SYNC_LENGTH])).collect())
}

/// The QPSK of the pairs of a Gold sequence: $[(1 - 2c(2m)) + j(1 - 2c(2m+1))]/\sqrt{2}$.
fn qpsk_of_gold(c_init: u32, n: usize) -> Vec<C64> {
    let c = gold_3gpp(c_init, 2 * n);
    c.chunks_exact(2).map(|p| C64::new(bpsk(p[0]), bpsk(p[1])) * FRAC_1_SQRT_2).collect()
}

/// The 144 values of the PBCH DMRS of the cell `n_cell_id` and the SSB index
/// $\bar{\imath}_{\mathrm{SSB}}$ (0 to 7; see the module).
///
/// # Errors
/// [`com_core::Error::Config`] for a cell identity above 1007 or $\bar{\imath}_{\mathrm{SSB}} > 7$.
pub fn pbch_dmrs(n_cell_id: u32, i_ssb_bar: u32) -> Result<Vec<C64>> {
    ensure(n_cell_id < 1008 && i_ssb_bar < 8, || {
        format!("the cell identity is below 1008 and the SSB index below 8, got ({n_cell_id}, {i_ssb_bar})")
    })?;
    let c_init = (1 << 11) * (i_ssb_bar + 1) * (n_cell_id / 4 + 1) + (1 << 6) * (i_ssb_bar + 1) + n_cell_id % 4;
    Ok(qpsk_of_gold(c_init, 144))
}

/// What a resource element of the SSB carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsbElement {
    /// Nothing (the guard subcarriers around the PSS and the SSS).
    Empty,
    /// The PSS.
    Pss,
    /// The SSS.
    Sss,
    /// The PBCH data.
    Pbch,
    /// The PBCH DMRS.
    PbchDmrs,
}

/// What the subcarrier `k` (0 to 239) of the symbol `l` (0 to 3) of the SSB of the cell
/// `n_cell_id` carries (see the module).
pub fn ssb_element(n_cell_id: u32, k: usize, l: usize) -> SsbElement {
    let sync = (SYNC_FIRST_SUBCARRIER..SYNC_FIRST_SUBCARRIER + SYNC_LENGTH).contains(&k);
    let pbch = match l {
        1 | 3 => true,
        2 => !(48..192).contains(&k),
        _ => false,
    };
    match (l, sync, pbch) {
        (0, true, _) => SsbElement::Pss,
        (2, true, _) => SsbElement::Sss,
        (_, _, true) if k % 4 == (n_cell_id % 4) as usize => SsbElement::PbchDmrs,
        (_, _, true) => SsbElement::Pbch,
        _ => SsbElement::Empty,
    }
}

/// The resource elements `(k, l)` of one kind in the SSB, frequency first (the order
/// of the mapping).
pub fn ssb_indices(n_cell_id: u32, kind: SsbElement) -> Vec<(usize, usize)> {
    (0..SSB_SYMBOLS)
        .flat_map(|l| (0..SSB_SUBCARRIERS).map(move |k| (k, l)))
        .filter(|&(k, l)| ssb_element(n_cell_id, k, l) == kind)
        .collect()
}

/// The SSB of the cell `n_cell_id` with the SSB index $\bar{\imath}_{\mathrm{SSB}}$:
/// `grid[l * 240 + k]`, the PSS and the SSS of unit amplitude, the PBCH DMRS, and the
/// 432 PBCH symbols `pbch` (zeros when `None`).
///
/// # Errors
/// [`com_core::Error::Config`] for a cell identity above 1007, an SSB index above 7, or a PBCH
/// of another length than 432.
pub fn ssb_grid(n_cell_id: u32, i_ssb_bar: u32, pbch: Option<&[C64]>) -> Result<Vec<C64>> {
    let mut g = vec![C64::new(0.0, 0.0); SSB_SYMBOLS * SSB_SUBCARRIERS];
    let (n_id1, n_id2) = (n_cell_id / 3, n_cell_id % 3);
    let dmrs = pbch_dmrs(n_cell_id, i_ssb_bar)?;
    for (i, v) in pss(n_id2)?.into_iter().enumerate() {
        g[SYNC_FIRST_SUBCARRIER + i] = C64::new(v, 0.0);
    }
    for (i, v) in sss(n_id1, n_id2)?.into_iter().enumerate() {
        g[2 * SSB_SUBCARRIERS + SYNC_FIRST_SUBCARRIER + i] = C64::new(v, 0.0);
    }
    for ((k, l), v) in ssb_indices(n_cell_id, SsbElement::PbchDmrs).into_iter().zip(dmrs) {
        g[l * SSB_SUBCARRIERS + k] = v;
    }
    if let Some(d) = pbch {
        let idx = ssb_indices(n_cell_id, SsbElement::Pbch);
        ensure(d.len() == idx.len(), || format!("the PBCH carries {} symbols, got {}", idx.len(), d.len()))?;
        for ((k, l), v) in idx.into_iter().zip(d) {
            g[l * SSB_SUBCARRIERS + k] = *v;
        }
    }
    Ok(g)
}

/// The cell identity of a received SSB grid (`grid[l * 240 + k]`, 960 values), by the
/// PSS then the SSS, coherently with the channel of the PSS (see the module); returns
/// the identity and the normalised correlations $|\langle y, d \rangle| / (\|y\| \|d\|)$ of the PSS and of the SSS.
///
/// # Errors
/// [`com_core::Error::Config`] for a grid of another size.
pub fn detect_cell_id(grid: &[C64]) -> Result<(u32, f64, f64)> {
    ensure(grid.len() == SSB_SYMBOLS * SSB_SUBCARRIERS, || {
        format!("an SSB grid has {} values, got {}", SSB_SYMBOLS * SSB_SUBCARRIERS, grid.len())
    })?;
    let at = |l: usize| &grid[l * SSB_SUBCARRIERS + SYNC_FIRST_SUBCARRIER..][..SYNC_LENGTH];
    let (yp, ys) = (at(0), at(2));
    let energy = |y: &[C64]| y.iter().map(C64::norm_sqr).sum::<f64>().sqrt().max(1e-300);
    let norm = (SYNC_LENGTH as f64).sqrt();
    let corr = |y: &[C64], d: &[f64]| y.iter().zip(d).map(|(a, b)| a * *b).sum::<C64>();
    let mut best2 = (0, C64::new(0.0, 0.0));
    for n2 in 0..3 {
        let c = corr(yp, &pss(n2)?);
        if c.norm() > best2.1.norm() {
            best2 = (n2, c);
        }
    }
    // the flat channel estimated on the PSS, then a coherent correlation of the SSS
    let h = best2.1 / SYNC_LENGTH as f64;
    let mut best1 = (0, f64::NEG_INFINITY);
    for n1 in 0..336 {
        let c = (h.conj() * corr(ys, &sss(n1, best2.0)?)).re;
        if c > best1.1 {
            best1 = (n1, c);
        }
    }
    let rs = corr(ys, &sss(best1.0, best2.0)?).norm() / (energy(ys) * norm);
    Ok((3 * best1.0 + best2.0, best2.1.norm() / (energy(yp) * norm), rs))
}

/// The DMRS of a PDSCH of mapping type A over a slot of 14 symbols (see the module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PdschDmrs {
    /// Configuration type 1 (one subcarrier in two) or 2 (two in six).
    pub config_type: u8,
    /// `dmrs-TypeA-Position`: the first DMRS symbol, 2 or 3.
    pub type_a_position: usize,
    /// `dmrs-AdditionalPosition`, 0 to 3.
    pub additional_position: usize,
    /// The scrambling identity $N_{\mathrm{ID}}$ (the cell identity by default).
    pub n_id: u32,
    /// The slot $n_{s}$ in the frame.
    pub slot: u32,
}

impl PdschDmrs {
    /// The DMRS symbols of the slot.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a configuration type other than 1 or 2, a first symbol
    /// other than 2 or 3, an additional position above 3, or $l_{0} = 3$ with three
    /// additional positions (not allowed).
    pub fn symbols(&self) -> Result<Vec<usize>> {
        ensure(self.config_type == 1 || self.config_type == 2, || {
            format!("the DMRS configuration type is 1 or 2, got {}", self.config_type)
        })?;
        ensure(self.type_a_position == 2 || self.type_a_position == 3, || {
            format!("dmrs-TypeA-Position is 2 or 3, got {}", self.type_a_position)
        })?;
        ensure(self.additional_position <= 3, || {
            format!("dmrs-AdditionalPosition is 0 to 3, got {}", self.additional_position)
        })?;
        ensure(!(self.additional_position == 3 && self.type_a_position == 3), || {
            "dmrs-AdditionalPosition 3 needs dmrs-TypeA-Position 2".to_string()
        })?;
        let l0 = self.type_a_position;
        Ok(match self.additional_position {
            0 => vec![l0],
            1 => vec![l0, 11],
            2 => vec![l0, 7, 11],
            _ => vec![l0, 5, 8, 11],
        })
    }

    /// The DMRS subcarriers of a grid of `n_rb` resource blocks.
    pub fn subcarriers(&self, n_rb: usize) -> Vec<usize> {
        let k = SUBCARRIERS_PER_RB * n_rb;
        if self.config_type == 1 { (0..k).step_by(2).collect() } else { (0..k).filter(|i| i % 6 < 2).collect() }
    }

    /// The DMRS values of the symbol `l` on [`PdschDmrs::subcarriers`] (the grid starts
    /// at common resource block 0).
    pub fn values(&self, n_rb: usize, l: usize) -> Vec<C64> {
        let n = self.subcarriers(n_rb).len();
        let a = (1u64 << 17) * (14 * u64::from(self.slot) + l as u64 + 1) * (2 * u64::from(self.n_id) + 1);
        let c_init = ((a + 2 * u64::from(self.n_id)) % (1u64 << 31)) as u32;
        qpsk_of_gold(c_init, n)
    }

    /// The DMRS of the slot as a [`PilotTable`] of `12 n_rb` subcarriers and 14 symbols.
    ///
    /// # Errors
    /// As [`PdschDmrs::symbols`].
    pub fn pilot_table(&self, n_rb: usize) -> Result<PilotTable> {
        let ks = self.subcarriers(n_rb);
        let mut pilots = Vec::new();
        for l in self.symbols()? {
            for (k, v) in ks.iter().zip(self.values(n_rb, l)) {
                pilots.push(((*k, l), v));
            }
        }
        PilotTable::new(SUBCARRIERS_PER_RB * n_rb, 14, pilots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_numerologies_of_lin_et_al_and_of_lte() {
        // Lin et al. 2019, Sec. II: 2^mu 15 kHz, 2^mu slots of 14 symbols per subframe
        for mu in 0..=4 {
            let nr = Numerology::new(mu, false).unwrap();
            assert_eq!(nr.subcarrier_spacing(), 15e3 * f64::from(1u32 << mu));
            assert_eq!((nr.slots_per_subframe(), nr.symbols_per_slot()), (1 << mu, 14));
            // a subframe lasts 1 ms at every FFT size
            for n_fft in [256, 2048, 4096] {
                let total: usize = nr.cp_lengths(n_fft).unwrap().iter().map(|c| c + n_fft).sum();
                assert!((total as f64 / nr.sample_rate(n_fft) - 1e-3).abs() < 1e-15);
            }
        }
        // LTE (Astély et al. 2009): 15 kHz, 4.7 µs and 16.7 µs prefixes, 14 or 12 symbols
        let lte = Numerology::new(0, false).unwrap();
        let fs = lte.sample_rate(2048);
        assert_eq!(fs, 30.72e6);
        let cp = lte.cp_lengths(2048).unwrap();
        assert_eq!((cp[0], cp[1], cp[7]), (160, 144, 160));
        assert!((cp[1] as f64 / fs - 4.7e-6).abs() < 0.05e-6);
        let ext = Numerology::new(0, true).unwrap();
        assert_eq!(ext.symbols_per_slot(), 12);
        assert!((ext.cp_lengths(2048).unwrap()[0] as f64 / fs - 16.7e-6).abs() < 0.05e-6);
        assert!(Numerology::new(5, false).is_err() && Numerology::new(1, true).is_err());
        assert!(lte.cp_lengths(100).is_err());
    }

    #[test]
    fn the_ssb_of_lin_et_al() {
        // 4 symbols, 240 subcarriers; PSS and SSS of length 127, BPSK; 1008 identities
        let id = 421;
        let count = |k| ssb_indices(id, k).len();
        assert_eq!((count(SsbElement::Pss), count(SsbElement::Sss)), (127, 127));
        assert_eq!((count(SsbElement::PbchDmrs), count(SsbElement::Pbch)), (144, 432));
        // the m-sequence of the PSS: 64 ones and 63 minus ones; the three PSS nearly
        // orthogonal
        let p: Vec<Vec<f64>> = (0..3).map(|n| pss(n).unwrap()).collect();
        assert_eq!(p[0].iter().filter(|v| **v < 0.0).count(), 64);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            let c: f64 = p[a].iter().zip(&p[b]).map(|(x, y)| x * y).sum();
            assert_eq!(c, -1.0);
        }
        // 1008 distinct SSS-PSS pairs
        let mut seen = std::collections::HashSet::new();
        for n1 in 0..336 {
            for n2 in 0..3 {
                let s: Vec<i8> = sss(n1, n2).unwrap().iter().map(|v| *v as i8).collect();
                assert!(seen.insert((s, n2)));
            }
        }
        assert!(pss(3).is_err() && sss(336, 0).is_err() && pbch_dmrs(1008, 0).is_err());
    }

    #[test]
    fn a_slot_modulates_and_the_dmrs_table_fits_it() {
        let nr = Numerology::new(1, false).unwrap();
        let (n_rb, n_fft) = (24, 512);
        let k = SUBCARRIERS_PER_RB * n_rb;
        let grid: Vec<C64> = (0..14 * k).map(|i| C64::new(((i * 7) % 3) as f64 - 1.0, 1.0) * FRAC_1_SQRT_2).collect();
        let x = modulate(&nr, &grid, k, n_fft, 0).unwrap();
        let cp = nr.cp_lengths(n_fft).unwrap();
        assert_eq!(x.len(), 14 * n_fft + cp[..14].iter().sum::<usize>());
        // the prefix repeats the end of its symbol
        assert!((0..cp[0]).all(|i| (x[i] - x[i + n_fft]).norm() < 1e-12));
        let dmrs = PdschDmrs { config_type: 1, type_a_position: 2, additional_position: 2, n_id: 1, slot: 3 };
        assert_eq!(dmrs.symbols().unwrap(), vec![2, 7, 11]);
        let t = dmrs.pilot_table(n_rb).unwrap();
        assert_eq!(t.len(), 3 * k / 2);
        let t2 = PdschDmrs { config_type: 2, ..dmrs }.pilot_table(n_rb).unwrap();
        assert_eq!(t2.len(), 3 * k / 3);
        assert!(PdschDmrs { type_a_position: 3, additional_position: 3, ..dmrs }.symbols().is_err());
    }
}

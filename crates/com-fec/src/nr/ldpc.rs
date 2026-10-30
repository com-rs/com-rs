//! The LDPC code of the data channels of 5G NR (TS 38.212, Sec. 5.2.2, 5.3.2, 5.4.2).
//!
//! # Model
//!
//! A code block of $K'$ information bits is encoded by base graph 1 ($46 \times 68$,
//! $K_{b} = 22$) or 2 ($42 \times 52$, $K_{b} \le 10$), chosen as the shared channels do
//! (Sec. 7.2.2): base graph 2 when $K' \le 292$, or $K' \le 3824$ and $R \le 0.67$, or
//! $R \le 0.25$, with $R = K'/E$; base graph 1 otherwise. The lifting size $Z_{c}$ is the
//! smallest of Table 5.3.2-1 with $K_{b} Z_{c} \ge K'$ ($K_{b} = 22$ for base graph 1;
//! for base graph 2, 10, 9, 8 or 6 for $K'$ above 640, 560, 192 or below), and every
//! nonzero entry $V_{i,j}$ of the base graph becomes the $Z_{c} \times Z_{c}$ identity
//! cyclically shifted by $V_{i,j} \bmod Z_{c}$ (the column of the set $i_{\mathrm{LS}}$
//! that contains $Z_{c}$). The $K = 22 Z_{c}$ (or $10 Z_{c}$) systematic bits are the
//! information bits followed by $K - K'$ filler bits (zeros); the encoder of
//! [`crate::ldpc::LdpcCode::from_core_rows`] computes the $N = 66 Z_{c}$ (or
//! $50 Z_{c}$) other bits of $d$ — the first $2 Z_{c}$ systematic bits are never
//! transmitted.
//!
//! Rate matching (Sec. 5.4.2.1, without limited buffer, $N_{\mathrm{cb}} = N$): the $E$
//! output bits are read from the circular buffer $d$ from $k_{0}$, skipping the filler
//! bits, where $k_{0} = 0$, $\lfloor 17 N_{\mathrm{cb}} / (66 Z_{c}) \rfloor Z_{c}$,
//! $\lfloor 33 N_{\mathrm{cb}} / (66 Z_{c}) \rfloor Z_{c}$,
//! $\lfloor 56 N_{\mathrm{cb}} / (66 Z_{c}) \rfloor Z_{c}$ for the redundancy versions 0
//! to 3 (13, 25, 43 over $50 Z_{c}$ for base graph 2); then the bit interleaver of
//! Sec. 5.4.2.2 writes them row by row in $Q_{m}$ rows and reads them column by column
//! ($Q_{m}$ the bits per modulation symbol, 1 for none).
//!
//! The decoder places the received LLRs on the bits they carry (summed where the
//! circular buffer wraps), zero on the punctured and unsent bits and a large positive
//! value on the filler bits, and runs the belief propagation of [`crate::ldpc`].
//!
//! # Examples
//! ```
//! use com_fec::block::BlockCode;
//! use com_fec::nr::NrLdpc;
//! let code = NrLdpc::new(100, 300)?;                       // K' = 100, E = 300
//! assert_eq!((code.base_graph(), code.lifting_size()), (2, 18));
//! let info: Vec<u8> = (0..100).map(|i| (i % 3 == 0) as u8).collect();
//! let c = code.encode_block(&info)?;
//! let llr: Vec<f64> = c.iter().map(|b| if *b == 0 { 4.0 } else { -4.0 }).collect();
//! assert_eq!(code.decode_block(&llr)?, info);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. 3GPP TS 38.212 V18.5.0, *NR; Multiplexing and channel coding*, Sec. 5.2.2, 5.3.2,
//!    5.4.2, 7.2.2 (ETSI TS 138 212, 2025).
//! 2. T. Richardson, S. Kudekar, "Design of low-density parity check codes for 5G new
//!    radio," *IEEE Commun. Mag.* 56(3), 28–34, 2018.

use super::tables::{BG1, BG2, LIFTING_SETS};
use crate::block::{BlockCode, SoftDecode, container_traits, expect_len};
use crate::interleaving::Interleaver;
use crate::ldpc::{LdpcCode, LdpcDecoder};
use crate::puncture::BitSelection;
use com_core::signal::KNOWN_BIT_LLR;
use com_core::{Result, ensure};

/// A base graph: its nonzero entries $(i, j, V_{i,j})$.
type BaseGraph = [(u8, u8, [u16; 8])];

/// The LDPC code of 5G NR for one code block (see the module).
#[derive(Clone, Debug)]
pub struct NrLdpc {
    bg: u8,
    z: usize,
    k_info: usize,
    e: usize,
    /// The bit of the full codeword (all $K + N$ columns) carried by every output bit,
    /// after the bit interleaver.
    map: BitSelection,
    /// The code of the lifted parity-check matrix and its decoder settings.
    code: LdpcCode,
}

/// The base graph of 5G NR for $K'$ bits at the rate $K'/E$ (Sec. 7.2.2).
fn base_graph(k_info: usize, e: usize) -> u8 {
    let r = k_info as f64 / e as f64;
    if k_info <= 292 || (k_info <= 3824 && r <= 0.67) || r <= 0.25 { 2 } else { 1 }
}

impl NrLdpc {
    /// The code of `k_info` information bits rate-matched to `e` bits, redundancy
    /// version 0, without bit interleaving.
    ///
    /// # Errors
    /// As [`NrLdpc::with_options`].
    pub fn new(k_info: usize, e: usize) -> Result<NrLdpc> {
        NrLdpc::with_options(k_info, e, None, 0, 1)
    }

    /// The code of `k_info` information bits rate-matched to `e` bits, the base graph
    /// chosen as in Sec. 7.2.2 or forced by `base_graph`, the redundancy version `rv`
    /// and the bit interleaver of `qm` bits per symbol (1: none).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a block larger than one code block (8448 bits for
    /// base graph 1, 3840 for base graph 2), `e` not a multiple of `qm`, an `rv` above
    /// 3, or a base graph other than 1 or 2.
    pub fn with_options(k_info: usize, e: usize, base_graph_override: Option<u8>, rv: u8, qm: usize) -> Result<NrLdpc> {
        ensure(k_info >= 1 && e >= 1, || format!("need K' ≥ 1 and E ≥ 1, got {k_info} and {e}"))?;
        ensure(rv <= 3 && qm >= 1 && e % qm == 0, || format!("rv ≤ 3 and Qm | E, got rv {rv}, Qm {qm}, E {e}"))?;
        let bg = base_graph_override.unwrap_or_else(|| base_graph(k_info, e));
        ensure(bg == 1 || bg == 2, || format!("the base graph is 1 or 2, got {bg}"))?;
        let k_cb = if bg == 1 { 8448 } else { 3840 };
        ensure(k_info <= k_cb, || {
            format!("{k_info} bits exceed one code block of base graph {bg} ({k_cb}); segment the block first")
        })?;
        let kb = match (bg, k_info) {
            (1, _) => 22,
            (_, b) if b > 640 => 10,
            (_, b) if b > 560 => 9,
            (_, b) if b > 192 => 8,
            _ => 6,
        };
        let (z, ils) = LIFTING_SETS
            .iter()
            .enumerate()
            .flat_map(|(i, set)| set.iter().map(move |&z| (z, i)))
            .filter(|&(z, _)| kb * z >= k_info)
            .min()
            .ok_or_else(|| com_core::Error::Config(format!("no lifting size for K' = {k_info}")))?;
        let (table, cols_k, rows_n): (&BaseGraph, usize, usize) = if bg == 1 { (&BG1, 22, 46) } else { (&BG2, 10, 42) };
        let k = cols_k * z;
        let n_full = (cols_k + rows_n) * z;
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); rows_n * z];
        for &(i, j, v) in table {
            let shift = v[ils] as usize % z;
            for r in 0..z {
                rows[i as usize * z + r].push(j as usize * z + (r + shift) % z);
            }
        }
        let code = LdpcCode::from_core_rows(n_full, rows, k, 4 * z)?;
        // the circular buffer d: the full codeword without its first 2Z bits
        let ncb = n_full - 2 * z;
        let num = match (bg, rv) {
            (_, 0) => 0,
            (1, 1) => 17,
            (1, 2) => 33,
            (1, _) => 56,
            (_, 1) => 13,
            (_, 2) => 25,
            _ => 43,
        };
        let den = if bg == 1 { 66 } else { 50 };
        let k0 = num * ncb / (den * z) * z;
        let filler = |col: usize| col >= k_info && col < k;
        let mut sel = Vec::with_capacity(e);
        let mut j = 0;
        while sel.len() < e {
            let col = 2 * z + (k0 + j) % ncb;
            if !filler(col) {
                sel.push(col);
            }
            j += 1;
        }
        // the bit interleaver: f[i + j Qm] = e[i E/Qm + j], written row by row into Qm
        // rows and read column by column
        let map = BitSelection::new(Interleaver::block(qm, e / qm).interleave(&sel), code.n())?;
        Ok(NrLdpc { bg, z, k_info, e, map, code })
    }

    /// The base graph, 1 or 2.
    pub fn base_graph(&self) -> u8 {
        self.bg
    }

    /// The lifting size $Z_{c}$.
    pub fn lifting_size(&self) -> usize {
        self.z
    }

    /// The bit of the full codeword carried by every output bit.
    pub fn bit_map(&self) -> &[usize] {
        self.map.positions()
    }

    /// The code of the lifted parity-check matrix.
    pub fn code(&self) -> &LdpcCode {
        &self.code
    }

    /// The settings of the decoder, which can change without changing the code.
    pub fn decoder_mut(&mut self) -> &mut LdpcDecoder {
        &mut self.code.decoder
    }

    /// The LLRs of the full codeword: the received LLRs summed over the copies of every
    /// bit, zero on the unsent bits, [`KNOWN_BIT_LLR`] on the filler bits.
    fn full_llr(&self, llr: &[f64]) -> Result<Vec<f64>> {
        let mut full = self.map.accumulate(llr)?;
        for v in full.iter_mut().take(self.code.k()).skip(self.k_info) {
            *v = KNOWN_BIT_LLR;
        }
        Ok(full)
    }
}

impl BlockCode for NrLdpc {
    fn k(&self) -> usize {
        self.k_info
    }

    fn n(&self) -> usize {
        self.e
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k_info, "bits")?;
        let mut u = info.to_vec();
        u.resize(self.code.k(), 0);
        self.map.select(&self.code.encode_block(&u)?)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), self.e, "LLRs")?;
        let (post, _) = self.code.decode_llr(&self.full_llr(llr)?)?;
        Ok(com_core::signal::hard_decisions(&post[..self.k_info]))
    }
}

impl SoftDecode for NrLdpc {
    /// The a posteriori LLRs of the $K'$ information bits after the belief propagation
    /// of the full codeword (the a priori LLRs added to theirs).
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> {
        expect_len(llr.len(), self.e, "LLRs")?;
        expect_len(out.len(), self.k_info, "a posteriori LLRs")?;
        let mut full = self.full_llr(llr)?;
        if let Some(a) = apriori {
            expect_len(a.len(), self.k_info, "a priori LLRs")?;
            for (v, a) in full.iter_mut().zip(a) {
                *v += a;
            }
        }
        let (post, it) = self.code.decode_llr(&full)?;
        out.copy_from_slice(&post[..self.k_info]);
        Ok(it)
    }
}

container_traits!(NrLdpc);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_base_graphs_have_their_dimensions() {
        assert_eq!((BG1.len(), BG2.len()), (316, 197));
        assert_eq!(BG1.iter().map(|e| e.0).max(), Some(45));
        assert_eq!(BG1.iter().map(|e| e.1).max(), Some(67));
        assert_eq!(BG2.iter().map(|e| e.0).max(), Some(41));
        assert_eq!(BG2.iter().map(|e| e.1).max(), Some(51));
        // the extension: row i ≥ 4 closes on column K_b + i
        for (t, kb) in [(&BG1[..], 22u8), (&BG2[..], 10)] {
            for i in 4..=t.iter().map(|e| e.0).max().unwrap() {
                let last = t.iter().filter(|e| e.0 == i).map(|e| e.1).max().unwrap();
                assert_eq!(last, kb + i);
            }
        }
    }

    #[test]
    fn every_lifting_size_encodes_codewords() {
        for set in LIFTING_SETS {
            for &z in set.iter().filter(|&&z| z <= 64) {
                for bg in [1u8, 2] {
                    let kb = if bg == 1 { 22 } else { 10 };
                    let k_info = kb * z - 3;
                    let code = NrLdpc::with_options(k_info, 3 * k_info, Some(bg), 0, 1).unwrap();
                    assert_eq!(code.lifting_size(), z.max(code.lifting_size()));
                    let info: Vec<u8> = (0..k_info).map(|i| ((i * 7 + z) % 5 < 2) as u8).collect();
                    let mut u = info.clone();
                    u.resize(code.code().k(), 0);
                    assert!(code.code().is_codeword(&code.code().encode_block(&u).unwrap()));
                    assert_eq!(code.encode_block(&info).unwrap().len(), 3 * k_info);
                }
            }
        }
        assert!(NrLdpc::new(9000, 10000).is_err());
    }

    /// The rate matching as it was written before the bit selection: the bit
    /// interleaver by its index formula, the LLRs summed position by position.
    #[test]
    fn the_selection_gives_the_former_rate_matching() {
        let mut rng = com_core::rng::Rng::new(21);
        for (k_info, e, rv, qm) in [(100, 300, 0, 1), (500, 1200, 2, 4), (1000, 1500, 3, 6), (60, 1000, 1, 2)] {
            let code = NrLdpc::with_options(k_info, e, None, rv, qm).unwrap();
            let k = code.code.k();
            let z = code.z;
            let ncb = code.code.n() - 2 * z;
            let (num, den) = match (code.bg, rv) {
                (_, 0) => (0, 1),
                (1, 1) => (17, 66),
                (1, 2) => (33, 66),
                (1, _) => (56, 66),
                (_, 1) => (13, 50),
                (_, 2) => (25, 50),
                _ => (43, 50),
            };
            let k0 = num * ncb / (den * z) * z;
            let mut sel = Vec::new();
            let mut j = 0;
            while sel.len() < e {
                let col = 2 * z + (k0 + j) % ncb;
                if !(col >= k_info && col < k) {
                    sel.push(col);
                }
                j += 1;
            }
            let map: Vec<usize> = (0..e).map(|t| sel[(t % qm) * (e / qm) + t / qm]).collect();
            assert_eq!(code.bit_map(), map);
            let llr: Vec<f64> = (0..e).map(|_| rng.gaussian()).collect();
            let mut full = vec![0.0; code.code.n()];
            for (&i, &l) in map.iter().zip(&llr) {
                full[i] += l;
            }
            for v in full.iter_mut().take(k).skip(k_info) {
                *v = 30.0;
            }
            let new = code.full_llr(&llr).unwrap();
            assert!(new.iter().zip(&full).all(|(a, b)| a.to_bits() == b.to_bits()));
            let (post, _) = code.code.decode_llr(&full).unwrap();
            let app = code.app(&llr, None).unwrap();
            assert!(app.iter().zip(&post[..k_info]).all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_eq!(code.decode_block(&llr).unwrap(), com_core::signal::hard_decisions(&post[..k_info]));
        }
    }
}

//! Convolutional codes: trellis, Viterbi and BCJR decoders.
//!
//! A [`Trellis`] describes a rate-$1/n$ code with $2^{K-1}$ states: from every state, an
//! input bit leads to a next state and emits $n$ coded bits. Two families are built:
//!
//! - feed-forward codes ([`Trellis::feedforward`]) from octal generators, the most
//!   significant bit tapping the current input — the $(133, 171)_{8}$ code of constraint
//!   length 7 ([`ConvolutionalCode::k7`]) is the classical one (802.11, DVB-S, LTE);
//! - recursive systematic codes ([`Trellis::recursive`]), the constituents of turbo
//!   codes, with a feedback and a feed-forward polynomial (LTE: $(13, 15)_{8}$).
//!
//! # Decoders
//!
//! - [`viterbi`]: maximum-likelihood sequence detection; with LLR inputs, the branch
//!   metric is the correlation $\sum_{i} (1 - 2 c_{i}) L_{i} / 2$ (soft decisions), with
//!   hard decisions it is the Hamming distance — about 2 dB apart (Viterbi 1967;
//!   Forney 1973);
//! - [`bcjr`]: the a posteriori LLR of every input bit (Bahl, Cocke, Jelinek & Raviv
//!   1974), in the log domain with the exact Jacobian logarithm (log-MAP) or its
//!   maximum (max-log-MAP).
//!
//! # Examples
//! ```
//! use com_fec::block::{BlockCode, bpsk_llr};
//! use com_fec::conv::ConvolutionalCode;
//! let code = ConvolutionalCode::k7(100);
//! let info: Vec<u8> = (0..100).map(|i| ((i * 7) % 3 == 0) as u8).collect();
//! let c = code.encode_block(&info)?;
//! assert_eq!(c.len(), 2 * (100 + 6));                           // terminated
//! let y: Vec<f64> = c.iter().map(|b| 1.0 - 2.0 * f64::from(*b)).collect();
//! assert_eq!(code.decode_block(&bpsk_llr(&y, 0.5))?, info);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum
//!    decoding algorithm," *IEEE Trans. Inf. Theory*, 13(2), 260–269, 1967.
//! 2. G. D. Forney, "The Viterbi algorithm," *Proc. IEEE*, 61(3), 268–278, 1973.
//! 3. L. R. Bahl, J. Cocke, F. Jelinek and J. Raviv, "Optimal decoding of linear codes
//!    for minimizing symbol error rate," *IEEE Trans. Inf. Theory*, 20(2), 284–287, 1974.
//! 4. P. Robertson, E. Villebrun and P. Hoeher, "A comparison of optimal and sub-optimal
//!    MAP decoding algorithms operating in the log domain," *Proc. IEEE ICC*, 1995.

use crate::block::{BlockCode, container_traits, expect_len, max_star};
use com_core::{Result, ensure};

fn parity(x: u32) -> u8 {
    (x.count_ones() & 1) as u8
}

/// The trellis of a rate-$1/n$ binary convolutional code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trellis {
    /// Constraint length $K$ (memory $K - 1$).
    pub constraint: u32,
    /// Coded bits per input bit $n$.
    pub n_out: usize,
    /// `next[s][u]`: the state after input `u` in state `s`.
    pub next: Vec<[usize; 2]>,
    /// `out[s][u]`: the coded bits, first output in the most significant position.
    pub out: Vec<[u32; 2]>,
    /// `tail[s]`: the input that drives state `s` towards zero (0 for feed-forward codes).
    pub tail: Vec<u8>,
}

impl Trellis {
    /// Number of states $2^{K-1}$.
    pub fn n_states(&self) -> usize {
        self.next.len()
    }

    /// The feed-forward code of constraint length `constraint` with the generators
    /// `generators` (octal digits written as integers, e.g. `0o133`).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a constraint length outside 1 to 24 or no generator.
    pub fn feedforward(constraint: u32, generators: &[u32]) -> Result<Trellis> {
        let m = Self::memory(constraint)?;
        ensure(!generators.is_empty(), || "a convolutional code needs at least one generator".into())?;
        let states = 1usize << m;
        let mut next = Vec::with_capacity(states);
        let mut out = Vec::with_capacity(states);
        for s in 0..states as u32 {
            let mut nx = [0usize; 2];
            let mut o = [0u32; 2];
            for u in 0..2u32 {
                let reg = (u << m) | s;
                nx[u as usize] = (reg >> 1) as usize;
                o[u as usize] = generators.iter().fold(0, |acc, g| (acc << 1) | u32::from(parity(reg & g)));
            }
            next.push(nx);
            out.push(o);
        }
        Ok(Trellis { constraint, n_out: generators.len(), next, out, tail: vec![0; states] })
    }

    /// The memory `constraint - 1` of a code, checked.
    fn memory(constraint: u32) -> Result<u32> {
        ensure((1..=24).contains(&constraint), || format!("the constraint length must be 1 to 24, got {constraint}"))?;
        Ok(constraint - 1)
    }

    /// The recursive systematic code with the feedback polynomial `feedback` and the
    /// feed-forward polynomial `forward` (octal, e.g. `0o13`, `0o15`): outputs the input
    /// bit, then the parity bit.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for a constraint length outside 1 to 24.
    pub fn recursive(constraint: u32, feedback: u32, forward: u32) -> Result<Trellis> {
        let m = Self::memory(constraint)?;
        let states = 1usize << m;
        let fb = feedback & ((1 << m) - 1);
        let mut next = Vec::with_capacity(states);
        let mut out = Vec::with_capacity(states);
        let mut tail = Vec::with_capacity(states);
        for s in 0..states as u32 {
            let mut nx = [0usize; 2];
            let mut o = [0u32; 2];
            for u in 0..2u32 {
                let w = u ^ u32::from(parity(s & fb));
                let reg = (w << m) | s;
                nx[u as usize] = (reg >> 1) as usize;
                o[u as usize] = (u << 1) | u32::from(parity(reg & forward));
            }
            next.push(nx);
            out.push(o);
            tail.push(parity(s & fb));
        }
        Ok(Trellis { constraint, n_out: 2, next, out, tail })
    }

    /// Encode `bits` from state 0; with `terminate`, append the $K - 1$ tail inputs that
    /// return to state 0. Returns the coded bits.
    pub fn encode(&self, bits: &[u8], terminate: bool) -> Vec<u8> {
        let mut s = 0usize;
        let tail = if terminate { self.constraint as usize - 1 } else { 0 };
        let mut out = Vec::with_capacity((bits.len() + tail) * self.n_out);
        let mut step = |s: &mut usize, u: u8| {
            let o = self.out[*s][usize::from(u)];
            for i in (0..self.n_out).rev() {
                out.push(((o >> i) & 1) as u8);
            }
            *s = self.next[*s][usize::from(u)];
        };
        for &b in bits {
            step(&mut s, b & 1);
        }
        for _ in 0..tail {
            let u = self.tail[s];
            step(&mut s, u);
        }
        out
    }

    /// The log-likelihood of a branch, $\sum_{i} \pm L_{i}/2$.
    fn branch(&self, o: u32, llr: &[f64]) -> f64 {
        llr.iter().enumerate().map(|(i, l)| if (o >> (self.n_out - 1 - i)) & 1 == 0 { l / 2.0 } else { -l / 2.0 }).sum()
    }

    /// The log-likelihoods of the branches of one step into `metrics[s][u]` (flat,
    /// `2 s + u`): through the $2^{n}$ output words when they are fewer than the
    /// branches, each computed once.
    fn branches(&self, llr: &[f64], words: &mut Vec<f64>, metrics: &mut [f64]) {
        let n_words = if self.n_out < 16 { 1usize << self.n_out } else { usize::MAX };
        if n_words <= metrics.len() {
            words.clear();
            words.extend((0..n_words as u32).map(|o| self.branch(o, llr)));
            for (m, o) in metrics.iter_mut().zip(self.out.iter().flatten()) {
                *m = words[*o as usize];
            }
        } else {
            for (m, o) in metrics.iter_mut().zip(self.out.iter().flatten()) {
                *m = self.branch(*o, llr);
            }
        }
    }
}

/// Viterbi decoding of `steps = llr.len() / n` trellis steps from state 0; with
/// `terminated`, the path ends in state 0 and the last $K - 1$ inputs (the tail) are
/// dropped. Returns the decided input bits.
///
/// # Errors
/// [`com_core::Error::Shape`] if the number of LLRs is not a multiple of $n$.
pub fn viterbi(trellis: &Trellis, llr: &[f64], terminated: bool) -> Result<Vec<u8>> {
    let n = trellis.n_out;
    ensure(llr.len() % n == 0, || format!("{} LLRs is not a multiple of {n}", llr.len()))
        .map_err(|e| com_core::Error::Shape(e.to_string()))?;
    let steps = llr.len() / n;
    let ns = trellis.n_states();
    let mut metric = vec![f64::NEG_INFINITY; ns];
    metric[0] = 0.0;
    let mut next = vec![f64::NEG_INFINITY; ns];
    let (mut words, mut gamma) = (Vec::new(), vec![0.0; 2 * ns]);
    // the survivor of every step and state: its predecessor and input, 2 s + u (the
    // states number at most 2^23)
    let mut from = vec![0u32; steps * ns];
    for (t, row) in from.chunks_exact_mut(ns).enumerate() {
        trellis.branches(&llr[t * n..(t + 1) * n], &mut words, &mut gamma);
        next.fill(f64::NEG_INFINITY);
        for s in 0..ns {
            if metric[s] == f64::NEG_INFINITY {
                continue;
            }
            for u in 0..2 {
                let ns2 = trellis.next[s][u];
                let m = metric[s] + gamma[2 * s + u];
                if m > next[ns2] {
                    next[ns2] = m;
                    row[ns2] = (2 * s + u) as u32;
                }
            }
        }
        std::mem::swap(&mut metric, &mut next);
    }
    let mut s = if terminated { 0 } else { (0..ns).max_by(|a, b| metric[*a].total_cmp(&metric[*b])).unwrap_or(0) };
    let mut bits = vec![0u8; steps];
    for t in (0..steps).rev() {
        let su = from[t * ns + s] as usize;
        bits[t] = (su & 1) as u8;
        s = su >> 1;
    }
    if terminated {
        bits.truncate(steps.saturating_sub(trellis.constraint as usize - 1));
    }
    Ok(bits)
}

/// The BCJR algorithm in the log domain: the a posteriori LLR of every input bit,
/// given the channel LLRs of the coded bits (`n` per step) and the a priori LLRs of the
/// inputs (`apriori`, zero if empty). `exact` selects log-MAP (else max-log-MAP);
/// `terminated` forces the final state to zero.
///
/// # Errors
/// [`com_core::Error::Shape`] for inconsistent lengths.
pub fn bcjr(trellis: &Trellis, llr: &[f64], apriori: &[f64], terminated: bool, exact: bool) -> Result<Vec<f64>> {
    let mut app = Vec::new();
    bcjr_into(trellis, llr, apriori, terminated, exact, &mut BcjrWorkspace::default(), &mut app)?;
    Ok(app)
}

/// The arrays of [`bcjr`], kept between the calls of an iterative decoder.
#[derive(Default)]
pub(crate) struct BcjrWorkspace {
    /// $\gamma_{t}(s, u)$ at `(t ns + s) 2 + u`.
    gamma: Vec<f64>,
    /// $\alpha_{t}(s)$, then $\beta_{t}(s)$, at `t ns + s`, $t = 0, \dots, T$.
    alpha: Vec<f64>,
    beta: Vec<f64>,
    words: Vec<f64>,
}

/// [`bcjr`] into `app`, with the arrays of `ws`.
pub(crate) fn bcjr_into(
    trellis: &Trellis,
    llr: &[f64],
    apriori: &[f64],
    terminated: bool,
    exact: bool,
    ws: &mut BcjrWorkspace,
    app: &mut Vec<f64>,
) -> Result<()> {
    let n = trellis.n_out;
    if llr.len() % n != 0 || (!apriori.is_empty() && apriori.len() != llr.len() / n) {
        return Err(com_core::Error::Shape("inconsistent channel and a priori LLRs".into()));
    }
    let steps = llr.len() / n;
    let ns = trellis.n_states();
    let ninf = f64::NEG_INFINITY;
    // the branch metrics, once per step and branch
    ws.gamma.resize(steps * 2 * ns, 0.0);
    for (t, g) in ws.gamma.chunks_exact_mut(2 * ns).enumerate() {
        trellis.branches(&llr[t * n..(t + 1) * n], &mut ws.words, g);
        let la = apriori.get(t).copied().unwrap_or(0.0);
        for (i, g) in g.iter_mut().enumerate() {
            *g += if i % 2 == 0 { la / 2.0 } else { -la / 2.0 };
        }
    }
    let gamma = &ws.gamma;
    let alpha = &mut ws.alpha;
    alpha.clear();
    alpha.resize((steps + 1) * ns, ninf);
    alpha[0] = 0.0;
    for t in 0..steps {
        let (now, later) = alpha[t * ns..(t + 2) * ns].split_at_mut(ns);
        for s in 0..ns {
            if now[s] == ninf {
                continue;
            }
            for u in 0..2 {
                let s2 = trellis.next[s][u];
                later[s2] = max_star(later[s2], now[s] + gamma[(t * ns + s) * 2 + u], exact);
            }
        }
    }
    let beta = &mut ws.beta;
    beta.clear();
    beta.resize((steps + 1) * ns, ninf);
    if terminated {
        beta[steps * ns] = 0.0;
    } else {
        beta[steps * ns..].fill(0.0);
    }
    for t in (0..steps).rev() {
        let (now, later) = beta[t * ns..(t + 2) * ns].split_at_mut(ns);
        for s in 0..ns {
            let mut b = ninf;
            for u in 0..2 {
                b = max_star(b, gamma[(t * ns + s) * 2 + u] + later[trellis.next[s][u]], exact);
            }
            now[s] = b;
        }
    }
    app.clear();
    for t in 0..steps {
        let mut l = [ninf; 2];
        for s in 0..ns {
            let a = alpha[t * ns + s];
            if a == ninf {
                continue;
            }
            for u in 0..2 {
                l[u] = max_star(l[u], a + gamma[(t * ns + s) * 2 + u] + beta[(t + 1) * ns + trellis.next[s][u]], exact);
            }
        }
        app.push(l[0] - l[1]);
    }
    Ok(())
}

/// A terminated rate-$1/n$ convolutional code over blocks of `k` information bits,
/// decoded by soft-decision Viterbi.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvolutionalCode {
    /// The trellis.
    pub trellis: Trellis,
    /// Information bits per block.
    pub k: usize,
    /// Decode hard decisions (the signs of the LLRs) instead of the LLRs.
    pub hard_decision: bool,
}

impl ConvolutionalCode {
    /// The $(133, 171)_{8}$ code of constraint length 7, rate 1/2, on blocks of `k` bits.
    pub fn k7(k: usize) -> ConvolutionalCode {
        let trellis = Trellis::feedforward(7, &[0o133, 0o171]).unwrap_or_else(|_| unreachable!("a valid code"));
        ConvolutionalCode { trellis, k, hard_decision: false }
    }
}

impl BlockCode for ConvolutionalCode {
    fn k(&self) -> usize {
        self.k
    }

    fn n(&self) -> usize {
        self.trellis.n_out * (self.k + self.trellis.constraint as usize - 1)
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k, "bits")?;
        Ok(self.trellis.encode(info, true))
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), self.n(), "LLRs")?;
        if self.hard_decision {
            // a zero LLR (a punctured bit) keeps a zero metric
            let h: Vec<f64> = llr
                .iter()
                .map(|l| {
                    if *l < 0.0 {
                        -1.0
                    } else if *l > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            viterbi(&self.trellis, &h, true)
        } else {
            viterbi(&self.trellis, llr, true)
        }
    }
}

container_traits!(ConvolutionalCode);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::testing::awgn_llr;
    use com_core::rng::Rng;
    use com_core::signal::hard_decisions;

    #[test]
    fn the_k7_code_has_free_distance_10() {
        // the weight of the impulse response is the free distance of (133, 171)
        let t = Trellis::feedforward(7, &[0o133, 0o171]).unwrap();
        let c = t.encode(&[1, 0, 0, 0, 0, 0, 0], true);
        assert_eq!(c.iter().map(|b| u32::from(*b)).sum::<u32>(), 10);
    }

    #[test]
    fn viterbi_corrects_errors_and_bcjr_agrees() {
        let code = ConvolutionalCode::k7(200);
        let mut r = Rng::new(1);
        let info: Vec<u8> = (0..200).map(|_| r.bit()).collect();
        let c = code.encode_block(&info).unwrap();
        let y = awgn_llr(&c, 0.6, &mut r);
        assert!(hard_decisions(&y).iter().zip(&c).filter(|(a, b)| a != b).count() > 5);
        assert_eq!(code.decode_block(&y).unwrap(), info);
        let app = bcjr(&code.trellis, &y, &[], true, true).unwrap();
        assert_eq!(hard_decisions(&app[..200]), info);
        let maxlog = bcjr(&code.trellis, &y, &[], true, false).unwrap();
        assert_eq!(hard_decisions(&maxlog[..200]), info);
    }

    #[test]
    fn branch_tables_and_reused_workspaces_change_nothing() {
        let mut r = Rng::new(5);
        // a table of the 2^n output words (k7, the turbo constituent) and the branches
        // computed one by one (five outputs, four states)
        let codes = [
            Trellis::feedforward(7, &[0o133, 0o171]).unwrap(),
            Trellis::recursive(4, 0o13, 0o15).unwrap(),
            Trellis::feedforward(3, &[5, 7, 7, 5, 6]).unwrap(),
        ];
        let (mut ws, mut app) = (BcjrWorkspace::default(), Vec::new());
        for t in &codes {
            let (ns, n) = (t.n_states(), t.n_out);
            let l: Vec<f64> = (0..n).map(|_| r.gaussian()).collect();
            let (mut words, mut m) = (Vec::new(), vec![0.0; 2 * ns]);
            t.branches(&l, &mut words, &mut m);
            for s in 0..ns {
                for u in 0..2 {
                    assert_eq!(m[2 * s + u].to_bits(), t.branch(t.out[s][u], &l).to_bits());
                }
            }
            for len in [30, 7, 50] {
                let bits: Vec<u8> = (0..len).map(|_| r.bit()).collect();
                let c = t.encode(&bits, true);
                let clean: Vec<f64> = c.iter().map(|b| 4.0 * (1.0 - 2.0 * f64::from(*b))).collect();
                assert_eq!(viterbi(t, &clean, true).unwrap(), bits);
                let y: Vec<f64> = clean.iter().map(|v| v + 2.0 * r.gaussian()).collect();
                let la: Vec<f64> = (0..y.len() / n).map(|_| r.gaussian()).collect();
                for (terminated, exact) in [(true, true), (false, false)] {
                    bcjr_into(t, &y, &la, terminated, exact, &mut ws, &mut app).unwrap();
                    let fresh = bcjr(t, &y, &la, terminated, exact).unwrap();
                    assert!(
                        app.iter().zip(&fresh).all(|(a, b)| a.to_bits() == b.to_bits()) && app.len() == fresh.len()
                    );
                }
            }
        }
    }

    #[test]
    fn recursive_codes_terminate_and_are_systematic() {
        let t = Trellis::recursive(4, 0o13, 0o15).unwrap();
        let bits = [1, 0, 1, 1, 0, 0, 1, 0];
        let c = t.encode(&bits, true);
        assert_eq!(c.len(), 2 * (8 + 3));
        // systematic: every other bit is the input
        assert_eq!((0..8).map(|i| c[2 * i]).collect::<Vec<_>>(), bits);
        // the tail returns to state 0
        let mut s = 0;
        for &b in &bits {
            s = t.next[s][usize::from(b)];
        }
        for _ in 0..3 {
            s = t.next[s][usize::from(t.tail[s])];
        }
        assert_eq!(s, 0);
    }
}

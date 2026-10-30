//! Polar codes.
//!
//! Arıkan's transform $x = u\, F^{\otimes m}$, $F = \begin{pmatrix} 1 & 0 \\\\ 1 & 1
//! \end{pmatrix}$, $N = 2^{m}$, turns $N$ uses of a channel into $N$ synthetic channels that
//! polarise: some become almost noiseless, the others almost useless. The $K$ most
//! reliable positions carry information (with the CRC bits, for CRC-aided list
//! decoding), the others are frozen to zero.
//!
//! # Construction
//!
//! The reliabilities come from the Gaussian approximation of density evolution
//! (Trifonov 2012) at a design $E_{b}/N_{0}$: the mean LLR $\mu$ of a synthetic channel
//! becomes $\phi^{-1}\left(1 - (1 - \phi(\mu))^{2}\right)$ for the worse branch and $2 \mu$
//! for the better one, with the function $\phi$ of Chung, Richardson & Urbanke (2001).
//!
//! # Decoding
//!
//! - successive cancellation (SC, Arıkan 2009), with the min-sum $f$ and the $g$ updates
//!   $f(a, b) = \operatorname{sgn}(a)\operatorname{sgn}(b)\min(|a|, |b|)$,
//!   $g(a, b, \hat u) = b + (1 - 2\hat u) a$;
//! - SC list (Tal & Vardy 2015) of size $L$ with the LLR-based path metric
//!   $\mathrm{PM} \mathrel{+}= \ln\left(1 + e^{-(1 - 2\hat u)\lambda}\right)$
//!   (Balatsoukas-Stimming, Parizi & Burg 2015), and the CRC choosing among the
//!   surviving paths (CA-SCL, the decoder of 5G NR control channels). The CRC bits are
//!   not coding gain: on short blocks a short CRC (6 bits) pays off, a long one costs
//!   more rate than it brings.
//!
//! Both run on the per-layer LLR and partial-sum arrays of Tal & Vardy (2015, Sec. IV),
//! shared between the paths and copied on write, in $O(L n \log n)$ per frame.
//!
//! # Examples
//! ```
//! use com_fec::block::BlockCode;
//! use com_fec::crc::Crc;
//! use com_fec::polar::PolarCode;
//! let code = PolarCode::new(128, 64, 2.0)?.with_crc(Crc::crc6())?.with_list(8)?;
//! let info: Vec<u8> = (0..64).map(|i| (i % 3 == 0) as u8).collect();
//! let c = code.encode_block(&info)?;
//! let llr: Vec<f64> = c.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect(); // noiseless
//! assert_eq!(code.decode_block(&llr)?, info);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. E. Arıkan, "Channel polarization: a method for constructing capacity-achieving
//!    codes for symmetric binary-input memoryless channels," *IEEE Trans. Inf. Theory*,
//!    55(7), 3051–3073, 2009.
//! 2. P. Trifonov, "Efficient design and decoding of polar codes," *IEEE Trans.
//!    Commun.*, 60(11), 3221–3227, 2012.
//! 3. S.-Y. Chung, T. J. Richardson and R. L. Urbanke, "Analysis of sum-product decoding
//!    of low-density parity-check codes using a Gaussian approximation," *IEEE Trans.
//!    Inf. Theory*, 47(2), 657–670, 2001.
//! 4. I. Tal and A. Vardy, "List decoding of polar codes," *IEEE Trans. Inf. Theory*,
//!    61(5), 2213–2226, 2015.
//! 5. A. Balatsoukas-Stimming, M. B. Parizi and A. Burg, "LLR-based successive
//!    cancellation list decoding of polar codes," *IEEE Trans. Signal Process.*, 63(19),
//!    5165–5179, 2015.

use crate::block::{BlockCode, container_traits, expect_len};
use crate::crc::Crc;
use com_core::{Result, ensure};
use com_math::{db2lin, ln_1p_exp};

/// The function $\phi$ of the Gaussian approximation (Chung, Richardson & Urbanke 2001).
fn phi(x: f64) -> f64 {
    if x <= 0.0 {
        1.0
    } else if x < 10.0 {
        (-0.4527 * x.powf(0.86) + 0.0218).exp()
    } else {
        (std::f64::consts::PI / x).sqrt() * (-x / 4.0).exp() * (1.0 - 10.0 / (7.0 * x))
    }
}

/// $\phi^{-1}$ by bisection ($\phi$ decreases from 1 to 0).
fn phi_inv(y: f64) -> f64 {
    if y >= 1.0 {
        return 0.0;
    }
    let (lo, mut hi) = (0.0f64, 1.0f64);
    while phi(hi) > y {
        hi *= 2.0;
        if hi > 1e6 {
            return hi;
        }
    }
    com_math::bisect(|x| phi(x) > y, lo, hi, 100)
}

/// The mean LLR of every synthetic channel of a transform of size `n` (a power of two)
/// whose channels have the mean LLR `mu`.
///
/// The outer stage of the transform acts first on the channel LLRs: the most
/// significant bit of a channel index chooses its branch (the check node
/// $\phi^{-1}(1 - (1 - \phi(\mu))^{2})$ for 0, the variable node $2\mu$ for 1), the
/// lower bits the next stages. After $s$ stages the $2^{s}$ values of the prefixes are
/// distinct, so each is computed once: $n - 1$ inversions of $\phi$ in all, instead of
/// one per channel and stage.
fn reliabilities(n: usize, mu: f64) -> Vec<f64> {
    let mut v = vec![mu];
    while v.len() < n {
        // prefix p (the high bits of the index) becomes 2p (bit 0) and 2p + 1 (bit 1)
        v = v.iter().flat_map(|&m| [phi_inv(1.0 - (1.0 - phi(m)).powi(2)), 2.0 * m]).collect();
    }
    v
}

/// The transform $x = u F^{\otimes m}$ in place.
fn transform(u: &mut [u8]) {
    let n = u.len();
    let mut s = 1;
    while s < n {
        for i in 0..n {
            if i & s == 0 {
                u[i] ^= u[i + s];
            }
        }
        s <<= 1;
    }
}

/// The min-sum $f(a, b) = \operatorname{sgn}(a)\operatorname{sgn}(b)\min(|a|, |b|)$.
#[inline]
fn f(a: f64, b: f64) -> f64 {
    a.signum() * b.signum() * a.abs().min(b.abs())
}

/// The update $g(a, b, \hat u) = b + (1 - 2\hat u) a$.
#[inline]
fn g(a: f64, b: f64, u: u8) -> f64 {
    b + if u == 0 { a } else { -a }
}

/// The arrays of one layer of the list decoder, shared between the paths with
/// reference counts and copied on write (the lazy copy of Tal & Vardy 2015, Sec. IV).
struct Pool<T> {
    /// The length of an array.
    len: usize,
    /// The arrays, one after the other.
    data: Vec<T>,
    /// `of[path]`: the array of a path.
    of: Vec<usize>,
    /// `refs[array]`: the number of paths that use an array.
    refs: Vec<usize>,
    /// The arrays no path uses.
    free: Vec<usize>,
}

impl<T: Copy + Default> Pool<T> {
    fn new(len: usize, arrays: usize, paths: usize) -> Pool<T> {
        Pool {
            len,
            data: vec![T::default(); len * arrays],
            of: vec![0; paths],
            refs: vec![0; arrays],
            free: Vec::new(),
        }
    }

    /// Path 0 alone, on array 0.
    fn reset(&mut self) {
        self.refs.fill(0);
        self.refs[0] = 1;
        self.of[0] = 0;
        self.free.clear();
        self.free.extend((1..self.refs.len()).rev());
    }

    /// Path `to` shares the array of path `from`.
    fn share(&mut self, from: usize, to: usize) {
        let a = self.of[from];
        self.of[to] = a;
        self.refs[a] += 1;
    }

    /// Path `path` stops using its array.
    fn release(&mut self, path: usize) {
        let a = self.of[path];
        self.refs[a] -= 1;
        if self.refs[a] == 0 {
            self.free.push(a);
        }
    }

    /// The array of `path`, made its own before it writes into it: a shared array is
    /// replaced by a free one that receives the first `keep` values of the shared one.
    fn writable(&mut self, path: usize, keep: usize) -> usize {
        let a = self.of[path];
        if self.refs[a] == 1 {
            return a;
        }
        self.refs[a] -= 1;
        let b = self.free.pop().unwrap_or_else(|| unreachable!("at most one array per path"));
        self.refs[b] = 1;
        self.of[path] = b;
        self.data.copy_within(a * self.len..a * self.len + keep, b * self.len);
        b
    }

    fn array(&self, path: usize) -> &[T] {
        let a = self.of[path];
        &self.data[a * self.len..(a + 1) * self.len]
    }

    fn array_mut(&mut self, a: usize) -> &mut [T] {
        &mut self.data[a * self.len..(a + 1) * self.len]
    }
}

/// The SC list decoder on the arrays of Tal & Vardy (2015, Sec. IV), in the LLR form of
/// Balatsoukas-Stimming, Bastani Parizi & Burg (2015): layer $\lambda = 0, \dots, m$ of
/// the decoding tree holds, for every path, the LLRs of its current node
/// ($2^{m - \lambda}$ values, the channel at $\lambda = 0$) and the partial sums of the
/// codeword under construction below it. Leaf $\varphi$ recomputes the layers from the one where
/// its branch turns right ($m$ minus the trailing zeros of $\varphi$) down, which is
/// $O(n \log n)$ per path and frame instead of $O(n^{2} \log n)$ for a leaf computed from
/// the channel; the paths share the arrays they have not written since they split.
struct ListDecoder {
    m: usize,
    list: usize,
    /// `llr[λ]`, λ = 0 (the channel, a single array) to m (the leaf LLR).
    llr: Vec<Pool<f64>>,
    /// `bits[λ]`, λ = 0 to m − 1: the codeword of the node of the layer, its left half
    /// written when its left child is decided, its right half then.
    bits: Vec<Pool<u8>>,
    /// The slots of the active paths, by rank: the order of the sorted candidates,
    /// which breaks the ties of the metrics.
    order: Vec<usize>,
    /// The unused path slots.
    free: Vec<usize>,
    /// Path metric, per slot.
    pm: Vec<f64>,
    /// The parity-check register of every slot after the decisions of its path.
    reg: Vec<[u8; 5]>,
    /// The candidates of a leaf: metric, rank of the parent path, bit.
    cand: Vec<(f64, usize, u8)>,
    /// The leaf LLR of every path, by rank.
    lam: Vec<f64>,
    /// Whether a parent rank already gave its slot to a child.
    claimed: Vec<bool>,
    /// The slots of the next paths, by rank.
    next: Vec<usize>,
    /// The decisions: (rank of the parent, bit), `start[φ]..start[φ + 1]` for leaf φ.
    hist: Vec<(u32, u8)>,
    start: Vec<usize>,
}

impl ListDecoder {
    fn new(n: usize, list: usize) -> ListDecoder {
        let m = n.trailing_zeros() as usize;
        let llr = (0..=m).map(|l| Pool::new(n >> l, if l == 0 { 1 } else { list }, list)).collect();
        let bits = (0..m).map(|l| Pool::new(n >> l, list, list)).collect();
        ListDecoder {
            m,
            list,
            llr,
            bits,
            order: Vec::with_capacity(list),
            free: Vec::with_capacity(list),
            pm: vec![0.0; list],
            reg: vec![[0; 5]; list],
            cand: Vec::with_capacity(2 * list),
            lam: Vec::with_capacity(list),
            claimed: Vec::with_capacity(list),
            next: Vec::with_capacity(list),
            hist: Vec::with_capacity(n * list),
            start: Vec::with_capacity(n + 1),
        }
    }

    /// The LLR of leaf `phi` for the path in slot `path`.
    fn leaf_llr(&mut self, path: usize, phi: usize) -> f64 {
        let m = self.m;
        let first = if phi == 0 { 1 } else { m - phi.trailing_zeros() as usize };
        for layer in first..=m {
            let half = 1usize << (m - layer);
            let (above, below) = self.llr.split_at_mut(layer);
            let (a, b) = above[layer - 1].array(path).split_at(half);
            let pool = &mut below[0];
            let dst = pool.writable(path, 0);
            let child = pool.array_mut(dst);
            if (phi >> (m - layer)) & 1 == 0 {
                for ((c, x), y) in child.iter_mut().zip(a).zip(b) {
                    *c = f(*x, *y);
                }
            } else {
                let partial = &self.bits[layer - 1].array(path)[..half];
                for (((c, x), y), &s) in child.iter_mut().zip(a).zip(b).zip(partial) {
                    *c = g(*x, *y, s);
                }
            }
        }
        self.llr[m].array(path)[0]
    }

    /// Record the decision `bit` of leaf `phi` in the partial sums of slot `path`.
    fn set_bit(&mut self, path: usize, phi: usize, bit: u8) {
        let m = self.m;
        let mut side = phi & 1;
        let pool = &mut self.bits[m - 1];
        let a = pool.writable(path, side);
        pool.array_mut(a)[side] = bit;
        // a completed right child completes its parent: x = (x_left ⊕ x_right, x_right)
        let (mut layer, mut len) = (m - 1, 1);
        while side == 1 && layer > 0 {
            let (above, below) = self.bits.split_at_mut(layer);
            let pool = &mut below[0];
            let a = pool.of[path];
            let node = pool.array_mut(a);
            let (l, r) = node.split_at_mut(len);
            for (x, y) in l.iter_mut().zip(r.iter()) {
                *x ^= y;
            }
            side = (phi >> (m - layer)) & 1;
            let parent = &mut above[layer - 1];
            let b = parent.writable(path, side * 2 * len);
            parent.array_mut(b)[side * 2 * len..(side + 1) * 2 * len].copy_from_slice(node);
            layer -= 1;
            len *= 2;
        }
    }

    /// The decisions $u$ and the metrics of the surviving paths, the most likely first.
    fn decode(&mut self, llr: &[f64], info: &[bool], pc: &[bool]) -> Vec<(Vec<u8>, f64)> {
        let n = llr.len();
        self.reset(llr);
        for phi in 0..n {
            self.candidates(phi, info, pc);
            self.release_unclaimed();
            self.assign_slots();
            self.commit(phi, n, info[phi] && !pc[phi]);
        }
        self.start.push(self.hist.len());
        self.trace_back(n)
    }

    /// One path, slot 0, on the channel LLRs `llr`.
    fn reset(&mut self, llr: &[f64]) {
        for p in &mut self.llr {
            p.reset();
        }
        for p in &mut self.bits {
            p.reset();
        }
        self.llr[0].data.copy_from_slice(llr);
        self.order.clear();
        self.order.push(0);
        self.free.clear();
        self.free.extend((1..self.list).rev());
        self.pm[0] = 0.0;
        self.reg[0] = [0; 5];
        self.hist.clear();
        self.start.clear();
    }

    /// The `list` best children (metric, rank of the parent, bit) of the paths at leaf
    /// `phi`: the frozen bit, the parity-check bit of the register, or both values of
    /// an information bit.
    fn candidates(&mut self, phi: usize, info: &[bool], pc: &[bool]) {
        self.lam.clear();
        for r in 0..self.order.len() {
            let lam = self.leaf_llr(self.order[r], phi);
            self.lam.push(lam);
        }
        self.cand.clear();
        for (r, (&slot, &lam)) in self.order.iter().zip(&self.lam).enumerate() {
            let pm = self.pm[slot];
            let cost = |bit: u8| {
                let s = if bit == 0 { lam } else { -lam };
                pm + ln_1p_exp(-s)
            };
            if pc[phi] {
                // the register before u_φ, rotated once more: its second cell
                let bit = self.reg[slot][1];
                self.cand.push((cost(bit), r, bit));
            } else if info[phi] {
                self.cand.push((cost(0), r, 0));
                self.cand.push((cost(1), r, 1));
            } else {
                self.cand.push((cost(0), r, 0));
            }
        }
        // a stable sort by metric: the ties keep the order (rank, bit)
        self.cand.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then((x.1, x.2).cmp(&(y.1, y.2))));
        self.cand.truncate(self.list);
    }

    /// The paths without a surviving child free their slot and arrays.
    fn release_unclaimed(&mut self) {
        self.claimed.clear();
        self.claimed.resize(self.order.len(), false);
        for c in &self.cand {
            self.claimed[c.1] = true;
        }
        for r in 0..self.order.len() {
            if !self.claimed[r] {
                let slot = self.order[r];
                for p in &mut self.llr {
                    p.release(slot);
                }
                for p in &mut self.bits {
                    p.release(slot);
                }
                self.free.push(slot);
            }
        }
    }

    /// The slot of every candidate: the first child of a parent keeps its slot, a
    /// second one takes a free slot that shares the arrays of the parent.
    fn assign_slots(&mut self) {
        self.claimed.fill(false);
        self.next.clear();
        for k in 0..self.cand.len() {
            let r = self.cand[k].1;
            let parent = self.order[r];
            let slot = if self.claimed[r] {
                let slot = self.free.pop().unwrap_or_else(|| unreachable!("at most L paths"));
                for p in &mut self.llr {
                    p.share(parent, slot);
                }
                for p in &mut self.bits {
                    p.share(parent, slot);
                }
                self.reg[slot] = self.reg[parent];
                slot
            } else {
                self.claimed[r] = true;
                parent
            };
            self.next.push(slot);
        }
    }

    /// The metrics, registers, history and partial sums of the candidates at leaf
    /// `phi` of `n`; `data` when the leaf carries an information bit (not a parity
    /// check), which enters the register.
    fn commit(&mut self, phi: usize, n: usize, data: bool) {
        self.start.push(self.hist.len());
        for k in 0..self.cand.len() {
            let (metric, r, bit) = self.cand[k];
            let slot = self.next[k];
            self.pm[slot] = metric;
            let reg = &mut self.reg[slot];
            reg.rotate_left(1);
            if data {
                reg[0] ^= bit;
            }
            self.hist.push((r as u32, bit));
            if phi + 1 < n {
                self.set_bit(slot, phi, bit);
            }
        }
        std::mem::swap(&mut self.order, &mut self.next);
    }

    /// The decisions of every surviving path, from the history of its ancestors.
    fn trace_back(&self, n: usize) -> Vec<(Vec<u8>, f64)> {
        (0..self.order.len())
            .map(|rank| {
                let mut u = vec![0u8; n];
                let mut r = rank;
                for phi in (0..n).rev() {
                    let (parent, bit) = self.hist[self.start[phi] + r];
                    u[phi] = bit;
                    r = parent as usize;
                }
                (u, self.pm[self.order[rank]])
            })
            .collect()
    }
}

/// A polar code of length $N$ with $K$ information bits.
#[derive(Clone, Debug, PartialEq)]
pub struct PolarCode {
    n: usize,
    k: usize,
    /// The positions that carry information (and CRC) bits, in increasing order.
    info_positions: Vec<usize>,
    /// The design $E_{b}/N_{0}$ of the construction (dB); `None` for the positions of a
    /// table ([`PolarCode::with_positions`]).
    design_ebn0_db: Option<f64>,
    /// The CRC appended to the information bits (CA-SCL).
    crc: Option<Crc>,
    /// List size $L$ (1: successive cancellation).
    list: usize,
    /// The parity-check positions of a PC-polar code (see [`PolarCode::with_positions`]).
    pc_positions: Vec<usize>,
}

/// The parity-check bits of PC-polar codes (TS 38.212, Sec. 5.3.1.2): a cyclic register
/// of length 5 runs over $u_{0}, u_{1}, \dots$; a parity-check position takes its first
/// cell, an information position adds its bit to it.
fn pc_register(u: &[u8], info: &[bool], pc: &[bool], upto: usize) -> u8 {
    let mut y = [0u8; 5];
    for i in 0..upto {
        y.rotate_left(1);
        if info[i] && !pc[i] {
            y[0] ^= u[i];
        }
    }
    y.rotate_left(1);
    y[0]
}

impl PolarCode {
    /// The code of length `n` (a power of two) with `k` information bits, built by the
    /// Gaussian approximation at the design $E_{b}/N_{0}$ `design_ebn0_db`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless $N$ is a power of two, $0 < K \le N$ and the
    /// design $E_{b}/N_{0}$ is finite.
    pub fn new(n: usize, k: usize, design_ebn0_db: f64) -> Result<PolarCode> {
        ensure(n.is_power_of_two() && n >= 2, || format!("the length must be a power of two, got {n}"))?;
        ensure(k >= 1 && k <= n, || format!("0 < K ≤ N, got K = {k}"))?;
        ensure(design_ebn0_db.is_finite(), || format!("the design Eb/N0 must be finite, got {design_ebn0_db}"))?;
        let mut code = PolarCode {
            n,
            k,
            info_positions: Vec::new(),
            design_ebn0_db: Some(design_ebn0_db),
            crc: None,
            list: 1,
            pc_positions: Vec::new(),
        };
        code.construct(design_ebn0_db)?;
        Ok(code)
    }

    fn construct(&mut self, design_ebn0_db: f64) -> Result<()> {
        let extra = self.crc.as_ref().map_or(0, |c| c.width() as usize);
        let a = self.k + extra;
        ensure(a <= self.n, || format!("{} information and CRC bits exceed N = {}", a, self.n))?;
        // BPSK: mean channel LLR 2/σ² = 4 R Eb/N0 with the rate of the information bits
        let rate = self.k as f64 / self.n as f64;
        let mu = 4.0 * rate * db2lin(design_ebn0_db);
        let mut order: Vec<(f64, usize)> = reliabilities(self.n, mu).into_iter().zip(0..self.n).collect();
        order.sort_by(|x, y| y.0.total_cmp(&x.0).then(y.1.cmp(&x.1)));
        let mut pos: Vec<usize> = order.iter().take(a).map(|o| o.1).collect();
        pos.sort_unstable();
        self.info_positions = pos;
        Ok(())
    }

    /// The same code with a CRC appended to the information bits (the information set
    /// grows by its degree, $K$ is kept; CA-SCL keeps the path that passes it). A second
    /// call replaces the CRC of the first.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the information bits and the CRC exceed the block,
    /// or for a code built by [`PolarCode::with_positions`], whose information set is
    /// fixed by its table (a standard with a CRC checks it outside the code, as
    /// [`crate::nr::NrPolar`] with [`PolarCode::decode_list`]).
    pub fn with_crc(mut self, crc: Crc) -> Result<PolarCode> {
        let Some(design_ebn0_db) = self.design_ebn0_db else {
            return Err(com_core::Error::Config(
                "with_crc: the information positions of a code built by with_positions are fixed; \
                 check the CRC outside the code (decode_list)"
                    .into(),
            ));
        };
        self.crc = Some(crc);
        self.construct(design_ebn0_db)?;
        Ok(self)
    }

    /// The same code decoded by SC list with `list` paths (1: successive
    /// cancellation).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for `list = 0`.
    pub fn with_list(mut self, list: usize) -> Result<PolarCode> {
        ensure(list >= 1, || "the list decoder needs at least one path".into())?;
        self.list = list;
        Ok(self)
    }

    /// The CRC appended to the information bits (CA-SCL), if any ([`PolarCode::with_crc`]).
    pub fn crc(&self) -> Option<&Crc> {
        self.crc.as_ref()
    }

    /// The list size $L$ of the decoder (1: successive cancellation).
    pub fn list(&self) -> usize {
        self.list
    }

    /// The code of length `n` (a power of two) whose information (and CRC) bits sit at
    /// `info_positions` and parity-check bits at `pc_positions`, all other positions
    /// frozen to zero: the codes of a standard whose construction is a table (5G NR,
    /// [`crate::nr::NrPolar`]). A parity-check bit is the first cell of a cyclic
    /// register of length 5 that every information bit before it updates (TS 38.212,
    /// Sec. 5.3.1.2); the list decoder sets it from the decisions of its path.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless $N$ is a power of two and the positions are
    /// distinct and below $N$.
    pub fn with_positions(n: usize, info_positions: &[usize], pc_positions: &[usize]) -> Result<PolarCode> {
        ensure(n.is_power_of_two() && n >= 2, || format!("the length must be a power of two, got {n}"))?;
        let mut seen = vec![false; n];
        for &p in info_positions.iter().chain(pc_positions) {
            ensure(p < n && !seen[p], || format!("position {p} repeated or beyond N = {n}"))?;
            seen[p] = true;
        }
        let mut info = info_positions.to_vec();
        info.sort_unstable();
        let mut pc = pc_positions.to_vec();
        pc.sort_unstable();
        ensure(!info.is_empty(), || "the code has no information positions".to_string())?;
        Ok(PolarCode {
            n,
            k: info.len(),
            info_positions: info,
            design_ebn0_db: None,
            crc: None,
            list: 1,
            pc_positions: pc,
        })
    }

    /// The positions carrying information and CRC bits.
    pub fn info_positions(&self) -> &[usize] {
        &self.info_positions
    }

    /// The parity-check positions (empty but for [`PolarCode::with_positions`]).
    pub fn pc_positions(&self) -> &[usize] {
        &self.pc_positions
    }

    /// The information (and CRC) bits of the surviving paths of the list decoder, the
    /// most likely first, for a CRC checked outside the code (after a deinterleaver).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] if `llr` does not have $N$ values.
    pub fn decode_list(&self, llr: &[f64]) -> Result<Vec<Vec<u8>>> {
        expect_len(llr.len(), self.n, "LLRs")?;
        Ok(self.decode_paths(llr).iter().map(|(u, _)| self.info_positions.iter().map(|&p| u[p]).collect()).collect())
    }

    fn masks(&self) -> (Vec<bool>, Vec<bool>) {
        let mut info = vec![false; self.n];
        let mut pc = vec![false; self.n];
        for &p in &self.info_positions {
            info[p] = true;
        }
        for &p in &self.pc_positions {
            info[p] = true;
            pc[p] = true;
        }
        (info, pc)
    }

    /// The decisions $u$ and the metrics of the surviving paths of the list decoder, the
    /// most likely first.
    fn decode_paths(&self, llr: &[f64]) -> Vec<(Vec<u8>, f64)> {
        let (info, pc) = self.masks();
        // the workspace (about L n values) costs less than the noise of the timing to
        // allocate per frame; `BlockCode` takes `&self` and cannot keep it
        ListDecoder::new(self.n, self.list).decode(llr, &info, &pc)
    }
}

impl BlockCode for PolarCode {
    fn k(&self) -> usize {
        self.k
    }

    fn n(&self) -> usize {
        self.n
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k, "bits")?;
        let payload = match &self.crc {
            Some(c) => c.attach(info),
            None => info.to_vec(),
        };
        let mut u = vec![0u8; self.n];
        for (&p, &b) in self.info_positions.iter().zip(&payload) {
            u[p] = b & 1;
        }
        if !self.pc_positions.is_empty() {
            let (info, pc) = self.masks();
            for &p in &self.pc_positions {
                u[p] = pc_register(&u, &info, &pc, p);
            }
        }
        transform(&mut u);
        Ok(u)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), self.n, "LLRs")?;
        let paths = self.decode_paths(llr);
        let payload = |u: &[u8]| -> Vec<u8> { self.info_positions.iter().map(|&p| u[p]).collect() };
        let chosen = match &self.crc {
            Some(c) => {
                paths.iter().map(|(u, _)| payload(u)).find(|w| c.check(w)).unwrap_or_else(|| payload(&paths[0].0))
            }
            None => payload(&paths[0].0),
        };
        Ok(chosen[..self.k].to_vec())
    }
}

container_traits!(PolarCode);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::testing::awgn_llr;
    use com_core::rng::Rng;

    #[test]
    fn the_transform_is_an_involution_and_sc_decodes_noiseless_words() {
        let mut r = Rng::new(1);
        let mut u: Vec<u8> = (0..64).map(|_| r.bit()).collect();
        let u0 = u.clone();
        transform(&mut u);
        transform(&mut u);
        assert_eq!(u, u0);
        let code = PolarCode::new(64, 32, 1.0).unwrap();
        let info: Vec<u8> = (0..32).map(|_| r.bit()).collect();
        let c = code.encode_block(&info).unwrap();
        let y: Vec<f64> = c.iter().map(|b| 4.0 * (1.0 - 2.0 * f64::from(*b))).collect();
        assert_eq!(code.decode_block(&y).unwrap(), info);
    }

    #[test]
    fn the_construction_freezes_the_worst_channels() {
        // N = 8, K = 4: the classical information set {3, 5, 6, 7}
        let code = PolarCode::new(8, 4, 2.0).unwrap();
        assert_eq!(code.info_positions(), &[3, 5, 6, 7]);
    }

    #[test]
    fn the_list_and_the_crc_lower_the_block_error_rate() {
        let base = PolarCode::new(128, 64, 2.0).unwrap();
        // a short CRC: 11 bits would cost 17 % of the rate of a block of 64 bits
        let scl = base.clone().with_crc(Crc::crc6()).unwrap().with_list(8).unwrap();
        let mut r = Rng::new(2);
        let (mut e_sc, mut e_scl) = (0, 0);
        let s2 = 1.0 / 10f64.powf(0.2); // Eb/N0 = 2 dB at rate 1/2
        for _ in 0..300 {
            let info: Vec<u8> = (0..64).map(|_| r.bit()).collect();
            for (code, e) in [(&base, &mut e_sc), (&scl, &mut e_scl)] {
                let c = code.encode_block(&info).unwrap();
                let y = awgn_llr(&c, s2, &mut r);
                if code.decode_block(&y).unwrap() != info {
                    *e += 1;
                }
            }
        }
        assert!(e_scl * 2 < e_sc, "SC {e_sc} vs CA-SCL {e_scl} block errors");
    }

    #[test]
    fn with_crc_rejects_the_fixed_positions_of_a_table() {
        let table = PolarCode::with_positions(16, &[7, 11, 13, 14, 15], &[12]).unwrap();
        let err = table.clone().with_crc(Crc::crc6()).unwrap_err();
        assert!(matches!(err, com_core::Error::Config(_)), "{err}");
        // the table code is untouched: no CRC, positions disjoint from the parity checks
        assert!(table.crc().is_none());
        assert!(table.info_positions().iter().all(|p| !table.pc_positions().contains(p)));
    }

    #[test]
    fn with_crc_twice_keeps_k_and_the_last_crc() {
        let once = PolarCode::new(128, 64, 2.0).unwrap().with_crc(Crc::crc6()).unwrap();
        let twice =
            PolarCode::new(128, 64, 2.0).unwrap().with_crc(Crc::crc11()).unwrap().with_crc(Crc::crc6()).unwrap();
        assert_eq!(once, twice);
        assert_eq!(BlockCode::k(&twice), 64);
        assert_eq!(twice.info_positions().len(), 64 + 6);
        assert_eq!(twice.crc().map(Crc::width), Some(6));
    }

    /// The decoder before the arrays of Tal & Vardy: every leaf computed from the channel
    /// with the decisions of its path, $O(L n^{2} \log n)$, and the paths copied whole.
    fn reference_paths(code: &PolarCode, llr: &[f64]) -> Vec<(Vec<u8>, f64)> {
        fn leaf_llr(i: usize, ch: &[f64], u: &[u8]) -> f64 {
            let n = ch.len();
            if n == 1 {
                return ch[0];
            }
            let h = n / 2;
            let (a, b) = ch.split_at(h);
            if i < h {
                let v: Vec<f64> = a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect();
                leaf_llr(i, &v, u)
            } else {
                let mut x = u[..h].to_vec();
                transform(&mut x);
                let v: Vec<f64> =
                    a.iter().zip(b).zip(&x).map(|((p, q), &xb)| q + if xb == 0 { *p } else { -p }).collect();
                leaf_llr(i - h, &v, &u[h..])
            }
        }
        let (info, pc) = code.masks();
        let mut paths: Vec<(Vec<u8>, f64)> = vec![(Vec::with_capacity(code.n), 0.0)];
        for i in 0..code.n {
            let mut next = Vec::with_capacity(paths.len() * 2);
            for (u, pm) in &paths {
                let mut full = u.clone();
                full.resize(code.n, 0);
                let lam = leaf_llr(i, llr, &full);
                let cost = |bit: u8| {
                    let s = if bit == 0 { lam } else { -lam };
                    pm + ln_1p_exp(-s)
                };
                let bits: &[u8] = if pc[i] {
                    &[pc_register(&full, &info, &pc, i)]
                } else if info[i] {
                    &[0, 1]
                } else {
                    &[0]
                };
                for &bit in bits {
                    let mut v = u.clone();
                    v.push(bit);
                    next.push((v, cost(bit)));
                }
            }
            next.sort_by(|a, b| a.1.total_cmp(&b.1));
            next.truncate(code.list);
            paths = next;
        }
        paths
    }

    #[test]
    fn the_list_decoder_matches_the_leaf_by_leaf_reference() {
        let mut r = Rng::new(2015);
        let mut cases = 0;
        for m in 3..=10 {
            let n = 1usize << m;
            for list in [1, 2, 8, 32] {
                let frames = 3;
                for frame in 0..frames {
                    let k = 1 + r.below(n - 1);
                    let code = match frame % 3 {
                        0 => PolarCode::new(n, k, -1.0 + 4.0 * r.uniform()).unwrap(),
                        1 if k + 6 <= n => PolarCode::new(n, k, 2.0).unwrap().with_crc(Crc::crc6()).unwrap(),
                        _ => {
                            // a table code with parity-check positions
                            let mut pos: Vec<usize> = (0..n).collect();
                            r.shuffle(&mut pos);
                            let n_pc = (n / 8).min(n - k);
                            PolarCode::with_positions(n, &pos[..k], &pos[k..k + n_pc]).unwrap()
                        }
                    }
                    .with_list(list)
                    .unwrap();
                    let info: Vec<u8> = (0..code.k).map(|_| r.bit()).collect();
                    let c = code.encode_block(&info).unwrap();
                    let s2 = 0.3 + 1.5 * r.uniform();
                    let mut llr = awgn_llr(&c, s2, &mut r);
                    if frame == 1 {
                        // integer LLRs and zeros: ties between the paths
                        for l in &mut llr {
                            *l = (*l / 2.0).round();
                        }
                    }
                    let got = code.decode_paths(&llr);
                    let want = reference_paths(&code, &llr);
                    assert_eq!(got.len(), want.len(), "n {n}, L {list}");
                    for ((gu, gm), (wu, wm)) in got.iter().zip(&want) {
                        assert_eq!(gu, wu, "n {n}, K {k}, L {list}, frame {frame}");
                        assert_eq!(gm.to_bits(), wm.to_bits(), "n {n}, K {k}, L {list}, frame {frame}");
                    }
                    cases += 1;
                }
            }
        }
        assert!(cases > 40);
    }

    #[test]
    fn a_non_finite_design_ebn0_is_rejected() {
        assert!(PolarCode::new(64, 32, f64::NAN).is_err());
        assert!(PolarCode::new(64, 32, f64::INFINITY).is_err());
        assert!(PolarCode::new(64, 32, 1.0).unwrap().with_list(0).is_err());
    }

    /// The reliability of one channel by the recursion over the stages, as the
    /// construction computed it before the values of the prefixes were shared.
    fn former_reliability(i: usize, n: usize, mu: f64) -> f64 {
        if n == 1 {
            return mu;
        }
        let m = if i < n / 2 { phi_inv(1.0 - (1.0 - phi(mu)).powi(2)) } else { 2.0 * mu };
        former_reliability(i % (n / 2), n / 2, m)
    }

    #[test]
    fn the_shared_reliabilities_are_those_of_the_recursion() {
        for n in [2usize, 8, 64, 1024] {
            for mu in [0.05, 1.3, 7.0, 40.0] {
                let r = reliabilities(n, mu);
                assert_eq!(r.len(), n);
                for (i, v) in r.iter().enumerate() {
                    assert_eq!(v.to_bits(), former_reliability(i, n, mu).to_bits(), "n {n}, mu {mu}, i {i}");
                }
            }
        }
    }
}

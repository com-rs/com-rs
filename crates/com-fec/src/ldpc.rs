//! Low-density parity-check (LDPC) codes.
//!
//! A code is the null space of a sparse parity-check matrix $H$ ($m \times n$):
//! $H c = 0$. Its Tanner graph links every variable node (coded bit) to the check
//! nodes (parity equations) it takes part in.
//!
//! # Construction
//!
//! [`LdpcCode::peg`] builds $H$ by progressive edge growth (Hu, Eleftheriou & Arnold
//! 2005): every new edge of a variable node goes to the check node farthest from it in
//! the current graph (lowest degree among the farthest), which maximises the local girth.
//! [`LdpcCode::from_rows`] takes any $H$ (for instance the lifted base graph of a
//! standard), and [`LdpcCode::from_alist`] reads it from MacKay's `alist` format (the
//! format of the matrices of MacKay's encyclopedia and of AFF3CT). The encoder brings $H$ to the systematic form $[A \mid I]$ by Gaussian
//! elimination over GF(2) (with a column permutation), so that the parity bits are
//! $p = A u$.
//!
//! # Decoding
//!
//! Message passing on the Tanner graph, with the LLR convention of the library
//! (positive for a 0). A variable node sends the sum of its channel LLR and of the
//! other incoming messages; a check node sends
//!
//! $$
//! L_{c \to v} = 2 \operatorname{atanh} \prod_{v' \ne v} \tanh\frac{L_{v' \to c}}{2}
//! \quad \text{(sum-product)}, \qquad
//! L_{c \to v} \approx \alpha \prod_{v' \ne v} \operatorname{sgn} L_{v' \to c}
//! \min_{v' \ne v} |L_{v' \to c}| \quad \text{(min-sum)},
//! $$
//!
//! with the normalisation $\alpha$ (Chen & Fossorier 2002) or an offset. The schedule is
//! flooding, or layered (row by row with immediate updates, Hocevar 2004), which
//! converges in about half the iterations; decoding stops as soon as $H \hat c = 0$.
//!
//! The decoder settings of a new code ([`LdpcDecoder`], the field `decoder`) are the
//! defaults of the crate, which the specifications (`com_spec`) and the Python bindings
//! also take: [`DEFAULT_RULE`] (sum-product), [`DEFAULT_LAYERED`] (layered) and
//! [`DEFAULT_MAX_ITERATIONS`] (50);
//! [`DEFAULT_ALPHA`] is the scale or the offset of a min-sum rule named without it,
//! [`DEFAULT_VAR_DEGREE`] and [`DEFAULT_PEG_SEED`] those of a PEG construction.
//!
//! # Examples
//! ```
//! use com_fec::block::BlockCode;
//! use com_fec::ldpc::LdpcCode;
//! let code = LdpcCode::peg(256, 128, 3, 1)?;                 // (256, 128), degree 3
//! assert_eq!(code.k(), 128);
//! let info: Vec<u8> = (0..128).map(|i| (i % 5 == 2) as u8).collect();
//! let c = code.encode_block(&info)?;
//! assert!(code.is_codeword(&c));
//! let llr: Vec<f64> = c.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect(); // noiseless
//! assert_eq!(code.decode_block(&llr)?, info);
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. R. G. Gallager, "Low-density parity-check codes," *IRE Trans. Inf. Theory*, 8(1),
//!    21–28, 1962.
//! 2. X.-Y. Hu, E. Eleftheriou and D. M. Arnold, "Regular and irregular progressive
//!    edge-growth Tanner graphs," *IEEE Trans. Inf. Theory*, 51(1), 386–398, 2005.
//! 3. J. Chen and M. P. C. Fossorier, "Near optimum universal belief propagation based
//!    decoding of low-density parity check codes," *IEEE Trans. Commun.*, 50(3), 406–414,
//!    2002.
//! 4. D. E. Hocevar, "A reduced complexity decoder architecture via layered decoding of
//!    LDPC codes," *Proc. IEEE SiPS*, 107–112, 2004.
//! 5. T. Richardson and R. Urbanke, *Modern Coding Theory*, Cambridge University Press,
//!    2008.

use crate::block::{BlockCode, SoftDecode, container_traits, expect_len};
use com_core::{Error, Result, ensure};
use com_math::gf2;
use std::collections::VecDeque;

/// The default check-node rule: sum-product, the exact belief propagation (the
/// reference of the performance tables of the standards).
pub const DEFAULT_RULE: CheckRule = CheckRule::SumProduct;
/// The default scale $\alpha$ of the normalised min-sum, or offset $\beta$ of the offset
/// min-sum, when a rule is named without it.
pub const DEFAULT_ALPHA: f64 = 0.8;
/// The default schedule: layered.
pub const DEFAULT_LAYERED: bool = true;
/// The default maximum number of iterations.
pub const DEFAULT_MAX_ITERATIONS: usize = 50;
/// The default degree of the variable nodes of a PEG construction.
pub const DEFAULT_VAR_DEGREE: usize = 3;
/// The default seed of a PEG construction.
pub const DEFAULT_PEG_SEED: u64 = 1;

/// The bound of $|\prod \tanh|$ in the check node of the sum-product rule, so that its
/// $\operatorname{atanh}$ stays finite (a message of at most $2 \operatorname{atanh}(1 -
/// 10^{-12}) \approx 28.3$).
const TANH_BOUND: f64 = 0.999_999_999_999;

/// The check-node rule of the decoder.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum CheckRule {
    /// Sum-product (belief propagation, exact).
    SumProduct,
    /// Min-sum scaled by $\alpha$ (1 for plain min-sum, 0.75 to 0.8 typically).
    NormalizedMinSum(f64),
    /// Min-sum with the offset $\beta$ subtracted from the magnitudes.
    OffsetMinSum(f64),
}

impl CheckRule {
    /// The rule of a name: `sum_product`, `min_sum` (plain), `normalized_min_sum`
    /// (scaled by `alpha`) or `offset_min_sum` (offset `alpha`), case-insensitive.
    ///
    /// # Errors
    /// [`Error::Config`] for another name.
    pub fn from_name(rule: &str, alpha: f64) -> Result<CheckRule> {
        match rule.to_ascii_lowercase().as_str() {
            "sum_product" => Ok(CheckRule::SumProduct),
            "min_sum" => Ok(CheckRule::NormalizedMinSum(1.0)),
            "normalized_min_sum" => Ok(CheckRule::NormalizedMinSum(alpha)),
            "offset_min_sum" => Ok(CheckRule::OffsetMinSum(alpha)),
            other => Err(Error::Config(format!(
                "rule must be sum_product, min_sum, normalized_min_sum or offset_min_sum, got {other:?}"
            ))),
        }
    }

    /// The name of the rule, as [`CheckRule::from_name`] reads it (`min_sum` is the
    /// normalised min-sum of scale 1).
    ///
    /// # Examples
    /// ```
    /// use com_fec::ldpc::{CheckRule, DEFAULT_RULE};
    /// assert_eq!(DEFAULT_RULE.name(), "sum_product");
    /// assert_eq!(CheckRule::from_name("min_sum", 0.5)?.name(), "normalized_min_sum");
    /// # Ok::<(), com_core::Error>(())
    /// ```
    pub fn name(self) -> &'static str {
        match self {
            CheckRule::SumProduct => "sum_product",
            CheckRule::NormalizedMinSum(_) => "normalized_min_sum",
            CheckRule::OffsetMinSum(_) => "offset_min_sum",
        }
    }
}

/// The systematic encoder of a code.
#[derive(Clone, Debug, PartialEq)]
enum Encoder {
    /// The positions of the parity bits, and the information bits each one sums
    /// (from the Gaussian elimination of $H$).
    Sums(Vec<(usize, Vec<usize>)>),
    /// Irregular repeat-accumulate: the information bits are the first $k$, parity bit
    /// $p_{r}$ (position $k + r$) closes check $r$ with $p_{r-1}$ (the dual-diagonal part
    /// of $H$): $p_{r} = p_{r-1} \oplus \bigoplus_{v \in r,\, v < k} c_{v}$.
    Accumulate,
    /// A core and an extension (the structure of the codes of 5G NR): the information
    /// bits are the first $k$; the first `core` checks involve the bits below
    /// $k + \mathrm{core}$ only, and core parity bit $j$ is the sum of the syndromes of
    /// the information bits over the checks `solve[j]` (the inverse of the core); check
    /// $r \ge \mathrm{core}$ then closes on its own parity bit $k + r$, the last of its
    /// row.
    Core { core: usize, solve: Vec<Vec<usize>> },
}

/// The settings of the belief-propagation decoder of an [`LdpcCode`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LdpcDecoder {
    /// The check-node rule.
    pub rule: CheckRule,
    /// Layered (true) or flooding schedule.
    pub layered: bool,
    /// Maximum iterations (0: the channel LLRs are returned as they are).
    pub max_iterations: usize,
}

impl Default for LdpcDecoder {
    /// [`DEFAULT_RULE`], [`DEFAULT_LAYERED`], [`DEFAULT_MAX_ITERATIONS`].
    fn default() -> LdpcDecoder {
        LdpcDecoder { rule: DEFAULT_RULE, layered: DEFAULT_LAYERED, max_iterations: DEFAULT_MAX_ITERATIONS }
    }
}

/// An LDPC code with its systematic encoder and its decoder settings.
#[derive(Clone, Debug, PartialEq)]
pub struct LdpcCode {
    n: usize,
    /// The variable nodes of every check node (the rows of $H$).
    rows: Vec<Vec<usize>>,
    /// The messages of check `c` sit at `start[c]..start[c + 1]` of the edges (row
    /// after row), computed once with the rows.
    start: Vec<usize>,
    /// Positions of the information bits in the codeword.
    info: Vec<usize>,
    /// How the parity bits follow from the information bits.
    encoder: Encoder,
    /// The decoder settings.
    pub decoder: LdpcDecoder,
}

impl LdpcCode {
    /// The code of checked rows, its information positions and its encoder, with the
    /// edge offsets of the decoder and the default settings.
    fn assemble(n: usize, rows: Vec<Vec<usize>>, info: Vec<usize>, encoder: Encoder) -> LdpcCode {
        let mut start = Vec::with_capacity(rows.len() + 1);
        start.push(0);
        for row in &rows {
            start.push(start[start.len() - 1] + row.len());
        }
        LdpcCode { n, rows, start, info, encoder, decoder: LdpcDecoder::default() }
    }

    /// The code of the parity-check matrix whose rows list the variable nodes of every
    /// check; $k = n - \operatorname{rank} H$.
    ///
    /// # Errors
    /// [`Error::Config`] for a variable index out of range or a code without
    /// information bits.
    pub fn from_rows(n: usize, rows: Vec<Vec<usize>>) -> Result<LdpcCode> {
        ensure(rows.iter().flatten().all(|v| *v < n), || "a check refers to a missing variable".into())?;
        // the reduced parity-check matrix: every pivot (parity) bit is a sum of
        // information bits
        let mut m = gf2::Matrix::from_rows(n, &rows);
        let pivots = m.reduce();
        let is_pivot: Vec<bool> = (0..n).map(|c| pivots.contains(&c)).collect();
        let info: Vec<usize> = (0..n).filter(|c| !is_pivot[*c]).collect();
        ensure(!info.is_empty(), || "the code has no information bits".into())?;
        let parity = pivots
            .iter()
            .enumerate()
            .map(|(row, &c)| (c, info.iter().copied().filter(|&v| m.get(row, v)).collect()))
            .collect();
        Ok(LdpcCode::assemble(n, rows, info, Encoder::Sums(parity)))
    }

    /// An irregular repeat-accumulate code: check $r$ of `rows` holds information bits
    /// (positions below $k = n - m$, $m$ the number of checks) and the parity bits
    /// $k + r$ and, for $r > 0$, $k + r - 1$ only (the dual-diagonal structure of the
    /// DVB-S2, DVB-T2 and IEEE 802.11n codes). The encoder accumulates the checks, in
    /// $O(n)$, without the elimination of [`LdpcCode::from_rows`].
    ///
    /// # Errors
    /// [`Error::Config`] if the checks do not have this structure.
    pub fn accumulate(n: usize, rows: Vec<Vec<usize>>) -> Result<LdpcCode> {
        let m = rows.len();
        ensure(m > 0 && m < n, || format!("{m} checks for {n} bits"))?;
        let k = n - m;
        for (r, row) in rows.iter().enumerate() {
            let parity: Vec<usize> = row.iter().copied().filter(|&v| v >= k).collect();
            let expected: Vec<usize> = if r == 0 { vec![k] } else { vec![k + r - 1, k + r] };
            let mut sorted = parity.clone();
            sorted.sort_unstable();
            ensure(sorted == expected, || {
                format!("check {r} is not dual-diagonal: parity bits {parity:?}, expected {expected:?}")
            })?;
            ensure(row.iter().all(|&v| v < n), || "a check refers to a missing variable".into())?;
        }
        Ok(LdpcCode::assemble(n, rows, (0..k).collect(), Encoder::Accumulate))
    }

    /// A code whose $m$ checks have a core and an extension, the structure of the LDPC
    /// codes of 5G NR (TS 38.212, Sec. 5.3.2): the information bits are the first $k$;
    /// the first `core` checks involve the bits below $k + \mathrm{core}$ only, with an
    /// invertible square block on the core parity bits $k, \dots, k + \mathrm{core} - 1$;
    /// every other check $r$ contains the parity bit $k + r$ and no bit above it. The
    /// encoder inverts the core block once, by Gauss–Jordan elimination over GF(2), then
    /// computes the extension parity bits one after the other, without the elimination of
    /// the whole matrix of [`LdpcCode::from_rows`].
    ///
    /// # Errors
    /// [`Error::Config`] if the checks do not have this structure or the core block is
    /// singular.
    pub fn from_core_rows(n: usize, rows: Vec<Vec<usize>>, k: usize, core: usize) -> Result<LdpcCode> {
        let m = rows.len();
        ensure(k > 0 && k + m == n && core <= m, || format!("{m} checks, {core} in the core, for k = {k}, n = {n}"))?;
        ensure(rows.iter().flatten().all(|v| *v < n), || "a check refers to a missing variable".into())?;
        for (r, row) in rows.iter().enumerate() {
            let top = row.iter().copied().max().unwrap_or(0);
            if r < core {
                ensure(top < k + core, || format!("core check {r} reaches bit {top} beyond the core"))?;
            } else {
                ensure(top == k + r && row.iter().filter(|&&v| v == k + r).count() == 1, || {
                    format!("extension check {r} does not close on its parity bit {}", k + r)
                })?;
            }
        }
        // parity bit j = (B^-1 s)_j: the checks of row j of the inverse of the core block
        let solve = core_inverse(&rows, k, core)?;
        Ok(LdpcCode::assemble(n, rows, (0..k).collect(), Encoder::Core { core, solve }))
    }

    /// A code of length `n` with `n - k` checks built by progressive edge growth, every
    /// variable node of degree `var_degree` (seeded tie breaks).
    ///
    /// # Errors
    /// [`Error::Config`] for $k \ge n$ or a degree larger than the number of checks.
    pub fn peg(n: usize, k: usize, var_degree: usize, seed: u64) -> Result<LdpcCode> {
        ensure(k < n && k > 0, || format!("PEG needs 0 < k < n, got k = {k}, n = {n}"))?;
        let m = n - k;
        ensure(var_degree >= 1 && var_degree <= m, || format!("variable degree {var_degree} for {m} checks"))?;
        let mut rng = com_core::rng::Rng::new(seed);
        let mut check_vars: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut var_checks: Vec<Vec<usize>> = vec![Vec::new(); n];
        let all: Vec<usize> = (0..m).collect();
        for v in 0..n {
            for e in 0..var_degree {
                let c = if e == 0 {
                    lowest_degree(&all, &check_vars, &mut rng)
                } else {
                    lowest_degree(&farthest_checks(v, &check_vars, &var_checks), &check_vars, &mut rng)
                };
                check_vars[c].push(v);
                var_checks[v].push(c);
            }
        }
        LdpcCode::from_rows(n, check_vars)
    }

    /// The code of a parity-check matrix written in MacKay's `alist` format (the format
    /// of MacKay's *Encyclopedia of Sparse Graph Codes* and of the matrices shipped with
    /// AFF3CT), with the systematic encoder of [`LdpcCode::from_rows`]. The rows of $H$
    /// keep the order of the file, which is the order of a layered schedule.
    ///
    /// The format is a list of integers: $n$ and $m$ (columns and rows of $H$), the
    /// largest column and row degrees, the $n$ column degrees, the $m$ row degrees, then
    /// for every column the (1-based) rows of its ones, and for every row the columns of
    /// its ones. The lists may be padded with zeros up to the largest degree or not;
    /// the text after a `#` is a comment. The two descriptions of the matrix must agree.
    ///
    /// # Errors
    /// [`Error::Config`] for a malformed file: a missing or extra number, a degree or an
    /// index out of range, a repeated one, or column and row lists that disagree; or
    /// for a matrix that [`LdpcCode::from_rows`] rejects.
    ///
    /// # Examples
    /// ```
    /// use com_fec::block::BlockCode;
    /// use com_fec::ldpc::LdpcCode;
    /// // the (7, 4) Hamming code: n = 7 columns, m = 3 rows
    /// let alist = "7 3\n3 4\n3 2 2 2 1 1 1\n4 4 4\n\
    ///              1 2 3\n1 2 0\n1 3 0\n2 3 0\n1 0 0\n2 0 0\n3 0 0\n\
    ///              1 2 3 5\n1 2 4 6\n1 3 4 7\n";
    /// let code = LdpcCode::from_alist(alist)?;
    /// assert_eq!((code.n(), code.k()), (7, 4));
    /// assert_eq!(code.rows()[1], [0, 1, 3, 5]);
    /// assert_eq!(LdpcCode::from_alist(&code.to_alist())?, code);
    /// # Ok::<(), com_core::Error>(())
    /// ```
    ///
    /// # References
    /// 1. D. J. C. MacKay, *Encyclopedia of Sparse Graph Codes*, "alist" format,
    ///    <http://www.inference.org.uk/mackay/codes/alist.html>.
    pub fn from_alist(text: &str) -> Result<LdpcCode> {
        let (n, rows) = parse_alist(text).map_err(|e| Error::Config(format!("alist: {e}")))?;
        LdpcCode::from_rows(n, rows)
    }

    /// [`LdpcCode::from_alist`] on the content of the file `path`.
    ///
    /// # Errors
    /// [`Error::Io`] if the file cannot be read, otherwise those of
    /// [`LdpcCode::from_alist`].
    pub fn from_alist_file(path: impl AsRef<std::path::Path>) -> Result<LdpcCode> {
        LdpcCode::from_alist(&std::fs::read_to_string(path)?)
    }

    /// The parity-check matrix in the `alist` format of [`LdpcCode::from_alist`], with
    /// the lists padded with zeros (the format MacKay's files use).
    pub fn to_alist(&self) -> String {
        let m = self.rows.len();
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        for (r, row) in self.rows.iter().enumerate() {
            for &v in row {
                cols[v].push(r);
            }
        }
        let max_col = cols.iter().map(Vec::len).max().unwrap_or(0);
        let max_row = self.rows.iter().map(Vec::len).max().unwrap_or(0);
        let line = |values: &mut dyn Iterator<Item = usize>| {
            let mut s = values.map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
            s.push('\n');
            s
        };
        let padded = |list: &[usize], len: usize| -> String {
            line(&mut list.iter().map(|v| v + 1).chain(std::iter::repeat_n(0, len - list.len())))
        };
        let mut out = format!("{} {m}\n{max_col} {max_row}\n", self.n);
        out.push_str(&line(&mut cols.iter().map(Vec::len)));
        out.push_str(&line(&mut self.rows.iter().map(Vec::len)));
        for c in &cols {
            out.push_str(&padded(c, max_col));
        }
        for r in &self.rows {
            out.push_str(&padded(r, max_row));
        }
        out
    }

    /// The positions of the information bits in the codeword, in the order of the
    /// information bits of [`BlockCode::encode_block`] (the code is systematic).
    pub fn info_positions(&self) -> &[usize] {
        &self.info
    }

    /// The parity checks: the variable nodes of every row of $H$.
    pub fn rows(&self) -> &[Vec<usize>] {
        &self.rows
    }

    /// Whether $H c = 0$.
    pub fn is_codeword(&self, c: &[u8]) -> bool {
        self.rows.iter().all(|r| r.iter().fold(0u8, |a, &v| a ^ c[v]) == 0)
    }

    /// The length of the shortest cycle through every variable node of the Tanner
    /// graph, minimised over the graph (the girth; 0 without cycle).
    pub fn girth(&self) -> usize {
        let mut var_checks: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        for (c, r) in self.rows.iter().enumerate() {
            for &v in r {
                var_checks[v].push(c);
            }
        }
        let mut best = usize::MAX;
        // BFS on the bipartite graph from every variable node
        let nodes = self.n + self.rows.len();
        for s in 0..self.n {
            let mut dist = vec![usize::MAX; nodes];
            let mut parent = vec![usize::MAX; nodes];
            dist[s] = 0;
            let mut q = VecDeque::from([s]);
            while let Some(x) = q.pop_front() {
                let nbrs: Vec<usize> = if x < self.n {
                    var_checks[x].iter().map(|c| self.n + c).collect()
                } else {
                    self.rows[x - self.n].clone()
                };
                for y in nbrs {
                    if dist[y] == usize::MAX {
                        dist[y] = dist[x] + 1;
                        parent[y] = x;
                        q.push_back(y);
                    } else if parent[x] != y {
                        best = best.min(dist[x] + dist[y] + 1);
                    }
                }
            }
        }
        if best == usize::MAX { 0 } else { best }
    }

    fn check_message(&self, incoming: &[f64], out: &mut [f64], scratch: &mut Vec<f64>) {
        match self.decoder.rule {
            CheckRule::SumProduct => {
                // the product of the other tanh(L/2) by prefix and suffix products, O(d)
                let d = incoming.len();
                scratch.clear();
                scratch.extend(incoming.iter().map(|l| (l / 2.0).tanh()));
                let mut prefix = 1.0;
                for (o, t) in out.iter_mut().zip(scratch.iter()) {
                    *o = prefix;
                    prefix *= t;
                }
                let mut suffix = 1.0;
                for i in (0..d).rev() {
                    let p = out[i] * suffix;
                    suffix *= scratch[i];
                    out[i] = 2.0 * p.clamp(-TANH_BOUND, TANH_BOUND).atanh();
                }
            }
            CheckRule::NormalizedMinSum(_) | CheckRule::OffsetMinSum(_) => {
                let sign: f64 = incoming.iter().map(|l| if *l < 0.0 { -1.0 } else { 1.0 }).product();
                let (mut m1, mut m2, mut i1) = (f64::INFINITY, f64::INFINITY, 0);
                for (i, l) in incoming.iter().enumerate() {
                    let a = l.abs();
                    if a < m1 {
                        m2 = m1;
                        m1 = a;
                        i1 = i;
                    } else if a < m2 {
                        m2 = a;
                    }
                }
                for (i, o) in out.iter_mut().enumerate() {
                    let mag = if i == i1 { m2 } else { m1 };
                    let s = sign * if incoming[i] < 0.0 { -1.0 } else { 1.0 };
                    let mag = match self.decoder.rule {
                        CheckRule::NormalizedMinSum(a) => a * mag,
                        CheckRule::OffsetMinSum(b) => (mag - b).max(0.0),
                        CheckRule::SumProduct => mag,
                    };
                    *o = s * mag;
                }
            }
        }
    }

    /// Decode a block of $n$ channel LLRs: the a posteriori LLRs of the coded bits and
    /// the number of iterations used.
    ///
    /// # Errors
    /// [`Error::Shape`] if `llr` does not have $n$ values.
    pub fn decode_llr(&self, llr: &[f64]) -> Result<(Vec<f64>, usize)> {
        expect_len(llr.len(), self.n, "LLRs")?;
        let mut post = llr.to_vec();
        let it = self.propagate(llr, &mut post);
        Ok((post, it))
    }

    /// Belief propagation from the channel LLRs `channel` ($n$ values), `post` holding
    /// them on entry and the a posteriori LLRs on exit: the iterations used.
    fn propagate(&self, channel: &[f64], post: &mut [f64]) -> usize {
        if self.hard_decisions_are_a_codeword(post) {
            return 0;
        }
        // the check-to-variable messages, row after row: those of check c at
        // start[c]..start[c + 1]; the flooding schedule writes the next ones apart
        let LdpcDecoder { layered, max_iterations, .. } = self.decoder;
        let start = &self.start;
        let edges = start[self.rows.len()];
        let mut c2v = vec![0.0; edges];
        let mut new_c2v = if layered { Vec::new() } else { vec![0.0; edges] };
        let (mut incoming, mut scratch) = (Vec::new(), Vec::new());
        for it in 1..=max_iterations {
            if layered {
                for (c, row) in self.rows.iter().enumerate() {
                    let msgs = &mut c2v[start[c]..start[c + 1]];
                    incoming.clear();
                    incoming.extend(row.iter().zip(msgs.iter()).map(|(&v, m)| post[v] - m));
                    self.check_message(&incoming, msgs, &mut scratch);
                    for ((&v, o), i) in row.iter().zip(msgs.iter()).zip(&incoming) {
                        post[v] = i + o;
                    }
                }
            } else {
                for (c, row) in self.rows.iter().enumerate() {
                    incoming.clear();
                    incoming.extend(row.iter().zip(&c2v[start[c]..start[c + 1]]).map(|(&v, m)| post[v] - m));
                    self.check_message(&incoming, &mut new_c2v[start[c]..start[c + 1]], &mut scratch);
                }
                post.copy_from_slice(channel);
                for (c, row) in self.rows.iter().enumerate() {
                    for (&v, m) in row.iter().zip(&new_c2v[start[c]..start[c + 1]]) {
                        post[v] += m;
                    }
                }
                std::mem::swap(&mut c2v, &mut new_c2v);
            }
            if self.hard_decisions_are_a_codeword(post) {
                return it;
            }
        }
        max_iterations
    }

    /// Whether the hard decisions of the LLRs `post` ($1$ for a negative LLR) satisfy
    /// every check, without forming them.
    fn hard_decisions_are_a_codeword(&self, post: &[f64]) -> bool {
        self.rows.iter().all(|r| r.iter().fold(0u8, |a, &v| a ^ u8::from(post[v] < 0.0)) == 0)
    }
}

/// The inverse of the core block $B$ (core × core: the core checks on the core parity bits
/// $k, \dots, k + \mathrm{core} - 1$) of [`LdpcCode::from_core_rows`], row by row: row $j$
/// lists the checks whose syndromes sum to the parity bit $k + j$.
///
/// It reduces $[I \mid B]$ over GF(2) ([`gf2::Matrix::reduce`], pivots searched from the
/// last column): when $B$ is invertible every pivot falls in its columns, the reduced
/// matrix is $[P B^{-1} \mid P]$ for a permutation $P$, and the row whose pivot is
/// column $j$ of $B$ holds row $j$ of $B^{-1}$.
fn core_inverse(rows: &[Vec<usize>], k: usize, core: usize) -> Result<Vec<Vec<usize>>> {
    let augmented: Vec<Vec<usize>> = rows
        .iter()
        .take(core)
        .enumerate()
        .map(|(r, row)| std::iter::once(r).chain(row.iter().filter(|&&v| v >= k).map(|&v| core + v - k)).collect())
        .collect();
    let mut a = gf2::Matrix::from_rows(2 * core, &augmented);
    let pivots = a.reduce();
    let rank = pivots.iter().filter(|&&c| c >= core).count();
    ensure(rank == core, || format!("the core block is singular (rank {rank} of {core})"))?;
    let mut solve = vec![Vec::new(); core];
    for (t, &c) in pivots.iter().enumerate() {
        solve[c - core] = (0..core).filter(|&j| a.get(t, j)).collect();
    }
    Ok(solve)
}

/// The check of lowest degree among `cands` (in the graph `check_vars`), the ties
/// broken by `rng`.
fn lowest_degree(cands: &[usize], check_vars: &[Vec<usize>], rng: &mut com_core::rng::Rng) -> usize {
    let d = cands.iter().map(|&c| check_vars[c].len()).min().unwrap_or(0);
    let best: Vec<usize> = cands.iter().copied().filter(|&c| check_vars[c].len() == d).collect();
    best[rng.below(best.len())]
}

/// The candidate checks of a new edge of the variable node `v` in progressive edge
/// growth (Hu, Eleftheriou & Arnold 2005): the checks the breadth-first expansion from
/// `v` never reaches, or, when it reaches them all, those of its last level not yet
/// linked to `v` (any check not linked to `v` if none).
fn farthest_checks(v: usize, check_vars: &[Vec<usize>], var_checks: &[Vec<usize>]) -> Vec<usize> {
    let m = check_vars.len();
    let mut reached = vec![false; m];
    let mut frontier: VecDeque<usize> = VecDeque::new();
    for &c in &var_checks[v] {
        reached[c] = true;
        frontier.push_back(c);
    }
    let mut last_new: Vec<usize> = var_checks[v].clone();
    loop {
        // the checks of the next level
        let mut new = Vec::new();
        let mut next = VecDeque::new();
        while let Some(c) = frontier.pop_front() {
            for &u in &check_vars[c] {
                for &c2 in &var_checks[u] {
                    if !reached[c2] {
                        reached[c2] = true;
                        new.push(c2);
                        next.push_back(c2);
                    }
                }
            }
        }
        if new.is_empty() || reached.iter().all(|r| *r) {
            let unreached: Vec<usize> = (0..m).filter(|&c| !reached[c]).collect();
            if !unreached.is_empty() {
                return unreached;
            }
            let last = if new.is_empty() { last_new } else { new };
            let cands: Vec<usize> = last.into_iter().filter(|c| !var_checks[v].contains(c)).collect();
            return if cands.is_empty() { (0..m).filter(|c| !var_checks[v].contains(c)).collect() } else { cands };
        }
        last_new = new;
        frontier = next;
    }
}

/// The number of columns and the rows (0-based column indices, in the order of the
/// file) of an `alist` text; the message of the error names what is wrong.
fn parse_alist(text: &str) -> std::result::Result<(usize, Vec<Vec<usize>>), String> {
    let mut it = alist_numbers(text)?.into_iter();
    let mut next = |what: &str| it.next().ok_or_else(|| format!("the file ends before {what}"));
    let (n, m) = (next("the number of columns")?, next("the number of rows")?);
    if n == 0 || m == 0 {
        return Err(format!("a matrix of {m} × {n}"));
    }
    let (max_col, max_row) = (next("the largest column degree")?, next("the largest row degree")?);
    let mut col_deg = Vec::with_capacity(n);
    for _ in 0..n {
        col_deg.push(next("the column degrees")?);
    }
    let mut row_deg = Vec::with_capacity(m);
    for _ in 0..m {
        row_deg.push(next("the row degrees")?);
    }
    check_degrees(&col_deg, max_col, "column")?;
    check_degrees(&row_deg, max_row, "row")?;
    let edges: usize = col_deg.iter().sum();
    if row_deg.iter().sum::<usize>() != edges {
        return Err(format!(
            "the column degrees count {edges} ones, the row degrees {}",
            row_deg.iter().sum::<usize>()
        ));
    }
    let rest: Vec<usize> = it.collect();
    // the lists are padded with zeros up to the largest degree, or not
    let padded = if rest.len() == n * max_col + m * max_row {
        true
    } else if rest.len() == 2 * edges {
        false
    } else {
        return Err(format!(
            "{} numbers after the degrees: {} with zero padding or {} without",
            rest.len(),
            n * max_col + m * max_row,
            2 * edges
        ));
    };
    let mut pos = 0;
    let cols = alist_lists(&rest, &mut pos, &col_deg, padded.then_some(max_col), m, "column")?;
    let rows = alist_lists(&rest, &mut pos, &row_deg, padded.then_some(max_row), n, "row")?;
    for (c, col) in cols.iter().enumerate() {
        for &r in col {
            if !rows[r].contains(&c) {
                return Err(format!("column {} lists row {} but row {} does not list it", c + 1, r + 1, r + 1));
            }
        }
    }
    Ok((n, rows))
}

/// The numbers of an `alist` text, the comments (after `#`) left out.
fn alist_numbers(text: &str) -> std::result::Result<Vec<usize>, String> {
    let mut numbers = Vec::new();
    for (l, line) in text.lines().enumerate() {
        let content = line.split('#').next().unwrap_or("");
        for token in content.split_whitespace() {
            numbers.push(token.parse::<usize>().map_err(|_| format!("line {}: {token:?} is not a number", l + 1))?);
        }
    }
    Ok(numbers)
}

/// The degrees of the columns (or rows) are at most, and reach, the declared maximum.
fn check_degrees(degrees: &[usize], max: usize, what: &str) -> std::result::Result<(), String> {
    if degrees.iter().any(|&d| d > max) || degrees.iter().max() != Some(&max) {
        return Err(format!("the {what} degrees do not have the maximum {max}"));
    }
    Ok(())
}

/// The lists of the columns (or rows) of degrees `degrees` read from `rest` at `pos`
/// (0-based indices below `bound`), each padded to `width` with zeros when given.
fn alist_lists(
    rest: &[usize],
    pos: &mut usize,
    degrees: &[usize],
    width: Option<usize>,
    bound: usize,
    what: &str,
) -> std::result::Result<Vec<Vec<usize>>, String> {
    let mut out = Vec::with_capacity(degrees.len());
    for (i, &d) in degrees.iter().enumerate() {
        let len = width.unwrap_or(d);
        let entries = &rest[*pos..*pos + len];
        *pos += len;
        let (ones, zeros) = entries.split_at(d);
        if zeros.iter().any(|&z| z != 0) {
            return Err(format!("{what} {}: more entries than its degree {d}", i + 1));
        }
        let mut list = Vec::with_capacity(d);
        for &e in ones {
            if e == 0 || e > bound {
                return Err(format!("{what} {}: the index {e} is outside 1 to {bound}", i + 1));
            }
            if list.contains(&(e - 1)) {
                return Err(format!("{what} {}: the index {e} is repeated", i + 1));
            }
            list.push(e - 1);
        }
        out.push(list);
    }
    Ok(out)
}

impl BlockCode for LdpcCode {
    fn k(&self) -> usize {
        self.info.len()
    }

    fn n(&self) -> usize {
        self.n
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        expect_len(info.len(), self.k(), "bits")?;
        let mut c = vec![0u8; self.n];
        for (&p, &b) in self.info.iter().zip(info) {
            c[p] = b & 1;
        }
        match &self.encoder {
            Encoder::Sums(parity) => {
                for (p, deps) in parity {
                    c[*p] = deps.iter().fold(0, |a, &v| a ^ c[v]);
                }
            }
            Encoder::Accumulate => {
                let k = self.info.len();
                let mut prev = 0u8;
                for (r, row) in self.rows.iter().enumerate() {
                    let s = row.iter().filter(|&&v| v < k).fold(0, |a, &v| a ^ c[v]);
                    prev ^= s;
                    c[k + r] = prev;
                }
            }
            Encoder::Core { core, solve } => {
                let k = self.info.len();
                let syndrome: Vec<u8> = self.rows[..*core]
                    .iter()
                    .map(|row| row.iter().filter(|&&v| v < k).fold(0, |a, &v| a ^ c[v]))
                    .collect();
                for (j, checks) in solve.iter().enumerate() {
                    c[k + j] = checks.iter().fold(0, |a, &r| a ^ syndrome[r]);
                }
                for (r, row) in self.rows.iter().enumerate().skip(*core) {
                    c[k + r] = row.iter().filter(|&&v| v != k + r).fold(0, |a, &v| a ^ c[v]);
                }
            }
        }
        // the encoders solve H c = 0 by construction (the tests check every family)
        debug_assert!(self.is_codeword(&c), "the encoder produced a non-codeword");
        Ok(c)
    }

    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        let (post, _) = self.decode_llr(llr)?;
        Ok(self.info.iter().map(|&p| u8::from(post[p] < 0.0)).collect())
    }
}

impl SoftDecode for LdpcCode {
    /// The a posteriori LLRs of the information bits of [`LdpcCode::decode_llr`]; the a
    /// priori LLRs add to the channel LLRs of the information positions.
    fn app_into(&self, llr: &[f64], apriori: Option<&[f64]>, out: &mut [f64]) -> Result<usize> {
        expect_len(llr.len(), self.n, "LLRs")?;
        expect_len(out.len(), self.k(), "a posteriori LLRs")?;
        let mut channel = llr.to_vec();
        if let Some(a) = apriori {
            expect_len(a.len(), self.k(), "a priori LLRs")?;
            for (&p, a) in self.info.iter().zip(a) {
                channel[p] += a;
            }
        }
        let mut post = channel.clone();
        let it = self.propagate(&channel, &mut post);
        for (o, &p) in out.iter_mut().zip(&self.info) {
            *o = post[p];
        }
        Ok(it)
    }
}

container_traits!(LdpcCode);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::testing::awgn_llr;

    /// The progressive edge growth as it was written in one function, eight levels deep.
    fn former_peg_rows(n: usize, k: usize, var_degree: usize, seed: u64) -> Vec<Vec<usize>> {
        let m = n - k;
        let mut rng = com_core::rng::Rng::new(seed);
        let mut check_vars: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut var_checks: Vec<Vec<usize>> = vec![Vec::new(); n];
        let lowest = |cands: &[usize], cv: &[Vec<usize>], rng: &mut com_core::rng::Rng| -> usize {
            let d = cands.iter().map(|&c| cv[c].len()).min().unwrap_or(0);
            let best: Vec<usize> = cands.iter().copied().filter(|&c| cv[c].len() == d).collect();
            best[rng.below(best.len())]
        };
        for v in 0..n {
            for e in 0..var_degree {
                let c = if e == 0 {
                    let all: Vec<usize> = (0..m).collect();
                    lowest(&all, &check_vars, &mut rng)
                } else {
                    // breadth-first expansion from v until no new check is reached
                    let mut reached = vec![false; m];
                    let mut frontier: VecDeque<usize> = VecDeque::new();
                    for &c in &var_checks[v] {
                        reached[c] = true;
                        frontier.push_back(c);
                    }
                    let mut last_new: Vec<usize> = var_checks[v].clone();
                    loop {
                        let mut new = Vec::new();
                        let mut next = VecDeque::new();
                        while let Some(c) = frontier.pop_front() {
                            for &u in &check_vars[c] {
                                for &c2 in &var_checks[u] {
                                    if !reached[c2] {
                                        reached[c2] = true;
                                        new.push(c2);
                                        next.push_back(c2);
                                    }
                                }
                            }
                        }
                        if new.is_empty() || reached.iter().all(|r| *r) {
                            let unreached: Vec<usize> = (0..m).filter(|&c| !reached[c]).collect();
                            break if unreached.is_empty() {
                                let cands: Vec<usize> = if new.is_empty() { last_new.clone() } else { new };
                                let cands: Vec<usize> =
                                    cands.into_iter().filter(|c| !var_checks[v].contains(c)).collect();
                                let cands = if cands.is_empty() {
                                    (0..m).filter(|c| !var_checks[v].contains(c)).collect()
                                } else {
                                    cands
                                };
                                lowest(&cands, &check_vars, &mut rng)
                            } else {
                                lowest(&unreached, &check_vars, &mut rng)
                            };
                        }
                        last_new = new;
                        frontier = next;
                    }
                };
                check_vars[c].push(v);
                var_checks[v].push(c);
            }
        }
        check_vars
    }

    #[test]
    fn the_peg_graphs_are_those_of_the_former_construction() {
        for (n, k, d, seed) in [(64, 32, 3, 1), (256, 128, 3, 7), (96, 48, 2, 3), (40, 10, 5, 2), (20, 18, 2, 1)] {
            let code = LdpcCode::peg(n, k, d, seed).unwrap();
            assert_eq!(code.rows(), former_peg_rows(n, k, d, seed), "({n}, {k}), degree {d}");
        }
    }

    /// The Gauss–Jordan elimination of `[B | I]` that `from_core_rows` wrote itself
    /// before it used `gf2::Matrix::reduce`.
    fn former_core_inverse(rows: &[Vec<usize>], k: usize, core: usize) -> Option<Vec<Vec<usize>>> {
        let words = core.div_ceil(64);
        let mut a: Vec<Vec<u64>> = vec![vec![0; 2 * words]; core];
        for (r, row) in rows.iter().take(core).enumerate() {
            for &v in row.iter().filter(|&&v| v >= k) {
                a[r][(v - k) / 64] ^= 1 << ((v - k) % 64);
            }
            a[r][words + r / 64] |= 1 << (r % 64);
        }
        for col in 0..core {
            let (w, b) = (col / 64, 1u64 << (col % 64));
            let pivot = (col..core).find(|&r| a[r][w] & b != 0)?;
            a.swap(col, pivot);
            let prow = a[col].clone();
            for (r, row) in a.iter_mut().enumerate() {
                if r != col && row[w] & b != 0 {
                    for (x, y) in row.iter_mut().zip(&prow) {
                        *x ^= y;
                    }
                }
            }
        }
        Some(a.iter().map(|row| (0..core).filter(|&c| row[words + c / 64] & (1 << (c % 64)) != 0).collect()).collect())
    }

    /// The inverse of the core block of both base graphs of 5G NR, at lifting sizes of
    /// every set, is the one of the former elimination; a singular block is an error.
    #[test]
    fn the_core_inverse_is_the_former_elimination() {
        use crate::nr::tables::{BG1, BG2, LIFTING_SETS};
        for (table, cols_k) in [(&BG1[..], 22), (&BG2[..], 10)] {
            for (ils, z) in [(0, 2), (1, 384), (2, 10), (3, 7), (4, 36), (5, 352), (6, 208), (7, 15)] {
                assert!(LIFTING_SETS[ils].contains(&z));
                let rows_n = table.iter().map(|e| e.0 as usize).max().unwrap() + 1;
                let mut rows: Vec<Vec<usize>> = vec![Vec::new(); rows_n * z];
                for &(i, j, v) in table {
                    let shift = v[ils] as usize % z;
                    for r in 0..z {
                        rows[i as usize * z + r].push(j as usize * z + (r + shift) % z);
                    }
                }
                let (k, core) = (cols_k * z, 4 * z);
                let new = core_inverse(&rows, k, core).unwrap();
                assert_eq!(Some(new), former_core_inverse(&rows, k, core), "z = {z}");
            }
        }
        // two equal core checks: singular for both
        let rows = vec![vec![0, 2], vec![0, 2], vec![1, 2, 3, 4]];
        assert!(core_inverse(&rows, 2, 2).is_err() && former_core_inverse(&rows, 2, 2).is_none());
        assert!(LdpcCode::from_core_rows(5, rows, 2, 2).is_err());
    }

    #[test]
    fn the_container_traits_fill_the_output_in_place() {
        use com_core::signal::{Bits, Llr};
        use com_core::traits::{Decode, Encode};
        let mut code = LdpcCode::peg(96, 48, 3, 1).unwrap();
        let rows: Vec<Vec<u8>> = (0..2).map(|s| (0..96).map(|i| u8::from((s + i) % 3 == 0)).collect()).collect();
        let info = Bits::from_streams(&rows).unwrap();
        let mut coded = Bits::zeros(2, 192);
        let p = coded.as_slice().as_ptr();
        code.encode_into(&info, &mut coded).unwrap();
        assert_eq!(coded.as_slice().as_ptr(), p);
        assert_eq!(coded.stream(1), code.encode_stream(info.stream(1)).unwrap());
        let llr = Llr::new(coded.as_array().mapv(|b| 4.0 * (1.0 - 2.0 * f64::from(b))));
        let mut decoded = Bits::zeros(2, 96);
        let p = decoded.as_slice().as_ptr();
        code.decode_into(&llr, &mut decoded).unwrap();
        assert_eq!((decoded.as_slice().as_ptr(), decoded.as_slice()), (p, info.as_slice()));
        assert!(code.encode_into(&Bits::zeros(1, 95), &mut coded).is_err());
    }
    use com_core::rng::Rng;
    use com_core::signal::hard_decisions;

    #[test]
    fn peg_builds_a_regular_code_without_four_cycles() {
        let code = LdpcCode::peg(504, 252, 3, 2).unwrap();
        assert_eq!(code.k(), 252);
        assert!(code.girth() >= 6, "girth {}", code.girth());
        let mut deg = vec![0; 504];
        for r in code.rows() {
            for &v in r {
                deg[v] += 1;
            }
        }
        assert!(deg.iter().all(|d| *d == 3));
        let mut r = Rng::new(3);
        for _ in 0..20 {
            let info: Vec<u8> = (0..252).map(|_| r.bit()).collect();
            assert!(code.is_codeword(&code.encode_block(&info).unwrap()));
        }
    }

    #[test]
    fn every_schedule_and_rule_decodes_and_layered_converges_faster() {
        let mut code = LdpcCode::peg(504, 252, 3, 4).unwrap();
        let mut r = Rng::new(5);
        let info: Vec<u8> = (0..252).map(|_| r.bit()).collect();
        let c = code.encode_block(&info).unwrap();
        // Eb/N0 = 2.5 dB at rate 1/2: σ² = 1/(2 R Eb/N0)
        let s2 = 1.0 / 10f64.powf(0.25);
        let y = awgn_llr(&c, s2, &mut r);
        let mut its = Vec::new();
        for layered in [false, true] {
            for rule in [CheckRule::SumProduct, CheckRule::NormalizedMinSum(0.8), CheckRule::OffsetMinSum(0.15)] {
                code.decoder.layered = layered;
                code.decoder.rule = rule;
                let (post, it) = code.decode_llr(&y).unwrap();
                assert!(code.is_codeword(&hard_decisions(&post)), "{rule:?} layered={layered}");
                assert_eq!(code.decode_block(&y).unwrap(), info);
                if rule == CheckRule::SumProduct {
                    its.push(it);
                }
            }
        }
        assert!(its[1] <= its[0], "flooding {} vs layered {} iterations", its[0], its[1]);
    }

    #[test]
    fn alist_files_round_trip_with_or_without_padding_and_comments() {
        let code = LdpcCode::peg(96, 48, 3, 7).unwrap();
        let text = code.to_alist();
        assert_eq!(LdpcCode::from_alist(&text).unwrap(), code);
        // the same matrix without the zero padding, with comments and blank lines
        let hamming_padded = "7 3\n3 4\n3 2 2 2 1 1 1\n4 4 4\n1 2 3\n1 2 0\n1 3 0\n2 3 0\n1 0 0\n2 0 0\n\
                              3 0 0\n1 2 3 5\n1 2 4 6\n1 3 4 7\n";
        let hamming_bare = "# the (7, 4) Hamming code\n7 3\n\n3 4 # degrees\n3 2 2 2 1 1 1\n4 4 4\n\
                            1 2 3\n1 2\n1 3\n2 3\n1\n2\n3\n1 2 3 5\n1 2 4 6\n1 3 4 7\n";
        let a = LdpcCode::from_alist(hamming_padded).unwrap();
        assert_eq!(LdpcCode::from_alist(hamming_bare).unwrap(), a);
        assert_eq!(a.rows(), &[vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]);
        let dir = std::env::temp_dir().join(format!("com-fec-alist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hamming.alist");
        std::fs::write(&path, hamming_bare).unwrap();
        assert_eq!(LdpcCode::from_alist_file(&path).unwrap(), a);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(matches!(LdpcCode::from_alist_file(dir.join("missing.alist")), Err(Error::Io(_))));
    }

    #[test]
    fn malformed_alist_files_are_rejected() {
        let good = "7 3\n3 4\n3 2 2 2 1 1 1\n4 4 4\n1 2 3\n1 2\n1 3\n2 3\n1\n2\n3\n1 2 3 5\n1 2 4 6\n1 3 4 7\n";
        assert!(LdpcCode::from_alist(good).is_ok());
        let cases = [
            ("", "ends before"),
            ("7 3\n3 4\n3 2 2 2 1 1 1\n", "ends before"),
            (&good.replace("1 3 4 7", "1 3 4 x"), "not a number"),
            (&good.replace("1 3 4 7", "1 3 4"), "numbers after the degrees"),
            (&good.replace("1 3 4 7", "1 3 4 9"), "outside"),
            (&good.replace("1 3 4 7", "1 3 3 7"), "repeated"),
            (&good.replace("1 3 4 7", "1 3 5 7"), "does not list"),
            (&good.replace("3 4\n3 2", "3 5\n3 2"), "maximum"),
        ];
        for (text, message) in cases {
            match LdpcCode::from_alist(text) {
                Err(Error::Config(e)) => assert!(e.contains(message), "{e:?} should name {message:?}"),
                other => panic!("{text:?} gave {other:?}"),
            }
        }
    }

    #[test]
    fn from_rows_handles_a_hamming_code() {
        // the (7, 4) Hamming code
        let code = LdpcCode::from_rows(7, vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]).unwrap();
        assert_eq!(code.k(), 4);
        assert!(LdpcCode::from_rows(3, vec![vec![0, 5]]).is_err());
    }
}

/// Checks against the published constructions (`references/`).
#[cfg(test)]
mod literature {
    use super::*;

    /// X.-Y. Hu, E. Eleftheriou and D. M. Arnold, "Regular and irregular progressive
    /// edge-growth Tanner graphs," IEEE Trans. Inf. Theory 51(1), 2005, Sec. VI
    /// (pp. 393-394): the regular (504, 252) PEG code with symbol degree 3 has girth 8
    /// (a few nodes 10), and its check degrees are 6 except for a few nodes of degrees 5
    /// and 7 (the paper: 8 of each).
    #[test]
    fn hu_2005_peg_504_252_has_girth_8() {
        for seed in [0, 1, 2] {
            let code = LdpcCode::peg(504, 252, 3, seed).unwrap();
            assert_eq!(code.girth(), 8, "seed {seed}");
            let mut degrees = std::collections::BTreeMap::new();
            for r in code.rows() {
                *degrees.entry(r.len()).or_insert(0usize) += 1;
            }
            assert_eq!(degrees.keys().copied().collect::<Vec<_>>(), [5, 6, 7], "seed {seed}: {degrees:?}");
            assert_eq!(degrees[&5], degrees[&7]);
            assert!(degrees[&5] <= 10 && degrees[&6] >= 230, "seed {seed}: {degrees:?}");
        }
    }
}

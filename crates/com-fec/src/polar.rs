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
//! # Examples
//! ```
//! use com_fec::block::{BlockCode, bpsk_llr};
//! use com_fec::crc::Crc;
//! use com_fec::polar::PolarCode;
//! let code = PolarCode::new(128, 64, 2.0)?.with_crc(Crc::crc6())?.with_list(8);
//! let info: Vec<u8> = (0..64).map(|i| (i % 3 == 0) as u8).collect();
//! let c = code.encode_block(&info)?;
//! let y: Vec<f64> = c.iter().map(|b| 1.0 - 2.0 * f64::from(*b)).collect();
//! assert_eq!(code.decode_block(&bpsk_llr(&y, 0.5))?, info);
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

/// The mean LLR of the synthetic channel `i` of a transform of size `n` whose
/// channels have the mean LLR `mu`.
fn reliability(i: usize, n: usize, mu: f64) -> f64 {
    if n == 1 {
        return mu;
    }
    // the outer stage of the transform acts first on the channel LLRs: the most
    // significant bit of `i` chooses its branch, the lower bits the next stages
    let m = if i < n / 2 { phi_inv(1.0 - (1.0 - phi(mu)).powi(2)) } else { 2.0 * mu };
    reliability(i % (n / 2), n / 2, m)
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

fn f(a: f64, b: f64) -> f64 {
    a.signum() * b.signum() * a.abs().min(b.abs())
}

/// The LLR of leaf `i` given the channel LLRs `ch` and the decisions `u[..i]`.
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
        let v: Vec<f64> = a.iter().zip(b).zip(&x).map(|((p, q), &xb)| q + if xb == 0 { *p } else { -p }).collect();
        leaf_llr(i - h, &v, &u[h..])
    }
}

/// A polar code of length $N$ with $K$ information bits.
#[derive(Clone, Debug, PartialEq)]
pub struct PolarCode {
    n: usize,
    k: usize,
    /// The positions that carry information (and CRC) bits, in increasing order.
    info_positions: Vec<usize>,
    /// The design $E_{b}/N_{0}$ of the construction (dB).
    design_ebn0_db: f64,
    /// The CRC appended to the information bits (CA-SCL).
    pub crc: Option<Crc>,
    /// List size $L$ (1: successive cancellation).
    pub list: usize,
}

impl PolarCode {
    /// The code of length `n` (a power of two) with `k` information bits, built by the
    /// Gaussian approximation at the design $E_{b}/N_{0}$ `design_ebn0_db`.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] unless $N$ is a power of two and $0 < K \le N$.
    pub fn new(n: usize, k: usize, design_ebn0_db: f64) -> Result<PolarCode> {
        ensure(n.is_power_of_two() && n >= 2, || format!("the length must be a power of two, got {n}"))?;
        ensure(k >= 1 && k <= n, || format!("0 < K ≤ N, got K = {k}"))?;
        let mut code = PolarCode { n, k, info_positions: Vec::new(), design_ebn0_db, crc: None, list: 1 };
        code.construct()?;
        Ok(code)
    }

    fn construct(&mut self) -> Result<()> {
        let extra = self.crc.as_ref().map_or(0, |c| c.width() as usize);
        let a = self.k + extra;
        ensure(a <= self.n, || format!("{} information and CRC bits exceed N = {}", a, self.n))?;
        // BPSK: mean channel LLR 2/σ² = 4 R Eb/N0 with the rate of the information bits
        let rate = self.k as f64 / self.n as f64;
        let mu = 4.0 * rate * db2lin(self.design_ebn0_db);
        let mut order: Vec<(f64, usize)> = (0..self.n).map(|i| (reliability(i, self.n, mu), i)).collect();
        order.sort_by(|x, y| y.0.total_cmp(&x.0).then(y.1.cmp(&x.1)));
        let mut pos: Vec<usize> = order.iter().take(a).map(|o| o.1).collect();
        pos.sort_unstable();
        self.info_positions = pos;
        Ok(())
    }

    /// The same code with a CRC appended to the information bits (the information set
    /// grows by its degree; CA-SCL keeps the path that passes it).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the information bits and the CRC exceed the block.
    pub fn with_crc(mut self, crc: Crc) -> Result<PolarCode> {
        self.crc = Some(crc);
        self.construct()?;
        Ok(self)
    }

    /// The same code decoded by SC list with `list` paths.
    #[must_use]
    pub fn with_list(mut self, list: usize) -> PolarCode {
        self.list = list.max(1);
        self
    }

    /// The positions carrying information and CRC bits.
    pub fn info_positions(&self) -> &[usize] {
        &self.info_positions
    }

    fn decode_paths(&self, llr: &[f64]) -> Vec<(Vec<u8>, f64)> {
        let mut info = vec![false; self.n];
        for &p in &self.info_positions {
            info[p] = true;
        }
        let mut paths: Vec<(Vec<u8>, f64)> = vec![(Vec::with_capacity(self.n), 0.0)];
        for i in 0..self.n {
            let mut next = Vec::with_capacity(paths.len() * 2);
            for (u, pm) in &paths {
                let mut full = u.clone();
                full.resize(self.n, 0);
                let lam = leaf_llr(i, llr, &full);
                let cost = |bit: u8| {
                    let s = if bit == 0 { lam } else { -lam };
                    pm + ln_1p_exp(-s)
                };
                if info[i] {
                    for bit in 0..2u8 {
                        let mut v = u.clone();
                        v.push(bit);
                        next.push((v, cost(bit)));
                    }
                } else {
                    let mut v = u.clone();
                    v.push(0);
                    next.push((v, cost(0)));
                }
            }
            next.sort_by(|a, b| a.1.total_cmp(&b.1));
            next.truncate(self.list.max(1));
            paths = next;
        }
        paths
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
        let scl = base.clone().with_crc(Crc::crc6()).unwrap().with_list(8);
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
}

//! Probabilistic amplitude shaping (PAS): a distribution matcher and a systematic
//! binary code combined so that the amplitudes are shaped and the signs, uniform, carry
//! the parity bits.
//!
//! # Model
//!
//! A real dimension carries the $2^{m}$-PAM symbols of
//! [`Constellation::pam`](com_mapping::constellation::Constellation::pam) scaled by the
//! half spacing $\Delta$, $x = \pm (2a + 1) \Delta$, labelled by a sign bit ($0$: $+$)
//! and the $m - 1$ bits of the binary reflected Gray code of the amplitude index $a$
//! ([`com_math::gray`]). For a codeword of $N$ bits of a systematic code of
//! $K \ge n_{c}(m - 1)$ information bits, $n_{c} = N/m$:
//!
//! 1. the [`Ccdm`] maps $k$ uniform bits to $n_{c}$ amplitudes of the target
//!    distribution $p_{A}$;
//! 2. the $n_{c}(m - 1)$ amplitude bits fill the first information positions of the
//!    code (label after label, or sorted by bit level with the bit mapper of
//!    [`Pas::with_bit_mapper`]), the $\gamma n_{c} = K - n_{c}(m - 1)$ other information
//!    positions carry uniform data bits;
//! 3. the $n_{c}$ positions that are not amplitude bits (the extra data bits and the
//!    $N - K$ parity bits, uniform) are the signs.
//!
//! The rate is $(k + \gamma n_{c})/n_{c}$ bits per real dimension. The receiver computes
//! the bit LLRs of every PAM symbol with the priors $p_{A}(a)/2$ (bit-metric decoding),
//!
//! $$
//! L(b) = \ln \sum_{x:\\, b = 0} p_{A}(a_{x}) e^{-(y - x)^{2} / (2 \sigma^{2})}
//!      - \ln \sum_{x:\\, b = 1} p_{A}(a_{x}) e^{-(y - x)^{2} / (2 \sigma^{2})},
//! $$
//!
//! with $\sigma^{2}$ the variance of the real noise, decodes the code, and dematches
//! the amplitudes. Over the AWGN channel PAS approaches the capacity within about 1 dB
//! with a code of rate $(m-1)/m$ only.
//!
//! The LLRs are those of [`com_mapping::demapper`] for the PAM whose point of label
//! $(s, g(a))$ is $(1 - 2s)(2a + 1)\Delta$, the priors $p_{A}(a)/2$ and the complex noise
//! variance $2\sigma^{2}$.
//!
//! # Examples
//! ```
//! use com_fec::ldpc::LdpcCode;
//! use com_systems::pas::Pas;
//! // 8-PAM per dimension (64-QAM), an LDPC code of rate 2/3, Maxwell-Boltzmann amplitudes
//! let code = LdpcCode::peg(768, 512, 3, 1)?;
//! let pa: Vec<f64> = (0..4).map(|a| (-0.08 * ((2 * a + 1) as f64).powi(2)).exp()).collect();
//! let pas = Pas::new(code, 3, &pa, 1.0)?;
//! let bits: Vec<u8> = (0..pas.info_bits()).map(|i| (i % 3 == 1) as u8).collect();
//! let x = pas.encode(&bits)?;
//! assert_eq!(x.len(), 256);
//! assert_eq!(pas.decode(&x, 0.01)?, bits);   // a clean channel
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. G. Böcherer, F. Steiner and P. Schulte, "Bandwidth efficient and rate-matched
//!    low-density parity-check coded modulation," *IEEE Trans. Commun.*, 63(12),
//!    4651–4665, 2015.
//! 2. P. Schulte and G. Böcherer, "Constant composition distribution matching," *IEEE
//!    Trans. Inf. Theory*, 62(1), 430–434, 2016.

use com_core::{Result, ensure, expect_len};
use com_fec::block::BlockCode;
use com_fec::ccdm::Ccdm;
use com_fec::ldpc::LdpcCode;
use com_mapping::constellation::Constellation;
use com_math::{C64, Jacobian, bits_to_index, gray, gray_inverse, index_to_bits};

/// A PAS transceiver of one real dimension (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Pas {
    code: LdpcCode,
    dm: Ccdm,
    m: usize,
    /// $2^{m}$-PAM scaled by $\Delta$: the level $i$ (in increasing order) carries the
    /// label $g(i)$.
    pam: Constellation,
    /// The same levels labelled as PAS: the point of label $s 2^{m-1} + g(a)$ is
    /// $(1 - 2s)(2a + 1)\Delta$, the sign bit first, then the label of the amplitude.
    labelled: Constellation,
    /// The prior of every point of `labelled`, $p_{A}(a)/2$.
    label_priors: Vec<f64>,
    /// The index, among the information bits of the code, of bit $j$ of the label of
    /// amplitude $i$ (entry $i(m-1) + j$).
    amplitude_info: Vec<usize>,
    amplitude_positions: Vec<usize>,
    extra_positions: Vec<usize>,
    sign_positions: Vec<usize>,
}

impl Pas {
    /// PAS with the systematic LDPC `code`, $2^{m}$-PAM symbols $\pm(2a+1)\Delta$ and the
    /// target amplitude distribution `amplitudes` ($2^{m-1}$ values, normalised inside).
    /// The $m - 1$ bits of every amplitude label take consecutive information positions
    /// ([`Pas::with_bit_mapper`] sorts them by level).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if the code length is not a multiple of $m$, if the
    /// code has fewer than $n_{c}(m-1)$ information bits, or for an invalid distribution.
    pub fn new(code: LdpcCode, m: usize, amplitudes: &[f64], delta: f64) -> Result<Pas> {
        ensure(m >= 2, || "PAS needs at least 4-PAM (m ≥ 2)".into())?;
        ensure(amplitudes.len() == 1 << (m - 1), || {
            format!("{}-PAM has {} amplitudes, got {} probabilities", 1 << m, 1 << (m - 1), amplitudes.len())
        })?;
        ensure(delta > 0.0, || "the half spacing must be positive".into())?;
        let n = code.n();
        ensure(n % m == 0, || format!("the code length {n} is not a multiple of {m} bits per symbol"))?;
        let n_c = n / m;
        let n_amp = n_c * (m - 1);
        ensure(code.k() >= n_amp, || {
            format!(
                "the code has {} information bits, the amplitudes need {n_amp} (rate at least {}/{m})",
                code.k(),
                m - 1
            )
        })?;
        let dm = Ccdm::from_distribution(amplitudes, n_c)?;
        let pam = Constellation::pam(1 << m)?.scaled(delta);
        // the level (1 - 2s)(2a + 1)Δ is the point of the PAM of index 2^{m-1} + a
        // (positive) or 2^{m-1} - 1 - a, and takes the label s 2^{m-1} + g(a)
        let half = 1 << (m - 1);
        let mut points = vec![C64::new(0.0, 0.0); 1 << m];
        let mut label_priors = vec![0.0; 1 << m];
        for (a, p) in dm.distribution().iter().enumerate() {
            for (s, i) in [(0, half + a), (1, half - 1 - a)] {
                points[s * half + gray(a)] = pam.points()[gray(i)];
                label_priors[s * half + gray(a)] = p / 2.0;
            }
        }
        let labelled = Constellation::from_points(points)?;
        let mut pas = Pas {
            code,
            dm,
            m,
            pam,
            labelled,
            label_priors,
            amplitude_info: Vec::new(),
            amplitude_positions: Vec::new(),
            extra_positions: Vec::new(),
            sign_positions: Vec::new(),
        };
        pas.place_bits(None);
        Ok(pas)
    }

    /// The same PAS with the bit mapper of Böcherer, Steiner & Schulte (2015, Sec.
    /// VII-B): the amplitude bits are sorted by bit level, the $n_{c}$ bits of one level
    /// in a row, and the levels follow in the order `levels`, a permutation of
    /// $2, \dots, m$ (level $1$ is the sign, level $\ell \ge 2$ bit $\ell - 2$ of the
    /// amplitude label, most significant first). The paper's bit mapper $\pi = (3, 2, 1)$
    /// of 8-ASK is `levels = [3, 2]`: the $n_{c}$ least significant amplitude bits take
    /// the first information positions (the variable nodes of the highest degrees of the
    /// DVB-S2 codes), then the most significant ones, then the signs.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] if `levels` is not a permutation of $2, \dots, m$.
    pub fn with_bit_mapper(mut self, levels: &[usize]) -> Result<Pas> {
        let m = self.m;
        let mut sorted = levels.to_vec();
        sorted.sort_unstable();
        ensure(sorted == (2..=m).collect::<Vec<_>>(), || {
            format!("the bit levels must be a permutation of 2, …, {m}, got {levels:?}")
        })?;
        self.place_bits(Some(levels));
        Ok(self)
    }

    /// The positions in the codeword of the amplitude bits (consecutive, or sorted by
    /// the bit levels `levels`), of the extra data bits and of the signs.
    fn place_bits(&mut self, levels: Option<&[usize]>) {
        let (n, m) = (self.code.n(), self.m);
        let n_c = n / m;
        let mb = m - 1;
        let n_amp = n_c * mb;
        let info = self.code.info_positions();
        self.amplitude_info = match levels {
            None => (0..n_amp).collect(),
            Some(levels) => {
                // the slot of every level in the codeword
                let mut slot = vec![0; mb];
                for (r, &l) in levels.iter().enumerate() {
                    slot[l - 2] = r;
                }
                (0..n_amp).map(|t| slot[t % mb] * n_c + t / mb).collect()
            }
        };
        self.amplitude_positions = self.amplitude_info.iter().map(|&t| info[t]).collect();
        self.extra_positions = info[n_amp..].to_vec();
        let mut is_amp = vec![false; n];
        for &p in &self.amplitude_positions {
            is_amp[p] = true;
        }
        self.sign_positions = (0..n).filter(|&p| !is_amp[p]).collect();
    }

    /// The level $(1 - 2s)(2a + 1)\Delta$ of the sign bit `s` and the amplitude `a`.
    fn level(&self, s: usize, a: usize) -> f64 {
        self.labelled.points()[(s << (self.m - 1)) | gray(a)].re
    }

    /// PAM symbols per codeword $n_{c}$.
    pub fn n_symbols(&self) -> usize {
        self.sign_positions.len()
    }

    /// Information bits per codeword: those of the matcher and the uniform data bits of
    /// the signs.
    pub fn info_bits(&self) -> usize {
        self.dm.k() + self.extra_positions.len()
    }

    /// Information bits per PAM symbol.
    pub fn rate(&self) -> f64 {
        self.info_bits() as f64 / self.n_symbols() as f64
    }

    /// The distribution matcher.
    pub fn matcher(&self) -> &Ccdm {
        &self.dm
    }

    /// The code.
    pub fn code(&self) -> &LdpcCode {
        &self.code
    }

    /// The $2^{m}$-PAM constellation scaled by $\Delta$ (its Gray labels are not the
    /// labels of PAS, see the module).
    pub fn constellation(&self) -> &Constellation {
        &self.pam
    }

    /// The probability of every PAM level `-(2^m - 1)Δ, …, (2^m - 1)Δ` in increasing
    /// order: the composition of the matcher, the signs uniform.
    pub fn level_distribution(&self) -> Vec<f64> {
        let pa = self.dm.distribution();
        pa.iter().rev().chain(&pa).map(|p| p / 2.0).collect()
    }

    /// The codeword of [`Pas::info_bits`] bits (the matcher's bits first): the Gray labels
    /// of the matched amplitudes and the extra data bits in the information positions of
    /// the code, and its parity.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of bits.
    pub fn encode_codeword(&self, bits: &[u8]) -> Result<Vec<u8>> {
        expect_len(bits.len(), self.info_bits(), "bits")?;
        let (dm_bits, extra) = bits.split_at(self.dm.k());
        let a = self.dm.match_bits(dm_bits)?;
        let mb = self.m - 1;
        let n_amp = a.len() * mb;
        let mut info = vec![0u8; self.code.k()];
        for (i, &ai) in a.iter().enumerate() {
            for (j, b) in index_to_bits(gray(ai), mb).enumerate() {
                info[self.amplitude_info[i * mb + j]] = b;
            }
        }
        info[n_amp..].copy_from_slice(extra);
        self.code.encode_block(&info)
    }

    /// The PAM symbols of one codeword from [`Pas::info_bits`] bits (the matcher's bits
    /// first): the levels of the signs and amplitude labels of [`Pas::encode_codeword`].
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of bits.
    pub fn encode(&self, bits: &[u8]) -> Result<Vec<f64>> {
        let c = self.encode_codeword(bits)?;
        let mb = self.m - 1;
        Ok(self
            .sign_positions
            .iter()
            .enumerate()
            .map(|(i, &sp)| {
                let label: Vec<u8> = (0..mb).map(|j| c[self.amplitude_positions[i * mb + j]]).collect();
                self.level(usize::from(c[sp]), gray_inverse(bits_to_index(&label)))
            })
            .collect())
    }

    /// The bit LLRs of the codeword (positive: bit 0) from the received PAM symbols `y`
    /// with real noise of variance `noise_var`, with the priors of the amplitudes.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of symbols.
    pub fn llr(&self, y: &[f64], noise_var: f64) -> Result<Vec<f64>> {
        self.llr_with(y, &vec![noise_var; y.len()])
    }

    /// [`Pas::llr`] with the noise variance of every symbol.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of symbols or variances.
    pub fn llr_with(&self, y: &[f64], noise_var: &[f64]) -> Result<Vec<f64>> {
        expect_len(y.len(), self.n_symbols(), "symbols")?;
        expect_len(noise_var.len(), self.n_symbols(), "noise variances")?;
        let mb = self.m - 1;
        let z: Vec<C64> = y.iter().map(|&v| C64::new(v, 0.0)).collect();
        // the complex noise variance of the demapper, kept positive and finite as the
        // demapper requires (a zero variance is the limit of a small one)
        let var: Vec<f64> = noise_var.iter().map(|&v| (2.0 * v.max(1e-300)).min(f64::MAX)).collect();
        let llr = self.labelled.llr_with_variances(&z, &var, Some(&self.label_priors), Jacobian::Exact)?;
        let mut out = vec![0.0; self.code.n()];
        for (i, l) in llr.chunks_exact(self.m).enumerate() {
            out[self.sign_positions[i]] = l[0];
            for (j, &lj) in l[1..].iter().enumerate() {
                out[self.amplitude_positions[i * mb + j]] = lj;
            }
        }
        Ok(out)
    }

    /// The information bits of one codeword from the received PAM symbols `y` (real
    /// noise of variance `noise_var`).
    ///
    /// # Errors
    /// As [`Pas::llr`].
    pub fn decode(&self, y: &[f64], noise_var: f64) -> Result<Vec<u8>> {
        self.decode_with(y, &vec![noise_var; y.len()])
    }

    /// [`Pas::decode`] with the noise variance of every symbol.
    ///
    /// # Errors
    /// As [`Pas::llr_with`].
    pub fn decode_with(&self, y: &[f64], noise_var: &[f64]) -> Result<Vec<u8>> {
        let llr = self.llr_with(y, noise_var)?;
        let info = self.code.decode_block(&llr)?;
        let mb = self.m - 1;
        let n_amp = self.n_symbols() * mb;
        let labels: Vec<u8> = self.amplitude_info.iter().map(|&t| info[t]).collect();
        let a: Vec<usize> = labels.chunks(mb).map(|w| gray_inverse(bits_to_index(w))).collect();
        let mut out = self.dm.dematch(&a)?;
        out.extend_from_slice(&info[n_amp..]);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    #[test]
    fn the_levels_are_those_of_the_pam_and_the_labels_of_pas() {
        let code = LdpcCode::peg(384, 256, 3, 1).unwrap();
        let pas = Pas::new(code, 3, &[0.4, 0.3, 0.2, 0.1], 0.5).unwrap();
        for s in 0..2 {
            for a in 0..4 {
                assert_eq!(pas.level(s, a), (1.0 - 2.0 * s as f64) * (2 * a + 1) as f64 * 0.5);
            }
        }
    }

    #[test]
    fn the_amplitudes_follow_the_composition_and_the_signs_are_uniform() {
        let code = LdpcCode::peg(1536, 1024, 3, 2).unwrap();
        let pa = [0.45, 0.3, 0.17, 0.08];
        let pas = Pas::new(code, 3, &pa, 1.0).unwrap();
        let mut rng = Rng::new(5);
        let mut count = [0usize; 8];
        for _ in 0..20 {
            let bits: Vec<u8> = (0..pas.info_bits()).map(|_| rng.bit()).collect();
            for x in pas.encode(&bits).unwrap() {
                count[((x + 7.0) / 2.0).round() as usize] += 1;
            }
        }
        let total = (20 * pas.n_symbols()) as f64;
        let level = pas.level_distribution();
        for (c, p) in count.iter().zip(&level) {
            assert!((*c as f64 / total - p).abs() < 0.01, "{count:?} {level:?}");
        }
        // the rate is that of the matcher plus the extra data bits of the signs
        assert!(pas.rate() > 1.5 && pas.rate() < 2.0, "{}", pas.rate());
    }

    #[test]
    fn the_bit_mapper_sorts_the_amplitude_bits_by_level() {
        // 16-PAM (m = 4), 96 symbols; levels 4, 2, 3: the least significant amplitude
        // bits first, then the most significant, then the middle ones
        let code = LdpcCode::peg(384, 300, 3, 4).unwrap();
        let pa = [0.3, 0.2, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03];
        let pas = Pas::new(code.clone(), 4, &pa, 1.0).unwrap().with_bit_mapper(&[4, 2, 3]).unwrap();
        let info = code.info_positions();
        for i in 0..96 {
            assert_eq!(pas.amplitude_positions[i * 3 + 2], info[i]);
            assert_eq!(pas.amplitude_positions[i * 3], info[96 + i]);
            assert_eq!(pas.amplitude_positions[i * 3 + 1], info[192 + i]);
        }
        let mut rng = Rng::new(9);
        let bits: Vec<u8> = (0..pas.info_bits()).map(|_| rng.bit()).collect();
        let x = pas.encode(&bits).unwrap();
        // the codeword read back from the levels carries the amplitudes of the symbols,
        // and is the codeword of the encoder
        let llr = pas.llr(&x, 1e-4).unwrap();
        for (i, xi) in x.iter().enumerate() {
            let a = ((xi.abs() - 1.0) / 2.0).round() as usize;
            let label: Vec<u8> = (0..3).map(|j| u8::from(llr[pas.amplitude_positions[i * 3 + j]] < 0.0)).collect();
            assert_eq!(gray_inverse(bits_to_index(&label)), a);
        }
        let c = pas.encode_codeword(&bits).unwrap();
        assert!(llr.iter().zip(&c).all(|(l, b)| (*l < 0.0) == (*b == 1)));
        assert_eq!(pas.decode(&x, 1e-4).unwrap(), bits);
        // a mapper that is not a permutation of the levels is refused
        let plain = Pas::new(code, 4, &pa, 1.0).unwrap();
        assert!(plain.clone().with_bit_mapper(&[4, 2, 2]).is_err());
        assert!(plain.with_bit_mapper(&[1, 2, 3]).is_err());
    }

    /// `encode` reads the amplitudes back from the labels of the codeword: the levels of
    /// the former encoder (the matched amplitudes and the signs of the codeword), bit for
    /// bit, with and without the bit mapper.
    #[test]
    fn the_symbols_are_the_levels_of_the_matched_amplitudes() {
        let code = LdpcCode::peg(384, 300, 3, 4).unwrap();
        let pa = [0.3, 0.2, 0.15, 0.12, 0.1, 0.06, 0.04, 0.03];
        let plain = Pas::new(code, 4, &pa, 0.7).unwrap();
        let mut rng = Rng::new(21);
        for pas in [plain.clone(), plain.with_bit_mapper(&[3, 4, 2]).unwrap()] {
            let bits: Vec<u8> = (0..pas.info_bits()).map(|_| rng.bit()).collect();
            let a = pas.dm.match_bits(&bits[..pas.dm.k()]).unwrap();
            let c = pas.encode_codeword(&bits).unwrap();
            let former: Vec<f64> =
                a.iter().zip(&pas.sign_positions).map(|(&ai, &sp)| pas.level(usize::from(c[sp]), ai)).collect();
            assert_eq!(pas.encode(&bits).unwrap(), former);
            assert!(pas.code().is_codeword(&c));
        }
    }

    #[test]
    fn the_llrs_are_those_of_the_formula_of_the_module() {
        // L(b) = ln Σ_{b = 0} p_A(a) e^{-(y - x)²/(2σ²)} - ln Σ_{b = 1} …, over the sign
        // and the Gray label of the amplitude; a zero variance stays a small one
        let code = LdpcCode::peg(384, 256, 3, 1).unwrap();
        let pas = Pas::new(code, 3, &[0.4, 0.3, 0.2, 0.1], 0.5).unwrap();
        // the priors are the composition of the matcher
        let pa = pas.matcher().distribution();
        let mut rng = Rng::new(2);
        let y: Vec<f64> = (0..pas.n_symbols()).map(|_| 2.0 * rng.gaussian()).collect();
        let var: Vec<f64> = (0..pas.n_symbols()).map(|i| [0.3, 0.05, 1.5][i % 3]).collect();
        let llr = pas.llr_with(&y, &var).unwrap();
        for (i, (&yi, &v)) in y.iter().zip(&var).enumerate() {
            let lse = |keep: &dyn Fn(usize, usize) -> bool| {
                let t = (0..8).filter(|&j| keep(j / 4, j % 4));
                com_math::log_sum_exp_iter(
                    t.map(|j| pa[j % 4].ln() - (yi - pas.level(j / 4, j % 4)).powi(2) / (2.0 * v)),
                )
            };
            let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-12 * b.abs().max(1.0), "{a} {b}");
            close(llr[pas.sign_positions[i]], lse(&|s, _| s == 0) - lse(&|s, _| s == 1));
            for j in 0..2 {
                let bit = |a: usize| (gray(a) >> (1 - j)) & 1;
                close(llr[pas.amplitude_positions[i * 2 + j]], lse(&|_, a| bit(a) == 0) - lse(&|_, a| bit(a) == 1));
            }
        }
        let x = pas.encode(&vec![0; pas.info_bits()]).unwrap();
        assert!(pas.llr(&x, 0.0).unwrap().iter().all(|l| l.is_finite()));
    }
}

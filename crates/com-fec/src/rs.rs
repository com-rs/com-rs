//! Reed–Solomon codes over $\mathrm{GF}(2^{m})$: systematic encoding, errors-and-erasures
//! decoding, shortening.
//!
//! # Model
//!
//! In the field $\mathrm{GF}(2^{m})$ of [`com_math::gf2m`] (primitive element $\alpha$, root
//! of the chosen primitive polynomial), the $t$-error-correcting Reed–Solomon code has
//! the generator polynomial
//!
//! $$
//! g(x) = \prod_{i=0}^{2t-1} \left(x - \beta^{b+i}\right), \qquad \beta = \alpha^{c},
//! $$
//!
//! with a first consecutive root $b$ (1 in the textbooks, 0 in DVB and QR codes, 112 in
//! CCSDS) and a root step $c$ coprime with $2^{m} - 1$ (1 everywhere but CCSDS, whose
//! roots are the powers of $\beta = \alpha^{11}$). The code of length $2^{m} - 1$ and
//! minimum distance $2t + 1$ is shortened to $n$ symbols by fixing its first
//! information symbols to zero; a codeword carries $k = n - 2t$ information symbols.
//! The symbols are `u16` values below $2^{m}$ ($m \le 16$); [`pack_symbols`] and
//! [`unpack_symbols`] convert them to and from bits ($m$ per symbol, most significant
//! first), so that the code also works on bits like the other codes of the crate
//! ([`ReedSolomon::encode`], [`ReedSolomon::decode`] and the
//! [`BlockCode`] interface on $km$ and $nm$ bits).
//!
//! The encoder is systematic: the $k$ message symbols (the highest degree first, in
//! transmission order), then the $2t$ coefficients of $x^{2t} u(x) \bmod g(x)$. The
//! decoder is the classical one:
//!
//! 1. the syndromes $S_{j} = r(\beta^{b+j})$, $j = 0, \dots, 2t - 1$;
//! 2. with $s$ erased positions of locators $Z_{l}$, the erasure locator
//!    $\Gamma(x) = \prod_{l} (1 - Z_{l} x)$ and the modified syndromes
//!    $T(x) = S(x)\Gamma(x) \bmod x^{2t}$ (Forney 1965);
//! 3. the error locator $\Lambda(x)$ by the Berlekamp–Massey algorithm on
//!    $T_{s}, \dots, T_{2t-1}$, of degree at most $\lfloor (2t - s)/2 \rfloor$;
//! 4. the roots of $\Psi(x) = \Lambda(x)\Gamma(x)$ by a Chien search over the $n$
//!    positions;
//! 5. the values by Forney's formula, with $\Omega(x) = S(x)\Psi(x) \bmod x^{2t}$,
//!
//! $$
//! Y_{l} = \frac{Z_{l}^{1-b}\, \Omega\!\left(Z_{l}^{-1}\right)}{\Psi'\!\left(Z_{l}^{-1}\right)}.
//! $$
//!
//! It corrects $e$ errors and $s$ erasures whenever $2e + s \le 2t$ and reports a
//! failure (`None`) when the locator has the wrong degree, fewer roots than its degree,
//! or gives a zero error value: the word then has more errors than the code corrects.
//! Beyond $t$ errors a bounded-distance decoder either fails or lands on another
//! codeword, so the closed forms [`word_error_probability`],
//! [`symbol_error_probability`] and [`bit_error_probability`] count every word with
//! more than $t$ symbol errors as lost.
//!
//! The dual-basis representation of the CCSDS symbols (the Berlekamp basis of CCSDS
//! 131.0-B, Sec. 4.2) is out of scope: the code is built with the CCSDS field, roots and
//! root step, but the symbols are polynomial-basis values.
//!
//! # Examples
//! ```
//! use com_fec::rs::ReedSolomon;
//! // the (255, 223) code over GF(256), t = 16
//! let code = ReedSolomon::new(8, 16)?;
//! assert_eq!((code.n(), code.k()), (255, 223));
//! let msg: Vec<u16> = (0..223).map(|i| (i * 7 % 256) as u16).collect();
//! let mut c = code.encode_symbols(&msg)?;
//! for i in [0, 5, 100, 254] {
//!     c[i] ^= 0x5A;                                     // four symbol errors
//! }
//! c[200] = 0;                                           // and one erasure
//! assert_eq!(code.decode_erasures(&c, &[200])?, (msg.clone(), Some(5)));
//! // on bits: 223 × 8 information bits in, 255 × 8 coded bits out
//! let bits = vec![1u8; 223 * 8];
//! let coded = code.encode(&bits)?;
//! assert_eq!(coded.len(), 255 * 8);
//! assert_eq!(code.decode(&coded)?, (bits, Some(0)));
//! # Ok::<(), com_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. I. S. Reed and G. Solomon, "Polynomial codes over certain finite fields,"
//!    *J. SIAM*, 8(2), 300–304, 1960.
//! 2. S. Lin and D. J. Costello, *Error Control Coding*, 2nd ed., Pearson, 2004,
//!    ch. 7 (Sec. 7.3: the generator polynomial, Sec. 7.4: the Berlekamp algorithm).
//! 3. G. D. Forney, "On decoding BCH codes," *IEEE Trans. Inf. Theory*, 11(4), 549–557,
//!    1965 (the error values and the erasures).
//! 4. J. L. Massey, "Shift-register synthesis and BCH decoding," *IEEE Trans. Inf.
//!    Theory*, 15(1), 122–127, 1969.
//! 5. S. B. Wicker, *Error Control Systems for Digital Communication and Storage*,
//!    Prentice Hall, 1995, ch. 9 (errors-and-erasures decoding).
//! 6. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008,
//!    Sec. 7.5.2 (the block and bit error probabilities of hard-decision decoding).
//! 7. B. Sklar, *Digital Communications: Fundamentals and Applications*, 2nd ed.,
//!    Prentice Hall, 2001, Sec. 8.1.2 (the bit error probability of Reed–Solomon codes).

use crate::algebraic::{berlekamp_massey, chien_search, eval_at_power};
use crate::block::{BlockCode, container_traits, expect_len};
use com_core::signal::hard_decisions;
use com_core::{Error, Result, ensure};
use com_math::gf2m::{Field, primitive_polynomial};
use com_math::{bits_to_index, index_to_bits};

/// A (shortened) Reed–Solomon code over $\mathrm{GF}(2^{m})$ (see the module).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReedSolomon {
    field: Field,
    t: usize,
    n: usize,
    first_root: usize,
    root_step: usize,
    /// $g(x)$, lowest degree first, monic of degree $2t$.
    generator: Vec<u16>,
}

impl ReedSolomon {
    /// The $t$-error-correcting code of length $2^{m} - 1$ over the field of the
    /// conventional primitive polynomial ([`primitive_polynomial`]), roots
    /// $\alpha, \dots, \alpha^{2t}$ ($b = 1$).
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for $m$ outside 2 to 16 or $2t \ge 2^{m} - 1$.
    pub fn new(m: u32, t: usize) -> Result<ReedSolomon> {
        let n = (1usize << m.min(16)) - 1;
        ReedSolomon::shortened(m, t, n)
    }

    /// The code of [`ReedSolomon::new`] shortened to `n` symbols.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for $m$ outside 2 to 16 or a length that leaves no
    /// information symbol or exceeds $2^{m} - 1$.
    pub fn shortened(m: u32, t: usize, n: usize) -> Result<ReedSolomon> {
        let poly = primitive_polynomial(m)
            .ok_or_else(|| Error::Config(format!("RS: the symbol size must be 2 to 16 bits, got {m}")))?;
        ReedSolomon::with_field(m, poly, 1, 1, t, n)
    }

    /// The code of `n` symbols correcting `t` errors over the field of the primitive
    /// polynomial `poly` (bit $i$ the coefficient of $x^{i}$), with the generator roots
    /// $\beta^{b}, \dots, \beta^{b+2t-1}$, $\beta = \alpha^{c}$: `first_root` is $b$
    /// and `root_step` is $c$.
    ///
    /// # Errors
    /// [`com_core::Error::Config`] for $m$ outside 2 to 16, a polynomial that is not
    /// primitive, a root step that is not coprime with $2^{m} - 1$, $t = 0$, or a length
    /// that leaves no information symbol or exceeds $2^{m} - 1$.
    pub fn with_field(
        m: u32,
        poly: u32,
        first_root: usize,
        root_step: usize,
        t: usize,
        n: usize,
    ) -> Result<ReedSolomon> {
        ensure((2..=16).contains(&m), || format!("RS: the symbol size must be 2 to 16 bits, got {m}"))?;
        let field = Field::new(m, poly).map_err(|e| e.context("RS"))?;
        let order = field.order();
        ensure(t >= 1, || "RS: t must be at least 1".into())?;
        ensure(root_step >= 1 && com_math::gcd(root_step, order) == 1, || {
            format!("RS: the root step {root_step} must be coprime with {order}")
        })?;
        ensure(2 * t < n && n <= order, || {
            format!("RS: the length {n} must exceed the {} parity symbols and not exceed {order}", 2 * t)
        })?;
        // g(x) = Π (x + β^{b+i}), β = α^c
        let mut generator = vec![1u32];
        for i in 0..2 * t {
            let r = field.pow(root_step * ((first_root + i) % order));
            let mut next = vec![0u32; generator.len() + 1];
            for (d, &a) in generator.iter().enumerate() {
                next[d + 1] ^= a;
                next[d] ^= field.mul(a, r);
            }
            generator = next;
        }
        let generator = generator.iter().map(|&a| a as u16).collect();
        Ok(ReedSolomon { field, t, n, first_root, root_step, generator })
    }

    /// Codeword length $n$ in symbols.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Message length $k = n - 2t$ in symbols.
    pub fn k(&self) -> usize {
        self.n - 2 * self.t
    }

    /// Errors corrected $t$.
    pub fn t(&self) -> usize {
        self.t
    }

    /// Bits per symbol $m$.
    pub fn m(&self) -> u32 {
        self.field.m()
    }

    /// The first consecutive root $b$.
    pub fn first_root(&self) -> usize {
        self.first_root
    }

    /// The root step $c$ ($\beta = \alpha^{c}$).
    pub fn root_step(&self) -> usize {
        self.root_step
    }

    /// The field.
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// The generator polynomial $g(x)$, lowest degree first (monic, degree $2t$).
    pub fn generator(&self) -> &[u16] {
        &self.generator
    }

    /// Checks that the symbols fit in $m$ bits.
    fn check_symbols(&self, symbols: &[u16]) -> Result<()> {
        let max = self.field.order() as u16;
        match symbols.iter().find(|&&s| s > max) {
            Some(s) => Err(Error::Config(format!("RS: the symbol {s} exceeds 2^{} - 1 = {max}", self.m()))),
            None => Ok(()),
        }
    }

    /// The codeword of $k$ message symbols: the message, then the $2t$ parity symbols
    /// (highest degree first).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of symbols,
    /// [`com_core::Error::Config`] for a symbol that does not fit in $m$ bits.
    pub fn encode_symbols(&self, msg: &[u16]) -> Result<Vec<u16>> {
        expect_len(msg.len(), self.k(), "symbols")?;
        self.check_symbols(msg)?;
        let f = &self.field;
        let r = 2 * self.t;
        // division by g(x): a shift register of r symbols, reg[j] the coefficient of x^j
        let mut reg = vec![0u32; r];
        for &u in msg {
            let fb = u32::from(u) ^ reg[r - 1];
            for j in (1..r).rev() {
                reg[j] = reg[j - 1] ^ f.mul(fb, u32::from(self.generator[j]));
            }
            reg[0] = f.mul(fb, u32::from(self.generator[0]));
        }
        let mut c = msg.to_vec();
        c.extend(reg.iter().rev().map(|&a| a as u16));
        Ok(c)
    }

    /// Decode $n$ received symbols: the message and the number of corrected symbols
    /// (`None` if the word has more errors than the code corrects and they were
    /// detected; the message symbols are then returned as received).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of symbols,
    /// [`com_core::Error::Config`] for a symbol that does not fit in $m$ bits.
    pub fn decode_symbols(&self, r: &[u16]) -> Result<(Vec<u16>, Option<usize>)> {
        self.decode_erasures(r, &[])
    }

    /// Decode $n$ received symbols with the positions `erasures` (0 to $n - 1$, in
    /// transmission order) known to be unreliable: the message and the number of
    /// corrected symbols (erasures included), `None` for a decoding failure (more than
    /// $2t - s$ errors in the $n - s$ other positions, when detected; the message
    /// symbols are then returned as received).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of symbols,
    /// [`com_core::Error::Config`] for a symbol that does not fit in $m$ bits, an
    /// erasure position outside the word, a repeated one, or more than $2t$ erasures.
    pub fn decode_erasures(&self, r: &[u16], erasures: &[usize]) -> Result<(Vec<u16>, Option<usize>)> {
        let n = self.n;
        expect_len(r.len(), n, "symbols")?;
        self.check_symbols(r)?;
        let two_t = 2 * self.t;
        let s = erasures.len();
        ensure(s <= two_t, || format!("RS: {s} erasures exceed the {two_t} the code can fill"))?;
        let mut erased = vec![false; n];
        for &p in erasures {
            ensure(p < n, || format!("RS: the erasure position {p} is outside the word of {n} symbols"))?;
            ensure(!erased[p], || format!("RS: the erasure position {p} is repeated"))?;
            erased[p] = true;
        }
        let f = &self.field;
        let order = f.order();
        let k = self.k();
        let (b, c) = (self.first_root, self.root_step);
        // the locator of position i is Z_i = β^{e_i}, e_i = n - 1 - i
        let locator_exp = |i: usize| c * (n - 1 - i) % order;
        let mut word: Vec<u32> = r.iter().map(|&v| u32::from(v)).collect();
        for &p in erasures {
            word[p] = 0;
        }
        // syndromes S_j = r(β^{b+j})
        let mut syn = vec![0u32; two_t];
        for (i, &v) in word.iter().enumerate() {
            if v != 0 {
                let e = locator_exp(i);
                for (j, sj) in syn.iter_mut().enumerate() {
                    *sj ^= f.mul(v, f.pow(e * ((b + j) % order)));
                }
            }
        }
        let msg_as_received = || r[..k].to_vec();
        if syn.iter().all(|&v| v == 0) && s == 0 {
            return Ok((msg_as_received(), Some(0)));
        }
        // erasure locator Γ(x) = Π (1 + Z_l x), and T = S Γ mod x^{2t}
        let mut gamma = vec![1u32];
        for &p in erasures {
            let z = f.pow(locator_exp(p));
            let mut next = vec![0u32; gamma.len() + 1];
            for (d, &a) in gamma.iter().enumerate() {
                next[d] ^= a;
                next[d + 1] ^= f.mul(a, z);
            }
            gamma = next;
        }
        let modified = poly_mul_mod(f, &syn, &gamma, two_t);
        // Berlekamp–Massey on T_s … T_{2t-1}
        // a valid locator: its length L fits the syndromes (2L ≤ 2t - s) and is its degree
        let (lambda, l) = berlekamp_massey(f, &modified[s..]);
        if 2 * l > two_t - s || lambda.len() - 1 != l {
            return Ok((msg_as_received(), None));
        }
        // Ψ = Λ Γ, its roots by a Chien search over the n positions
        let psi = poly_mul_mod(f, &lambda, &gamma, lambda.len() + gamma.len() - 1);
        let degree = psi.len() - 1;
        let roots = chien_search(f, &psi, n, locator_exp);
        if roots.len() != degree {
            return Ok((msg_as_received(), None));
        }
        // Forney: Y = Z^{1-b} Ω(Z⁻¹) / Ψ'(Z⁻¹), Ω = S Ψ mod x^{2t}
        let omega = poly_mul_mod(f, &syn, &psi, two_t);
        let exp_1mb = (order + 1 - b % order) % order;
        let mut corrected = 0;
        for &i in &roots {
            let ze = locator_exp(i);
            let inv = (order - ze) % order;
            let num = eval_at_power(f, &omega, inv);
            let den = psi
                .iter()
                .enumerate()
                .skip(1)
                .step_by(2)
                .fold(0u32, |acc, (d, &a)| acc ^ f.mul(a, f.pow((d - 1) * inv)));
            if den == 0 {
                return Ok((msg_as_received(), None));
            }
            let y = f.mul(f.pow(ze * exp_1mb), f.div(num, den));
            if y == 0 && !erased[i] {
                return Ok((msg_as_received(), None));
            }
            word[i] ^= y;
            corrected += usize::from(word[i] != u32::from(r[i]));
        }
        Ok((word[..k].iter().map(|&v| v as u16).collect(), Some(corrected)))
    }

    /// The codeword of $km$ message bits as $nm$ bits ($m$ per symbol, most significant
    /// first): the message bits, then the parity.
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of bits.
    pub fn encode(&self, bits: &[u8]) -> Result<Vec<u8>> {
        let m = self.m();
        expect_len(bits.len(), self.k() * m as usize, "bits")?;
        Ok(unpack_symbols(&self.encode_symbols(&pack_symbols(bits, m)?)?, m))
    }

    /// Decode $nm$ hard bits: the $km$ message bits and the number of corrected
    /// symbols (`None` for a detected decoding failure, the message bits then returned
    /// as received).
    ///
    /// # Errors
    /// [`com_core::Error::Shape`] for another number of bits.
    pub fn decode(&self, bits: &[u8]) -> Result<(Vec<u8>, Option<usize>)> {
        let m = self.m();
        expect_len(bits.len(), self.n * m as usize, "bits")?;
        let (msg, corrected) = self.decode_symbols(&pack_symbols(bits, m)?)?;
        Ok((unpack_symbols(&msg, m), corrected))
    }
}

impl BlockCode for ReedSolomon {
    fn k(&self) -> usize {
        ReedSolomon::k(self) * self.m() as usize
    }

    fn n(&self) -> usize {
        self.n * self.m() as usize
    }

    fn encode_block(&self, info: &[u8]) -> Result<Vec<u8>> {
        self.encode(info)
    }

    /// Hard decisions of the LLRs, then [`ReedSolomon::decode`]; a decoding failure
    /// returns the message bits as received.
    fn decode_block(&self, llr: &[f64]) -> Result<Vec<u8>> {
        expect_len(llr.len(), BlockCode::n(self), "LLRs")?;
        Ok(self.decode(&hard_decisions(llr))?.0)
    }
}

container_traits!(ReedSolomon);

/// Pack bits into symbols of `m` bits, the most significant first.
///
/// # Errors
/// [`com_core::Error::Shape`] if the number of bits is not a multiple of `m`,
/// [`com_core::Error::Config`] for `m` outside 1 to 16.
pub fn pack_symbols(bits: &[u8], m: u32) -> Result<Vec<u16>> {
    ensure((1..=16).contains(&m), || format!("the symbol size must be 1 to 16 bits, got {m}"))?;
    let m = m as usize;
    if bits.len() % m != 0 {
        return Err(Error::Shape(format!("{} bits is not a multiple of the symbol size {m}", bits.len())));
    }
    Ok(bits.chunks(m).map(|c| bits_to_index(c) as u16).collect())
}

/// Unpack symbols of `m` bits (1 to 16) into bits, the most significant first; the
/// bits of a symbol above $2^{m}$ are dropped.
pub fn unpack_symbols(symbols: &[u16], m: u32) -> Vec<u8> {
    let m = m.clamp(1, 16) as usize;
    symbols.iter().flat_map(|&s| index_to_bits(usize::from(s), m)).collect()
}

/// The product of two polynomials over the field, truncated to `len` coefficients
/// (lowest degree first).
fn poly_mul_mod(f: &Field, a: &[u32], b: &[u32], len: usize) -> Vec<u32> {
    let mut out = vec![0u32; len];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate().take(len.saturating_sub(i)) {
            out[i + j] ^= f.mul(x, y);
        }
    }
    out
}

/// $\ln \binom{n}{i} + i \ln p + (n - i) \ln (1 - p)$ for $i = 0, \dots, n$: the
/// logarithms of the binomial probabilities of `i` symbol errors among `n`.
fn log_binomial_terms(n: usize, p: f64) -> Vec<f64> {
    let (lp, lq) = (p.ln(), (1.0 - p).ln());
    let mut terms = Vec::with_capacity(n + 1);
    let mut log_c = 0.0;
    for i in 0..=n {
        if i > 0 {
            log_c += ((n + 1 - i) as f64).ln() - (i as f64).ln();
        }
        terms.push(log_c + i as f64 * lp + (n - i) as f64 * lq);
    }
    terms
}

/// The word error probability of a bounded-distance decoder of a $t$-error-correcting
/// code of $n$ symbols over a memoryless channel of symbol error probability $p_{s}$
/// (Proakis & Salehi, Sec. 7.5.2),
///
/// $$
/// P_{w} = \sum_{i=t+1}^{n} \binom{n}{i} p_{s}^{i} (1 - p_{s})^{n-i}.
/// $$
pub fn word_error_probability(n: usize, t: usize, p_s: f64) -> f64 {
    if !(0.0..=1.0).contains(&p_s) || t >= n {
        return 0.0;
    }
    if p_s == 0.0 {
        return 0.0;
    }
    if p_s == 1.0 {
        return 1.0;
    }
    log_binomial_terms(n, p_s).iter().skip(t + 1).map(|l| l.exp()).sum()
}

/// The symbol error probability after decoding, the usual approximation that a word of
/// $i > t$ errors keeps its $i$ wrong symbols (Proakis & Salehi, Sec. 7.5.2; Sklar,
/// Sec. 8.1.2),
///
/// $$
/// P_{s}^{\mathrm{out}} \approx \frac{1}{n} \sum_{i=t+1}^{n} i \binom{n}{i} p_{s}^{i} (1 - p_{s})^{n-i}.
/// $$
pub fn symbol_error_probability(n: usize, t: usize, p_s: f64) -> f64 {
    if !(0.0..=1.0).contains(&p_s) || t >= n || p_s == 0.0 {
        return 0.0;
    }
    if p_s == 1.0 {
        return 1.0;
    }
    log_binomial_terms(n, p_s).iter().enumerate().skip(t + 1).map(|(i, l)| i as f64 * l.exp()).sum::<f64>() / n as f64
}

/// The bit error probability after decoding of a code over $\mathrm{GF}(2^{m})$: a wrong
/// symbol carries on average $2^{m-1}/(2^{m} - 1)$ wrong bits per bit (Sklar,
/// Sec. 8.1.2),
///
/// $$
/// P_{b} \approx \frac{2^{m-1}}{2^{m} - 1}\, P_{s}^{\mathrm{out}}.
/// $$
pub fn bit_error_probability(n: usize, t: usize, m: u32, p_s: f64) -> f64 {
    let q = 2f64.powi(m as i32);
    q / 2.0 / (q - 1.0) * symbol_error_probability(n, t, p_s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use com_core::rng::Rng;

    /// A random word with `errors` distinct symbol errors outside `avoid`.
    pub(super) fn corrupt(c: &mut [u16], errors: usize, m: u32, avoid: &[usize], rng: &mut Rng) -> Vec<usize> {
        let mut positions = Vec::with_capacity(errors);
        while positions.len() < errors {
            let p = rng.below(c.len());
            if !positions.contains(&p) && !avoid.contains(&p) {
                positions.push(p);
                let e = 1 + rng.below((1 << m) - 1) as u16;
                c[p] ^= e;
            }
        }
        positions
    }

    #[test]
    fn errors_and_erasures_are_corrected_up_to_2e_plus_s_equal_2t() {
        let code = ReedSolomon::shortened(8, 4, 60).unwrap();
        assert_eq!(code.k(), 52);
        let mut rng = Rng::new(11);
        for s in 0..=8 {
            for e in 0..=(8 - s) / 2 {
                let msg: Vec<u16> = (0..52).map(|_| rng.below(256) as u16).collect();
                let mut c = code.encode_symbols(&msg).unwrap();
                let erased = corrupt(&mut c, s, 8, &[], &mut rng);
                corrupt(&mut c, e, 8, &erased, &mut rng);
                let (m, n) = code.decode_erasures(&c, &erased).unwrap();
                assert_eq!(m, msg, "s = {s}, e = {e}");
                assert_eq!(n, Some(s + e), "s = {s}, e = {e}");
            }
        }
    }

    #[test]
    fn invalid_inputs_are_errors_not_panics() {
        assert!(ReedSolomon::new(1, 1).is_err());
        assert!(ReedSolomon::new(17, 1).is_err());
        assert!(ReedSolomon::new(3, 0).is_err());
        assert!(ReedSolomon::shortened(3, 2, 4).is_err());
        assert!(ReedSolomon::shortened(3, 1, 8).is_err());
        assert!(ReedSolomon::with_field(4, 0b1_1111, 1, 1, 1, 15).is_err());
        assert!(ReedSolomon::with_field(4, 0b1_0011, 1, 3, 1, 15).is_err());
        let code = ReedSolomon::new(4, 2).unwrap();
        assert!(code.encode_symbols(&[0; 10]).is_err());
        assert!(code.encode_symbols(&[16; 11]).is_err());
        assert!(code.decode_symbols(&[0; 14]).is_err());
        assert!(code.decode_erasures(&[0; 15], &[15]).is_err());
        assert!(code.decode_erasures(&[0; 15], &[1, 1]).is_err());
        assert!(code.decode_erasures(&[0; 15], &[0, 1, 2, 3, 4]).is_err());
        assert!(code.encode(&[0; 43]).is_err());
        assert!(pack_symbols(&[1, 0, 1], 2).is_err());
        assert_eq!(unpack_symbols(&[0b1011], 3), vec![0, 1, 1]);
        assert_eq!(pack_symbols(&[1, 0, 1, 1, 1, 0], 3).unwrap(), vec![5, 6]);
    }

    #[test]
    fn the_block_code_interface_counts_bits() {
        let code = ReedSolomon::shortened(6, 3, 30).unwrap();
        assert_eq!((BlockCode::k(&code), BlockCode::n(&code)), (24 * 6, 30 * 6));
        let mut rng = Rng::new(3);
        let info: Vec<u8> = (0..144).map(|_| rng.bit()).collect();
        let c = code.encode_block(&info).unwrap();
        let mut llr: Vec<f64> = c.iter().map(|&b| if b == 1 { -2.0 } else { 2.0 }).collect();
        // three symbols hit: 2 bits in symbol 0, 1 in symbol 7, 6 in symbol 29
        for i in [0, 1, 42, 174, 175, 176, 177, 178, 179] {
            llr[i] = -llr[i];
        }
        assert_eq!(code.decode_block(&llr).unwrap(), info);
    }

    #[test]
    fn the_closed_forms_have_the_right_limits() {
        assert_eq!(word_error_probability(255, 16, 0.0), 0.0);
        assert_eq!(word_error_probability(255, 16, 1.0), 1.0);
        // t = 0: every word with an error is lost, 1 - (1 - p)^n
        let p = 0.01;
        assert!((word_error_probability(7, 0, p) - (1.0 - (1.0 - p).powi(7))).abs() < 1e-15);
        // the output symbol error rate is below the input one and above zero
        let out = symbol_error_probability(255, 16, 0.05);
        assert!(out > 0.0 && out < 0.05);
        assert!((bit_error_probability(255, 16, 8, 0.05) / out - 128.0 / 255.0).abs() < 1e-12);
    }
}

#[cfg(test)]
mod literature {
    use super::*;
    use com_core::rng::Rng;

    /// S. Lin and D. J. Costello, *Error Control Coding*, 2nd ed., 2004, Sec. 7.3: the
    /// triple-error-correcting (15, 9) code over GF(2^4) built with p(x) = 1 + x + x⁴
    /// (Table 2.8) has g(x) = (x + α)(x + α²)⋯(x + α⁶) = α⁶ + α⁹x + α⁶x² + α⁴x³ +
    /// α¹⁴x⁴ + α¹⁰x⁵ + x⁶.
    #[test]
    fn lin_costello_sec_7_3_generator_of_the_15_9_code() {
        let code = ReedSolomon::with_field(4, 0b1_0011, 1, 1, 3, 15).unwrap();
        let f = code.field();
        let expected: Vec<u16> = [6, 9, 6, 4, 14, 10].iter().map(|&e| f.pow(e) as u16).chain([1]).collect();
        assert_eq!(code.generator(), expected);
    }

    /// S. Lin and D. J. Costello, 2nd ed., Sec. 7.4 (the Berlekamp algorithm for the
    /// (15, 9) code): the received polynomial r(x) = α⁷x³ + α³x⁶ + α⁴x¹² (the all-zero
    /// codeword with three errors) is decoded to the all-zero word; its syndromes are
    /// S₁ = α¹², S₂ = 1, S₃ = α¹⁴, S₄ = α¹⁰, S₅ = 0, S₆ = α¹².
    #[test]
    fn lin_costello_sec_7_4_three_errors_of_the_15_9_code() {
        let code = ReedSolomon::with_field(4, 0b1_0011, 1, 1, 3, 15).unwrap();
        let f = code.field();
        let mut r = vec![0u16; 15];
        // position i carries x^{14 - i}
        r[14 - 3] = f.pow(7) as u16;
        r[14 - 6] = f.pow(3) as u16;
        r[14 - 12] = f.pow(4) as u16;
        let syndromes: Vec<u32> = (1..=6)
            .map(|j| r.iter().enumerate().fold(0u32, |acc, (i, &v)| acc ^ f.mul(u32::from(v), f.pow(j * (14 - i)))))
            .collect();
        assert_eq!(syndromes, [f.pow(12), 1, f.pow(14), f.pow(10), 0, f.pow(12)]);
        assert_eq!(code.decode_symbols(&r).unwrap(), (vec![0; 9], Some(3)));
    }

    /// ETSI EN 300 744 V1.6.1, Sec. 4.3.2 (DVB-T; also EN 300 421 Sec. 4.4 for DVB-S):
    /// the RS(204, 188, t = 8) shortened code with g(x) = (x + λ⁰)(x + λ¹)⋯(x + λ¹⁵),
    /// λ = `02_HEX`, over the field of p(x) = x⁸ + x⁴ + x³ + x² + 1. The standard prints
    /// g(x) in factored form; its expansion is the degree-16 generator polynomial of
    /// ISO/IEC 18004:2015 (QR codes, same field, same roots), Table A.1:
    /// x¹⁶ + α¹²⁰x¹⁵ + α¹⁰⁴x¹⁴ + α¹⁰⁷x¹³ + α¹⁰⁹x¹² + α¹⁰²x¹¹ + α¹⁶¹x¹⁰ + α⁷⁶x⁹ + α³x⁸ +
    /// α⁹¹x⁷ + α¹⁹¹x⁶ + α¹⁴⁷x⁵ + α¹⁶⁹x⁴ + α¹⁸²x³ + α¹⁹⁴x² + α²²⁵x + α¹²⁰.
    #[test]
    fn etsi_en_300_744_sec_4_3_2_generator_of_rs_204_188() {
        let code = ReedSolomon::with_field(8, 0x11D, 0, 1, 8, 204).unwrap();
        assert_eq!((code.n(), code.k()), (204, 188));
        let f = code.field();
        let exponents = [120, 104, 107, 109, 102, 161, 76, 3, 91, 191, 147, 169, 182, 194, 225, 120];
        let mut expected: Vec<u16> = exponents.iter().rev().map(|&e| f.pow(e) as u16).collect();
        expected.push(1);
        assert_eq!(code.generator(), expected);
        for j in 0..16 {
            let root = f.pow(j);
            let v = code
                .generator()
                .iter()
                .enumerate()
                .fold(0u32, |acc, (d, &a)| acc ^ f.mul(u32::from(a), f.pow(d * f.log(root))));
            assert_eq!(v, 0, "λ^{j} is a root");
        }
        // the shortened code is the (255, 239) code with 51 leading zero symbols
        let full = ReedSolomon::with_field(8, 0x11D, 0, 1, 8, 255).unwrap();
        let msg: Vec<u16> = (0..188).map(|i| (i * 13 % 256) as u16).collect();
        let mut padded = vec![0u16; 51];
        padded.extend(&msg);
        assert_eq!(full.encode_symbols(&padded).unwrap()[51..], code.encode_symbols(&msg).unwrap()[..]);
    }

    /// CCSDS 131.0-B-5, *TM Synchronization and Channel Coding*, Sec. 4.2: the (255, 223)
    /// code over the field of F(x) = x⁸ + x⁷ + x² + x + 1 with g(x) = Π_{j=112}^{143}
    /// (x − α^{11j}). The root set is symmetric (112 + 143 = 255), so g(x) is
    /// self-reciprocal: `g_i = g_{32-i}` (the standard notes this symmetry). The dual
    /// (Berlekamp) basis of the symbols is not represented.
    #[test]
    fn ccsds_131_0_b_sec_4_2_roots_of_rs_255_223() {
        let code = ReedSolomon::with_field(8, 0x187, 112, 11, 16, 255).unwrap();
        let f = code.field();
        let g = code.generator();
        assert_eq!(g.len(), 33);
        for j in 112..=143 {
            let root_exp = 11 * j;
            let v = g.iter().enumerate().fold(0u32, |acc, (d, &a)| acc ^ f.mul(u32::from(a), f.pow(d * root_exp)));
            assert_eq!(v, 0, "α^(11·{j}) is a root");
        }
        for i in 0..=32 {
            assert_eq!(g[i], g[32 - i], "g is self-reciprocal");
        }
        let mut rng = Rng::new(5);
        let msg: Vec<u16> = (0..223).map(|_| rng.below(256) as u16).collect();
        let mut c = code.encode_symbols(&msg).unwrap();
        super::tests::corrupt(&mut c, 16, 8, &[], &mut rng);
        assert_eq!(code.decode_symbols(&c).unwrap(), (msg, Some(16)));
    }

    /// The (255, 223) code corrects every pattern of 16 symbol errors; with 17 errors a
    /// bounded-distance decoder fails or miscorrects, and never returns the message:
    /// 300 words of each, seeded.
    #[test]
    fn rs_255_223_corrects_16_errors_and_not_17() {
        let code = ReedSolomon::new(8, 16).unwrap();
        let mut rng = Rng::new(255);
        let (mut failures, mut miscorrections) = (0, 0);
        for trial in 0..300 {
            let msg: Vec<u16> = (0..223).map(|_| rng.below(256) as u16).collect();
            let mut c = code.encode_symbols(&msg).unwrap();
            super::tests::corrupt(&mut c, 16, 8, &[], &mut rng);
            assert_eq!(code.decode_symbols(&c).unwrap(), (msg.clone(), Some(16)), "trial {trial}");
            let mut c = code.encode_symbols(&msg).unwrap();
            super::tests::corrupt(&mut c, 17, 8, &[], &mut rng);
            let (decoded, status) = code.decode_symbols(&c).unwrap();
            if status.is_none() {
                failures += 1;
            } else {
                assert_ne!(decoded, msg, "trial {trial}: 17 errors cannot be corrected");
                miscorrections += 1;
            }
        }
        // the miscorrection probability of a (255, 223) decoder is about 1/16! ≈ 5e-14
        // (McEliece & Swanson 1986): with 300 trials, every word is a detected failure
        assert_eq!((failures, miscorrections), (300, 0));
    }

    /// Monte Carlo over a symbol error channel at `p_s = 0.01` against the closed form of
    /// the word error probability: the (255, 247) code (t = 4), 20 000 words.
    #[test]
    fn word_error_rate_matches_the_closed_form_at_p_s_0_01() {
        let code = ReedSolomon::new(8, 4).unwrap();
        let p_s = 0.01;
        let trials = 20_000;
        let mut rng = Rng::new(2024);
        let mut lost = 0usize;
        for _ in 0..trials {
            let msg: Vec<u16> = (0..247).map(|_| rng.below(256) as u16).collect();
            let mut c = code.encode_symbols(&msg).unwrap();
            for v in &mut c {
                if rng.uniform() < p_s {
                    *v ^= 1 + rng.below(255) as u16;
                }
            }
            let (decoded, status) = code.decode_symbols(&c).unwrap();
            if status.is_none() || decoded != msg {
                lost += 1;
            }
        }
        let p_w = word_error_probability(255, 4, p_s);
        let measured = lost as f64 / trials as f64;
        // the count of lost words is binomial: standard error sqrt(p (1 - p) / N) ≈ 0.0023
        // at p ≈ 0.11, N = 20 000; the tolerance is four standard errors (P ≈ 6e-5 of a
        // false failure)
        let tol = 4.0 * (p_w * (1.0 - p_w) / trials as f64).sqrt();
        assert!((measured - p_w).abs() < tol, "measured {measured}, closed form {p_w}, tolerance {tol}");
    }
}

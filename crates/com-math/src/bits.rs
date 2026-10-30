//! Integers as bit strings: the reflected binary Gray code and the conversions between
//! an index and its bits, most significant bit first.
//!
//! # Model
//!
//! The binary reflected Gray code of $i$ and its inverse are
//!
//! $$
//! g(i) = i \oplus \lfloor i / 2 \rfloor, \qquad
//! g^{-1}(g) = \bigoplus_{k \ge 0} \left\lfloor g / 2^{k} \right\rfloor,
//! $$
//!
//! so that $g(i)$ and $g(i + 1)$ differ in one bit. An index $i$ of $n$ bits is written
//! $b_{0} b_{1} \dots b_{n-1}$ with $i = \sum_{k=0}^{n-1} b_{k} 2^{n-1-k}$ (the most
//! significant bit first, the order of the labels of every constellation and of the
//! symbols of every code of com-rs).
//!
//! # Examples
//! ```
//! use com_math::{bits_to_index, gray, gray_inverse, index_to_bits};
//! assert_eq!(gray(5), 7);                                   // 101 → 111
//! assert_eq!(gray_inverse(7), 5);
//! assert_eq!(bits_to_index(&[1, 0, 1, 1]), 11);
//! assert_eq!(index_to_bits(11, 4).collect::<Vec<u8>>(), [1, 0, 1, 1]);
//! ```

use com_core::{Error, Result};

/// The binary reflected Gray code $g(i) = i \oplus \lfloor i/2 \rfloor$.
///
/// # Examples
/// ```
/// use com_math::gray;
/// assert_eq!((0..8).map(gray).collect::<Vec<_>>(), [0, 1, 3, 2, 6, 7, 5, 4]);
/// ```
#[inline]
pub fn gray(i: usize) -> usize {
    i ^ (i >> 1)
}

/// The inverse of [`gray`]: $g^{-1}(g) = \bigoplus_{k \ge 0} \lfloor g / 2^{k} \rfloor$.
///
/// # Examples
/// ```
/// use com_math::{gray, gray_inverse};
/// assert!((0..1024).all(|i| gray_inverse(gray(i)) == i));
/// ```
#[inline]
pub fn gray_inverse(mut g: usize) -> usize {
    let mut i = 0;
    while g > 0 {
        i ^= g;
        g >>= 1;
    }
    i
}

/// The index $i = \sum_{k} b_{k} 2^{n-1-k}$ of the bits $b_{0} \dots b_{n-1}$, the most
/// significant first (only the lowest bit of every value is read).
///
/// # Examples
/// ```
/// use com_math::bits_to_index;
/// assert_eq!(bits_to_index(&[1, 1, 0]), 6);
/// assert_eq!(bits_to_index(&[]), 0);
/// ```
#[inline]
pub fn bits_to_index(bits: &[u8]) -> usize {
    bits.iter().fold(0usize, |acc, &b| (acc << 1) | usize::from(b & 1))
}

/// The `n_bits` bits of `index`, the most significant first (the bits above $2^{n}$
/// are dropped).
///
/// # Examples
/// ```
/// use com_math::index_to_bits;
/// assert_eq!(index_to_bits(6, 3).collect::<Vec<u8>>(), [1, 1, 0]);
/// assert_eq!(index_to_bits(6, 2).collect::<Vec<u8>>(), [1, 0]);
/// ```
#[inline]
pub fn index_to_bits(index: usize, n_bits: usize) -> impl Iterator<Item = u8> {
    (0..n_bits).rev().map(move |k| ((index >> k) & 1) as u8)
}

/// The indices of the blocks of `n_bits` bits of `bits`, each the [`bits_to_index`] of
/// its block (the labels of the symbols of a bit stream, the symbols of a code over
/// $\mathrm{GF}(2^{m})$).
///
/// # Errors
/// [`Error::Config`] for `n_bits = 0`, [`Error::Shape`] if the number of bits is not a
/// multiple of `n_bits`.
///
/// # Examples
/// ```
/// use com_math::bits::bits_to_indices;
/// assert_eq!(bits_to_indices(&[1, 0, 1, 1, 1, 0], 3)?, [5, 6]);
/// assert!(bits_to_indices(&[1, 0, 1], 2).is_err());
/// # Ok::<(), com_core::Error>(())
/// ```
pub fn bits_to_indices(bits: &[u8], n_bits: usize) -> Result<Vec<usize>> {
    if n_bits == 0 {
        return Err(Error::Config("a symbol needs at least one bit".into()));
    }
    if bits.len() % n_bits != 0 {
        return Err(Error::Shape(format!("{} bits is not a multiple of {n_bits} bits per symbol", bits.len())));
    }
    Ok(bits.chunks(n_bits).map(bits_to_index).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_neighbours_differ_by_one_bit_and_the_conversions_invert() {
        for i in 0..256 {
            assert_eq!(gray_inverse(gray(i)), i);
            assert_eq!((gray(i) ^ gray(i + 1)).count_ones(), 1);
            let bits: Vec<u8> = index_to_bits(i, 8).collect();
            assert_eq!(bits_to_index(&bits), i);
        }
    }
}

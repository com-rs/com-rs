//! Algebraic codes and puncturing: Reed–Solomon and BCH codes, the puncturing patterns
//! of the convolutional and turbo codes, the free distance of a punctured code.

use crate::{BArr, IArr, IntoPy, bvec, fvec, iarr, map_real, uvec};
use comrs::fec::bch::Bch;
use comrs::fec::block::{self, BlockCode};
use comrs::fec::conv::Trellis;
use comrs::fec::puncture::{self, Puncturer};
use comrs::fec::rs::{self, ReedSolomon};
use comrs::math::gf2m::primitive_polynomial;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A puncturing pattern from a Python argument: the name of a pattern of a standard
/// (`"ieee80211_2_3"`, `"ieee80211_3_4"`, `"ieee80211_5_6"`, `"dvbs_2_3"`,
/// `"dvbs_3_4"`, `"dvbs_5_6"`, `"dvbs_7_8"`, `"turbo_1_2"`) or the rows of a pattern
/// (`[[1, 1, 0], [1, 0, 1]]`, row `i` the output `i`).
pub(crate) fn puncturer(pattern: &Bound<'_, PyAny>) -> PyResult<Puncturer> {
    if let Ok(name) = pattern.extract::<String>() {
        return Puncturer::from_name(&name).py();
    }
    let rows: Vec<Vec<u8>> = pattern
        .extract()
        .map_err(|_| PyValueError::new_err("the puncturing pattern is a name or a list of rows of 0 and 1"))?;
    let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
    Puncturer::new(&refs).py()
}

/// The rows of a pattern (row `i` the output `i`, column `p` the phase `p`).
pub(crate) fn pattern_rows(p: &Puncturer) -> Vec<Vec<u32>> {
    (0..p.n_out()).map(|i| (0..p.period()).map(|phase| u32::from(p.keeps(phase, i))).collect()).collect()
}

/// A symbol array (non-negative integers below 2^16).
fn symbols(x: &Bound<'_, PyAny>) -> PyResult<Vec<u16>> {
    uvec(x)?
        .into_iter()
        .map(|s| u16::try_from(s).map_err(|_| PyValueError::new_err(format!("the symbol {s} exceeds 16 bits"))))
        .collect()
}

fn symbols_out(py: Python<'_>, v: Vec<u16>) -> IArr<'_> {
    iarr(py, v.into_iter().map(usize::from).collect())
}

/// A Reed–Solomon code over GF(2^m) (symbols of ``m`` bits) correcting ``t`` symbols
/// per word, with systematic encoding, errors-and-erasures decoding (Berlekamp–Massey,
/// Chien search, Forney 1965) and shortening (Lin & Costello 2004, ch. 7).
///
/// The generator polynomial has the roots ``β^b, …, β^(b+2t-1)`` with ``β = α^c`` for
/// the primitive element ``α`` of the field of the primitive polynomial ``poly``:
/// ``first_root`` is ``b`` (1 in the textbooks, 0 in DVB, 112 in CCSDS) and
/// ``root_step`` is ``c`` (11 in CCSDS). The code of length ``2^m - 1`` is shortened to
/// ``n`` symbols; a word carries ``k = n - 2t`` information symbols.
///
/// The code works on symbols (``encode_symbols``, ``decode_symbols``) and on bits,
/// ``m`` per symbol, the most significant first: ``encode`` and ``decode`` take whole
/// blocks of ``k m`` bits and ``n m`` LLRs like the other codes (hard decisions),
/// ``decode_bits`` one block of ``n m`` hard bits with the number of corrections.
///
/// Args:
///     m (int): bits per symbol, 2 to 16.
///     t (int): symbol errors corrected per word.
///     n (int | None): word length in symbols (``2^m - 1`` by default, less for a
///         shortened code).
///     first_root (int): exponent ``b`` of the first root of the generator.
///     root_step (int): exponent ``c`` of ``β = α^c``, coprime with ``2^m - 1``.
///     poly (int | None): the primitive polynomial of the field, bit ``i`` the
///         coefficient of ``x^i`` (the conventional one of Lin & Costello, Table 2.7,
///         by default: ``0x11D`` for ``m = 8``).
///
/// Examples:
///     The RS(204, 188) outer code of DVB-T and DVB-S (ETSI EN 300 744 V1.6.1,
///     Sec. 4.3.2; EN 300 421, Sec. 4.4): ``t = 8``, roots ``λ^0, …, λ^15`` over the
///     field of ``x^8 + x^4 + x^3 + x^2 + 1``, shortened from (255, 239):
///
///     >>> import comrs, numpy as np
///     >>> dvb = comrs.ReedSolomonCode(8, 8, n=204, first_root=0)
///     >>> (dvb.n, dvb.k, dvb.t)
///     (204, 188, 8)
///     >>> msg = np.arange(188) % 256
///     >>> c = dvb.encode_symbols(msg)
///     >>> c[[3, 50, 200]] ^= 0x5A
///     >>> decoded, corrected = dvb.decode_symbols(c)
///     >>> bool(np.array_equal(decoded, msg)), corrected
///     (True, 3)
///
///     The RS(255, 223) code of CCSDS 131.0-B-5, Sec. 4.2: ``t = 16`` over the field of
///     ``x^8 + x^7 + x^2 + x + 1`` with the roots ``α^(11 j)``, ``j = 112, …, 143``:
///
///     >>> ccsds = comrs.ReedSolomonCode(8, 16, poly=0x187, first_root=112, root_step=11)
///     >>> (ccsds.n, ccsds.k, round(ccsds.rate, 4))
///     (255, 223, 0.8745)
///     >>> g = ccsds.generator
///     >>> bool(np.array_equal(g, g[::-1]))  # self-reciprocal
///     True
#[pyclass(name = "ReedSolomonCode", module = "comrs")]
pub struct PyReedSolomon {
    inner: ReedSolomon,
    poly: u32,
}

#[pymethods]
impl PyReedSolomon {
    #[new]
    #[pyo3(signature = (m, t, *, n=None, first_root=1, root_step=1, poly=None))]
    fn new(
        m: u32,
        t: usize,
        n: Option<usize>,
        first_root: usize,
        root_step: usize,
        poly: Option<u32>,
    ) -> PyResult<Self> {
        let poly = match poly {
            Some(p) => p,
            None => primitive_polynomial(m)
                .ok_or_else(|| PyValueError::new_err(format!("the symbol size must be 2 to 16 bits, got {m}")))?,
        };
        let n = n.unwrap_or((1usize << m.min(16)) - 1);
        Ok(PyReedSolomon { inner: ReedSolomon::with_field(m, poly, first_root, root_step, t, n).py()?, poly })
    }

    /// int: Word length in symbols.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// int: Information symbols per word, ``n - 2t``.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Symbol errors corrected per word.
    #[getter]
    fn t(&self) -> usize {
        self.inner.t()
    }

    /// int: Bits per symbol.
    #[getter]
    fn m(&self) -> u32 {
        self.inner.m()
    }

    /// float: Code rate ``k / n``.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// int: Exponent of the first root of the generator.
    #[getter]
    fn first_root(&self) -> usize {
        self.inner.first_root()
    }

    /// int: Exponent ``c`` of ``β = α^c``.
    #[getter]
    fn root_step(&self) -> usize {
        self.inner.root_step()
    }

    /// int: The primitive polynomial of the field, bit ``i`` the coefficient of ``x^i``.
    #[getter]
    fn poly(&self) -> u32 {
        self.poly
    }

    /// numpy.ndarray: The generator polynomial ``g(x)``, lowest degree first (``2t + 1``
    /// symbols, int64).
    #[getter]
    fn generator<'py>(&self, py: Python<'py>) -> IArr<'py> {
        symbols_out(py, self.inner.generator().to_vec())
    }

    /// The word of a message of symbols: the message, then the ``2t`` parity symbols.
    ///
    /// Args:
    ///     msg (array_like): ``k`` message symbols, below ``2^m``.
    ///
    /// Returns:
    ///     numpy.ndarray: the ``n`` symbols of the word (int64).
    fn encode_symbols<'py>(&self, py: Python<'py>, msg: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        Ok(symbols_out(py, self.inner.encode_symbols(&symbols(msg)?).py()?))
    }

    /// Decode a word of received symbols, some positions known to be unreliable. The
    /// decoder corrects ``e`` errors and ``s`` erasures whenever ``2e + s <= 2t``.
    ///
    /// Args:
    ///     received (array_like): the ``n`` received symbols.
    ///     erasures (array_like, optional): the erased positions (0 to ``n - 1``).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int | None]: the ``k`` message symbols and the number of
    ///     corrected symbols (erasures included), ``None`` for a detected decoding
    ///     failure (the message symbols are then returned as received).
    #[pyo3(signature = (received, *, erasures=None))]
    fn decode_symbols<'py>(
        &self,
        py: Python<'py>,
        received: &Bound<'py, PyAny>,
        erasures: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(IArr<'py>, Option<usize>)> {
        let r = symbols(received)?;
        let (msg, corrected) = match erasures {
            Some(e) => self.inner.decode_erasures(&r, &uvec(e)?).py()?,
            None => self.inner.decode_symbols(&r).py()?,
        };
        Ok((symbols_out(py, msg), corrected))
    }

    /// Decode one block of hard bits.
    ///
    /// Args:
    ///     bits (array_like): ``n m`` received bits.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int | None]: the ``k m`` message bits and the number of
    ///     corrected symbols (``None`` for a detected decoding failure).
    fn decode_bits<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<(BArr<'py>, Option<usize>)> {
        let (msg, corrected) = self.inner.decode(&bvec(bits)?).py()?;
        Ok((msg.into_pyarray(py), corrected))
    }

    /// Encode whole blocks of bits (``m`` per symbol, most significant first).
    ///
    /// Args:
    ///     bits (array_like): a multiple of ``k m`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n m`` coded bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> code = comrs.ReedSolomonCode(4, 2)
    ///     >>> code.encode(comrs.bits(2 * code.k * code.m, comrs.Rng(1))).size
    ///     120
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        let b = bvec(bits)?;
        Ok(py.detach(|| self.inner.encode_stream(&b)).py()?.into_pyarray(py))
    }

    /// Decode whole blocks of LLRs by their signs (hard decisions); a block whose
    /// decoding fails keeps its message bits as received.
    ///
    /// Args:
    ///     llr (array_like): a multiple of ``n m`` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: ``k m`` bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> code = comrs.ReedSolomonCode(4, 2)
    ///     >>> b = comrs.bits(code.k * code.m, comrs.Rng(1))
    ///     >>> c = code.encode(b)
    ///     >>> c[[0, 9]] ^= 1  # two wrong bits, in two symbols
    ///     >>> bool(np.array_equal(code.decode(1.0 - 2.0 * c), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        let l = fvec(llr)?;
        Ok(py.detach(|| self.inner.decode_stream(&l)).py()?.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "ReedSolomonCode(n={}, k={}, t={}, m={})",
            self.inner.n(),
            self.inner.k(),
            self.inner.t(),
            self.inner.m()
        )
    }
}

/// Pack bits into symbols of ``m`` bits, the most significant first.
///
/// Args:
///     bits (array_like): a multiple of ``m`` bits.
///     m (int): bits per symbol, 1 to 16.
///
/// Returns:
///     numpy.ndarray: the symbols (int64).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.pack_symbols([1, 0, 1, 1, 1, 1, 0, 0], 4).tolist()
///     [11, 12]
#[pyfunction]
fn pack_symbols<'py>(py: Python<'py>, bits: &Bound<'py, PyAny>, m: u32) -> PyResult<IArr<'py>> {
    Ok(symbols_out(py, rs::pack_symbols(&bvec(bits)?, m).py()?))
}

/// Unpack symbols of ``m`` bits into bits, the most significant first.
///
/// Args:
///     symbols (array_like): the symbols, below ``2^m``.
///     m (int): bits per symbol, 1 to 16.
///
/// Returns:
///     numpy.ndarray: ``m`` bits per symbol (uint8).
///
/// Examples:
///     >>> import comrs
///     >>> comrs.unpack_symbols([11, 12], 4).tolist()
///     [1, 0, 1, 1, 1, 1, 0, 0]
#[pyfunction]
fn unpack_symbols<'py>(py: Python<'py>, symbols: &Bound<'py, PyAny>, m: u32) -> PyResult<BArr<'py>> {
    Ok(rs::unpack_symbols(&self::symbols(symbols)?, m).into_pyarray(py))
}

/// The word error rate of a bounded-distance decoder of a ``t``-error-correcting code
/// of ``n`` symbols over a memoryless channel of symbol error probability ``p_s``
/// (Proakis & Salehi, *Digital Communications*, 5th ed., Sec. 7.5.2): the probability
/// of more than ``t`` wrong symbols.
///
/// Args:
///     n (int): word length (symbols).
///     t (int): symbol errors corrected per word.
///     p_s (float | array_like): symbol error probability of the channel.
///
/// Returns:
///     float | numpy.ndarray: the word error probability at every ``p_s``.
///
/// Examples:
///     >>> import comrs
///     >>> pw = comrs.theory_rs_wer(255, 16, [1e-3, 1e-2])
///     >>> bool(pw[0] < 1e-15), round(float(pw[1]), 12)
///     (True, 1.4e-09)
#[pyfunction]
fn theory_rs_wer(n: usize, t: usize, p_s: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p_s, |p| Ok(rs::word_error_probability(n, t, p)))
}

/// The symbol error rate after decoding, approximating that a word of ``i > t``
/// errors keeps its ``i`` wrong symbols (Proakis & Salehi, Sec. 7.5.2; Sklar, *Digital
/// Communications*, 2nd ed., Sec. 8.1.2).
///
/// Args:
///     n (int): word length (symbols).
///     t (int): symbol errors corrected per word.
///     p_s (float | array_like): symbol error probability of the channel.
///
/// Returns:
///     float | numpy.ndarray: the decoded symbol error probability at every ``p_s``.
///
/// Examples:
///     >>> import comrs
///     >>> bool(comrs.theory_rs_ser(255, 16, 1e-2) < comrs.theory_rs_wer(255, 16, 1e-2))
///     True
#[pyfunction]
fn theory_rs_ser(n: usize, t: usize, p_s: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p_s, |p| Ok(rs::symbol_error_probability(n, t, p)))
}

/// The bit error rate after decoding of a code over GF(2^m): a wrong symbol carries on
/// average ``2^(m-1) / (2^m - 1)`` wrong bits per bit (Sklar, Sec. 8.1.2).
///
/// Args:
///     n (int): word length (symbols).
///     t (int): symbol errors corrected per word.
///     m (int): bits per symbol.
///     p_s (float | array_like): symbol error probability of the channel.
///
/// Returns:
///     float | numpy.ndarray: the decoded bit error probability at every ``p_s``.
#[pyfunction]
fn theory_rs_ber(n: usize, t: usize, m: u32, p_s: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(p_s, |p| Ok(rs::bit_error_probability(n, t, m, p)))
}

/// One BCH word as a block code: hard decisions of the LLRs, a decoding failure keeping
/// the message bits as received (the convention of the Reed–Solomon code).
struct BchBlock<'a>(&'a Bch);

impl BlockCode for BchBlock<'_> {
    fn k(&self) -> usize {
        self.0.k()
    }

    fn n(&self) -> usize {
        self.0.n()
    }

    fn encode_block(&self, info: &[u8]) -> comrs::Result<Vec<u8>> {
        self.0.encode(info)
    }

    fn decode_block(&self, llr: &[f64]) -> comrs::Result<Vec<u8>> {
        Ok(self.0.decode(&block::hard(llr))?.0)
    }
}

/// A binary BCH code of GF(2^m), narrow sense, of designed distance ``2t + 1`` (Lin &
/// Costello, *Error Control Coding*, 2nd ed., 2004, ch. 6), shortened to ``n`` bits:
/// the outer code of DVB-S2 (ETSI EN 302 307-1 V1.4.1, Sec. 5.3.1). Systematic
/// encoding; decoding of hard decisions by Berlekamp–Massey and a Chien search.
///
/// The code follows the contract of every code of the package: ``encode`` maps whole
/// blocks of ``k`` bits to ``n`` coded bits, ``decode`` whole blocks of ``n`` LLRs to
/// ``k`` bits (by their signs, a block whose decoding fails keeping its message bits as
/// received); ``decode_bits`` decodes one word of hard bits with the number of
/// corrections.
///
/// Args:
///     m (int): degree of the field, the code length being ``2^m - 1`` before
///         shortening.
///     t (int): bit errors corrected per word.
///     n (int | None): word length in bits (``2^m - 1`` by default).
///     poly (int | None): the primitive polynomial of the field, bit ``i`` the
///         coefficient of ``x^i`` (the conventional one by default).
///
/// Examples:
///     The (15, 7) code correcting two errors, ``g(x) = 1 + x^4 + x^6 + x^7 + x^8``
///     (Lin & Costello, ch. 6):
///
///     >>> import comrs, numpy as np
///     >>> code = comrs.BchCode(4, 2)
///     >>> (code.n, code.k, code.generator.tolist())
///     (15, 7, [1, 0, 0, 0, 1, 0, 1, 1, 1])
///     >>> msg = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
///     >>> c = code.encode(msg)
///     >>> c[[2, 11]] ^= 1
///     >>> decoded, corrected = code.decode_bits(c)
///     >>> bool(np.array_equal(decoded, msg)), corrected
///     (True, 2)
#[pyclass(name = "BchCode", module = "comrs")]
pub struct PyBch {
    inner: Bch,
}

#[pymethods]
impl PyBch {
    #[new]
    #[pyo3(signature = (m, t, *, n=None, poly=None))]
    fn new(m: u32, t: usize, n: Option<usize>, poly: Option<u32>) -> PyResult<Self> {
        let poly = match poly {
            Some(p) => p,
            None => primitive_polynomial(m)
                .ok_or_else(|| PyValueError::new_err(format!("the field degree must be 2 to 16, got {m}")))?,
        };
        let n = n.unwrap_or((1usize << m.min(16)) - 1);
        Ok(PyBch { inner: Bch::new(m, poly, t, n).py()? })
    }

    /// int: Word length in bits.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// int: Message bits per word, ``n - deg g``.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Bit errors corrected per word.
    #[getter]
    fn t(&self) -> usize {
        self.inner.t()
    }

    /// float: Code rate ``k / n``.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.k() as f64 / self.inner.n() as f64
    }

    /// numpy.ndarray: The generator polynomial ``g(x)``, lowest degree first (uint8).
    #[getter]
    fn generator<'py>(&self, py: Python<'py>) -> BArr<'py> {
        self.inner.generator().to_vec().into_pyarray(py)
    }

    /// Encode whole blocks: every word is its message, then its parity bits.
    ///
    /// Args:
    ///     bits (array_like): a multiple of ``k`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n`` coded bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.BchCode(4, 2).encode(comrs.bits(14, comrs.Rng(1))).size
    ///     30
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        let b = bvec(bits)?;
        Ok(py.detach(|| BchBlock(&self.inner).encode_stream(&b)).py()?.into_pyarray(py))
    }

    /// Decode whole blocks of LLRs by their signs (hard decisions); a block whose
    /// decoding fails keeps its message bits as received.
    ///
    /// Args:
    ///     llr (array_like): a multiple of ``n`` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: ``k`` bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> code = comrs.BchCode(4, 2)
    ///     >>> b = comrs.bits(14, comrs.Rng(1))
    ///     >>> c = code.encode(b)
    ///     >>> c[[1, 20]] ^= 1  # one error in each word
    ///     >>> bool(np.array_equal(code.decode(1.0 - 2.0 * c), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        let l = fvec(llr)?;
        Ok(py.detach(|| BchBlock(&self.inner).decode_stream(&l)).py()?.into_pyarray(py))
    }

    /// Decode one word of hard bits.
    ///
    /// Args:
    ///     bits (array_like): ``n`` received bits.
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int | None]: the ``k`` message bits and the number of
    ///     corrected errors (``None`` for a detected decoding failure, the message bits
    ///     then returned as received).
    fn decode_bits<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<(BArr<'py>, Option<usize>)> {
        let (msg, corrected) = self.inner.decode(&bvec(bits)?).py()?;
        Ok((msg.into_pyarray(py), corrected))
    }

    fn __repr__(&self) -> String {
        format!("BchCode(n={}, k={}, t={})", self.inner.n(), self.inner.k(), self.inner.t())
    }
}

/// The free distance of a feed-forward convolutional code, punctured or not: a
/// shortest path on the periodically time-varying trellis. Reproduces Yasuda, Kashiki
/// & Hirata, *IEEE Trans. Commun.* 32(3), 1984, Table I.
///
/// Args:
///     generators (list[int]): octal generators.
///     constraint (int): constraint length `K`.
///     puncture (str | list[list[int]] | None): a pattern name or its rows, as for
///         :class:`ConvolutionalCode` (none for the mother code).
///
/// Returns:
///     int: the free distance.
///
/// Examples:
///     >>> import comrs
///     >>> comrs.free_distance()
///     10
///     >>> [comrs.free_distance(puncture=p) for p in ("dvbs_2_3", "dvbs_3_4", "dvbs_5_6", "dvbs_7_8")]
///     [6, 5, 4, 3]
#[pyfunction]
#[pyo3(signature = (generators=vec![0o133, 0o171], constraint=7, *, puncture=None))]
fn free_distance(generators: Vec<u32>, constraint: u32, puncture: Option<&Bound<'_, PyAny>>) -> PyResult<u32> {
    if !(2..=16).contains(&constraint) || generators.is_empty() {
        return Err(PyValueError::new_err("constraint length 2 to 16 and at least one generator"));
    }
    let trellis = Trellis::feedforward(constraint, &generators).py()?;
    let p = match puncture {
        Some(p) => puncturer(p)?,
        None => Puncturer::identity(trellis.n_out).py()?,
    };
    puncture::free_distance(&trellis, &p).py()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReedSolomon>()?;
    m.add_class::<PyBch>()?;
    m.add_function(wrap_pyfunction!(pack_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(theory_rs_wer, m)?)?;
    m.add_function(wrap_pyfunction!(theory_rs_ser, m)?)?;
    m.add_function(wrap_pyfunction!(theory_rs_ber, m)?)?;
    m.add_function(wrap_pyfunction!(free_distance, m)?)?;
    Ok(())
}

//! Channel coding: CRC, scrambling, interleaving, convolutional, turbo, LDPC and polar
//! codes, and the finite-length limits (the Reed–Solomon and BCH codes and the
//! puncturing patterns are in [`crate::fec_codes`]).

use crate::fec_codes::{pattern_rows, puncturer};
use crate::{BArr, FArr, IArr, IntoPy, bvec, count, extract_error, farr, fvec, iarr, map_real, py_bool, uvec};
use comrs::fec::block::BlockCode;
use comrs::fec::bounds;
use comrs::fec::ccdm::Ccdm;
use comrs::fec::conv::{ConvolutionalCode, Trellis};
use comrs::fec::crc::Crc;
use comrs::fec::interleaving::Interleaver;
use comrs::fec::ldpc::{self, CheckRule, LdpcCode};
use comrs::fec::polar::PolarCode;
use comrs::fec::puncture::{Punctured, PuncturedTurbo, Puncturer};
use comrs::fec::scrambling::Scrambler;
use comrs::fec::turbo::TurboCode;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Encode whole blocks of bits with a code, the GIL released.
fn encode<'py, C>(py: Python<'py>, code: &C, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>>
where
    C: BlockCode + Sync + ?Sized,
{
    let b = bvec(bits)?;
    Ok(py.detach(|| code.encode_stream(&b)).py()?.into_pyarray(py))
}

/// Decode whole blocks of LLRs with a code, the GIL released.
fn decode<'py, C>(py: Python<'py>, code: &C, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>>
where
    C: BlockCode + Sync + ?Sized,
{
    let l = fvec(llr)?;
    Ok(py.detach(|| code.decode_stream(&l)).py()?.into_pyarray(py))
}

/// A cyclic redundancy check of 5G NR (3GPP TS 38.212, Sec. 5.1).
///
/// Args:
///     name (str): `"crc24a"`, `"crc24b"`, `"crc24c"`, `"crc16"`, `"crc11"` or `"crc6"`.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> crc = comrs.Crc("crc24a")
///     >>> w = crc.attach(np.array([1, 0, 1, 1], dtype=np.uint8))
///     >>> bool(crc.check(w)), w.size
///     (True, 28)
#[pyclass(name = "Crc", module = "comrs")]
pub struct PyCrc {
    inner: Crc,
}

#[pymethods]
impl PyCrc {
    fn __repr__(&self) -> String {
        format!("Crc('{}')", self.inner.name())
    }

    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Ok(PyCrc { inner: Crc::from_name(name).py()? })
    }

    /// int: Degree of the polynomial (number of parity bits).
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// The parity bits of a message.
    ///
    /// Args:
    ///     bits (array_like): the message.
    ///
    /// Returns:
    ///     numpy.ndarray: `width` parity bits (uint8).
    fn checksum<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.checksum(&bvec(bits)?).into_pyarray(py))
    }

    /// A message followed by its parity bits.
    ///
    /// Args:
    ///     bits (array_like): the message.
    ///
    /// Returns:
    ///     numpy.ndarray: the message and the `width` parity bits (uint8).
    fn attach<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.attach(&bvec(bits)?).into_pyarray(py))
    }

    /// Whether a word passes the check.
    ///
    /// Args:
    ///     word (array_like): information then parity bits.
    ///
    /// Returns:
    ///     bool: whether the parity bits match.
    fn check(&self, word: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.inner.check(&bvec(word)?))
    }
}

/// The Gold scrambler of 3GPP (TS 38.211, Sec. 5.2.1).
///
/// Args:
///     c_init (int): the initial value of the second register.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> s = comrs.Scrambler(0x1234)
///     >>> b = np.array([1, 0, 1, 1], dtype=np.uint8)
///     >>> s.scramble(s.scramble(b)).tolist()
///     [1, 0, 1, 1]
#[pyclass(name = "Scrambler", module = "comrs")]
pub struct PyScrambler {
    inner: Scrambler,
}

#[pymethods]
impl PyScrambler {
    fn __repr__(&self) -> String {
        format!("Scrambler(c_init={})", self.inner.c_init)
    }

    #[new]
    fn new(c_init: u32) -> Self {
        PyScrambler { inner: Scrambler { c_init } }
    }

    /// The first bits of the sequence.
    ///
    /// Args:
    ///     n (int): number of bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` bits (uint8).
    fn sequence<'py>(&self, py: Python<'py>, n: i64) -> PyResult<BArr<'py>> {
        Ok(self.inner.sequence(count(n, "n")?).into_pyarray(py))
    }

    /// Scramble (or descramble) bits.
    ///
    /// Args:
    ///     bits (array_like): the bits.
    ///
    /// Returns:
    ///     numpy.ndarray: the bits XOR the sequence (uint8).
    fn scramble<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.scramble(&bvec(bits)?).into_pyarray(py))
    }

    /// Descramble LLRs (flip the signs where the sequence is 1).
    ///
    /// Args:
    ///     llr (array_like): the LLRs of scrambled bits.
    ///
    /// Returns:
    ///     numpy.ndarray: the LLRs of the bits (float64).
    fn descramble_llr<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<FArr<'py>> {
        Ok(farr(py, self.inner.descramble_llr(&fvec(llr)?)))
    }
}

/// A permutation of the positions of a block: `Interleaver.random(n, seed=…)`,
/// `Interleaver.block(rows, cols)` or `Interleaver.qpp(k, f1, f2)`. It permutes any
/// array and keeps its dtype (bits, LLRs, symbols).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> il = comrs.Interleaver.block(2, 3)
///     >>> il.interleave(np.arange(6)).tolist()
///     [0, 3, 1, 4, 2, 5]
///     >>> il.deinterleave(il.interleave(np.arange(6))).tolist()
///     [0, 1, 2, 3, 4, 5]
#[pyclass(name = "Interleaver", module = "comrs")]
pub struct PyInterleaver {
    inner: Interleaver,
}

#[pymethods]
impl PyInterleaver {
    /// A uniform random permutation.
    ///
    /// Args:
    ///     n (int): the block length.
    ///     seed (int): the seed of the permutation.
    ///
    /// Returns:
    ///     Interleaver: the interleaver.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> il = comrs.Interleaver.random(8, seed=1)
    ///     >>> sorted(il.perm.tolist()) == list(range(8))
    ///     True
    #[staticmethod]
    #[pyo3(signature = (n, *, seed=0))]
    fn random(n: i64, seed: u64) -> PyResult<Self> {
        Ok(PyInterleaver { inner: Interleaver::random(count(n, "n")?, seed) })
    }

    /// Write row by row, read column by column.
    ///
    /// Args:
    ///     rows (int): rows of the array.
    ///     cols (int): columns of the array.
    ///
    /// Returns:
    ///     Interleaver: the interleaver of `rows · cols` positions.
    #[staticmethod]
    fn block(rows: usize, cols: usize) -> Self {
        PyInterleaver { inner: Interleaver::block(rows, cols) }
    }

    /// The quadratic permutation polynomial `(f1 i + f2 i²) mod k`.
    ///
    /// Args:
    ///     k (int): the block length.
    ///     f1 (int): the linear coefficient.
    ///     f2 (int): the quadratic coefficient.
    ///
    /// Returns:
    ///     Interleaver: the interleaver.
    ///
    /// Raises:
    ///     ValueError: if the polynomial is not a permutation.
    #[staticmethod]
    fn qpp(k: usize, f1: usize, f2: usize) -> PyResult<Self> {
        Ok(PyInterleaver { inner: Interleaver::qpp(k, f1, f2).py()? })
    }

    /// numpy.ndarray: The permutation (int64): output `i` is input `perm[i]`.
    #[getter]
    fn perm<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.perm.clone())
    }

    /// int: The block length.
    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }

    /// Interleave a block: `y[i] = x[perm[i]]`.
    ///
    /// Args:
    ///     x (array_like): one block of any dtype (bits, LLRs, symbols).
    ///
    /// Returns:
    ///     numpy.ndarray: the permuted block, of the dtype of `x`.
    fn interleave<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let x = self.one_block(py, x)?;
        x.get_item(iarr(py, self.inner.perm.clone()))
    }

    /// The inverse permutation: `x[perm[i]] = y[i]`.
    ///
    /// Args:
    ///     y (array_like): one interleaved block of any dtype.
    ///
    /// Returns:
    ///     numpy.ndarray: the block in its original order, of the dtype of `y`.
    fn deinterleave<'py>(&self, py: Python<'py>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let y = self.one_block(py, y)?;
        let out = py.import("numpy")?.call_method1("empty_like", (&y,))?;
        out.set_item(iarr(py, self.inner.perm.clone()), &y)?;
        Ok(out)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Interleaver(len={})", self.inner.len())
    }
}

impl PyInterleaver {
    /// A 1-D array of one block, its dtype kept.
    fn one_block<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let a = py.import("numpy")?.call_method1("asarray", (x,))?;
        let shape: Vec<usize> = a.getattr("shape")?.extract()?;
        if shape != [self.inner.len()] {
            return Err(PyValueError::new_err(format!(
                "a block of shape {shape:?} for an interleaver of {}",
                self.inner.len()
            )));
        }
        Ok(a)
    }
}

/// A terminated rate-1/n convolutional code on blocks of `k` bits, decoded by Viterbi
/// (Viterbi 1967): the (133, 171) code of constraint length 7 by default, punctured by
/// `puncture` (Cain, Clark & Geist 1979): the name of a pattern of a standard
/// (`"ieee80211_2_3"`, `"ieee80211_3_4"`, `"ieee80211_5_6"` of IEEE 802.11-2020,
/// Sec. 17.3.5.6 and 19.3.11.6; `"dvbs_2_3"`, `"dvbs_3_4"`, `"dvbs_5_6"`, `"dvbs_7_8"`
/// of ETSI EN 300 421, Table 1) or the rows of a pattern, row `i` the output `i`
/// (`[[1, 1, 0], [1, 0, 1]]`). The receiver puts a zero LLR at every punctured
/// position.
///
/// The code follows the contract of every code of the package: `encode` maps whole
/// blocks of `k` bits to `n` coded bits, `decode` whole blocks of `n` LLRs (positive
/// for a 0) to `k` bits.
///
/// Args:
///     k (int): information bits per block.
///     generators (list[int]): octal generators (write them as `0o133`).
///     constraint (int): constraint length `K`.
///     hard_decision (bool): decode the signs of the LLRs only.
///     puncture (str | list[list[int]] | None): the puncturing pattern.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.ConvolutionalCode(100)
///     >>> b = comrs.bits(100, comrs.Rng(1))
///     >>> c = code.encode(b)
///     >>> llr = 4.0 * (1.0 - 2.0 * c)
///     >>> bool(np.array_equal(code.decode(llr), b))
///     True
///     >>> punctured = comrs.ConvolutionalCode(100, puncture="ieee80211_3_4")
///     >>> (punctured.n, round(punctured.rate, 3))
///     (142, 0.704)
///     >>> bool(np.array_equal(punctured.decode(4.0 * (1.0 - 2.0 * punctured.encode(b))), b))
///     True
#[pyclass(name = "ConvolutionalCode", module = "comrs")]
pub struct PyConv {
    inner: Punctured<ConvolutionalCode>,
    punctured: bool,
}

#[pymethods]
impl PyConv {
    fn __repr__(&self) -> String {
        format!(
            "ConvolutionalCode(k={}, n={}, constraint={}, punctured={})",
            self.inner.k(),
            self.inner.n(),
            self.inner.code.trellis.constraint,
            py_bool(self.punctured)
        )
    }

    #[new]
    #[pyo3(signature = (k, *, generators=vec![0o133, 0o171], constraint=7, hard_decision=false, puncture=None))]
    fn new(
        k: usize,
        generators: Vec<u32>,
        constraint: u32,
        hard_decision: bool,
        puncture: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let code = ConvolutionalCode { trellis: Trellis::feedforward(constraint, &generators).py()?, k, hard_decision };
        let p = match puncture {
            Some(p) => puncturer(p)?,
            None => Puncturer::identity(code.trellis.n_out).py()?,
        };
        Ok(PyConv { inner: Punctured::new(code, p).py()?, punctured: puncture.is_some() })
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Coded bits per block (tail included, after puncturing).
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// float: Code rate `k / n` (after puncturing).
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// list[list[int]] | None: The rows of the puncturing pattern (row `i` the output
    /// `i`), `None` without puncturing.
    #[getter]
    fn puncture(&self) -> Option<Vec<Vec<u32>>> {
        self.punctured.then(|| pattern_rows(&self.inner.puncturer))
    }

    /// The free distance of the (punctured) code (see :func:`free_distance`).
    ///
    /// Returns:
    ///     int: the free distance.
    fn free_distance(&self) -> PyResult<u32> {
        comrs::fec::puncture::free_distance(&self.inner.code.trellis, &self.inner.puncturer).py()
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` coded bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.ConvolutionalCode(10).encode(comrs.bits(20, comrs.Rng(1))).size
    ///     64
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (Viterbi).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0; 0 at the
    ///         punctured positions is added by the code).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> code = comrs.ConvolutionalCode(10)
    ///     >>> b = comrs.bits(20, comrs.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(1.0 - 2.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// The turbo code of LTE: two (13, 15) recursive systematic codes in parallel, rate
/// about 1/3, decoded by iterative log-MAP (Berrou, Glavieux & Thitimajshima 1993).
/// The interleaver is the QPP `(f1, f2)` if given (`"lte"`: the coefficients of LTE for
/// `k`, TS 36.212 Table 5.1.3-3), else a random permutation.
/// `puncture` punctures the systematic and the two parity streams by a pattern of
/// three rows applied at every information index (the tail bits are kept): the name
/// `"turbo_1_2"` (systematic 1 1, parity 1: 1 0, parity 2: 0 1, the rate-1/2 code of
/// Berrou et al. 1993, Sec. II) or the rows of a pattern.
///
/// Args:
///     k (int): information bits per block.
///     iterations (int): decoding iterations.
///     exact (bool): log-MAP (else max-log-MAP).
///     qpp (str | tuple[int, int] | None): the QPP interleaver `(f1, f2)`, or `"lte"`.
///     seed (int): seed of the random interleaver.
///     puncture (str | list[list[int]] | None): the puncturing pattern (three rows).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.TurboCode(40, qpp=(3, 10), puncture="turbo_1_2")
///     >>> (code.n, round(code.rate, 3))
///     (92, 0.435)
#[pyclass(name = "TurboCode", module = "comrs")]
pub struct PyTurbo {
    inner: Turbo,
}

/// A turbo code, punctured or not (the unpunctured codeword keeps its stream order).
enum Turbo {
    Plain(TurboCode),
    Punctured(PuncturedTurbo),
}

impl Turbo {
    fn code(&self) -> &TurboCode {
        match self {
            Turbo::Plain(c) => c,
            Turbo::Punctured(p) => &p.code,
        }
    }

    fn code_mut(&mut self) -> &mut TurboCode {
        match self {
            Turbo::Plain(c) => c,
            Turbo::Punctured(p) => &mut p.code,
        }
    }

    fn block(&self) -> &(dyn BlockCode + Sync) {
        match self {
            Turbo::Plain(c) => c,
            Turbo::Punctured(p) => p,
        }
    }
}

#[pymethods]
impl PyTurbo {
    fn __repr__(&self) -> String {
        format!(
            "TurboCode(k={}, n={}, iterations={}, punctured={})",
            self.inner.block().k(),
            self.inner.block().n(),
            self.inner.code().iterations,
            py_bool(matches!(self.inner, Turbo::Punctured(_)))
        )
    }

    #[new]
    #[pyo3(signature = (k, *, iterations=8, exact=true, qpp=None, seed=7, puncture=None))]
    fn new(
        k: usize,
        iterations: usize,
        exact: bool,
        qpp: Option<&Bound<'_, PyAny>>,
        seed: u64,
        puncture: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let il = match qpp {
            Some(q) if q.is_instance_of::<PyString>() => Interleaver::from_name(&q.extract::<String>()?, k).py()?,
            Some(q) => {
                let (f1, f2): (usize, usize) =
                    q.extract().map_err(|e| extract_error(q.py(), &e, "qpp is \"lte\" or a pair (f1, f2)"))?;
                Interleaver::qpp(k, f1, f2).py()?
            }
            None => Interleaver::random(k, seed),
        };
        let mut code = TurboCode::with_interleaver(il);
        code.iterations = iterations;
        code.exact = exact;
        let inner = match puncture {
            Some(p) => Turbo::Punctured(PuncturedTurbo::new(code, puncturer(p)?).py()?),
            None => Turbo::Plain(code),
        };
        Ok(PyTurbo { inner })
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.block().k()
    }

    /// int: Coded bits per block (`3k + 12` without puncturing).
    #[getter]
    fn n(&self) -> usize {
        self.inner.block().n()
    }

    /// float: Code rate (after puncturing).
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.block().code_rate()
    }

    /// list[list[int]] | None: The rows of the puncturing pattern (systematic, parity
    /// 1, parity 2), `None` without puncturing.
    #[getter]
    fn puncture(&self) -> Option<Vec<Vec<u32>>> {
        match &self.inner {
            Turbo::Plain(_) => None,
            Turbo::Punctured(p) => Some(pattern_rows(&p.puncturer)),
        }
    }

    /// int: Decoding iterations (read and write).
    #[getter]
    fn get_iterations(&self) -> usize {
        self.inner.code().iterations
    }

    #[setter]
    fn set_iterations(&mut self, n: usize) {
        self.inner.code_mut().iterations = n;
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` coded bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.TurboCode(40).encode(comrs.bits(80, comrs.Rng(1))).size
    ///     264
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, self.inner.block(), bits)
    }

    /// Decode whole blocks of LLRs (iterative log-MAP).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> code = comrs.TurboCode(40)
    ///     >>> b = comrs.bits(40, comrs.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, self.inner.block(), llr)
    }
}

/// An LDPC code: `LdpcCode.peg(n, k, var_degree=…, seed=…)` (progressive edge growth,
/// Hu, Eleftheriou & Arnold 2005) or `LdpcCode(n, rows)` from the rows of a
/// parity-check matrix. Decoded by belief propagation (see `set_decoder`): `rule` is
/// `"sum_product"` (the default, as in the specifications), `"min_sum"`,
/// `"normalized_min_sum"` (scaled by `alpha`, 0.8 by default) or `"offset_min_sum"`
/// (offset `alpha`); `layered` selects the layered schedule (Hocevar 2004, the
/// default); 50 iterations at most by default. The PEG construction uses the variable
/// degree 3 and the seed 1 by default.
///
/// Args:
///     n (int): coded bits per block.
///     rows (list[list[int]]): the positions of the ones of every row of the
///         parity-check matrix.
#[pyclass(name = "LdpcCode", module = "comrs")]
pub struct PyLdpc {
    inner: LdpcCode,
}

#[pymethods]
impl PyLdpc {
    fn __repr__(&self) -> String {
        format!("LdpcCode(n={}, k={})", self.inner.n(), self.inner.k())
    }

    #[new]
    fn new(n: usize, rows: Vec<Vec<usize>>) -> PyResult<Self> {
        Ok(PyLdpc { inner: LdpcCode::from_rows(n, rows).py()? })
    }

    /// A code built by progressive edge growth.
    ///
    /// Args:
    ///     n (int): coded bits per block.
    ///     k (int): information bits per block.
    ///     var_degree (int): the degree of every variable node.
    ///     seed (int): the seed of the construction.
    ///
    /// Returns:
    ///     LdpcCode: the code.
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> code = comrs.LdpcCode.peg(96, 48)
    ///     >>> code.n, code.k, code.girth() >= 6
    ///     (96, 48, True)
    #[staticmethod]
    #[pyo3(signature = (n, k, *, var_degree=ldpc::DEFAULT_VAR_DEGREE, seed=ldpc::DEFAULT_PEG_SEED))]
    fn peg(n: usize, k: usize, var_degree: usize, seed: u64) -> PyResult<Self> {
        Ok(PyLdpc { inner: LdpcCode::peg(n, k, var_degree, seed).py()? })
    }

    /// Choose the decoder.
    ///
    /// Args:
    ///     rule (str): the check-node rule, ``"sum_product"``, ``"min_sum"``,
    ///         ``"normalized_min_sum"`` (scaled by ``alpha``) or ``"offset_min_sum"``
    ///         (with the offset ``alpha``), as the specifications.
    ///     alpha (float): the scale or the offset of the min-sum rules.
    ///     layered (bool): the layered schedule (else flooding).
    ///     max_iterations (int): iterations at most.
    #[pyo3(signature = (
        rule=ldpc::DEFAULT_RULE.name(),
        *,
        alpha=ldpc::DEFAULT_ALPHA,
        layered=ldpc::DEFAULT_LAYERED,
        max_iterations=ldpc::DEFAULT_MAX_ITERATIONS,
    ))]
    fn set_decoder(&mut self, rule: &str, alpha: f64, layered: bool, max_iterations: usize) -> PyResult<()> {
        self.inner.rule = CheckRule::from_name(rule, alpha).py()?;
        self.inner.layered = layered;
        self.inner.max_iterations = max_iterations;
        Ok(())
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Coded bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// float: Code rate.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// list[list[int]]: The rows of the parity-check matrix (the positions of their
    /// ones).
    #[getter]
    fn rows(&self) -> Vec<Vec<usize>> {
        self.inner.rows().to_vec()
    }

    /// The girth of the Tanner graph.
    ///
    /// Returns:
    ///     int: the length of its shortest cycle.
    fn girth(&self) -> usize {
        self.inner.girth()
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` coded bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.LdpcCode.peg(96, 48).encode(comrs.bits(96, comrs.Rng(1))).size
    ///     192
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (belief propagation).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> code = comrs.LdpcCode.peg(96, 48)
    ///     >>> b = comrs.bits(48, comrs.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }

    /// Decode one block into the a posteriori LLRs of the coded bits.
    ///
    /// Args:
    ///     llr (array_like): `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, int]: the `n` a posteriori LLRs and the iterations run.
    fn decode_llr<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<(FArr<'py>, usize)> {
        let l = fvec(llr)?;
        let (post, it) = py.detach(|| self.inner.decode_llr(&l)).py()?;
        Ok((farr(py, post), it))
    }
}

/// A polar code of length `n` (a power of two) and `k` information bits, built by the
/// Gaussian approximation at `design_ebn0_db` (Trifonov 2012); decoded by SC
/// (Arıkan 2009), or SC list of size `list_size` (Tal & Vardy 2015) with the CRC `crc`
/// choosing the path (CA-SCL).
///
/// Args:
///     n (int): coded bits per block, a power of two.
///     k (int): information bits per block.
///     design_ebn0_db (float): the `Eb/N0` of the construction (dB).
///     crc (str, optional): the CRC of the list decoder (a name of `Crc`).
///     list_size (int): the list size `L` (1 for SC).
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.PolarCode(128, 64, crc="crc6", list_size=8)
///     >>> b = comrs.bits(64, comrs.Rng(2))
///     >>> llr = 4.0 * (1.0 - 2.0 * code.encode(b))
///     >>> bool(np.array_equal(code.decode(llr), b))
///     True
#[pyclass(name = "PolarCode", module = "comrs")]
pub struct PyPolar {
    inner: PolarCode,
}

#[pymethods]
impl PyPolar {
    fn __repr__(&self) -> String {
        format!("PolarCode(n={}, k={})", self.inner.n(), self.inner.k())
    }

    #[new]
    #[pyo3(signature = (n, k, *, design_ebn0_db=2.0, crc=None, list_size=1))]
    fn new(n: usize, k: usize, design_ebn0_db: f64, crc: Option<&str>, list_size: usize) -> PyResult<Self> {
        let mut code = PolarCode::new(n, k, design_ebn0_db).py()?;
        if let Some(c) = crc {
            code = code.with_crc(Crc::from_name(c).py()?).py()?;
        }
        Ok(PyPolar { inner: code.with_list(list_size) })
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Coded bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// float: Code rate.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.code_rate()
    }

    /// numpy.ndarray: The positions of the information and CRC bits (int64).
    #[getter]
    fn info_positions<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.info_positions().to_vec())
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` coded bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs
    ///     >>> comrs.PolarCode(64, 32).encode(comrs.bits(64, comrs.Rng(1))).size
    ///     128
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (SC or CA-SCL).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    ///
    /// Examples:
    ///     >>> import comrs, numpy as np
    ///     >>> code = comrs.PolarCode(64, 32)
    ///     >>> b = comrs.bits(32, comrs.Rng(1))
    ///     >>> bool(np.array_equal(code.decode(2.0 - 4.0 * code.encode(b)), b))
    ///     True
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// The LDPC code of the data channels of 5G NR (3GPP TS 38.212, Sec. 5.3.2, 5.4.2): one
/// code block of `k` information bits rate-matched to `n` bits; base graph 1 or 2 chosen
/// as the shared channels do (or forced), the smallest lifting size `Z` with
/// `K_b Z ≥ k`, filler bits, the first `2Z` bits punctured, the circular buffer read
/// from the start of redundancy version `rv`, the bit interleaver of `qm` bits per
/// symbol; decoded by belief propagation.
///
/// Args:
///     k (int): information bits per block (at most 8448 for base graph 1, 3840 for 2).
///     n (int): transmitted bits per block.
///     base_graph (int, optional): 1 or 2 (default: the rule of Sec. 7.2.2).
///     rv (int): the redundancy version, 0 to 3.
///     qm (int): bits per modulation symbol of the bit interleaver (1: none).
///     rule (str): the check-node rule (as `LdpcCode.set_decoder`).
///     alpha (float): the scale or the offset of the min-sum rules.
///     iterations (int): the maximum number of iterations.
///     layered (bool): the layered schedule (default), or flooding (``False``); both stop
///         early on a codeword.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.NrLdpc(100, 300)
///     >>> code.base_graph, code.lifting_size
///     (2, 18)
///     >>> b = comrs.bits(100, comrs.Rng(1))
///     >>> bool(np.array_equal(code.decode(4.0 - 8.0 * code.encode(b)), b))
///     True
#[pyclass(name = "NrLdpc", module = "comrs")]
pub struct PyNrLdpc {
    inner: comrs::fec::nr::NrLdpc,
}

#[pymethods]
impl PyNrLdpc {
    fn __repr__(&self) -> String {
        format!(
            "NrLdpc(k={}, n={}, base_graph={}, lifting_size={})",
            self.inner.k(),
            self.inner.n(),
            self.inner.base_graph(),
            self.inner.lifting_size()
        )
    }

    #[new]
    #[pyo3(signature = (
        k,
        n,
        *,
        base_graph = None,
        rv = 0,
        qm = 1,
        rule = ldpc::DEFAULT_RULE.name(),
        alpha = ldpc::DEFAULT_ALPHA,
        iterations = ldpc::DEFAULT_MAX_ITERATIONS,
        layered = ldpc::DEFAULT_LAYERED,
    ))]
    #[allow(clippy::too_many_arguments, reason = "keyword-only options of the Python constructor")]
    fn new(
        k: usize,
        n: usize,
        base_graph: Option<u8>,
        rv: u8,
        qm: usize,
        rule: &str,
        alpha: f64,
        iterations: usize,
        layered: bool,
    ) -> PyResult<Self> {
        let mut code = comrs::fec::nr::NrLdpc::with_options(k, n, base_graph, rv, qm).py()?;
        code.code.rule = CheckRule::from_name(rule, alpha).py()?;
        code.code.max_iterations = iterations;
        code.code.layered = layered;
        Ok(PyNrLdpc { inner: code })
    }

    /// int: Information bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Transmitted bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// int: The base graph, 1 or 2.
    #[getter]
    fn base_graph(&self) -> u8 {
        self.inner.base_graph()
    }

    /// int: The lifting size `Z`.
    #[getter]
    fn lifting_size(&self) -> usize {
        self.inner.lifting_size()
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` bits per block (uint8).
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs.
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// The polar code of the control channels of 5G NR (3GPP TS 38.212, Sec. 5.3.1, 5.4.1):
/// `k` payload bits with their CRC rate-matched to `n` bits (puncturing, shortening or
/// repetition by the sub-block interleaver), the frozen set of the polar sequence of
/// Table 5.3.1.2-1; decoded by CRC-aided SC list.
///
/// ``"uplink"``: the uplink control information (CRC6 and 3 parity-check bits for
/// 12 ≤ k ≤ 19, CRC11 above, the triangular interleaver of the coded bits).
/// ``"downlink"``: the downlink control information and the broadcast channel (CRC24C,
/// the input interleaver; the RNTI masking of a DCI is left to the caller).
///
/// Args:
///     k (int): payload bits per block.
///     n (int): transmitted bits per block (`E`).
///     link (str): ``"uplink"`` or ``"downlink"``.
///     list_size (int): the list size `L`.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> code = comrs.NrPolar(30, 100, list_size=8)
///     >>> code.mother_length
///     128
///     >>> b = comrs.bits(30, comrs.Rng(1))
///     >>> bool(np.array_equal(code.decode(3.0 - 6.0 * code.encode(b)), b))
///     True
#[pyclass(name = "NrPolar", module = "comrs")]
pub struct PyNrPolar {
    inner: comrs::fec::nr::NrPolar,
}

#[pymethods]
impl PyNrPolar {
    fn __repr__(&self) -> String {
        format!("NrPolar(k={}, n={}, mother_length={})", self.inner.k(), self.inner.n(), self.inner.mother_length())
    }

    #[new]
    #[pyo3(signature = (k, n, *, link="uplink", list_size=8))]
    fn new(k: usize, n: usize, link: &str, list_size: usize) -> PyResult<Self> {
        let code = comrs::fec::nr::NrPolar::from_name(link, k, n).py()?;
        Ok(PyNrPolar { inner: code.with_list(list_size) })
    }

    /// int: Payload bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// int: Transmitted bits per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// int: The mother code length `N`.
    #[getter]
    fn mother_length(&self) -> usize {
        self.inner.mother_length()
    }

    /// Encode whole blocks.
    ///
    /// Args:
    ///     bits (array_like): a multiple of `k` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: `n` bits per block (uint8).
    fn encode<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        encode(py, &self.inner, bits)
    }

    /// Decode whole blocks of LLRs (CRC-aided SC list).
    ///
    /// Args:
    ///     llr (array_like): a multiple of `n` LLRs (positive for a 0).
    ///
    /// Returns:
    ///     numpy.ndarray: `k` bits per block (uint8).
    fn decode<'py>(&self, py: Python<'py>, llr: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        decode(py, &self.inner, llr)
    }
}

/// Capacity of the binary-input AWGN channel for a code rate (the noise variance is
/// `1/(2 R Eb/N0)`).
///
/// Args:
///     ebn0_db (float | array_like): `Eb/N0` (dB).
///     rate (float): the code rate `R`.
///
/// Returns:
///     float | numpy.ndarray: the capacity (bits per channel use) at every `Eb/N0`.
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.biawgn_capacity(comrs.shannon_limit_ebn0_db(0.5), 0.5), 6)
///     0.5
#[pyfunction]
fn biawgn_capacity(ebn0_db: &Bound<'_, PyAny>, rate: f64) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::biawgn_capacity(bounds::bpsk_noise_variance(e, rate))))
}

/// The normal approximation of the smallest block error probability of an `(n, k)`
/// binary code over BPSK and AWGN (Polyanskiy, Poor & Verdú 2010).
///
/// Args:
///     n (int): block length.
///     k (int): information bits.
///     ebn0_db (float | array_like): `Eb/N0` (dB).
///
/// Returns:
///     float | numpy.ndarray: the block error probability at every `Eb/N0`.
///
/// Examples:
///     >>> import comrs
///     >>> p = comrs.normal_approximation(256, 128, [1.0, 3.0])
///     >>> bool(p[0] > p[1])
///     True
#[pyfunction]
fn normal_approximation(n: usize, k: usize, ebn0_db: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(ebn0_db, |e| Ok(bounds::normal_approximation(n, k, e)))
}

/// The smallest `Eb/N0` at which BPSK supports a code rate.
///
/// Args:
///     rate (float | array_like): the code rate `R`.
///
/// Returns:
///     float | numpy.ndarray: the limit (dB).
///
/// Examples:
///     >>> import comrs
///     >>> round(comrs.shannon_limit_ebn0_db(0.5), 3)
///     0.187
#[pyfunction]
fn shannon_limit_ebn0_db(rate: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    map_real(rate, |r| Ok(bounds::shannon_limit_ebn0_db(r)))
}

/// Constant-composition distribution matching (Schulte & Böcherer 2016): ``k`` uniform
/// bits to a sequence of ``n`` symbols of a fixed composition, by enumerative coding.
///
/// Args:
///     composition (list[int]): occurrences of every symbol in a block.
///
/// Examples:
///     >>> import comrs, numpy as np
///     >>> m = comrs.Ccdm([2, 1, 1])
///     >>> b = np.zeros(m.k, dtype=np.uint8)
///     >>> m.dematch(m.matched(b)).tolist() == b.tolist()
///     True
#[pyclass(name = "Ccdm", module = "comrs")]
pub struct PyCcdm {
    inner: Ccdm,
}

#[pymethods]
impl PyCcdm {
    fn __repr__(&self) -> String {
        format!("Ccdm(n={}, k={})", self.inner.n(), self.inner.k())
    }

    #[new]
    fn new(composition: Vec<usize>) -> PyResult<Self> {
        Ok(PyCcdm { inner: Ccdm::new(composition).py()? })
    }

    /// The matcher of blocks of ``n`` symbols nearest to a distribution.
    ///
    /// Args:
    ///     p (array_like): the target distribution of the symbols.
    ///     n (int): symbols per block.
    ///
    /// Returns:
    ///     Ccdm: the matcher.
    #[staticmethod]
    fn from_distribution(p: &Bound<'_, PyAny>, n: i64) -> PyResult<Self> {
        Ok(PyCcdm { inner: Ccdm::from_distribution(&fvec(p)?, count(n, "n")?).py()? })
    }

    /// numpy.ndarray: Occurrences of every symbol in a block (int64).
    #[getter]
    fn composition<'py>(&self, py: Python<'py>) -> IArr<'py> {
        iarr(py, self.inner.composition().to_vec())
    }

    /// int: Symbols per block.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// int: Bits per block.
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }

    /// float: Bits per symbol, ``k / n``.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    /// numpy.ndarray: The distribution of the symbols, ``composition / n`` (float64).
    #[getter]
    fn distribution<'py>(&self, py: Python<'py>) -> FArr<'py> {
        farr(py, self.inner.distribution())
    }

    /// The symbols of one block of bits.
    ///
    /// Args:
    ///     bits (array_like): ``k`` bits.
    ///
    /// Returns:
    ///     numpy.ndarray: ``n`` symbol indices (int64).
    fn matched<'py>(&self, py: Python<'py>, bits: &Bound<'py, PyAny>) -> PyResult<IArr<'py>> {
        let s = self.inner.matched(&bvec(bits)?).py()?;
        Ok(iarr(py, s))
    }

    /// The bits of a block of symbols (a used-up symbol is replaced by the nearest
    /// available one).
    ///
    /// Args:
    ///     symbols (array_like): ``n`` symbol indices.
    ///
    /// Returns:
    ///     numpy.ndarray: ``k`` bits (uint8).
    fn dematch<'py>(&self, py: Python<'py>, symbols: &Bound<'py, PyAny>) -> PyResult<BArr<'py>> {
        Ok(self.inner.dematch(&uvec(symbols)?).py()?.into_pyarray(py))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCcdm>()?;
    m.add_class::<PyCrc>()?;
    m.add_class::<PyScrambler>()?;
    m.add_class::<PyInterleaver>()?;
    m.add_class::<PyConv>()?;
    m.add_class::<PyTurbo>()?;
    m.add_class::<PyLdpc>()?;
    m.add_class::<PyPolar>()?;
    m.add_class::<PyNrLdpc>()?;
    m.add_class::<PyNrPolar>()?;
    m.add_function(wrap_pyfunction!(biawgn_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(normal_approximation, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_limit_ebn0_db, m)?)?;
    Ok(())
}

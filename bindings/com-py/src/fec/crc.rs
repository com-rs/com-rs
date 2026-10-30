//! The CRCs of 5G NR, the scrambler and the interleavers.

use super::*;

/// A cyclic redundancy check of 5G NR (3GPP TS 38.212, Sec. 5.1).
///
/// Args:
///     name (str): `"crc24a"`, `"crc24b"`, `"crc24c"`, `"crc16"`, `"crc11"` or `"crc6"`.
///
/// Raises:
///     ValueError: for an invalid parameter (the message names it).
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
    ///
    /// Raises:
    ///     ValueError: for bits that are not 0 or 1, or not a 1-D array.
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
    ///
    /// Raises:
    ///     ValueError: for bits that are not 0 or 1, or not a 1-D array.
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
    ///
    /// Raises:
    ///     ValueError: for bits that are not 0 or 1, or not a 1-D array.
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
    ///
    /// Raises:
    ///     ValueError: for a negative count.
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
    ///
    /// Raises:
    ///     ValueError: for bits that are not 0 or 1, or not a 1-D array.
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
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
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
    /// Raises:
    ///     ValueError: for a negative count.
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
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
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
    ///
    /// Raises:
    ///     ValueError: for an input that is not an array of numbers of the expected shape.
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

/// Register the bindings of this module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCrc>()?;
    m.add_class::<PyScrambler>()?;
    m.add_class::<PyInterleaver>()?;
    Ok(())
}

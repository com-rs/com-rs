# com-rs against other Python link-level libraries

The same links simulated with com-rs and with
[komm](https://github.com/rwnobrega/komm),
[OptiCommPy](https://github.com/edsonportosilva/OptiCommPy) (with and without numba)
and [Sionna](https://github.com/NVlabs/sionna) (PyTorch, CPU): do the results agree,
with the closed form when there is one and with each other otherwise, and how long
does each library take? `tools/benchmarks_page.py` renders the results into
`docs/python/benchmarks.rst`, which states the rules (same work, intervals, warm-up,
median of five runs, threads, precision).

| Script | Scenario | komm | OptiCommPy | Sionna | com-rs |
|---|---|:-:|:-:|:-:|:-:|
| `s1_awgn.py` | uncoded Gray QPSK and 16-QAM over AWGN, BER vs the exact closed form | yes | yes | yes | yes |
| `s2_rayleigh.py` | BPSK and QPSK over flat Rayleigh fading, BER vs theory | — | — | yes | yes |
| `s3_mimo.py` | 4 × 4 16-QAM, i.i.d. Rayleigh: ZF (vs theory), MMSE, ML | — | — | yes | yes |
| `s4_coding.py` | (133, 171) soft Viterbi; 5G NR LDPC (1024, 2048); one LDPC code and the same LLRs given to three BP decoders | Viterbi | BP decoder | yes | yes |
| `s5_ssfm.py` | split-step Fourier, 16-QAM, 32 GBd, 10 × 80 km, 1 km step: NMSE of the fields | — | yes | yes | yes |

com-rs runs three ways: the functions of the `comrs` package called block by block
(`comrs`), its parallel Monte Carlo runner from Python (`comrs-spec`, `comrs.run_spec`)
and the `comrs` binary of Rust on the same specification (`comrs-rust`, scenarios 1
and 4). "OptiCommPy without numba" is `NUMBA_DISABLE_JIT=1`, numba's own switch: the
`@njit` functions of OptiCommPy run as plain Python (OptiCommPy has no option of its
own); its results are those of the compiled code, so it is only timed. OptiCommPy is
GPL-3.0: the scripts import it, nothing of it is copied.

## Reproducing

On a quiet machine, from the root of the repository (Python 3.12; the versions are
pinned in `requirements.txt`):

```bash
uv venv --python 3.12 benchmarks/libraries/.venv
source benchmarks/libraries/.venv/bin/activate
uv pip install -r benchmarks/libraries/requirements.txt
maturin develop --release                    # comrs, from this checkout
cd benchmarks/libraries
python run_all.py                            # builds target/release/comrs first
python ../../tools/benchmarks_page.py        # docs/python/benchmarks.rst
```

`python run_all.py --quick` runs everything at small sizes in about two minutes, to
check that it works; its timings mean nothing. Other options: `--scenarios
s1_awgn,s4_coding`, `--libs comrs,sionna`, `--threads 1` (default `1,all`),
`--no-rust-build`, `--keep-work`. One scenario and one library:

```bash
python s3_mimo.py --lib sionna --threads 1 --out /tmp/s3_sionna.json [--quick]
```

(a script started by hand re-executes itself with the thread settings of `--threads`).

## What `run_all.py` does

Every (scenario, library, threads) runs in its own process, with every thread pool at
one thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS` = 1,
`torch.set_num_threads(1)`, `threads=1` for com-rs) or at the defaults of the
libraries (every core). The error rates come from the single-thread runs; the
all-thread runs only time. OptiCommPy with numba starts from an empty numba cache, so
that its first call measures the compilation. Each scenario writes
`results/<scenario>.json`: the error rates with their 95 % intervals and the closed
form, the timings (first call, then the median of five runs), the thread settings,
the precision of every library, the machine (`sysctl`, `sw_vers`), the Python and
package versions, the com-rs commit and the command of every run.

The repository's `.gitignore` ignores every `results/` directory: commit the result
files with `git add -f benchmarks/libraries/results/*.json`.

## Expected duration

About two minutes with `--quick`. The full run (10⁷ bits per point, 2000 frames per
point, 2¹⁴ symbols through the fibre, both thread modes) is estimated at 15 to 20
minutes by scaling a quick run, plus the build of `target/release/comrs` when it is
out of date. Most of it is Sionna's exhaustive ML detector (scenario 3) and its
decoders (scenario 4), and the exhaustive ML search of com-rs (scenario 3).
`--scenarios` and `--libs` run a part of it.

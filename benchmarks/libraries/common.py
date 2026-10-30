"""Shared machinery of the library benchmarks: arguments, threads, timing, intervals,
machine and version information, JSON output.

Every scenario script runs one library at a time, in its own process, so that the
thread settings (read by NumPy, numba, PyTorch and rayon when they start) apply from
the first import and that one library's imports cannot slow another one down. A
script started by hand re-executes itself with the environment of its ``--threads``
option; ``run_all.py`` starts the scripts with that environment directly.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

# The variants of the libraries, in the order of the tables.
LIBRARIES = {
    "comrs": "com-rs from Python: the functions of the comrs package, called block by block",
    "comrs-spec": "com-rs from Python: comrs.run_spec, the parallel Monte Carlo runner of Rust",
    "comrs-rust": "com-rs from Rust: the comrs command-line binary on the same specification",
    "komm": "komm (NumPy)",
    "opticommpy": "OptiCommPy with numba (its @njit kernels compiled)",
    "opticommpy-nojit": "OptiCommPy without numba (NUMBA_DISABLE_JIT=1: the same @njit kernels run as plain Python)",
    "sionna": "Sionna PHY (PyTorch, CPU, eager mode)",
}

# The variables that set the size of the thread pools, all set to 1 for the
# single-thread runs and left unset (the library default: every core) otherwise.
THREAD_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
]

REPEATS = 5


def thread_env(lib: str, threads: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of a run of ``lib`` on ``threads`` ("1" or "all")."""
    env = dict(os.environ if base is None else base)
    for v in THREAD_VARS:
        env.pop(v, None)
    if threads == "1":
        env.update({v: "1" for v in THREAD_VARS})
    env.pop("NUMBA_DISABLE_JIT", None)
    if lib == "opticommpy-nojit":
        env["NUMBA_DISABLE_JIT"] = "1"
    if lib == "opticommpy":
        # a fresh cache: the first call compiles the kernels instead of loading them
        cache = RESULTS / "_work" / f"numba_cache_{os.getpid()}_{time.time_ns()}"
        env["NUMBA_CACHE_DIR"] = str(cache)
    env["BENCH_ENV"] = f"{lib}:{threads}"
    return env


def parse_args(cases: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=sys.modules["__main__"].__doc__)
    p.add_argument("--lib", required=True, choices=list(LIBRARIES))
    p.add_argument("--threads", default="1", choices=["1", "all"])
    p.add_argument("--quick", action="store_true", help="small sizes, to check that it runs")
    p.add_argument("--timing-only", action="store_true", help="skip the error-rate sweep")
    p.add_argument("--cases", default=",".join(cases), help="comma-separated cases")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--work", default=str(RESULTS / "_work"), help="scratch directory")
    p.add_argument("--out", required=True, help="JSON file of this run")
    args = p.parse_args()
    args.cases = [c for c in args.cases.split(",") if c]
    if os.environ.get("BENCH_ENV") != f"{args.lib}:{args.threads}":
        env = thread_env(args.lib, args.threads)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return args


def setup_torch(threads: str) -> dict:
    """Set the PyTorch thread pools (before any computation) and return them."""
    import torch

    if threads == "1":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    return {
        "torch_intra_op_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def time_it(fn: Callable[[], object], repeats: int = REPEATS) -> dict:
    """One warm-up call (JIT compilation, caches, first allocations), reported apart,
    then the median of ``repeats`` timed calls (``time.perf_counter``)."""
    t0 = time.perf_counter()
    fn()
    warmup = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "warmup_s": warmup,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "times_s": times,
        "repeats": repeats,
    }


def wilson(errors: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a proportion (95 % by default)."""
    if total == 0:
        return (0.0, 1.0)
    p = errors / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    h = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate_point(errors: int, total: int, theory: float | None = None, **extra) -> dict:
    """An error rate of independent trials, its 95 % interval and its z-score against
    ``theory`` (binomial standard error at the theory)."""
    lo, hi = wilson(errors, total)
    p = errors / total if total else float("nan")
    d = {
        "errors": int(errors),
        "total": int(total),
        "value": p,
        "lo": lo,
        "hi": hi,
        "ci": "binomial",
        **extra,
    }
    if theory is not None:
        d["theory"] = float(theory)
        sd = math.sqrt(max(theory * (1 - theory), 1e-300) / total)
        d["z"] = (p - theory) / sd
    return d


def frame_ber_point(errors_per_frame, k: int, theory: float | None = None, **extra) -> dict:
    """The bit error rate of coded frames: the errors of a frame are not independent
    (they come in bursts), so the interval is that of the mean of the per-frame error
    counts (central limit theorem over the frames), not a binomial interval."""
    import numpy as np

    e = np.asarray(errors_per_frame, dtype=float)
    f = e.size
    p = e.sum() / (f * k)
    se = e.std(ddof=1) / math.sqrt(f) / k if f > 1 else float("inf")
    # no error: the rule of three (95 %) on the frames, the interval of the mean is empty
    hi = 3.0 / (f * k) if e.sum() == 0 else p + 1.96 * se
    d = {
        "errors": int(e.sum()),
        "total": int(f * k),
        "frames": int(f),
        "value": float(p),
        "lo": float(max(0.0, p - 1.96 * se)),
        "hi": float(hi),
        "se": float(se),
        "ci": "per-frame",
        **extra,
    }
    if theory is not None:
        d["theory"] = float(theory)
        d["z"] = float((p - theory) / se) if se > 0 else float("nan")
    return d


def versions() -> dict[str, str]:
    out = {}
    for pkg in [
        "comrs",
        "komm",
        "OptiCommPy",
        "sionna",
        "torch",
        "numba",
        "llvmlite",
        "numpy",
        "scipy",
    ]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_info() -> dict:
    """The machine, gathered automatically (macOS: sysctl and sw_vers)."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _run(["sysctl", "-n", "hw.physicalcpu"]),
        "core_levels": [],
        "ram_gb": None,
        "macos": _run(["sw_vers", "-productVersion"]),
        "rustc": _run(["rustc", "--version"]),
    }
    for i in range(int(_run(["sysctl", "-n", "hw.nperflevels"]) or 0)):
        name = _run(["sysctl", "-n", f"hw.perflevel{i}.name"])
        n = _run(["sysctl", "-n", f"hw.perflevel{i}.physicalcpu"])
        info["core_levels"].append(f"{n} {name}")
    mem = _run(["sysctl", "-n", "hw.memsize"])
    if mem:
        info["ram_gb"] = round(int(mem) / 2**30, 1)
    return info


def git_commit() -> dict:
    commit = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    return {"commit": commit, "dirty": bool(status)}


def threading_info() -> dict:
    """The thread pools of the libraries loaded by the run, as they ended up."""
    info: dict = {"cpu_count": os.cpu_count()}
    if "numba" in sys.modules:
        import numba

        info["numba_threads"] = numba.get_num_threads()
        info["numba_jit_disabled"] = bool(numba.config.DISABLE_JIT)
    if "torch" in sys.modules:
        import torch

        info["torch_intra_op_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
    return info


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")


def main(scenario: str, cases: dict[str, dict], runners: dict[str, Callable]) -> None:
    """The entry point of a scenario script: run the cases of ``--lib`` that it
    supports and write the JSON of the run."""
    args = parse_args(list(cases))
    todo = [c for c in args.cases if cases[c].get(args.lib) is True]
    meta = {
        "scenario": scenario,
        "library": args.lib,
        "threads": args.threads,
        "quick": args.quick,
        "timing_only": args.timing_only,
        "env": {
            v: os.environ.get(v) for v in [*THREAD_VARS, "NUMBA_DISABLE_JIT", "NUMBA_CACHE_DIR"]
        },
        "versions": versions(),
    }
    t0 = time.perf_counter()
    results = runners[args.lib](args, todo) if todo else []
    meta["wall_s"] = time.perf_counter() - t0
    meta["threading"] = threading_info()
    write_json(args.out, {**meta, "cases": results})

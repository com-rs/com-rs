"""Scenario 1: uncoded Gray QPSK and 16-QAM over AWGN, BER against the exact closed form.

Every library draws its own bits and noise, maps with its own Gray mapper, adds the
noise of its own channel for the Eb/N0 of the point and decides by minimum distance
(Sionna: hard decisions of max-log LLRs, the same decisions). The BER is counted on
the same number of bits per point for every library, in blocks of the same size.
Timing: one block of ``TIMING_BITS`` bits at Eb/N0 = 6 dB (bits, mapping, noise,
decisions, error count), median of 5 runs after a warm-up run.

    python s1_awgn.py --lib comrs --threads 1 [--quick] --out results/_work/s1.json
"""

from __future__ import annotations

import math
import time

import common
import numpy as np
import theory

SCENARIO = "s1_awgn"
TITLE = "Uncoded QPSK and 16-QAM over AWGN"
ORDERS = {"qpsk": 4, "16qam": 16}
ALL = {lib: True for lib in common.LIBRARIES}
CASES = {"qpsk": ALL, "16qam": ALL}
EBN0 = [float(e) for e in range(11)]
TIMING_EBN0 = 6.0
TIMING_TRIALS = 40  # trials of the specification runner per timing run


def sizes(quick: bool) -> dict:
    if quick:
        return {"bits_per_point": 200_000, "block": 100_000, "timing_bits": 200_000, "trials": 10}
    return {
        "bits_per_point": 10_000_000,
        "block": 1_000_000,
        "timing_bits": 1_000_000,
        "trials": 100,
    }


PARAMETERS = {
    "ebn0_db": EBN0,
    "bits_per_point": "10^7 (quick: 2·10^5), in blocks of 10^6 (quick: 10^5)",
    "timing": "one block of 10^6 bits (quick: 2·10^5) at Eb/N0 = 6 dB",
    "theory": "exact BER of Gray square QAM, Cho & Yoon, IEEE Trans. Commun. 50(7), 2002",
}


# ---------------------------------------------------------------------------- chains
# chain(n_bits, ebn0_db) -> bit errors, for one library and one constellation


def chain_comrs(m: int, seed: int):
    import comrs

    c = comrs.Constellation("qam", m).normalized()
    k = int(math.log2(m))
    rng = comrs.Rng(seed)

    def run(n: int, ebn0: float) -> int:
        b = comrs.bits(n, rng)
        y = comrs.awgn_ebn0(c.map_bits(b), ebn0, k, rng)
        return int(np.count_nonzero(c.demap_bits(y) != b))

    return run, "float64"


def chain_komm(m: int, seed: int):
    import komm

    const = komm.QAMConstellation(m)
    lab = komm.ReflectedRectangularLabeling(int(math.log2(m)))
    k = int(math.log2(m))
    es = const.mean_energy()
    rng = np.random.default_rng(seed)

    def run(n: int, ebn0: float) -> int:
        b = rng.integers(0, 2, n)
        x = const.indices_to_symbols(lab.bits_to_indices(b))
        n0 = es / (k * 10 ** (ebn0 / 10))
        y = komm.GaussianChannel(n0, rng=rng).transmit(x)
        bh = lab.indices_to_bits(const.closest_indices(y))
        return int(np.count_nonzero(bh.ravel() != b))

    return run, "float64"


def chain_opticommpy(m: int, seed: int):
    from optic.comm.modulation import demodulateGray, modulateGray
    from optic.models.channels import awgn
    from optic.utils import parameters

    k = int(math.log2(m))
    rng = np.random.default_rng(seed)
    state = {"block": 0}

    def run(n: int, ebn0: float) -> int:
        b = rng.integers(0, 2, n)
        x = modulateGray(b, m, "qam")
        p = parameters()
        p.snr = ebn0 + 10 * math.log10(k)  # Es/N0 (Fs = B = 1)
        p.seed = seed * 1_000_003 + state["block"]  # awgn re-seeds its generator per call
        state["block"] += 1
        y = awgn(x, p)
        bh = demodulateGray(y, m, "qam")
        return int(np.count_nonzero(bh != b))

    return run, "complex64 symbols (its Gray constellation), float64 noise"


def chain_sionna(m: int, seed: int):
    import torch
    from sionna.phy import config
    from sionna.phy.channel import AWGN
    from sionna.phy.mapping import BinarySource, Demapper, Mapper
    from sionna.phy.utils import ebnodb2no

    config.seed = seed
    k = int(math.log2(m))
    source, mapper, channel = BinarySource(), Mapper("qam", k), AWGN()
    demapper = Demapper("maxlog", "qam", k, hard_out=True)

    def run(n: int, ebn0: float) -> int:
        with torch.inference_mode():
            b = source([1, n])
            no = ebnodb2no(ebn0, k, 1.0)
            y = channel(mapper(b), no)
            bh = demapper(y, no)
            return int((bh != b).sum())

    return run, "float32 (config.precision = 'single')"


CHAINS = {
    "comrs": chain_comrs,
    "komm": chain_komm,
    "opticommpy": chain_opticommpy,
    "opticommpy-nojit": chain_opticommpy,
    "sionna": chain_sionna,
}


def run_chains(args, todo: list[str]) -> list[dict]:
    if args.lib == "sionna":
        threading = common.setup_torch(args.threads)
    else:
        threading = {}
    sz = sizes(args.quick)
    out = []
    for case in todo:
        m = ORDERS[case]
        run, precision = CHAINS[args.lib](m, args.seed)
        res = {"case": case, "precision": precision, **threading}
        n = sz["timing_bits"]
        if args.lib == "opticommpy-nojit":
            n = min(n, 100_000)  # plain-Python kernels: a smaller block, the time is per bit
        t0 = time.perf_counter()
        run(n, TIMING_EBN0)  # the first call: JIT compilation (numba), lazy initialisation
        res["first_call_s"] = time.perf_counter() - t0
        if not args.timing_only:
            pts = []
            t0 = time.perf_counter()
            for e in EBN0:
                errors = 0
                for _ in range(sz["bits_per_point"] // sz["block"]):
                    errors += run(sz["block"], e)
                pts.append(
                    common.rate_point(
                        errors, sz["bits_per_point"], float(theory.ber_qam_awgn(m, e)), ebn0_db=e
                    )
                )
            res["points"] = pts
            res["sweep_s"] = time.perf_counter() - t0
        last = {}
        res["timing"] = common.time_it(
            lambda n=n, run=run, last=last: last.__setitem__("e", run(n, TIMING_EBN0))
        )
        res["timing"].update(units=n, unit="bit", workload=f"{n} bits at Eb/N0 = {TIMING_EBN0} dB")
        res["timing_ber"] = last["e"] / n
        out.append(res)
    return out


# ------------------------------------------------------------- specification runner


def spec(m: int, ebn0: list[float], bits: int, trials: int, seed: int) -> dict:
    k = int(math.log2(m))
    return {
        "name": f"bench_awgn_{m}qam",
        "seed": seed,
        "batch": trials,
        "grid": {"ebn0_db": ebn0},
        "link": [
            {
                "type": "source",
                "constellation": {"kind": "qam", "m": m},
                "n_symbols": bits // (k * trials),
            },
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
        ],
        "metrics": {"report": ["ber"], "scale": "none"},
        "stop": {"min_trials": trials, "max_trials": trials},
    }


def spec_points(results: list[dict], m: int) -> list[dict]:
    pts = []
    for r in results:
        e = float(r["condition"]["ebn0_db"])
        ber = r["metrics"]["ber"]
        errors = ber["count"] if "count" in ber else ber["errors"]  # run_spec / the binary's report
        pts.append(
            common.rate_point(errors, ber["total"], float(theory.ber_qam_awgn(m, e)), ebn0_db=e)
        )
    return pts


def run_spec_lib(args, todo: list[str], *, rust: bool = False) -> list[dict]:
    """``comrs-spec`` (``comrs.run_spec``), or with ``rust`` ``comrs-rust`` (the binary)."""
    sz = sizes(args.quick)
    n = sz["timing_bits"]
    workload = f"{n} bits at Eb/N0 = {TIMING_EBN0} dB, {TIMING_TRIALS} trials"
    if rust:
        workload += " (process wall time, start-up included)"
    out = []
    for case in todo:
        m = ORDERS[case]
        out.append(
            common.spec_case(
                args,
                {"case": case, "precision": "float64"},
                spec(m, EBN0, sz["bits_per_point"], sz["trials"], args.seed),
                spec(m, [TIMING_EBN0], n, TIMING_TRIALS, args.seed + 1),
                lambda rows, m=m: spec_points(rows, m),
                rust=rust,
                units=n,
                unit="bit",
                workload=workload,
            )
        )
    return out


def run_rust_lib(args, todo: list[str]) -> list[dict]:
    return run_spec_lib(args, todo, rust=True)


RUNNERS = {
    **{lib: run_chains for lib in CHAINS},
    "comrs-spec": run_spec_lib,
    "comrs-rust": run_rust_lib,
}

if __name__ == "__main__":
    common.main(SCENARIO, CASES, RUNNERS)

"""Scenario 2: BPSK and Gray QPSK over flat Rayleigh fading (one coefficient per
symbol, perfectly known at the receiver), BER against the closed form.

Every library draws its own bits, channel coefficients h ~ CN(0, 1) and noise, and
detects with its own coherent detector given h (zero forcing over the 1 x 1 channel:
comrs.mimo_detect, Sionna's LinearDetector, the mimo_rx block of the specification).
komm and OptiCommPy have no fading channel. Timing: one block of ``TIMING_BITS`` bits
at Eb/N0 = 10 dB, median of 5 runs after a warm-up run.

    python s2_rayleigh.py --lib sionna --threads 1 [--quick] --out results/_work/s2.json
"""

from __future__ import annotations

import math
import time

import common
import numpy as np
import theory

SCENARIO = "s2_rayleigh"
TITLE = "BPSK and QPSK over flat Rayleigh fading"
ORDERS = {"bpsk": 2, "qpsk": 4}
SUPPORT = {
    "comrs": True,
    "comrs-spec": True,
    "komm": "no fading channel",
    "opticommpy": "no fading channel",
    "opticommpy-nojit": "no fading channel",
    "sionna": True,
}
CASES = {"bpsk": SUPPORT, "qpsk": SUPPORT}
EBN0 = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
TIMING_EBN0 = 10.0
TIMING_TRIALS = 40

PARAMETERS = {
    "ebn0_db": EBN0,
    "bits_per_point": "10^7 (quick: 2·10^5), in blocks of 10^6 (quick: 10^5)",
    "channel": "y = h x + w, h ~ CN(0, 1) independent for every symbol, known at the receiver",
    "timing": "one block of 10^6 bits (quick: 2·10^5) at Eb/N0 = 10 dB",
    "theory": "BPSK: (1 - sqrt(G / (1 + G))) / 2; Gray QPSK: the same per bit (Proakis & Salehi, 5th ed., Sec. 13.3)",
}


def sizes(quick: bool) -> dict:
    if quick:
        return {"bits_per_point": 200_000, "block": 100_000, "timing_bits": 200_000, "trials": 10}
    return {
        "bits_per_point": 10_000_000,
        "block": 1_000_000,
        "timing_bits": 1_000_000,
        "trials": 100,
    }


def chain_comrs(m: int, seed: int):
    import comrs

    c = comrs.Constellation("psk" if m == 2 else "qam", m).normalized()
    k = int(math.log2(m))
    rng = comrs.Rng(seed)

    def run(n: int, ebn0: float) -> int:
        b = comrs.bits(n, rng)
        x = c.map_bits(b)
        h = comrs.mimo_channels(x.size, 1, 1, rng)
        n0 = 1.0 / (k * 10 ** (ebn0 / 10))
        y = h[:, 0, 0] * x + rng.complex_noise(x.size, n0)
        labels = comrs.mimo_detect(h, y[None, :], c, n0, "zf")
        bh = comrs.unpack_symbols(labels.ravel(), k)
        return int(np.count_nonzero(bh != b))

    return run, "float64"


def chain_sionna(m: int, seed: int):
    import torch
    from sionna.phy import config
    from sionna.phy.channel import FlatFadingChannel
    from sionna.phy.mapping import BinarySource, Mapper
    from sionna.phy.mimo import LinearDetector
    from sionna.phy.utils import ebnodb2no

    config.seed = seed
    k = int(math.log2(m))
    kind = "pam" if m == 2 else "qam"
    source, mapper = BinarySource(), Mapper(kind, k)
    channel = FlatFadingChannel(1, 1, return_channel=True)
    detector = LinearDetector(
        "zf", "bit", "maxlog", constellation_type=kind, num_bits_per_symbol=k, hard_out=True
    )

    def run(n: int, ebn0: float) -> int:
        with torch.inference_mode():
            b = source([n // k, k])
            x = mapper(b)  # [n/k, 1]
            no = ebnodb2no(ebn0, k, 1.0)
            y, h = channel(x, no)
            s = torch.full((n // k, 1, 1), float(no), dtype=y.dtype)
            bh = detector(y, h, s)  # [n/k, 1, k]
            return int((bh.reshape(b.shape) != b).sum())

    return run, "float32 (config.precision = 'single')"


CHAINS = {"comrs": chain_comrs, "sionna": chain_sionna}


def run_chains(args, todo: list[str]) -> list[dict]:
    threading = common.setup_torch(args.threads) if args.lib == "sionna" else {}
    sz = sizes(args.quick)
    out = []
    for case in todo:
        m = ORDERS[case]
        run, precision = CHAINS[args.lib](m, args.seed)
        res = {"case": case, "precision": precision, **threading}
        n = sz["timing_bits"]
        t0 = time.perf_counter()
        run(n, TIMING_EBN0)
        res["first_call_s"] = time.perf_counter() - t0
        if not args.timing_only:
            pts = []
            t0 = time.perf_counter()
            for e in EBN0:
                errors = sum(
                    run(sz["block"], e) for _ in range(sz["bits_per_point"] // sz["block"])
                )
                pts.append(
                    common.rate_point(
                        errors,
                        sz["bits_per_point"],
                        float(theory.ber_qam_rayleigh(m, e)),
                        ebn0_db=e,
                    )
                )
            res["points"] = pts
            res["sweep_s"] = time.perf_counter() - t0
        res["timing"] = common.time_it(lambda n=n, run=run: run(n, TIMING_EBN0))
        res["timing"].update(units=n, unit="bit", workload=f"{n} bits at Eb/N0 = {TIMING_EBN0} dB")
        out.append(res)
    return out


def spec(m: int, ebn0: list[float], bits: int, trials: int, seed: int) -> dict:
    k = int(math.log2(m))
    return {
        "name": f"bench_rayleigh_{m}",
        "seed": seed,
        "batch": trials,
        "grid": {"ebn0_db": ebn0},
        "link": [
            {
                "type": "source",
                "constellation": {"kind": "psk" if m == 2 else "qam", "m": m},
                "n_symbols": bits // (k * trials),
            },
            {"type": "rayleigh", "n_rx": 1, "coherence": 1},
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
            {"type": "mimo_rx", "detector": "zf"},
        ],
        "metrics": {"report": ["ber"], "scale": "none"},
        "stop": {"min_trials": trials, "max_trials": trials},
    }


def spec_points(rows: list[dict], m: int) -> list[dict]:
    pts = []
    for row in rows:
        e = float(row["condition"]["ebn0_db"])
        ber = row["metrics"]["ber"]
        pts.append(
            common.rate_point(
                ber["count"], ber["total"], float(theory.ber_qam_rayleigh(m, e)), ebn0_db=e
            )
        )
    return pts


def run_spec_lib(args, todo: list[str]) -> list[dict]:
    sz = sizes(args.quick)
    n = sz["timing_bits"]
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
                units=n,
                unit="bit",
                workload=f"{n} bits at Eb/N0 = {TIMING_EBN0} dB, {TIMING_TRIALS} trials",
            )
        )
    return out


RUNNERS = {"comrs": run_chains, "sionna": run_chains, "comrs-spec": run_spec_lib}

if __name__ == "__main__":
    common.main(SCENARIO, CASES, RUNNERS)

"""Scenario 3: 4 x 4 spatial multiplexing of Gray 16-QAM over i.i.d. Rayleigh fading
(one channel matrix per vector, known at the receiver), detected by zero forcing, MMSE
and maximum likelihood.

Every stream carries unit-energy symbols (no power split between the antennas) and
Eb/N0 is the energy per bit received on one antenna over N0: N0 = 1 / (4 Eb/N0).
Zero forcing over 4 x 4 leaves one diversity branch per stream, so its BER is that of
16-QAM over SISO Rayleigh fading (Winters, Salz & Gitlin 1994); MMSE and ML have no
closed form here and are compared between the libraries. Maximum likelihood is an
exhaustive search over the 16^4 = 65536 vectors in Sionna and in comrs ("ml"), and
the Schnorr-Euchner sphere decoder in comrs ("sphere", the same decisions).
The specification runner splits the power between the antennas (mimo_tx), so it runs
at Eb/N0 + 10 log10(4) dB to have the same SNR. Timing: ``TIMING_VECTORS`` vectors at
Eb/N0 = 12 dB (channel, noise, detection), median of 5 runs after a warm-up run.

    python s3_mimo.py --lib comrs --threads 1 [--quick] --out results/_work/s3.json
"""

from __future__ import annotations

import math
import time

import common
import numpy as np
import theory

SCENARIO = "s3_mimo"
TITLE = "4 x 4 MIMO, 16-QAM, i.i.d. Rayleigh fading: ZF, MMSE and ML detection"
NT = NR = 4
M = 16
K = 4
NO_MIMO = "no MIMO detection"
SUPPORT = {
    "comrs": True,
    "comrs-spec": True,
    "komm": NO_MIMO,
    "opticommpy": "no MIMO detection (only the 2 x 2 adaptive equalisers of polarisation multiplexing)",
    "opticommpy-nojit": "no MIMO detection (only the 2 x 2 adaptive equalisers of polarisation multiplexing)",
    "sionna": True,
}
CASES = {
    "zf": SUPPORT,
    "mmse": SUPPORT,
    "ml": SUPPORT,
    "sphere": {
        **SUPPORT,
        "sionna": "no sphere decoder (its exhaustive ML detector is the case ml)",
    },
}
SPEC_DETECTOR = {"zf": "zf", "mmse": "mmse", "ml": "ml", "sphere": "sphere"}
EBN0 = [0.0, 4.0, 8.0, 12.0, 16.0]
TIMING_EBN0 = 12.0
SPEC_OFFSET_DB = 10 * math.log10(NT)

PARAMETERS = {
    "ebn0_db": EBN0,
    "vectors_per_point": "linear detectors and sphere: 10^5 (quick: 5·10^3); exhaustive ML: 2·10^4 (quick: 10^3)",
    "timing": "10^4 vectors at Eb/N0 = 12 dB (exhaustive ML: 10^3; quick: 2·10^3 and 2·10^2)",
    "theory": "ZF: 16-QAM over SISO Rayleigh fading (Winters, Salz & Gitlin, IEEE Trans. Commun. 42(2/3/4), 1994)",
}


def sizes(quick: bool, case: str) -> dict:
    heavy = case == "ml"
    if quick:
        return {
            "vectors": 1_000 if heavy else 5_000,
            "timing": 200 if heavy else 2_000,
            "blocks": 1,
        }
    return {
        "vectors": 20_000 if heavy else 100_000,
        "timing": 1_000 if heavy else 10_000,
        "blocks": 10,
    }


def theory_ber(case: str, ebn0: float) -> float | None:
    return float(theory.ber_qam_rayleigh(M, ebn0)) if case == "zf" else None


def chain_comrs(case: str, seed: int):
    import comrs

    c = comrs.Constellation("qam", M).normalized()
    rng = comrs.Rng(seed)

    def run(n: int, ebn0: float) -> np.ndarray:
        b = comrs.bits(n * NT * K, rng)
        x = c.map_bits(b).reshape(n, NT)
        h = comrs.mimo_channels(n, NR, NT, rng)
        n0 = 1.0 / (K * 10 ** (ebn0 / 10))
        # y = H x + w (the Python API draws H; the product is NumPy's)
        y = np.matmul(h, x[:, :, None])[:, :, 0] + rng.complex_noise(n * NR, n0).reshape(n, NR)
        labels = comrs.mimo_detect(h, y.T, c, n0, case)  # stream-major (NT, n)
        bh = comrs.unpack_symbols(labels.T.ravel(), K)
        return np.count_nonzero((bh != b).reshape(n, NT * K), axis=1)  # bit errors per vector

    return run, "float64"


def chain_sionna(case: str, seed: int):
    import torch
    from sionna.phy import config
    from sionna.phy.channel import FlatFadingChannel
    from sionna.phy.mapping import BinarySource, Mapper
    from sionna.phy.mimo import LinearDetector, MaximumLikelihoodDetector
    from sionna.phy.utils import ebnodb2no

    config.seed = seed
    source, mapper = BinarySource(), Mapper("qam", K)
    channel = FlatFadingChannel(NT, NR, return_channel=True)
    if case == "ml":
        det = MaximumLikelihoodDetector(
            "bit", "maxlog", NT, constellation_type="qam", num_bits_per_symbol=K, hard_out=True
        )
        chunk = 64  # vectors per call: the search holds 65536 candidates per vector
    else:
        eq = {"zf": "zf", "mmse": "lmmse"}[case]
        det = LinearDetector(
            eq, "bit", "maxlog", constellation_type="qam", num_bits_per_symbol=K, hard_out=True
        )
        chunk = None

    def run(n: int, ebn0: float) -> np.ndarray:
        with torch.inference_mode():
            b = source([n, NT * K])
            x = mapper(b)  # [n, NT]
            no = ebnodb2no(ebn0, K, 1.0)  # Es / (K Eb/N0), Es = 1
            y, h = channel(x, no)
            s = (float(no) * torch.eye(NR, dtype=y.dtype)).expand(n, NR, NR)
            if chunk is None:
                bh = det(y, h, s)
            else:
                bh = torch.cat(
                    [
                        det(y[i : i + chunk], h[i : i + chunk], s[i : i + chunk])
                        for i in range(0, n, chunk)
                    ]
                )
            return (bh.reshape(b.shape) != b).sum(dim=1).numpy()  # bit errors per vector

    return run, "float32 (config.precision = 'single')"


CHAINS = {"comrs": chain_comrs, "sionna": chain_sionna}


def run_chains(args, todo: list[str]) -> list[dict]:
    threading = common.setup_torch(args.threads) if args.lib == "sionna" else {}
    out = []
    for case in todo:
        sz = sizes(args.quick, case)
        run, precision = CHAINS[args.lib](case, args.seed)
        res = {"case": case, "precision": precision, **threading}
        n = sz["timing"]
        t0 = time.perf_counter()
        run(n, TIMING_EBN0)
        res["first_call_s"] = time.perf_counter() - t0
        if not args.timing_only:
            pts = []
            t0 = time.perf_counter()
            for e in EBN0:
                # the 16 bits of a vector share its channel: the interval is per vector
                errs = np.concatenate(
                    [run(sz["vectors"] // sz["blocks"], e) for _ in range(sz["blocks"])]
                )
                pts.append(
                    common.frame_ber_point(
                        errs, NT * K, theory_ber(case, e), ebn0_db=e, vectors=errs.size
                    )
                )
            res["points"] = pts
            res["sweep_s"] = time.perf_counter() - t0
        res["timing"] = common.time_it(lambda n=n, run=run: run(n, TIMING_EBN0))
        res["timing"].update(
            units=n, unit="vector", workload=f"{n} vectors at Eb/N0 = {TIMING_EBN0} dB"
        )
        out.append(res)
    return out


def spec(case: str, ebn0: list[float], vectors: int, trials: int, seed: int) -> dict:
    return {
        "name": f"bench_mimo_{case}",
        "seed": seed,
        "batch": trials,
        "grid": {"ebn0_db": [e + SPEC_OFFSET_DB for e in ebn0]},
        "link": [
            {"type": "source", "constellation": "16-QAM", "n_symbols": NT * vectors // trials},
            {"type": "mimo_tx", "n_tx": NT, "scheme": "spatial"},
            {"type": "rayleigh", "n_rx": NR, "coherence": 1},
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
            {"type": "mimo_rx", "detector": SPEC_DETECTOR[case]},
        ],
        "metrics": {"report": ["ber"], "scale": "none"},
        "stop": {"min_trials": trials, "max_trials": trials},
    }


def run_spec_lib(args, todo: list[str]) -> list[dict]:
    trials = 40
    out = []
    for case in todo:
        sz = sizes(args.quick, case)

        def points(rows: list[dict], case=case, vectors=sz["vectors"]) -> list[dict]:
            pts = []
            for e, row in zip(EBN0, rows, strict=True):
                ber = row["metrics"]["ber"]
                pts.append(
                    common.rate_point(
                        ber["count"], ber["total"], theory_ber(case, e), ebn0_db=e, vectors=vectors
                    )
                )
            return pts

        n = sz["timing"]
        out.append(
            common.spec_case(
                args,
                {"case": case, "precision": "float64", "ebn0_offset_db": SPEC_OFFSET_DB},
                spec(case, EBN0, sz["vectors"], trials, args.seed),
                spec(case, [TIMING_EBN0], n, trials, args.seed + 1),
                points,
                units=n,
                unit="vector",
                workload=f"{n} vectors at Eb/N0 = {TIMING_EBN0} dB, {trials} trials",
            )
        )
    return out


RUNNERS = {"comrs": run_chains, "sionna": run_chains, "comrs-spec": run_spec_lib}

if __name__ == "__main__":
    common.main(SCENARIO, CASES, RUNNERS)

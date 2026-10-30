"""Scenario 4: channel coding over BPSK and AWGN.

- ``conv``: the rate-1/2 convolutional code of constraint length 7, generators
  (133, 171) in octal, 1024 information bits per frame terminated by 6 tail bits
  (rate 1024/2060), soft-decision Viterbi decoding. komm counts the taps from the
  least significant bit: the same code is ``ConvolutionalCode([[0o155, 0o117]])``
  there (checked on the impulse response). Reference: the union bound with the
  distance spectrum of the trellis (Viterbi 1971).
- ``nr_ldpc``: the LDPC code of 5G NR (3GPP TS 38.212, Sec. 5.3.2), k = 1024, rate
  matched to n = 2048 (base graph 2 by the rule of Sec. 7.2.2, lifting size 104, the
  first 2Z bits punctured; the same in both libraries), sum-product
  decoding by flooding, 50 iterations at most, in both libraries (comrs's default is
  the layered schedule, which converges in about half the iterations; it is set to
  flooding here so that the error rates compare the same decoder). comrs stops early on
  a codeword, Sionna runs the 50 iterations (its default).
- ``ldpc_same_llr``: one (1024, 512) LDPC code (PEG construction of comrs, variable
  degree 3, seed 1), its parity-check matrix given to the three decoders that accept
  one (comrs, OptiCommPy, Sionna), and the same channel LLRs (drawn by comrs from the
  same seed in every process): flooding sum-product, 50 iterations at most. The
  decoders see identical inputs, so they should take the same decisions frame by
  frame.

Every library uses its own encoder, BPSK mapper, AWGN channel and LLR computation in
``conv`` and ``nr_ldpc`` (independent seeds); Eb/N0 counts the energy per information
bit. Timings: the decoder alone on ``TIMING_FRAMES`` frames of LLRs at the timing
point, and the whole chain (bits, encoding, mapping, noise, LLRs, decoding, error
count) on the same number of frames; median of 5 runs after a warm-up run.

    python s4_coding.py --lib sionna --threads 1 [--quick] --out results/_work/s4.json
"""

from __future__ import annotations

import time

import common
import numpy as np
import theory

SCENARIO = "s4_coding"
TITLE = "Channel coding over BPSK and AWGN"
K_CONV = 1024
CONV_GEN = [0o133, 0o171]
K_NR, N_NR = 1024, 2048
N_PEG, K_PEG = 1024, 512
ITERATIONS = 50
EBN0 = {
    "conv": [1.0, 2.0, 3.0, 4.0],
    "nr_ldpc": [1.0, 1.5, 2.0, 2.5],
    "ldpc_same_llr": [1.5, 2.0, 2.5, 3.0],
}
TIMING_EBN0 = {"conv": 3.0, "nr_ldpc": 2.0, "ldpc_same_llr": 2.0}
SPEC_ONLY_OWN = "the runner draws its own channel (this case feeds the same LLRs to every decoder)"
CASES = {
    "conv": {
        "comrs": True,
        "comrs-spec": True,
        "comrs-rust": True,
        "komm": True,
        "opticommpy": "no convolutional code",
        "opticommpy-nojit": "no convolutional code",
        "sionna": True,
    },
    "nr_ldpc": {
        "comrs": True,
        "comrs-spec": "the specification format has no 5G NR LDPC code (NrLdpc: Python and Rust API)",
        "comrs-rust": "the specification format has no 5G NR LDPC code (NrLdpc: Python and Rust API)",
        "komm": "no LDPC code",
        "opticommpy": "no 5G NR LDPC code (its decoder takes a parity-check matrix: see ldpc_same_llr)",
        "opticommpy-nojit": "no 5G NR LDPC code (its decoder takes a parity-check matrix: see ldpc_same_llr)",
        "sionna": True,
    },
    "ldpc_same_llr": {
        "comrs": True,
        "comrs-spec": SPEC_ONLY_OWN,
        "comrs-rust": SPEC_ONLY_OWN,
        "komm": "no LDPC (belief-propagation) decoder",
        "opticommpy": True,
        "opticommpy-nojit": True,
        "sionna": True,
    },
}

PARAMETERS = {
    "conv": "K = 7, (133, 171), k = 1024 + 6 tail bits, soft Viterbi; Eb/N0 = 1, 2, 3, 4 dB",
    "nr_ldpc": "5G NR LDPC, k = 1024, n = 2048 (base graph 2, Z = 104), flooding sum-product, 50 iterations; Eb/N0 = 1, 1.5, 2, 2.5 dB",
    "ldpc_same_llr": "PEG (1024, 512), flooding sum-product, 50 iterations, identical LLRs; Eb/N0 = 1.5, 2, 2.5, 3 dB",
    "frames_per_point": "2000 (quick: 100)",
    "timing": "200 frames (quick: 20) at Eb/N0 = 3 dB (conv) or 2 dB (LDPC); OptiCommPy without numba: 2 frames",
}


def sizes(quick: bool) -> dict:
    return (
        {"frames": 100, "timing": 20, "block": 50}
        if quick
        else {"frames": 2000, "timing": 200, "block": 200}
    )


def n0_of(ebn0: float, rate: float) -> float:
    """N0 of unit-energy BPSK for an Eb/N0 per information bit (complex noise of
    variance N0, N0/2 on the real axis)."""
    return 1.0 / (rate * 10 ** (ebn0 / 10))


# ---------------------------------------------------------------------- the codecs
# A codec gives: encode(F) -> (info bits [F, k], channel input), llrs(F, ebn0) ->
# (reference bits, LLRs in the library's convention), decode(llrs) -> decided bits
# [F, k_out], and the number k_out of bits compared per frame.


def codec_comrs(case: str, seed: int):
    import comrs

    rng = comrs.Rng(seed)
    bpsk = comrs.Constellation("psk", 2)
    if case == "conv":
        code = comrs.ConvolutionalCode(K_CONV, generators=CONV_GEN, constraint=7)
    elif case == "nr_ldpc":
        code = comrs.NrLdpc(K_NR, N_NR, iterations=ITERATIONS, rule="sum_product", layered=False)
    else:
        return codec_same_llr_comrs(seed)
    k, n = code.k, code.n

    def llrs(f: int, ebn0: float):
        b = comrs.bits(f * k, rng)
        x = bpsk.modulate(code.encode(b))
        n0 = n0_of(ebn0, k / n)
        y = x + rng.complex_noise(x.size, n0)
        return b.reshape(f, k), bpsk.llr(y, n0)

    def decode(llr):
        return code.decode(llr).reshape(-1, k)

    return llrs, decode, k, "float64"


def peg_code():
    import comrs

    code = comrs.LdpcCode.peg(N_PEG, K_PEG, var_degree=3, seed=1)
    code.set_decoder("sum_product", layered=False, max_iterations=ITERATIONS)
    return code


def same_llrs(seed: int):
    """The codewords and LLRs of ldpc_same_llr: the same in every process."""
    import comrs

    code = peg_code()
    rng = comrs.Rng(seed)
    bpsk = comrs.Constellation("psk", 2)

    def llrs(f: int, ebn0: float):
        b = comrs.bits(f * code.k, rng)
        c = code.encode(b)
        n0 = n0_of(ebn0, code.k / code.n)
        y = bpsk.modulate(c) + rng.complex_noise(c.size, n0)
        return c.reshape(f, code.n), bpsk.llr(y, n0).reshape(f, code.n)  # positive for a 0

    return code, llrs


def codec_same_llr_comrs(seed: int):
    code, llrs = same_llrs(seed)

    def decode(llr):
        return np.stack([(code.decode_llr(row)[0] < 0).astype(np.uint8) for row in llr])

    return llrs, decode, code.n, "float64"


def codec_opticommpy(case: str, seed: int):
    import scipy.sparse as sp
    from optic.comm.fec import decodeLDPC
    from optic.utils import parameters

    code, llrs = same_llrs(seed)
    rows = code.rows
    h = sp.csr_matrix(
        (
            np.ones(sum(map(len, rows)), dtype=np.uint8),
            (np.repeat(np.arange(len(rows)), list(map(len, rows))), np.concatenate(rows)),
        ),
        shape=(len(rows), code.n),
    )
    p = parameters()
    p.H, p.maxIter, p.alg = h, ITERATIONS, "SPA"

    def decode(llr):
        bits, _out, _fail = decodeLDPC(np.ascontiguousarray(llr.T), p)  # columns are codewords
        return bits.T.astype(np.uint8)

    return llrs, decode, code.n, "float32 (decodeLDPC default prec = np.float32)"


def codec_komm(case: str, seed: int):
    import komm

    # (133, 171) with the current input on the most significant bit = (155, 117) in komm
    cc = komm.ConvolutionalCode([[0o155, 0o117]])
    code = komm.TerminatedConvolutionalCode(cc, num_blocks=K_CONV, mode="zero-termination")
    decoder = komm.ViterbiDecoder(code, input_type="soft")
    bpsk, lab = komm.PAMConstellation(2), komm.NaturalLabeling(1)
    rng = np.random.default_rng(seed)
    k, n = code.dimension, code.length

    def llrs(f: int, ebn0: float):
        b = rng.integers(0, 2, (f, k))
        x = bpsk.indices_to_symbols(lab.bits_to_indices(code.encode(b).ravel()))
        sigma2 = n0_of(ebn0, k / n) / 2  # real channel: N0/2 per dimension
        y = komm.GaussianChannel(sigma2, rng=rng).transmit(x)
        return b, lab.marginalize(bpsk.posteriors(y, sigma2))  # log P(0) / P(1)

    def decode(llr):
        return decoder.decode(llr).reshape(-1, k)

    return llrs, decode, k, "float64"


def codec_sionna(case: str, seed: int):
    import torch
    from sionna.phy import config
    from sionna.phy.channel import AWGN
    from sionna.phy.fec.conv import ConvEncoder, ViterbiDecoder
    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder, LDPCBPDecoder
    from sionna.phy.mapping import BinarySource, Demapper, Mapper

    config.seed = seed
    if case == "ldpc_same_llr":
        code, same = same_llrs(seed)
        dec = LDPCBPDecoder(peg_pcm(code), num_iter=ITERATIONS, hard_out=True)

        def decode(llr):
            with torch.inference_mode():
                # Sionna's LLRs are logits, log P(1) / P(0): the opposite sign
                return dec(torch.from_numpy(-llr.astype(np.float32))).numpy().astype(np.uint8)

        return same, decode, code.n, "float32 (config.precision = 'single')"

    if case == "conv":
        enc = ConvEncoder(rate=0.5, constraint_length=7, terminate=True)
        dec = ViterbiDecoder(encoder=enc, method="soft_llr", return_info_bits=True)
        k, n = K_CONV, 2 * (K_CONV + 6)
    else:
        enc = LDPC5GEncoder(K_NR, N_NR)
        dec = LDPC5GDecoder(enc, num_iter=ITERATIONS, hard_out=True, return_infobits=True)
        k, n = K_NR, N_NR
    source, mapper, channel = BinarySource(), Mapper("pam", 1), AWGN()
    demapper = Demapper("app", "pam", 1)

    def llrs(f: int, ebn0: float):
        with torch.inference_mode():
            b = source([f, k])
            no = n0_of(ebn0, k / n)
            y = channel(mapper(enc(b)), no)
            return b, demapper(y, no)  # logits, log P(1) / P(0)

    def decode(llr):
        with torch.inference_mode():
            return dec(llr)

    return llrs, decode, k, "float32 (config.precision = 'single')"


def peg_pcm(code):
    h = np.zeros((len(code.rows), code.n), dtype=np.float32)
    for i, r in enumerate(code.rows):
        h[i, r] = 1
    return h


CODECS = {
    "comrs": codec_comrs,
    "komm": codec_komm,
    "opticommpy": codec_opticommpy,
    "opticommpy-nojit": codec_opticommpy,
    "sionna": codec_sionna,
}


def as_np(a) -> np.ndarray:
    return a.numpy() if hasattr(a, "numpy") else np.asarray(a)


def run_codecs(args, todo: list[str]) -> list[dict]:
    threading = common.setup_torch(args.threads) if args.lib == "sionna" else {}
    sz = sizes(args.quick)
    out = []
    for case in todo:
        llrs, decode, k_out, precision = CODECS[args.lib](case, args.seed)
        res = {"case": case, "precision": precision, **threading}

        def chain(f: int, ebn0: float, llrs=llrs, decode=decode) -> np.ndarray:
            ref, llr = llrs(f, ebn0)
            return np.count_nonzero(
                as_np(decode(llr)).reshape(f, -1) != as_np(ref).reshape(f, -1), axis=1
            )

        nt = 2 if args.lib == "opticommpy-nojit" else sz["timing"]
        e_t = TIMING_EBN0[case]
        t0 = time.perf_counter()
        chain(min(nt, 2), e_t)
        res["first_call_s"] = time.perf_counter() - t0
        if not args.timing_only:
            pts = []
            t0 = time.perf_counter()
            for e in EBN0[case]:
                errs = np.concatenate(
                    [chain(sz["block"], e) for _ in range(sz["frames"] // sz["block"])]
                )
                bound = (
                    float(theory.conv_union_bound(CONV_GEN, 7, K_CONV / (2 * (K_CONV + 6)), e))
                    if case == "conv"
                    else None
                )
                pt = common.frame_ber_point(errs, k_out, None, ebn0_db=e)
                if bound is not None:
                    pt["union_bound"] = bound
                fe = int(np.count_nonzero(errs))
                pt["bler"] = common.rate_point(fe, errs.size)
                if case == "ldpc_same_llr":
                    pt["errors_per_frame"] = errs.astype(int).tolist()
                pts.append(pt)
            res["points"] = pts
            res["sweep_s"] = time.perf_counter() - t0
            res["bits_per_frame"] = k_out
        _, llr = llrs(nt, e_t)
        res["timing"] = common.time_it(lambda llr=llr, decode=decode: decode(llr))
        res["timing"].update(
            units=nt, unit="frame", workload=f"decoder only, {nt} frames at Eb/N0 = {e_t} dB"
        )
        res["timing_chain"] = common.time_it(lambda nt=nt, e_t=e_t, chain=chain: chain(nt, e_t))
        res["timing_chain"].update(
            units=nt, unit="frame", workload=f"whole chain, {nt} frames at Eb/N0 = {e_t} dB"
        )
        out.append(res)
    return out


# ------------------------------------------------------------- specification runner


def spec(ebn0: list[float], frames: int, trials: int, seed: int) -> dict:
    code = {"type": "conv", "k": K_CONV}
    return {
        "name": "bench_conv",
        "seed": seed,
        "batch": trials,
        "grid": {"ebn0_db": ebn0},
        "link": [
            {"type": "fec_encode", "code": code, "blocks": frames // trials},
            {"type": "map", "constellation": {"kind": "psk", "m": 2}},
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
            "demap",
            {"type": "fec_decode", "code": code},
        ],
        "metrics": {"report": ["coded_ber", "bler"]},  # coded_ber: decoded information bits
        "stop": {"min_trials": trials, "max_trials": trials},
    }


def spec_points(rows: list[dict]) -> list[dict]:
    pts = []
    for r in rows:
        e = float(r["condition"]["ebn0_db"])
        m = r["metrics"]
        get = lambda d: d["count"] if "count" in d else d["errors"]  # noqa: E731
        # a binomial interval: the runner keeps the totals, not the errors of every frame
        pt = common.rate_point(get(m["coded_ber"]), m["coded_ber"]["total"], None, ebn0_db=e)
        pt["union_bound"] = float(
            theory.conv_union_bound(CONV_GEN, 7, K_CONV / (2 * (K_CONV + 6)), e)
        )
        pt["bler"] = common.rate_point(get(m["bler"]), m["bler"]["total"])
        pts.append(pt)
    return pts


def run_spec_lib(args, todo: list[str]) -> list[dict]:
    import comrs

    sz = sizes(args.quick)
    threads = 1 if args.threads == "1" else 0
    out = []
    for case in todo:
        res = {"case": case, "precision": "float64", "threads_option": threads}
        trials = 20
        if not args.timing_only:
            t0 = time.perf_counter()
            r = comrs.run_spec(spec(EBN0[case], sz["frames"], trials, args.seed), threads=threads)
            res["sweep_s"] = time.perf_counter() - t0
            res["points"] = spec_points(r)
            res["bits_per_frame"] = K_CONV
        nt, e_t = sz["timing"], TIMING_EBN0[case]
        s = spec([e_t], nt, trials, args.seed + 1)
        res["timing_chain"] = common.time_it(lambda s=s: comrs.run_spec(s, threads=threads))
        res["timing_chain"].update(
            units=nt,
            unit="frame",
            workload=f"whole chain, {nt} frames at Eb/N0 = {e_t} dB, {trials} trials",
        )
        out.append(res)
    return out


def run_rust_lib(args, todo: list[str]) -> list[dict]:
    from pathlib import Path

    from s1_awgn import run_rust_cli, rust_binary

    sz = sizes(args.quick)
    threads = 1 if args.threads == "1" else 0
    work = Path(args.work) / "rust"
    out = []
    for case in todo:
        res = {
            "case": case,
            "precision": "float64",
            "threads_option": threads,
            "binary": str(rust_binary()),
        }
        trials = 20
        if not args.timing_only:
            report, wall = run_rust_cli(
                spec(EBN0[case], sz["frames"], trials, args.seed), work, threads
            )
            res["sweep_s"] = wall
            res["points"] = spec_points(report["conditions"])
            res["bits_per_frame"] = K_CONV
        nt, e_t = sz["timing"], TIMING_EBN0[case]
        s = spec([e_t], nt, trials, args.seed + 1)
        res["timing_chain"] = common.time_it(lambda s=s: run_rust_cli(s, work, threads))
        res["timing_chain"].update(
            units=nt,
            unit="frame",
            workload=f"whole chain, {nt} frames at Eb/N0 = {e_t} dB, {trials} trials (process wall time)",
        )
        out.append(res)
    return out


def postprocess(merged: dict, work) -> dict:
    """ldpc_same_llr: the decoders saw the same LLRs; compare their errors frame by frame
    with those of comrs (same number of bit errors in a frame, same frame errors)."""
    per_lib = {}
    for run in merged["runs"]:
        if run.get("threads") != "1":
            continue
        for case in run.get("cases", []):
            if case["case"] == "ldpc_same_llr" and "points" in case:
                per_lib[run["library"]] = case["points"]
    base = per_lib.get("comrs")
    rows = []
    for lib, pts in per_lib.items():
        if lib == "comrs" or base is None:
            continue
        for p, q in zip(pts, base, strict=True):
            a, b = np.array(p["errors_per_frame"]), np.array(q["errors_per_frame"])
            rows.append(
                {
                    "library": lib,
                    "ebn0_db": p["ebn0_db"],
                    "frames": int(a.size),
                    "same_bit_errors": int(np.count_nonzero(a == b)),
                    "same_frame_outcome": int(np.count_nonzero((a > 0) == (b > 0))),
                }
            )
    return {"same_llr_vs_comrs": rows}


RUNNERS = {
    **{lib: run_codecs for lib in CODECS},
    "comrs-spec": run_spec_lib,
    "comrs-rust": run_rust_lib,
}

if __name__ == "__main__":
    common.main(SCENARIO, CASES, RUNNERS)

"""Speed of comrs against OptiCommPy on the same inputs.

Every case runs the two implementations on the same arrays, checks that they agree
(the oracle tests check the values in detail), and keeps the best of a few timed runs
after a warm-up run (which compiles the numba functions of OptiCommPy). comrs is timed
on one thread for the functions (OptiCommPy is single-threaded) and, for the Monte
Carlo campaign, on one thread and on every core.

Usage::

    python benchmarks/bench_opticommpy.py [--quick] [--out benchmarks/results.md]

writes a Markdown table and a JSON file next to it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from optic.comm import metrics as om
from optic.comm import modulation as omod
from optic.dsp import carrierRecovery as ocr
from optic.dsp import clockRecovery as ock
from optic.dsp import core as ocore
from optic.dsp import equalization as oeq
from optic.models import channels as och
from optic.utils import parameters

import comrs


def params(**kw):
    p = parameters()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def best_time(f: Callable[[], object], repeats: int) -> float:
    """Best wall time of ``repeats`` runs after one warm-up run (seconds)."""
    f()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def rel_err(a, b) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = min(a.size, b.size)
    return float(np.linalg.norm(a[:n] - b[:n]) / max(np.linalg.norm(b[:n]), 1e-300))


def cases(quick: bool) -> list[dict]:
    k = 4 if quick else 1
    rng = comrs.Rng(1)
    c16 = comrs.Constellation("qam", 16).normalized()
    out = []

    # split-step Fourier: 5 x 80 km, 0.5 km steps, 16384 symbols at 4 sps
    n = 65536 // k
    x = comrs.pnorm(
        comrs.fir_filter(
            comrs.pulse_shape("rrc", 4, n_taps=1025, rolloff=0.1),
            comrs.upsample(comrs.symbols(c16, n // 4, rng)[1], 4),
        )
    )
    x = x * np.sqrt(comrs.dbm2w(2.0))
    f = comrs.FiberParams(n_spans=5)
    link = comrs.Ssfm(f, step_km=0.5, amplification="ideal")
    p = params(
        Ltotal=400.0,
        Lspan=80.0,
        hz=0.5,
        alpha=0.2,
        D=17.0,
        gamma=1.3,
        Fc=193.1e12,
        Fs=128e9,
        amp="ideal",
        prgsBar=False,
    )
    out.append(
        dict(
            name=f"split-step Fourier, 5 x 80 km, 800 steps, {n} samples",
            ref=lambda: och.ssfm(x, p),
            ours=lambda: link.propagate(x, 128e9),
            repeats=1,
        )
    )
    out.append(
        dict(
            name="split-step Fourier in f32, same link",
            ref=lambda: och.ssfm(x, p),
            ours=lambda: link.propagate(x, 128e9, precision="f32"),
            repeats=1,
            tol=1e-4,
        )
    )

    # linear fibre channel, 2^20 samples
    y = rng.complex_noise(2**20 // k)
    pl = params(L=1000.0, alpha=0.2, D=17.0, Fc=193.1e12, Fs=64e9)
    out.append(
        dict(
            name=f"linear fibre channel, {y.size} samples",
            ref=lambda: och.linearFiberChannel(y, pl),
            ours=lambda: comrs.linear_fiber_channel(y, comrs.FiberParams(), 1000.0, 64e9),
            repeats=3,
        )
    )

    # FIR filtering with a long root-raised cosine
    h = comrs.pulse_shape("rrc", 2, n_taps=1024, rolloff=0.1)
    z = rng.complex_noise(2**18 // k)
    out.append(
        dict(
            name=f"FIR filter, 1024 taps, {z.size} samples",
            ref=lambda: ocore.firFilter(h, z),
            ours=lambda: comrs.fir_filter(h, z),
            repeats=3,
        )
    )

    # carrier recovery on 16-QAM
    _idx, s = comrs.symbols(c16, 2**15 // k, rng)
    r = comrs.awgn(s * np.exp(1j * (0.3 + 1e-4 * np.arange(s.size))), 18.0, rng)
    out.append(
        dict(
            name=f"blind phase search, B = 64, N = 35, {r.size} symbols",
            ref=lambda: ocr.bps(r.reshape(-1, 1), 35, c16.points, 64),
            ours=lambda: comrs.bps(r, c16, n=35, b=64),
            repeats=3,
        )
    )
    out.append(
        dict(
            name=f"Viterbi-Viterbi, {r.size} symbols",
            ref=lambda: ocr.viterbi(r.reshape(-1, 1), 35, 4),
            ours=lambda: comrs.viterbi_viterbi(r, n=35, m=4),
            repeats=3,
        )
    )
    pil = np.arange(0, r.size, 64)
    tau = 1 / (2 * np.pi * 10e6)
    out.append(
        dict(
            name=f"decision-directed PLL, {r.size} symbols",
            ref=lambda: ocr.ddpll(
                r.reshape(-1, 1), 1 / 32e9, 0.1, tau, tau, c16.points, s.reshape(-1, 1), pil
            ),
            ours=lambda: comrs.ddpll(r, c16, fs=32e9, kv=0.1, tau1=tau, tau2=tau, pilot_indices=pil, pilot_symbols=s[pil]),
            repeats=3,
        )
    )

    # clock recovery and adaptive equalisation at 2 sps
    g = comrs.pulse_shape("rrc", 2, n_taps=128, rolloff=0.5)
    w = comrs.pnorm(comrs.fir_filter(g, comrs.fir_filter(g, comrs.upsample(s, 2))))
    out.append(
        dict(
            name=f"Gardner clock recovery, {w.size} samples",
            ref=lambda: ock.gardnerClockRecovery(w, params(kp=1e-3, ki=1e-6, isNyquist=True)),
            ours=lambda: comrs.gardner_clock_recovery(w)[0],
            repeats=3,
            tol=1e-4,
        )
    )
    pe = params(
        nTaps=15,
        SpS=2,
        mu=[1e-3],
        alg=["cma"],
        M=16,
        constType="qam",
        numIter=1,
        prgsBar=False,
        domain="time",
        prec=np.complex128,
    )
    out.append(
        dict(
            name=f"adaptive equaliser (CMA, 15 taps), {s.size} symbols",
            ref=lambda: oeq.mimoAdaptEqualizer(w, pe),
            ours=lambda: comrs.adaptive_equalizer(w, c16, algorithm="cma", n_taps=15, sps=2, mu=1e-3)[0],
            repeats=1,
            check=False,
        )
    )

    # metrics
    c64 = comrs.Constellation("qam", 64).normalized()
    _, s64 = comrs.symbols(c64, 2**20 // k, rng)
    r64 = comrs.awgn(s64, 20.0, rng)
    out.append(
        dict(
            name=f"BER, SER and SNR (fastBERcalc), 64-QAM, {s64.size} symbols",
            ref=lambda: om.fastBERcalc(r64, s64, 64, "qam"),
            ours=lambda: comrs.fast_ber_calc(r64, s64, c64),
            repeats=3,
            check=False,
        )
    )
    r16 = comrs.awgn(s, 10.0, rng)
    out.append(
        dict(
            name=f"GMI (monteCarloGMI), 16-QAM, {s.size} symbols",
            ref=lambda: om.monteCarloGMI(r16, s, 16, "qam"),
            ours=lambda: comrs.monte_carlo_gmi(r16, s, c16),
            repeats=3,
            check=False,
        )
    )
    return out


def campaign(quick: bool) -> dict:
    """BER of 16-QAM over AWGN at 8 values of Eb/N0, 100 blocks of 4096 symbols each."""
    trials = 25 if quick else 100
    ebn0 = list(range(0, 16, 2))
    n = 4096

    def opticommpy():
        rng = np.random.default_rng(1)
        out = []
        for e in ebn0:
            err = 0
            for _ in range(trials):
                bits = rng.integers(0, 2, 4 * n)
                # demodulateGray decides on the unnormalised grid; awgn measures the power
                x = omod.modulateGray(bits, 16, "qam")
                snr = e + 10 * np.log10(4)
                y = och.awgn(x, params(SNR=snr, Fs=1.0, B=1.0))
                err += int(np.sum(omod.demodulateGray(y, 16, "qam") != bits))
            out.append(err / (4 * n * trials))
        return out

    spec = {
        "name": "bench",
        "grid": {"ebn0_db": ebn0},
        "link": [
            {"type": "source", "constellation": "16-QAM", "n_symbols": n},
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
        ],
        "metrics": {"report": ["ber"], "scale": "none"},
        "stop": {"max_trials": trials},
        "batch": trials,
    }
    t_ref = best_time(opticommpy, 1)
    t1 = best_time(lambda: comrs.run_spec(spec, threads=1), 1)
    tn = best_time(lambda: comrs.run_spec(spec, threads=0), 1)
    ber_ref = opticommpy()
    ber = [r["metrics"]["ber"]["value"] for r in comrs.run_spec(spec)]
    ok = all(abs(a / b - 1) < 0.1 for a, b in zip(ber, ber_ref, strict=False) if b > 1e-4)
    return dict(
        name=f"Monte Carlo BER campaign: 16-QAM, 8 points x {trials} blocks of {n} symbols",
        ref=t_ref,
        ours=t1,
        ours_all=tn,
        agree=ok,
    )


def machine() -> dict:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        cpu = ""
    import numba
    import optic

    return dict(
        cpu=cpu or platform.processor(),
        cores=os.cpu_count(),
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        opticommpy=getattr(optic, "__version__", "0.11"),
        comrs=comrs.__version__,
        system=f"{platform.system()} {platform.release()}",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="smaller inputs (a quick check)")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results.md"))
    args = ap.parse_args()
    rows = []
    for case in cases(args.quick):
        a = case["ref"]()
        b = case["ours"]()
        agree = True
        if case.get("check", True):
            a = a[0] if isinstance(a, tuple) else a
            agree = rel_err(b, a) < case.get("tol", 1e-6)
        t_ref = best_time(case["ref"], case["repeats"])
        t = best_time(case["ours"], case["repeats"])
        rows.append(
            dict(name=case["name"], ref=t_ref, ours=t, agree=agree, checked=case.get("check", True))
        )
        print(
            f"{case['name']}: OptiCommPy {t_ref * 1e3:.1f} ms, comrs {t * 1e3:.1f} ms, x{t_ref / t:.1f}",
            flush=True,
        )
    camp = campaign(args.quick)
    print(
        f"{camp['name']}: OptiCommPy {camp['ref']:.2f} s, comrs {camp['ours']:.2f} s (1 thread), {camp['ours_all']:.2f} s (all cores)"
    )
    info = machine()
    lines = [
        "# comrs against OptiCommPy",
        "",
        f"Machine: {info['cpu']}, {info['cores']} cores, {info['system']}; Python {info['python']}, NumPy {info['numpy']}, "
        f"numba {info['numba']}, OptiCommPy {info['opticommpy']}, comrs {info['comrs']}.",
        "",
        "Best of a few runs after a warm-up (numba compiled), one thread for comrs; generated by",
        "`python benchmarks/bench_opticommpy.py`" + (" --quick" if args.quick else "") + ".",
        "",
        "| Operation | OptiCommPy | comrs | Speed-up | Same result |",
        "|---|---:|---:|---:|:---:|",
    ]
    fmt = lambda t: f"{t * 1e3:.1f} ms" if t < 1 else f"{t:.2f} s"  # noqa: E731
    for r in rows:
        same = ("yes" if r["agree"] else "no") if r["checked"] else "oracle tests"
        lines.append(
            f"| {r['name']} | {fmt(r['ref'])} | {fmt(r['ours'])} | {r['ref'] / r['ours']:.1f} | {same} |"
        )
    lines.append(
        f"| {camp['name']}, one thread | {fmt(camp['ref'])} | {fmt(camp['ours'])} | {camp['ref'] / camp['ours']:.1f} | {'yes' if camp['agree'] else 'no'} |"
    )
    lines.append(
        f"| same campaign, {info['cores']} cores | {fmt(camp['ref'])} | {fmt(camp['ours_all'])} | {camp['ref'] / camp['ours_all']:.1f} | {'yes' if camp['agree'] else 'no'} |"
    )
    lines += [
        "",
        '"Same result" compares the outputs to 1e-6 relative (1e-4 for the single-precision split step and the',
        'Gardner loop, which OptiCommPy computes in complex64); "oracle tests" marks the cases compared value by',
        "value in tests/test_oracle.py instead (the equaliser runs here in the time domain and double precision",
        "on both sides for a like-for-like comparison). The campaign compares the BER of the two implementations.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    args.out.with_suffix(".json").write_text(
        json.dumps(dict(machine=info, cases=rows, campaign=camp), indent=2)
    )
    print(f"written {args.out}")


if __name__ == "__main__":
    main()

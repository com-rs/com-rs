"""Scenario 5: single-polarisation split-step Fourier propagation of 32 GBd 16-QAM over
10 x 80 km of standard single-mode fibre, a fixed step of 1 km (80 steps per span),
noiseless amplification at the end of every span (gain e^{alpha L}).

The input field is the same for every library: 16-QAM symbols of comrs (seed 1),
shaped by a root-raised cosine (roll-off 0.1, 4 samples per symbol, 128 GS/s) and
scaled to 4 dBm; it is drawn again, identically, by every process. Every library
propagates it with its own split-step solver (symmetric steps, nonlinear phase over
the whole step in all three); run_all.py compares the output fields (NMSE) with that
of comrs in double precision and with a step-converged reference (comrs, 0.05 km).

Sionna's equation is the complex conjugate of the one of comrs and OptiCommPy
(dE/dz = ... + j beta_2/2 d^2E/dt^2 - j gamma |E|^2 E): its field is compared in their
convention, conj(sionna(conj(x))), and the NMSE of the plain output is reported too.
Sionna propagates one span per call (SSFM) and amplifies with its EDFA of noise figure
0 (no noise). OptiCommPy's ssfm is NumPy only (no numba kernel): its two variants run
the same code. Timing: one propagation over the 10 spans, median of 5 runs after a
warm-up run.

    python s5_ssfm.py --lib opticommpy --threads 1 [--quick] --out results/_work/s5.json
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import common
import numpy as np

SCENARIO = "s5_ssfm"
TITLE = "Split-step Fourier propagation, 16-QAM, 32 GBd, 10 x 80 km"
CASES = {
    "ssfm": {
        "comrs": True,
        "komm": "no fibre model",
        "opticommpy": True,
        "opticommpy-nojit": True,
        "sionna": True,
    }
}
ALPHA_DB_KM, D_PS_NM_KM, GAMMA, FC = 0.2, 17.0, 1.3, 193.1e12
SPAN_KM, N_SPANS, STEP_KM = 80.0, 10, 1.0
REF_STEP_KM = 0.05
SYMBOL_RATE, SPS = 32e9, 4
FS = SYMBOL_RATE * SPS
POWER_DBM = 4.0

PARAMETERS = {
    "fibre": "alpha = 0.2 dB/km, D = 17 ps/nm/km, gamma = 1.3 /W/km, fc = 193.1 THz",
    "link": "10 x 80 km, ideal (noiseless) amplification after every span",
    "step": "1 km (800 steps); reference: comrs at 0.05 km (16000 steps)",
    "signal": "16-QAM, 32 GBd, RRC roll-off 0.1, 4 samples per symbol, 4 dBm; 2^14 symbols (quick: 2^11)",
    "timing": "one propagation over the 10 spans",
}


def n_symbols(quick: bool) -> int:
    return 2**11 if quick else 2**14


def input_field(quick: bool) -> np.ndarray:
    import comrs

    rng = comrs.Rng(1)
    c = comrs.Constellation("qam", 16).normalized()
    _, s = comrs.symbols(c, n_symbols(quick), rng)
    h = comrs.pulse_shape("rrc", SPS, n_taps=1025, rolloff=0.1)
    x = comrs.pnorm(comrs.fir_filter(h, comrs.upsample(s, SPS)))
    return x * math.sqrt(comrs.dbm2w(POWER_DBM))


def beta2_ps2_km() -> float:
    c = 299_792_458.0
    lam = c / FC
    return -(D_PS_NM_KM * 1e-6) * lam**2 / (2 * math.pi * c) * 1e24 * 1e3  # s^2/m -> ps^2/km


def prop_comrs(precision: str, step_km: float = STEP_KM):
    import comrs

    f = comrs.FiberParams(
        alpha_db_km=ALPHA_DB_KM,
        d_ps_nm_km=D_PS_NM_KM,
        gamma_w_km=GAMMA,
        span_km=SPAN_KM,
        n_spans=N_SPANS,
        fc_hz=FC,
    )
    link = comrs.Ssfm(f, step_km=step_km, amplification="ideal")
    return lambda x: link.propagate(x, FS, precision=precision)


def prop_opticommpy():
    from optic.models.channels import ssfm
    from optic.utils import parameters

    p = parameters()
    p.Ltotal, p.Lspan, p.hz = SPAN_KM * N_SPANS, SPAN_KM, STEP_KM
    p.alpha, p.D, p.gamma, p.Fc, p.Fs = ALPHA_DB_KM, D_PS_NM_KM, GAMMA, FC, FS
    p.amp, p.prgsBar, p.prec = "ideal", False, np.complex128
    return lambda x: ssfm(x, p)


def prop_sionna(precision: str):
    import torch
    from sionna.phy.channel.optical import EDFA, SSFM

    alpha = ALPHA_DB_KM / (10 * math.log10(math.e))  # 1/km
    span = SSFM(
        alpha=alpha,
        beta_2=beta2_ps2_km(),
        f_c=FC,
        gamma=GAMMA,
        length=SPAN_KM,
        n_ssfm=round(SPAN_KM / STEP_KM),
        sample_duration=1e12 / FS,  # ps
        t_norm=1e-12,
        with_amplification=False,
        precision=precision,
    )
    amp = EDFA(g=math.exp(alpha * SPAN_KM), f=0.0, f_c=FC, dt=1 / FS, precision=precision)
    cdtype = torch.complex64 if precision == "single" else torch.complex128

    def run(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            u = torch.from_numpy(x).to(cdtype)
            for _ in range(N_SPANS):
                u = amp(span(u))
            return u.numpy()

    return run


def run_lib(args, todo: list[str]) -> list[dict]:
    threading = common.setup_torch(args.threads) if args.lib == "sionna" else {}
    x = input_field(args.quick)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.lib == "comrs":
        variants = {"f64": (prop_comrs("f64"), "float64"), "f32": (prop_comrs("f32"), "float32")}
    elif args.lib == "sionna":
        variants = {
            "single": (prop_sionna("single"), "float32 (precision = 'single', the default)"),
            "double": (prop_sionna("double"), "float64 (precision = 'double')"),
        }
    else:
        variants = {"f64": (prop_opticommpy(), "float64 (prec = np.complex128)")}
    out = []
    for name, (prop, precision) in variants.items():
        res = {
            "case": "ssfm",
            "variant": name,
            "precision": precision,
            "samples": int(x.size),
            **threading,
        }
        if args.lib == "sionna":
            fwd = lambda prop=prop: np.conj(prop(np.conj(x)))  # noqa: E731
        else:
            fwd = lambda prop=prop: prop(x)  # noqa: E731
        t0 = time.perf_counter()
        y = fwd()
        res["first_call_s"] = time.perf_counter() - t0
        if not args.timing_only:
            np.save(work / f"{SCENARIO}_{args.lib}_{name}.npy", np.asarray(y, dtype=np.complex128))
            if args.lib == "sionna":
                np.save(
                    work / f"{SCENARIO}_{args.lib}_{name}_plain.npy",
                    np.asarray(prop(x), dtype=np.complex128),
                )
        res["timing"] = common.time_it(fwd)
        res["timing"].update(
            units=1,
            unit="propagation",
            workload=f"{x.size} samples, {N_SPANS} x {SPAN_KM:g} km, {STEP_KM:g} km steps",
        )
        out.append(res)
    if args.lib == "comrs" and not args.timing_only:
        t0 = time.perf_counter()
        np.save(work / f"{SCENARIO}_reference.npy", prop_comrs("f64", REF_STEP_KM)(x))
        out.append(
            {
                "case": "ssfm",
                "variant": "reference",
                "precision": "float64",
                "step_km": REF_STEP_KM,
                "wall_s": time.perf_counter() - t0,
            }
        )
    return out


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


def postprocess(merged: dict, work: Path) -> dict:
    """The NMSE of every output field against comrs (f64, 1 km) and the reference."""
    fields = {}
    for path in sorted(work.glob(f"{SCENARIO}_*.npy")):
        fields[path.stem.removeprefix(f"{SCENARIO}_")] = np.load(path)
    base, ref = fields.get("comrs_f64"), fields.get("reference")
    rows = []
    for name, y in fields.items():
        if name == "reference":
            continue
        rows.append(
            {
                "field": name,
                "nmse_vs_comrs_f64": nmse(y, base) if base is not None else None,
                "nmse_vs_reference": nmse(y, ref) if ref is not None else None,
            }
        )
    return {"nmse": rows, "reference_step_km": REF_STEP_KM}


RUNNERS = {"comrs": run_lib, "opticommpy": run_lib, "opticommpy-nojit": run_lib, "sionna": run_lib}

if __name__ == "__main__":
    common.main(SCENARIO, CASES, RUNNERS)

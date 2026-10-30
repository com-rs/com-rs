#!/usr/bin/env python3
"""Record the error rates of AFF3CT for the codes that com-rs validates against it.

AFF3CT (A. Cassagne et al., "AFF3CT: A Fast Forward Error Correction Toolbox!",
*SoftwareX* 10, 2019, MIT licence) simulates every configuration of ``CONFIGS`` (BPSK
over AWGN); this script runs it, parses its result table and writes
``validation/aff3ct/<name>.json``: the provenance (the tool, its version, the exact
command line, the date), the parameters AFF3CT reports (those of the machine removed)
and, for every Eb/N0, the frames, bit and frame errors, so that confidence intervals can
be computed. ``validation/tests/aff3ct.rs`` simulates the same configurations with
com-rs and compares the two. The fixtures are outputs of AFF3CT runs, not its code nor
its data files; the LDPC matrices and the polar frozen sets given to AFF3CT
(``validation/aff3ct/matrices/``, ``validation/aff3ct/frozen/``) are written by com-rs
(``COMRS_RECORD_AFF3CT=1 cargo test -p com-validation --test aff3ct``)::

    python3 tools/record_aff3ct.py --aff3ct /path/to/aff3ct          # or AFF3CT=...
    python3 tools/record_aff3ct.py --only rs_n255_k239_t8 --force    # re-run one

The command lines follow those of the AFF3CT reference database
(github.com/aff3ct/error_rate_references), which is only read for them: none of its
files nor numbers are copied.

Reproducibility: AFF3CT is not reproducible run to run, even with a fixed seed (``-S``)
and one thread: streampu (``Sequence``) collects the modules of the chain in a
``std::set`` of pointers and AFF3CT seeds them in that order, so the source and the
channel exchange their seeds depending on the heap addresses. A fixture whose command
line and AFF3CT version are unchanged is therefore kept as it is (no diff); ``--force``
runs it again, and the date of a re-run is kept only if the data are the same. The output
is deterministic otherwise: sorted keys, fixed layout.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from _common import ROOT

OUT = ROOT / "validation" / "aff3ct"
SEED = 1
FRAME_ERRORS = 200
# the reports of the machine, not of the configuration
MACHINE_KEYS = {
    ("Simulation", "Date (UTC)"),
    ("Simulation", "Git version"),
    ("Simulation", "Multi-threading (t)"),
    ("Terminal", "Frequency (ms)"),
}
CASSAGNE = (
    "A. Cassagne, O. Hartmann, M. Léonardon, K. He, C. Leroux, R. Tajan, O. Aumage, "
    'D. Barthou, T. Tonnellier, V. Pignoly, B. Le Gal and C. Jégo, "AFF3CT: A Fast '
    'Forward Error Correction Toolbox!," SoftwareX 10, 100345, 2019'
)


@dataclass(frozen=True)
class Config:
    """One AFF3CT simulation: its arguments (without the noise range and the stopping
    rule), the Eb/N0 range ``(min, max, step)`` in dB, and what it validates."""

    name: str
    description: str
    reference: str
    args: list[str]
    ebn0: tuple[float, float, float]
    extra: list[str] = field(default_factory=list)


LIN_COSTELLO = "S. Lin and D. J. Costello, Error Control Coding, 2nd ed., Pearson, 2004"
TS36212 = "3GPP TS 36.212, E-UTRA; Multiplexing and channel coding, Sec. 5.1.3.2 and Table 5.1.3-3"
HU2005 = (
    'X.-Y. Hu, E. Eleftheriou and D. M. Arnold, "Regular and irregular progressive '
    'edge-growth Tanner graphs," IEEE Trans. Inf. Theory 51(1), 2005'
)
ARIKAN = (
    'E. Arıkan, "Channel polarization," IEEE Trans. Inf. Theory 55(7), 2009; '
    'I. Tal and A. Vardy, "List decoding of polar codes," IEEE Trans. Inf. Theory 61(5), 2015'
)


def _rs(n: int, k: int, t: int, ebn0: tuple[float, float, float]) -> Config:
    return Config(
        f"rs_n{n}_k{k}_t{t}",
        f"Reed–Solomon ({n}, {k}), t = {t}, algebraic (Berlekamp–Massey) hard decoding",
        f"I. S. Reed and G. Solomon, J. SIAM 8(2), 1960; {LIN_COSTELLO}, ch. 7",
        ["-C", "RS", "-N", str(n), "-K", str(k), "-T", str(t)],
        ebn0,
    )


def _bch(n: int, k: int, t: int, ebn0: tuple[float, float, float]) -> Config:
    return Config(
        f"bch_n{n}_k{k}_t{t}",
        f"binary BCH ({n}, {k}), t = {t}, algebraic (Berlekamp–Massey) hard decoding",
        f"{LIN_COSTELLO}, ch. 6",
        ["-C", "BCH", "-N", str(n), "-K", str(k), "-T", str(t)],
        ebn0,
    )


def _ldpc(
    name: str, desc: str, matrix: str, decoder: list[str], ebn0: tuple[float, float, float]
) -> Config:
    n, k = (int(v) for v in re.findall(r"\d+", matrix)[-2:])
    return Config(
        name,
        desc,
        HU2005,
        [
            "-C",
            "LDPC",
            "-N",
            str(n),
            "-K",
            str(k),
            "--dec-h-path",
            f"validation/aff3ct/matrices/{matrix}",
            "--src-type",
            "AZCW",
            "--enc-type",
            "AZCW",
            "--sim-coded",
            *decoder,
        ],
        ebn0,
    )


def _polar(
    n: int,
    k: int,
    decoder: list[str],
    crc: list[str],
    name: str,
    desc: str,
    ebn0: tuple[float, float, float],
) -> Config:
    a = k + (int(crc[3]) if crc else 0)
    return Config(
        name,
        desc,
        ARIKAN,
        [
            "-C",
            "POLAR",
            "-N",
            str(n),
            "-K",
            str(a),
            "--enc-no-sys",
            "--enc-fb-gen-method",
            "FILE",
            "--enc-fb-awgn-path",
            f"validation/aff3ct/frozen/polar_n{n}_k{a}.txt",
            "--dec-implem",
            "NAIVE",
            *decoder,
            *crc,
        ],
        ebn0,
    )


def _turbo(k: int, iterations: int, max_star: str, ebn0: tuple[float, float, float]) -> Config:
    kind = {"MAX": "max-log-MAP", "MAXS": "log-MAP"}[max_star]
    return Config(
        f"turbo_lte_k{k}_{kind.replace('-', '').lower()}_i{iterations}",
        f"LTE turbo code, K = {k} (QPP interleaver), {iterations} iterations of {kind} BCJR",
        f"C. Berrou, A. Glavieux and P. Thitimajshima, ICC 1993; {TS36212}",
        [
            "-C",
            "TURBO",
            "-K",
            str(k),
            "--enc-sub-std",
            "LTE",
            "--itl-type",
            "LTE",
            "-i",
            str(iterations),
            "--dec-sub-type",
            "BCJR",
            "--dec-sub-implem",
            "STD",
            "--dec-sub-max",
            max_star,
        ],
        ebn0,
    )


CONFIGS: list[Config] = [
    # Reed–Solomon: AFF3CT's field polynomials (Galois::select_polynomial) and roots
    # α¹ … α²ᵗ (RS_polynomial_generator)
    _rs(15, 11, 2, (3.0, 7.5, 0.5)),
    _rs(63, 55, 4, (4.0, 7.25, 0.25)),
    _rs(255, 239, 8, (5.0, 7.25, 0.25)),
    # binary BCH
    _bch(63, 45, 3, (3.0, 7.0, 0.5)),
    _bch(127, 92, 5, (3.0, 6.5, 0.5)),
    _bch(255, 223, 4, (4.0, 7.0, 0.5)),
    # the (133, 171) convolutional code of constraint length 7, terminated
    Config(
        "conv_k7_k64_viterbi",
        "convolutional code (133, 171), K = 7, 64 information bits, terminated, soft Viterbi",
        'A. J. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum '
        'decoding algorithm," IEEE Trans. Inf. Theory 13(2), 1967',
        ["-C", "CONV", "-K", "64", "--enc-poly", "{0133,0171}", "-D", "VITERBI"],
        (0.0, 4.5, 0.5),
    ),
    # LDPC: the matrices are PEG constructions of com-rs, read by AFF3CT
    _ldpc(
        "ldpc_peg_n576_k288_flooding_spa_i50",
        "PEG (576, 288) LDPC, degree 3, flooding sum-product, 50 iterations",
        "peg_n576_k288.alist",
        ["-D", "BP_FLOODING", "--dec-implem", "SPA", "-i", "50"],
        (1.0, 3.0, 0.25),
    ),
    _ldpc(
        "ldpc_peg_n576_k288_layered_nms075_i20",
        "PEG (576, 288) LDPC, degree 3, layered normalised min-sum (α = 0.75), 20 iterations",
        "peg_n576_k288.alist",
        ["-D", "BP_HORIZONTAL_LAYERED", "--dec-implem", "NMS", "--dec-norm", "0.75", "-i", "20"],
        (1.0, 3.0, 0.25),
    ),
    _ldpc(
        "ldpc_peg_n576_k288_flooding_oms015_i20",
        "PEG (576, 288) LDPC, degree 3, flooding offset min-sum (β = 0.15), 20 iterations",
        "peg_n576_k288.alist",
        ["-D", "BP_FLOODING", "--dec-implem", "OMS", "--dec-off", "0.15", "-i", "20"],
        (1.0, 3.5, 0.25),
    ),
    _ldpc(
        "ldpc_peg_n648_k540_layered_spa_i20",
        "PEG (648, 540) LDPC, degree 3, layered sum-product, 20 iterations",
        "peg_n648_k540.alist",
        ["-D", "BP_HORIZONTAL_LAYERED", "--dec-implem", "SPA", "-i", "20"],
        (2.5, 4.75, 0.25),
    ),
    Config(
        "ldpc_dvbs2_short_r1_2_layered_spa_i50",
        "DVB-S2 LDPC, short frame, rate 1/2 (16200, 7200), layered sum-product, 50 iterations",
        "ETSI EN 302 307-1 V1.4.1, Sec. 5.3.2 and Table B.1 (short frame, rate 1/2)",
        [
            "-C",
            "LDPC",
            "-N",
            "16200",
            "-K",
            "7200",
            "--enc-type",
            "LDPC_DVBS2",
            "--sim-coded",
            "-D",
            "BP_HORIZONTAL_LAYERED",
            "--dec-implem",
            "SPA",
            "-i",
            "50",
        ],
        (0.5, 0.9, 0.1),
    ),
    # polar codes: the frozen sets of com-rs (Gaussian approximation), read from a file
    _polar(
        256,
        128,
        ["-D", "SC"],
        [],
        "polar_n256_k128_sc",
        "polar (256, 128), SC (min-sum f), frozen set of com-rs",
        (1.0, 4.5, 0.5),
    ),
    _polar(
        1024,
        512,
        ["-D", "SC"],
        [],
        "polar_n1024_k512_sc",
        "polar (1024, 512), SC (min-sum f), frozen set of com-rs",
        (1.5, 3.5, 0.25),
    ),
    _polar(
        256,
        128,
        ["-D", "SCL", "-L", "8"],
        ["--crc-poly", "11-5G", "--crc-size", "11"],
        "polar_n256_k128_cascl_l8_crc11",
        "polar (256, 128) + CRC11 of 5G NR, CA-SCL, L = 8, frozen set of com-rs",
        (0.5, 3.5, 0.5),
    ),
    # LTE turbo codes
    _turbo(40, 8, "MAXS", (0.0, 4.5, 0.5)),
    _turbo(1024, 6, "MAX", (0.5, 1.5, 0.125)),
]


# ----------------------------------------------------------------------------- running


def version(binary: str) -> tuple[str, str]:
    """The version tag and the build description printed by ``aff3ct --version``."""
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True).stdout
    first = out.splitlines()[0]
    m = re.match(r"\S+\s+\((?P<build>[^)]*)\)\s+(?P<tag>\S+)", first)
    if not m:
        raise RuntimeError(f"cannot read the version of AFF3CT from {first!r}")
    return m["tag"], m["build"]


def arguments(cfg: Config, frame_errors: int = FRAME_ERRORS) -> list[str]:
    """The full argument list of a configuration (noise range, stopping rule, seed)."""
    lo, hi, step = cfg.ebn0
    return [
        *cfg.args,
        "-m",
        f"{lo:g}",
        "-M",
        f"{hi + step / 2:g}",
        "-s",
        f"{step:g}",
        "-e",
        str(frame_errors),
        "-S",
        str(SEED),
        "-p",
        "64",
        *cfg.extra,
    ]


def run(binary: str, cfg: Config, threads: int, frame_errors: int = FRAME_ERRORS) -> str:
    cmd = [
        binary,
        *arguments(cfg, frame_errors),
        "-t",
        str(threads),
        "--full-legend",
        "--no-colors",
        "--ter-freq",
        "0",
    ]
    print("running", shlex.join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"{cfg.name}: AFF3CT failed\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


# ----------------------------------------------------------------------------- parsing

SECTION = re.compile(r"^# \* (?P<name>.+?) -+$")
SUBSECTION = re.compile(r"^#    (?P<name>[A-Za-z].*?) -+$")
ENTRY = re.compile(r"^#    (?P<sub>   )?\*\* (?P<key>.+?)\s+= (?P<value>.*)$")


def parse(text: str) -> tuple[dict, list[dict]]:
    """The parameters (``{section: {key: value, subsection: {...}}}``) and the rows of the
    result table of an AFF3CT run."""
    params: dict[str, dict] = {}
    section: dict | None = None
    current: dict | None = None
    rows = []
    for line in text.splitlines():
        if m := SECTION.match(line):
            section = current = params.setdefault(m["name"], {})
            continue
        if (m := SUBSECTION.match(line)) and section is not None and "||" not in line:
            current = section.setdefault(m["name"], {})
            continue
        if (m := ENTRY.match(line)) and current is not None:
            current[m["key"]] = m["value"].strip()
            continue
        if line.startswith("#") or "||" not in line:
            continue
        cells = [c.strip() for c in line.replace("||", "|").split("|")]
        esn0, ebn0, frames, be, fe, ber, fer = cells[:7]
        for count in (frames, be, fe):
            if not count.isdigit():
                raise ValueError(
                    f"a count printed as {count!r}: lower the number of frames ({line!r})"
                )
        rows.append(
            {
                "esn0_db": float(esn0),
                "ebn0_db": float(ebn0),
                "frames": int(frames),
                "bit_errors": int(be),
                "frame_errors": int(fe),
                "ber": float(ber),
                "fer": float(fer),
            }
        )
    for sec, key in MACHINE_KEYS:
        params.get(sec, {}).pop(key, None)
    return params, rows


def bit_rate(params: dict) -> list[int]:
    """The rate AFF3CT converts Eb/N0 with, ``[numerator, denominator]``."""
    m = re.search(r"\((\d+)/(\d+)\)", params["Simulation"]["Bit rate"])
    if not m:
        raise ValueError(f"cannot read the bit rate {params['Simulation']['Bit rate']!r}")
    r = Fraction(int(m[1]), int(m[2]))
    return [r.numerator, r.denominator]


# ----------------------------------------------------------------------------- writing


def document(
    cfg: Config, tag: str, build: str, text: str, frame_errors: int = FRAME_ERRORS
) -> dict:
    params, rows = parse(text)
    if not rows:
        raise RuntimeError(f"{cfg.name}: no result in the output of AFF3CT\n{text}")
    bits = int(params["Monitor"]["Size (K)"])
    for r in rows:
        ber = r["bit_errors"] / (r["frames"] * bits)
        if abs(ber - r["ber"]) > 0.006 * r["ber"] + 1e-300:
            raise RuntimeError(
                f"{cfg.name}: BER {r['ber']} is not {r['bit_errors']}/({r['frames']}·{bits})"
            )
    return {
        "name": cfg.name,
        "description": cfg.description,
        "tool": "AFF3CT",
        "version": tag,
        "build": build,
        "command": "aff3ct " + shlex.join(arguments(cfg, frame_errors)),
        "references": [CASSAGNE, cfg.reference],
        "parameters": params,
        "bit_rate": bit_rate(params),
        "bits_per_frame": bits,
        "data": rows,
    }


def dumps(doc: dict) -> str:
    """The document indented, one line per point of the curve."""
    rows = doc["data"]
    head = json.dumps({**doc, "data": "\x00"}, sort_keys=True, indent=1, ensure_ascii=False)
    body = ",\n".join("  " + json.dumps(r, sort_keys=True) for r in rows)
    return head.replace('"\\u0000"', "[\n" + body + "\n ]") + "\n"


def record(
    cfg: Config,
    binary: str,
    threads: int,
    force: bool,
    out: Path = OUT,
    frame_errors: int = FRAME_ERRORS,
) -> str:
    path = out / f"{cfg.name}.json"
    tag, build = version(binary)
    command = "aff3ct " + shlex.join(arguments(cfg, frame_errors))
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if old and not force and old.get("command") == command and old.get("version") == tag:
        return "unchanged"
    doc = document(cfg, tag, build, run(binary, cfg, threads, frame_errors), frame_errors)
    same = old is not None and {k: v for k, v in old.items() if k != "recorded"} == doc
    doc["recorded"] = old["recorded"] if same else datetime.date.today().isoformat()
    text = dumps(doc)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return "written"
    return "same"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--aff3ct", default=os.environ.get("AFF3CT"), help="the AFF3CT binary (default: $AFF3CT)"
    )
    p.add_argument(
        "--only", nargs="*", default=None, help="the configurations to record (default: all)"
    )
    p.add_argument(
        "--force", action="store_true", help="run again the configurations already recorded"
    )
    p.add_argument("--threads", type=int, default=0, help="threads of AFF3CT (0: every core)")
    p.add_argument(
        "--list", action="store_true", help="print the configurations and their commands"
    )
    p.add_argument(
        "--out", type=Path, default=OUT, help="where to write (default: validation/aff3ct)"
    )
    p.add_argument(
        "--frame-errors",
        type=int,
        default=FRAME_ERRORS,
        help=f"frame errors per point (default {FRAME_ERRORS}; fewer for a trial run)",
    )
    a = p.parse_args()
    selected = [c for c in CONFIGS if a.only is None or c.name in a.only]
    if a.only is not None and len(selected) != len(set(a.only)):
        known = {c.name for c in CONFIGS}
        p.error(f"unknown configurations: {sorted(set(a.only) - known)}")
    if a.list:
        for c in selected:
            print(f"{c.name}\n    aff3ct {shlex.join(arguments(c, a.frame_errors))}")
        return 0
    if not a.aff3ct:
        p.error("give the AFF3CT binary with --aff3ct or the AFF3CT environment variable")
    for c in selected:
        status = record(c, a.aff3ct, a.threads, a.force, a.out.resolve(), a.frame_errors)
        print(f"{c.name}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

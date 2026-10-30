"""The highlights of com-rs — lightweight, fast, safe, validated — computed from the
repository and written on the front pages and in ``docs/python/why.rst``.

Every number comes from a file of the repository, never typed by hand:

- *lightweight*: the dependencies of the manifests (``Cargo.toml``, ``pyproject.toml``,
  ``Cargo.lock``) and the installed footprint of every library of the benchmark
  (``benchmarks/libraries/results/footprint.json``, from ``footprint.py``);
- *fast*: the recorded benchmark (``benchmarks/libraries/results/s*.json``);
- *safe*: the workspace lints, the error type and the tests of the code;
- *validated*: ``validation/literature.yaml``, ``validation/maturity.yaml`` and the
  cards of ``validation/cards/``.

    python tools/highlights.py           # README.md, docs/python/index.rst, docs/python/why.rst
    python tools/highlights.py --check   # fail if one of them is out of date (just check)

To refresh the numbers: ``just bench-libraries`` (the benchmark and the footprint, on a
quiet machine), then this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

import yaml
from _common import ROOT, write_or_check
from benchmarks_page import CASE_NAMES, RESULTS, SCENARIOS, case_results, load, runs_by

README = ROOT / "README.md"
INDEX = ROOT / "docs" / "python" / "index.rst"
PAGE = ROOT / "docs" / "python" / "why.rst"
FIX = "python tools/highlights.py"
COMRS = ["comrs", "comrs-spec", "comrs-rust"]
SCENARIO_NAMES = {
    "s1_awgn": "AWGN",
    "s2_rayleigh": "flat Rayleigh",
    "s3_mimo": "4 × 4 MIMO, 16-QAM",
    "s4_coding": "coding",
    "s5_ssfm": "fibre, 10 × 80 km",
}
OTHERS = {"komm": "komm", "opticommpy": "OptiCommPy", "sionna": "Sionna"}
FOOTPRINT_NAMES = {
    "comrs": "com-rs",
    "komm": "komm",
    "OptiCommPy": "OptiCommPy",
    "Sionna": "Sionna",
}


# --- the facts ---------------------------------------------------------------------


def rust_dependencies() -> list[str]:
    """The external crates of the library (the workspace dependencies but the
    benchmark harness)."""
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    deps = manifest["workspace"]["dependencies"]
    return sorted(
        n for n in deps if not n.startswith("com") and n not in {"criterion", "pyo3", "numpy"}
    )


def native_crates() -> list[str]:
    """The crates of ``Cargo.lock`` that bind a system library (``*-sys``)."""
    return sorted(
        set(re.findall(r'^name = "([^"]+-sys)"', (ROOT / "Cargo.lock").read_text(), re.MULTILINE))
    )


def python_dependencies() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return [re.match(r"[A-Za-z0-9._-]+", d).group(0) for d in project.get("dependencies", [])]


def unsafe_forbidden() -> bool:
    lints = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]["lints"][
        "rust"
    ]
    return lints.get("unsafe_code") == "forbid"


def count(pattern: str, files: list[Path]) -> int:
    rx = re.compile(pattern, re.MULTILINE)
    return sum(len(rx.findall(f.read_text(encoding="utf-8"))) for f in files)


def tests() -> dict[str, int]:
    crates = sorted((ROOT / "crates").rglob("*.rs"))
    validation = sorted((ROOT / "validation" / "tests").glob("*.rs"))
    python = sorted((ROOT / "tests").glob("test_*.py"))
    return {
        "rust": count(r"^\s*#\[test\]", crates),
        "validation": count(r"^\s*#\[test\]", validation),
        "python": count(r"^def test_", python),
    }


def validation() -> dict:
    lit = yaml.safe_load((ROOT / "validation" / "literature.yaml").read_text(encoding="utf-8"))
    mat = yaml.safe_load((ROOT / "validation" / "maturity.yaml").read_text(encoding="utf-8"))
    grades = {g: sum(1 for m in mat if m.get("level") == g) for g in "ABC"}
    cards = sorted((ROOT / "validation" / "cards").glob("*.yaml"))
    badges = {b: sum(1 for e in lit if e["badge"] == b) for b in ("green", "orange", "red")}
    remarks = sum(len(e.get("remarks", [])) for e in lit)
    return {
        "documents": len(lit),
        "badges": badges,
        "remarks": remarks,
        "grades": grades,
        "cards": len(cards),
    }


def footprint() -> dict | None:
    path = RESULTS / "footprint.json"
    return json.loads(path.read_text()) if path.exists() else None


def per_unit(c: dict) -> float:
    return c["timing"]["median_s"] / c["timing"]["units"]


def speed(data: dict[str, dict]) -> list[dict]:
    """For every case timed by com-rs and another library: the best time of com-rs and
    the time of every other library, on one thread and on all threads."""
    rows = []
    for s in SCENARIOS:
        d = data.get(s)
        if not d:
            continue
        runs = runs_by(d)
        for case in d["support"]:
            row = {"scenario": s, "case": case, "unit": None, "comrs": {}, "others": {}}
            for t in ("1", "all"):
                best = None
                for lib in COMRS:
                    for c in case_results(runs.get((lib, t)), case):
                        if "timing" in c and c.get("variant", "f64") in ("", "f64"):
                            row["unit"] = c["timing"]["unit"]
                            best = per_unit(c) if best is None else min(best, per_unit(c))
                if best is not None:
                    row["comrs"][t] = best
                for lib in OTHERS:
                    for c in case_results(runs.get((lib, t)), case):
                        if "timing" in c and c.get("variant", "") in (
                            "",
                            "f64",
                            "float64",
                            "f32",
                            "float32",
                        ):
                            prev = row["others"].setdefault(lib, {}).get(t)
                            row["others"][lib][t] = (
                                per_unit(c) if prev is None else min(prev, per_unit(c))
                            )
            if row["comrs"] and row["others"]:
                rows.append(row)
    return rows


def ratio_range(rows: list[dict], lib: str, t: str) -> tuple[float, float] | None:
    r = [
        row["others"][lib][t] / row["comrs"][t]
        for row in rows
        if t in row["comrs"] and t in row["others"].get(lib, {})
    ]
    return (min(r), max(r)) if r else None


def times(x: float) -> str:
    """A ratio as a factor: ``3.4×`` faster, or ``0.8×`` (slower)."""
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"


def factor(x: float) -> str:
    """A ratio as a number: ``33``, ``2.1``."""
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def megabytes(n: int) -> str:
    return f"{n / 1e6:.0f} MB" if n >= 1e7 else f"{n / 1e6:.1f} MB"


# --- the texts ---------------------------------------------------------------------


def facts() -> dict:
    data = load()
    rows = speed(data)
    return {
        "rust": rust_dependencies(),
        "native": native_crates(),
        "python": python_dependencies(),
        "unsafe_forbidden": unsafe_forbidden(),
        "tests": tests(),
        "validation": validation(),
        "footprint": footprint(),
        "rows": rows,
        "machine": next(iter(data.values()))["machine"] if data else None,
        "commit": next(iter(data.values()))["git"]["commit"][:12] if data else None,
    }


def bullets(f: dict) -> list[tuple[str, str]]:
    """(title, text) of the four highlights, plain text with ``code`` marks."""
    fp = f["footprint"]
    if fp:
        libs = fp["libraries"]
        others = ", ".join(
            f"{FOOTPRINT_NAMES[k]} {megabytes(v['total_bytes'])}"
            for k, v in libs.items()
            if k != "comrs"
        )
        light = (
            f"one run-time dependency in Python ({', '.join(f['python'])}), "
            f"{len(f['rust'])} direct crates in Rust and no C, C++ or Fortran library to build; "
            f"{megabytes(libs['comrs']['total_bytes'])} installed with its dependencies "
            f"(the benchmark's environment: {others})"
        )
    else:
        light = f"one run-time dependency in Python ({', '.join(f['python'])}), {len(f['rust'])} crates in Rust"
    lo = {lib: ratio_range(f["rows"], lib, "1") for lib in OTHERS}
    parts = [
        f"{factor(a)} to {factor(b)} times as long with {OTHERS[lib]}"
        for lib, r in lo.items()
        if r
        for a, b in [r]
    ]
    fast = (
        "compiled Rust, Monte Carlo runs on every core; on one CPU thread, the links of the "
        f"benchmark take {', '.join(parts[:-1])} and {parts[-1]} as with com-rs"
    )
    t = f["tests"]
    safe = (
        ("`unsafe` code forbidden in every crate, " if f["unsafe_forbidden"] else "")
        + "invalid inputs are errors (one error type), results independent of the thread "
        f"count; {t['rust']} unit tests, {t['validation']} validation tests, {t['python']} Python tests"
    )
    v = f["validation"]
    valid = (
        f"compared with {v['documents']} papers, standards and reference tools "
        f"({v['badges']['green']} fully reproduced, {v['badges']['orange']} partly), "
        f"{v['remarks']} inconsistencies found in them; {v['cards']} validation cards rerun "
        "a comparison without the paper"
    )
    return [("Lightweight", light), ("Fast", fast), ("Safe", safe), ("Validated", valid)]


def md(text: str) -> str:
    return text


def rst(text: str) -> str:
    return re.sub(r"`([^`]+)`", r"``\1``", text)


def readme_block(f: dict) -> str:
    lines = ["<!-- highlights: generated by tools/highlights.py, do not edit -->"]
    for title, text in bullets(f):
        lines.append(f"- **{title}**: {md(text)}.")
    lines.append(
        "\nThe numbers are recomputed by `tools/highlights.py` from the recorded benchmark "
        "and the validation files; the details are on the page "
        "[Why com-rs](https://com-rs.github.io/com-rs/why.html)."
    )
    return "\n".join(lines) + "\n"


def rst_bullets(f: dict) -> list[str]:
    return [f"- **{title}**: {rst(text)}." for title, text in bullets(f)]


def index_block(f: dict) -> str:
    lines = [".. highlights: generated by tools/highlights.py, do not edit", "", *rst_bullets(f)]
    lines += ["", "The details, and how these numbers are produced: :doc:`why`.", ""]
    return "\n".join(lines)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = [".. list-table::", "   :header-rows: 1", ""]
    for r in [header, *rows]:
        out.append(f"   * - {r[0]}")
        out += [f"     - {c}" for c in r[1:]]
    return [*out, ""]


def page(f: dict) -> str:
    v, t, fp = f["validation"], f["tests"], f["footprint"]
    out = [
        "Why com-rs",
        "==========",
        "",
        ".. generated by tools/highlights.py from the repository: do not edit",
        "",
        "Four properties, each measured from the files of the repository rather than",
        "claimed: the numbers below are recomputed by ``tools/highlights.py`` and checked by",
        "``just check`` (see *Updating* at the end).",
        "",
        *rst_bullets(f),
        "",
        "Lightweight",
        "-----------",
        "",
        f"The Python package needs {', '.join(f'``{p}``' for p in f['python'])} only at run time "
        "(matplotlib is optional, for the plots). The Rust library uses "
        f"{len(f['rust'])} crates of the Rust ecosystem: "
        + ", ".join(f"``{c}``" for c in f["rust"])
        + ". It links no BLAS, LAPACK or FFTW and builds no C, C++ or Fortran code: the "
        "special functions, the quadratures, the linear algebra and the solvers are "
        "written in the library, so that ``cargo build`` works on any platform with Rust "
        "alone and the results do not change with the version of a numerical library."
        + (
            " In ``Cargo.lock``, which also covers the bindings and the benchmarks, the only"
            f" crates that bind a system library are {', '.join(f'``{c}``' for c in f['native'])},"
            " the operating-system interfaces of Windows."
            if f["native"]
            else ""
        ),
        "",
    ]
    if fp:
        rows = []
        for k, lib in fp["libraries"].items():
            rows.append(
                [
                    f"{FOOTPRINT_NAMES[k]} {lib['version']}",
                    ", ".join(lib["direct"]) or "—",
                    str(len(lib["closure"]) - 1),
                    megabytes(lib["total_bytes"]),
                ]
            )
        out += [
            "What a user installs to run each library of the :doc:`benchmark <benchmarks>`:",
            "the run-time requirements (without extras), every distribution they pull in,",
            "and the size on disk of the whole, measured in the benchmark's environment",
            "(``benchmarks/libraries/footprint.py``).",
            "",
            *table(
                ["Library", "Direct requirements", "Distributions pulled in", "Installed size"],
                rows,
            ),
            "Sionna and OptiCommPy are heavier for reasons of their own: PyTorch (GPU,",
            "automatic differentiation, learning) and the ray tracer for Sionna, numba and",
            "SciPy for OptiCommPy.",
            "",
        ]
    out += [
        "Fast",
        "----",
        "",
        "Every block is compiled Rust; a Python call processes whole arrays, and the",
        "Monte Carlo runner (``comrs.run_spec``, ``Chain.run``, the ``comrs`` command) runs",
        "the trials on every core, with results independent of the thread count. Times of",
        "the :doc:`benchmark <benchmarks>` (median of five runs; the rules, the limits of",
        "the comparison and the code are on that page), as the time of the other library",
        "over the best time of com-rs on the same link (above 1: com-rs is faster):",
        "",
    ]
    rows = []
    for row in f["rows"]:
        cells = [f"{SCENARIO_NAMES[row['scenario']]}: {CASE_NAMES.get(row['case'], row['case'])}"]
        for lib in OTHERS:
            o = row["others"].get(lib, {})
            c1 = f"{times(o['1'] / row['comrs']['1'])}" if "1" in o and "1" in row["comrs"] else "—"
            ca = (
                f"{times(o['all'] / row['comrs']['all'])}"
                if "all" in o and "all" in row["comrs"]
                else "—"
            )
            cells.append(f"{c1} / {ca}" if c1 != "—" else "—")
        rows.append(cells)
    out += table(["Case", *[f"{n} (1 thread / all)" for n in OTHERS.values()]], rows)
    if f["machine"]:
        m = f["machine"]
        out += [
            f"Measured on {m['cpu']} ({m['logical_cores']} cores, {m['ram_gb']:.0f} GB, macOS "
            f"{m['macos']}), com-rs commit ``{f['commit']}``. The CPU only: Sionna is built",
            "for GPUs, where its times would differ.",
            "",
        ]
    out += [
        "Safe",
        "----",
        "",
        "Rust checks memory safety at compile time"
        + (
            ", and ``unsafe`` code is forbidden in every crate of the library "
            '(``unsafe_code = "forbid"`` in the workspace lints; the Python extension '
            "excepted, whose PyO3 macros expand to it)"
            if f["unsafe_forbidden"]
            else ""
        )
        + ". An invalid input — a negative variance, lengths that differ, a parameter out "
        "of range — returns an error of one shared type, raised as ``ValueError`` in "
        "Python, instead of a wrong result or a crash. Random blocks draw from explicit, "
        "reproducible streams: the same seed gives the same result on 1 or 12 threads. "
        "Clippy runs in its pedantic mode with every warning an error.",
        "",
        *table(
            ["Tests", "Count"],
            [
                ["Unit tests of the crates", str(t["rust"])],
                ["Validation tests (against papers, standards and tools)", str(t["validation"])],
                ["Python tests (functions; many are parametrised)", str(t["python"])],
            ],
        ),
        "Validated",
        "---------",
        "",
        "Every module is compared with an external reference: the figures and tables of",
        "papers, the tables of standards, the recorded outputs of reference",
        "implementations (OptiCommPy, GNPy, AFF3CT, Sionna, py3gpp), in tests named after",
        "the reference. The comparisons that are rerun without reading the paper again",
        "are written as :doc:`validation cards <comparisons>` (``just card <id>``).",
        "",
        *table(
            ["Evidence", "Count"],
            [
                ["Documents and tools compared", str(v["documents"])],
                ["— fully reproduced", str(v["badges"]["green"])],
                ["— partly reproduced", str(v["badges"]["orange"])],
                ["— implemented, not yet compared", str(v["badges"]["red"])],
                ["Inconsistencies found in the documents", str(v["remarks"])],
                ["Validation cards", str(v["cards"])],
                [
                    "Modules at maturity A / B / C",
                    f"{v['grades']['A']} / {v['grades']['B']} / {v['grades']['C']}",
                ],
            ],
        ),
        "The comparisons, paper by paper, are in ``docs/LITERATURE.md`` and the evidence",
        "per module in :doc:`maturity`.",
        "",
        "Updating",
        "--------",
        "",
        "The numbers of this page come from files that scripts regenerate:",
        "",
        ".. code-block:: bash",
        "",
        "   just bench-libraries               # the benchmark and the footprint (quiet machine)",
        "   python tools/literature_table.py   # after editing validation/literature.yaml",
        "   python tools/maturity_table.py     # after editing validation/maturity.yaml",
        "   python tools/highlights.py         # this page and the front pages",
        "",
        "``just check`` fails when one of these pages is older than its sources.",
        "",
    ]
    return "\n".join(out)


def splice(text: str, block: str, start: str, end: str) -> str:
    i, j = text.index(start), text.index(end)
    return text[: i + len(start)] + "\n" + block + text[j:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if a page is out of date")
    args = ap.parse_args()
    f = facts()
    status = 0
    readme = splice(
        README.read_text(encoding="utf-8"),
        readme_block(f),
        "<!-- highlights:start -->",
        "<!-- highlights:end -->",
    )
    index = splice(
        INDEX.read_text(encoding="utf-8"),
        index_block(f),
        ".. highlights:start",
        ".. highlights:end",
    )
    for path, text in ((README, readme), (INDEX, index), (PAGE, page(f))):
        status |= write_or_check(path, text, args.check, fix=FIX)
    return status


if __name__ == "__main__":
    sys.exit(main())

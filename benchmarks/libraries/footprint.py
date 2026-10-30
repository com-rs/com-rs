"""The footprint of every library of the benchmark: what a user installs to run it.

For each library, the distributions it requires at run time (its ``Requires-Dist``
without extras, then theirs, recursively, in this environment) and the size they take
on disk (the files listed by their ``RECORD``; for comrs built by ``maturin develop``,
the compiled extension). Run in the environment of the benchmark, after the libraries
are installed:

    python footprint.py            # writes results/footprint.json
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
LIBRARIES = {"comrs": "comrs", "komm": "komm", "OptiCommPy": "opticommpy", "Sionna": "sionna"}
NAME = re.compile(r"^[A-Za-z0-9._-]+")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(dist: metadata.Distribution) -> list[str]:
    """The run-time requirements of a distribution: its ``Requires-Dist`` without the
    extras (and without the requirements whose marker excludes this platform)."""
    from packaging.requirements import Requirement

    out = []
    for line in dist.requires or []:
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        out.append(canonical(req.name))
    return out


def closure(name: str) -> list[str]:
    """The distribution and everything it requires, recursively."""
    seen: list[str] = []
    todo = [canonical(name)]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.append(n)
        todo += requirements(metadata.distribution(n))
    return seen


def size(name: str) -> int:
    """The bytes of the files of a distribution (its RECORD; for an editable install,
    the files of its import package)."""
    dist = metadata.distribution(name)
    total = 0
    for f in dist.files or []:
        p = Path(dist.locate_file(f))
        if p.is_file() and "__pycache__" not in p.parts:
            total += p.stat().st_size
    if canonical(name) == "comrs":
        import comrs

        package = Path(comrs.__file__).parent
        total = sum(
            p.stat().st_size
            for p in package.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    return total


def main() -> int:
    out = {"machine": common.machine_info(), "libraries": {}}
    for label, dist in LIBRARIES.items():
        names = closure(dist)
        direct = requirements(metadata.distribution(dist))
        sizes = {n: size(n) for n in names}
        out["libraries"][label] = {
            "distribution": dist,
            "version": metadata.version(dist),
            "direct": sorted(direct),
            "closure": sorted(names),
            "bytes": sizes,
            "total_bytes": sum(sizes.values()),
        }
    path = HERE / "results" / "footprint.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

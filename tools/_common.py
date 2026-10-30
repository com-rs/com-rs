"""What the scripts of `tools/` share: the root of the repository, the themes of the
documentation, whether a cited test exists, the built extension, the formatting by
`ruff format`, and writing or checking a generated file.

The scripts run as `python tools/<script>.py`, which puts `tools/` on the import path
(`from _common import ROOT`); `docs/python/conf.py` reads the themes from here too.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]

# The themes of the documentation, in the order of a link: the sections of the API
# reference, of the maturity table and of the comparisons with the literature, and the
# sub-galleries of the examples (the folder of `docs/python/examples/`, None for a
# theme without one).
THEMES: list[tuple[str, str | None]] = [
    ("Foundations", "foundations"),
    ("Coding and shaping", "coding"),
    ("Waveforms", "waveforms"),
    ("MIMO", "mimo"),
    ("Wireless channels", None),
    ("Optical", "optical"),
    ("Hardware impairments", "hardware"),
    ("Receiver DSP", "receiver"),
    ("Metrics and analysis", "analysis"),
    ("Systems and standards", "systems"),
    ("Simulation and specifications", "simulation"),
]
THEME_TITLES = [title for title, _ in THEMES]
GALLERY_THEMES = [folder for _, folder in THEMES if folder is not None]


# ------------------------------------------------------------------- cited tests


def defines(file: Path, name: str) -> bool:
    """Whether `file` defines `name` as a whole identifier: `fn name` or `mod name` in
    Rust, `def name` or `class name` in Python, a `name:` key elsewhere (the checks of a
    specification); a comment or a longer name does not count."""
    text = file.read_text(encoding="utf-8")
    word = re.escape(name)
    if file.suffix == ".rs":
        pattern = rf"\b(?:fn\s+{word}\s*[(<]|mod\s+{word}\s*[{{;])"
    elif file.suffix == ".py":
        pattern = rf"^\s*(?:async\s+)?(?:def|class)\s+{word}\b"
    else:
        pattern = rf"(?<![\w.-]){word}(?![\w.-])"
    return re.search(pattern, text, re.MULTILINE) is not None


def missing_test(path: str, name: str = "") -> str | None:
    """Why the test `path::name` cited by `validation/literature.yaml` or
    `validation/maturity.yaml` does not exist, None when it does: `path` must exist and,
    with a `name`, define its last part (`tests::name` is found as `name`)."""
    file = ROOT / path
    if not file.exists():
        return f"no file {path}"
    if name and file.is_file() and not defines(file, name.split("::")[-1]):
        return f"no {name} in {path}"
    return None


# ------------------------------------------------------------------- generators


@functools.cache
def extension() -> ModuleType:
    """The built extension `comrs._comrs`, loaded without the package (whose generated
    modules `gen_blocks.py` and `gen_stubs.py` write), once per run."""
    found = sorted((ROOT / "python" / "comrs").glob("_comrs*.so")) + sorted(
        (ROOT / "python" / "comrs").glob("_comrs*.pyd")
    )
    if not found:
        raise SystemExit("build the extension first: maturin develop")
    spec = importlib.util.spec_from_file_location("_comrs", found[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ruff_format(text: str, path: Path, script: str) -> str:
    """`text` formatted by `ruff format` with the settings of the repository, as the
    file `path`; `script` names the generator in the error raised without ruff."""
    ruff = shutil.which("ruff")
    if ruff is None:
        raise SystemExit(f"{script} needs ruff (pip install --group lint)")
    cfg = ["--config", str(ROOT / "pyproject.toml")]
    out = subprocess.run(
        [ruff, "format", *cfg, "--stdin-filename", str(path), "-"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


# ------------------------------------------------------------------- outputs


def stale(path: Path, text: str) -> bool:
    """Whether the generated file `path` is missing or differs from `text`."""
    return not path.exists() or path.read_text(encoding="utf-8") != text


def write(path: Path, text: str, message: str = "written {path}") -> None:
    """Write the generated file and say so (`message` with the relative `{path}`)."""
    path.write_text(text, encoding="utf-8")
    print(message.format(path=path.relative_to(ROOT)))


def write_or_check(
    path: Path,
    text: str,
    check: bool,
    *,
    fix: str,
    written: str = "written {path}",
    up_to_date: bool = False,
    stream: TextIO | None = None,
) -> int:
    """The exit status of a generator: with `check`, 1 (and the command `fix` that
    regenerates it, on `stream`, stderr by default) when `path` is out of date, 0
    otherwise (saying so with `up_to_date`); without, writes `text` to `path`."""
    rel = path.relative_to(ROOT)
    if check:
        if stale(path, text):
            print(f"{rel} is out of date: run {fix}", file=stream or sys.stderr)
            return 1
        if up_to_date:
            print(f"{rel} is up to date")
        return 0
    write(path, text, written)
    return 0


def fail(header: str, items: list) -> int:
    """Print `header` and one indented line per item on stderr; the exit status 1."""
    print(header, *items, sep="\n  ", file=sys.stderr)
    return 1

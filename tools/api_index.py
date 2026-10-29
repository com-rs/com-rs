#!/usr/bin/env python3
"""The index of the public API: every public item of the Rust crates, of the Python
bindings and of the Python package, with its signature and the first line of its
documentation, in `docs/API.md`.

The index is the inventory to read before adding a function, so that a prototype that
exists is reused instead of written again. `just api-index` regenerates it; CI checks
that it is up to date (`--check`); `--duplicates` lists the names defined in more than
one module, the candidates for a shared helper.

The parser is deliberately simple (line-based): it understands the style of this
repository (one item per `pub` line, `///` documentation above it, `#[cfg(test)]`
closing the public part of a file), not the whole Rust grammar.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "API.md"
RUST_ITEM = re.compile(
    r"^(?P<indent>\s*)pub\s+"
    r"(?P<kind>(?:const\s+|async\s+|unsafe\s+)?fn|struct|enum|trait|type|const|static)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_IMPL = re.compile(
    r"^impl(?:<[^>]*>)?\s+(?:(?P<trait>[A-Za-z0-9_:<>, ]+?)\s+for\s+)?(?P<type>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_MACRO_ITEM = re.compile(r"^\s*(?:pub\s+)?struct\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\{")
PY_ITEM = re.compile(
    r"^(?P<indent>\s*)(?:async\s+)?(?P<kind>def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)


def rust_items(path: Path) -> list[tuple[str, str, str]]:
    """`(kind, signature, doc)` of the public items of a Rust file, methods prefixed by
    their type (`Ssfm::propagate`); the private items of `impl` blocks of public types
    are left out, and so is everything after `#[cfg(test)]`."""
    lines = path.read_text().splitlines()
    out = []
    impl_type: str | None = None
    impl_indent = 0
    doc: list[str] = []
    in_block_params = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#[cfg(test)]"):
            break
        if stripped.startswith("///"):
            doc.append(stripped[3:].strip())
            continue
        if stripped.startswith("//!") or stripped.startswith("#["):
            continue
        if stripped.startswith("block_params!"):
            in_block_params = True
        if in_block_params:
            if m := RUST_MACRO_ITEM.match(line):
                out.append(("params", m.group("name"), " ".join(doc) or "parameters of a block"))
            if stripped == "}" and line == "}":
                in_block_params = False
            doc = []
            continue
        if m := RUST_IMPL.match(line):
            impl_type = m.group("type")
            impl_indent = len(line) - len(line.lstrip())
            doc = []
            continue
        if impl_type and stripped == "}" and len(line) - len(line.lstrip()) == impl_indent:
            impl_type = None
            doc = []
            continue
        m = RUST_ITEM.match(line)
        if not m:
            if stripped and not stripped.startswith("//"):
                doc = []
            continue
        kind, name = m.group("kind"), m.group("name")
        # the signature: the lines up to the body or the semicolon
        sig_lines = []
        for j in range(i, min(i + 12, len(lines))):
            s = lines[j].strip()
            sig_lines.append(s)
            if s.endswith("{") or s.endswith(";") or s.endswith("}"):
                break
        sig = " ".join(sig_lines)
        sig = re.sub(r"\s+", " ", sig)
        sig = re.sub(r"\s*\{$", "", sig).rstrip(";").strip()
        sig = re.sub(r"^pub\s+", "", sig)
        sig = re.sub(r"\s*where .*$", "", sig)
        if len(sig) > 160:  # a long constant or array: the name is what matters
            sig = sig[:157].rstrip() + "…"
        if kind.endswith("fn") and impl_type:
            sig = sig.replace(f"fn {name}", f"fn {impl_type}::{name}", 1)
            name = f"{impl_type}::{name}"
        first = doc[0] if doc else ""
        out.append((kind.split()[-1], sig, first))
        doc = []
    return out


def python_items(path: Path) -> list[tuple[str, str, str]]:
    """`(kind, signature, doc)` of the public functions and classes of a Python file,
    methods prefixed by their class."""
    lines = path.read_text().splitlines()
    out = []
    cls: str | None = None
    for i, line in enumerate(lines):
        m = PY_ITEM.match(line)
        if not m:
            continue
        indent, kind, name = len(m.group("indent")), m.group("kind"), m.group("name")
        if name.startswith("_") and name != "__init__":
            continue
        if indent == 0:
            cls = name if kind == "class" else None
        elif indent == 4 and cls:
            name = f"{cls}.{name}"
        else:
            continue
        sig_lines = []
        for j in range(i, min(i + 12, len(lines))):
            s = lines[j].strip()
            sig_lines.append(s)
            if s.endswith(":"):
                break
        sig = re.sub(r"\s+", " ", " ".join(sig_lines)).rstrip(":")
        sig = re.sub(rf"^(def|class)\s+{re.escape(name.split('.')[-1])}", rf"\1 {name}", sig)
        doc = ""
        for j in range(i + 1, min(i + 4, len(lines))):
            s = lines[j].strip()
            if s.startswith(('"""', "'''")):
                doc = s.strip("\"'").strip()
                break
            if s and not s.startswith("#"):
                break
        out.append((kind, sig, doc))
    return out


def collect() -> dict[str, dict[str, list[tuple[str, str, str]]]]:
    """`{crate: {file: items}}`, in path order."""
    index: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(dict)
    for crate in sorted((ROOT / "crates").iterdir()):
        src = crate / "src"
        if not src.is_dir():
            continue
        for f in sorted(src.rglob("*.rs")):
            if f.parent.name == "bin":
                continue
            items = rust_items(f)
            if items:
                index[crate.name][str(f.relative_to(crate))] = items
    bindings = ROOT / "bindings" / "com-py" / "src"
    for f in sorted(bindings.rglob("*.rs")):
        items = rust_items(f)
        if items:
            index["com-py (bindings)"][str(f.relative_to(bindings.parent))] = items
    for f in sorted((ROOT / "python" / "comrs").glob("*.py")):
        items = python_items(f)
        if items:
            index["comrs (Python)"][str(f.relative_to(ROOT))] = items
    return index


def render(index) -> str:
    lines = [
        "# Index of the public API",
        "",
        "Generated by `tools/api_index.py` (`just api-index`); CI checks that it is up to date.",
        "One line per public item: its signature and the first line of its documentation.",
        "**Read it before adding a function**: a prototype that exists is reused, or moved",
        "one layer down, instead of written again (`docs/ARCHITECTURE.md`, placement).",
        "",
    ]
    total = 0
    for crate, files in index.items():
        n = sum(len(v) for v in files.values())
        total += n
        lines += [f"## {crate} ({n} items)", ""]
        for path, items in files.items():
            lines += [f"### `{path}`", ""]
            for _kind, sig, doc in items:
                sig = sig.replace("`", "")
                lines.append(f"- `{sig}`" + (f" — {doc}" if doc else ""))
            lines.append("")
    lines.insert(6, f"{total} items.")
    lines.insert(7, "")
    return "\n".join(lines) + "\n"


def duplicates(index) -> list[tuple[str, list[str]]]:
    """Function names defined in more than one module (methods by their last
    component), with the places."""
    where: dict[str, list[str]] = defaultdict(list)
    for crate, files in index.items():
        for path, items in files.items():
            for kind, sig, _ in items:
                if kind not in ("fn", "def"):
                    continue
                m = re.match(r"(?:fn|def)\s+([A-Za-z0-9_:.]+)", sig)
                if not m:
                    continue
                name = m.group(1).split("::")[-1].split(".")[-1]
                if name in (
                    "new",
                    "default",
                    "fmt",
                    "from",
                    "len",
                    "is_empty",
                    "__init__",
                    "build",
                    "parse",
                    "apply",
                    "name",
                    "infos",
                    "iter",
                    "get",
                    "set",
                    "clone",
                    "with",
                    "into",
                ):
                    continue
                where[name].append(f"{crate}:{path}")
    return sorted((n, p) for n, p in where.items() if len(set(p)) > 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail when docs/API.md is out of date")
    ap.add_argument(
        "--duplicates", action="store_true", help="list the names defined in several modules"
    )
    a = ap.parse_args()
    index = collect()
    if a.duplicates:
        for name, places in duplicates(index):
            print(f"{name}: {', '.join(sorted(set(places)))}")
        return 0
    text = render(index)
    if a.check:
        if OUTPUT.read_text() != text:
            print(
                f"{OUTPUT.relative_to(ROOT)} is out of date: run `just api-index`", file=sys.stderr
            )
            return 1
        return 0
    OUTPUT.write_text(text)
    print(f"written {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

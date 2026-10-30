#!/usr/bin/env python3
"""The index of the public API: every public item of the Rust crates, of the Python
bindings and of the Python package, with its signature and the first line of its
documentation, in `docs/API.md`.

The index is the inventory to read before adding a function, so that a prototype that
exists is reused instead of written again. `just api-index` regenerates it; CI checks
that it is up to date (`--check`); `--duplicates` lists the names (of functions
and of types) defined in more than one module, the candidates for a shared helper, and `--duplicates --check` fails on such
a name that `docs/api-duplicates.txt` does not accept.

The Rust parser is deliberately simple (line-based): it understands the style of this
repository (one item per `pub` line, `///` documentation above it, `#[cfg(test)]`
closing the public part of a file), not the whole Rust grammar. The Python files are
read by the parser of Python (`ast`), so that a signature is never cut.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

from _common import ROOT, fail, write_or_check

OUTPUT = ROOT / "docs" / "API.md"
ACCEPTED = ROOT / "docs" / "api-duplicates.txt"
RUST_ITEM = re.compile(
    r"^(?P<indent>\s*)pub\s+"
    r"(?P<kind>(?:const\s+|async\s+|unsafe\s+)?fn|struct|enum|trait|type|const|static)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_IMPL = re.compile(
    r"^impl(?:<[^>]*>)?\s+(?:(?P<trait>[A-Za-z0-9_:<>, ]+?)\s+for\s+)?(?P<type>[A-Za-z_][A-Za-z0-9_]*)"
)
RUST_MACRO_ITEM = re.compile(r"^\s*(?:pub\s+)?struct\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\{")


def rust_items(path: Path) -> list[tuple[str, str, str]]:
    """`(kind, signature, doc)` of the public items of a Rust file, methods prefixed by
    their type (`Ssfm::propagate`); the private items of `impl` blocks of public types
    are left out, and so is everything after `#[cfg(test)]`."""
    lines = path.read_text(encoding="utf-8").splitlines()
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


def _segment(src: str, node: ast.AST) -> str:
    """The source text of a node, on one line."""
    return re.sub(r"\s+", " ", ast.get_source_segment(src, node) or ast.unparse(node))


def _parameters(src: str, a: ast.arguments) -> str:
    """The parameters of a function as written (annotations and defaults), with the
    markers `/` and `*`."""

    def one(arg: ast.arg, default: ast.expr | None, prefix: str = "") -> str:
        text = prefix + _segment(src, arg)
        if default is None:
            return text
        sep = " = " if arg.annotation is not None else "="
        return f"{text}{sep}{_segment(src, default)}"

    positional = a.posonlyargs + a.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(a.defaults))
    defaults += a.defaults
    parts = [one(arg, d) for arg, d in zip(positional, defaults, strict=True)]
    if a.posonlyargs:
        parts.insert(len(a.posonlyargs), "/")
    if a.vararg is not None:
        parts.append(one(a.vararg, None, "*"))
    elif a.kwonlyargs:
        parts.append("*")
    parts += [one(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True)]
    if a.kwarg is not None:
        parts.append(one(a.kwarg, None, "**"))
    return ", ".join(parts)


def _python_item(src: str, node: ast.AST, name: str) -> tuple[str, str, str] | None:
    """`(kind, signature, doc)` of a function or class node, None for another node."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        sig = f"{prefix} {name}({_parameters(src, node.args)})"
        if node.returns is not None:
            sig += f" -> {_segment(src, node.returns)}"
        kind = "def"
    elif isinstance(node, ast.ClassDef):
        bases = [_segment(src, b) for b in node.bases]
        bases += [f"{k.arg}={_segment(src, k.value)}" if k.arg else f"**{_segment(src, k.value)}"
                  for k in node.keywords]  # fmt: skip
        sig = f"class {name}" + (f"({', '.join(bases)})" if bases else "")
        kind = "class"
    else:
        return None
    doc = ast.get_docstring(node) or ""
    return kind, sig, doc.strip().splitlines()[0] if doc.strip() else ""


def python_items(path: Path) -> list[tuple[str, str, str]]:
    """`(kind, signature, doc)` of the public functions and classes of a Python file,
    methods (and nested classes) prefixed by their class, read by the parser of Python
    (`ast`): the whole signature, its annotations and its return type."""
    src = path.read_text(encoding="utf-8")
    out = []
    for node in ast.parse(src).body:
        name = getattr(node, "name", "")
        if not name or name.startswith("_"):
            continue
        item = _python_item(src, node, name)
        if item is None:
            continue
        out.append(item)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                cname = getattr(child, "name", "")
                if not cname or (cname.startswith("_") and cname != "__init__"):
                    continue
                if (sub := _python_item(src, child, f"{name}.{cname}")) is not None:
                    out.append(sub)
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


# The names of a function that are an interface, not a copy, whatever the module
COMMON_VERBS = {
    "new", "default", "fmt", "from", "len", "is_empty", "__init__", "build", "parse",
    "apply", "name", "infos", "iter", "get", "set", "clone", "with", "into",
}  # fmt: skip
# The kinds of item compared: the functions, and the types (Rust structs, enums,
# traits and aliases; Python classes). The parameters of a declarative block
# (`block_params!`) are left out: they are named after the type the block wraps.
FUNCTIONS = {"fn", "def"}
TYPES = {"struct", "enum", "trait", "type", "class"}
PYTHON = "comrs (Python)"
ITEM_NAME = re.compile(r"(?:(?:async\s+)?fn|def|struct|enum|trait|type|class)\s+([A-Za-z0-9_:.]+)")


def duplicates(index) -> list[tuple[str, list[str]]]:
    """Names defined in more than one module, with the places: the functions (methods
    by their last component), and the types (structs, enums, traits, classes).

    A type is compared with the types of its language only: a Python class named after
    the Rust type it wraps (the classes of `comrs.blocks`) is its interface, not a
    copy; and a type never meets a function (`Constellation` and `constellation`)."""
    where: dict[str, list[str]] = defaultdict(list)
    types: dict[tuple[bool, str], list[str]] = defaultdict(list)
    for crate, files in index.items():
        for path, items in files.items():
            for kind, sig, _ in items:
                m = ITEM_NAME.match(sig)
                if not m:
                    continue
                if kind in FUNCTIONS:
                    name = m.group(1).split("::")[-1].split(".")[-1]
                    if name not in COMMON_VERBS:
                        where[name].append(f"{crate}:{path}")
                elif kind in TYPES and "." not in m.group(1):  # not a nested class
                    types[(crate == PYTHON, m.group(1))].append(f"{crate}:{path}")
    found = [(n, p) for n, p in where.items() if len(set(p)) > 1]
    found += [(n, p) for (_, n), p in types.items() if len(set(p)) > 1]
    return sorted(found)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail when docs/API.md is out of date")
    ap.add_argument(
        "--duplicates", action="store_true", help="list the names defined in several modules"
    )
    a = ap.parse_args()
    index = collect()
    if a.duplicates:
        found = duplicates(index)
        if not a.check:
            for name, places in found:
                print(f"{name}: {', '.join(sorted(set(places)))}")
            return 0
        accepted = {
            line.split("#")[0].strip()
            for line in ACCEPTED.read_text(encoding="utf-8").splitlines()
            if line.split("#")[0].strip()
        }
        new = [f"{n}: {', '.join(sorted(set(p)))}" for n, p in found if n not in accepted]
        gone = sorted(accepted - {n for n, _ in found})
        status = 0
        if new:
            status = fail(
                "names defined in several modules, not in docs/api-duplicates.txt (factor them, "
                "or add them with the reason they are not copies):",
                new,
            )
        if gone:
            status = fail(
                "names of docs/api-duplicates.txt no longer defined twice (remove them):", gone
            )
        return status
    return write_or_check(OUTPUT, render(index), a.check, fix="`just api-index`")


if __name__ == "__main__":
    sys.exit(main())

"""The validation cards of `validation/cards/`: check them, list them, run one.

    python tools/cards.py --check     # every card consistent with the repository
    python tools/cards.py list        # id, runner, tier, literature entry
    python tools/cards.py run <id>    # rerun the comparison of one card

A card is run by its runner: a function of `comrs` (`pytest tests/test_cards.py -k <id>`),
a Rust test of `validation/tests/` (`cargo test --release -p com-validation`, ignored
tests included) or a specification of `specs/validation/` (`comrs verify`). The check
makes sure that the runner exists and points back at the card (a Rust test loads it with
`Card::load("<id>")`), that the literature entry exists, that the reference file is in
a theme folder of `references/`, and that every card a Rust test loads exists.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml
from cited_tests import defines

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "validation" / "cards"
THEME = re.compile(r"^### .*\(`([a-z-]+)/`\)$", re.MULTILINE)
LOAD = re.compile(r'Card::load\("([a-z0-9_]+)"\)')


def cards() -> list[dict]:
    return [
        yaml.safe_load(p.read_text(encoding="utf-8")) | {"_file": p}
        for p in sorted(CARDS.glob("*.yaml"))
    ]


def runner(card: dict) -> tuple[str, str]:
    ((kind, target),) = card["run"].items()
    return kind, target


def problems() -> list[str]:
    out = []
    entries = yaml.safe_load((ROOT / "validation" / "literature.yaml").read_text(encoding="utf-8"))
    literature = {e["id"] for e in entries}
    readme = (ROOT / "references" / "README.md").read_text(encoding="utf-8")
    themes = {"standards", *THEME.findall(readme)}
    known = set()
    for card in cards():
        cid = card["id"]
        known.add(cid)
        if card["_file"].stem != cid:
            out.append(f"{card['_file'].name}: the id is {cid}")
        if card.get("literature") and card["literature"] not in literature:
            out.append(f"{cid}: no entry {card['literature']} in validation/literature.yaml")
        file = card["source"].get("file")
        if file and file.split("/")[0] not in themes:
            out.append(f"{cid}: {file} is not in a theme folder of references/")
        kind, target = runner(card)
        if kind == "test":
            path, _, name = target.partition("::")
            if not (ROOT / path).is_file() or (name and not defines(ROOT / path, name)):
                out.append(f"{cid}: no test {target}")
            elif f'Card::load("{cid}")' not in (ROOT / path).read_text(encoding="utf-8"):
                out.append(f"{cid}: {path} does not load the card")
        elif kind == "spec" and not (ROOT / target).is_file():
            out.append(f"{cid}: no specification {target}")
    for test in sorted((ROOT / "validation" / "tests").glob("*.rs")):
        for cid in LOAD.findall(test.read_text(encoding="utf-8")):
            if cid not in known:
                out.append(f"{test.relative_to(ROOT)}: no card {cid}")
    return out


def command(card: dict) -> list[str]:
    kind, target = runner(card)
    if kind == "function":
        return [sys.executable, "-m", "pytest", "-q", "tests/test_cards.py", "-k", card["id"]]
    if kind == "test":
        path, _, name = target.partition("::")
        only = ["--exact", name] if name else []
        return ["cargo", "test", "--release", "-p", "com-validation", "--test", Path(path).stem, "--",
                "--include-ignored", *only, "--nocapture"]  # fmt: skip
    return ["cargo", "run", "--release", "-p", "com-spec", "--bin", "comrs", "--", "verify", target]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check every card")
    parser.add_argument("action", nargs="?", choices=["list", "run"])
    parser.add_argument("id", nargs="?")
    args = parser.parse_args()
    if args.check:
        bad = problems()
        for p in bad:
            print(p, file=sys.stderr)
        return 1 if bad else 0
    if args.action == "list":
        for card in cards():
            kind, target = runner(card)
            print(
                f"{card['id']:38} {kind:8} {card.get('tier', 'ci'):9} {card.get('literature', '-'):15} {target}"
            )
        return 0
    if args.action == "run" and args.id:
        match = [c for c in cards() if c["id"] == args.id]
        if not match:
            print(f"no card {args.id}", file=sys.stderr)
            return 1
        cmd = command(match[0])
        print(" ".join(cmd))
        return subprocess.call(cmd, cwd=ROOT)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

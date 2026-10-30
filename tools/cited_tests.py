"""Whether a test cited by `validation/literature.yaml` or `validation/maturity.yaml`
exists: shared by `literature_table.py` and `maturity_table.py`."""

from __future__ import annotations

import re
from pathlib import Path


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

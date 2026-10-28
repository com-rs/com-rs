"""Chains and declarative specifications: the conversions of :meth:`comrs.Chain.to_spec`
and :meth:`comrs.Chain.from_spec`.

A specification is a mapping (or a YAML, JSON or TOML text or file, read by the Rust
parser) with the campaign settings (``name``, ``seed``, ``grid``, ``metrics``,
``stop``, ``fiber``…) and the ``link``, a list of blocks ``{"type": …, "label": …,
<parameters>}`` whose parameters may be grid variables (``"$name"``).

Example:
    >>> from comrs import spec
    >>> blocks, settings = spec.chain_blocks({"name": "x", "link": [
    ...     {"type": "source"}, {"type": "awgn", "snr_db": "$snr_db"}, {"type": "awgn", "snr_db": 3}]})
    >>> list(blocks), settings
    (['source', 'awgn_2', 'awgn_3'], {'name': 'x'})
    >>> blocks["awgn_2"].params
    {'snr_db': var('snr_db')}
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from comrs.block import Native, var

__all__ = ["chain_blocks", "load", "settings"]

_VAR = re.compile(r"\$(?:([A-Za-z_]\w*)|\{([A-Za-z_]\w*)\})")


def _from_spec_value(value: Any) -> Any:
    """A parameter value of a specification, ``"$name"`` and ``"${name}"`` as
    variables."""
    if isinstance(value, str):
        m = _VAR.fullmatch(value)
        return var(m.group(1) or m.group(2)) if m else value
    if isinstance(value, dict):
        return {k: _from_spec_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_spec_value(v) for v in value]
    return value


def _yaml(text: str) -> dict[str, Any]:
    """Parse a YAML text with the Rust parser of the specifications (no Python
    dependency)."""
    from comrs._comrs import _yaml_to_json

    return json.loads(_yaml_to_json(text))


def load(spec: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """A specification as a dict.

    Args:
        spec (str | os.PathLike | dict): a mapping (copied), a YAML or JSON text, or a
            file.

    Returns:
        dict: the specification.
    """
    if isinstance(spec, Mapping):
        return copy.deepcopy(dict(spec))
    path = Path(spec)
    text = path.read_text() if "\n" not in str(spec) and path.is_file() else str(spec)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _yaml(text)


def chain_blocks(spec: Mapping[str, Any]) -> tuple[dict[str, Native], dict[str, Any]]:
    """The blocks of the link of a specification and its other fields.

    The blocks are named by their labels (or types), a repeated name taking the
    position of its block as the Rust runner numbers them (``awgn_2``, ``awgn_3``).

    Args:
        spec (dict): the specification.

    Returns:
        tuple[dict[str, Native], dict]: the named blocks, in order, and the settings.
    """
    from comrs._comrs import _link_labels

    d = copy.deepcopy(dict(spec))
    link = d.pop("link")
    blocks: dict[str, Native] = {}
    for label, item in zip(_link_labels(link), link, strict=True):
        b = {"type": item} if isinstance(item, str) else dict(item)
        kind = b.pop("type")
        b.pop("label", None)
        blocks[label] = Native(kind, **{k: _from_spec_value(v) for k, v in b.items()})
    return blocks, d


def settings(
    base: Mapping[str, Any],
    *,
    name: str | None = None,
    grid: Mapping[str, Any] | None = None,
    metrics: list[str] | Mapping[str, Any] | None = None,
    stop: Mapping[str, Any] | None = None,
    seed: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Campaign settings: ``base`` updated by the arguments given, the metrics as a
    mapping and the grid axes as lists.

    Args:
        base (dict): the settings of the chain.
        name (str | None): the campaign name (``"chain"`` by default).
        grid (dict | None): the grid, ``{variable: values}``.
        metrics (list[str] | dict | None): the reported metrics or the ``metrics``
            mapping.
        stop (dict | None): the stopping rule.
        seed (int | None): the base seed.
        **extra: other fields (``batch``, ``fiber``, ``confidence``…).

    Returns:
        dict: the settings.
    """
    out = copy.deepcopy(dict(base))
    out.setdefault("name", "chain")
    given = {"name": name, "grid": grid, "metrics": metrics, "stop": stop, "seed": seed}
    out.update({k: v for k, v in given.items() if v is not None})
    out.update(extra)
    if isinstance(out.get("metrics"), list | tuple):
        out["metrics"] = {"report": list(out["metrics"])}
    if isinstance(out.get("grid"), Mapping):
        out["grid"] = {k: v if isinstance(v, Mapping) else list(v) for k, v in out["grid"].items()}
    return out

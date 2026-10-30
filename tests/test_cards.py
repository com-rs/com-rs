"""The validation cards of ``validation/cards/``: one test per case of every card.

A card states the function under test, its inputs, the expected outputs with their
tolerance and the exact source of the expectation (see ``validation/cards/README.md``).
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from oracle import _load

import comrs

yaml = pytest.importorskip("yaml")

CARDS = Path(__file__).resolve().parent.parent / "validation" / "cards"
ORACLES = Path(__file__).resolve().parent / "oracle"


def cards() -> list[dict[str, Any]]:
    return [yaml.safe_load(p.read_text()) | {"_file": p.name} for p in sorted(CARDS.glob("*.yaml"))]


def resolve(name: str) -> Any:
    module, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def build(value: Any) -> Any:
    if isinstance(value, dict) and "construct" in value:
        return resolve(value["construct"])(
            **{k: build(v) for k, v in value.get("args", {}).items()}
        )
    return value


def expected_value(value: Any) -> Any:
    if isinstance(value, dict) and "oracle" in value:
        data = _load(ORACLES / f"{value['oracle']}.json")
        for key in value["path"]:
            data = data[key]
        return data
    return value


CASES = [
    pytest.param(card, i, id=f"{card['id']}[{i}]")
    for card in cards()
    for i in range(len(card["cases"]))
]


# a number in exponent notation, which YAML 1.1 (PyYAML) reads as a string unless it has
# a dot and a signed exponent (21.4e9, 1e+10 and 1e-3 are strings; 2.14e+10 is a float)
NUMBER_AS_STRING = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def test_the_regex_of_numbers_read_as_strings_matches_them():
    as_read = yaml.safe_load("[21.4e9, 1e+10, 1e-3, -.5E7, 2.14e+10, 1.0e-3, 1e3x, e5, x1e5]")
    assert [bool(NUMBER_AS_STRING.match(str(v))) for v in as_read] == [
        True, True, True, True, False, False, False, False, False
    ]  # fmt: skip
    assert [isinstance(v, str) for v in as_read[:4]] == [True] * 4


def test_no_number_is_read_as_a_string():
    # YAML 1.1 (PyYAML) reads 21.4e9 as a string: write 2.14e+10
    for card in cards():
        for case in card["cases"]:
            bad = [s for s in strings(case) if NUMBER_AS_STRING.match(s)]
            assert not bad, f"{card['id']}: {bad} read as strings; write 2.14e+10 (a dot, a sign)"


def test_every_card_follows_the_schema():
    schema = json.loads((CARDS / "schema.json").read_text())
    jsonschema = pytest.importorskip("jsonschema")
    for card in cards():
        jsonschema.validate({k: v for k, v in card.items() if k != "_file"}, schema)
        assert card["_file"] == f"{card['id']}.yaml"


@pytest.mark.parametrize(("card", "i"), CASES)
def test_card(card, i):
    case = card["cases"][i]
    got = resolve(card["function"])(**{k: build(v) for k, v in case["inputs"].items()})
    output = case.get("output", card.get("output"))
    if output is not None:
        got = got[output]
    want = expected_value(case["expected"])
    tol = card.get("tolerance", {}) | case.get("tolerance", {})
    got, want = np.asarray(got), np.asarray(want)
    where = f"{card['id']} case {i} ({card['source']['reference']}, {card['source']['locator']})"
    if tol.get("exact"):
        np.testing.assert_array_equal(got, want, err_msg=where)
    else:
        np.testing.assert_allclose(
            got, want, rtol=tol.get("rel", 0.0), atol=tol.get("abs", 0.0), err_msg=where
        )
    assert comrs  # the package under test

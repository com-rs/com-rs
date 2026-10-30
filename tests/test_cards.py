"""The validation cards of ``validation/cards/``: their schema, and one test per case of
every card whose runner is a Python function.

A card states what is compared with which source (a paper with its figure or table, a
standard, a tool), the expected outputs with their tolerance and the reason for it, and
how the measured values are produced: a function of ``comrs`` (run here), a Rust test of
``validation/tests/`` or a specification of ``specs/validation/`` (run by
``tools/cards.py run <id>``; see ``validation/cards/README.md``).
"""

from __future__ import annotations

import importlib
import json
import re
from typing import Any

import numpy as np
import pytest
from oracle import recorded

import comrs

yaml = pytest.importorskip("yaml")

# the loader of tools/cards.py (`just card <id>`), which needs PyYAML: one reading of the
# cards for the tool and the tests
from cards import CARDS, cards  # noqa: E402


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
        data = recorded(value["oracle"])
        for key in value["path"]:
            data = data[key]
        return data
    return value


CASES = [
    pytest.param(card, i, id=f"{card['id']}[{i}]")
    for card in cards()
    if "function" in card["run"]
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
        for case in card.get("cases", []):
            bad = [s for s in strings(case) if NUMBER_AS_STRING.match(s)]
            assert not bad, f"{card['id']}: {bad} read as strings; write 2.14e+10 (a dot, a sign)"


def test_every_card_follows_the_schema():
    schema = json.loads((CARDS / "schema.json").read_text(encoding="utf-8"))
    jsonschema = pytest.importorskip("jsonschema")
    for card in cards():
        jsonschema.validate({k: v for k, v in card.items() if k != "_file"}, schema)
        assert card["_file"].name == f"{card['id']}.yaml"


def test_every_tolerance_that_is_not_exact_says_why():
    for card in cards():
        for case in card.get("cases", []):
            tol = card.get("tolerance", {}) | case.get("tolerance", {})
            if tol and not tol.get("exact"):
                assert "why" in case or "why" in card, f"{card['id']}: the tolerance has no why"


def accepted(got: np.ndarray, want: Any, tol: dict[str, Any]) -> bool:
    """The rule of the cards (and of ``comrs verify``): ``|v - e| <= atol + rtol |e|``, or
    ``e/f <= v <= f e`` with ``factor``, or a range ``{min, max}``."""
    if isinstance(want, dict):
        lo, hi = want.get("min", -np.inf), want.get("max", np.inf)
        return bool(np.all((got >= lo) & (got <= hi)))
    want = np.asarray(want)
    ok = np.zeros(np.shape(got), dtype=bool)
    if "atol" in tol or "rtol" in tol:
        ok |= np.abs(got - want) <= tol.get("atol", 0.0) + tol.get("rtol", 0.0) * np.abs(want)
    if "factor" in tol:
        f = tol["factor"]
        ok |= (want > 0) & (got >= want / f) & (got <= want * f)
    return bool(np.all(ok))


def test_the_rule_of_the_cards():
    assert accepted(np.asarray(0.30), 0.26, {"factor": 1.6})
    assert not accepted(np.asarray(0.5), 0.26, {"factor": 1.6})
    assert accepted(np.asarray([1.0, 2.05]), [1.0, 2.0], {"rtol": 0.05})
    assert accepted(np.asarray(14.0), {"min": 10}, {}) and not accepted(
        np.asarray(1.2), {"max": 1}, {}
    )


@pytest.mark.parametrize(("card", "i"), CASES)
def test_card(card, i):
    case = card["cases"][i]
    function = resolve(card["run"]["function"])
    assert getattr(comrs, function.__name__, None) is function, "the runner is not in comrs"
    got = function(**{k: build(v) for k, v in case["inputs"].items()})
    output = case.get("output", card.get("output"))
    if output is not None:
        got = got[output]
    want = expected_value(case["expected"])
    tol = card.get("tolerance", {}) | case.get("tolerance", {})
    got = np.asarray(got)
    where = f"{card['id']} case {i} ({card['source']['reference']}, {card['source']['locator']})"
    if tol.get("exact"):
        np.testing.assert_array_equal(got, np.asarray(want), err_msg=where)
    else:
        assert accepted(got, want, tol), f"{where}: got {got}, expected {want} ({tol})"

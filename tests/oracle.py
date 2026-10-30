"""Recorded oracles: the numbers of the reference implementations, read from files.

The tests of ``test_oracle.py`` (OptiCommPy) and ``test_gnpy_oracle.py`` (GNPy) compare
comrs with the outputs of these tools on the same inputs. The outputs are recorded once
into ``tests/oracle/<name>.json`` (small values) and ``tests/oracle/<name>.npz`` (arrays
of more than 64 values), with their provenance: the tool and its version, the versions
of Python and NumPy, the date. The tests read the files and the tools are only needed
to (re)record them::

    pip install --group oracle
    COMRS_RECORD_ORACLE=1 python -m pytest tests/test_oracle.py tests/test_gnpy_oracle.py

Re-recording is a deliberate act, named in the commit message with the version of the
tool. The files are written deterministically (sorted keys, ``repr`` of the floats, a
fixed timestamp in the archive, the date kept when nothing else changed), so that
re-recording an unchanged oracle produces no diff. ``tools/record_sionna_vectors.py``
writes its vectors with the same functions (:func:`archive`, :func:`dumps`), and
``tests/test_cards.py`` reads the expected values of the cards with :func:`recorded`.

A missing fixture skips its test on a development machine, and fails it in CI (the
variable ``CI`` set, as GitHub Actions sets it): a deleted fixture must not turn its
tests into skips there (:func:`missing`).

The outputs of a program are not covered by its licence: the fixtures are results of
OptiCommPy (GPL-3.0), GNPy (BSD-3-Clause) and py3gpp (AGPL-3.0), not their code nor their
data files.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import platform
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

DIR = Path(__file__).parent / "oracle"
RECORD = os.environ.get("COMRS_RECORD_ORACLE", "") not in ("", "0")
INLINE = 64  # arrays of at most this many values stay in the JSON document
DISTRIBUTIONS = {"OptiCommPy": "OptiCommPy", "GNPy": "gnpy", "py3gpp": "py3gpp"}
# a fixed timestamp for the entries of the archive: `np.savez` would stamp the time
STAMP = (1980, 1, 1, 0, 0, 0)


def oracle(name: str, compute: Callable[[], Any], *, tool: str) -> Any:
    """The recorded output of ``compute`` (which imports and runs ``tool``).

    In recording mode (``COMRS_RECORD_ORACLE=1``), runs ``compute``, writes the result
    to ``tests/oracle/<name>.json`` (and ``.npz``) and returns it as the file gives it;
    otherwise reads the file, or skips the test (fails it in CI, :func:`missing`) with a
    message naming the missing file.
    """
    if tool not in DISTRIBUTIONS:
        raise ValueError(f"unknown tool {tool!r}: one of {sorted(DISTRIBUTIONS)}")
    path = DIR / f"{name}.json"
    if RECORD:
        _record(name, path, compute(), tool)
    elif not path.exists():
        missing(
            f"recorded oracle {path.relative_to(DIR.parent.parent)} is missing: "
            f"install {tool} (pip install --group oracle) and record it with "
            "COMRS_RECORD_ORACLE=1 python -m pytest"
        )
    return _load(path)


def in_ci() -> bool:
    """Whether the tests run in continuous integration (the variable ``CI`` set)."""
    return os.environ.get("CI", "").lower() not in ("", "0", "false")


def missing(message: str, *, module: bool = False) -> NoReturn:
    """Skip the test (or, with ``module``, the module) whose recorded data are missing,
    or fail it in CI, where every fixture must be present."""
    import pytest

    if in_ci():
        pytest.fail(f"{message} (a fixture missing in CI fails)", pytrace=False)
    pytest.skip(message, allow_module_level=module)


def recorded(name: str) -> Any:
    """The recorded oracle ``tests/oracle/<name>.json`` (and its ``.npz``)."""
    return _load(DIR / f"{name}.json")


# ------------------------------------------------------------------------- writing


def _record(name: str, path: Path, value: Any, tool: str) -> None:
    from importlib.metadata import version

    arrays: dict[str, np.ndarray] = {}
    data = _encode(value, "data", arrays)
    doc: dict[str, Any] = {
        "name": name,
        "tool": tool,
        "version": version(DISTRIBUTIONS[tool]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "data": data,
    }
    npz = path.with_suffix(".npz")
    if arrays:
        doc["npz"] = npz.name
    doc["recorded"] = _date(path, npz, doc, arrays)
    DIR.mkdir(exist_ok=True)
    text = dumps(doc)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    if arrays:
        blob = archive(arrays)
        if not npz.exists() or npz.read_bytes() != blob:
            npz.write_bytes(blob)
    elif npz.exists():
        npz.unlink()


def _date(path: Path, npz: Path, doc: dict, arrays: dict) -> str:
    """Today, unless the same values are already recorded: then their date is kept."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        same = {k: v for k, v in old.items() if k != "recorded"} == doc
        if same and arrays:
            same = npz.exists() and npz.read_bytes() == archive(arrays)
        if same:
            return old["recorded"]
    return datetime.date.today().isoformat()


def dumps(doc: dict) -> str:
    """The document as JSON: sorted keys, indented, except the values of the inline
    arrays, one line each."""
    inline: list[list] = []
    text = json.dumps(_placeholders(doc, inline), sort_keys=True, indent=1) + "\n"
    for i, values in enumerate(inline):
        text = text.replace(f'"\\u0000{i}\\u0000"', json.dumps(values), 1)
    return text


def _placeholders(value: Any, inline: list[list]) -> Any:
    """A copy of the document whose inline arrays hold a placeholder for their values."""
    if isinstance(value, dict):
        if "__array__" in value:
            inline.append(value["__array__"]["values"])
            spec = {**value["__array__"], "values": f"\x00{len(inline) - 1}\x00"}
            return {"__array__": spec}
        return {k: _placeholders(v, inline) for k, v in value.items()}
    if isinstance(value, list):
        return [_placeholders(v, inline) for v in value]
    return value


def archive(arrays: dict[str, np.ndarray]) -> bytes:
    """The ``.npz`` archive of ``arrays``, deterministic: the entries in the order of
    the keys, each ``<key>.npy``, compressed, with a fixed timestamp."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            zf.writestr(zipfile.ZipInfo(f"{key}.npy", date_time=STAMP), buf.getvalue(),
                        compress_type=zipfile.ZIP_DEFLATED)  # fmt: skip
    return out.getvalue()


def _encode(value: Any, key: str, arrays: dict[str, np.ndarray]) -> Any:
    """The JSON form of ``value``; the large arrays are collected into ``arrays``."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise TypeError(f"cannot record an array of dtype {value.dtype}")
        if value.size > INLINE:
            arrays[key] = value
            return {"__npz__": key}
        flat = value.ravel()
        if value.dtype.kind == "c":
            values = [[float(z.real), float(z.imag)] for z in flat]
        else:
            values = [v.item() for v in flat]
        return {
            "__array__": {"dtype": value.dtype.str, "shape": list(value.shape), "values": values}
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v, f"{key}.{i}", arrays) for i, v in enumerate(value)]}
    if isinstance(value, list):
        return [_encode(v, f"{key}.{i}", arrays) for i, v in enumerate(value)]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("the keys of a recorded dict must be strings")
        return {k: _encode(v, f"{key}.{k}", arrays) for k, v in value.items()}
    raise TypeError(f"cannot record a value of type {type(value).__name__}")


# ------------------------------------------------------------------------- reading


def _load(path: Path) -> Any:
    doc = json.loads(path.read_text(encoding="utf-8"))
    arrays = np.load(path.with_suffix(".npz")) if "npz" in doc else None
    return _decode(doc["data"], arrays)


def _decode(value: Any, arrays: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v, arrays) for v in value]
    if not isinstance(value, dict):
        return value
    if "__complex__" in value:
        return complex(*value["__complex__"])
    if "__npz__" in value:
        return arrays[value["__npz__"]]
    if "__array__" in value:
        spec = value["__array__"]
        dtype = np.dtype(spec["dtype"])
        values = spec["values"]
        if dtype.kind == "c":
            values = [complex(re, im) for re, im in values]
        return np.array(values, dtype=dtype).reshape(spec["shape"])
    if "__tuple__" in value:
        return tuple(_decode(v, arrays) for v in value["__tuple__"])
    return {k: _decode(v, arrays) for k, v in value.items()}

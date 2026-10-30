"""Shared helpers of the Python tests.

- ``tests/`` is on the import path, for the recorded oracles (``from oracle import
  oracle``) and these helpers (``from conftest import qpsk_chain``), and so is
  ``docs/python``, for the closed forms that the gallery shares with the tests until
  the package binds them (``from _gallery_utils import ber_rayleigh_mrc``);
- :func:`qpsk_chain` and :func:`qpsk_spec` are the QPSK link over AWGN that many tests
  run, as a chain and as a specification;
- the ``slow`` marker labels the long tests (a WDM link with the GN model, the Raman
  amplification of a WDM link, the dual-polarisation back-propagation): they run by
  default, as in CI; ``python -m pytest -m "not slow"`` leaves them out.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import comrs
from comrs import blocks as B

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
SPECS = ROOT / "specs"
for folder in (TESTS, ROOT / "docs" / "python"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def qpsk_chain(n_symbols: int = 1000, *, snr: str = "ebn0_db", **source: Any) -> comrs.Chain:
    """QPSK over AWGN: ``tx`` (the source) and ``channel``, whose SNR is the grid variable
    ``snr`` (``ebn0_db`` for an ``awgn_ebn0`` block, ``snr_db`` for an ``awgn`` block of
    unit reference power)."""
    tx = B.Source(constellation="QPSK", n_symbols=n_symbols, **source)
    if snr == "ebn0_db":
        channel = B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db"))
    else:
        channel = B.Awgn(snr_db=comrs.var(snr), reference="unit")
    return comrs.Chain(tx=tx, channel=channel)


def qpsk_spec(n_symbols: int = 2000, **settings: Any) -> dict[str, Any]:
    """The specification of QPSK over AWGN at the grid variable ``ebn0_db`` (BER
    reported, 64 trials by default), updated by ``settings``."""
    spec = {
        "name": "qpsk",
        "grid": {"ebn0_db": [6]},
        "link": [
            {"type": "source", "constellation": "QPSK", "n_symbols": n_symbols},
            {"type": "awgn_ebn0", "ebn0_db": "$ebn0_db"},
        ],
        "metrics": {"report": ["ber"]},
        "stop": {"max_trials": 64},
    }
    spec.update(settings)
    return spec

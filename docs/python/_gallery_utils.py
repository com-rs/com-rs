"""The folders of the repository, for the examples and tutorials of the gallery that read
its specifications and recordings (the colours are those of :mod:`comrs.plot`).

sphinx-gallery runs every example with ``docs/python`` on the import path (``conf.py``
puts it there), so that an example imports this module as ``from _gallery_utils import
SPECS``; to run an example directly, put the folder on the path too::

    PYTHONPATH=docs/python python docs/python/examples/simulation/plot_01_chains.py

The gallery draws every example in the style of :func:`comrs.plot.style` (``conf.py``
sets it before each one); an example run directly gets it with ``comrs.plot.style()``.
"""

from __future__ import annotations

from pathlib import Path

# --- the repository -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "specs"

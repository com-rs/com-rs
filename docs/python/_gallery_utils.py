"""What the examples and tutorials of the gallery share: the folders of the repository,
the colours of the figures, and the closed forms that the package does not offer yet.

sphinx-gallery runs every example with ``docs/python`` on the import path (``conf.py``
puts it there), so that an example imports this module as ``from _gallery_utils import
SPECS``; to run an example directly, put the folder on the path too::

    PYTHONPATH=docs/python python docs/python/examples/mimo/plot_02_space_time_diversity.py

The module needs NumPy only (``tests/test_performance.py`` uses its closed form).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# --- the repository -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "specs"

# --- colours: the palette of the documentation, defined once ------------------------
#
# ``conf.reset_style`` builds the style of every figure from these; an example names a
# colour instead of writing its value.

# neutrals, from the darkest
TEXT = "#0b0b0b"  # titles and text
INK = "#1f1f1d"  # a curve or a label in black
LABEL = "#2b2a28"  # axis labels; the darkest curve of a family
MUTED = "#52514e"  # ticks; closed forms and annotations
GREY = "#8a8984"  # reference lines: capacity, AWGN, limits, chance
GREY_WARM = "#8a8780"  # values read on a published figure, cut-off lines
EDGE = "#b9b8b3"  # the axes
RULE = "#d0cfca"  # lines through the origin
GRID = "#e6e5e1"  # the grid, shaded bands
PAPER = "#f3f1ec"  # the empty cells of a resource grid
BACKGROUND = "#fcfcfb"  # figures and axes, and the edges of the markers on them

# the categorical slots, in their fixed order (a validated light palette): the colour
# cycle of every figure
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
AMBER = "#eda100"
PINK = "#e87ba4"
DEEP_GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"
SERIES = [BLUE, ORANGE, GREEN, AMBER, PINK, DEEP_GREEN, VIOLET, RED]

# a second categorical set, for the curves that a figure compares by hand (ColorBrewer
# Dark2, two more blues and a red)
TEAL = "#1b9e77"
RUST = "#d95f02"
PURPLE = "#7570b3"
COBALT = "#2a6fdb"
SKY = "#8fb8e6"
CRIMSON = "#c0392b"

# --- closed forms not bound yet ------------------------------------------------------


def ber_rayleigh_mrc(mean_ebn0_lin, branches: int):
    """BER of BPSK (and of Gray QPSK) with maximal-ratio combining of ``branches``
    i.i.d. Rayleigh branches of mean SNR per bit ``mean_ebn0_lin`` each (linear; a
    scalar or an array), Proakis & Salehi, eq. 13.4-15.

    Until the binding ``comrs.theory_ber_rayleigh_mrc`` of
    ``com_mapping::theory::ber_rayleigh_mrc`` lands: the examples and the tests then
    call it and this function goes.
    """
    mu = np.sqrt(mean_ebn0_lin / (1 + mean_ebn0_lin))
    return ((1 - mu) / 2) ** branches * sum(
        math.comb(branches - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(branches)
    )

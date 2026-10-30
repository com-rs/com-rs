"""
Constellations and Gray labels
==============================

The constellations of ``comrs`` carry the Gray labels of OptiCommPy: point ``i`` of
``c.points`` carries the bits of ``i``, most significant bit first, and two nearest
neighbours differ by a single bit. A symbol error to a neighbour therefore costs one
bit error, which is what makes the approximation :math:`P_b \\approx P_s / \\log_2 M`
of the bit error rate accurate at high SNR.
"""

import matplotlib.pyplot as plt
import numpy as np
from _gallery_utils import BACKGROUND, BLUE, MUTED

import comrs

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
for ax, (kind, m) in zip(axes, [("qam", 16), ("psk", 8), ("pam", 4)], strict=True):
    c = comrs.Constellation(kind, m).normalized()
    p = c.points
    k = c.bits_per_symbol
    ax.scatter(p.real, p.imag, s=46, color=BLUE, zorder=3, edgecolor=BACKGROUND, linewidth=1.5)
    for i, z in enumerate(p):
        ax.annotate(
            format(i, f"0{k}b"),
            (z.real, z.imag),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    lim = 1.35 * np.abs(p).max()
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", title=f"{m}-{kind.upper()}")
    ax.set_xlabel("in phase")
axes[0].set_ylabel("quadrature")
fig.tight_layout()

# %%
# Neighbours differ by one bit: check it on 64-QAM.
c = comrs.Constellation("qam", 64).normalized()
p = c.points
d = np.abs(p[:, None] - p[None, :])
dmin = d[d > 0].min()
pairs = np.argwhere(np.isclose(d, dmin))
print("nearest-neighbour pairs:", len(pairs) // 2)
print("bits that differ:", {bin(i ^ j).count("1") for i, j in pairs})

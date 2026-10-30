"""
PAPR of OFDM: the CCDF and its closed forms
===========================================

An OFDM block is the sum of :math:`N` independent subcarriers: by the central limit
theorem its samples are nearly complex Gaussian, and rare blocks have large peaks. The
complementary cumulative distribution :math:`\\Pr(\\mathrm{PAPR} > \\gamma)` is the
standard figure of merit.

- At the Nyquist rate the :math:`N` samples are nearly independent Rayleigh amplitudes:
  :math:`1 - (1 - e^{-\\gamma})^{N}` (van Nee & Prasad, *OFDM for Wireless Multimedia
  Communications*, 2000).
- The peaks of the continuous signal fall between those samples: an oversampling by
  :math:`L = 4` captures them, and their distribution is
  :math:`1 - \\exp\\bigl(-N \\sqrt{\\pi\\gamma/3}\\, e^{-\\gamma}\\bigr)` (Ochiai & Imai,
  *IEEE Trans. Commun.* 49(2), 2001), an asymptotic result for large :math:`N`: tight
  from a few hundred subcarriers, slightly pessimistic for :math:`N = 64`.
- DFT-spread OFDM (localised SC-FDMA, the LTE uplink) transmits a single-carrier signal
  and removes most of the PAPR (Myung, Lim & Goodman, *IEEE Veh. Technol. Mag.* 1(3),
  2006).
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs.plot import BACKGROUND, GREY, MUTED

BLOCKS = 4000
t = np.arange(4.0, 12.6, 0.1)
c = comrs.Constellation("qam", 16).normalized()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
for n in [64, 256, 1024]:
    ofdm = comrs.Ofdm(n, n)
    _, d = comrs.symbols(c, n * BLOCKS, comrs.Rng(n))
    p = ofdm.papr_db(ofdm.modulate(d))
    (line,) = ax1.semilogy(t, comrs.theory_papr_ccdf(n, t), linewidth=1.4)
    ax1.semilogy(
        t[::4],
        comrs.ccdf(p, t[::4]),
        "o",
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        label=f"N = {n}",
    )
ax1.plot([], [], color=MUTED, linewidth=1.4, label="Ochiai & Imai")
ax1.set(
    xlabel=r"$\gamma$ (dB)",
    ylabel=r"$\Pr(\mathrm{PAPR} > \gamma)$",
    ylim=(1e-3, 1.1),
    xlim=(4, 12.5),
    title="OFDM, oversampled (L = 4)",
)
ax1.legend()

n = 256
_, d = comrs.symbols(c, n * BLOCKS, comrs.Rng(7))
for L, marker in [(1, "s"), (4, "o")]:
    ofdm = comrs.Ofdm(n, n, oversampling=L)
    p = ofdm.papr_db(ofdm.modulate(d))
    ax2.semilogy(
        t,
        comrs.ccdf(p, t),
        marker=marker,
        markevery=6,
        markeredgecolor=BACKGROUND,
        label=f"OFDM, L = {L}",
    )
ax2.semilogy(
    t,
    comrs.theory_papr_ccdf(n, t, oversampled=False),
    color=GREY,
    linestyle=":",
    linewidth=1.4,
    label="van Nee & Prasad",
)
sc = comrs.Ofdm(1024, 256, dft_spread=True)
p = sc.papr_db(sc.modulate(d))
ax2.semilogy(
    t, comrs.ccdf(p, t), marker="D", markevery=6, markeredgecolor=BACKGROUND, label="DFT-s-OFDM"
)
ax2.set(xlabel=r"$\gamma$ (dB)", xlim=(2, 12.5), title=f"N = {n}: sampling and SC-FDMA")
ax2.legend()
fig.tight_layout()

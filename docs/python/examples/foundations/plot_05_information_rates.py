"""
Information rates of QAM
========================

The mutual information :math:`I(X; Y)` of a constellation over AWGN is the rate a
receiver that decodes symbols can achieve; the generalised mutual information (GMI)
is the rate of a bit-interleaved receiver that decodes bits independently, the usual
coded-modulation receiver. Both stay below the Shannon capacity
:math:`\\log_2(1 + \\mathrm{SNR})` and saturate at :math:`\\log_2 M`; with Gray labels
the gap between them is small.

The curves are computed by Gauss–Hermite quadrature (:func:`comrs.theory_mi`,
:func:`comrs.theory_gmi`); the markers are Monte Carlo estimates on 100 000 noisy
16-QAM symbols (:func:`comrs.information_rates`).
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs.plot import BACKGROUND, GREY, ORANGE

snr = np.arange(-5.0, 30.5, 1.0)
fig, ax = plt.subplots()
ax.plot(snr, np.log2(1 + 10 ** (snr / 10)), color=GREY, linewidth=1.4, label="Shannon capacity")
for m in [4, 16, 64]:
    mi = [comrs.theory_mi(m, s, "qam") for s in snr]
    gmi = [comrs.theory_gmi(m, s, "qam") for s in snr]
    (line,) = ax.plot(snr, mi, label=f"{m}-QAM, MI")
    ax.plot(snr, gmi, color=line.get_color(), linestyle="--", linewidth=1.4, label=f"{m}-QAM, GMI")

# Monte Carlo estimates for 16-QAM
c = comrs.Constellation("qam", 16).normalized()
est = []
for i, s in enumerate(snr[::3]):
    rng = comrs.Rng.stream(11, "mi", i, 0)
    idx, x = comrs.symbols(c, 100_000, rng)
    est.append(comrs.information_rates(comrs.awgn(x, s, rng), idx, c))
ax.plot(
    snr[::3],
    [e["mi"] for e in est],
    "o",
    color=ORANGE,
    markeredgecolor=BACKGROUND,
    label="16-QAM, MI simulated",
)
ax.set(
    xlabel="SNR $E_s/N_0$ (dB)",
    ylabel="bits per symbol",
    xlim=(-5, 30),
    ylim=(0, 7),
    title="Information rates over AWGN",
)
ax.legend(ncol=2, loc="lower right")
fig.tight_layout()

# %%
# The gap to capacity at 3.5 bits per symbol, which needs an SNR of
# 10 log10(2^3.5 - 1) = 10.13 dB: 16-QAM, close to saturation, is 1.4 dB away; 64-QAM
# 0.7 dB (what remains is the shaping gap of uniform square constellations).
target = 3.5
for m in [16, 64]:
    lo, hi = 0.0, 30.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if comrs.theory_mi(m, mid, "qam") > target else (mid, hi)
    print(
        f"{m}-QAM reaches {target} bits/symbol at {lo:.2f} dB (capacity: {10 * np.log10(2**target - 1):.2f} dB)"
    )

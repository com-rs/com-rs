"""
BPSK over flat Rayleigh fading
==============================

A narrowband channel whose gain :math:`h \\sim \\mathcal{CN}(0, 1)` changes from symbol
to symbol: with coherent detection and an average SNR per bit :math:`\\bar\\gamma`,

.. math::

   P_b = \\frac{1}{2}\\left(1 - \\sqrt{\\frac{\\bar\\gamma}{1 + \\bar\\gamma}}\\right)
   \\approx \\frac{1}{4\\bar\\gamma},

a decay of one decade per 10 dB instead of the exponential waterfall of AWGN: deep
fades dominate the errors (Proakis & Salehi, Sec. 13.3). The fading gains are drawn
with the same generator as the noise, so the whole link is reproducible from its
seed.
"""

import matplotlib.pyplot as plt
import numpy as np
from _gallery_utils import BACKGROUND, GREY

import comrs

N = 200_000
points = np.arange(0.0, 41.0, 4.0)
sim_awgn, sim_ray = [], []
for i, eb_db in enumerate(points):
    g = comrs.db2lin(eb_db)
    errors = {"awgn": 0, "rayleigh": 0}
    total = 0
    t = 0
    while min(errors.values()) < 200 and total < 4e7:
        rng = comrs.Rng.stream(5, "rayleigh", i, t)
        bits = comrs.bits(N, rng)
        x = 1.0 - 2.0 * bits
        w = rng.complex_noise(N, 1.0 / g)
        h = rng.complex_noise(N, 1.0)
        errors["awgn"] += int((((x + w).real < 0) != (bits == 1)).sum())
        errors["rayleigh"] += int((((np.conj(h) * (h * x + w)).real < 0) != (bits == 1)).sum())
        total += N
        t += 1
    sim_awgn.append(errors["awgn"] / total)
    sim_ray.append(errors["rayleigh"] / total)

# %%
eb = np.linspace(0, 40, 401)
g = 10 ** (eb / 10)
fig, ax = plt.subplots()
(l1,) = ax.semilogy(
    eb, [comrs.qfunc(np.sqrt(2 * v)) for v in g], linewidth=1.4, label="AWGN, theory"
)
(l2,) = ax.semilogy(eb, 0.5 * (1 - np.sqrt(g / (1 + g))), linewidth=1.4, label="Rayleigh, theory")
ax.semilogy(eb, 1 / (4 * g), color=GREY, linestyle=":", linewidth=1.2, label=r"$1/(4\bar\gamma)$")
keep = np.array(sim_awgn) > 0
ax.plot(
    points[keep],
    np.array(sim_awgn)[keep],
    "o",
    color=l1.get_color(),
    markeredgecolor=BACKGROUND,
    label="AWGN, simulated",
)
ax.plot(
    points,
    sim_ray,
    "s",
    color=l2.get_color(),
    markeredgecolor=BACKGROUND,
    label="Rayleigh, simulated",
)
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 40),
    xlabel=r"average $E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="BPSK with coherent detection",
)
ax.legend()
fig.tight_layout()

# %%
# At a BER of 1e-4, fading costs about 25 dB.
for name, f in [
    ("AWGN", lambda v: comrs.qfunc(np.sqrt(2 * v))),
    ("Rayleigh", lambda v: 0.5 * (1 - np.sqrt(v / (1 + v)))),
]:
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(10 ** (mid / 10)) > 1e-4 else (lo, mid)
    print(f"{name:>8}: {lo:.1f} dB at a BER of 1e-4")

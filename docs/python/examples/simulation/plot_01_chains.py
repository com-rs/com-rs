"""
A coherent receiver as a chain of blocks
========================================

The declarative link ``specs/coherent_link.yaml`` — 16-QAM at 32 GBd over standard
fibre, with laser phase noise, a detuned local oscillator, and the standard receiver
DSP (dispersion compensation, matched filter, Gardner clock recovery, fourth-power
frequency estimation, blind phase search, frame synchronisation) — read as a
:class:`comrs.Chain`.

**Trace.** ``Chain.trace`` returns the frame after every block: the constellation
spins under the frequency offset, is stopped by the frequency estimator (Leven et al.
2007) and its residual phase noise removed by blind phase search (Pfau, Hoffmann &
Noé 2009).

**Variants.** ``Chain.replace`` changes a block by name. Raising the linewidth of both
lasers from 100 kHz to 1 MHz leaves blind phase search with a larger linewidth-symbol
duration product, :math:`\\Delta\\nu T = 3 \\cdot 10^{-5}` per laser instead of
:math:`3 \\cdot 10^{-6}`, and a larger penalty with respect to 16-QAM over AWGN at the
same SNR. Both campaigns run on the parallel Rust runner.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs import blocks as B

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
spec = next(p / "specs/coherent_link.yaml" for p in [here, *here.parents] if (p / "specs").is_dir())
link = comrs.Chain.from_spec(spec)
print(link)

# %%
# The constellation after the main stages of the receiver.
frames = link.trace({"length_km": 80, "snr_db": 20}, comrs.Rng(3))
stages = {
    "decimate": "after clock recovery",
    "foe": "after frequency estimation",
    "bps": "after blind phase search",
}
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharex=True, sharey=True)
for ax, (name, title) in zip(axes, stages.items(), strict=True):
    y = frames[name].samples[2560:-256]
    ax.plot(y.real, y.imag, ".", markersize=1.0, alpha=0.3, color="#2a6fdb")
    ax.set(title=title, aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    ax.set_xlabel("in phase")
axes[0].set_ylabel("quadrature")
fig.tight_layout()

# %%
# Bit error rate against the SNR, for two laser linewidths.
snr = np.arange(12.0, 22.1, 2.0)
campaign = {
    "grid": {"length_km": [80], "snr_db": list(snr)},
    "stop": {"min_errors": {"ber": 200}, "min_trials": 8, "max_trials": 256},
}
variants = {
    "Δν = 100 kHz": link,
    "Δν = 1 MHz": link.replace(
        tx_laser=B.PhaseNoise(linewidth_hz=1e6), lo_laser=B.PhaseNoise(linewidth_hz=1e6)
    ),
}
fig, ax = plt.subplots()
s = np.linspace(10, 24, 141)
ax.semilogy(
    s,
    [comrs.theory_ber(16, e - 10 * np.log10(4), "qam") for e in s],
    color="#8a8984",
    linewidth=1.4,
    label="16-QAM over AWGN",
)
for (name, chain), marker in zip(variants.items(), ["o", "s"], strict=True):
    res = chain.run(**campaign)
    ber = np.array([r["metrics"]["ber"]["value"] for r in res])
    keep = ber > 0
    ax.semilogy(snr[keep], ber[keep], marker=marker, markeredgecolor="#fcfcfb", label=name)
ax.set(
    xlabel="SNR (dB)",
    ylabel="bit error rate",
    ylim=(1e-6, 0.1),
    title="Coherent 16-QAM receiver, 80 km",
)
ax.legend(loc="lower left")
fig.tight_layout()

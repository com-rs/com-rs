"""
Häger & Pfister 2021, Fig. 5: linear equalisation and back-propagation
======================================================================

Fig. 5 of C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
communication systems", *IEEE J. Sel. Areas Commun.* 39(1), 2021 (arXiv:2010.14258),
gives the effective SNR against the launch power of a single channel over 25 spans of
80 km of standard fibre at 10.7 GBd with Gaussian symbols, for linear equalisation
(dispersion compensation alone) and digital back-propagation (DBP) with one step per
span. The link is that of the paper: :math:`\\beta_2 = -21.683` ps²/km at 194.6 THz,
:math:`\\gamma = 1.3` /W/km, 0.2 dB/km, a noise figure of 5 dB, root-raised-cosine
pulses of roll-off 0.1, propagation at 6 samples per symbol and receivers at 2.

The published values were read from the vector paths of the figure (about 0.05 dB)
and are the ``expect`` entries of ``specs/validation/hager2021_fig5.yaml``, which CI
checks within 0.3 dB; the agreement is within 0.13 dB. This script runs that
specification with two trials per point instead of four and draws the published
points on the simulated curves. The validation found that the coarse-step DBP applied
the nonlinear phase over the step length instead of the effective length (60 % of the
phase with one step per span), fixed since.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

import comrs

here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
root = next(p for p in [here, *here.parents] if (p / "specs").is_dir())
text = (root / "specs/validation/hager2021_fig5.yaml").read_text()
spec = yaml.safe_load(text)
results = comrs.run_spec(text.replace("stop: {max_trials: 4}", "stop: {max_trials: 2}"))

fig, ax = plt.subplots()
labels = {"linear": "linear equalisation", "dbp1": "DBP, 1 step per span"}
for (receiver, label), marker in zip(labels.items(), ["o", "s"], strict=True):
    rows = [r for r in results if r["condition"]["receiver"] == receiver]
    x = np.array([r["condition"]["power_dbm"] for r in rows])
    snr = np.array([r["metrics"]["snr_eff_db"]["value"] for r in rows])
    (line,) = ax.plot(x, snr, marker=marker, markeredgecolor="#fcfcfb", label=f"comrs, {label}")
    published = next(e for e in spec["expect"] if e["where"]["receiver"] == receiver)["curve"]
    ax.plot(published["x"], published["y"], "x", color=line.get_color(), markersize=7, mew=1.6,
            label=f"Häger & Pfister, Fig. 5, {label}")  # fmt: skip
ax.set(
    xlim=(-12, 3),
    ylim=(10, 27),
    xlabel="launch power (dBm)",
    ylabel="effective SNR (dB)",
    title="25 × 80 km, 10.7 GBd (Häger & Pfister 2021, Fig. 5)",
)
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# The largest difference with the published points, over the powers common to both.
for receiver, label in labels.items():
    rows = {r["condition"]["power_dbm"]: r["metrics"]["snr_eff_db"]["value"] for r in results
            if r["condition"]["receiver"] == receiver}  # fmt: skip
    published = next(e for e in spec["expect"] if e["where"]["receiver"] == receiver)["curve"]
    diffs = [
        abs(rows[x] - y) for x, y in zip(published["x"], published["y"], strict=True) if x in rows
    ]
    print(f"{label}: within {max(diffs):.2f} dB of the published points")

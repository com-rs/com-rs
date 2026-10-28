"""
Bit error rate of M-QAM over AWGN
=================================

The waterfall curves of Gray-mapped 4-, 16- and 64-QAM over the additive white
Gaussian noise channel, against the closed form

.. math::

   P_b \\approx \\frac{4}{\\log_2 M}\\left(1 - \\frac{1}{\\sqrt{M}}\\right)
   Q\\left(\\sqrt{\\frac{3 \\log_2 M}{M - 1}\\frac{E_b}{N_0}}\\right)

(Proakis & Salehi, *Digital Communications*, Sec. 4.3). The campaign is a declarative
specification run on every core by the Rust runner: each point stops after 200 bit
errors, and the sweep of a constellation stops once its BER falls below
:math:`10^{-6}`. The error bars are the 95 % Wilson intervals.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs

results = comrs.run_spec(
    """
name: qam_awgn
seed: 2024
grid:
  m: [4, 16, 64]
  ebn0_db: {from: 0, to: 20, step: 1}
link:
  - {type: source, constellation: {kind: qam, m: $m}, n_symbols: 8192}
  - {type: awgn_ebn0, ebn0_db: $ebn0_db}
metrics: {report: [ber]}
stop: {min_errors: {ber: 200}, min_trials: 8, max_trials: 4000}
callbacks:
  skip_below: {metric: ber, threshold: 1e-6, along: ebn0_db}
"""
)

# %%
# Simulated points with their intervals, and the closed form.
fig, ax = plt.subplots()
eb = np.linspace(0, 20, 201)
for m in [4, 16, 64]:
    rows = [r for r in results if r["condition"]["m"] == m and r["status"] == "completed"]
    x = np.array([r["condition"]["ebn0_db"] for r in rows])
    ber = np.array([r["metrics"]["ber"]["value"] for r in rows])
    lo = np.array([r["metrics"]["ber"]["lo"] for r in rows])
    hi = np.array([r["metrics"]["ber"]["hi"] for r in rows])
    keep = ber > 0
    (line,) = ax.semilogy(eb, [comrs.theory_ber(m, e, "qam") for e in eb], linewidth=1.4)
    ax.errorbar(
        x[keep],
        ber[keep],
        yerr=[ber[keep] - lo[keep], hi[keep] - ber[keep]],
        fmt="o",
        color=line.get_color(),
        markeredgecolor="#fcfcfb",
        capsize=0,
        label=f"{m}-QAM",
    )
ax.plot([], [], color="#52514e", linewidth=1.4, label="theory")
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 20),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="Gray-mapped QAM over AWGN",
)
ax.legend()
fig.tight_layout()

# %%
# The Eb/N0 needed for a BER of 1e-5 grows by 3.8 dB from QPSK to 16-QAM, and by
# 4.4 dB more to 64-QAM: the price of two more bits per symbol.
for m in [4, 16, 64]:
    lo, hi = 0.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if comrs.theory_ber(m, mid, "qam") > 1e-5 else (lo, mid)
    print(f"{m:>2}-QAM: Eb/N0 = {lo:.2f} dB at a BER of 1e-5")

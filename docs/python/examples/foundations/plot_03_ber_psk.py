"""
M-PSK over AWGN with the Python runner
======================================

The bit error rate of Gray-mapped BPSK, QPSK, 8-PSK and 16-PSK, simulated with the
Monte Carlo runner of :mod:`comrs.sim`: the trial is a Python function, every point
runs until 200 bit errors (or 400 trials), and the result of every point is an error
rate with its Wilson interval. BPSK and QPSK share the same curve
:math:`Q(\\sqrt{2 E_b/N_0})`; beyond four points, every doubling of the order costs
3.5 to 4.5 dB at a BER of :math:`10^{-5}`, since the points get closer on the circle:

.. math::

   P_b \\approx \\frac{2}{\\log_2 M} Q\\left(\\sqrt{2 \\log_2 M \\frac{E_b}{N_0}}\\sin\\frac{\\pi}{M}\\right).
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs import sim
from comrs.plot import BACKGROUND, MUTED

ORDERS = [2, 4, 8, 16]


def trial(cond, rng, rec, c):
    k = c.bits_per_symbol
    bits = comrs.bits(k * 8192, rng)
    y = comrs.awgn_ebn0(c.modulate(bits), cond["ebn0_db"], k, rng)
    rec.errors("ber", int((c.demodulate(y) != bits).sum()), bits.size)


results = sim.run(
    trial,
    sim.grid(m=ORDERS, ebn0_db=np.arange(0.0, 21.0, 2.0)),
    name="psk",
    seed=7,
    setup=lambda cond: comrs.Constellation("psk", cond["m"]).normalized(),
    stop=sim.Stop.min_errors("ber", 200) | sim.Stop.max_trials(400),
)

# %%
fig, ax = plt.subplots()
eb = np.linspace(0, 20, 201)
labels = {2: "BPSK", 4: "QPSK", 8: "8-PSK", 16: "16-PSK"}
for m in ORDERS:
    pts = [r for r in results if r.condition["m"] == m and r.metrics["ber"].value > 0]
    x = np.array([r.condition["ebn0_db"] for r in pts])
    p = np.array([r.metrics["ber"].value for r in pts])
    ci = np.array([r.metrics["ber"].interval(0.95) for r in pts])
    (line,) = ax.semilogy(
        eb,
        [comrs.theory_ber(m, e, "psk") for e in eb],
        linewidth=1.4,
        linestyle="--" if m == 4 else "-",
    )
    ax.errorbar(
        x,
        p,
        yerr=[p - ci[:, 0], ci[:, 1] - p],
        fmt="o" if m != 4 else "s",
        color=line.get_color(),
        markeredgecolor=BACKGROUND,
        label=labels[m],
    )
ax.plot([], [], color=MUTED, linewidth=1.4, label="theory")
ax.set(
    ylim=(1e-6, 0.5),
    xlim=(0, 20),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="Gray-mapped PSK over AWGN",
)
ax.legend()
fig.tight_layout()

# %%
# Every point carries its own trial count: the high-SNR points need many more trials
# to collect their errors.
for r in results[:6]:
    lo, hi = r.metrics["ber"].interval(0.95)
    print(
        f"m={r.condition['m']:>2} Eb/N0={r.condition['ebn0_db']:4.1f} dB  BER={r.metrics['ber'].value:.3e}"
        f"  [{lo:.2e}, {hi:.2e}]  trials={r.trials}"
    )

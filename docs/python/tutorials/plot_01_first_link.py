"""
A first link: 16-QAM over AWGN
==============================

The smallest complete simulation: draw bits, map them on a constellation, add noise,
decide, count the errors and compare with the closed form. Every step is one call on
NumPy arrays. The same link is then written as a chain of named blocks and run as a
Monte Carlo campaign on every core, with a stopping rule and confidence intervals.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs import blocks as B
from comrs.plot import BACKGROUND, MUTED

# %%
# **The generator.** There is no global random state: every function that draws takes
# a :class:`comrs.Rng` and advances it, so a script is reproducible from its seed
# (and gives the same numbers as the Rust crates).
rng = comrs.Rng(2024)

# %%
# **The constellation.** ``Constellation(kind, m)`` carries the points and their Gray
# labels; ``normalized()`` scales them to a unit average energy, the convention of the
# closed forms. ``demodulate`` on the points themselves returns the label of each.
c = comrs.Constellation("qam", 16).normalized()
k = c.bits_per_symbol
labels = c.demodulate(c.points).reshape(-1, k)
print(c, "-", k, "bits per symbol, energy", np.mean(np.abs(c.points) ** 2))

fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.plot(c.points.real, c.points.imag, "o", markersize=7)
for p, bits in zip(c.points, labels, strict=True):
    ax.annotate("".join(map(str, bits)), (p.real, p.imag), (6, 6), textcoords="offset points",
                fontsize=8, color=MUTED)  # fmt: skip
ax.set(aspect="equal", xlabel="in phase", ylabel="quadrature", title="16-QAM, Gray labels")
fig.tight_layout()

# %%
# **Bits, symbols, noise, decisions.** :func:`comrs.awgn_ebn0` adds the noise for an
# :math:`E_b/N_0` in dB, given the bits per symbol; ``demodulate`` decides the nearest
# point and returns its bits; :func:`comrs.ber` counts the differences.
bits = comrs.bits(k * 100_000, rng)
x = c.modulate(bits)
y = comrs.awgn_ebn0(x, 10.0, k, rng)
decided = c.demodulate(y)
print(f"BER {comrs.ber(decided, bits):.2e}, closed form {comrs.theory_ber(16, 10.0, 'qam'):.2e}")

fig, ax = plt.subplots(figsize=(4.4, 4.4))
comrs.plot.constellation(y, ax=ax, reference=c, max_points=8000)
ax.set_title("received at $E_b/N_0$ = 10 dB")
fig.tight_layout()

# %%
# **A sweep by hand.** A loop over :math:`E_b/N_0` gives a first curve. What sets the
# precision of a point is its number of errors: with 100 errors the relative standard
# error is about 10 %, with 400 about 5 %. The points of the loop below are left out
# once they have fewer than 20 errors.
ebn0 = np.arange(0.0, 15.0, 1.0)
ber_loop = []
for e in ebn0:
    bits = comrs.bits(k * 50_000, rng)
    y = comrs.awgn_ebn0(c.modulate(bits), e, k, rng)
    errors = int((c.demodulate(y) != bits).sum())
    ber_loop.append(errors / bits.size if errors >= 20 else np.nan)

# %%
# **The same link as a chain.** A :class:`comrs.Chain` names the blocks of the link;
# :func:`comrs.var` marks the parameter that the campaign sweeps. ``run`` evaluates
# the chain on a grid of conditions on the parallel Rust runner, until a stopping
# rule is met (here 500 bit errors, or 2000 trials of 4096 symbols), and reports
# every metric with its 95 % confidence interval.
link = comrs.Chain(
    tx=B.Source(constellation="16-QAM", n_symbols=4096),
    channel=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
)
results = link.run(
    grid={"ebn0_db": list(ebn0)},
    metrics=["ber", "ser"],
    stop={"min_errors": {"ber": 500}, "max_trials": 2000},
    seed=1,
)
r = results.where(ebn0_db=10.0)[0]
print(r["condition"], r["trials"], "trials,", r["metrics"]["ber"])

# %%
# The loop, the campaign with its intervals, and the closed form of Gray-mapped QAM
# (Proakis & Salehi, *Digital Communications*, 5th ed., Sec. 4.3-2).
x, ber, lo, hi = results.series("ebn0_db", "ber", interval=True)
fine = np.linspace(0, 14, 141)
fig, ax = plt.subplots()
ax.semilogy(fine, comrs.theory_ber(16, fine, "qam"), color=MUTED, linewidth=1.4,
            label="closed form")  # fmt: skip
ax.semilogy(ebn0, ber_loop, "s", markeredgecolor=BACKGROUND, label="loop, 50 000 symbols")
ax.errorbar(x, ber, yerr=[ber - lo, hi - ber], fmt="o", markeredgecolor=BACKGROUND, capsize=2,
            label="campaign, 500 errors per point")  # fmt: skip
ax.set(ylim=(1e-5, 0.5), xlabel="$E_b/N_0$ (dB)", ylabel="bit error rate", title="16-QAM over AWGN")
ax.legend()
fig.tight_layout()

# %%
# **Where to go next.** :doc:`plot_02_classic_chain` adds pulse shaping, a matched
# filter and impairments to this link; the :doc:`user guide <../user_guide/chains>`
# describes chains, probes and variants; :doc:`plot_04_specifications` runs the same
# campaign from a YAML file and the command line.

"""
Alamouti 1998, Fig. 4: transmit diversity against MRRC
=======================================================

Fig. 4 of S. M. Alamouti, "A simple transmit diversity technique for wireless
communications", *IEEE J. Sel. Areas Commun.* 16(8), 1998, compares the BER of
uncoded coherent BPSK over Rayleigh fading for five systems: no diversity, maximal
ratio receive combining (MRRC) with two and four receive antennas, and the two-branch
transmit diversity scheme of the paper with one and two receive antennas. The
assumptions of the paper's Sec. IV are reproduced: the total transmit power of the
two antennas equals that of the single antenna of MRRC, the fading amplitudes of
every transmit-receive pair are uncorrelated Rayleigh with the same average power,
and the receiver knows the channel.

The lines are the closed form of MRC with :math:`L` branches (Proakis & Salehi, eq.
13.4-15), the mean SNR of a branch being :math:`E_b/N_0` for MRRC and
:math:`E_b/N_0 - 3\\,\\mathrm{dB}` for the transmit diversity scheme, which the paper
notes (its Sec. IV) as the price of splitting the power. The points are the campaign
on the parallel Rust runner; ``specs/mimo_diversity.yaml`` checks the 2 × 2 case in
CI. Open Fig. 4 of the paper next to this figure: the five curves are the same.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs import blocks as B
from comrs.plot import BACKGROUND, MUTED


def system(n_tx, n_rx):
    """BPSK over i.i.d. Rayleigh fading: MRRC with one transmit antenna, the
    two-branch transmit diversity scheme with two (fading constant over the two
    symbol periods of a code block)."""
    return comrs.Chain(
        tx=B.Source(constellation="BPSK", n_symbols=4096),
        mimo=B.MimoTx(n_tx=n_tx, scheme="alamouti" if n_tx == 2 else "spatial"),
        fading=B.Rayleigh(n_rx=n_rx, coherence=2 if n_tx == 2 else 1),
        noise=B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db")),
        rx=B.MimoRx(detector="zf"),
    )


# (n_tx, n_rx, diversity order, share of the power per transmit antenna, last Eb/N0)
systems = {
    "no diversity (1 Tx, 1 Rx)": (1, 1, 1, 1.0, 40),
    "MRRC (1 Tx, 2 Rx)": (1, 2, 2, 1.0, 28),
    "MRRC (1 Tx, 4 Rx)": (1, 4, 4, 1.0, 16),
    "new scheme (2 Tx, 1 Rx)": (2, 1, 2, 0.5, 32),
    "new scheme (2 Tx, 2 Rx)": (2, 2, 4, 0.5, 20),
}
fig, ax = plt.subplots(figsize=(7.5, 5.2))
eb = np.linspace(0, 50, 251)
for (name, (nt, nr, order, share, last)), marker in zip(
    systems.items(), ["o", "s", "D", "^", "v"], strict=True
):
    (line,) = ax.semilogy(
        eb,
        comrs.theory_ber_rayleigh_mrc(comrs.lin2db(share * 10 ** (eb / 10)), order),
        linewidth=1.4,
    )
    res = system(nt, nr).run(
        grid={"ebn0_db": list(range(0, last + 1, 4))},
        metrics={"report": ["ber"], "scale": "none"},
        stop={"min_errors": {"ber": 200}, "min_trials": 4, "max_trials": 3000},
        seed=10 * nt + nr,
    )
    x, ber, lo, hi = res.series("ebn0_db", "ber", interval=True, min_count=20)
    ax.errorbar(x, ber, yerr=[ber - lo, hi - ber], fmt=marker, color=line.get_color(),
                markeredgecolor=BACKGROUND, capsize=0, label=name)  # fmt: skip
ax.plot([], [], color=MUTED, linewidth=1.4, label="closed form, MRC of $L$ branches")
ax.set(
    xlim=(0, 50),
    ylim=(1e-6, 1),
    xlabel="$E_b/N_0$ (dB)",
    ylabel="bit error rate",
    title="Coherent BPSK over Rayleigh fading (Alamouti 1998, Fig. 4)",
)
ax.legend(loc="lower left")
fig.tight_layout()

# %%
# The two systems of diversity order four differ by exactly 3 dB, and the two of
# order two as well: the transmit diversity scheme obtains the diversity of MRRC
# with half the power per antenna, which is the conclusion of the paper.
for order in [2, 4]:
    g = 10 ** (np.linspace(10, 30, 201) / 10)
    ratio = comrs.theory_ber_rayleigh_mrc(comrs.lin2db(g), order) / comrs.theory_ber_rayleigh_mrc(
        comrs.lin2db(g / 2), order
    )
    print(
        f"order {order}: the BER ratio of MRRC to the new scheme tends to {ratio[-1]:.3f} = 2^-{order}"
    )

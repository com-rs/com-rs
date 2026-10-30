"""
Filtered multicarrier and OTFS
==============================

Three answers to the weaknesses of CP-OFDM, side by side.

1. **Spectral containment.** CP-OFDM keeps the sidelobes of its rectangular pulses,
   which fall as :math:`1/f^{2}`. FBMC/OQAM (``comrs.Fbmc``) shapes every subcarrier
   with the PHYDYAS prototype (Bellanger, overlapping factor :math:`K = 4`) and drops
   the prefix; UFMC (``comrs.Ufmc``, F. Schaich and T. Wild, "Waveform contenders for
   5G: OFDM vs. FBMC vs. UFMC," *Proc. ISCCSP*, 2014) filters sub-bands of 12
   subcarriers with Dolph–Chebyshev filters of :math:`L = 16` taps, whose ramp-up and
   ramp-down replace the prefix. The densities below (Welch, 36 of 64 subcarriers
   active, same overhead of 15 samples for CP-OFDM and UFMC) show the emission in the
   empty part of the band. The filters of UFMC also attenuate the subcarriers at the
   edges of each sub-band (the dips in the band), which the one-tap equaliser undoes.
2. **Delay–Doppler channel.** OTFS (``comrs.Otfs``) puts the symbols on a grid of
   :math:`M` delays and :math:`N` Doppler shifts. A channel of a few paths of whole
   delays and Doppler bins acts on that grid as a 2-D circular convolution (P.
   Raviteja, Y. Hong, E. Viterbo and E. Biglieri, *IEEE Trans. Veh. Technol.* 68(1),
   2019, Theorem 1): a pilot comes out as one peak per path, at its delay and Doppler
   shift, with its gain.

``validation/tests/nissel2017.rs`` compares FBMC and OFDM in doubly-selective channels
with Nissel & Rupp (2017), and ``validation/tests/raviteja2019.rs`` checks Theorem 1
through the time-domain channel.
"""

import matplotlib.pyplot as plt
import numpy as np

import comrs
from comrs.plot import GRID, LABEL

N, K, BLOCKS = 64, 36, 2000
rng = comrs.Rng(2014)
qpsk = comrs.Constellation("psk", 4).normalized()


def symbols(n):
    return qpsk.modulate(comrs.bits(2 * n, rng))


def density_db(x):
    f, p = comrs.psd(x, 1.0, nfft=512)
    return f * N, comrs.lin2db(p / p.max())


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), width_ratios=[1.4, 1])

# 1. the densities of CP-OFDM, FBMC/OQAM and UFMC at one sample per subcarrier spacing
ofdm = comrs.Ofdm(N, K, oversampling=1, n_cp=15)
ufmc = comrs.Ufmc(N, 3, 12, 16, sidelobe_db=40.0)
fbmc = comrs.Fbmc(N, overlap=4)
active = np.r_[0 : K // 2, N - K // 2 : N]  # the K subcarriers centred on the carrier
grid = np.zeros((2 * BLOCKS, N))
grid[:, active] = comrs.oqam_stagger(symbols(K * BLOCKS), K).reshape(2 * BLOCKS, K)
for name, x in [
    ("CP-OFDM", ofdm.modulate(symbols(K * BLOCKS))),
    ("FBMC/OQAM, PHYDYAS K = 4", fbmc.modulate(grid.ravel())),
    ("UFMC, Dolph–Chebyshev L = 16", ufmc.modulate(symbols(K * BLOCKS))),
]:
    ax1.plot(*density_db(x), linewidth=1.4, label=name)
ax1.axvspan(-K / 2, K / 2, color=GRID, alpha=0.5, linewidth=0)
ax1.set(
    xlim=(-N / 2, N / 2),
    ylim=(-90, 3),
    xlabel="frequency (subcarrier spacings)",
    ylabel="density (dB, normalised)",
    title="Out-of-band emission",
)
ax1.legend(loc="lower center", fontsize=8)

# 2. a pilot through a delay–Doppler channel, on the OTFS grid
M, Nd = 32, 16
fs = M * 15e3
bin_hz = fs / (M * Nd)  # one Doppler bin, 1/(N T)
otfs = comrs.Otfs(M, Nd, n_cp=8)
paths = [(1.0, 0, 0.0), (0.6j, 4, 3 * bin_hz), (-0.4, 9, -2 * bin_hz)]
x = np.zeros(M * Nd, complex)
x[M * (Nd // 2) + 4] = 1.0  # delay 4, Doppler bin N/2
y = (otfs.effective_channel(paths, fs) @ x).reshape(Nd, M)
image = ax2.imshow(
    np.abs(y).T, origin="lower", aspect="auto", cmap="Blues", extent=(-0.5, Nd - 0.5, -0.5, M - 0.5)
)
fig.colorbar(image, ax=ax2, label="|y|")
for g, d, nu in paths:
    ax2.annotate(
        f"|h| = {abs(g):.1f}",
        (Nd // 2 + round(nu / bin_hz), 4 + d),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=8,
        color=LABEL,
    )
ax2.set(
    xlabel="Doppler bin n",
    ylabel="delay bin m",
    title="A pilot through three paths",
    ylim=(-0.5, 16.5),
)
ax2.grid(False)
fig.tight_layout()
plt.show()

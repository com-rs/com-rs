"""
OFDM synchronisation and MIMO-OFDM
==================================

**Timing and frequency.** The preamble of Schmidl & Cox (*IEEE Trans. Commun.* 45(12),
1997) has two identical halves: their correlation peaks — on a plateau as long as the
cyclic prefix — at the start of the symbol, and its phase gives the carrier frequency
offset, here in units of the subcarrier spacing. The error variance of the offset
falls as :math:`1/\\mathrm{SNR}`.

**MIMO-OFDM.** With two transmit and two receive antennas over the EPA profile, every
subcarrier carries a flat 2 × 2 MIMO channel; the detectors of the MIMO module run
subcarrier by subcarrier (Stüber et al., *Proc. IEEE* 92(2), 2004). ML (here the sphere
decoder) collects the receive diversity that the linear detectors lose.
"""

import matplotlib.pyplot as plt
import numpy as np
from _gallery_utils import mimo_ofdm_ser

import comrs
from comrs.plot import AMBER, BACKGROUND, GREY

ofdm = comrs.Ofdm(128, 100, n_cp=16)
c = comrs.Constellation("qam", 4).normalized()
pre = comrs.ofdm_preamble(ofdm, seed=3)
rng = comrs.Rng(1)
data = ofdm.modulate(c.symbols(rng.integers(4, 2 * 100)))
r = np.concatenate([np.zeros(300), pre, data])
r = r + rng.complex_noise(r.size, 0.1)
metric = comrs.schmidl_cox(r, ofdm.fft_len // 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
ax1.plot(metric, linewidth=1.4)
ax1.axvspan(300, 300 + 4 * 16, color=AMBER, alpha=0.25, label="cyclic prefix")
ax1.set(xlabel="sample", ylabel="M(d)", title="Schmidl & Cox timing metric, 10 dB", xlim=(0, 1100))
ax1.legend()


def cfo_rmse(snr_db, trials=200):
    s2 = comrs.db2lin(-snr_db)
    err = []
    for t in range(trials):
        g = comrs.Rng.stream(2, "cfo", int(snr_db + 100), t)
        eps = g.uniform(1)[0] - 0.5
        d = ofdm.modulate(c.symbols(g.integers(4, 100)))
        x = np.concatenate([np.zeros(150), pre, d])
        x = (x + g.complex_noise(x.size, s2)) * np.exp(
            2j * np.pi * eps * np.arange(x.size) / ofdm.fft_len
        )
        err.append(comrs.ofdm_synchronize(x, ofdm)["cfo"] - eps)
    return np.sqrt(np.mean(np.square(err)))


snrs = np.arange(-5.0, 26.0, 5.0)
ax2.semilogy(
    snrs, [cfo_rmse(s) for s in snrs], marker="o", markeredgecolor=BACKGROUND, label="simulated"
)
ax2.semilogy(
    snrs,
    cfo_rmse(10.0) * 10 ** (-(snrs - 10) / 20),
    color=GREY,
    linestyle=":",
    linewidth=1.2,
    label=r"$\propto 1/\sqrt{\mathrm{SNR}}$",
)
ax2.set(
    xlabel="SNR (dB)", ylabel="RMS error (subcarrier spacings)", title="frequency offset estimate"
)
ax2.legend()
fig.tight_layout()

# %%
# 2 × 2 MIMO-OFDM over EPA (1.92 MHz, 96 of 128 subcarriers), perfect CSI.
mofdm = comrs.Ofdm(128, 96, oversampling=1, n_cp=16)
ch = comrs.TdlChannel("epa", 1.92e6, n_rx=2, n_tx=2)


def ser(snr_db, method, frames=150):
    """Four OFDM symbols per frame, one channel per frame, per-subcarrier detection
    (shared with the test that checks these curves)."""
    return mimo_ofdm_ser(mofdm, ch, c, snr_db, method, frames, seed=3, condition=int(snr_db))


fig, ax = plt.subplots()
snr = np.arange(0.0, 31.0, 5.0)
for (name, m), marker in zip(
    {"ZF": "zf", "MMSE": "mmse", "MMSE-SIC": "mmse_sic", "ML (sphere)": "sphere"}.items(),
    ["o", "s", "D", "^"],
    strict=True,
):
    v = np.array([ser(s, m) for s in snr])
    keep = v > 0
    ax.semilogy(snr[keep], v[keep], marker=marker, markeredgecolor=BACKGROUND, label=name)
ax.set(
    xlabel=r"$E_s/N_0$ per receive antenna (dB)",
    ylabel="symbol error rate",
    title="2 × 2 MIMO-OFDM, QPSK, EPA",
)
ax.legend()
fig.tight_layout()

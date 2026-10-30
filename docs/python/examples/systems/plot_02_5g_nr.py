"""
5G NR: numerology, synchronisation block and reference signals
==============================================================

NR keeps the OFDM of LTE and scales it: the subcarrier spacing is
:math:`\\Delta f = 2^{\\mu} \\cdot 15` kHz, a 1 ms subframe holds :math:`2^{\\mu}` slots
of 14 symbols, and the cyclic prefix shrinks as :math:`2^{-\\mu}` (X. Lin et al.,
"5G New Radio: Unveiling the Essentials of the Next Generation Wireless Access
Technology", *IEEE Commun. Standards Mag.* 3(3), 2019, Sec. II; :math:`\\mu = 0` is the
numerology of LTE, D. Astély et al., *IEEE Commun. Mag.* 47(4), 2009). This page builds
the pieces a receiver meets first: the numerology (:func:`comrs.nr_numerology`,
:func:`comrs.nr_ofdm_modulate`), the synchronisation signal block that carries the cell
identity (:func:`comrs.nr_ssb_grid`, :func:`comrs.nr_detect_cell_id`) and the
demodulation reference signals of the data channel (:func:`comrs.nr_pdsch_dmrs`).

The sequences, the positions and the modulation follow TS 38.211 and agree with py3gpp,
an open reimplementation of the 5G functions of MATLAB (``tests/test_nr_oracle.py``):
bit for bit for the sequences and the positions, to 10⁻¹⁴ for the waveforms.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import comrs
from comrs.plot import BLUE, CRIMSON, PAPER, SKY, TEAL

# %%
# Numerology
# ----------
#
# At the sampling rate :math:`N_{\mathrm{FFT}} \Delta f`, the first symbol of every half
# subframe has a longer prefix, so that the half subframe lasts exactly 0.5 ms.

print(" mu   Δf (kHz)  slot (µs)  prefix (µs)  long prefix (µs)")
for mu in range(5):
    info = comrs.nr_numerology(mu, n_fft=4096)
    fs, cp = info["sample_rate"], info["cp_lengths"]
    slot = 1e3 / info["slots_per_subframe"]
    print(
        f"{mu:3d} {info['subcarrier_spacing'] / 1e3:9.0f} {slot:10.1f} {cp[1] / fs * 1e6:12.2f} {cp[0] / fs * 1e6:17.2f}"
    )

# %%
# The synchronisation signal block
# --------------------------------
#
# 4 symbols × 240 subcarriers: the PSS (an m-sequence of length 127, one of 3), the SSS
# (a Gold sequence of length 127, one of 336), the broadcast channel PBCH and its DMRS on
# one subcarrier in four, shifted by the cell identity (Lin et al., Sec. III-A and their
# Fig. 2). The 1008 identities are :math:`3 N_{\mathrm{ID}}^{(1)} + N_{\mathrm{ID}}^{(2)}`.

cell = 417
kinds = ["pss", "sss", "pbch", "pbch_dmrs"]
layout = np.zeros(960, int)
for i, kind in enumerate(kinds, start=1):
    layout[comrs.nr_ssb_indices(cell, kind)] = i
cmap = ListedColormap([PAPER, CRIMSON, BLUE, SKY, TEAL])
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.imshow(layout.reshape(4, 240), aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=4)
ax.set(xlabel="subcarrier", ylabel="OFDM symbol", yticks=range(4), title=f"SSB of cell {cell}")
for i, (name, c) in enumerate(
    zip(["PSS", "SSS", "PBCH", "PBCH DMRS"], cmap.colors[1:], strict=True)
):
    ax.text(1.01, 0.9 - 0.22 * i, name, color=c, transform=ax.transAxes, fontsize=8, weight="bold")
fig.tight_layout()

# %%
# Cell search
# -----------
#
# The PSS is found by correlation with the three candidates; the channel, assumed flat,
# is estimated on it; the SSS is then correlated coherently with the 336 candidates of
# that PSS. The 127 subcarriers of each sequence give 21 dB of correlation gain: with
# white noise and a random phase the identity is found down to -8 dB per resource
# element. Over a flat Rayleigh channel the failures come from the fades, whose
# probability falls only as 1/SNR (one branch of diversity): the reason why a terminal
# combines several SSBs of a burst before it decides.

rng = comrs.Rng(30)
snrs = np.arange(-12.0, 9.0, 2.0)
trials = 200
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for fading, color, label in [
    (False, BLUE, "white noise, random phase"),
    (True, CRIMSON, "flat Rayleigh fading"),
]:
    p_ok = []
    for snr in snrs:
        ok = 0
        for _ in range(trials):
            cid = int(rng.integers(1008, 1)[0])
            h = rng.complex_noise(1, 1.0)[0] if fading else np.exp(2j * np.pi * rng.uniform(1)[0])
            y = h * comrs.nr_ssb_grid(cid).T.ravel() + rng.complex_noise(960, comrs.db2lin(-snr))
            ok += comrs.nr_detect_cell_id(y.reshape(4, 240).T)[0] == cid
        p_ok.append(ok / trials)
    ax.plot(snrs, p_ok, "o-", color=color, label=label)
ax.set(xlabel="SNR per resource element (dB)", ylabel="P(correct cell identity)", ylim=(0, 1.02))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()

# %%
# Demodulation reference signals of the PDSCH
# -------------------------------------------
#
# Front-loaded (the first DMRS symbol is the third or fourth of the slot), with up to
# three additional symbols for mobility, on one subcarrier in two (configuration type 1)
# or two in six (type 2), QPSK of the Gold sequence of 3GPP (Lin et al., Sec. VI-A).
# ``pilot_table`` of the Rust module hands them to the channel estimators of
# ``com_waveform::ofdm::estimation``.

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
for ax, (ctype, add) in zip(axes, [(1, 2), (2, 3)], strict=True):
    d = comrs.nr_pdsch_dmrs(4, config_type=ctype, additional_position=add, n_id=cell)
    m = np.zeros(14 * 48)
    m[d["indices"]] = 1
    ax.imshow(
        m.reshape(14, 48).T,
        aspect="auto",
        cmap=ListedColormap([PAPER, TEAL]),
        origin="lower",
    )
    ax.set(xlabel="OFDM symbol", title=f"type {ctype}, symbols {d['symbols'].tolist()}")
axes[0].set_ylabel("subcarrier (4 resource blocks)")
fig.tight_layout()

# %%
# The waveform
# ------------
#
# :func:`comrs.nr_ofdm_modulate` maps a grid (subcarriers × symbols) to samples with
# the prefixes of the numerology; here a slot at 30 kHz on 24 resource blocks: its
# spectrum occupies :math:`12 \times 24 \times 30` kHz = 8.64 MHz.

mu, n_rb, n_fft = 1, 24, 512
k = 12 * n_rb
g = (np.sign(rng.gaussian(k * 14)) + 1j * np.sign(rng.gaussian(k * 14))).reshape(k, 14) / np.sqrt(2)
x = comrs.nr_ofdm_modulate(g, mu, n_fft)
fs = comrs.nr_numerology(mu, n_fft=n_fft)["sample_rate"]
spec = np.fft.fftshift(
    np.abs(np.fft.fft(x[: (x.size // n_fft) * n_fft].reshape(-1, n_fft), axis=1)) ** 2
).mean(0)
f = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / fs)) / 1e6
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(f, comrs.lin2db(spec / spec.max()), color=BLUE, linewidth=1)
ax.set(xlabel="frequency (MHz)", ylabel="PSD (dB)", ylim=(-40, 3))
ax.grid(alpha=0.3)
fig.tight_layout()
print(f"{x.size} samples at {fs / 1e6:.2f} MS/s: {x.size / fs * 1e3:.3f} ms")

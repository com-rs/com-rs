"""
Chromatic dispersion and its compensation
=========================================

Standard single-mode fibre spreads a 32 GBd signal over many symbols:
:math:`D = 17` ps/(nm·km) accumulate 1360 ps/nm over 80 km, and the received 16-QAM
constellation collapses into a cloud. Dispersion is an all-pass filter,
:math:`e^{j \\beta_2 \\omega^2 L / 2}`, so the receiver undoes it exactly in the
frequency domain (:func:`comrs.cdc`), or with an FIR whose length grows with the
accumulated dispersion (:func:`comrs.cd_fir_taps`).
"""

import matplotlib.pyplot as plt

import comrs

RS, SPS = 32e9, 2
FS = RS * SPS
fiber = comrs.FiberParams(n_spans=1, span_km=80.0)
c = comrs.Constellation("qam", 16).normalized()
rng = comrs.Rng(4)
_, x = comrs.symbols(c, 8192, rng)
h = comrs.pulse_shape("rrc", SPS, n_taps=2 * 128 + 1, rolloff=0.1)
tx = comrs.fir_filter(h, comrs.upsample(x, SPS))
# the span loss is compensated by a noisy amplifier: an OSNR-limited link at 20 dB SNR
rx = comrs.awgn(comrs.linear_fiber_channel(tx, fiber, 80.0, FS), 20.0, rng, fs=FS, bandwidth=RS)


def symbols(r):
    return comrs.pnorm(comrs.downsample(comrs.fir_filter(h, r), SPS))[200:-200]


cases = {
    "back to back": comrs.awgn(tx, 20.0, rng, fs=FS, bandwidth=RS),
    "80 km, no compensation": rx,
    "80 km, compensated": comrs.cdc(rx, fiber, 80.0, FS),
}
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6), sharex=True, sharey=True)
for ax, (name, r) in zip(axes, cases.items(), strict=True):
    z = symbols(r)
    ber, _, snr = comrs.fast_ber_calc(z, x[200:-200], c)
    ax.scatter(z.real, z.imag, s=2, color="#2a78d6", alpha=0.35, linewidth=0)
    ax.set(title=name, aspect="equal", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8))
    ax.text(
        0.03,
        0.03,
        f"SNR {snr:.1f} dB\nBER {ber:.1e}",
        transform=ax.transAxes,
        fontsize=8,
        color="#2b2a28",
        va="bottom",
    )
fig.tight_layout()

# %%
# The FIR alternative: OptiCommPy's length rule, and its accuracy against the exact
# compensation.
n = comrs.cd_taps_needed(fiber, 80.0, RS, FS)
taps = comrs.cd_fir_taps(fiber, 80.0, FS, max(n, 1) | 1)
z = symbols(comrs.convolve(taps, rx)[len(taps) // 2 : len(taps) // 2 + rx.size])
print(f"{n} taps for 80 km: SNR {comrs.fast_ber_calc(z, x[200:-200], c)[2]:.1f} dB")

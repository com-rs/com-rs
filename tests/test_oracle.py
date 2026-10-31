"""comrs against OptiCommPy, its test oracle.

Every deterministic function of comrs that has an OptiCommPy counterpart is compared on
the same inputs, to machine precision unless stated otherwise. The documented
deviations are tested as such:

- ``theory_ber`` of BPSK is the exact ``Q(√(2E_b/N_0))``; OptiCommPy's general PSK
  formula gives twice that value;
- ``monte_carlo_mi`` snaps the normalised transmitted symbols to the constellation;
- ``pulse_shape`` uses a symmetric grid, equal to OptiCommPy's for an even length only;
- OptiCommPy's clock recovery computes in ``complex64``;
- with a ``seed``, OptiCommPy's ``edfa`` re-seeds its generator on every call, so that
  ``manakovSSF`` adds the same ASE to both polarisations and after every span; the
  comparisons of noisy links leave its seed unset.

The outputs of OptiCommPy are recorded fixtures (``tests/oracle/<name>.json`` and
``.npz``, see ``tests/oracle.py``): every ``oracle(...)`` call below names the fixture
and the function that computes it, and OptiCommPy (GPL-3.0) is imported only inside
that function, when recording. The inputs are generated here from seeded generators, so
that both sides see the same values; the WDM field of OptiCommPy's transmitter, drawn by
its own generator, is recorded as a fixture too. The tests run from the files alone and
skip when a file is missing. To re-record, with OptiCommPy installed
(``pip install --group oracle``)::

    COMRS_RECORD_ORACLE=1 python -m pytest tests/test_oracle.py

Re-recording is a deliberate act: the commit that changes the fixtures says why and
names the version of OptiCommPy (the ``version`` field of the files).
"""

import json

import numpy as np
import oracle as recorded_oracles
import pytest
from oracle import DIR, oracle

import comrs

TOOL = "OptiCommPy"


@pytest.mark.parametrize(("ci", "outcome"), [("true", "fail"), ("", "skip")])
def test_a_missing_oracle_fails_in_ci_and_skips_elsewhere(monkeypatch, ci, outcome):
    monkeypatch.setattr(recorded_oracles, "RECORD", False)
    monkeypatch.setenv("CI", ci)
    expected = pytest.fail.Exception if outcome == "fail" else pytest.skip.Exception
    with pytest.raises(expected, match="no_such_oracle"):
        oracle("no_such_oracle", lambda: None, tool=TOOL)


def test_every_recorded_oracle_has_its_arrays():
    docs = sorted(DIR.glob("*.json"))
    assert docs, "tests/oracle/ holds no recorded oracle"
    for path in docs:
        npz = json.loads(path.read_text(encoding="utf-8")).get("npz")
        assert npz is None or (DIR / npz).exists(), f"{path.name}: {npz} is missing"


CONSTELLATIONS = [
    ("qam", 4),
    ("qam", 16),
    ("qam", 64),
    ("qam", 256),
    ("psk", 2),
    ("psk", 4),
    ("psk", 8),
    ("psk", 16),
    ("pam", 2),
    ("pam", 4),
    ("pam", 8),
]


def params(**kw):
    """An OptiCommPy parameter object."""
    from optic.utils import parameters

    p = parameters()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def noisy_symbols(kind, m, snr_db, n=4000, seed=0):
    """Transmitted and received symbols of a normalised constellation over AWGN."""
    rng = comrs.Rng(seed)
    c = comrs.Constellation(kind, m).normalized()
    _, x = comrs.symbols(c, n, rng)
    return c, x, comrs.awgn(x, snr_db, rng)


# ----------------------------------------------------------------------- modulation


@pytest.mark.parametrize(("kind", "m"), CONSTELLATIONS)
def test_gray_mapping_points_and_labels(kind, m):
    def compute():
        from optic.comm import modulation as omod

        return omod.grayMapping(m, kind).ravel()

    ours = comrs.Constellation(kind, m).points
    ref = oracle(f"gray_mapping_{kind}_{m}", compute, tool=TOOL)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_gray_code(n):
    def compute():
        from optic.comm import modulation as omod

        return [int("".join(map(str, row)), 2) for row in omod.grayCode(n)]

    ref = oracle(f"gray_code_{n}", compute, tool=TOOL)
    np.testing.assert_array_equal(np.sort(comrs.gray_code(n)), np.sort(ref))


@pytest.mark.parametrize(("kind", "m"), CONSTELLATIONS)
def test_modulate_and_demodulate(kind, m):
    rng = comrs.Rng(m)
    c = comrs.Constellation(kind, m)
    b = comrs.bits(c.bits_per_symbol * 300, rng)
    x = c.map_bits(b)
    y = x + 0.2 * rng.complex_noise(x.size) * (1 if kind != "pam" else 0)

    def compute():
        from optic.comm import modulation as omod

        return {
            "modulated": omod.modulateGray(b, m, kind).ravel(),
            "demodulated": omod.demodulateGray(y, m, kind).ravel(),
        }

    ref = oracle(f"modulate_demodulate_{kind}_{m}", compute, tool=TOOL)
    np.testing.assert_allclose(x, ref["modulated"], atol=1e-12)
    np.testing.assert_array_equal(c.demap_bits(y), ref["demodulated"])


# -------------------------------------------------------------------------- metrics


@pytest.mark.parametrize(
    ("kind", "m", "ebn0"),
    [
        ("qam", 4, 6.0),
        ("qam", 16, 10.0),
        ("qam", 64, 14.0),
        ("psk", 8, 12.0),
        ("psk", 4, 7.0),
        ("pam", 4, 12.0),
        ("pam", 8, 16.0),
    ],
)
def test_theory_ber(kind, m, ebn0):
    def compute():
        from optic.comm import metrics as om

        return float(om.theoryBER(m, ebn0, kind))

    ref = oracle(f"theory_ber_{kind}_{m}_{ebn0}", compute, tool=TOOL)
    assert comrs.theory_ber(m, ebn0, kind) == pytest.approx(ref, rel=1e-6)


def test_theory_ber_of_bpsk_is_exact_where_opticommpy_doubles_it():
    def compute():
        from optic.comm import metrics as om

        return float(om.theoryBER(2, 6.0, "psk"))

    ours = comrs.theory_ber(2, 6.0, "psk")
    assert ours == pytest.approx(comrs.qfunc(np.sqrt(2 * 10**0.6)), rel=1e-12)
    assert oracle("theory_ber_bpsk_6", compute, tool=TOOL) == pytest.approx(2 * ours, rel=1e-6)


@pytest.mark.parametrize(
    ("kind", "m", "snr"), [("qam", 16, 14.0), ("qam", 64, 20.0), ("psk", 8, 14.0), ("qam", 4, 8.0)]
)
def test_fast_ber_calc(kind, m, snr):
    c, x, y = noisy_symbols(kind, m, snr)
    y = y * np.exp(0.3j) * 1.7  # a gain and a phase that the preprocessing removes

    def compute():
        from optic.comm import metrics as om

        rb, rs, rsnr = om.fastBERcalc(y, x, m, kind)
        return [float(rb[0]), float(rs[0]), float(rsnr[0])]

    ber, ser, snr_db = comrs.fast_ber_calc(y, x, c)
    rb, rs, rsnr = oracle(f"fast_ber_calc_{kind}_{m}_{snr}", compute, tool=TOOL)
    assert ber == pytest.approx(rb, abs=1e-12)
    assert ser == pytest.approx(rs, abs=1e-12)
    assert snr_db == pytest.approx(rsnr, abs=1e-9)


@pytest.mark.parametrize(
    ("kind", "m", "snr"), [("qam", 16, 12.0), ("qam", 64, 18.0), ("psk", 8, 10.0)]
)
def test_evm(kind, m, snr):
    c, x, y = noisy_symbols(kind, m, snr)

    def compute():
        from optic.comm import metrics as om

        return {
            "with_reference": float(om.calcEVM(y, m, kind, x)[0]),
            "decided": float(om.calcEVM(y, m, kind)[0]),
        }

    ref = oracle(f"evm_{kind}_{m}_{snr}", compute, tool=TOOL)
    assert comrs.evm(y, c, tx=x) == pytest.approx(ref["with_reference"], rel=1e-9)
    # without the reference, OptiCommPy's decisions run in numba with fastmath: the
    # values agree to 1e-7 relative only
    assert comrs.evm(y, c) == pytest.approx(ref["decided"], rel=1e-6)


@pytest.mark.parametrize(
    ("kind", "m", "snr"), [("qam", 16, 8.0), ("qam", 16, 14.0), ("qam", 64, 16.0), ("psk", 8, 10.0)]
)
def test_monte_carlo_gmi(kind, m, snr):
    c, x, y = noisy_symbols(kind, m, snr)

    def compute():
        from optic.comm import metrics as om

        rg, rn = om.monteCarloGMI(y, x, m, kind)
        return [float(rg[0]), float(rn[0])]

    gmi, ngmi = comrs.monte_carlo_gmi(y, x, c)
    rg, rn = oracle(f"monte_carlo_gmi_{kind}_{m}_{snr}", compute, tool=TOOL)
    assert gmi == pytest.approx(rg, abs=1e-6)
    assert ngmi == pytest.approx(rn, abs=1e-6)


@pytest.mark.parametrize(
    ("kind", "m", "snr"), [("qam", 16, 8.0), ("qam", 16, 14.0), ("psk", 8, 10.0)]
)
def test_monte_carlo_mi_close_to_opticommpy_below_saturation(kind, m, snr):
    c, x, y = noisy_symbols(kind, m, snr, n=8000)

    def compute():
        from optic.comm import metrics as om

        return float(om.monteCarloMI(y, x, m, kind)[0])

    ours = comrs.monte_carlo_mi(y, x, c)
    ref = oracle(f"monte_carlo_mi_{kind}_{m}_{snr}", compute, tool=TOOL)
    # the snapping of the transmitted symbols changes the estimate by much less than
    # its Monte Carlo spread
    assert ours == pytest.approx(ref, abs=0.03)
    assert ours <= np.log2(m) + 1e-9


def test_information_rates_without_noise_are_log2_m():
    c, x, _ = noisy_symbols("qam", 16, 30.0)
    assert comrs.monte_carlo_mi(x, x, c) == pytest.approx(4.0, abs=1e-9)
    assert comrs.monte_carlo_gmi(x, x, c)[0] == pytest.approx(4.0, abs=1e-9)


# ------------------------------------------------------------------------ filtering


@pytest.mark.parametrize(
    ("kind", "sps", "n", "rolloff"),
    [
        ("rrc", 2, 256, 0.1),
        ("rrc", 4, 1024, 0.01),
        ("rc", 2, 512, 0.25),
        ("rc", 8, 64, 0.5),
        ("rect", 4, 1, 0.0),
        ("nrz", 8, 1, 0.0),
    ],
)
def test_pulse_shape_even_lengths(kind, sps, n, rolloff):
    def compute():
        from optic.dsp import core as ocore

        return ocore.pulseShape(params(pulseType=kind, SpS=sps, nFilterTaps=n, rollOff=rolloff))

    ref = oracle(f"pulse_shape_{kind}_{sps}_{n}_{rolloff}", compute, tool=TOOL)
    np.testing.assert_allclose(
        comrs.pulse_shape(kind, sps, n_taps=n, rolloff=rolloff), ref, rtol=1e-9, atol=1e-14
    )


def test_pulse_shape_odd_length_sits_on_the_samples():
    h = comrs.pulse_shape("rrc", 2, n_taps=1025, rolloff=0.2)
    assert np.argmax(h) == 512 and np.allclose(h, h[::-1])
    # the matched pair is Nyquist at the symbol instants: g * g has zeros every 2 samples
    g = np.convolve(h, h)
    centre = np.argmax(g)
    assert np.max(np.abs(g[centre + 2 : centre + 40 : 2])) < 2e-3 * g[centre]


@pytest.mark.parametrize("n_taps", [5, 32, 101])
def test_fir_filter(n_taps):
    rng = comrs.Rng(n_taps)
    x = rng.complex_noise(500)
    h = rng.gaussian(n_taps)

    def compute():
        from optic.dsp import core as ocore

        return ocore.firFilter(h, x)

    ref = oracle(f"fir_filter_{n_taps}", compute, tool=TOOL)
    np.testing.assert_allclose(comrs.fir_filter(h, x), ref, atol=1e-12)


@pytest.mark.parametrize("kind", ["rect", "gauss"])
def test_low_pass_fir(kind):
    def compute():
        from optic.dsp import core as ocore

        return ocore.lowPassFIR(10e9, 64e9, 101, kind)

    ref = oracle(f"low_pass_fir_{kind}", compute, tool=TOOL)
    np.testing.assert_allclose(
        comrs.low_pass_fir(10e9, 64e9, 101, kind=kind), ref, rtol=1e-10, atol=1e-15
    )


@pytest.mark.parametrize("n", [2, 4, 7, 35])
def test_moving_average(n):
    x = comrs.Rng(n).complex_noise(200)

    def compute():
        from optic.dsp import core as ocore

        return ocore.movingAverage(x.reshape(-1, 1), n).ravel()

    ref = oracle(f"moving_average_{n}", compute, tool=TOOL)
    np.testing.assert_allclose(comrs.moving_average(x, n), ref, atol=1e-12)


def test_moving_average_of_one_sample_is_the_identity():
    # OptiCommPy's movingAverage fails for N = 1 (an empty slice)
    x = comrs.Rng(0).complex_noise(50)
    np.testing.assert_allclose(comrs.moving_average(x, 1), x)


def test_upsample_and_decimate():
    x = comrs.Rng(4).complex_noise(64)
    s = np.repeat(np.array([1, -1, 1, 1, -1, -1, 1, -1] * 16, dtype=complex), 4)
    s = np.roll(s * np.tile([0.2, 0.6, 1.0, 0.6], 128), 1)

    def compute():
        from optic.dsp import core as ocore

        return {
            "upsampled": ocore.upsample(x, 4),
            "decimated": ocore.decimate(s, params(SpSin=4, SpSout=1)),
        }

    ref = oracle("upsample_decimate", compute, tool=TOOL)
    np.testing.assert_allclose(comrs.upsample(x, 4), ref["upsampled"])
    np.testing.assert_allclose(comrs.decimate(s, 4, 1), ref["decimated"])


# ----------------------------------------------------------------------- channels


@pytest.mark.parametrize("length", [1.0, 80.0, 1000.0])
def test_linear_fiber_channel(length):
    x = comrs.Rng(1).complex_noise(1024, 1e-3)
    f = comrs.FiberParams(alpha_db_km=0.2, d_ps_nm_km=17.0)

    def compute():
        from optic.models import channels as och

        return och.linearFiberChannel(x, params(L=length, alpha=0.2, D=17.0, Fc=193.1e12, Fs=64e9))

    ref = oracle(f"linear_fiber_channel_{length}", compute, tool=TOOL)
    np.testing.assert_allclose(
        comrs.linear_fiber_channel(x, f, length, 64e9), ref, rtol=1e-9, atol=1e-15
    )


@pytest.mark.parametrize(("amp", "gamma"), [("ideal", 1.3), ("none", 1.3), ("ideal", 0.0)])
def test_split_step_fourier(amp, gamma):
    rng = comrs.Rng(3)
    sps = 4
    c = comrs.Constellation("qam", 16).normalized()
    _, sym = comrs.symbols(c, 256, rng)
    x = comrs.fir_filter(
        comrs.pulse_shape("rrc", sps, n_taps=256, rolloff=0.1), comrs.upsample(sym, sps)
    )
    x = comrs.pnorm(x) * np.sqrt(comrs.dbm2w(6.0))
    f = comrs.FiberParams(
        alpha_db_km=0.2, d_ps_nm_km=17.0, gamma_w_km=gamma, span_km=50.0, n_spans=3
    )

    def compute():
        from optic.models import channels as och

        p = params(
            Ltotal=150.0,
            Lspan=50.0,
            hz=0.5,
            alpha=0.2,
            D=17.0,
            gamma=gamma,
            Fc=193.1e12,
            Fs=128e9,
            amp=None if amp == "none" else amp,
            prgsBar=False,
        )
        return och.ssfm(x, p)

    ours = comrs.Ssfm(f, step_km=0.5, amplification=amp).propagate(x, 128e9)
    ref = oracle(f"ssfm_{amp}_{gamma}", compute, tool=TOOL)
    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-12 * np.abs(ref).max())


WDM_TX = dict(
    M=16, Rs=32e9, SpS=8, nBits=4 * 1024, pulseRollOff=0.1, powerPerChannel=2.0,
    nChannels=3, wdmGridSpacing=50e9, nPolModes=2, prgsBar=False, seed=1,
)  # fmt: skip


def wdm_tx_field():
    """The dual-polarisation WDM field of OptiCommPy's transmitter (its own generator,
    seeded): recorded once, the input of the Manakov comparisons."""

    def compute():
        from optic.models.tx import simpleWDMTx

        sig, _, _ = simpleWDMTx(params(**WDM_TX))
        return sig

    return oracle("wdm_tx_field", compute, tool=TOOL)


@pytest.mark.parametrize(
    ("amp", "gamma", "step"),
    [("ideal", 1.3, None), ("none", 1.3, None), ("ideal", 1.3, 0.5), ("ideal", 0.0, None)],
)
def test_manakov_split_step_with_wdm(amp, gamma, step):
    # the WDM field of OptiCommPy's transmitter, propagated by both implementations
    sig = wdm_tx_field()
    fs = WDM_TX["Rs"] * WDM_TX["SpS"]

    def compute():
        from optic.models import channels as och

        p = params(
            Ltotal=160, Lspan=80, alpha=0.2, D=16.0, gamma=gamma, Fc=193.1e12, Fs=fs,
            amp=None if amp == "none" else amp, prgsBar=False, saveSpanN=[],
            nlprMethod=step is None, hz=step if step is not None else 0.5,
        )  # fmt: skip
        return och.manakovSSF(sig.copy(), p)

    ref = oracle(f"manakov_ssfm_{amp}_{gamma}_{step}", compute, tool=TOOL)
    f = comrs.FiberParams(
        alpha_db_km=0.2, d_ps_nm_km=16.0, gamma_w_km=gamma, span_km=80.0, n_spans=2
    )
    x, y = comrs.ManakovSsfm(
        f, step_km=step, nonlinear_step="trapezoidal", amplification=amp
    ).propagate(sig.T, fs)
    scale = np.abs(ref).max()
    np.testing.assert_allclose(x, ref[:, 0], rtol=1e-8, atol=1e-10 * scale)
    np.testing.assert_allclose(y, ref[:, 1], rtol=1e-8, atol=1e-10 * scale)


def test_the_midpoint_manakov_step_converges_faster_than_opticommpy():
    # the same WDM field over 160 km: steps of 1 km against a solution of 20 m steps
    sig = wdm_tx_field()
    fs = WDM_TX["Rs"] * WDM_TX["SpS"]
    f = comrs.FiberParams(alpha_db_km=0.2, d_ps_nm_km=16.0, gamma_w_km=1.3, span_km=80.0, n_spans=2)

    def run(step, nonlinear):
        link = comrs.ManakovSsfm(f, step_km=step, nonlinear_step=nonlinear, amplification="ideal")
        x, y = link.propagate(sig.T, fs)
        return np.concatenate([x, y])

    ref = run(0.02, "midpoint")
    err = {
        n: np.linalg.norm(run(1.0, n) - ref) / np.linalg.norm(ref)
        for n in ("trapezoidal", "midpoint")
    }
    # 7.9e-3 against 1.22e-2 (three channels with loss; 6.4 times on the lossless soliton
    # collision of validation/tests/sinkin2003.rs)
    assert err["midpoint"] < 0.75 * err["trapezoidal"], err
    assert "nonlinear_step='midpoint'" in repr(comrs.ManakovSsfm(f))
    with pytest.raises(ValueError, match="midpoint"):
        comrs.ManakovSsfm(f, nonlinear_step="euler")


@pytest.mark.parametrize("n", [1, 4, 5])
def test_wdm_grid(n):
    def compute():
        from optic.models.tx import simpleWDMTx

        tx = params(nBits=64, SpS=2, nChannels=n, wdmGridSpacing=50e9, prgsBar=False, seed=1)
        _, _, tx = simpleWDMTx(tx)
        return np.asarray(tx.wdmFreqGrid[:n], dtype=float)

    np.testing.assert_allclose(comrs.wdm_grid(n, 50e9), oracle(f"wdm_grid_{n}", compute, tool=TOOL))


def test_wdm_link_has_the_snr_of_opticommpy():
    # the same link simulated by both libraries end to end: OptiCommPy's WDM transmitter
    # (MZM-based IQ modulators driven at 0.5 V_π) and Manakov propagation with EDFAs,
    # then the same receiver (dispersion compensation, matched filter, sampling at the
    # best instant, the least-squares gain of every polarisation; OptiCommPy's `edc`
    # alone limits the SNR to 23 dB, its linear fibre of opposite dispersion is used
    # instead); comrs runs it as a chain on the Rust runner. The SNRs, estimated over
    # independent draws (OptiCommPy's side is recorded: four seeded draws of its own
    # generator), agree within their spread.
    from comrs import blocks as B

    rs, sps, n_spans = 32e9, 8, 3
    fiber = {"n_spans": n_spans, "span_km": 80, "d_ps_nm_km": 16.0, "gamma_w_km": 1.3}
    link = comrs.Chain(
        B.Source(constellation="16-QAM", n_symbols=4096, symbol_rate=rs, pols=2),
        B.WdmTx(n_channels=3, spacing_hz=50e9, sps=sps, power_dbm=comrs.var("p"),
                rolloff=0.1, mzm_scale=0.5, n_taps=1024),
        B.Manakov(),
        B.WdmRx(sps=sps),
        B.Cdc(),
        B.MatchedFilter(rolloff=0.1, n_taps=1024),
        B.Decimate(),
        B.FrameSync(start=1024, length=512, max_delay=8, symmetry=0),
    )  # fmt: skip

    def snr_of_opticommpy(power):
        from optic.dsp import core as ocore
        from optic.models import channels as och
        from optic.models.tx import simpleWDMTx

        snr = []
        for seed in range(4):
            tx = params(
                M=16, Rs=rs, SpS=sps, nBits=4 * 4096, pulseRollOff=0.1, nFilterTaps=1024,
                powerPerChannel=power, nChannels=3, wdmGridSpacing=50e9, nPolModes=2,
                prgsBar=False, seed=100 + 10 * seed,
            )  # fmt: skip
            sig, symb, tx = simpleWDMTx(tx)
            ch = params(
                Ltotal=80 * n_spans, Lspan=80, alpha=0.2, D=16.0, gamma=1.3, Fc=193.1e12,
                Fs=rs * sps, amp="edfa", NF=4.5, prgsBar=False, saveSpanN=[],
            )  # fmt: skip
            rx = och.manakovSSF(sig, ch)
            # dispersion compensation: the linear fibre of opposite dispersion
            rx = och.linearFiberChannel(
                rx, params(L=80 * n_spans, alpha=0.0, D=-16.0, Fc=193.1e12, Fs=rs * sps)
            )
            h = ocore.pulseShape(params(pulseType="rrc", SpS=sps, nFilterTaps=1024, rollOff=0.1))
            for pol in range(2):
                # the two even-length filters delay by one sample (40 dB back to back)
                z = ocore.firFilter(h / np.linalg.norm(h), rx[:, pol])[1::sps][256:-256]
                x = symb[256:-256, pol, 1]
                g = np.vdot(z, x) / np.vdot(z, z)
                snr.append(np.sum(abs(x) ** 2) / np.sum(abs(g * z - x) ** 2))
        return float(10 * np.log10(np.mean(snr)))

    for power in [-6.0, 3.0]:
        ours = link.run(
            grid={"p": [power]},
            metrics={"report": ["snr_db"], "skip_start": 256, "skip_end": 256, "scale": "ls"},
            stop={"max_trials": 4},
            fiber=fiber,
        )[0]["metrics"]["snr_db"]["value"]
        theirs = oracle(f"wdm_link_snr_{power}", lambda p=power: snr_of_opticommpy(p), tool=TOOL)
        assert abs(ours - theirs) < 0.3, (power, ours, theirs)


# ------------------------------------------------------------------------ devices


def test_mzm_and_phase_modulator():
    rng = comrs.Rng(5)
    e = rng.complex_noise(100)
    u = rng.gaussian(100)
    settings = [(-1.0, 60.0), (-2.0, 25.0), (0.3, 40.0)]

    def compute():
        from optic.models import devices as odev

        return {
            "mzm": [odev.mzm(e, u, params(Vpi=2.0, Vb=vb, ER=er)) for vb, er in settings],
            "pm": odev.pm(e, u, 3.0),
        }

    ref = oracle("mzm_phase_modulator", compute, tool=TOOL)
    for (vb, er), r in zip(settings, ref["mzm"], strict=True):
        np.testing.assert_allclose(comrs.mzm(e, u, vpi=2.0, vb=vb, er_db=er), r, atol=1e-12)
    np.testing.assert_allclose(comrs.phase_modulator(e, u, vpi=3.0), ref["pm"], atol=1e-12)


def test_iq_modulator():
    rng = comrs.Rng(6)
    e = rng.complex_noise(100)
    u = rng.complex_noise(100)

    def compute():
        from optic.models import devices as odev

        return odev.iqm(e, u, params(Vpi=2.0, VbI=-2.0, VbQ=-1.8, Vphi=1.1, ERI=30.0, ERQ=50.0))

    ref = oracle("iq_modulator", compute, tool=TOOL)
    np.testing.assert_allclose(
        comrs.iqm(e, u, vpi=2.0, vb_i=-2.0, vb_q=-1.8, vphi=1.1, er_i_db=30.0, er_q_db=50.0),
        ref,
        atol=1e-12,
    )


def test_optical_hybrid_and_coherent_receiver():
    rng = comrs.Rng(7)
    es, elo = rng.complex_noise(64), rng.complex_noise(64)

    def compute():
        from optic.models import devices as odev

        return {
            "hybrid": list(odev.opticalHybrid2x4(es, elo)),
            "receiver": odev.coherentReceiver(
                es, elo, params(Fs=64e9), params(ideal=True, Fs=64e9)
            ),
        }

    ref = oracle("hybrid_coherent_receiver", compute, tool=TOOL)
    for ours, r in zip(comrs.hybrid_2x4(es, elo), ref["hybrid"], strict=True):
        np.testing.assert_allclose(ours, r, atol=1e-12)
    ideal = comrs.Photodiode(ideal=True)
    ours = comrs.coherent_receiver(es, elo, 64e9, rng, photodiode=ideal)
    np.testing.assert_allclose(ours, ref["receiver"], atol=1e-12)
    np.testing.assert_allclose(ours, es * np.conj(elo), atol=1e-12)


def test_noiseless_photodiode_with_bandwidth_limitation():
    e = comrs.Rng(8).complex_noise(400, 1e-3)

    def compute():
        from optic.models import devices as odev

        return odev.photodiode(
            e, params(shotNoise=False, thermalNoise=False, B=20e9, N=63, Fs=64e9)
        )

    pd = comrs.Photodiode(shot_noise=False, thermal_noise=False, bandwidth_hz=20e9, n_taps=63)
    ours = pd.detect(e, 64e9, comrs.Rng(0))
    np.testing.assert_allclose(ours, oracle("photodiode_noiseless", compute, tool=TOOL), atol=1e-15)


# --------------------------------------------------------------- carrier recovery


def test_fourth_power_foe():
    _c, _x, y = noisy_symbols("qam", 16, 20.0, n=4096)
    y = y * np.exp(2j * np.pi * 300e6 * np.arange(y.size) / 32e9)

    def compute():
        from optic.dsp import carrierRecovery as ocr

        ref_out, ref_fo = ocr.fourthPowerFOE(y.reshape(-1, 1), 32e9)
        return {"out": ref_out.ravel(), "fo": float(np.ravel(ref_fo)[0])}

    ref = oracle("fourth_power_foe", compute, tool=TOOL)
    out, fo = comrs.fourth_power_foe(y, 32e9, refined=False)
    assert fo == pytest.approx(ref["fo"], rel=1e-12)
    np.testing.assert_allclose(out, ref["out"], atol=1e-9)


def test_viterbi_viterbi():
    _c, _x, y = noisy_symbols("qam", 4, 15.0, n=2000)
    y = y * np.exp(1j * (0.2 + 1e-3 * np.arange(y.size)))

    def compute():
        from optic.dsp import carrierRecovery as ocr

        return ocr.viterbi(y.reshape(-1, 1), 35, 4).ravel()

    ref = oracle("viterbi_viterbi", compute, tool=TOOL)
    np.testing.assert_allclose(comrs.viterbi_viterbi(y, n=35, m=4), ref, atol=1e-9)


def test_blind_phase_search():
    c, _x, y = noisy_symbols("qam", 16, 18.0, n=1500)
    y = y * np.exp(0.4j)
    const = c.points

    def compute():
        from optic.dsp import carrierRecovery as ocr

        return ocr.bps(y.reshape(-1, 1), 20, const, 32).ravel()

    ref = oracle("blind_phase_search", compute, tool=TOOL)
    np.testing.assert_allclose(comrs.bps(y, c, n=20, b=32), ref, atol=1e-12)


def test_decision_directed_pll_with_pilots():
    c, x, y = noisy_symbols("qam", 16, 20.0, n=1500)
    y = y * np.exp(1j * (0.1 + 2e-4 * np.arange(y.size)))
    pilots = np.arange(0, y.size, 16)
    ts, kv, t1, t2 = 1 / 32e9, 0.1, 1 / (2 * np.pi * 10e6), 1 / (2 * np.pi * 10e6)

    def compute():
        from optic.dsp import carrierRecovery as ocr

        return ocr.ddpll(
            y.reshape(-1, 1), ts, kv, t1, t2, c.points, x.reshape(-1, 1), pilots
        ).ravel()

    ref = oracle("ddpll_pilots", compute, tool=TOOL)
    ours = comrs.ddpll(
        y, c, fs=1 / ts, kv=kv, tau1=t1, tau2=t2, pilot_indices=pilots, pilot_symbols=x[pilots]
    )
    np.testing.assert_allclose(ours, ref, atol=1e-9)


def test_unwrap_matches_numpy():
    phi = comrs.Rng(9).uniform(300) * 6 - 3
    for period in [2 * np.pi, np.pi / 2]:
        np.testing.assert_allclose(
            comrs.unwrap(phi, period=period), np.unwrap(phi, period=period), atol=1e-12
        )


# ----------------------------------------------------------------- clock recovery


def test_timing_error_detectors_and_interpolator():
    rng = comrs.Rng(10)
    draws = []
    for _ in range(20):
        draws.append((rng.complex_noise(3), rng.complex_noise(4), rng.uniform(1)[0] * 2 - 1))

    def compute():
        from optic.dsp import clockRecovery as ock

        return {
            "gardner": [float(ock.gardnerTED(x3)) for x3, _, _ in draws],
            "gardner_nyquist": [float(ock.gardnerTEDnyquist(x3)) for x3, _, _ in draws],
            "cubic": [complex(ock.interpolator(x4, t)) for _, x4, t in draws],
        }

    ref = oracle("timing_error_detectors_interpolator", compute, tool=TOOL)
    for i, (x3, x4, t) in enumerate(draws):
        assert comrs.gardner_ted(x3) == pytest.approx(ref["gardner"][i], abs=1e-12)
        assert comrs.gardner_ted_nyquist(x3) == pytest.approx(ref["gardner_nyquist"][i], abs=1e-12)
        assert comrs.lagrange_cubic(x4, t) == pytest.approx(ref["cubic"][i], abs=1e-12)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # OptiCommPy's drift log on a short signal
def test_gardner_clock_recovery():
    rng = comrs.Rng(11)
    c = comrs.Constellation("qam", 4).normalized()
    _, s = comrs.symbols(c, 3000, rng)
    h = comrs.pulse_shape("rrc", 2, n_taps=128, rolloff=0.5)
    x = comrs.fir_filter(h, comrs.fir_filter(h, comrs.upsample(s, 2)))
    x = comrs.pnorm(comrs.delay_fft(x, 0.3 / 64e9, 64e9))

    def compute():
        from optic.dsp import clockRecovery as ock

        return ock.gardnerClockRecovery(x, params(kp=1e-3, ki=1e-6, isNyquist=True, maxPPM=500))

    ours, _timing = comrs.gardner_clock_recovery(x)
    ref = oracle("gardner_clock_recovery", compute, tool=TOOL)
    n = min(ours.size, ref.size)
    # OptiCommPy stores the output in complex64: agreement to single precision
    np.testing.assert_allclose(ours[:n], ref[:n], atol=2e-5)


# ------------------------------------------------------------- information theory


@pytest.mark.parametrize(
    ("kind", "m", "snr"),
    [("qam", 4, 3.0), ("qam", 16, 5.0), ("qam", 16, 12.0), ("psk", 8, 8.0), ("pam", 4, 10.0)],
)
def test_theoretical_mi_and_gmi(kind, m, snr):
    def compute():
        from optic.comm import metrics as om

        return {"mi": float(om.theoryMI(m, kind, snr)), "gmi": float(om.theoryGMI(m, kind, snr))}

    ref = oracle(f"theory_mi_gmi_{kind}_{m}_{snr}", compute, tool=TOOL)
    # OptiCommPy integrates with dblquad to an absolute tolerance of 1e-3
    assert comrs.theory_mi(m, snr, kind) == pytest.approx(ref["mi"], abs=2e-3)
    assert comrs.theory_gmi(m, snr, kind) == pytest.approx(ref["gmi"], abs=2e-3)


# --------------------------------------------------------------------- equalisation


def equalizer_params(alg):
    return params(
        nTaps=15, SpS=2, mu=[1e-3], alg=[alg], M=16, constType="qam", numIter=1,
        prgsBar=False, domain="time", prec=np.complex128,
    )  # fmt: skip


@pytest.mark.filterwarnings("ignore::Warning")
@pytest.mark.parametrize("alg", ["cma", "rde", "nlms", "dd-lms"])
def test_adaptive_equalizer(alg):
    c = comrs.Constellation("qam", 16).normalized()
    _, s = comrs.symbols(c, 4000, comrs.Rng(1))
    g = comrs.pulse_shape("rrc", 2, n_taps=128, rolloff=0.5)
    w = comrs.pnorm(comrs.fir_filter(g, comrs.fir_filter(g, comrs.upsample(s, 2))))
    w = w + 0.2 * np.roll(w, 3)

    def compute():
        from optic.dsp import equalization as oeq

        p = equalizer_params(alg)
        out = oeq.mimoAdaptEqualizer(w, p, s) if alg == "nlms" else oeq.mimoAdaptEqualizer(w, p)
        return np.asarray(out).ravel()

    ref = oracle(f"adaptive_equalizer_{alg}", compute, tool=TOOL)
    ours = comrs.adaptive_equalizer(
        w, c, algorithm=alg, n_taps=15, sps=2, mu=1e-3, reference=s if alg == "nlms" else None
    )[0][0]
    # time-domain OptiCommPy in double precision: same updates, other summation order
    assert np.linalg.norm(ours - ref) / np.linalg.norm(ref) < 1e-6


@pytest.mark.filterwarnings("ignore::Warning")
@pytest.mark.parametrize("alg", ["cma", "rde", "nlms", "dd-lms"])
def test_adaptive_equalizer_2x2(alg):
    # polarisation demultiplexing: two modes mixed by a rotation with a phase and a
    # crosstalk, the 2 x 2 butterfly of both libraries on the same input
    c = comrs.Constellation("qam", 16).normalized()
    _, s = comrs.symbols(c, 8000, comrs.Rng(1))
    s = s.reshape(2, -1)
    g = comrs.pulse_shape("rrc", 2, n_taps=128, rolloff=0.5)
    w = np.stack(
        [comrs.pnorm(comrs.fir_filter(g, comrs.fir_filter(g, comrs.upsample(m, 2)))) for m in s]
    )
    th = 0.6
    rot = np.array(
        [[np.cos(th), -np.sin(th) * np.exp(0.3j)], [np.sin(th), np.cos(th) * np.exp(0.3j)]]
    )
    x = rot @ w
    x[1] = x[1] + 0.1 * np.roll(x[0], 2)

    def compute():
        from optic.dsp import equalization as oeq

        p = equalizer_params(alg)
        out = (
            oeq.mimoAdaptEqualizer(x.T, p, s.T) if alg == "nlms" else oeq.mimoAdaptEqualizer(x.T, p)
        )
        return np.asarray(out)

    ref = oracle(f"adaptive_equalizer_2x2_{alg}", compute, tool=TOOL)
    ours = comrs.adaptive_equalizer(
        x, c, algorithm=alg, n_taps=15, sps=2, mu=1e-3, reference=s if alg == "nlms" else None
    )[0]
    assert np.linalg.norm(ours.T - ref) / np.linalg.norm(ref) < 1e-6


@pytest.mark.parametrize("snr", [6.0, 14.0])
def test_shaped_information_rates(snr):
    # Maxwell-Boltzmann 64-QAM: OptiCommPy's estimators with the priors px
    c = comrs.Constellation("qam", 64).normalized()
    px = np.asarray(comrs.maxwell_boltzmann(c, 2.0))
    rng = comrs.Rng(4)
    _, x = comrs.symbols(c, 20_000, rng, px=px)
    y = comrs.awgn(x, snr, rng)

    def compute():
        from optic.comm import metrics as om

        ref_gmi, ref_ngmi = om.monteCarloGMI(y.copy(), x.copy(), 64, "qam", px)
        ref_mi = om.monteCarloMI(y.copy(), x.copy(), 64, "qam", px)
        return {"gmi": float(ref_gmi[0]), "ngmi": float(ref_ngmi[0]), "mi": float(ref_mi[0])}

    ref = oracle(f"shaped_information_rates_{snr}", compute, tool=TOOL)
    gmi, ngmi = comrs.monte_carlo_gmi(y, x, c, px=px)
    # the same formula; the LLRs are computed in another order (log-sum-exp)
    assert gmi == pytest.approx(ref["gmi"], rel=1e-6)
    assert ngmi == pytest.approx(ref["ngmi"], rel=1e-6)
    # OptiCommPy's MI uses the normalised reference itself; comrs snaps it to the grid
    assert comrs.monte_carlo_mi(y, x, c, px=px) == pytest.approx(ref["mi"], abs=5e-3)

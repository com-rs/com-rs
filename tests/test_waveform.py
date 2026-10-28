"""Constant-envelope waveforms: CPM (MSK, GMSK) and CE-OFDM against the literature."""

import numpy as np
import pytest

import comrs

# ------------------------------------------------------------------------------- CPM


def test_msk_round_trip_is_exact_without_noise():
    msk = comrs.Cpm.msk(4)
    rng = comrs.Rng(3)
    bits = comrs.bits(500, rng)
    s = msk.modulate(bits)
    assert s.size == 2000 and s.dtype == np.complex128
    np.testing.assert_allclose(np.abs(s), 1.0, atol=1e-12)
    np.testing.assert_array_equal(msk.ml_detect(s), bits)
    np.testing.assert_array_equal(msk.differential_detect(s), bits)
    idx, states = msk.ml_detect_path(s)
    assert states.size == bits.size + 1
    np.testing.assert_array_equal(idx, bits)
    # the precoded link: the information bits are read from the phase states
    _, states = msk.ml_detect_path(msk.modulate(comrs.msk_precode(bits)))
    np.testing.assert_array_equal(comrs.msk_bits(states), bits)
    np.testing.assert_array_equal(msk.msk_detect(msk.modulate(comrs.msk_precode(bits))), bits)
    # symbol conventions: indices below M, levels ±1, ±3, …
    np.testing.assert_array_equal(msk.levels([0, 1]), [-1.0, 1.0])
    quat = comrs.Cpm("1/4", "rc", length=2, sps=4, order=4)
    np.testing.assert_array_equal(quat.levels([0, 1, 2, 3]), [-3.0, -1.0, 1.0, 3.0])
    a = [0, 3, 1, 2, 2, 0, 1, 3, 3, 0]
    np.testing.assert_array_equal(quat.ml_detect(quat.modulate(a)), a)
    assert (quat.n_phase_states, quat.n_states) == (8, 32)


def test_cpm_parameters_and_errors():
    c = comrs.Cpm((2, 8), "rec", sps=4)
    assert c.h == 0.25 and c.h_fraction == (1, 4) and c.pulse == "rec" and c.bt is None
    np.testing.assert_allclose(c.phase_states, np.pi * np.arange(8) / 4)
    g = comrs.Cpm.gmsk(0.3, length=3, sps=8)
    assert g.bt == 0.3 and g.length == 3 and g.sps == 8 and g.phase_response.size == 24
    assert "gmsk" in repr(g)
    with pytest.raises(ValueError, match="fraction"):
        comrs.Cpm("half")
    with pytest.raises(ValueError, match="order"):
        comrs.Cpm(order=8)
    with pytest.raises(ValueError, match="rec, rc or gmsk"):
        comrs.Cpm(pulse="sinc")
    with pytest.raises(ValueError, match="below 2"):
        comrs.Cpm().modulate([0, 2])
    with pytest.raises(ValueError, match="multiple"):
        comrs.Cpm(sps=4).ml_detect(np.ones(6))
    with pytest.raises(ValueError, match="binary"):
        comrs.Cpm(order=4).differential_detect(np.ones(8))


@pytest.mark.parametrize("ebn0_db", [4.0, 6.0])
def test_precoded_msk_has_the_ber_of_bpsk_over_awgn(ebn0_db):
    # Proakis & Salehi, Sec. 3.3-2: MSK is OQPSK with half-sinusoid pulses, so the
    # coherent detector of the precoded bits has the BER Q(√(2 E_b/N_0)) of BPSK
    sps = 4
    msk = comrs.Cpm.msk(sps)
    rng = comrs.Rng(1)
    bits = comrs.bits(400_000, rng)
    r = comrs.awgn_ebn0(msk.modulate(comrs.msk_precode(bits)), ebn0_db, 1, rng, sps=sps)
    _, states = msk.ml_detect_path(r)
    errors = int((comrs.msk_bits(states) != bits).sum())
    assert errors > 300
    ber = errors / bits.size
    assert ber == pytest.approx(comrs.theory_ber(2, ebn0_db, "psk"), rel=0.1)


def test_msk_power_spectrum_of_proakis():
    # Proakis & Salehi, Sec. 3.4-3 (Fig. 3.4-5): the closed form integrates to one,
    # its first null is at fT = 0.75 and its 3 dB bandwidth is about 0.59/T
    f = np.linspace(-20.0, 20.0, 400_001)
    p = comrs.msk_psd(f)
    assert np.trapezoid(p, f) == pytest.approx(1.0, abs=1e-3)
    assert comrs.msk_psd(0.0) == pytest.approx(16 / np.pi**2)
    assert comrs.msk_psd(0.25) == pytest.approx(1.0, abs=1e-6)
    assert comrs.msk_psd(0.75) == pytest.approx(0.0, abs=1e-12)
    lobe = (f > 0.6) & (f < 0.9)
    assert f[lobe][np.argmin(p[lobe])] == pytest.approx(0.75, abs=1e-4)
    half = f[(f >= 0) & (p < p[f.size // 2] / 2)][0]
    assert 2 * half == pytest.approx(0.59, abs=0.01)
    assert comrs.msk_psd(0.295) / comrs.msk_psd(0.0) == pytest.approx(0.5, abs=0.01)
    # the Welch spectrum of a random MSK signal follows the closed form within 1 dB
    # on the main lobe, with a null deeper than 25 dB at 0.75/T
    sps = 8
    msk = comrs.Cpm.msk(sps)
    s = msk.modulate(comrs.bits(400 * 64, comrs.Rng(77)))
    freqs, d = comrs.psd(s, fs=sps, nfft=64 * sps)
    want = comrs.msk_psd(freqs)
    main = want > 1e-2 * comrs.msk_psd(0.0)
    err_db = 10 * np.log10(d[main] / want[main])
    assert np.abs(err_db).max() < 1.0 and np.sqrt(np.mean(err_db**2)) < 0.3
    peak = d[freqs.size // 2]
    sel = (freqs > 0.6) & (freqs < 0.9)
    null = np.argmin(d[sel])
    assert freqs[sel][null] == pytest.approx(0.75, abs=0.02)
    assert 10 * np.log10(d[sel][null] / peak) < -25


def test_gmsk_has_a_unit_envelope_and_a_smaller_distance_than_msk():
    g = comrs.Cpm.gmsk(0.3, length=3, sps=8)
    rng = comrs.Rng(9)
    bits = comrs.bits(2000, rng)
    s = g.modulate(bits)
    assert s.size == 16_000
    np.testing.assert_allclose(np.abs(s), 1.0, atol=1e-12)
    # the phase is continuous: increments below 2πh·(M-1)·max g / sps
    phi = np.unwrap(np.angle(s))
    assert np.abs(np.diff(phi)).max() < np.pi / 8 + 1e-9
    np.testing.assert_array_equal(g.ml_detect(s), bits)
    # Aulin & Sundberg (1981) / Murota & Hirade (1981): MSK has d² = 2 and the
    # Gaussian pulse at BT = 0.3 loses a fraction of a dB
    assert comrs.Cpm.msk(4).min_distance2(max_len=3) == pytest.approx(2.0, abs=1e-3)
    d2 = comrs.Cpm.gmsk(0.3, length=3, sps=4).min_distance2(max_len=4)
    assert 1.6 < d2 < 2.0


# --------------------------------------------------------------------------- CE-OFDM


def test_ce_ofdm_round_trip_and_cyclic_prefix():
    ce = comrs.CeOfdm(32, h=0.6, oversampling=16, n_cp=5)
    assert ce.n_fft == 2 * 16 * 33 and ce.block_len == ce.n_fft + 5
    assert ce.sps == pytest.approx(ce.block_len / 32)
    c = comrs.Constellation("qam", 16).normalized()
    _, d = comrs.symbols(c, 32 * 50, comrs.Rng(2))
    x = ce.ofdm_signal(d)
    assert x.dtype == np.float64 and x.size == 50 * ce.n_fft
    assert np.mean(x**2) == pytest.approx(1.0, abs=0.05)
    s = ce.modulate(d)
    assert s.size == 50 * ce.block_len
    np.testing.assert_allclose(np.abs(s), 1.0, atol=1e-12)
    # the prefix repeats the end of the block
    np.testing.assert_allclose(s[:5], s[ce.n_fft : ce.n_fft + 5], atol=1e-12)
    # without noise the unwrapped receiver recovers the symbols despite the wraps of
    # a phase of standard deviation 2πh = 3.8 rad, which the wrapped receiver does not
    np.testing.assert_allclose(ce.demodulate(s, unwrap="stream"), d, atol=1e-8)
    assert np.abs(ce.demodulate(s, unwrap="none") - d).max() > 0.1
    theta = ce.received_phase(s, unwrap="stream")
    assert theta.size == s.size and np.abs(np.diff(theta)).max() < np.pi
    with pytest.raises(ValueError, match="multiple"):
        ce.demodulate(s[:-1])
    with pytest.raises(ValueError, match="unwrap"):
        ce.demodulate(s, unwrap="phase")
    with pytest.raises(ValueError, match="positive"):
        comrs.CeOfdm(32, h=0.0)


def test_ce_ofdm_linear_regime_snr_and_rates():
    # SNR = (2πh)² E_s/N_0 · N/(N + N_cp) per data subcarrier (Sec. III-A of
    # Thompson et al. 2008); the rates are those of the constellation over AWGN
    ce = comrs.CeOfdm(32, h=0.05, oversampling=4)
    g = (2 * np.pi * 0.05) ** 2 * 100.0
    assert ce.snr(20.0) == pytest.approx(g, rel=1e-9)
    cp = comrs.CeOfdm(32, h=0.05, oversampling=4, n_cp=ce.n_fft // 4)
    assert cp.snr(20.0) == pytest.approx(0.8 * g, rel=1e-12)
    assert cp.noise_variance(20.0) == pytest.approx(cp.block_len / 32 / 100.0)
    ebn0_db = 10 * np.log10(g / 4)
    assert ce.ser(16, 20.0) == pytest.approx(comrs.theory_ser(16, ebn0_db, "qam"), rel=1e-9)
    assert ce.ber(4, 20.0, "psk") == pytest.approx(comrs.theory_ber(4, 10 * np.log10(g / 2), "psk"))
    # eq. (15) of the paper with 4-PAM per real subcarrier (16-QAM), at a BER below
    # 1e-2 its nearest-neighbour term is within 1 % of the exact Gray BER
    h, ebn0 = 0.1, 14.0
    ce = comrs.CeOfdm(64, h=h, oversampling=4)
    esn0_db = ebn0 + 10 * np.log10(2 * 2)
    arg = 2 * np.pi * h * np.sqrt(6 * 2 / 15 * comrs.db2lin(ebn0))
    paper = 2 * 3 / (4 * 2) * comrs.qfunc(arg)
    assert 1e-4 < paper < 1e-2
    assert ce.ber(16, esn0_db) == pytest.approx(paper, rel=0.01)
    assert ce.ber(16, esn0_db) >= paper * (1 - 1e-12)


def test_ce_ofdm_16qam_link_follows_eq_15_of_thompson_2008():
    # specs/validation/thompson2008_eq15.yaml: 16-QAM, K = 126, J = 2, 2πh = 0.5,
    # E_s/N_0 = 20 dB, the simulated SER within 15 % of the linear theory
    ce = comrs.CeOfdm(126, h=0.5 / (2 * np.pi), oversampling=2)
    c = comrs.Constellation("qam", 16).normalized()
    rng = comrs.Rng(2008)
    _, d = comrs.symbols(c, 126 * 2000, rng)
    s = ce.modulate(d)
    r = s + rng.complex_noise(s.size, ce.noise_variance(20.0))
    e = ce.demodulate(r, unwrap="stream")
    errors = int((c.decide(e) != c.decide(d)).sum())
    assert errors > 1000
    ser = errors / d.size
    assert ser == pytest.approx(ce.ser(16, 20.0), rel=0.15)

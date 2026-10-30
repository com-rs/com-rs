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
    quat = comrs.Cpm("1/4", pulse="rc", length=2, sps=4, order=4)
    np.testing.assert_array_equal(quat.levels([0, 1, 2, 3]), [-3.0, -1.0, 1.0, 3.0])
    a = [0, 3, 1, 2, 2, 0, 1, 3, 3, 0]
    np.testing.assert_array_equal(quat.ml_detect(quat.modulate(a)), a)
    assert (quat.n_phase_states, quat.n_states) == (8, 32)


def test_cpm_parameters_and_errors():
    c = comrs.Cpm((2, 8), pulse="rec", sps=4)
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
    p = comrs.theory_msk_psd(f)
    assert np.trapezoid(p, f) == pytest.approx(1.0, abs=1e-3)
    assert comrs.theory_msk_psd(0.0) == pytest.approx(16 / np.pi**2)
    assert comrs.theory_msk_psd(0.25) == pytest.approx(1.0, abs=1e-6)
    assert comrs.theory_msk_psd(0.75) == pytest.approx(0.0, abs=1e-12)
    lobe = (f > 0.6) & (f < 0.9)
    assert f[lobe][np.argmin(p[lobe])] == pytest.approx(0.75, abs=1e-4)
    half = f[(f >= 0) & (p < p[f.size // 2] / 2)][0]
    assert 2 * half == pytest.approx(0.59, abs=0.01)
    assert comrs.theory_msk_psd(0.295) / comrs.theory_msk_psd(0.0) == pytest.approx(0.5, abs=0.01)
    # the Welch spectrum of a random MSK signal follows the closed form within 1 dB
    # on the main lobe, with a null deeper than 25 dB at 0.75/T
    sps = 8
    msk = comrs.Cpm.msk(sps)
    s = msk.modulate(comrs.bits(400 * 64, comrs.Rng(77)))
    freqs, d = comrs.psd(s, fs=sps, nfft=64 * sps)
    want = comrs.theory_msk_psd(freqs)
    main = want > 1e-2 * comrs.theory_msk_psd(0.0)
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
    assert ce.ser_esn0(16, 20.0) == pytest.approx(comrs.theory_ser(16, ebn0_db, "qam"), rel=1e-9)
    assert ce.ber_esn0(4, 20.0, "psk") == pytest.approx(
        comrs.theory_ber(4, 10 * np.log10(g / 2), "psk")
    )
    # eq. (15) of the paper with 4-PAM per real subcarrier (16-QAM), at a BER below
    # 1e-2 its nearest-neighbour term is within 1 % of the exact Gray BER
    h, ebn0 = 0.1, 14.0
    ce = comrs.CeOfdm(64, h=h, oversampling=4)
    esn0_db = ebn0 + 10 * np.log10(2 * 2)
    arg = 2 * np.pi * h * np.sqrt(6 * 2 / 15 * comrs.db2lin(ebn0))
    paper = 2 * 3 / (4 * 2) * comrs.qfunc(arg)
    assert 1e-4 < paper < 1e-2
    assert ce.ber_esn0(16, esn0_db) == pytest.approx(paper, rel=0.01)
    assert ce.ber_esn0(16, esn0_db) >= paper * (1 - 1e-12)


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
    assert ser == pytest.approx(ce.ser_esn0(16, 20.0), rel=0.15)


def test_ce_ofdm_low_complexity_receivers_follow_the_bussgang_analysis():
    # the basic linear (Im r) and enhanced (Im r (2 - Re r)) receivers of Ahmed and
    # Zeidler (2015): the simulated BER of QPSK within 10 % of the theory, the enhanced
    # receiver ahead at 2πh = 0.7
    ce = comrs.CeOfdm(32, h=0.7 / (2 * np.pi), oversampling=2)
    c = comrs.Constellation("qam", 4).normalized()
    rng = comrs.Rng(15)
    _, d = comrs.symbols(c, 32 * 3000, rng)
    esn0_db = 8.0 + 10 * np.log10(2)
    s = ce.modulate(d)
    r = s + rng.complex_noise(s.size, ce.noise_variance(esn0_db))
    rates = {}
    for receiver in ["linear", "enhanced"]:
        e = ce.demodulate(r, receiver=receiver)
        wrong = np.bitwise_xor(c.decide(e), c.decide(d))
        ber = float(sum(bin(int(v)).count("1") for v in wrong)) / (2 * d.size)
        assert ber == pytest.approx(ce.ber_receiver(4, esn0_db, receiver=receiver), rel=0.1)
        rates[receiver] = ber
    assert rates["enhanced"] < 0.8 * rates["linear"]
    snr = ce.subcarrier_snr(esn0_db, receiver="enhanced")
    assert snr.shape == (32,) and snr.min() > 0
    with pytest.raises(ValueError, match="receiver"):
        ce.demodulate(r, receiver="pll")


def test_ce_ofdm_gamp_decodes_where_the_phase_wraps():
    # 2πh = 1.4 (López Morales et al. 2022): the phase detector errs on wraps, GAMP not
    ce = comrs.CeOfdm(63, h=1.4 / (2 * np.pi), oversampling=4)
    c = comrs.Constellation("qam", 4).normalized()
    rng = comrs.Rng(22)
    _, d = comrs.symbols(c, 63 * 4, rng)
    s = ce.modulate(d)
    sigma2 = ce.noise_variance(10.0 + 10 * np.log10(2))
    r = s + rng.complex_noise(s.size, sigma2)
    np.testing.assert_array_equal(c.decide(ce.gamp(r, sigma2, c)), c.decide(d))
    assert (c.decide(ce.demodulate(r)) != c.decide(d)).mean() > 0.05


def test_ce_ofdm_spectrum_gain_and_cycle_slips():
    # the intermodulation powers sum to 1 and give the carrier line of the spectrum
    ce = comrs.CeOfdm(32, h=1.0 / (2 * np.pi), oversampling=4)
    p = comrs.ce_ofdm_imp_powers(1.0, 40)
    assert p.sum() == pytest.approx(1.0) and p[0] == pytest.approx(np.exp(-1.0))
    psd = ce.psd()
    assert psd.size == ce.n_fft and psd.mean() == pytest.approx(1.0)
    assert psd[0] > ce.n_fft * p[0]
    # Guerreiro et al. 2016: G = (2πh)², 3.5 dB at 2πh = 1.5; eq. (8) is the linear
    # theory of QPSK
    assert 10 * np.log10(comrs.ce_ofdm_optimum_gain(1.5)) == pytest.approx(3.52, abs=0.01)
    assert comrs.ce_ofdm_optimum_gain(1.5, np.pi) < comrs.ce_ofdm_optimum_gain(1.5)
    lin = comrs.CeOfdm(32, h=0.6 / (2 * np.pi)).ber_esn0(4, 6.0 + 10 * np.log10(2))
    assert comrs.theory_ce_ofdm_ber_optimum(0.6, 6.0) == pytest.approx(lin, rel=1e-9)
    # Pacheco and Hatzinakos 2006: σ = 1.1 doubles the band of the message
    assert comrs.pm_occupied_bandwidth(1.1) == pytest.approx(2.0, abs=0.1)
    # a cycle slip weighs as 1/k on the subcarriers (Ahmed et al. 2012)
    resp = np.abs(ce.cycle_slip_response(ce.n_fft // 2))
    assert resp.shape == (32,) and resp[0] > 10 * resp[-1]


def test_exact_gray_ber_and_phase_density():
    # Cho and Yoon 2002: 4-PAM (3Q(x) + 2Q(3x) - Q(5x))/4 with x = √(4/5 E_b/N_0)
    x = np.sqrt(0.8 * comrs.db2lin(10.0))
    want = (3 * comrs.qfunc(x) + 2 * comrs.qfunc(3 * x) - comrs.qfunc(5 * x)) / 4
    assert comrs.theory_ber_gray(4, 10.0, "pam") == pytest.approx(want, rel=1e-12)
    assert comrs.theory_ber_gray(4, 4, "qam", mq=4) == pytest.approx(comrs.theory_ber_gray(16, 4))
    # the phase of a carrier in noise: BPSK errs when |θ| > π/2, Q(√(2ρ))
    theta = np.linspace(-np.pi, np.pi, 200_001)
    pdf = comrs.theory_phase_pdf(theta, 2.0)
    assert np.trapezoid(pdf, theta) == pytest.approx(1.0, abs=1e-6)
    tail = np.trapezoid(np.where(np.abs(theta) > np.pi / 2, pdf, 0.0), theta)
    assert tail == pytest.approx(comrs.qfunc(2.0), abs=1e-4)


# --------------------------------------------------------------- chirp spread spectrum


def test_css_round_trip_orthogonality_and_simulated_ser():
    css = comrs.Css(7)
    assert (css.m, css.symbol_len) == (128, 128)
    s = css.modulate(np.arange(128))
    np.testing.assert_allclose(np.abs(s), 1.0, atol=1e-12)
    np.testing.assert_array_equal(css.demodulate(s), np.arange(128))
    # orthogonal at one sample per chip (Chiani and Elzanaty 2019, Property 3)
    x = s.reshape(128, 128)
    np.testing.assert_allclose(x @ x.conj().T / 128, np.eye(128), atol=1e-10)
    # the simulated SER against the exact eq. (16) of Afisiadis et al. 2020
    rng = comrs.Rng(4)
    sym = rng.integers(128, 20_000)
    y = css.modulate(sym)
    y = y + rng.complex_noise(y.size, 1 / comrs.db2lin(-9.0))
    ser = float(np.mean(css.demodulate(y) != sym))
    assert ser == pytest.approx(comrs.theory_css_ser(7, -9.0), rel=0.15)
    assert comrs.theory_css_ber(7, -9.0) == pytest.approx(comrs.theory_css_ser(7, -9.0) * 128 / 254)
    with pytest.raises(ValueError, match="one sample per chip"):
        comrs.Css(7, oversampling=2, waveform="fsk")


def test_css_waveforms_spectrum_and_interference():
    # Chiani and Elzanaty 2019, Table I and Property 4
    assert comrs.css_max_cross_correlation(5) == pytest.approx(0.091, abs=5e-4)
    assert abs(comrs.css_cross_correlation(4, 3, 3) - 1) < 1e-15
    f, p = comrs.css_spectral_lines(3, 40 * 8)
    assert p.sum() == pytest.approx(1 / 8, abs=1e-4) and f.size == p.size
    g = comrs.css_continuous_spectrum(3, np.array([0.0, 0.25, 2.0]))
    assert g[0] > g[2] > 0
    # Afisiadis et al. 2020: the strongest interference bin is -round(τ) mod M
    r = comrs.css_interference_pattern(7, 40, 0, 10.4)
    assert int(np.argmax(np.abs(r))) == 128 - 10
    ser = comrs.theory_css_ser_interference(9, -12.0, -3.0)
    assert comrs.theory_css_ser(9, -12.0) < ser < 1e-2
    assert comrs.theory_css_ser_interference(9, -12.0, -3.0, frame=10) > ser


# ------------------------------------------------------------ filtered multicarrier, OTFS


def test_fbmc_and_ufmc_round_trips_and_shapes():
    f = comrs.Fbmc(16, overlap=4)
    c = np.exp(2j * np.pi * np.arange(16 * 6) / 7)
    s = f.modulate(comrs.oqam_stagger(c, 16))
    assert s.size == f.n_samples(12) and (f.n_fft, f.overlap) == (16, 4)
    back = comrs.oqam_destagger(f.demodulate(s).real, 16)
    assert np.max(np.abs(back - c)) < 1e-2
    # a signal of another length is not a whole number of half-symbols
    with pytest.raises(ValueError):
        f.demodulate(s[:-1])
    with pytest.raises(ValueError):
        comrs.oqam_stagger(c[:5], 16)
    u = comrs.Ufmc(64, 4, 12, 16, sidelobe_db=40.0)
    d = np.exp(2j * np.pi * np.arange(96) / 5)
    x = u.modulate(d)
    assert (x.size, u.block_len, u.n_active) == (2 * 79, 79, 48)
    np.testing.assert_allclose(u.demodulate(x), d, atol=1e-9)
    with pytest.raises(ValueError):
        comrs.Ufmc(64, 4, 12, 66)


def test_otfs_round_trip_and_effective_channel():
    o = comrs.Otfs(16, 8, n_cp=4)
    x = np.exp(2j * np.pi * np.arange(128) / 7)
    s = o.modulate(x)
    assert s.size == o.block_len == 132
    np.testing.assert_allclose(o.demodulate(s), x, atol=1e-12)
    # one path of delay 2 and one Doppler bin: the effective channel is a permutation
    # with unit-modulus entries, and it maps the frame through the channel
    fs = 128.0
    h = o.effective_channel([(1.0 + 0.0j, 2, 1.0)], fs)
    assert h.shape == (128, 128)
    np.testing.assert_allclose(np.abs(h).sum(axis=1), 1.0, atol=1e-12)
    # r[n] = h e^{j2πν(n - l)/fs} s[n - l], the samples before the frame being zero
    n = np.arange(s.size)
    r = np.concatenate([np.zeros(2), s[:-2]]) * np.exp(2j * np.pi * 1.0 * (n - 2) / fs)
    np.testing.assert_allclose(o.demodulate(r), h @ x, atol=1e-9)
    paths = comrs.delay_doppler_paths("eva", 100.0, 15.36e6, comrs.Rng(1))
    assert len(paths) == 9 and all(abs(p[2]) <= 100.0 for p in paths)


def test_information_rates_over_fading():
    awgn = comrs.theory_mi(16, 10.0)
    rayleigh = comrs.theory_mi(16, 10.0, fading="rayleigh")
    assert rayleigh < awgn
    gray = comrs.Constellation("qam", 16)
    # one kernel, one default quadrature: the capacity of a constellation object is
    # theory_mi of its family
    assert comrs.theory_cm_capacity(gray, 10.0, fading="rayleigh") == pytest.approx(
        rayleigh, abs=1e-12
    )
    assert comrs.theory_cm_capacity(gray, 10.0, fading="rayleigh", points=16) == pytest.approx(
        comrs.theory_mi(16, 10.0, fading="rayleigh", points=16), abs=1e-9
    )
    assert comrs.theory_gmi(16, 10.0, fading="rayleigh") <= rayleigh + 1e-9
    with pytest.raises(ValueError):
        comrs.theory_mi(16, 10.0, fading="rician")

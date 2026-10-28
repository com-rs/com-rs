"""Classical performance of digital communication links, against the textbook results.

Every test reproduces a curve of the documentation gallery at a few points and checks
it against its closed form or its well-known value:

- error rates of Gray-mapped QAM, PSK and PAM over AWGN (Proakis & Salehi, ch. 4);
- BPSK over flat Rayleigh fading with coherent detection (Proakis & Salehi, ch. 13);
- the Eb/N0 needed for a BER of 1e-5 (9.6 dB for BPSK);
- the mutual information of QAM against its quadrature and the Shannon capacity;
- Nyquist pulse shaping: a root-raised-cosine pair has no intersymbol interference;
- chromatic dispersion: compensation restores the back-to-back performance;
- nonlinear propagation: digital back-propagation beats dispersion compensation
  above the optimum launch power.

A simulated error rate passes when it lies within four binomial standard errors of
theory (each point collects at least a few hundred errors, so the relative standard
error is a few percent).
"""

import math
from pathlib import Path

import numpy as np
import pytest

import comrs

MIN_ERRORS = 400
BLOCK = 20_000


def error_rates(c, ebn0_db, seed, min_errors=MIN_ERRORS):
    """Bit and symbol errors of `c` over AWGN until `min_errors` bit errors."""
    k = c.bits_per_symbol
    b_err = s_err = n_bits = n_sym = 0
    trial = 0
    while b_err < min_errors:
        rng = comrs.Rng.stream(seed, "awgn", 0, trial)
        bits = comrs.bits(k * BLOCK, rng)
        x = c.modulate(bits)
        y = comrs.awgn_ebn0(x, ebn0_db, k, rng)
        b_err += int((c.demodulate(y) != bits).sum())
        s_err += int((c.decide(y) != c.indices(bits)).sum())
        n_bits += bits.size
        n_sym += BLOCK
        trial += 1
    return b_err, n_bits, s_err, n_sym


def assert_rate(errors, total, theory, rtol=0.0, label=""):
    """`errors/total` within four binomial standard errors of `theory`, plus `rtol`."""
    p = errors / total
    sigma = math.sqrt(theory * (1 - theory) / total)
    assert abs(p - theory) <= 4 * sigma + rtol * theory, (
        f"{label}: simulated {p:.4e} ({errors}/{total}), theory {theory:.4e}"
    )


@pytest.mark.parametrize(
    ("kind", "m", "ebn0_db"),
    [
        ("qam", 4, 6.0),
        ("qam", 4, 9.0),
        ("qam", 16, 8.0),
        ("qam", 16, 11.0),
        ("qam", 64, 12.0),
        ("qam", 64, 15.0),
        ("psk", 2, 7.0),
        ("psk", 8, 11.0),
        ("pam", 4, 10.0),
    ],
)
def test_error_rates_over_awgn_follow_the_closed_forms(kind, m, ebn0_db):
    c = comrs.Constellation(kind, m).normalized()
    b_err, n_bits, s_err, n_sym = error_rates(c, ebn0_db, seed=m)
    # the SER of square QAM and PAM is exact; the PSK one is the nearest-neighbour bound
    assert_rate(
        s_err, n_sym, comrs.theory_ser(m, ebn0_db, kind), rtol=0.03 * (kind == "psk"), label="SER"
    )
    # the Gray BER counts one bit error per symbol error to a nearest neighbour: a few
    # percent optimistic for dense constellations at moderate SNR
    assert_rate(b_err, n_bits, comrs.theory_ber(m, ebn0_db, kind), rtol=0.06, label="BER")


def test_qpsk_has_the_bit_error_rate_of_bpsk():
    # Gray-mapped QPSK is two independent BPSK on the quadratures
    for ebn0_db in [0.0, 4.0, 8.0]:
        assert comrs.theory_ber(4, ebn0_db, "qam") == pytest.approx(
            comrs.theory_ber(2, ebn0_db, "psk"), rel=1e-12
        )
    b_err, n_bits, _, _ = error_rates(comrs.Constellation("qam", 4).normalized(), 7.0, seed=7)
    assert_rate(b_err, n_bits, comrs.qfunc(math.sqrt(2 * comrs.db2lin(7.0))), label="QPSK")


@pytest.mark.parametrize("ebn0_db", [0.0, 10.0, 20.0])
def test_bpsk_over_flat_rayleigh_fading(ebn0_db):
    # y = h x + w with h ~ CN(0, 1), coherent detection: Pb = (1 - sqrt(g/(1+g)))/2
    g = comrs.db2lin(ebn0_db)
    errors = total = trial = 0
    while errors < MIN_ERRORS:
        rng = comrs.Rng.stream(3, "rayleigh", 0, trial)
        bits = comrs.bits(BLOCK, rng)
        x = 1.0 - 2.0 * bits
        h = rng.complex_noise(BLOCK, 1.0)
        y = h * x + rng.complex_noise(BLOCK, 1.0 / g)
        errors += int((((np.conj(h) * y).real < 0) != (bits == 1)).sum())
        total += BLOCK
        trial += 1
    assert_rate(errors, total, 0.5 * (1 - math.sqrt(g / (1 + g))), label=f"Rayleigh {ebn0_db} dB")


def test_required_ebn0_of_bpsk_for_a_ber_of_1e_5():
    # the textbook 9.6 dB: solve Q(sqrt(2 Eb/N0)) = 1e-5 by bisection
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if comrs.theory_ber(2, mid, "psk") > 1e-5 else (lo, mid)
    assert lo == pytest.approx(9.59, abs=0.01)


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_mutual_information_of_16_qam(snr_db):
    c = comrs.Constellation("qam", 16).normalized()
    rng = comrs.Rng(int(snr_db) + 1)
    idx, x = comrs.symbols(c, 100_000, rng)
    y = comrs.awgn(x, snr_db, rng)
    r = comrs.information_rates(y, idx, c)
    mi = comrs.theory_mi(16, snr_db, "qam")
    gmi = comrs.theory_gmi(16, snr_db, "qam")
    assert r["mi"] == pytest.approx(mi, abs=0.02)
    assert r["gmi"] == pytest.approx(gmi, abs=0.02)
    # GMI ≤ MI ≤ min(log2 M, log2(1 + SNR))
    assert gmi <= mi + 1e-9
    assert mi <= min(4.0, math.log2(1 + comrs.db2lin(snr_db))) + 1e-9


def test_root_raised_cosine_pair_has_no_intersymbol_interference():
    sps, n_sym = 8, 4000
    c = comrs.Constellation("qam", 16).normalized()
    _, x = comrs.symbols(c, n_sym, comrs.Rng(5))
    h = comrs.pulse_shape("rrc", sps, n_taps=8 * 64 + 1, rolloff=0.25)
    rx = comrs.fir_filter(h, comrs.fir_filter(h, comrs.upsample(x, sps)))
    z = comrs.downsample(rx, sps)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    core = slice(100, n_sym - 100)
    # the residual ISI of the truncated pair is 40 dB below the signal
    assert comrs.snr_db(z[core], x[core]) > 40.0
    assert np.array_equal(c.decide(z[core]), c.decide(x[core]))


def test_dispersion_compensation_restores_back_to_back():
    fiber = comrs.FiberParams(alpha_db_km=0.0, n_spans=1, span_km=80.0)
    rs, sps = 32e9, 2
    fs = rs * sps
    c = comrs.Constellation("qam", 16).normalized()
    _, x = comrs.symbols(c, 4096, comrs.Rng(6))
    h = comrs.pulse_shape("rrc", sps, n_taps=2 * 128 + 1, rolloff=0.1)
    tx = comrs.fir_filter(h, comrs.upsample(x, sps))
    y = comrs.linear_fiber_channel(tx, fiber, 80.0, fs)

    def symbols(r):
        z = comrs.downsample(comrs.fir_filter(h, r), sps)
        return comrs.pnorm(z)

    core = slice(300, 4096 - 300)
    # 1360 ps/nm close the eye completely...
    assert comrs.fast_ber_calc(symbols(y)[core], x[core], c)[0] > 0.1
    # ...and the frequency-domain compensation reopens it
    ber, _, snr = comrs.fast_ber_calc(symbols(comrs.cdc(y, fiber, 80.0, fs))[core], x[core], c)
    assert ber == 0.0 and snr > 30.0


def test_back_propagation_beats_dispersion_compensation_at_high_power():
    # 16-QAM at 32 GBd over 3 x 80 km with EDFAs at 6 dBm, above the optimum launch power
    fiber = comrs.FiberParams(n_spans=3)
    link = comrs.Ssfm(fiber, step_km=1.0)
    rs, sps, n = 32e9, 4, 4096
    fs = rs * sps
    c = comrs.Constellation("qam", 16).normalized()
    rng = comrs.Rng(8)
    _, x = comrs.symbols(c, n, rng)
    h = comrs.pulse_shape("rrc", sps, n_taps=4 * 256 + 1, rolloff=0.1)
    tx = comrs.pnorm(comrs.fir_filter(h, comrs.upsample(x, sps))) * math.sqrt(comrs.dbm2w(6.0))
    y = link.propagate(tx, fs, rng)

    def snr(r):
        z = comrs.pnorm(comrs.downsample(comrs.fir_filter(h, r), sps))
        return comrs.fast_ber_calc(z[200:-200], x[200:-200], c)[2]

    snr_cdc = snr(comrs.cdc(y, fiber, fiber.total_km, fs))
    snr_dbp = snr(link.backpropagate(y, fs, steps_per_span=20))
    assert snr_dbp > snr_cdc + 3.0, f"CDC {snr_cdc:.1f} dB, DBP {snr_dbp:.1f} dB"


# --- MIMO --------------------------------------------------------------------------


def mrc_bpsk(gamma, branches):
    """BPSK with maximal-ratio combining of `branches` i.i.d. Rayleigh branches of
    mean SNR `gamma` each (Proakis & Salehi, eq. 13.4-15)."""
    mu = math.sqrt(gamma / (1 + gamma))
    return ((1 - mu) / 2) ** branches * sum(
        math.comb(branches - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(branches)
    )


def bpsk_mimo_errors(n_rx, n_tx, ebn0_db, method, seed, min_errors=MIN_ERRORS):
    """Bit errors of BPSK spatial multiplexing over i.i.d. Rayleigh fading (one channel
    per vector, unit energy per stream, noise variance 1/(Eb/N0) per antenna)."""
    c = comrs.Constellation("psk", 2)
    s2 = 1 / comrs.db2lin(ebn0_db)
    errors = total = trial = 0
    n = 20_000
    while errors < min_errors:
        rng = comrs.Rng.stream(seed, method, n_rx * 10 + n_tx, trial)
        h = comrs.mimo_channel(n_rx, n_tx, rng, n=n)
        idx = rng.integers(2, n * n_tx).reshape(n, n_tx)
        x = c.points[idx]
        w = rng.complex_noise(n * n_rx, s2).reshape(n, n_rx)
        y = np.einsum("nij,nj->ni", h, x) + w
        errors += int((comrs.mimo_detect(h, y, c, s2, method) != idx).sum())
        total += idx.size
        trial += 1
    return errors, total


@pytest.mark.parametrize(("n_rx", "n_tx"), [(2, 2), (3, 2), (4, 2)])
def test_zero_forcing_has_the_diversity_of_mrc_with_nr_minus_nt_plus_one_branches(n_rx, n_tx):
    # after ZF, the SNR of a stream is chi-square with 2(n_r - n_t + 1) degrees of freedom
    ebn0_db = 10.0
    errors, total = bpsk_mimo_errors(n_rx, n_tx, ebn0_db, "zf", seed=21)
    assert_rate(
        errors, total, mrc_bpsk(comrs.db2lin(ebn0_db), n_rx - n_tx + 1), label=f"ZF {n_rx}x{n_tx}"
    )


def test_alamouti_is_mrc_of_two_branches_at_half_power():
    # 2 x n_r Alamouti = MRC of 2 n_r branches of mean SNR (Eb/N0)/2 (3 dB array loss)
    code = comrs.Ostbc.alamouti()
    for n_rx, ebn0_db in [(1, 10.0), (2, 6.0)]:
        g = comrs.db2lin(ebn0_db)
        errors = total = trial = 0
        blocks = 10_000
        while errors < MIN_ERRORS:
            rng = comrs.Rng.stream(22, "alamouti", n_rx, trial)
            bits = comrs.bits(2 * blocks, rng)
            s = 1.0 - 2.0 * bits
            x = code.encode(s)
            h = comrs.mimo_channel(n_rx, 2, rng, n=blocks)
            y = np.concatenate([h[b] @ x[:, 2 * b : 2 * b + 2] for b in range(blocks)], axis=1)
            y = y + rng.complex_noise(y.size, 1 / g).reshape(y.shape)
            est, _ = code.decode(h, y, 1 / g)
            errors += int(((est.real < 0) != (bits == 1)).sum())
            total += bits.size
            trial += 1
        assert_rate(errors, total, mrc_bpsk(g / 2, 2 * n_rx), label=f"Alamouti 2x{n_rx}")


def test_detectors_rank_ml_sic_linear():
    # 16-QAM over 4 x 4 Rayleigh at 20 dB: ML = sphere < MMSE-SIC < MMSE < ZF
    c = comrs.Constellation("qam", 16).normalized()
    rng = comrs.Rng(23)
    n, s2 = 3000, 4 / comrs.db2lin(20.0)
    h = comrs.mimo_channel(4, 4, rng, n=n)
    idx = rng.integers(16, n * 4).reshape(n, 4)
    y = np.einsum("nij,nj->ni", h, c.points[idx]) + rng.complex_noise(n * 4, s2).reshape(n, 4)
    ser = {
        m: float((comrs.mimo_detect(h, y, c, s2, m) != idx).mean())
        for m in ["zf", "mmse", "mmse_sic", "sphere"]
    }
    assert ser["sphere"] < ser["mmse_sic"] < ser["mmse"] < ser["zf"], ser
    # the sphere decoder is exact ML: K-best with every path kept agrees with it
    sub = slice(0, 200)
    assert np.array_equal(
        comrs.mimo_detect(h[sub], y[sub], c, s2, "sphere"),
        comrs.mimo_detect(h[sub], y[sub], c, s2, "kbest", k=4096),
    )


def large_mimo(n_rx, n_tx, snr_db, rng, n=60, correlation=0.0):
    """16-QAM vectors over a channel of unit-norm columns at the SNR β Es/N0."""
    c = comrs.Constellation("qam", 16).normalized()
    n0 = n_tx / n_rx / comrs.db2lin(snr_db)
    h = comrs.mimo_channel(
        n_rx, n_tx, rng, n=n, rx_correlation=correlation, tx_correlation=correlation
    ) / np.sqrt(n_rx)
    idx = rng.integers(16, n * n_tx).reshape(n, n_tx)
    y = np.einsum("nij,nj->ni", h, c.points[idx]) + rng.complex_noise(n * n_rx, n0).reshape(n, n_rx)
    return c, h, y, idx, n0


def test_message_passing_reaches_the_io_bound_of_large_mimo():
    # 128 x 64 16-QAM at 14 dB (Jeon et al. 2018, Fig. 6(a)): LAMA and OAMP near the
    # IO bound of the state evolution, ten times below MMSE
    rng = comrs.Rng(41)
    c, h, y, idx, n0 = large_mimo(128, 64, 14.0, rng)
    ser = {
        m: float((comrs.mimo_detect(h, y, c, n0, m, iterations=8) != idx).mean())
        for m in ["mmse", "amp", "oamp"]
    }
    s2 = comrs.amp_state_evolution(c, 0.5, n0, 300)["noise_var"][-1]
    io = comrs.theory_ser(16, comrs.lin2db(1 / (4 * s2)), "qam")
    assert ser["amp"] < 3 * io and ser["oamp"] < 3 * io, (ser, io)
    assert ser["amp"] < ser["mmse"] / 5 and ser["oamp"] < ser["mmse"] / 5, ser
    # the signs of the LLRs of the decoupled channel are the decisions (bitwise and
    # symbolwise MAP may differ in a few ambiguous symbols)
    llr = comrs.mimo_llr(h[:5], y[:5], c, n0, "oamp", iterations=8)
    bits = (llr < 0).astype(int).reshape(5, 64, 4)
    labels = comrs.mimo_detect(h[:5], y[:5], c, n0, "oamp", iterations=8)
    assert np.mean(bits @ np.array([8, 4, 2, 1]) != labels) < 0.02


def test_oamp_survives_a_correlated_channel_where_amp_fails():
    # antenna correlation 0.7 at both ends (Ma & Ping 2017: OAMP holds for
    # unitarily-invariant channels, AMP needs i.i.d. entries)
    rng = comrs.Rng(42)
    c, h, y, idx, n0 = large_mimo(128, 64, 18.0, rng, n=40, correlation=0.7)
    ser = {
        m: float((comrs.mimo_detect(h, y, c, n0, m, iterations=8) != idx).mean())
        for m in ["mmse", "amp", "oamp"]
    }
    assert ser["amp"] > 0.5 and ser["oamp"] < ser["mmse"] / 2, ser


def test_ergodic_capacity_of_rayleigh_fading():
    # SISO: E log2(1 + ρ|h|²) = log2(e) e^(1/ρ) E1(1/ρ) (Lee 1990); MIMO grows like
    # min(n_t, n_r) log2 ρ at high SNR (Telatar 1999)
    special = pytest.importorskip("scipy.special")
    rng = comrs.Rng(24)
    rho = 10.0
    c = comrs.mimo_capacity(comrs.mimo_channel(1, 1, rng, n=200_000), 10.0).mean()
    exact = math.log2(math.e) * math.exp(1 / rho) * special.exp1(1 / rho)
    assert c == pytest.approx(exact, rel=0.01)
    h = comrs.mimo_channel(4, 4, rng, n=4000)
    slope = (comrs.mimo_capacity(h, 40.0).mean() - comrs.mimo_capacity(h, 30.0).mean()) / math.log2(
        10.0
    )
    assert slope == pytest.approx(4.0, abs=0.15)
    # water-filling never loses
    assert np.all(
        comrs.mimo_capacity(h, 0.0, water_filling=True) >= comrs.mimo_capacity(h, 0.0) - 1e-9
    )


# --- OFDM: PAPR, its reduction, amplifiers and predistortion -----------------------


def ofdm_blocks(ofdm, kind, m, blocks, seed):
    c = comrs.Constellation(kind, m).normalized()
    _, d = comrs.symbols(c, ofdm.n_subcarriers * blocks, comrs.Rng(seed))
    return c, d


def test_papr_ccdf_of_ofdm_follows_the_closed_forms():
    # continuous signal (L = 4): Ochiai & Imai 2001; Nyquist rate (L = 1): van Nee & Prasad
    for oversampling, oversampled in [(4, True), (1, False)]:
        ofdm = comrs.Ofdm(256, 256, oversampling=oversampling)
        _, d = ofdm_blocks(ofdm, "qam", 16, 4000, seed=31)
        p = ofdm.papr_db(ofdm.modulate(d))
        t = np.array([8.0, 9.0, 10.0]) if oversampled else np.array([7.5, 8.5, 9.5])
        emp = comrs.ccdf(p, t)
        th = comrs.theory_papr_ccdf(256, t, oversampled=oversampled)
        assert np.all(np.abs(emp / th - 1) < 0.3), (oversampling, emp, th)


def test_slm_ccdf_is_the_power_of_the_original_one():
    # U independent candidates: P(PAPR > γ) = CCDF(γ)^U (Bäuml, Fischer & Huber 1996)
    ofdm = comrs.Ofdm(256, 256)
    _, d = ofdm_blocks(ofdm, "qam", 4, 4000, seed=32)
    x, side = ofdm.modulate(d, comrs.PaprReduction.slm(4, seed=3))
    t = np.array([7.0, 7.5])
    emp = comrs.ccdf(ofdm.papr_db(x), t)
    th = comrs.theory_papr_ccdf(256, t, candidates=4)
    assert np.all(np.abs(emp / th - 1) < 0.3), (emp, th)
    assert set(side) <= {0, 1, 2, 3}


def test_dft_spread_ofdm_has_a_lower_papr():
    plain = comrs.Ofdm(256, 64)
    sc = comrs.Ofdm(256, 64, dft_spread=True)
    _, d = ofdm_blocks(plain, "qam", 4, 2000, seed=33)
    p99 = [np.percentile(o.papr_db(o.modulate(d)), 99) for o in (plain, sc)]
    # localised SC-FDMA (sinc interpolation): about 3 dB less at 1 % (Myung, Lim & Goodman 2006)
    assert p99[1] < p99[0] - 2.5, p99


def test_nyquist_rate_clipping_follows_bussgang():
    # y = α x + d with α and σ_d² of Ochiai & Imai 2002 (Gaussian samples at L = 1)
    ofdm = comrs.Ofdm(256, 256, oversampling=1)
    _, d = ofdm_blocks(ofdm, "qam", 16, 1000, seed=34)
    x = ofdm.modulate(d)
    for cr in [1.0, 3.0, 5.0]:
        y, _ = ofdm.modulate(d, comrs.PaprReduction.clip(cr))
        th = comrs.clipping_bussgang(cr)
        alpha = np.vdot(x, y).real / np.vdot(x, x).real
        sdr = 10 * np.log10(alpha**2 / np.mean(np.abs(y - alpha * x) ** 2))
        assert alpha == pytest.approx(th["gain"], abs=0.005)
        assert sdr == pytest.approx(th["sdr_db"], abs=0.3)


def test_filtering_removes_the_out_of_band_regrowth_of_clipping():
    ofdm = comrs.Ofdm(256, 150)
    _, d = ofdm_blocks(ofdm, "qam", 16, 200, seed=35)
    fs, bw = 4.0, 150 / 256

    def aclr(x):
        return comrs.aclr_db(*comrs.periodogram(x, fs, nfft=ofdm.fft_len), bw, bw)

    clipped, _ = ofdm.modulate(d, comrs.PaprReduction.clip(4.0))
    filtered, _ = ofdm.modulate(d, comrs.PaprReduction.clip_filter(4.0, iterations=4))
    assert aclr(clipped) < 35.0
    assert aclr(filtered) > 100.0
    # and the repeated filtering keeps the PAPR close to the target
    assert np.percentile(ofdm.papr_db(filtered), 99) < 6.0


def test_lossless_techniques_reduce_the_papr_tail():
    ofdm = comrs.Ofdm(128, 128)
    c, d = ofdm_blocks(ofdm, "qam", 16, 1000, seed=36)
    base = np.percentile(ofdm.papr_db(ofdm.modulate(d)), 99.9)
    for r, gain in [
        (comrs.PaprReduction.slm(16), 2.5),
        (comrs.PaprReduction.pts(4, phases=4), 2.0),
        (comrs.PaprReduction.tone_reservation(128, 12, 4.0), 2.0),
        (comrs.PaprReduction.exponential(2.0), 6.0),
    ]:
        k = ofdm.data_len(r)
        dd = d[: (d.size // k) * k]
        x, side = ofdm.modulate(dd, r)
        assert np.percentile(ofdm.papr_db(x), 99.9) < base - gain, r
        assert np.allclose(ofdm.demodulate(x, r, side=side), dd, atol=1e-6), r
    # ACE keeps every decision and reduces the tail
    x, _ = ofdm.modulate(d, comrs.PaprReduction.ace(c, 4.0))
    assert np.array_equal(c.decide(ofdm.demodulate(x)), c.decide(d))
    assert np.percentile(ofdm.papr_db(x), 99.9) < base - 1.5


def test_digital_predistortion_linearises_the_amplifier():
    # the memory-polynomial PA of Ding et al. 2004 driven by OFDM: indirect learning
    # restores the spectrum (ACLR) and the constellation (EVM)
    ofdm = comrs.Ofdm(256, 150)
    _, d = ofdm_blocks(ofdm, "qam", 16, 200, seed=37)
    x = 0.15 * ofdm.modulate(d)
    g = 1.0513 + 0.0904j
    fs, bw = 4.0, 150 / 256

    def aclr(y):
        return comrs.aclr_db(*comrs.periodogram(y, fs, nfft=ofdm.fft_len), bw, bw)

    def evm_db(y):
        r = ofdm.demodulate(y / (0.15 * g))
        a = np.vdot(d, r) / np.vdot(d, d)
        return 10 * np.log10(np.mean(np.abs(r / a - d) ** 2))

    y0 = comrs.pa_memory_polynomial(x)
    dpd = comrs.MemoryPolynomialDpd([1, 3, 5], 3)
    dpd.train_indirect(x, comrs.pa_memory_polynomial, gain=g, iterations=2)
    y1 = comrs.pa_memory_polynomial(dpd.apply(x))
    assert aclr(y1) > aclr(y0) + 15.0
    assert evm_db(y1) < evm_db(y0) - 15.0
    # the static inverse of Rapp's amplifier at 9 dB of input back-off
    xr = x / 0.15 * 10 ** (-9 / 20)
    assert aclr(comrs.pa_rapp(comrs.rapp_predistort(xr))) > aclr(comrs.pa_rapp(xr)) + 10.0


# --- channel coding ----------------------------------------------------------------


def coded_errors(code, ebn0_db, blocks, seed):
    """Bit and block error rates of BPSK with `code` over AWGN at `ebn0_db`."""
    b = comrs.bits(code.k * blocks, comrs.Rng(seed))
    c = code.encode(b)
    s2 = 1 / (2 * code.rate * comrs.db2lin(ebn0_db))
    y = (1.0 - 2.0 * c) + np.sqrt(s2) * comrs.Rng(seed + 1).gaussian(c.size)
    e = (code.decode(2 * y / s2) != b).reshape(blocks, -1)
    return e.mean(), e.any(axis=1).mean()


def test_shannon_limits_of_bpsk():
    # the classical values of the binary-input AWGN channel
    assert comrs.shannon_limit_ebn0_db(0.5) == pytest.approx(0.187, abs=0.01)
    assert comrs.shannon_limit_ebn0_db(1 / 3) == pytest.approx(-0.495, abs=0.01)


def test_soft_decision_viterbi_gains_about_two_db():
    # the (133, 171) code: soft decisions at 3 dB beat hard decisions at 5 dB
    soft, _ = coded_errors(comrs.ConvolutionalCode(2000), 3.0, 100, seed=41)
    hard, _ = coded_errors(comrs.ConvolutionalCode(2000, hard_decision=True), 5.0, 100, seed=41)
    assert 0 < soft < hard


def test_turbo_code_works_within_1_5_db_of_the_shannon_limit():
    # K = 1024, rate ≈ 1/3, 8 iterations: block error rate below 5 % at 1.0 dB, i.e.
    # 1.5 dB above the limit of -0.5 dB (Berrou, Glavieux & Thitimajshima 1993)
    ber, bler = coded_errors(comrs.TurboCode(1024), 1.0, 200, seed=42)
    assert bler < 0.05 and ber < 1e-3


def test_ldpc_and_polar_codes_approach_the_normal_approximation():
    # never below the finite-length limit, close to it (Polyanskiy, Poor & Verdú 2010)
    # a regular (3, 6) PEG code: about 1.2 dB from the limit (irregular codes do better)
    ldpc = comrs.LdpcCode.peg(1024, 512, var_degree=3, seed=1)
    for eb in [1.5, 2.0]:
        _, bler = coded_errors(ldpc, eb, 500, seed=43)
        assert (
            comrs.normal_approximation(1024, 512, [eb])[0]
            <= bler
            <= comrs.normal_approximation(1024, 512, [eb - 1.3])[0]
        )
    polar = comrs.PolarCode(256, 128, design_ebn0_db=2.0, crc="crc6", list_size=8)
    sc = comrs.PolarCode(256, 128, design_ebn0_db=2.0)
    for eb in [2.0, 2.5]:
        _, bler = coded_errors(polar, eb, 1000, seed=44)
        assert (
            comrs.normal_approximation(256, 128, [eb])[0]
            <= bler
            <= comrs.normal_approximation(256, 128, [eb - 1.0])[0]
        )
        # CRC-aided list decoding beats successive cancellation (Tal & Vardy 2015)
        assert bler < coded_errors(sc, eb, 1000, seed=44)[1] / 3


# --- OFDM links: multipath, channel estimation, synchronisation, MIMO-OFDM -----------

OFDM_N, OFDM_K, OFDM_CP, OFDM_FS = 256, 200, 16, 3.84e6


def ofdm_link_ber(ebn0_db, estimator, frames, seed):
    """QPSK OFDM over static EVA realisations (one per frame of 14 symbols): BER and the
    MSE of the channel estimate."""
    ofdm = comrs.Ofdm(OFDM_N, OFDM_K, oversampling=1, n_cp=OFDM_CP)
    ch = comrs.TdlChannel("eva", OFDM_FS)
    c = comrs.Constellation("qam", 4).normalized()
    grid = comrs.ResourceGrid(OFDM_K, 14, pattern="lattice", freq=3, time=4, seed=1)
    err = tot = 0
    mse = 0.0
    for f in range(frames):
        rng = comrs.Rng.stream(seed, "ofdm", int(ebn0_db * 10), f)
        bits = comrs.bits(2 * grid.n_data, rng)
        x = ofdm.modulate(grid.map(c.modulate(bits)))
        y, taps = ch.fade(x, rng)
        s2 = OFDM_N / (2 * OFDM_K * comrs.db2lin(ebn0_db))  # Eb/N0 per subcarrier
        rx = ofdm.demodulate(y + rng.complex_noise(y.size, s2))
        h = np.tile(taps.frequency_response(0, ofdm.bins, OFDM_N), 14)
        nv = s2 * OFDM_K / OFDM_N
        if estimator == "perfect":
            hh = h
        elif estimator == "lmmse":
            delays = [float(d) for d in taps.delays]
            hh = comrs.ofdm_estimate(
                grid, rx, nv, "lmmse", delays=delays, powers=ch.powers, fft_len=OFDM_N
            )
        elif estimator == "robust":
            hh = comrs.ofdm_estimate(grid, rx, nv, "lmmse", max_delay=OFDM_CP, fft_len=OFDM_N)
        else:
            hh = comrs.ofdm_estimate(grid, rx, nv, "linear")
        mse += float(np.mean(np.abs(hh - h) ** 2)) / frames
        err += int((c.demodulate(grid.extract(rx / hh)) != bits).sum())
        tot += bits.size
    return err / tot, mse


def test_ofdm_turns_multipath_into_flat_rayleigh_subcarriers():
    # with a prefix longer than the channel, every subcarrier is a flat Rayleigh channel:
    # Gray QPSK with perfect CSI has the BER of BPSK over Rayleigh (Proakis & Salehi 13.3)
    for eb in [5.0, 10.0]:
        ber, _ = ofdm_link_ber(eb, "perfect", 400, seed=51)
        g = comrs.db2lin(eb)
        assert ber == pytest.approx(0.5 * (1 - math.sqrt(g / (1 + g))), rel=0.15)


def test_lmmse_channel_estimation_beats_interpolation():
    # LMMSE with the true profile < robust LMMSE (uniform over the prefix) < linear
    # interpolation (van de Beek et al. 1995; Edfors et al. 1998)
    perfect, _ = ofdm_link_ber(15.0, "perfect", 150, seed=52)
    results = {e: ofdm_link_ber(15.0, e, 150, seed=52) for e in ["lmmse", "robust", "linear"]}
    assert results["lmmse"][1] < results["robust"][1] < results["linear"][1] / 2
    assert results["lmmse"][0] < 1.2 * perfect


def test_schmidl_cox_synchronisation():
    ofdm = comrs.Ofdm(128, 100, n_cp=16)
    pre = comrs.ofdm_preamble(ofdm, seed=3)
    cp = 4 * 16
    rmse = []
    for snr_db in [0.0, 10.0, 20.0]:
        s2 = 1 / comrs.db2lin(snr_db)
        err = []
        for t in range(60):
            rng = comrs.Rng.stream(53, "sync", int(snr_db), t)
            eps = rng.uniform(1)[0] - 0.5
            data = ofdm.modulate(
                comrs.Constellation("qam", 4).normalized().symbols(rng.integers(4, 100))
            )
            r = np.concatenate([np.zeros(150), pre, data])
            r = r + rng.complex_noise(r.size, s2)
            r = r * np.exp(2j * np.pi * eps * np.arange(r.size) / ofdm.fft_len)
            e = comrs.ofdm_synchronize(r, ofdm)
            assert 150 - 8 <= e["timing"] <= 150 + cp + 8
            err.append(e["cfo"] - eps)
        rmse.append(float(np.sqrt(np.mean(np.square(err)))))
    # the error falls with the SNR (variance ∝ 1/SNR, Schmidl & Cox 1997)
    assert rmse[0] > rmse[1] > rmse[2] and rmse[1] < 0.01
    assert pre.size == ofdm.block_len


def test_mimo_ofdm_ml_detection_collects_the_receive_diversity():
    # 2 × 2 spatial multiplexing over EPA: per-subcarrier ML (sphere) beats MMSE
    ofdm = comrs.Ofdm(128, 96, oversampling=1, n_cp=16)
    ch = comrs.TdlChannel("epa", 1.92e6, n_rx=2, n_tx=2)
    c = comrs.Constellation("qam", 4).normalized()
    ser = {}
    for method in ["mmse", "sphere"]:
        err = tot = 0
        for f in range(150):
            rng = comrs.Rng.stream(54, "mimo-ofdm", 0, f)
            idx = rng.integers(4, 2 * 96 * 4).reshape(2, -1)
            x = np.stack([ofdm.modulate(c.points[i]) for i in idx])
            y, taps = ch.fade(x, rng)
            s2 = 2 * 128 / (96 * comrs.db2lin(20.0))
            y = y + rng.complex_noise(y.size, s2).reshape(y.shape)
            rx = np.stack([ofdm.demodulate(r) for r in y])  # (n_rx, 4 · 96)
            h = taps.frequency_response(0, ofdm.bins, 128)  # (96, 2, 2), static
            hh = np.tile(h, (4, 1, 1))
            d = comrs.mimo_detect(hh, rx.T, c, s2 * 96 / 128, method)
            err += int((d != idx.T).sum())
            tot += idx.size
        ser[method] = err / tot
    assert ser["sphere"] < ser["mmse"] / 2, ser


# --- wireless links as chains of native blocks (the parallel Rust runner) ------------


def chain_rate(chain, metric, stop, **cond):
    """Errors and total of a rate metric of a chain at one condition."""
    res = chain.run(
        grid={k: [v] for k, v in cond.items()}, metrics=[metric], stop=stop, seed=61, name=metric
    )
    m = res[0]["metrics"][metric]
    return m["count"], m["total"]


def test_chains_reproduce_the_diversity_of_alamouti_and_zero_forcing():
    from comrs import blocks as B

    stop = {"min_errors": {"ber": 400}, "max_trials": 400}
    snr = comrs.var("snr_db")
    # QPSK: E_b/N_0 = SNR/2; Alamouti splits the power over two antennas
    alamouti = comrs.Chain(
        B.Source(constellation="QPSK", n_symbols=10_000),
        B.MimoTx(scheme="alamouti"),
        B.Rayleigh(n_rx=1, coherence=2),
        B.Awgn(snr_db=snr),
        B.MimoRx(),
    )
    e, n = chain_rate(alamouti, "ber", stop, snr_db=12.0)
    assert_rate(e, n, mrc_bpsk(comrs.db2lin(12.0) / 4, 2), label="Alamouti 2x1")
    zf = comrs.Chain(
        B.Source(constellation="QPSK", n_symbols=10_000),
        B.MimoTx(n_tx=2),
        B.Rayleigh(n_rx=3, coherence=1),
        B.Awgn(snr_db=snr),
        B.MimoRx(detector="zf"),
    )
    e, n = chain_rate(zf, "ber", stop, snr_db=12.0)
    assert_rate(e, n, mrc_bpsk(comrs.db2lin(12.0) / 4, 2), label="ZF 3x2")


def test_ofdm_chain_over_eva_with_estimated_channels():
    from comrs import blocks as B

    def link(estimation, pilots):
        return comrs.Chain(
            B.Source(constellation="QPSK", n_symbols=4_800),
            B.OfdmTx(n_fft=256, n_subcarriers=200, n_cp=18, pilots=pilots),
            B.Tdl(profile="EVA", doppler_hz=70),
            B.Awgn(snr_db=comrs.var("snr_db")),
            B.OfdmRx(estimation=estimation),
        )

    stop = {"min_trials": 300, "max_trials": 300}
    lattice = {"pattern": "lattice", "freq": 3, "time": 4}
    ber = {}
    for est, pilots in [("perfect", "none"), ("lmmse", lattice), ("ls", lattice)]:
        e, n = chain_rate(link(est, pilots), "ber", stop, snr_db=15.0)
        ber[est] = e / n
    # every subcarrier is Rayleigh with perfect CSI; estimation costs a fraction of a dB,
    # less with LMMSE than with linear interpolation
    g = comrs.db2lin(15.0) / 2
    assert ber["perfect"] == pytest.approx(mrc_bpsk(g, 1), rel=0.15), ber
    assert ber["perfect"] < ber["lmmse"] < ber["ls"] < 1.6 * ber["perfect"], ber


def test_bicm_chain_with_ldpc_over_awgn_and_fast_fading():
    from comrs import blocks as B

    code = {"type": "ldpc", "n": 1024, "k": 512, "seed": 1}

    def link(fading):
        blocks = [
            B.FecEncode(code=code, blocks=4, interleave=True),
            B.Map(constellation="16-QAM"),
        ]
        if fading:
            blocks.append(B.Rayleigh(coherence=1))
        blocks += [B.AwgnEbn0(ebn0_db=comrs.var("ebn0_db"))]
        if fading:
            blocks.append(B.MimoRx())
        blocks += [B.Demap(), B.FecDecode(code=code, interleave=True)]
        return comrs.Chain(*blocks)

    stop = {"min_trials": 25, "max_trials": 25}
    # 16-QAM BICM at rate 1/2 (2 information bits per symbol): the BICM limit of the pair
    # is near E_b/N_0 = 2.2 dB; this (3, 6) LDPC code of 1024 bits falls in its
    # waterfall between 3 and 4.5 dB
    e, n = chain_rate(link(False), "coded_ber", stop, ebn0_db=5.0)
    assert e / n < 1e-4
    e, n = chain_rate(link(False), "bler", stop, ebn0_db=3.0)
    assert e / n > 0.5
    # over fast Rayleigh fading the interleaved code collects the time diversity: clean
    # at 8.5 dB, where the coded bits still see a BER above 10 %
    e, n = chain_rate(link(True), "coded_ber", stop, ebn0_db=8.5)
    assert e / n < 1e-4
    e, n = chain_rate(link(True), "ber", stop, ebn0_db=8.5)
    assert e / n > 0.1


# --- specifications that check themselves against the literature --------------------


SPECS = Path(__file__).resolve().parents[1] / "specs"


@pytest.mark.parametrize(
    "name",
    [
        "mimo_diversity",
        pytest.param("wdm_gn", marks=pytest.mark.slow),
        pytest.param("raman_wdm", marks=pytest.mark.slow),
        "polmux_cma",
        "polmux_cma_45deg",
        "shaping_awgn",
        "pas_64qam",
    ],
)
def test_specification_reproduces_its_closed_forms(name):
    # Alamouti and zero forcing against maximal-ratio combining; the Manakov simulation
    # of a WDM link against the GN model; blind 2 x 2 polarisation demultiplexing and
    # its singularity (see the comments of the specifications)
    r = comrs.verify_spec(SPECS / f"{name}.yaml")
    failed = [c for c in r["checks"] if not c["passed"]]
    assert r["passed"], failed


# --- dual-polarisation nonlinear propagation ------------------------------------------


@pytest.mark.slow
def test_dual_polarisation_back_propagation_removes_the_self_interference():
    # one DP-16QAM channel over 10 x 80 km at 6 dBm: dispersion compensation is far
    # beyond its optimum, back-propagation with 4 steps per span recovers most of it
    from comrs import blocks as B

    link = comrs.Chain(
        tx=B.Source(constellation="16-QAM", n_symbols=4096, symbol_rate=32e9, pols=2),
        wdm=B.WdmTx(n_channels=1, sps=4, power_dbm=6.0, rolloff=0.1),
        fiber=B.Manakov(),
        rx=B.WdmRx(sps=2),
        comp=B.Cdc(),
        mf=B.MatchedFilter(rolloff=0.1),
        dec=B.Decimate(),
        sync=B.FrameSync(start=1024, length=512, max_delay=16, symmetry=0),
    )
    fiber = {"n_spans": 10, "span_km": 80, "d_ps_nm_km": 16, "gamma_w_km": 1.3}

    def snr(chain):
        res = chain.run(
            metrics={"report": ["snr_db"], "skip_start": 256, "skip_end": 256},
            stop={"max_trials": 2},
            fiber=fiber,
        )
        return res[0]["metrics"]["snr_db"]["value"]

    cdc = snr(link)
    dbp = snr(link.replace(comp=B.ManakovDbp(steps_per_span=4)))
    assert dbp > cdc + 10.0, (cdc, dbp)


# --- probabilistic shaping -----------------------------------------------------------------


def test_pas_is_bicm_without_shaping_and_gains_with_it():
    # without shaping (λ = 0), PAS with an LDPC code of rate 2/3 is the BICM of 64-QAM
    # with that code; at 3 bits per symbol, the shaped PAS needs about 1 dB less than the
    # uniform 64-QAM with a code of rate 1/2 (Böcherer, Steiner & Schulte 2015)
    from comrs import blocks as B

    def pas(lam):
        p = {"constellation": "64-QAM", "mb_lambda": lam, "n": 1536}
        return comrs.Chain(
            B.PasEncode(**p, blocks=4), B.Awgn(snr_db=comrs.var("snr")), B.PasDecode(**p)
        )

    def bicm(constellation, k):
        code = {"type": "ldpc", "n": 1536, "k": k}
        return comrs.Chain(
            B.FecEncode(code=code, blocks=4, interleave=True),
            B.Map(constellation=constellation),
            B.Awgn(snr_db=comrs.var("snr")),
            B.Demap(),
            B.FecDecode(code=code, interleave=True),
        )

    stop = {"min_trials": 40, "max_trials": 40}

    def bler(chain, snr):
        return chain_rate(chain, "bler", stop, snr=snr)

    e0, n0 = bler(pas(0.0), 14.0)
    e1, n1 = bler(bicm("64-QAM", 1024), 14.0)
    assert abs(e0 / n0 - e1 / n1) < 0.15, (e0 / n0, e1 / n1)
    e, n = bler(pas(2.55), 11.0)
    assert e / n < 0.02
    e, n = bler(bicm("64-QAM", 768), 11.0)
    assert e / n > 0.2


def test_dvbs2_waterfall_sits_below_the_threshold_of_table_13():
    """8PSK 3/5, short frames: no frame lost 0.1 dB above the QEF Es/N0 of the normal
    frames (Table 13 of EN 302 307-1), every frame lost 0.6 dB below it."""
    from comrs import blocks as B

    link = comrs.Chain(
        B.Dvbs2Tx(modulation="8PSK", rate="3/5", frame="short"),
        B.Awgn(snr_db=comrs.var("esn0_db"), reference="unit"),
        B.Dvbs2Rx(modulation="8PSK", rate="3/5", frame="short"),
    )
    thr = comrs.Dvbs2("8PSK", "3/5").threshold_db
    stop = {"min_trials": 40, "max_trials": 40}
    e, n = chain_rate(link, "bler", stop, esn0_db=thr + 0.1)
    assert e / n < 0.05, e / n
    e, n = chain_rate(link, "bler", stop, esn0_db=thr - 0.6)
    assert e == n


def test_mzm_extinction_ratio_and_field_transfer():
    # E_out = E [c_I cos θ + j c_Q sin θ], θ = π(u + V_b)/(2V_π), with c_I = √(r/(r+1)) and
    # c_Q = 1/√(r+1): the ratio of the largest to the smallest power over the bias is r
    vpi = 2.0
    biases = np.linspace(-2 * vpi, 0.0, 4001)
    for er_db in (20.0, 30.0):
        r = comrs.db2lin(er_db)
        out = np.array(
            [comrs.mzm(np.ones(1), np.zeros(1), vpi=vpi, vb=vb, er_db=er_db)[0] for vb in biases]
        )
        th = np.pi * biases / (2 * vpi)
        expected = np.sqrt(r / (r + 1)) * np.cos(th) + 1j * np.sqrt(1 / (r + 1)) * np.sin(th)
        np.testing.assert_allclose(out, expected, atol=1e-12)
        p = np.abs(out) ** 2
        assert abs(10 * np.log10(p.max() / p.min()) - er_db) < 1e-6


def test_mzm_harmonics_follow_the_jacobi_anger_expansion():
    # a sine drive V_m sin x, m = π V_m / V_π: at the quadrature point the power is
    # (1 + sin(m sin x))/2, of odd harmonics J_k(m); at the null the field is
    # sin((m/2) sin x), of odd harmonics 2 J_k(m/2)
    special = pytest.importorskip("scipy.special")
    vpi, n, cycles = 2.0, 4096, 8
    x = 2 * np.pi * cycles * np.arange(n) / n
    for m in (0.5, 1.0, 2.0):
        u = m * vpi / np.pi * np.sin(x)
        p = np.abs(comrs.mzm(np.ones(n), u, vpi=vpi, vb=-vpi / 2, er_db=np.inf)) ** 2
        field = comrs.mzm(np.ones(n), u, vpi=vpi, vb=-vpi, er_db=np.inf).real
        sp, sf = np.fft.rfft(p) * 2 / n, np.fft.rfft(field) * 2 / n
        for k in (1, 3, 5):
            assert abs(abs(sp[k * cycles]) - abs(special.jv(k, m))) < 1e-12
            assert abs(abs(sf[k * cycles]) - 2 * abs(special.jv(k, m / 2))) < 1e-12
        for k in (2, 4):
            assert abs(sp[k * cycles]) < 1e-12 and abs(sf[k * cycles]) < 1e-12


def test_iq_modulator_compression_and_arcsine_predistortion():
    # biased at the null with a 90° phase: (E/√2)[sin(π u_I/(2V_π)) + j sin(π u_Q/(2V_π))]
    vpi = 2.0
    levels = np.array([-3.0, -1.0, 1.0, 3.0]) / 3
    ui, uq = np.meshgrid(levels, levels)
    s = (ui + 1j * uq).ravel()
    ideal = dict(vpi=vpi, vb_i=-vpi, vb_q=-vpi, vphi=vpi / 2, er_i_db=np.inf, er_q_db=np.inf)
    for swing in (0.2, 0.5, 0.8):
        u = swing * vpi * s
        e = comrs.iqm(np.ones(s.size), u, **ideal)
        expected = (
            np.sin(np.pi * u.real / (2 * vpi)) + 1j * np.sin(np.pi * u.imag / (2 * vpi))
        ) / np.sqrt(2)
        np.testing.assert_allclose(e, expected, atol=1e-12)
        # the arcsine drive makes the field proportional to the target
        target = 0.9 * swing * s
        pre = 2 * vpi / np.pi * (np.arcsin(target.real) + 1j * np.arcsin(target.imag))
        np.testing.assert_allclose(comrs.iqm(np.ones(s.size), pre, **ideal),
                                   target / np.sqrt(2), atol=1e-12)  # fmt: skip

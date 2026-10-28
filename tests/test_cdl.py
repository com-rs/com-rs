"""The CDL channels of 3GPP TR 38.901 V17.0.0 (Tables 7.7.1-1 to 7.7.1-5) in Python."""

import numpy as np
import pytest

import comrs

PROFILES = ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"]


@pytest.mark.parametrize("name, n_clusters", list(zip(PROFILES, [23, 23, 24, 13, 14], strict=True)))
def test_tr_38901_tables_7_7_1_have_one_tap_per_cluster(name, n_clusters):
    delays, powers_db = comrs.cdl_profile(name, 100e-9).T
    assert len(delays) == len(powers_db) == n_clusters
    assert delays[0] == 0.0  # the clusters in the order of the table
    assert abs(np.sum(10 ** (powers_db / 10)) - 1) < 1e-12


def test_tr_38901_section_7_7_3_cdl_c_at_100_ns_has_24_taps_of_100_ns_spread():
    # Sec. 7.7.3: the normalised delays have a unit RMS delay spread (1.00000 for
    # CDL-C from the rounded entries of Table 7.7.1-3) and (7.7-1) scales them
    taps = comrs.cdl_profile("CDL-C", 100e-9)
    assert taps.shape == (24, 2)
    ds = comrs.rms_delay_spread(taps)
    assert abs(ds / 100e-9 - 1) < 0.01, ds


@pytest.mark.parametrize("name, k_db", [("CDL-D", 13.3), ("CDL-E", 22.0)])
def test_tr_38901_tables_7_7_1_4_and_5_k_factor(name, k_db):
    # the specular row against the Laplacian row of cluster 1
    assert abs(comrs.cdl_k_factor_db(name) - k_db) < 1e-9
    # and the K-factor of the first tap of the realisations: the specular component
    # is the mean over the random phases of the rays, the Laplacian part the power
    # about it (10 000 draws: 1 % on the diffuse power, 0.04 dB)
    ch = comrs.CdlChannel(name, 100e-9, 1e6)
    rng = comrs.Rng(21)
    h = np.array([ch.draw(1, rng).coefficients()[0, 0, 0, 0] for _ in range(10_000)])
    specular = abs(h.mean()) ** 2
    diffuse = np.mean(np.abs(h - h.mean()) ** 2)
    assert abs(10 * np.log10(specular / diffuse) - k_db) < 0.2


def test_profiles_without_line_of_sight_have_no_k_factor():
    for name in ["CDL-A", "cdl-b", "c"]:
        assert comrs.cdl_k_factor_db(name) is None
    with pytest.raises(ValueError):
        comrs.cdl_k_factor_db("CDL-F")
    with pytest.raises(ValueError):
        comrs.cdl_profile("CDL-A", -1.0)


def test_a_2x2_channel_has_the_right_shapes_and_a_unit_mean_gain():
    fs = 30.72e6
    ch = comrs.CdlChannel("CDL-A", 100e-9, fs, n_rx=2, n_tx=2)
    assert (ch.n_tx, ch.n_rx) == (2, 2)
    assert abs(sum(ch.powers) - 1) < 1e-12
    assert ch.delays.tolist() == [round(d * fs) for d in comrs.cdl_profile("CDL-A", 100e-9)[:, 0]]
    rng = comrs.Rng(3)
    x = rng.complex_noise(2 * 128).reshape(2, 128)
    y, taps = ch.fade(x, rng)
    assert y.shape == (2, 128)
    assert isinstance(taps, comrs.ChannelTaps)
    c = taps.coefficients()
    assert c.shape == (23, 128, 2, 2)
    assert taps.n_samples == 128
    assert len(taps.delays) == 23
    # the powers are normalised: the gain of every link, summed over the taps, is 1
    # on average (600 draws of 23 Rayleigh-like taps: a few % of standard error)
    gain = np.zeros((2, 2))
    n = 600
    for _ in range(n):
        gain += np.sum(np.abs(ch.draw(1, rng).coefficients()) ** 2, axis=0)[0]
    np.testing.assert_allclose(gain / n, 1.0, rtol=0.1)


def test_one_antenna_accepts_a_1d_signal_and_a_static_channel_is_time_invariant():
    ch = comrs.CdlChannel("CDL-B", 300e-9, 15.36e6)
    rng = comrs.Rng(5)
    x = rng.complex_noise(256)
    y, taps = ch.fade(x, rng)
    assert y.shape == (256,)
    c = taps.coefficients()
    assert c.shape == (len(ch.delays), 256, 1, 1)
    # velocity 0: every tap is constant over the samples
    assert np.all(c[:, 1:] == c[:, :1])
    assert ch.max_doppler_hz == 0.0
    # the output is the convolution of the input by the taps
    want = np.zeros(256, dtype=complex)
    for h, d in zip(c[:, 0, 0, 0], taps.delays, strict=True):
        want[d:] += h * x[: 256 - d]
    np.testing.assert_allclose(y, want, atol=1e-12)


def test_a_moving_receiver_gives_time_varying_taps_within_the_maximum_doppler():
    ch = comrs.CdlChannel("CDL-D", 30e-9, 61.44e6, velocity_mps=10.0, fc_hz=3.5e9)
    assert abs(ch.max_doppler_hz - 116.74) < 0.01  # 10 m/s at 3.5 GHz
    taps = ch.draw(64, comrs.Rng(7))
    c = taps.coefficients()
    assert not np.array_equal(c[:, 1:], c[:, :1])


def test_dual_polarized_arrays_double_the_antennas():
    ch = comrs.CdlChannel("CDL-C", 100e-9, 1e6, n_rx=2, n_tx=1, dual_polarized=True)
    assert (ch.n_tx, ch.n_rx) == (2, 4)
    y, _ = ch.fade(np.ones((2, 32)), comrs.Rng(1))
    assert y.shape == (4, 32)


def test_same_seed_same_realisation_and_errors():
    ch = comrs.CdlChannel("CDL-E", 100e-9, 1e6, n_tx=2)
    x = np.ones((2, 16))
    y1, _ = ch.fade(x, comrs.Rng(9))
    y2, _ = ch.fade(x, comrs.Rng(9))
    np.testing.assert_array_equal(y1, y2)
    with pytest.raises(ValueError):
        ch.fade(np.ones(16), comrs.Rng(0))  # one stream for two transmit antennas
    with pytest.raises(ValueError):
        comrs.CdlChannel("CDL-X", 100e-9, 1e6)
    with pytest.raises(ValueError):
        comrs.CdlChannel("CDL-A", 100e-9, 0.0)
    with pytest.raises(ValueError):
        comrs.CdlChannel("CDL-A", 100e-9, 1e6, spacing=0.0)

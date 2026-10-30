"""Optical wireless channels: the indoor channel against Barry et al. (1993) and Komine
& Nakagawa (2004), free-space optics against Al-Habash, Andrews & Phillips (2001)."""

import numpy as np
import pytest

import comrs


def test_barry_line_of_sight_and_first_reflection():
    # configuration A of Barry et al. 1993: 1.23 µW along the line of sight (Fig. 3)
    room = comrs.OwcRoom(
        [5.0, 5.0, 3.0],
        {"north": 0.8, "south": 0.8, "east": 0.8, "west": 0.8, "ceiling": 0.8, "floor": 0.3},
    )
    down = comrs.owc_direction(-np.pi / 2, 0.0)
    lit = room.illuminate([2.5, 2.5, 3.0], down, 1.0, 1.0, bin_s=0.2e-9, partitions=[[50, 50, 30]])
    h = lit.impulse_responses([0.5, 1.0, 0.0], [0, 0, 1], 1e-4, np.radians(85))
    assert len(h) == 2
    assert h[0].total_w() == pytest.approx(1.23e-6, rel=0.01)
    assert h[1].total_w() == pytest.approx(0.505e-6, rel=0.05)
    both = h[0] + h[1]
    assert both.total_w() == pytest.approx(h[0].total_w() + h[1].total_w())
    assert abs(both.frequency_response([0.0])[0]) == pytest.approx(both.total_w())
    assert both.power_w.size == h[0].power_w.size
    s, isi = both.bit_split(10e-9, both.first_arrival_s())
    assert s + isi == pytest.approx(both.total_w())


def test_komine_illuminance():
    # Fig. 3 of Komine & Nakagawa 2004: the maximum of 1211.5 lx under a light
    off = (np.arange(60) - 29.5) * 0.01
    chips = np.array(
        [
            (cx + a, cy + b, 2.5)
            for cx, cy in [(1.25, 1.25), (3.75, 1.25), (1.25, 3.75), (3.75, 3.75)]
            for a in off
            for b in off
        ]
    )
    m = comrs.lambertian_mode(np.radians(70))
    assert m == pytest.approx(0.6461, abs=1e-4)
    e = comrs.owc_illuminance(chips, [0, 0, -1], m, 0.73, [1.2, 1.2, 0.85], [0, 0, 1])
    assert 1150 < e < 1215


def test_pin_fet_receiver():
    rx = comrs.PinFetReceiver.komine2004()
    assert rx.ook_snr_db(1e-3, 0.0, 1e-4, 100e6) > rx.ook_snr_db(1e-3, 1e-4, 1e-4, 100e6)


def test_gamma_gamma_parameters_and_draws():
    g = comrs.GammaGamma.from_rytov(25.0, "plane")
    assert (g.alpha, g.beta) == pytest.approx((8.05, 1.03), abs=0.01)
    assert g.scintillation_index() == pytest.approx(1.214, abs=1e-3)
    s = comrs.GammaGamma.from_rytov(5.0 / 0.4, "spherical")
    assert (s.alpha, s.beta) == pytest.approx((2.11, 1.24), abs=0.015)
    x = g.draw(200_000, comrs.Rng(3))
    assert x.mean() == pytest.approx(1.0, abs=0.01)
    i = np.linspace(1e-4, 30, 300_000)
    assert np.trapezoid(g.pdf(i), i) == pytest.approx(1.0, abs=3e-3)
    with pytest.raises(ValueError):
        comrs.GammaGamma.from_rytov(1.0, "cylindrical")


def test_fso_link():
    p = comrs.PointingError(2.5, 0.1, 0.3)
    assert 0 < p.a0() < 1
    assert p.collected(0.0) == pytest.approx(p.a0())
    link = comrs.Fso(3.0, turbulence="lognormal", scintillation_index=0.2, pointing=p)
    h = link.gains(50_000, comrs.Rng(1))
    assert np.all(h > 0)
    assert link.scintillation_index() == pytest.approx(0.2)
    assert comrs.fog_attenuation_db_per_km(1.0, 1550) < comrs.fog_attenuation_db_per_km(1.0, 785)
    assert comrs.rytov_variance(1e-14, 1550e-9, 1000.0) > 0
    with pytest.raises(ValueError):
        comrs.Fso(turbulence="lognormal")

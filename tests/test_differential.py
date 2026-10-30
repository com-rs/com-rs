"""Differential M-PSK in Python: round trips, noncoherent and coherent detection
against the closed forms of Proakis & Salehi (5th ed., Secs. 3.2-2 and 4.5) and
Pawula, Rice & Roberts (IEEE Trans. Commun. 30(8), 1982, Sec. IV)."""

import math

import numpy as np
import pytest

import comrs


def test_encode_decode_and_detection_round_trip():
    a = [0, 1, 3, 2, 1]
    dqpsk = comrs.Dpsk(4)
    c = dqpsk.encode(a)
    assert c.tolist() == [0, 0, 1, 0, 2, 3]
    assert dqpsk.decode(c).tolist() == a
    x = np.exp(2j * np.pi * c / 4)
    assert dqpsk.detect(x * np.exp(0.7j)).tolist() == a
    assert dqpsk.detect_coherent(x).tolist() == c.tolist()
    # the bit level is the index level through the Gray labels of the transitions
    bits = dqpsk.bits(a)
    np.testing.assert_allclose(dqpsk.map_bits(bits), x, atol=1e-12)
    assert dqpsk.indices(bits).tolist() == a
    with pytest.raises(ValueError):
        dqpsk.encode([0, 4])
    with pytest.raises(ValueError):
        dqpsk.decode([])
    with pytest.raises(ValueError):
        comrs.Dpsk(6)
    with pytest.raises(ValueError):
        comrs.Dpsk(4).map_bits(np.array([1, 0, 1], dtype=np.uint8))


@pytest.mark.parametrize("m", [2, 4, 8, 16])
def test_dpsk_is_invariant_to_a_constant_rotation(m):
    d = comrs.Dpsk(m)
    assert (d.order, d.bits_per_symbol) == (m, int(math.log2(m)))
    rng = comrs.Rng(8)
    bits = comrs.bits(300 * d.bits_per_symbol, rng)
    x = d.map_bits(bits)
    assert x.shape == (301,)
    assert x.dtype == np.complex128
    np.testing.assert_allclose(np.abs(x), 1.0, atol=1e-12)
    assert x[0] == 1.0
    assert d.bits(d.indices(bits)).tolist() == bits.tolist()
    for phi in [0.1, 1.0, 2.5, -3.0]:
        np.testing.assert_array_equal(d.demap_bits(x * np.exp(1j * phi)), bits)
    np.testing.assert_array_equal(d.demodulate_coherent(x), bits)
    # Gray labelling of the transitions: neighbouring phases differ by one bit
    for k in range(m):
        b1, b2 = d.bits([k]), d.bits([(k + 1) % m])
        assert np.sum(b1 != b2) == 1


def test_proakis_dbpsk_noncoherent_ber_at_8_db():
    # Proakis & Salehi, 5th ed., Sec. 4.5: Pb = ½ exp(-Eb/N0) = 9.1e-4 at 8 dB; one
    # million bits give about 900 errors (3.3 % standard error), 10 % accepted
    d = comrs.Dpsk(2)
    rng = comrs.Rng(31)
    n = 1_000_000
    bits = comrs.bits(n, rng)
    r = comrs.awgn_ebn0(d.map_bits(bits), 8.0, 1, rng)
    got = comrs.ber(d.demap_bits(r), bits)
    want = 0.5 * math.exp(-comrs.db2lin(8.0))
    assert abs(comrs.theory_ber_dpsk(2, 8.0) - want) < 1e-12
    assert got * n > 300
    assert abs(got / want - 1) < 0.1, (got, want)


def test_pawula_rice_roberts_dqpsk_noncoherent_ber_at_8_db():
    # Pawula, Rice & Roberts 1982, Sec. IV (the differential phase of two vectors of
    # equal SNR): Gray-coded DQPSK at 8 dB has Pb = 3.6e-3; 400 000 bits give about
    # 1 460 errors (2.6 % standard error), 10 % accepted
    d = comrs.Dpsk(4)
    rng = comrs.Rng(33)
    n = 400_000
    bits = comrs.bits(n, rng)
    r = comrs.awgn_ebn0(d.map_bits(bits), 8.0, 2, rng)
    got = comrs.ber(d.demap_bits(r), bits)
    want = comrs.theory_ber_dpsk(4, 8.0)
    assert abs(got / want - 1) < 0.1, (got, want)
    # the closed form is the sum over the Gray-weighted decision sectors of the tail
    g = comrs.db2lin(8.0) * 2
    tail = comrs.theory_differential_phase_tail
    assert abs(want - 0.5 * (tail(math.pi / 4, g) + tail(3 * math.pi / 4, g))) < 1e-12
    assert comrs.theory_ber_dpsk(8, 8.0) > want > comrs.theory_ber_dpsk(2, 8.0)


def test_proakis_coherent_detection_with_differential_decoding():
    # Proakis & Salehi, 5th ed., Sec. 3.2-2: Pb = 2p(1 - p), p = Q(sqrt(2 Eb/N0)):
    # 4.8e-3 at 6 dB, about 1 900 errors in 400 000 bits (2.3 % standard error)
    d = comrs.Dpsk(2)
    rng = comrs.Rng(32)
    n = 400_000
    bits = comrs.bits(n, rng)
    r = comrs.awgn_ebn0(d.map_bits(bits), 6.0, 1, rng)
    got = comrs.ber(d.demodulate_coherent(r), bits)
    p = comrs.qfunc(math.sqrt(2 * comrs.db2lin(6.0)))
    want = 2 * p * (1 - p)
    assert abs(comrs.theory_ber_dpsk(2, 6.0, detection="coherent") - want) < 1e-15
    with pytest.raises(ValueError, match="m = 2"):
        comrs.theory_ber_dpsk(4, 6.0, detection="coherent")
    with pytest.raises(ValueError, match="detection"):
        comrs.theory_ber_dpsk(2, 6.0, detection="optimal")
    # array-likes give arrays
    assert comrs.theory_ber_dpsk(2, [4.0, 6.0], detection="coherent")[1] == pytest.approx(want)
    assert abs(got / want - 1) < 0.1, (got, want)
    # noncoherent detection is worse than coherent at the same Eb/N0
    assert comrs.theory_ber_dpsk(2, 6.0) > want

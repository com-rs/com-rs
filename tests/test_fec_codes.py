"""Reed–Solomon and BCH codes, puncturing of the convolutional and turbo codes: round
trips, erasures, the parameters of the standards, free distances and a BER sanity check
over AWGN."""

import numpy as np
import pytest
from scipy.special import erfc
from scipy.stats import binom

import comrs

# ---------------------------------------------------------------- Reed–Solomon


def corrupt(word, positions, rng, m=8):
    """Flip every symbol of `positions` to another value of GF(2^m)."""
    w = np.array(word, dtype=np.int64)
    w[positions] ^= 1 + rng.integers(2**m - 1, len(positions))
    return w


def distinct_positions(n, count, rng):
    """`count` distinct positions among `n`, drawn from `rng`."""
    positions = []
    while len(positions) < count:
        p = int(rng.integers(n, 1)[0])
        if p not in positions:
            positions.append(p)
    return positions


def test_rs_properties_and_defaults():
    code = comrs.ReedSolomonCode(8, 16)
    assert (code.n, code.k, code.t, code.m) == (255, 223, 16, 8)
    assert code.rate == pytest.approx(223 / 255)
    assert (code.first_root, code.root_step, code.poly) == (1, 1, 0x11D)
    assert code.generator.size == 33 and code.generator[-1] == 1
    short = comrs.ReedSolomonCode(4, 3, n=12)
    assert (short.n, short.k) == (12, 6)


def test_rs_symbol_round_trip_with_errors_up_to_t():
    code = comrs.ReedSolomonCode(8, 8)
    rng = comrs.Rng(1)
    msg = rng.integers(256, code.k)
    c = code.encode_symbols(msg)
    assert c.size == 255 and np.array_equal(c[: code.k], msg)
    decoded, corrected = code.decode_symbols(c)
    assert corrected == 0 and np.array_equal(decoded, msg)
    for e in range(1, code.t + 1):
        r = corrupt(c, distinct_positions(255, e, rng), rng)
        decoded, corrected = code.decode_symbols(r)
        assert corrected == e and np.array_equal(decoded, msg), f"{e} errors"


def test_rs_erasures_and_failure():
    code = comrs.ReedSolomonCode(8, 8)
    rng = comrs.Rng(2)
    msg = np.arange(code.k) * 3 % 256
    c = code.encode_symbols(msg)
    # 2t erasures and no error
    erasures = list(range(0, 2 * code.t))
    r = corrupt(c, erasures, rng)
    decoded, corrected = code.decode_symbols(r, erasures=erasures)
    assert corrected == 2 * code.t and np.array_equal(decoded, msg)
    # e errors and s erasures with 2e + s = 2t
    erasures = [10, 20, 30, 40, 50, 60]
    errors = [100, 110, 120, 130, 140]
    r = corrupt(c, erasures + errors, rng)
    decoded, corrected = code.decode_symbols(r, erasures=erasures)
    assert corrected == 11 and np.array_equal(decoded, msg)
    # 2t + 1 errors is not corrected: a failure or a miscorrection, never the message
    r = corrupt(c, list(range(17)), rng)
    decoded, corrected = code.decode_symbols(r)
    assert corrected is None or not np.array_equal(decoded, msg)
    with pytest.raises(ValueError, match="erasures"):
        code.decode_symbols(c, erasures=list(range(17)))
    with pytest.raises(ValueError):
        code.decode_symbols(c[:-1])
    with pytest.raises(ValueError):
        code.encode_symbols(np.full(code.k, 256))


def test_rs_bit_interfaces():
    code = comrs.ReedSolomonCode(4, 2, n=11)
    rng = comrs.Rng(3)
    bits = comrs.bits(2 * code.k * code.m, rng)  # two blocks
    coded = code.encode(bits)
    assert coded.dtype == np.uint8 and coded.size == 2 * code.n * code.m
    # one block of hard bits, with one symbol in error
    block = coded[: code.n * code.m]
    r = block.copy()
    r[4 * 3 : 4 * 3 + 2] ^= 1
    msg, corrected = code.decode_bits(r)
    assert corrected == 1 and np.array_equal(msg, bits[: code.k * code.m])
    # whole blocks of LLRs (hard decisions on the signs)
    llr = 3.0 * (1.0 - 2.0 * coded.astype(float))
    llr[0] = -llr[0]
    llr[-1] = -llr[-1]
    assert np.array_equal(code.decode(llr), bits)
    # the symbol packing used by the bit interface
    symbols = comrs.pack_symbols(bits[: code.k * code.m], code.m)
    assert np.array_equal(code.encode_symbols(symbols), comrs.pack_symbols(block, code.m))
    assert np.array_equal(comrs.unpack_symbols(symbols, code.m), bits[: code.k * code.m])


def test_rs_dvb_204_188_etsi_en_300_744_sec_4_3_2():
    # ETSI EN 300 744 V1.6.1, Sec. 4.3.2: RS(204, 188, t = 8), g(x) = Π (x + λ^i),
    # i = 0..15, over x^8 + x^4 + x^3 + x^2 + 1; the expansion of g(x) is the degree-16
    # polynomial of ISO/IEC 18004:2015, Table A.1, whose coefficients are the powers
    # α^120, α^104, ..., α^225, α^120 (x^15 down to x^0).
    dvb = comrs.ReedSolomonCode(8, 8, n=204, first_root=0)
    assert (dvb.n, dvb.k, dvb.t, dvb.poly) == (204, 188, 8, 0x11D)
    exponents = [120, 104, 107, 109, 102, 161, 76, 3, 91, 191, 147, 169, 182, 194, 225, 120]
    alpha = [1]
    for _ in range(254):
        a = alpha[-1] << 1
        alpha.append(a ^ 0x11D if a & 0x100 else a)
    expected = [alpha[e] for e in reversed(exponents)] + [1]
    assert dvb.generator.tolist() == expected
    # the shortened code is the (255, 239) code with 51 leading zero symbols
    full = comrs.ReedSolomonCode(8, 8, n=255, first_root=0)
    msg = np.arange(188) * 13 % 256
    padded = np.concatenate([np.zeros(51, dtype=np.int64), msg])
    assert np.array_equal(full.encode_symbols(padded)[51:], dvb.encode_symbols(msg))
    r = corrupt(dvb.encode_symbols(msg), [0, 17, 100, 150, 203, 60, 61, 62], comrs.Rng(4))
    decoded, corrected = dvb.decode_symbols(r)
    assert corrected == 8 and np.array_equal(decoded, msg)


def test_rs_ccsds_255_223_sec_4_2():
    # CCSDS 131.0-B-5, Sec. 4.2: F(x) = x^8 + x^7 + x^2 + x + 1, roots α^(11 j),
    # j = 112..143; the root set is symmetric, so g(x) is self-reciprocal.
    ccsds = comrs.ReedSolomonCode(8, 16, poly=0x187, first_root=112, root_step=11)
    assert (ccsds.n, ccsds.k, ccsds.t) == (255, 223, 16)
    g = ccsds.generator
    assert g.size == 33 and np.array_equal(g, g[::-1])
    rng = comrs.Rng(5)
    msg = rng.integers(256, 223)
    c = ccsds.encode_symbols(msg)
    r = corrupt(c, distinct_positions(255, 16, rng), rng)
    decoded, corrected = ccsds.decode_symbols(r)
    assert corrected == 16 and np.array_equal(decoded, msg)
    with pytest.raises(ValueError, match="coprime"):
        comrs.ReedSolomonCode(8, 16, root_step=3)
    with pytest.raises(ValueError):
        comrs.ReedSolomonCode(8, 16, poly=0x11F)  # not primitive


def test_rs_closed_form_error_rates():
    # the (255, 223) code: the binomial tails of Proakis & Salehi, Sec. 7.5.2
    p_s = np.array([1e-3, 1e-2, 5e-2])
    pw = comrs.theory_rs_wer(255, 16, p_s)
    expected = np.array([binom.sf(16, 255, p) for p in p_s])
    np.testing.assert_allclose(pw, expected, rtol=1e-9)
    ps_out = comrs.theory_rs_ser(255, 16, p_s)
    i = np.arange(17, 256)
    expected = np.array([np.sum(i * binom.pmf(i, 255, p)) / 255 for p in p_s])
    np.testing.assert_allclose(ps_out, expected, rtol=1e-9)
    pb = comrs.theory_rs_ber(255, 16, 8, p_s)
    np.testing.assert_allclose(pb, 128 / 255 * ps_out, rtol=1e-12)
    assert np.all(np.diff(pw) > 0) and pw[0] < 1e-15


# ------------------------------------------------------------------------- BCH


def test_bch_15_7_lin_costello_ch_6():
    code = comrs.BchCode(4, 2)
    assert (code.n, code.k, code.t) == (15, 7, 2)
    assert code.rate == pytest.approx(7 / 15)
    assert code.generator.tolist() == [1, 0, 0, 0, 1, 0, 1, 1, 1]
    rng = comrs.Rng(6)
    msg = comrs.bits(7, rng)
    c = code.encode(msg)
    assert c.size == 15 and np.array_equal(c[:7], msg)
    for errors in ([], [3], [0, 14]):
        r = c.copy()
        r[errors] ^= 1
        decoded, corrected = code.decode_bits(r)
        assert corrected == len(errors) and np.array_equal(decoded, msg)
    # three errors: a failure or a miscorrection
    r = c.copy()
    r[[1, 5, 9]] ^= 1
    decoded, corrected = code.decode_bits(r)
    assert corrected is None or not np.array_equal(decoded, msg)


def test_bch_follows_the_contract_of_the_codes():
    """Whole blocks of bits in, whole blocks of LLRs in, as every code of the package."""
    code = comrs.BchCode(4, 2)
    rng = comrs.Rng(8)
    msg = comrs.bits(5 * code.k, rng)
    c = code.encode(msg)
    assert c.size == 5 * code.n
    for i in range(5):
        np.testing.assert_array_equal(
            c[i * code.n : (i + 1) * code.n], code.encode(msg[i * code.k : (i + 1) * code.k])
        )
    r = c.copy()
    r[[0, 16, 17, 40]] ^= 1  # 1, 2 and 1 errors in words 0, 1 and 2
    np.testing.assert_array_equal(code.decode(2.0 * (1.0 - 2.0 * r)), msg)
    with pytest.raises(ValueError):
        code.decode(np.ones(code.n + 1))


def test_bch_shortened_dvbs2_outer_code():
    # ETSI EN 302 307-1 V1.4.1, Table 5a: the short frame BCH of 12 errors shortened to
    # (3240, 3072) for the rate 1/2, over GF(2^14)
    code = comrs.BchCode(14, 12, n=3240)
    assert (code.n, code.k, code.t) == (3240, 3072, 12)
    rng = comrs.Rng(7)
    msg = comrs.bits(3072, rng)
    c = code.encode(msg)
    positions = distinct_positions(3240, 12, rng)
    r = c.copy()
    r[positions] ^= 1
    decoded, corrected = code.decode_bits(r)
    assert corrected == 12 and np.array_equal(decoded, msg)
    with pytest.raises(ValueError):
        comrs.BchCode(4, 2, n=20)
    with pytest.raises(ValueError):
        code.decode_bits(c[:-1])


# ------------------------------------------------------------------- puncturing

YASUDA_TABLE_I = {  # Yasuda, Kashiki & Hirata, IEEE Trans. Commun. 32(3), 1984, Table I
    "dvbs_2_3": 6,
    "dvbs_3_4": 5,
    "dvbs_5_6": 4,
    "dvbs_7_8": 3,
    "ieee80211_2_3": 6,
    "ieee80211_3_4": 5,
    "ieee80211_5_6": 4,
}


@pytest.mark.parametrize(("pattern", "d_free"), sorted(YASUDA_TABLE_I.items()))
def test_free_distance_yasuda_1984_table_i(pattern, d_free):
    assert comrs.free_distance(puncture=pattern) == d_free
    assert comrs.ConvolutionalCode(64, puncture=pattern).free_distance() == d_free


def test_free_distance_of_the_mother_code_and_custom_patterns():
    assert comrs.free_distance() == 10  # the (133, 171) code, Table I of Yasuda et al.
    assert comrs.ConvolutionalCode(10).free_distance() == 10
    assert comrs.free_distance(puncture=[[1, 0], [1, 1]]) == 5
    assert comrs.free_distance([0o7, 0o5], 3) == 5  # the (7, 5) code
    with pytest.raises(ValueError, match="rows"):
        comrs.free_distance(puncture="turbo_1_2")
    with pytest.raises(ValueError, match="unknown"):
        comrs.free_distance(puncture="dvbs_1_2")


@pytest.mark.parametrize(
    ("pattern", "rate", "rows"),
    [
        ("ieee80211_2_3", 2 / 3, [[1, 1], [1, 0]]),
        ("ieee80211_3_4", 3 / 4, [[1, 1, 0], [1, 0, 1]]),
        ("ieee80211_5_6", 5 / 6, [[1, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
        ("dvbs_2_3", 2 / 3, [[1, 1], [1, 0]]),
        ("dvbs_3_4", 3 / 4, [[1, 1, 0], [1, 0, 1]]),
        ("dvbs_5_6", 5 / 6, [[1, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
        ("dvbs_7_8", 7 / 8, [[1, 1, 1, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0, 1]]),
    ],
)
def test_punctured_convolutional_rates_and_round_trips(pattern, rate, rows):
    k = 840  # a multiple of every period, the tail aside
    code = comrs.ConvolutionalCode(k, puncture=pattern)
    assert code.puncture == rows
    # the tail of 6 steps is punctured too: the rate tends to the nominal one
    assert code.n == pytest.approx(k / rate, abs=2 * 6)
    assert code.rate == pytest.approx(k / code.n)
    assert comrs.ConvolutionalCode(k, puncture=rows).n == code.n
    rng = comrs.Rng(8)
    b = comrs.bits(2 * k, rng)
    c = code.encode(b)
    assert c.size == 2 * code.n
    assert np.array_equal(code.decode(2.0 * (1.0 - 2.0 * c.astype(float))), b)


def test_unpunctured_code_is_unchanged():
    code = comrs.ConvolutionalCode(50)
    assert code.puncture is None and (code.n, code.rate) == (112, 50 / 112)
    assert comrs.ConvolutionalCode(50, puncture=[[1], [1]]).n == 112
    with pytest.raises(ValueError):
        comrs.ConvolutionalCode(50, puncture=[[1, 1], [1, 0], [0, 1]])
    with pytest.raises(ValueError):
        comrs.ConvolutionalCode(50, puncture=[[1, 2], [1, 0]])


def test_punctured_turbo_code():
    plain = comrs.TurboCode(40, qpp=(3, 10))
    code = comrs.TurboCode(40, qpp=(3, 10), puncture="turbo_1_2")
    assert plain.puncture is None and code.puncture == [[1, 1], [1, 0], [0, 1]]
    assert plain.n == 132 and code.n == 2 * 40 + 12
    assert code.rate == pytest.approx(40 / 92)
    rng = comrs.Rng(9)
    b = comrs.bits(80, rng)
    c = code.encode(b)
    assert c.size == 2 * code.n
    # the systematic bits are all kept: they are the even positions of every block body
    assert np.array_equal(c[:80:2], b[:40])
    assert np.array_equal(code.decode(3.0 * (1.0 - 2.0 * c.astype(float))), b)
    code.iterations = 3
    assert code.iterations == 3
    with pytest.raises(ValueError, match="3 rows"):
        comrs.TurboCode(40, puncture="ieee80211_2_3")


def test_punctured_rate_3_4_over_awgn_is_better_than_uncoded():
    # soft Viterbi of the rate-3/4 (133, 171) code over BPSK and AWGN: the BER
    # decreases with Eb/N0 and is far below uncoded BPSK at 4.5 dB (uncoded: 9.4e-3,
    # coded about 1e-4)
    k = 2000
    code = comrs.ConvolutionalCode(k, puncture="ieee80211_3_4")
    rng = comrs.Rng(10)
    ber = []
    for ebn0_db in (2.5, 3.5, 4.5):
        esn0 = 10 ** (ebn0_db / 10) * code.rate
        errors, total = 0, 0
        for _ in range(100):
            b = comrs.bits(k, rng)
            x = 1.0 - 2.0 * code.encode(b).astype(float)
            y = comrs.awgn(x, 10 * np.log10(esn0), rng)
            llr = 4.0 * esn0 * y.real  # 2 y / (σ²/2), σ² = 1/SNR
            errors += int(np.sum(code.decode(llr) != b))
            total += k
        ber.append(errors / total)
    uncoded = 0.5 * erfc(np.sqrt(10 ** (4.5 / 10)))
    assert ber[0] > ber[1] > ber[2] > 0
    assert ber[2] < uncoded / 10

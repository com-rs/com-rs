"""The Python API: docstring examples and their sections, conversions, closed forms on
scalars and arrays, errors, reproducibility, and the Monte Carlo runner of Python
trials."""

import doctest
import inspect
import math
import pathlib
import re

import numpy as np
import pytest

import comrs
from comrs import _comrs, sim

FLAGS = doctest.NORMALIZE_WHITESPACE


def run_doctests(obj, name, *, any_module=False):
    """Run the examples of an object and of its members (with `any_module`, the members
    of the extension, which `DocTestFinder` skips unless told not to check their
    module)."""
    finder = doctest.DocTestFinder(exclude_empty=True)
    runner = doctest.DocTestRunner(optionflags=FLAGS)
    kw = {"module": False} if any_module else {}
    tests = finder.find(obj, name, globs={}, **kw)
    for t in tests:
        runner.run(t)
    return tests, runner.summarize(verbose=False)


def extension_members():
    """The public functions and classes of the extension."""
    return [
        (n, getattr(_comrs, n))
        for n in sorted(dir(_comrs))
        if not n.startswith("_")
        and (inspect.isclass(getattr(_comrs, n)) or callable(getattr(_comrs, n)))
    ]


@pytest.mark.parametrize("name, obj", extension_members(), ids=[n for n, _ in extension_members()])
def test_extension_docstring_examples(name, obj):
    _, result = run_doctests(obj, name, any_module=True)
    assert result.failed == 0


@pytest.mark.parametrize(
    "module",
    [
        comrs,
        comrs.block,
        comrs.blocks,
        comrs.chain,
        comrs.plot,
        comrs.results,
        comrs.sim,
        comrs.spec,
    ],
    ids=lambda m: m.__name__,
)
def test_module_docstring_examples(module):
    tests, result = run_doctests(module, module.__name__)
    assert any(t.examples for t in tests), f"no examples in {module.__name__}"
    assert result.failed == 0


CORE = [
    comrs.awgn,
    comrs.awgn_ebn0,
    comrs.ber,
    comrs.bits,
    comrs.theory_ser,
    comrs.Interleaver.random,
    comrs.LdpcCode.peg,
    comrs.Results.series,
    comrs.Results.profile,
    comrs.Chain.profile,
    comrs.Chain.bind,
    *[
        getattr(comrs.PaprReduction, m)
        for m in (
            "clip",
            "clip_filter",
            "slm",
            "pts",
            "tone_reservation",
            "ace",
            "mu_law",
            "exponential",
        )
    ],
    *[
        getattr(comrs.Dpsk, m)
        for m in (
            "encode",
            "decode",
            "detect",
            "detect_coherent",
            "modulate",
            "demodulate",
            "demodulate_coherent",
        )
    ],
    *[
        getattr(c, m)
        for c in (
            comrs.ConvolutionalCode,
            comrs.TurboCode,
            comrs.LdpcCode,
            comrs.PolarCode,
            comrs.ReedSolomonCode,
            comrs.BchCode,
        )
        for m in ("encode", "decode")
    ],
]


@pytest.mark.parametrize("obj", CORE, ids=lambda o: o.__qualname__)
def test_core_functions_have_examples(obj):
    assert ">>>" in (obj.__doc__ or ""), f"{obj.__qualname__} has no example"


def public_routines():
    """Every public function and method of the package, with its qualified name."""
    seen = set()
    for name in comrs.__all__:
        obj = getattr(comrs, name)
        if inspect.ismodule(obj) or name == "__version__":
            continue
        if inspect.isclass(obj):
            for attr, member in vars(obj).items():
                if attr.startswith("_"):
                    continue
                if isinstance(member, staticmethod | classmethod):
                    member = member.__func__
                if callable(member) and not inspect.isclass(member):
                    key = f"{name}.{attr}"
                    if key not in seen:
                        seen.add(key)
                        yield key, member
        elif callable(obj):
            yield name, obj


def parameters(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [
        p.name
        for p in sig.parameters.values()
        if p.name not in ("self", "cls", "py") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


@pytest.mark.parametrize(
    "name, obj", list(public_routines()), ids=[n for n, _ in public_routines()]
)
def test_every_public_routine_documents_its_arguments(name, obj):
    doc = obj.__doc__ or ""
    assert len(doc.strip()) > 20, f"{name} lacks a docstring"
    params = parameters(obj)
    if params:
        assert "Args:" in doc, f"{name} has no Args section"
        for p in params:
            assert re.search(rf"^\s+{p} \(", doc, re.M), f"{name}: argument {p} is not documented"


def test_every_public_class_is_documented():
    for name in comrs.__all__:
        obj = getattr(comrs, name)
        if inspect.isclass(obj):
            assert obj.__doc__ and len(obj.__doc__.strip()) > 20, f"{name} lacks a docstring"


def test_version():
    assert comrs.__version__ == "0.2.0"


def test_all_is_sorted_and_complete():
    assert comrs.__all__ == sorted(comrs.__all__)
    for name in comrs.__all__:
        assert hasattr(comrs, name)


# ---------------------------------------------------------------- reproducibility


def test_generator_matches_the_rust_reference_values():
    # xoshiro256** seeded by SplitMix64 from 0 (the value of the Rust unit test)
    assert comrs.Rng(0).next_u64() == 0x99EC5F36CB75F2B4


def test_streams_are_reproducible_and_distinct():
    a = comrs.Rng.stream(1, "exp", 2, 3).uniform(5)
    b = comrs.Rng.stream(1, "exp", 2, 3).uniform(5)
    np.testing.assert_array_equal(a, b)
    for other in [(2, "exp", 2, 3), (1, "other", 2, 3), (1, "exp", 1, 3), (1, "exp", 2, 4)]:
        assert not np.array_equal(a, comrs.Rng.stream(*other).uniform(5))


def test_functions_that_draw_advance_the_generator():
    rng = comrs.Rng(4)
    x = np.ones(8, dtype=complex)
    assert not np.array_equal(comrs.awgn(x, 0.0, rng), comrs.awgn(x, 0.0, rng))


# --------------------------------------------------------------------- conversions


def test_array_likes_are_converted():
    c = comrs.Constellation("qam", 4)
    assert c.modulate([0, 1, 1, 0]).dtype == np.complex128
    np.testing.assert_allclose(comrs.pnorm([1.0, -1.0]), [1.0, -1.0])
    np.testing.assert_allclose(comrs.fir_filter([1.0], np.arange(4)), np.arange(4))
    # a strided view is copied, not misread
    x = np.arange(10, dtype=complex)[::2]
    np.testing.assert_allclose(comrs.pnorm(x), x / np.sqrt(np.mean(np.abs(x) ** 2)))


@pytest.mark.parametrize(
    "f, arg",
    [
        (lambda x: comrs.theory_ber(16, x), 8.0),
        (lambda x: comrs.theory_ser(4, x, "psk"), 6.0),
        (lambda x: comrs.theory_mi(4, x), 5.0),
        (lambda x: comrs.theory_gmi(4, x), 5.0),
        (lambda x: comrs.theory_ber_dpsk(4, x), 6.0),
        (lambda x: comrs.theory_ber_rayleigh_mrc(x, 2), 10.0),
        (lambda x: comrs.theory_papr_ccdf(64, x), 8.0),
        (lambda x: comrs.theory_rs_wer(255, 8, x), 1e-2),
        (lambda x: comrs.theory_rs_ser(255, 8, x), 1e-2),
        (lambda x: comrs.theory_rs_ber(255, 8, 8, x), 1e-2),
        (lambda x: comrs.biawgn_capacity(x, 0.5), 1.0),
        (lambda x: comrs.normal_approximation(128, 64, x), 2.0),
        (comrs.shannon_limit_ebn0_db, 0.5),
        (comrs.qfunc, 1.0),
        (comrs.db2lin, 3.0),
        (comrs.lin2db, 2.0),
        (comrs.dbm2w, 0.0),
        (comrs.w2dbm, 1e-3),
        (comrs.msk_psd, 0.1),
        (lambda x: comrs.differential_phase_tail(math.pi / 4, x), 3.0),
    ],
)
def test_closed_forms_take_scalars_and_arrays(f, arg):
    scalar = f(arg)
    assert isinstance(scalar, float)
    assert f(np.float64(arg)) == scalar and f(np.asarray(arg)) == scalar
    grid = f([arg, arg])
    assert isinstance(grid, np.ndarray) and grid.shape == (2,) and grid[0] == scalar
    assert f(np.full((2, 3), arg)).shape == (2, 3)


@pytest.mark.parametrize("branches", [1, 2, 4])
def test_the_mrc_closed_form_is_bound(branches):
    # Proakis & Salehi, Digital Communications, 5th ed., eq. (13.4-15)
    for ebn0_db in (0.0, 10.0, 20.0):
        g = comrs.db2lin(ebn0_db)
        mu = math.sqrt(g / (1 + g))
        expected = ((1 - mu) / 2) ** branches * sum(
            math.comb(branches - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(branches)
        )
        got = comrs.theory_ber_rayleigh_mrc(ebn0_db, branches)
        assert got == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="branch"):
        comrs.theory_ber_rayleigh_mrc(10.0, 0)


def test_one_estimate_has_one_shape():
    # frame_metrics, run_spec and ConditionResult.to_dict give an estimate the same keys
    spec = {
        "name": "shape",
        "link": [
            {"type": "source", "constellation": "QPSK", "n_symbols": 64},
            {"type": "awgn_ebn0", "ebn0_db": 4.0},
        ],
        "metrics": {"report": ["ber", "snr_db"]},
        "stop": {"max_trials": 2},
    }
    keys = {"value", "lo", "hi", "kind", "count", "total"}
    run = comrs.run_spec(spec)[0]["metrics"]
    f = comrs.Frame()
    rng = comrs.Rng(1)
    for block in spec["link"]:
        comrs.LinkBlock(block).apply(f, rng)
    frame = comrs.frame_metrics(f, spec["metrics"])
    assert set(run) == set(frame) == {"ber", "snr_db"}
    for name in run:
        assert set(run[name]) == set(frame[name]) == keys
        assert run[name]["kind"] == frame[name]["kind"]
    assert frame["ber"]["total"] == 128 and frame["snr_db"]["count"] == 1


def test_lists_are_arrays():
    assert isinstance(comrs.wdm_grid(3, 50e9), np.ndarray)
    taps = comrs.power_delay_profile("eva")
    assert taps.shape == (9, 2)
    assert comrs.rms_delay_spread(taps) == comrs.rms_delay_spread(taps.tolist())
    ch = comrs.TdlChannel(taps, 15.36e6)
    assert ch.delays.dtype == np.int64 and isinstance(ch.powers, np.ndarray)
    assert comrs.Ofdm(64, 48).bins.dtype == np.int64
    assert comrs.PolarCode(64, 32).info_positions.dtype == np.int64
    assert comrs.ResourceGrid(12, 4).pilot_positions().shape[1] == 2


def test_interleavers_keep_the_dtype():
    il = comrs.Interleaver.random(16, seed=3)
    for x in (comrs.bits(16, comrs.Rng(1)), np.linspace(-1, 1, 16), np.exp(1j * np.arange(16))):
        y = il.interleave(x)
        assert y.dtype == x.dtype
        np.testing.assert_array_equal(y, x[il.perm])
        np.testing.assert_array_equal(il.deinterleave(y), x)
    with pytest.raises(ValueError, match="block"):
        il.interleave(np.ones(15))


def test_constellation_object():
    c = comrs.Constellation("psk", 8)
    assert (c.kind, c.order, c.bits_per_symbol, len(c)) == ("psk", 8, 3, 8)
    assert "psk" in repr(c)
    s = c.scaled(2.0)
    assert s.energy == pytest.approx(4.0)
    custom = comrs.Constellation.from_points([1, -1, 1j, -1j])
    assert custom.kind == "custom"
    idx = custom.decide([0.9, -0.1j - 1, 0.1 + 1.2j])
    assert idx.tolist() == [0, 1, 2]
    np.testing.assert_allclose(custom.project([0.9]), [1.0])
    q = comrs.Constellation.from_name("16-QAM")
    np.testing.assert_array_equal(q.points, comrs.Constellation("qam", 16).points)
    assert comrs.Constellation.from_name("qpsk").order == 4
    with pytest.raises(ValueError):
        comrs.Constellation.from_name("sixteen")


def test_fiber_parameters_are_mutable_and_derived_quantities_follow():
    f = comrs.FiberParams()
    assert f.n_spans == 1 and f.total_km == 80.0
    f.n_spans = 5
    f.span_km = 100.0
    assert f.total_km == 500.0
    assert f.span_gain_db == pytest.approx(20.0)
    assert "n_spans=5" in repr(f)


def test_equalizer_accepts_one_mode_and_schedules():
    rng = comrs.Rng(1)
    c = comrs.Constellation("qam", 4).normalized()
    _, s = comrs.symbols(c, 3000, rng)
    x = comrs.upsample(s, 2)
    x = x + 0.3 * np.roll(x, 2)
    out, taps, err = comrs.adaptive_equalizer(
        x, c, algorithm=[("cma", 1000), ("dd-lms", 0)], n_taps=11, sps=2, mu=5e-3
    )
    assert out.shape == (1, 3000) and taps.shape == (1, 11) and err.shape == (1, 3000)
    assert np.mean(err[0, -500:]) < 0.05


# ------------------------------------------------------------------------ errors


def test_invalid_parameters_raise_value_errors():
    for seconds in (float("inf"), float("nan"), -1.0):
        with pytest.raises(ValueError, match="max_time"):
            comrs.sim.Stop.max_time(seconds)
    with pytest.raises(ValueError, match="power of two"):
        comrs.Constellation("qam", 12)
    with pytest.raises(ValueError):
        comrs.Constellation("bogus", 4)
    with pytest.raises(ValueError, match="bits"):
        comrs.Constellation("qam", 4).modulate([0, 2, 1, 0])
    with pytest.raises(ValueError):
        comrs.pulse_shape("sinc", 2)
    with pytest.raises(ValueError):
        comrs.ber([0, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        comrs.Edfa(gain_db=10.0, nf_db=2.0).apply(np.ones(4), 1e9, comrs.Rng(0))
    with pytest.raises(ValueError, match="2-D"):
        comrs.adaptive_equalizer(np.ones((2, 2, 2)), comrs.Constellation("qam", 4))
    with pytest.raises(ValueError, match="pilot"):
        comrs.ddpll(np.ones(4), comrs.Constellation("qam", 4), pilot_indices=[0])
    with pytest.raises(ValueError, match="4 samples"):
        comrs.interpolate_cubic([1, 2, 3], 0.0)
    with pytest.raises(ValueError, match="rng"):
        comrs.Manakov(comrs.FiberParams(), pmd_ps_sqrt_km=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        comrs.bits(-1, comrs.Rng(0))
    with pytest.raises(ValueError, match="at most 24"):
        comrs.gray_code(40)  # aborted the process on the allocation of 2^40 words


def test_tuning_parameters_are_keyword_only():
    with pytest.raises(TypeError):
        comrs.pulse_shape("rrc", 4, 1025)
    with pytest.raises(TypeError):
        comrs.mimo_channel(2, 2, comrs.Rng(0), 10)
    with pytest.raises(TypeError):
        comrs.CdlChannel("CDL-A", 1e-7, 1e6, 2, 2)


def test_defaults_are_those_of_the_blocks(tmp_path):
    blocks = comrs.spec_blocks()
    p = blocks["pulse_shape"]["params"]
    h = comrs.pulse_shape("rrc", 2)
    assert h.size == int(p["n_taps"]["default"])
    np.testing.assert_array_equal(
        h, comrs.pulse_shape("rrc", 2, rolloff=float(p["rolloff"]["default"]))
    )
    # the K-best detector keeps the paths of the mimo_rx block
    rng = comrs.Rng(5)
    c = comrs.Constellation("qam", 16).normalized()
    h4 = comrs.mimo_channel(4, 4, rng, n=50)
    y = np.einsum(
        "nij,nj->ni", h4, c.points[rng.integers(16, 200).reshape(50, 4)]
    ) + 0.3 * rng.complex_noise(200).reshape(50, 4)
    k = int(blocks["mimo_rx"]["params"]["k"]["default"])
    np.testing.assert_array_equal(
        comrs.mimo_detect(h4, y, c, 0.09, "kbest"), comrs.mimo_detect(h4, y, c, 0.09, "kbest", k=k)
    )
    x = np.exp(2j * np.pi * 0.01 * np.arange(1024)) * c.points[rng.integers(16, 1024)]
    assert comrs.fourth_power_foe(x, 1.0)[1] == comrs.fourth_power_foe(x, 1.0, refined=True)[1]
    link = comrs.Ssfm(comrs.FiberParams(), step_km=10.0, amplification="ideal")
    z = 1e-2 * np.exp(1j * np.arange(64))
    np.testing.assert_array_equal(
        link.backpropagate(z, 32e9), link.backpropagate(z, 32e9, steps_per_span=1)
    )
    comrs.write_sigmf(tmp_path / "x", z, 1e6)
    assert comrs.read_sigmf(tmp_path / "x")[2]["global"]["core:datatype"] == "cf64_le"


# ------------------------------------------------------------- Python Monte Carlo


def bernoulli(cond, rng, rec):
    p = cond["p"]
    rec.errors("rate", int((rng.uniform(100) < p).sum()), 100)
    rec.value("draw", rng.gaussian(1)[0])


def test_python_runner_is_independent_of_the_batch_size():
    conds = sim.grid(p=[0.05, 0.5])
    a = sim.run(bernoulli, conds, name="b", seed=3, batch=7, stop=sim.Stop.max_trials(70))
    b = sim.run(bernoulli, conds, name="b", seed=3, batch=70, stop=sim.Stop.max_trials(70))
    for x, y in zip(a, b, strict=False):
        assert x.metrics["rate"] == y.metrics["rate"]
        assert x.metrics["draw"].value == pytest.approx(y.metrics["draw"].value, abs=1e-12)


def test_python_stopping_rules():
    rule = sim.Stop.min_errors("rate", 50) | sim.Stop.max_trials(1000)
    res = sim.run(bernoulli, sim.grid(p=[0.1, 1e-9]), name="b", batch=2, stop=rule)
    assert res[0].metrics["rate"].count >= 50 and res[0].trials < 20
    assert res[1].trials == 1000
    with pytest.raises(ValueError, match="forever"):
        sim.run(bernoulli, sim.grid(p=[0.1]), name="b", stop=sim.Stop.min_errors("rate", 1))
    rule = sim.Stop.precision("draw", 0.5) & sim.Stop.min_trials(4) | sim.Stop.max_trials(10_000)
    res = sim.run(
        lambda c, r, rec: rec.value("draw", 1.0 + r.gaussian(1)[0]),
        [{}],
        name="m",
        batch=1,
        stop=rule,
    )
    assert res[0].trials >= 4 and res[0].metrics["draw"].relative_precision() <= 0.5


def test_stop_mappings_are_parsed_by_the_rust_runner():
    # min_trials alone caps the trials (the rule of the specifications)
    assert sim.Stop.from_spec({"min_trials": 50}) == sim.Stop.max_trials(50)
    assert repr(sim.Stop.from_spec({})) == "trials ≥ 100"
    rule = sim.Stop.from_spec({"min_errors": {"ber": 10}, "min_trials": 5, "max_trials": 9})
    assert repr(rule) == "((errors(ber) ≥ 10 and trials ≥ 5) or trials ≥ 9)"
    res = sim.run(bernoulli, [{"p": 0.5}], name="s", batch=1, stop={"min_trials": 3})
    assert res[0].trials == 3


def test_python_statistics_are_the_rust_ones():
    rate = sim.ErrorRate(0, 1000)
    lo, hi = rate.interval(0.95)
    assert lo == 0.0 and hi == pytest.approx(3.83e-3, rel=1e-2)  # the rule of three, about
    m = sim.Mean()
    m.add(1.0)
    assert m.interval() == (-math.inf, math.inf)
    for x in [2.0, 3.0]:
        m.add(x)
    assert m.value == 2.0 and m.std == 1.0
    a, b = sim.Mean(), sim.Mean()
    for x in [1.0, 5.0]:
        a.add(x)
    for x in [2.0, 7.0, 9.0]:
        b.add(x)
    a.merge(b)
    assert a.value == pytest.approx(4.8) and a.std == pytest.approx(np.std([1, 5, 2, 7, 9], ddof=1))
    assert sim.z_score(0.95) == pytest.approx(1.959964, abs=1e-6)


def test_python_runner_writes_the_reports_and_reports_failures(tmp_path):
    path = tmp_path / "out" / "r.csv"
    sim.run(
        bernoulli,
        sim.grid(p=[0.2, 0.4]),
        name="b",
        stop=sim.Stop.max_trials(8),
        batch=4,
        csv_path=path,
        json_path=tmp_path / "r.json",
    )
    lines = path.read_text().splitlines()
    assert lines[0].startswith(
        "index,p,status,trials,wall_s,draw,draw_lo,draw_hi,draw_count,draw_total,rate"
    )
    assert len(lines) == 3
    assert '"scenario": "b"' in (tmp_path / "r.json").read_text()

    def failing(cond, rng, rec):
        raise ArithmeticError("diverged")

    with pytest.raises(ArithmeticError, match="diverged") as e:
        sim.run(failing, sim.grid(p=[0.3]), name="f")
    assert e.value.__notes__ == ["trial 0 of (p=0.3) failed"]

    def bad_setup(cond):
        raise KeyError("missing")

    with pytest.raises(KeyError) as e:
        sim.run(bernoulli, sim.grid(p=[0.3]), name="f", setup=bad_setup)
    assert e.value.__notes__ == ["setup of (p=0.3) failed"]


def test_python_runner_with_setup_and_record_type_checks():
    seen = []

    def setup(cond):
        seen.append(cond["m"])
        return comrs.Constellation("qam", cond["m"]).normalized()

    def trial(cond, rng, rec, c):
        b = comrs.bits(c.bits_per_symbol * 200, rng)
        y = comrs.awgn_ebn0(c.modulate(b), 12.0, c.bits_per_symbol, rng)
        rec.errors("ber", int((c.demodulate(y) != b).sum()), b.size)

    res = sim.run(
        trial, sim.grid(m=[4, 16]), name="qam", setup=setup, stop=sim.Stop.max_trials(10), batch=5
    )
    assert seen == [4, 16] and all(r.status == "completed" for r in res)
    assert res[1].condition == {"m": 16} and res[1].to_dict()["metrics"]["ber"]["kind"] == "rate"
    rec = sim.Record()
    rec.errors("x", 1, 2)
    with pytest.raises(TypeError):
        rec.value("x", 1.0)


def test_conditions_keep_their_python_values():
    marker = object()

    def trial(cond, rng, rec):
        assert cond["obj"] is marker
        rec.value("one", 1.0)

    res = sim.run(
        trial, [{"obj": marker}, {"obj": marker, "x": 2}], name="o", stop={"max_trials": 1}
    )
    assert res[0].condition["obj"] is marker and res[1].index == 1


def test_every_exported_name_is_in_the_api_reference():
    """The API reference (docs/python/api/index.rst) lists every name of ``comrs.__all__``
    once, in its theme, so that a new binding cannot be left undocumented."""
    page = pathlib.Path(__file__).parents[1] / "docs" / "python" / "api" / "index.rst"
    listed = re.findall(r"^   ([A-Za-z_][A-Za-z0-9_]*)$", page.read_text(), flags=re.M)
    exported = sorted(set(comrs.__all__) - {"__version__"})
    assert sorted(set(exported) - set(listed)) == [], "exported but not documented"
    assert sorted(set(listed) - set(exported)) == [], "documented but not exported"
    assert len(listed) == len(set(listed)), "listed twice"

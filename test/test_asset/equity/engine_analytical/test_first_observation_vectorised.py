"""The projection through the first observation evaluates its integrand on every node in one call.

``first_observation.through_first_observation`` takes the expectation over the Gaussian step to the first future
observation by a composite Gauss-Legendre rule at two orders. It calls ``hit_value`` and ``continuation`` once each,
on a 1-D array of every node spot of both orders, and sums the terms in the rule's own order. The engines' closures
and the closed forms they reach (single-barrier knock-out, touch terms, Black-Scholes, Ikeda-Kunitomo, the double
no-touch series) take that array. Their scalar paths are unchanged, so a value computed on the array matches the
same value computed node by node, and each engine resolves its schedule's records once per price.
"""
from __future__ import annotations

import math
import platform
import warnings

import numpy as np
import pytest

import test_discrete_barrier_between_observations as T
from quantark.asset.equity.engine.analytical import (
    barrier_analytical_engine,
    double_barrier_option_engine,
    double_sharkfin_option_analytical_engine,
    first_observation,
    one_touch_analytical_engine,
)
from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_no_touch import double_no_touch
from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
    DoubleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.analytical.first_observation import FirstObservation, through_first_observation
from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    AccumulatorKnockOutType,
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationType,
    OptionType,
    TouchType,
)
from quantark.util.numerical import pow_elementwise

#: the engine modules that hold a first-observation projection
PROJECTING = [barrier_analytical_engine, one_touch_analytical_engine, double_barrier_option_engine,
              double_sharkfin_option_analytical_engine]
MONTHLY = [k / 12 for k in range(1, 13)]
WEEKLY = [k / 52 for k in range(1, 53)]
#: NumPy's elementwise exp / log / pow / erfcx are the platform libm's here, so the array paths are bitwise the
#: scalar ones; elsewhere NumPy may use its own SIMD kernels (last-bit differences)
LIBM_UFUNCS = platform.machine().lower() in ("arm64", "aarch64")


def _env(spot, vol=0.20):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=T.VALUATION)


def _single(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def _double(times, upper=130.0, lower=75.0):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, upper_barrier=upper,
                                                          lower_barrier=lower) for t in times])


def _contracts(times, maturity=1.0):
    """(name, engine class, product builder) on one observation grid."""
    d = dict(maturity=maturity, observation_type=ObservationType.DISCRETE)
    return [
        ("UOC_r1", BarrierAnalyticalEngine, lambda: BarrierOption(
            strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT, rebate=1.0,
            observation_schedule=_single(times, 120.0), **d)),
        ("UOC_r5_hit", BarrierAnalyticalEngine, lambda: BarrierOption(
            strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT, rebate=5.0,
            pay_at_hit=True, observation_schedule=_single(times, 120.0), **d)),
        ("DIP_r5", BarrierAnalyticalEngine, lambda: BarrierOption(
            strike=100.0, option_type=OptionType.PUT, barrier=85.0, barrier_type=BarrierType.DOWN_IN, rebate=5.0,
            observation_schedule=_single(times, 85.0), **d)),
        ("OT_up_hit", OneTouchAnalyticalEngine, lambda: OneTouchOption(
            barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=5.0, payment_at_hit=True,
            touch_type=TouchType.ONE_TOUCH, observation_schedule=_single(times, 120.0), **d)),
        ("NT_dn", OneTouchAnalyticalEngine, lambda: OneTouchOption(
            barrier=85.0, barrier_direction=BarrierDirection.DOWN, rebate=5.0, touch_type=TouchType.NO_TOUCH,
            observation_schedule=_single(times, 85.0), **d)),
        ("DKO_C_r1", DoubleBarrierOptionAnalyticalEngine, lambda: DoubleBarrierOption(
            strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT, rebate=1.0, observation_schedule=_double(times), **d)),
        ("DKI_P_r5", DoubleBarrierOptionAnalyticalEngine, lambda: DoubleBarrierOption(
            strike=100.0, option_type=OptionType.PUT, upper_barrier=130.0, lower_barrier=75.0,
            barrier_type=DoubleBarrierType.KNOCK_IN, rebate=5.0, observation_schedule=_double(times), **d)),
        ("SSF_hit", SingleSharkfinOptionAnalyticalEngine, lambda: SingleSharkfinOption(
            strike=100.0, option_type=OptionType.CALL, barrier=120.0, knock_out_rebate=0.5, no_hit_rebate=3.0,
            pay_at_hit=True, observation_schedule=_single(times, 120.0), **d)),
        ("DSF_hit", DoubleSharkfinOptionAnalyticalEngine, lambda: DoubleSharkfinOption(
            strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0,
            knock_out_rebate=0.5, no_hit_rebate=3.0, pay_at_hit=True, observation_schedule=_double(times), **d)),
        ("DSF_put", DoubleSharkfinOptionAnalyticalEngine, lambda: DoubleSharkfinOption(
            strike=100.0, option_type=OptionType.PUT, upper_barrier=130.0, lower_barrier=75.0,
            knock_out_rebate=0.5, no_hit_rebate=3.0, observation_schedule=_double(times), **d)),
        ("ACC_term", AccumulatorAnalyticalEngine, lambda: AccumulatorOption(
            strike=95.0, knock_out_barrier=110.0, option_type=OptionType.CALL, maturity=maturity,
            daily_share_accumulation=1.0, gearing=2.0, knock_out_type=AccumulatorKnockOutType.TERMINATION,
            observation_dates=list(times))),
    ]


def _aged(build, years, keep_landing=False):
    def aged():
        product = build()
        product.time_shift(years, T.VALUATION, _env(100.0), keep_landing=keep_landing)
        return product
    return aged


#: (label, observation grid, maturity, (age, keep landing record) or None, vol): fresh monthly, between records,
#: on the penultimate record (the first future observation is the maturity: zero remaining time after it), a
#: first observation 1e-4 away, low vol
SETUPS = [
    ("fresh", MONTHLY, 1.0, None, 0.20),
    ("between", MONTHLY, 1.0, (0.30, False), 0.20),
    ("last_record", MONTHLY, 1.0, (11 / 12, True), 0.20),
    ("tiny_t1", [k / 1000 for k in range(1, 11)], 0.01, (0.0009, False), 0.20),
    ("low_vol", MONTHLY, 1.0, (0.30, False), 0.02),
]
SPOTS = [80.0, 105.0, 119.9, 121.0, 131.0]


def _price(engine_cls, build, spot, vol):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(engine_cls().price(build(), _env(spot, vol)))
        except Exception as exc:  # noqa: BLE001 -- an outcome compared as such
            return f"{type(exc).__name__}: {exc}"


class _Projections:
    """Wraps every engine module's ``through_first_observation`` so its two functions pass through ``wrap``."""

    def __init__(self, monkeypatch, wrap):
        self.calls = []        # per projection: {"hit_value": [sizes], "continuation": [sizes]}
        for module in PROJECTING:
            original = module.through_first_observation

            def projection(*, hit_value, continuation, _original=original, **kwargs):
                record = {"hit_value": [], "continuation": []}
                self.calls.append(record)
                return _original(hit_value=wrap(hit_value, record["hit_value"]),
                                 continuation=wrap(continuation, record["continuation"]), **kwargs)

            monkeypatch.setattr(module, "through_first_observation", projection)


def _counted(fn, sizes):
    def counted(spots):
        sizes.append(spots.size if isinstance(spots, np.ndarray) else None)
        return fn(spots)
    return counted


def _node_by_node(fn, sizes):
    def per_node(spots):
        sizes.append(spots.size)
        return np.concatenate([np.broadcast_to(np.asarray(fn(spots[i:i + 1]), dtype=float), (1,))
                               for i in range(spots.size)])
    return per_node


# ======================================================================================================================
# the integrand is evaluated once, on every node
# ======================================================================================================================

def test_each_function_is_evaluated_once_on_every_node():
    calls = {"hit_value": [], "continuation": []}

    def hit_value(s):
        calls["hit_value"].append(s)
        return np.maximum(s - 100.0, 0.0)

    def continuation(s):
        calls["continuation"].append(s)
        return np.sqrt(s)

    first = FirstObservation(time=0.25, lower=80.0, upper=125.0, settlement_time=0.25, payoff=0.0, remaining=3)
    value = through_first_observation(spot=100.0, first=first, rate=0.03, div=0.01, vol=0.2, hit_value=hit_value,
                                      continuation=continuation, remaining_time=0.75)
    assert math.isfinite(value)
    for name in ("hit_value", "continuation"):
        assert len(calls[name]) == 1, name
        nodes = calls[name][0]
        assert isinstance(nodes, np.ndarray) and nodes.ndim == 1
        assert nodes.size % (first_observation.LOW_ORDER + first_observation.HIGH_ORDER) == 0
    # the breached regions below 80 and above 125 are two regions of hit_value, both in the one call
    assert np.any(calls["hit_value"][0] < 80.0) and np.any(calls["hit_value"][0] > 125.0)
    assert np.all((calls["continuation"][0] > 80.0) & (calls["continuation"][0] < 125.0))


def test_a_constant_breach_value_is_read_once():
    calls = []

    def hit_value(s):
        calls.append(s)
        return 2.5

    first = FirstObservation(time=0.25, lower=None, upper=125.0, settlement_time=0.25, payoff=2.5, remaining=3)
    through_first_observation(spot=100.0, first=first, rate=0.03, div=0.01, vol=0.2, hit_value=hit_value,
                              continuation=lambda s: np.ones_like(s), remaining_time=0.75, hit_is_constant=True)
    assert len(calls) == 1


@pytest.mark.parametrize("name,engine_cls,build", _contracts(MONTHLY) + [
    (f"{name}_weekly", engine_cls, build) for name, engine_cls, build in _contracts(WEEKLY)
    if name in ("DSF_hit", "ACC_term")])
def test_engines_evaluate_each_function_once_per_projection(monkeypatch, name, engine_cls, build):
    projections = _Projections(monkeypatch, _counted)
    value = _price(engine_cls, build, 105.0, 0.20)
    assert isinstance(value, float), value
    assert projections.calls, "no projection through the first observation"
    for record in projections.calls:
        for fn, sizes in record.items():
            assert len(sizes) <= 1, (fn, len(sizes))
            assert all(size is not None for size in sizes), fn


# ======================================================================================================================
# one call on every node == node by node, bitwise (each node's value depends on that node only)
# ======================================================================================================================

@pytest.mark.parametrize("label,times,maturity,age,vol", SETUPS, ids=[s[0] for s in SETUPS])
def test_one_call_on_every_node_equals_node_by_node(monkeypatch, label, times, maturity, age, vol):
    contracts = _contracts(times, maturity)
    builds = [(name, cls, _aged(build, *age) if age is not None else build) for name, cls, build in contracts]
    vectorised = {(name, spot): _price(cls, build, spot, vol) for name, cls, build in builds for spot in SPOTS}
    projections = _Projections(monkeypatch, _node_by_node)
    by_node = {(name, spot): _price(cls, build, spot, vol) for name, cls, build in builds for spot in SPOTS}
    assert projections.calls, label
    assert by_node == vectorised


# ======================================================================================================================
# each engine resolves its schedule's records once per price
# ======================================================================================================================

@pytest.mark.parametrize("name,engine_cls,build", [c for c in _contracts(MONTHLY) if c[1] in (
    BarrierAnalyticalEngine, OneTouchAnalyticalEngine, DoubleBarrierOptionAnalyticalEngine,
    DoubleSharkfinOptionAnalyticalEngine)])
@pytest.mark.parametrize("spot", [105.0, 121.0])
def test_records_are_resolved_once_per_price(monkeypatch, name, engine_cls, build, spot):
    calls = []
    for module in PROJECTING:
        original = module.resolve_current_records

        def spy(*args, _original=original, **kwargs):
            calls.append(1)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, "resolve_current_records", spy)
    value = _price(engine_cls, build, spot, 0.20)
    assert isinstance(value, float), value
    assert len(calls) == 1


# ======================================================================================================================
# the closed forms' array paths are their scalar paths, elementwise
# ======================================================================================================================

def _assert_twin(array_values, scalar_values, scale=1.0):
    """Bitwise where NumPy's ufuncs are libm's; elsewhere within 1e-12 relative plus 1e-13 of ``scale`` (the size of
    the terms: a value near 0 is a difference of terms of the spot's size, whose last bits SIMD moves)."""
    array_values = np.asarray(array_values, dtype=float)
    scalar_values = np.asarray(scalar_values, dtype=float)
    assert array_values.shape == scalar_values.shape
    if LIBM_UFUNCS:
        assert np.array_equal(array_values, scalar_values, equal_nan=True)
    else:
        bound = 1e-12 * np.abs(scalar_values) + 1e-13 * np.asarray(scale, dtype=float)
        assert np.all(np.abs(array_values - scalar_values) <= bound), np.max(np.abs(array_values - scalar_values))


SPOT_GRID = np.array([40.0, 74.9, 75.0, 75.1, 90.0, 100.0, 104.9, 119.99, 120.0, 120.01, 129.9, 135.0, 300.0])


@pytest.mark.parametrize("exponent", [2.0, 0.5, -1.0, 1.7, -3.3, 0.0])
def test_pow_elementwise_is_libm_pow(exponent):
    base = np.linspace(0.05, 40.0, 2001)
    expected = [b ** exponent for b in base.tolist()]
    result = pow_elementwise(base, exponent)
    if LIBM_UFUNCS:
        assert result.tolist() == expected
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-15)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("is_up,barrier", [(True, 120.0), (True, 95.0), (False, 85.0), (False, 105.0)])
@pytest.mark.parametrize("vol", [0.2, 0.03])
def test_single_barrier_knock_out_array_path(is_call, is_up, barrier, vol):
    engine = BarrierAnalyticalEngine()
    kw = dict(strike=100.0, barrier=barrier, maturity=0.7, rate=0.03, carry=0.02, vol=vol, is_call=is_call,
              is_up=is_up)
    _assert_twin(engine._price_knock_out_closed_form(spot=SPOT_GRID, **kw),
                 [engine._price_knock_out_closed_form(spot=s, **kw) for s in SPOT_GRID.tolist()], scale=SPOT_GRID)


@pytest.mark.parametrize("term", ["_instant_touch_term", "_expiry_touch_term"])
@pytest.mark.parametrize("is_up,barrier", [(True, 120.0), (False, 85.0)])
def test_touch_terms_array_path(term, is_up, barrier):
    engine = OneTouchAnalyticalEngine()
    fn = getattr(engine, term)
    kw = dict(barrier=barrier, maturity=0.7, rate=0.03, div=0.01, vol=0.2, is_up=is_up)
    _assert_twin(fn(spot=SPOT_GRID, **kw), [fn(spot=s, **kw) for s in SPOT_GRID.tolist()])


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("vol,maturity", [(0.2, 0.7), (0.05, 0.7), (0.4, 0.004)])
def test_ikeda_kunitomo_array_path(is_call, vol, maturity):
    engine = DoubleBarrierOptionAnalyticalEngine()
    spots = SPOT_GRID[(SPOT_GRID > 75.0) & (SPOT_GRID < 130.0)]
    kw = dict(K=100.0, L=78.0, U=126.0, T=maturity, r=0.03, b=0.02, sigma=vol, is_call=is_call)
    _assert_twin(engine._price_knock_out_ikeda_kuintomo(S=spots, **kw),
                 [engine._price_knock_out_ikeda_kuintomo(S=s, **kw) for s in spots.tolist()], scale=spots)


def test_double_no_touch_array_path():
    # spots outside the corridor, on it, inside; maturities zero, short, long (the proven-zero branch)
    maturities = np.array([0.0, 1e-4, 0.08, 0.7, 5.0, 60.0])
    surface = double_no_touch(SPOT_GRID[:, None], maturities[None, :], 0.03, 0.01, 0.2, 75.0, 130.0)
    assert surface.shape == (SPOT_GRID.size, maturities.size)
    expected = [[double_no_touch(s, t, 0.03, 0.01, 0.2, 75.0, 130.0) for t in maturities.tolist()]
                for s in SPOT_GRID.tolist()]
    _assert_twin(surface, expected)
    row = double_no_touch(SPOT_GRID, 0.7, 0.03, 0.01, 0.2, 75.0, 130.0)
    _assert_twin(row, [double_no_touch(s, 0.7, 0.03, 0.01, 0.2, 75.0, 130.0) for s in SPOT_GRID.tolist()])


def test_double_no_touch_array_path_refuses_what_the_scalar_refuses():
    with pytest.raises(Exception) as scalar:
        double_no_touch(-1.0, 0.5, 0.03, 0.01, 0.2, 75.0, 130.0)
    with pytest.raises(type(scalar.value)):
        double_no_touch(np.array([100.0, -1.0]), 0.5, 0.03, 0.01, 0.2, 75.0, 130.0)
    with pytest.raises(type(scalar.value)):
        double_no_touch(np.array([100.0, math.nan]), 0.5, 0.03, 0.01, 0.2, 75.0, 130.0)

"""QUAD V2 ``EventStatistics`` (unified event statistics spec T16: AC7, AC8, AC9, AC10, AC13, AC14, AC17; D17, D23).

Every termination outcome is a backward channel of the price's own recursion; the knock-in path statistics come from
the forward adjoint pass of that recursion (``quad/v2/adjoint.py``). The adjoint identity (AC14): the pass's masses
reproduce every backward column whose continuations are note-state transitions -- the knock-out, maturity and
knocked-in settlement probabilities and cash -- to round-off. Independent pins: the ``disable_ko_after_ki`` twin's
knocked-in settlement (= P(KI before any KO)), the closed-form first passage (continuous KI, knock-out out of reach),
closed-form zero-volatility contracts.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.quad import (
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.engine.quad.v2 import statistics as v2_statistics
from quantark.asset.equity.engine.quad.v2.adjoint import forward_pass
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.asset.equity.lifecycle.cashflows import RealizedCashflow, ValuationPoint
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.param import QuadV2Params
from quantark.asset.equity.product.option import PhoenixOption, SnowballOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.cashleg.event_distribution import EventType
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.vol.vol_surface import TermStructureVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode

VALUATION = datetime(2024, 1, 1)
OBS = (0.25, 0.5, 0.75, 1.0)
#: AC14: the forward pass reproduces a backward column (measured <= 6e-14 on cash ~100, 1e-15 on probabilities)
ADJOINT_TOL = 1e-10
#: AC8 bounds, measured on every route here (probability residual <= 2e-15, reconciliation <= 1e-13 on pv ~100):
#: the recursion integrates every outcome's affine cash exactly against the Gaussian, so both are round-off
RESIDUAL_BOUND, RECONCILIATION_BOUND = 1e-12, 1e-11
#: P(KI before KO) against the ``disable_ko_after_ki`` twin's KI settlement (measured <= 3e-13): the two pass the
#: knocked-in mass through different (but exactly integrating) arithmetic
TWIN_BOUND = 1e-10


def env(spot=100.0, vol=0.22, r=0.03, q=0.01, valuation=VALUATION, calendar=None):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=r), div_yield=ContinuousDividendYield(div_yield=q),
                              valuation_date=valuation, calendar=calendar)


def ki_kw(ki, dates=OBS, barrier=75.0):
    if ki == "cki":
        return dict(ki_barrier=barrier, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    if ki == "dki":
        return dict(ki_barrier=barrier, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(dates))
    if ki == "daily":
        days = [k / 252 for k in range(1, 253)]
        return dict(ki_barrier=barrier, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=days)
    return dict(ki_barrier=None)


def snowball(ki="cki", ko_dates=OBS, ko=103.0, reverse=False, disable=False, ki_barrier=None, maturity=1.0,
             ki_dates=OBS):
    barrier = ki_barrier if ki_barrier is not None else (125.0 if reverse else 75.0)
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity, contract_multiplier=1.0, is_reverse=reverse,
        barrier_config=BarrierConfig(ko_barrier=97.0 if reverse and ko == 103.0 else ko, ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_dates), disable_ko_after_ki=disable,
                                     **ki_kw(ki, dates=ki_dates, barrier=barrier)),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )


def phoenix(memory=True, pay=CouponPayType.INSTANT, ki="cki", ko_dates=OBS, disable=False, barriers=None, ko=105.0,
            ki_dates=OBS):
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_dates), disable_ko_after_ki=disable,
                                     **ki_kw(ki, dates=ki_dates, barrier=70.0)),
        coupon_config=CouponBarrierConfig(coupon_barrier=barriers or [80.0, 108.0, 80.0, 80.0][:len(ko_dates)],
                                          coupon_rate=0.02, coupon_pay_type=pay,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=memory),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )


def _schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def ko_reset(ki="dki", disable=False, ko=103.0, ki_dates=OBS):
    if ki == "cki":
        kw = dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        kw = dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=_schedule(ki_dates, 80.0))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=2.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(OBS, ko), ki_barrier=80.0,
                                     disable_ko_after_ki=disable, **kw),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                                                                             2.0), 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
    )


def term_env():
    market = env()
    market.vol_surface = TermStructureVolSurface(times=[0.12, 0.29, 0.5, 1.0], vols=[0.18, 0.24, 0.22, 0.22])
    return market


SSE = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def dated(family, ki="dki"):
    from intraday.conftest import dated_ko_reset, dated_phoenix, dated_snowball

    product = {"snowball": dated_snowball, "phoenix": dated_phoenix, "ko_reset": dated_ko_reset}[family](
        SSE, datetime(2026, 3, 16))
    if ki == "cki":
        product.barrier_config = replace(product.barrier_config, ki_continuous=True,
                                         ki_observation_type=ObservationType.CONTINUOUS,
                                         ki_observation_schedule=None, ki_observation_dates=None)
    return product


def dated_env(spot=100.0):
    return env(spot=spot, valuation=datetime(2026, 3, 16), calendar=SSE)


#: (id, engine factory, product factory, market factory) -- every engine class, every product family and KI kind
ROUTES = [
    ("sb_cki", SnowballQuadEngineV2, lambda: snowball("cki"), env),
    ("sb_dki", SnowballQuadEngineV2, lambda: snowball("dki"), env),
    ("sb_daily_ki", SnowballQuadEngineV2, lambda: snowball("daily"), env),
    ("sb_reverse_cki", SnowballQuadEngineV2, lambda: snowball("cki", reverse=True), env),
    ("sb_reverse_dki", SnowballQuadEngineV2, lambda: snowball("dki", reverse=True), env),
    ("sb_disable", SnowballQuadEngineV2, lambda: snowball("cki", disable=True), env),
    ("sb_term_cki", SnowballQuadEngineV2, lambda: snowball("cki"), term_env),
    ("sb_fft", lambda: SnowballQuadEngineV2(QuadV2Params(backend="fft")), lambda: snowball("cki"), env),
    ("sb_direct", lambda: SnowballQuadEngineV2(QuadV2Params(backend="direct")), lambda: snowball("dki"), env),
    ("sb_fine", lambda: SnowballQuadEngineV2(QuadV2Params(order=10, cells_per_sd=3.0)), lambda: snowball("cki"), env),
    ("ph_mem_instant_cki", PhoenixQuadEngineV2, lambda: phoenix(True, CouponPayType.INSTANT), env),
    ("ph_mem_expiry_cki", PhoenixQuadEngineV2, lambda: phoenix(True, CouponPayType.EXPIRY), env),
    ("ph_nomem_instant_dki", PhoenixQuadEngineV2, lambda: phoenix(False, CouponPayType.INSTANT, "dki"), env),
    ("ph_nomem_expiry_dki", PhoenixQuadEngineV2, lambda: phoenix(False, CouponPayType.EXPIRY, "dki"), env),
    ("ph_mem_expiry_dki", PhoenixQuadEngineV2, lambda: phoenix(True, CouponPayType.EXPIRY, "dki"), env),
    ("kr_dki", KOResetSnowballQuadEngineV2, lambda: ko_reset("dki"), env),
    ("kr_cki", KOResetSnowballQuadEngineV2, lambda: ko_reset("cki"), env),
    ("dated_sb_dki", SnowballQuadEngineV2, lambda: dated("snowball"), dated_env),
    ("dated_sb_cki", SnowballQuadEngineV2, lambda: dated("snowball", "cki"), dated_env),
    ("dated_ph_dki", PhoenixQuadEngineV2, lambda: dated("phoenix"), dated_env),
    ("dated_kr_dki", KOResetSnowballQuadEngineV2, lambda: dated("ko_reset"), dated_env),
]
IDS = [r[0] for r in ROUTES]


def route(name):
    _, engine, product, market = next(r for r in ROUTES if r[0] == name)
    return engine(), product(), market()


def note_channels(context):
    """Channels whose continuations are note-state transitions (the EXPIRY coupons' termination states are not)."""
    contract = context._march.contract
    touched = set()
    for event in contract.events:
        for row in tuple(event.rows) + tuple(event.boundary_rows):
            for action in row:
                for _, _, selected in action.channel_terms:
                    touched |= set(selected)
    return [i for i in range(1, len(context.channels)) if i not in touched]


# --- AC7, D23 -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_full_request_fills_every_number_and_pv_is_price(name):
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    assert isinstance(stats, EventStatistics)
    assert set(stats.unavailable) <= {"knock_in.expected_time"}
    if stats.unavailable:
        assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert stats.ki_table is not None and stats.knock_in.probability is not None
    assert stats.knock_in.expected_loss is not None and stats.knock_in.expected_loss >= 0.0
    assert stats.provenance.statistics_model == stats.provenance.price_model == type(engine).__name__
    assert stats.standard_errors is None
    assert (stats.coupons is not None) == isinstance(product, PhoenixOption)          # AC11
    # D23: pv is price()'s value, bitwise (here read from the statistics context's own price channel); and on
    # price_with_events
    assert not engine._prepared._march.pointwise_recursion
    assert stats.pv == type(engine)(engine.params).price(product, market)
    result = type(engine)(engine.params).price_with_events(product, market)
    assert result.npv == stats.pv
    assert EventType.KI in result.event_distribution.stream_times


FFT_CASES = {
    "snowball_cki": (SnowballQuadEngineV2, lambda: snowball("cki")),
    "phoenix_mem_expiry_cki": (PhoenixQuadEngineV2, lambda: phoenix(True, CouponPayType.EXPIRY)),
    "ko_reset_dki": (KOResetSnowballQuadEngineV2, lambda: ko_reset("dki")),
}


@pytest.mark.parametrize("name", list(FFT_CASES))
def test_pv_is_price_bitwise_on_the_forced_fft_backend(name):
    """D23 on the FFT backend: the statistics context batches 13-31 channels through one FFT einsum where ``price()``
    batches one, so a channel-count-dependent reduction order (the x86_64 CI's BLAS) would break ``pv == price()``
    here first (CP6). Both runs really take the FFT path."""
    cls, make = FFT_CASES[name]
    pricer = cls(QuadV2Params(backend="fft"))
    price = pricer.price(make(), env())
    engine = cls(QuadV2Params(backend="fft"))
    stats = engine.calculate_event_stats(make(), env())
    assert "fft" in pricer._prepared.diagnostics["backends"] and "fft" in engine._prepared.diagnostics["backends"]
    assert len(engine._prepared.channels) > 10 * len(pricer._prepared.channels)
    assert stats.pv == price
    assert cls(QuadV2Params(backend="fft")).price_with_events(make(), env()).npv == price


# --- AC8 ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_partition_residuals_are_round_off(name):
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND
    assert abs(stats.reconciliation_error) <= RECONCILIATION_BOUND * max(1.0, abs(stats.pv))
    # cash to pay on deterministic-payment rows is the discounted cash over the payment DF (AC12)
    obs = stats.observations
    df = np.array([market.get_discount_factor(t) for t in obs.payment_times])
    np.testing.assert_allclose(obs.cash_to_pay * df, obs.discounted_cash, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name", ["sb_cki", "ph_mem_expiry_cki", "kr_dki"])
def test_residuals_stay_at_round_off_under_refinement(name):
    _, product, market = route(name)
    for params in (QuadV2Params(order=6, cells_per_sd=1.5), QuadV2Params(), QuadV2Params(order=10, cells_per_sd=3.0)):
        stats = type(route(name)[0])(params).calculate_event_stats(product, market)
        assert abs(stats.probability_residual) <= RESIDUAL_BOUND
        assert abs(stats.reconciliation_error) <= RECONCILIATION_BOUND * abs(stats.pv)


# --- AC14 -----------------------------------------------------------------------------------------------------------


ADJOINT_CASES = IDS + ["ko_at_valuation", "carried", "ki_at_valuation_cki", "ki_at_valuation_dki",
                       "ki_first_at_next_observation", "after_phase",
                       "receivable_alive", "phoenix_count_memory"]


def adjoint_case(name):
    if name in IDS:
        return route(name) + ({},)
    if name == "ko_at_valuation":
        return SnowballQuadEngineV2(), snowball("cki", ko_dates=(0.0, *OBS)), env(spot=110.0), {}
    if name == "carried":
        product = phoenix(True, CouponPayType.EXPIRY)
        product._otc_lifecycle_knocked_in = True
        return PhoenixQuadEngineV2(), product, env(), {}
    if name == "ki_at_valuation_cki":
        return SnowballQuadEngineV2(), snowball("cki"), env(spot=70.0), {}
    if name == "ki_at_valuation_dki":
        return SnowballQuadEngineV2(), snowball("dki", ko_dates=(0.0, *OBS), ki_dates=(0.0, *OBS)), env(spot=70.0), {}
    if name == "ki_first_at_next_observation":
        return SnowballQuadEngineV2(), snowball("dki", ko_dates=(0.0, *OBS)), env(spot=70.0), {}
    if name == "after_phase":
        return SnowballQuadEngineV2(), snowball("dki", ko_dates=(0.0, *OBS)), env(spot=105.0), {"event_phase": "after"}
    if name == "receivable_alive":
        return SnowballQuadEngineV2(), snowball("cki"), env(), {"lifecycle_state": receivable_state()}
    if name == "phoenix_count_memory":
        state = AutocallableLifecycleState(coupon_memory_count=1)
        product = phoenix(True, CouponPayType.INSTANT, "dki")
        product.coupon_config = replace(product.coupon_config, fixed_coupon_year_fraction=0.25)
        return PhoenixQuadEngineV2(), product, env(), {"lifecycle_state": state}
    raise KeyError(name)


@pytest.mark.parametrize("name", ADJOINT_CASES)
def test_forward_pass_reproduces_the_backward_columns(name):
    """AC14: Σ m0 x cash + m1 x asset over the pass's actions equals every note-state backward column read at the
    spot: each knock-out record's probability and cash, the maturity and knocked-in settlement units and cash."""
    engine, product, market, kw = adjoint_case(name)
    engine.calculate_event_stats(product, market, **kw)
    context = engine._prepared
    result = context.evaluate([market.spot])
    reconstructed = forward_pass(context._march, market.spot).reconstruct(len(context.channels))
    channels = note_channels(context)
    for i in channels:
        name_i = context.channels[i]
        assert reconstructed[i] == pytest.approx(float(result["components"][name_i][0]), abs=ADJOINT_TOL), name_i
    probability = [i for i in channels if "probability" in context.channels[i]]
    total = sum(reconstructed[i] for i in probability if not context.channels[i].startswith("coupon"))
    backward = sum(float(result["components"][context.channels[i]][0]) for i in probability
                   if not context.channels[i].startswith("coupon"))
    assert total == pytest.approx(backward, abs=ADJOINT_TOL)
    if 0 in channels or not context.diagnostics["expiry_coupons"]:
        # without deferred coupons the price itself is a note-state column
        assert reconstructed[0] == pytest.approx(float(result["price"][0]) - context.pending_pv,
                                                 abs=ADJOINT_TOL * 100)


@pytest.mark.parametrize("name", ["sb_cki", "sb_dki", "sb_daily_ki", "sb_reverse_cki", "sb_reverse_dki",
                                  "ph_mem_instant_cki", "ph_nomem_instant_dki", "kr_dki", "kr_cki", "sb_term_cki",
                                  "dated_sb_dki", "dated_ph_dki", "dated_kr_dki"])
def test_knock_in_before_ko_is_the_disable_ko_after_ki_twins_settlement(name):
    """Independent backward pin: a path that knocks in first never knocks out under ``disable_ko_after_ki``, so the
    twin's KI settlement is P(KI before any KO)."""
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    twin = deepcopy(product)
    twin.barrier_config = replace(twin.barrier_config, disable_ko_after_ki=True)
    reference = type(engine)(engine.params).calculate_event_stats(twin, market)
    assert stats.knock_in.probability == pytest.approx(reference.ki_settlement.probability, abs=TWIN_BOUND)
    assert stats.knock_in.probability >= stats.ki_settlement.probability - TWIN_BOUND


# --- AC13 -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_knock_in_table_identities(name):
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    table, ki = stats.ki_table, stats.knock_in
    assert float(table.probability.sum()) == pytest.approx(ki.probability, rel=1e-12, abs=1e-15)
    np.testing.assert_array_equal(table.payment_times, table.times)                       # D8
    assert np.all(table.probability >= -1e-12)
    assert np.all(np.diff(table.survival) <= 1e-12)
    continuous = product.barrier_config.ki_continuous
    mean = float(np.sum(table.times * table.probability) / ki.probability)
    if continuous:
        day = 1.0 / market.bus_days_in_year
        assert abs(mean - ki.expected_time) <= day
        assert table.interpolation_scheme == v2_statistics.FIRST_PASSAGE
        horizon = float(product.get_maturity(market)) if not isinstance(product, KnockOutResetSnowballOption) \
            else float(product.get_pre_maturity_time(market))
        assert table.times[-1] == pytest.approx(horizon)
        if market.calendar is None:
            assert table.dates is None
            np.testing.assert_allclose(np.diff(table.times[:-1]), day, rtol=1e-12)
        else:
            assert all(SSE.is_business_day(d) for d in table.dates[:-1])
        # rows are interpolated exactly where a day boundary is not an event time
        events = {0.0} | {float(t) for _, t in engine._prepared.diagnostics["canonical_times"]}
        starts = np.r_[0.0, table.times[:-1]]
        expected = [not (any(abs(e - b) <= 1e-10 for e in events) and any(abs(e - a) <= 1e-10 for e in events))
                    for a, b in zip(starts, table.times)]
        np.testing.assert_array_equal(table.interpolated, expected)
    else:
        assert mean == pytest.approx(ki.expected_time, rel=1e-12)
        records = [r for r in product.resolve_ki_observations(market) if r.observation_time >= 0]
        horizon = float(product.get_pre_maturity_time(market)) if isinstance(product, KnockOutResetSnowballOption) \
            else float(product.get_maturity(market))
        np.testing.assert_allclose(table.times, [r.observation_time for r in records
                                                 if r.observation_time <= horizon + 1e-12])
        assert not table.interpolated.any() and table.interpolation_scheme is None
        if market.calendar is not None:
            assert table.dates == tuple(r.observation_date for r in records
                                        if r.observation_time <= horizon + 1e-12)


def test_continuous_rows_follow_the_first_passage_law_within_an_interval():
    """One quarter-long reflection interval spreads its crossing mass over the days by the Brownian first-passage
    law: the mass up to each day matches the closed form when no knock-out can intervene."""
    from scipy.special import ndtr

    product = snowball("cki", ko_dates=(1.0,), ko=1e9)
    market = env()
    stats = SnowballQuadEngineV2().calculate_event_stats(product, market)
    cumulative = np.cumsum(stats.ki_table.probability)
    m = (0.03 - 0.01 - 0.5 * 0.22 ** 2)

    def first_hit(t):
        d, v = math.log(100.0 / 75.0), 0.22 ** 2 * t
        return float(ndtr((-d - m * t) / math.sqrt(v)) + math.exp(-2 * m * t * d / v) * ndtr((m * t - d) / math.sqrt(v)))

    for k in (10, 63, 126, 200):
        assert cumulative[k] == pytest.approx(first_hit(stats.ki_table.times[k]), abs=1e-10)


# --- AC9 ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
def test_no_knock_in_after_a_certain_knock_out(family):
    """Every path knocks out at 0.25 (the KO barrier below any reachable spot); the discrete KI observations come
    after it and cross often -- none of it counts (the legacy KO-independent monitor counted it)."""
    if family == "snowball":
        product, engine = snowball("dki", ko=1e-3, ki_dates=(0.5, 0.75, 1.0)), SnowballQuadEngineV2()
    elif family == "phoenix":
        product = phoenix(False, CouponPayType.INSTANT, "dki", ko=1e-3, ki_dates=(0.5, 0.75, 1.0))
        engine = PhoenixQuadEngineV2()
    else:
        product, engine = ko_reset("dki", ko=1e-3, ki_dates=(0.5, 0.75, 1.0)), KOResetSnowballQuadEngineV2()
    stats = engine.calculate_event_stats(product, env(vol=0.6))
    assert stats.observations.probability[0] == pytest.approx(1.0, abs=1e-12)
    assert stats.knock_in.probability == 0.0
    assert stats.knock_in.expected_loss == pytest.approx(0.0, abs=1e-12)
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert float(stats.ki_table.probability.sum()) == 0.0


# --- AC10 -----------------------------------------------------------------------------------------------------------


def _spy(monkeypatch):
    calls = []
    original = v2_statistics.forward_pass

    def spy(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    monkeypatch.setattr(v2_statistics, "forward_pass", spy)
    return calls


@pytest.mark.parametrize("family", ["snowball", "phoenix"])
def test_knocked_out_at_valuation(family, monkeypatch):
    calls = _spy(monkeypatch)
    if family == "snowball":
        product, engine = snowball("cki", ko_dates=(0.0, *OBS)), SnowballQuadEngineV2()
    else:
        product, engine = phoenix(True, CouponPayType.INSTANT, ko_dates=(0.0, *OBS),
                                  barriers=[80.0] * 5), PhoenixQuadEngineV2()
    stats = engine.calculate_event_stats(product, env(spot=110.0))
    assert stats.observations.probability[0] == 1.0 and stats.observations.times[0] == 0.0
    assert stats.maturity.probability == 0.0 and stats.ki_settlement.probability == 0.0
    assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0
    assert stats.knock_in.expected_loss == 0.0 and stats.expected_lifetime == 0.0
    assert stats.unavailable == {"knock_in.expected_time": stats.unavailable["knock_in.expected_time"]}
    assert stats.pv == engine.price(product, env(spot=110.0))
    assert not calls


@pytest.mark.parametrize("name", ["sb_cki", "ph_mem_expiry_cki", "kr_dki"])
def test_carried_knock_in(name, monkeypatch):
    calls = _spy(monkeypatch)
    engine, product, market = route(name)
    product._otc_lifecycle_knocked_in = True
    stats = engine.calculate_event_stats(product, market)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    assert stats.maturity.probability == 0.0 and stats.unavailable == {}
    assert not calls


@pytest.mark.parametrize("ki", ["cki", "dki"])
def test_knock_in_at_valuation(ki, monkeypatch):
    calls = _spy(monkeypatch)
    product = snowball(ki, ko_dates=(0.0, *OBS) if ki == "dki" else OBS, ki_dates=(0.0, *OBS))
    stats = SnowballQuadEngineV2().calculate_event_stats(product, env(spot=70.0))
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    assert stats.maturity.probability == pytest.approx(0.0, abs=1e-12)
    assert not calls


def test_no_knock_in_barrier(monkeypatch):
    calls = _spy(monkeypatch)
    stats = SnowballQuadEngineV2().calculate_event_stats(snowball("none"), env())
    assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0
    assert stats.ki_settlement.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert not calls


# --- D17 ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["sb_cki", "ph_mem_expiry_cki", "kr_dki"])
@pytest.mark.parametrize("streams", [frozenset({EventType.KO}),
                                     frozenset({EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})])
def test_pruned_request_skips_the_forward_pass(name, streams, monkeypatch):
    engine, product, market = route(name)
    full = engine.calculate_event_stats(product, market)
    calls = _spy(monkeypatch)
    pruned = type(engine)(engine.params).calculate_event_stats(product, market, streams=streams)
    assert not calls
    for path in ("knock_in.probability", "knock_in.expected_time", "ki_table"):
        assert pruned.unavailable[path] == "not requested"
    if pruned.coupons is not None:
        assert pruned.unavailable["coupons.probability"] == "not requested"
        np.testing.assert_array_equal(pruned.coupons.discounted_cash, full.coupons.discounted_cash)
    assert pruned.pv == full.pv
    np.testing.assert_array_equal(pruned.observations.probability, full.observations.probability)
    assert pruned.ki_settlement == full.ki_settlement and pruned.maturity == full.maturity
    assert pruned.knock_in.expected_loss == full.knock_in.expected_loss
    np.testing.assert_array_equal(pruned.ledger.discounted_cash, full.ledger.discounted_cash)


# --- AC17 -----------------------------------------------------------------------------------------------------------


def receivable_state():
    state = AutocallableLifecycleState(valuation_point=ValuationPoint(date=VALUATION))
    state.ledger.register(RealizedCashflow(
        cashflow_id="coupon-pending", event_type=LifecycleEventType.COUPON, amount=2.0,
        determination_date=VALUATION - timedelta(days=5), payment_date=VALUATION + timedelta(days=10)))
    return state


def ko_state(paid_days, knocked_in=False):
    state = AutocallableLifecycleState(knocked_in=knocked_in)
    state.mark_ko(VALUATION - timedelta(days=3), cashflow=105.0, settlement_date=VALUATION + timedelta(days=paid_days),
                  valuation_point=ValuationPoint(date=VALUATION))
    return state


def test_receivables_sit_in_the_cash_identity_not_in_the_events():
    engine, product, market = SnowballQuadEngineV2(), snowball("cki"), env()
    state = receivable_state()
    stats = engine.calculate_event_stats(product, market, lifecycle_state=state)
    rec = stats.receivables
    assert rec.source == ("coupon",) and rec.decision_dates == (VALUATION - timedelta(days=5),)
    assert rec.decision_times[0] == pytest.approx(-5 / 365)
    assert rec.cash_to_pay[0] == 2.0
    assert rec.discounted_cash[0] == pytest.approx(2.0 * market.get_discount_factor(10 / 365), rel=1e-12)
    assert stats.pv == engine.price(product, market, lifecycle_state=state)
    assert abs(stats.reconciliation_error) <= RECONCILIATION_BOUND * stats.pv
    plain = SnowballQuadEngineV2().calculate_event_stats(product, market)
    np.testing.assert_array_equal(stats.observations.probability, plain.observations.probability)
    assert stats.probability_residual == plain.probability_residual


@pytest.mark.parametrize("cls,factory", [(SnowballQuadEngineV2, lambda: snowball("cki")),
                                         (PhoenixQuadEngineV2, lambda: phoenix()),
                                         (KOResetSnowballQuadEngineV2, lambda: ko_reset())])
@pytest.mark.parametrize("knocked_in", [False, True])
def test_terminated_unpaid_returns_receivables_only(cls, factory, knocked_in):
    product, market, state = factory(), env(), ko_state(paid_days=10, knocked_in=knocked_in)
    stats = cls().calculate_event_stats(product, market, lifecycle_state=state)
    assert stats.terminated and stats.receivables.source == ("ko",)
    assert stats.receivables.cash_to_pay[0] == 105.0
    assert stats.pv == cls().price(product, market, lifecycle_state=state)
    assert abs(stats.reconciliation_error) <= 1e-12 and stats.probability_residual == 0.0
    assert stats.knock_in.probability == (1.0 if knocked_in else 0.0)
    result = cls().price_with_events(product, market, lifecycle_state=state)
    assert result.event_distribution.terminated and result.npv == stats.pv
    assert cls().calculate_event_stats_at_spots(product, market, [99.0, 101.0]) is not None


@pytest.mark.parametrize("cls,factory", [(SnowballQuadEngineV2, lambda: snowball("cki")),
                                         (PhoenixQuadEngineV2, lambda: phoenix())])
def test_terminated_settled_returns_none(cls, factory):
    from quantark.execution.context import default_context
    from quantark.execution.contracts import OutputKind, PricingOperation, PricingRequest
    from quantark.execution.errors import CapabilityError

    product, market, state = factory(), env(), ko_state(paid_days=-1)
    assert cls().calculate_event_stats(product, market, lifecycle_state=state) is None
    result = cls().price_with_events(product, market, lifecycle_state=state)
    assert result.event_distribution.terminated
    assert result.npv == cls().price(product, market, lifecycle_state=state)
    request = PricingRequest(product, market, operation=PricingOperation.EVENT_STATS,
                             outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}), lifecycle_state=state)
    with pytest.raises(CapabilityError, match="settled"):
        cls().execute(request, default_context(environ={}))


def test_matured_nothing_unpaid_returns_none():
    state = AutocallableLifecycleState(matured=True, alive=False)
    state.ledger.register(RealizedCashflow(
        cashflow_id="maturity", event_type=LifecycleEventType.MATURITY, amount=100.0,
        determination_date=VALUATION - timedelta(days=4), payment_date=VALUATION - timedelta(days=1)))
    state.valuation_point = ValuationPoint(date=VALUATION)
    assert SnowballQuadEngineV2().calculate_event_stats(snowball("cki"), env(), lifecycle_state=state) is None


def test_matured_unpaid_is_a_knocked_in_settlement_receivable():
    state = AutocallableLifecycleState(matured=True, alive=False, knocked_in=True)
    state.ledger.register(RealizedCashflow(
        cashflow_id="maturity", event_type=LifecycleEventType.MATURITY, amount=80.0,
        determination_date=VALUATION - timedelta(days=1), payment_date=VALUATION + timedelta(days=2)))
    state.valuation_point = ValuationPoint(date=VALUATION)
    stats = SnowballQuadEngineV2().calculate_event_stats(snowball("cki"), env(), lifecycle_state=state)
    assert stats.receivables.source == ("ki_settlement",)
    assert stats.knock_in.probability == 1.0


# --- refusals (never approximated) ----------------------------------------------------------------------------------


@pytest.mark.parametrize("market", [env(), env(vol=0.25, r=0.05), env(spot=90.0, vol=0.3, r=0.01)],
                         ids=["base", "vol25_r5", "spot90"])
def test_pointwise_continuous_backend_refuses_the_knock_in_fields(market):
    """The pointwise continuous readout is a batched readout over every channel together: the statistics context's
    price channel is not price()'s bitwise there (measured: 1 ULP at vol 25%, r 5%), so pv comes from price() itself."""
    engine = SnowballQuadEngineV2(QuadV2Params(continuous_grid_backend="pointwise"))
    product = snowball("cki")
    stats = engine.calculate_event_stats(product, market)
    assert engine._prepared._march.pointwise_recursion
    for path in ("knock_in.probability", "knock_in.expected_time", "ki_table"):
        assert stats.unavailable[path].startswith("undefined:") and "pointwise" in stats.unavailable[path]
    assert stats.pv == SnowballQuadEngineV2(QuadV2Params(continuous_grid_backend="pointwise")).price(product, market)
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND


@pytest.mark.parametrize("cls,factory,market", [
    (SnowballQuadEngineV2, lambda: snowball("cki"), env(vol=0.25, r=0.01)),
    (SnowballQuadEngineV2, lambda: snowball("cki"), env(spot=90.0, vol=0.25, r=0.01)),
    (PhoenixQuadEngineV2, lambda: phoenix(), env(vol=0.2, r=0.05)),
    (PhoenixQuadEngineV2, lambda: phoenix(), env(spot=110.0, vol=0.25, r=0.01)),
], ids=["sb_vol25_r1", "sb_spot90", "ph_vol20_r5", "ph_spot110"])
def test_pointwise_route_at_spots_pv_is_price_at_the_env_spot(cls, factory, market):
    """D23 holds for ``calculate_event_stats_at_spots`` too: on a pointwise route its pv at the env spot is price()'s
    value bitwise, as ``calculate_event_stats``'s is (each case measured 1 ULP off price() before the rule)."""
    product = factory()
    engine = cls(QuadV2Params(continuous_grid_backend="pointwise"))
    spots = [market.spot - 1.0, market.spot, market.spot + 1.0]
    stats = engine.calculate_event_stats_at_spots(product, market, spots)
    assert engine._prepared._march.pointwise_recursion
    price = cls(QuadV2Params(continuous_grid_backend="pointwise")).price(product, market)
    assert stats[1].pv == price
    assert stats[1].pv == engine.calculate_event_stats(product, market).pv


# --- the KO-reset layout --------------------------------------------------------------------------------------------


def test_ko_reset_layout():
    stats = KOResetSnowballQuadEngineV2().calculate_event_stats(ko_reset("dki"), env())
    obs = stats.observations
    assert obs.schedule == ("pre", "post") * 4 + ("post",) * 4
    np.testing.assert_allclose(obs.times, [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.5, 1.75, 2.0])
    assert stats.maturity.time == 1.0 and stats.ki_settlement.time == 2.0
    # F5: the first schedule's maturity mass leaves the survival at its own determination
    cumulative = 1.0 - np.cumsum(obs.probability)
    np.testing.assert_allclose(obs.survival, cumulative - np.where(obs.times >= 1.0, stats.maturity.probability, 0.0),
                               atol=1e-15)
    assert stats.coupons is None


# --- AC12: cash to pay on path-dependent (EXPIRY) coupon rows -------------------------------------------------------


@pytest.mark.parametrize("memory", [False, True])
def test_expiry_coupon_cash_to_pay_is_the_coupon_earned_at_the_row(memory):
    """Zero volatility, flat 10% carry: row 1 meets its barrier (3, or 6 with the missed row 0), row 2 knocks out with
    its coupon (3). Paid at the knock-out settlement, so E[C] != E[DF C] / DF(any fixed time): the row's cash to pay
    is the coupon earned there, computed directly (memory: its own undiscounted channel)."""
    from test_asset.equity.lifecycle.test_quad_v2_coupon_event_stats import environment, phoenix as fixed_phoenix

    p = fixed_phoenix(times=(0.2, 0.4, 0.6, 0.8), ko=[200.0, 200.0, 90.0, 200.0], coupon=[110.0, 90.0, 90.0, 90.0],
                      memory=memory, pay_type=CouponPayType.EXPIRY)
    stats = PhoenixQuadEngineV2().calculate_event_stats(p, environment(vol=0.0, r=0.1, q=0.1))
    first = 6.0 if memory else 3.0
    np.testing.assert_allclose(stats.coupons.cash_to_pay, [0.0, first, 3.0, 0.0], atol=1e-12)
    assert stats.coupons.payment_is_path_dependent.all()
    np.testing.assert_allclose(stats.coupons.discounted_cash, np.array([0.0, first, 3.0, 0.0]) * np.exp(-0.06),
                               atol=1e-12)

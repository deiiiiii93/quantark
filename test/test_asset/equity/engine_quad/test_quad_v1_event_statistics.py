"""QUAD V1 ``EventStatistics`` (unified event statistics spec T17: AC7, AC8, AC9, AC10, AC13, AC14; D17, D23).

Every termination outcome is a column of the V1 statistics recursion, marched through the price's own operators -- the
maturity outcome too (never ``pv - Σ KO``); the knock-in path statistics come from the forward adjoint pass, the exact
transpose of that recursion's backward step (``quad/v1_adjoint.py``). The adjoint identity (AC14): the pass's masses
reproduce every note-state backward column -- the knock-out units, the maturity and knocked-in settlement units and
cash -- to round-off. Independent pins: the ``disable_ko_after_ki`` twin's knocked-in settlement (= P(KI before any KO)
where the twin's events are the same), the closed-form first passage (continuous KI, knock-out out of reach), a dot
product test of every transposed operator.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.quad import v1_adjoint, v1_statistics
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v1_adjoint import forward_pass
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option import PhoenixOption, SnowballOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.cashleg.event_distribution import EventType
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.vol.vol_surface import TermStructureVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.enum.option_enums import PostKOScheduleMode

VALUATION = datetime(2024, 1, 1)
OBS = (0.25, 0.5, 0.75, 1.0)
#: AC14: the forward pass reproduces a backward column read at the spot (probabilities and cash on a 100 notional)
ADJOINT_TOL = 1e-10
#: P(KI before KO) against the ``disable_ko_after_ki`` twin's KI settlement where the twin's events coincide with the
#: note's (nodal events: the knock-out of a tie is a node mask either way)
TWIN_BOUND = 1e-10


def env(spot=100.0, vol=0.22, r=0.03, q=0.01, valuation=VALUATION):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=r), div_yield=ContinuousDividendYield(div_yield=q),
                              valuation_date=valuation)


def term_env():
    market = env()
    market.vol_surface = TermStructureVolSurface(times=[0.12, 0.29, 0.5, 1.0], vols=[0.18, 0.24, 0.22, 0.22])
    return market


def ki_kw(ki, dates=OBS, barrier=75.0):
    if ki == "cki":
        return dict(ki_barrier=barrier, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    if ki == "dki":
        return dict(ki_barrier=barrier, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(dates))
    if ki == "monthly":
        days = [k / 12 for k in range(1, 13)]
        return dict(ki_barrier=barrier, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=days)
    return dict(ki_barrier=None)


def snowball(ki="cki", ko_dates=OBS, ko=103.0, reverse=False, disable=False, ki_barrier=None, ki_dates=OBS):
    barrier = ki_barrier if ki_barrier is not None else (125.0 if reverse else 75.0)
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=reverse,
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


def sb(**kw):
    return lambda: SnowballQuadEngine(QuadParams(grid_points=401, **kw))


def ph(**kw):
    return lambda: PhoenixQuadEngine(QuadParams(grid_points=401, **kw))


def kr(**kw):
    return lambda: KOResetSnowballQuadEngine(QuadParams(grid_points=401, **kw))


NODAL = dict(event_projection=EventProjectionMode.NODAL, event_smoothing_cells=0)

#: (id, engine factory, product factory, market factory) -- every engine class, product family, KI kind and route
ROUTES = [
    ("sb_cki", sb(), lambda: snowball("cki"), env),
    ("sb_dki", sb(), lambda: snowball("dki"), env),
    ("sb_monthly_ki", sb(), lambda: snowball("monthly"), env),
    ("sb_reverse_cki", sb(), lambda: snowball("cki", reverse=True), env),
    ("sb_reverse_dki", sb(), lambda: snowball("dki", reverse=True), env),
    ("sb_disable", sb(), lambda: snowball("cki", disable=True), env),
    ("sb_term_cki", sb(), lambda: snowball("cki"), term_env),
    ("sb_nodal_dki", sb(**NODAL), lambda: snowball("dki"), env),
    ("sb_smoothed_dki", sb(event_projection=EventProjectionMode.NODAL), lambda: snowball("dki"), env),
    ("sb_transition_dki", sb(readout="transition"), lambda: snowball("dki"), env),
    ("sb_no_filter", sb(fft_filter_alpha=0.0, fft_padding_factor=1), lambda: snowball("cki"), env),
    ("sb_simpson", sb(integration_rule="simpson"), lambda: snowball("dki"), env),
    ("ph_mem_instant_cki", ph(), lambda: phoenix(True, CouponPayType.INSTANT), env),
    ("ph_mem_expiry_cki", ph(), lambda: phoenix(True, CouponPayType.EXPIRY), env),
    ("ph_nomem_instant_dki", ph(), lambda: phoenix(False, CouponPayType.INSTANT, "dki"), env),
    ("ph_nomem_expiry_dki", ph(), lambda: phoenix(False, CouponPayType.EXPIRY, "dki"), env),
    ("ph_mem_expiry_dki", ph(), lambda: phoenix(True, CouponPayType.EXPIRY, "dki"), env),
    ("ph_nodal_dki", ph(**NODAL), lambda: phoenix(True, CouponPayType.INSTANT, "dki"), env),
    ("kr_dki", kr(), lambda: ko_reset("dki"), env),
    ("kr_cki", kr(), lambda: ko_reset("cki"), env),
    ("kr_nodal_dki", kr(**NODAL), lambda: ko_reset("dki"), env),
]
IDS = [r[0] for r in ROUTES]


def route(name):
    _, engine, product, market = next(r for r in ROUTES if r[0] == name)
    return engine(), product(), market()


def _is_continuous(product):
    return bool(product.barrier_config.ki_continuous)


def _horizon(product, market):
    if isinstance(product, KnockOutResetSnowballOption):
        return float(product.get_pre_maturity_time(market))
    return float(product.get_maturity(market))


# --- AC7, D23 -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_full_request_fills_every_number_and_pv_is_price(name):
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    assert isinstance(stats, EventStatistics)
    assert set(stats.unavailable) <= {"knock_in.expected_time"}, stats.unavailable
    if stats.unavailable:
        assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert stats.ki_table is not None and stats.knock_in.probability is not None
    assert stats.knock_in.expected_loss is not None and stats.knock_in.expected_loss >= -1e-12
    assert stats.provenance.statistics_model == stats.provenance.price_model == "quad_v1"
    assert stats.standard_errors is None
    assert (stats.coupons is not None) == isinstance(product, PhoenixOption)          # AC11
    fresh = type(engine)(engine.params)
    assert stats.pv == fresh.price(product, market)                                   # D23, bitwise
    result = type(engine)(engine.params).price_with_events(product, market)
    assert result.npv == stats.pv
    assert EventType.KI in result.event_distribution.stream_times


# --- AC8 ------------------------------------------------------------------------------------------------------------

#: AC8 bounds of the V1 recursion, measured on every route here (T17 report, tmp/T17-measure.txt): the outcome
#: columns ride the price's own linear operators, so the partition and the cash identity close to round-off
#: (probability residual <= 1.4e-14, reconciliation <= 6e-16 x pv) -- except EXPIRY Phoenix coupons, whose discount in
#: ``price`` is the termination surface W knocked out by the nodal weight while the statistics columns take the
#: observation's joint projection: 6.7e-9 x pv at 401 nodes, 1.1e-9 at 1001, 6.8e-11 at 4001 (it shrinks)
RESIDUAL_BOUND = 1e-12
RECONCILIATION_BOUND = 1e-12
EXPIRY_RECONCILIATION_BOUND = 1e-8


def _reconciliation_bound(product):
    expiry = isinstance(product, PhoenixOption) and product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    return EXPIRY_RECONCILIATION_BOUND if expiry else RECONCILIATION_BOUND


@pytest.mark.parametrize("name", IDS)
def test_partition_residuals_are_bounded(name):
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND
    assert abs(stats.reconciliation_error) <= _reconciliation_bound(product) * max(1.0, abs(stats.pv))
    obs = stats.observations
    df = np.array([market.get_discount_factor(t) for t in obs.payment_times])
    np.testing.assert_allclose(obs.cash_to_pay * df, obs.discounted_cash, rtol=1e-12, atol=1e-14)   # AC12


def test_expiry_reconciliation_shrinks_under_refinement():
    _, product, market = route("ph_mem_expiry_cki")
    gaps = [abs(PhoenixQuadEngine(QuadParams(grid_points=points)).calculate_event_stats(product, market)
                .reconciliation_error) for points in (401, 1001, 4001)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] <= 1e-10 * 100.0


def test_maturity_is_its_own_column_not_the_price_residual(monkeypatch):
    """The maturity row is marched directly: perturbing the reported pv leaves it alone (the legacy row was
    ``pv - Σ KO cash``) and the perturbation shows in ``reconciliation_error``."""
    engine, product, market = route("sb_cki")
    stats = engine.calculate_event_stats(product, market)
    monkeypatch.setattr(SnowballQuadEngine, "price", lambda self, p, e: 1000.0)
    moved = SnowballQuadEngine(engine.params).calculate_event_stats(product, market)
    assert moved.maturity == stats.maturity and moved.ki_settlement == stats.ki_settlement
    assert moved.reconciliation_error == pytest.approx(stats.reconciliation_error + 1000.0 - stats.pv, abs=1e-9)


# --- AC14 -----------------------------------------------------------------------------------------------------------


ADJOINT_CASES = IDS + ["ko_reset_carried", "carried", "ki_at_valuation_cki", "ki_at_valuation_dki"]


def adjoint_case(name):
    if name in IDS:
        return route(name)
    if name == "carried":
        product = phoenix(True, CouponPayType.EXPIRY)
        product._otc_lifecycle_knocked_in = True
        return ph()(), product, env()
    if name == "ko_reset_carried":
        product = ko_reset("dki")
        product._otc_lifecycle_knocked_in = True
        return kr()(), product, env()
    if name == "ki_at_valuation_cki":
        return sb()(), snowball("cki"), env(spot=70.0)
    if name == "ki_at_valuation_dki":
        return sb()(), snowball("dki", ko_dates=(0.0, *OBS), ki_dates=(0.0, *OBS)), env(spot=70.0)
    raise KeyError(name)


@pytest.mark.parametrize("name", ADJOINT_CASES)
def test_forward_pass_reproduces_the_backward_columns(name):
    """AC14: the transposed march's masses equal every note-state backward column read at the spot: each knock-out
    record's unit, the maturity and knocked-in settlement units and cash."""
    engine, product, market = adjoint_case(name)
    engine.calculate_event_stats(product, market)
    log = engine._last_v1_march
    result = forward_pass(log)
    assert log.columns, "the statistics march logged no backward column"
    for key, backward in log.columns.items():
        if key[0] == "ko":
            got = result.sources.get(key, 0.0)
        else:
            got = result.terminal[key[1]]
        assert got == pytest.approx(backward, abs=ADJOINT_TOL), key
    ko = sum(result.sources.get(k, 0.0) for k in log.columns if k[0] == "ko")
    assert ko == pytest.approx(sum(v for k, v in log.columns.items() if k[0] == "ko"), abs=ADJOINT_TOL)


@pytest.mark.parametrize("name", ["sb_cki", "sb_transition_dki", "ph_nodal_dki", "kr_cki", "sb_no_filter",
                                  "sb_simpson"])
def test_every_transposed_operator_passes_the_dot_product_test(name):
    """``<A v, d> = <v, A^T d>`` for the diffusion, the bridge, the readout rows and every event map of the march."""
    engine, product, market = route(name)
    engine.calculate_event_stats(product, market)
    log = engine._last_v1_march
    mu = log.math_utils
    n = mu.grid.size
    rng = np.random.default_rng(7)
    v, d = rng.normal(size=n), rng.normal(size=n)
    for step in log.steps:
        tau, alpha, beta, prefactor, omega = step.diffusion
        forward = engine._diffuse_fft(v.copy(), mu, omega, prefactor, 0, n - 1, (n - 1) % 2, alpha, beta, tau)
        adjoint = v1_adjoint.diffuse_transpose(d, mu, omega, prefactor, alpha, beta, tau)[0]
        assert float(forward @ d) == pytest.approx(float(v @ adjoint), rel=1e-11, abs=1e-12)
        if step.bridge is not None:
            log_barrier, vol, dt, reverse = step.bridge
            zero = np.zeros(n)
            with_bridge = engine._diffuse_with_bridge(zero, v.copy(), mu, omega, prefactor, 0, n - 1, (n - 1) % 2,
                                                      log_barrier, alpha, beta, vol, dt, tau, reverse)
            plain = engine._diffuse_fft(zero, mu, omega, prefactor, 0, n - 1, (n - 1) % 2, alpha, beta, tau)
            bridged = v1_adjoint.bridge_transpose(d, engine, mu, prefactor, alpha, tau, log_barrier, vol, dt, reverse)
            assert float((with_bridge - plain) @ d) == pytest.approx(float(v @ bridged), rel=1e-11, abs=1e-12)
        if step.event is not None:
            banded = v1_adjoint._BandedEvent(step.event, n, step.time)
            e = rng.normal(size=n)
            w_out, w_in = step.event(v[None, :].copy(), e[None, :].copy(), None)
            lhs = float(w_out[0] @ d) + float(w_in[0] @ e)
            rhs = float(v @ (banded.transpose("out", "out", d) + banded.transpose("out", "in", e))) + float(
                e @ (banded.transpose("in", "out", d) + banded.transpose("in", "in", e)))
            assert lhs == pytest.approx(rhs, rel=1e-11, abs=1e-12)
    if log.transition is not None:
        prefactor, alpha, beta, tau = log.transition
        row = v1_adjoint.transition_readout_weights(mu, prefactor, alpha, beta, tau)
        direct = engine._readout_at_spot(v, mu, prefactor, 0, n - 1, (n - 1) % 2, alpha, beta, tau)
        assert float(row @ v) == pytest.approx(float(direct), rel=1e-11, abs=1e-12)
    assert float(v1_adjoint.interpolation_weights(mu) @ v) == pytest.approx(mu.interpolate(v, 0.0), abs=1e-13)


@pytest.mark.parametrize("name", ["sb_nodal_dki", "kr_nodal_dki", "ph_nodal_dki", "sb_cki", "sb_reverse_cki",
                                  "kr_cki", "sb_term_cki"])
def test_knock_in_before_ko_is_the_disable_ko_after_ki_twins_settlement(name):
    """Independent backward pin: a path that knocks in first never knocks out under ``disable_ko_after_ki``, so the
    twin's KI settlement is P(KI before any KO). Exact where the twin's events coincide with the note's: hard nodal
    events (discrete KI) or continuous KI (the bridge moves the knocked-in mass before the event)."""
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    twin = deepcopy(product)
    twin.barrier_config = replace(twin.barrier_config, disable_ko_after_ki=True)
    reference = type(engine)(engine.params).calculate_event_stats(twin, market)
    assert stats.knock_in.probability == pytest.approx(reference.ki_settlement.probability, abs=TWIN_BOUND)
    assert stats.knock_in.probability >= stats.ki_settlement.probability - TWIN_BOUND


@pytest.mark.parametrize("name", ["sb_dki", "kr_dki", "ph_nomem_instant_dki"])
def test_cell_average_twin_agrees_to_the_projection_resolution(name):
    """Cell-average events project a knock-in and a knock-out at the same instant in a different order in the twin
    (knock-out, then knock-in) than in the note (knock-in, then the knocked-in knock-out): the two differ by the
    straddling cells' mass only, which shrinks under refinement."""
    _, product, market = route(name)
    cls = type(route(name)[0])
    gaps = []
    for points in (401, 1601):
        stats = cls(QuadParams(grid_points=points)).calculate_event_stats(product, market)
        twin = deepcopy(product)
        twin.barrier_config = replace(twin.barrier_config, disable_ko_after_ki=True)
        reference = cls(QuadParams(grid_points=points)).calculate_event_stats(twin, market)
        gaps.append(abs(stats.knock_in.probability - reference.ki_settlement.probability))
    assert gaps[1] <= max(gaps[0], 1e-10) and gaps[1] <= 1e-4


# --- AC13 -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", IDS)
def test_knock_in_table_identities(name):
    engine, product, market = route(name)
    stats = engine.calculate_event_stats(product, market)
    table, ki = stats.ki_table, stats.knock_in
    assert float(table.probability.sum()) == pytest.approx(ki.probability, rel=1e-12, abs=1e-15)
    np.testing.assert_array_equal(table.payment_times, table.times)                       # D8
    assert np.all(table.probability >= -1e-12)
    mean = float(np.sum(table.times * table.probability) / ki.probability)
    horizon = _horizon(product, market)
    if _is_continuous(product):
        day = 1.0 / market.bus_days_in_year
        assert abs(mean - ki.expected_time) <= day
        assert table.interpolation_scheme == v1_statistics.FIRST_PASSAGE
        assert table.times[-1] == pytest.approx(horizon)
        assert table.dates is None
        np.testing.assert_allclose(np.diff(table.times[:-1]), day, rtol=1e-12)
    else:
        assert mean == pytest.approx(ki.expected_time, rel=1e-12)
        records = [r for r in product.resolve_ki_observations(market) if r.observation_time >= 0]
        np.testing.assert_allclose(table.times, [r.observation_time for r in records
                                                 if r.observation_time <= horizon + 1e-12])
        assert not table.interpolated.any() and table.interpolation_scheme is None


#: a settlement lag on every knock-out record, at a 10% rate: DF(settle)/DF(obs) = e^{-0.01} -- a factor ~1e-2 off
#: the knock-out probability if the survival read the discounted unit instead of the probability
LAG = 0.1


def _lagged(times, level, lag=LAG):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level,
                                                          settlement_time=float(t) + lag) for t in times])


def lagged_snowball(ki_barrier=1.0, ki="dki"):
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_lagged(OBS, 103.0), **ki_kw(ki, barrier=ki_barrier)),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )


def lagged_phoenix(ki_barrier=1.0):
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_lagged(OBS, 105.0), **ki_kw("dki", barrier=ki_barrier)),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0, 108.0, 80.0, 80.0], coupon_rate=0.02,
                                          coupon_pay_type=CouponPayType.INSTANT,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=True),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )


def lagged_ko_reset(ki_barrier=1.0):
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=2.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_lagged(OBS, 103.0), ki_barrier=ki_barrier,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule(OBS, ki_barrier)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_lagged((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                                                                           2.0), 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
    )


LAGGED = {
    "snowball": (sb(), lagged_snowball),
    "phoenix": (ph(), lagged_phoenix),
    "ko_reset": (kr(), lagged_ko_reset),
}


@pytest.mark.parametrize("family", list(LAGGED))
def test_lagged_settlement_survival_is_the_knock_out_survival_when_knock_in_is_unreachable(family):
    """The forward pass's knock-out source is the backward unit DF(settle)/DF(obs); the KI table's survival must
    subtract the knock-out PROBABILITY (CP6). With the knock-in out of reach, the KI table's survival on each row is
    the observation table's survival (both schedules of the KO-reset are lagged)."""
    engine, factory = LAGGED[family]
    stats = engine().calculate_event_stats(factory(), env(r=0.10, q=0.0))
    obs, table = stats.observations, stats.ki_table
    assert abs(stats.knock_in.probability) <= 1e-15                              # out of reach (round-off)
    assert float(np.sum(obs.probability)) > 0.1                                   # the KO mass is material
    np.testing.assert_allclose(obs.payment_times - obs.times, LAG, atol=1e-12)
    # P(no knock-out by the row): the observation table's survival on a snowball / Phoenix; the KO-reset's also ends
    # the first schedule's maturity at its last row, which is not a knock-out, so read the knock-out mass itself
    ko_by_row = np.array([np.sum(obs.probability[obs.times <= t + 1e-12]) for t in table.times])
    np.testing.assert_allclose(table.survival, 1.0 - ko_by_row, rtol=0.0, atol=1e-12)
    if family != "ko_reset":
        rows = [int(np.flatnonzero(np.abs(obs.times - t) <= 1e-12)[0]) for t in table.times]
        np.testing.assert_allclose(table.survival, obs.survival[rows], rtol=0.0, atol=1e-12)


def test_lagged_settlement_survival_matches_quad_v2_with_reachable_knock_in():
    """V1 and V2 KI-table survival agree on a lagged note whose knock-in is reachable, within the two engines'
    discretisation gap (measured 5.3e-4 on the KI masses, 6.3e-4 on survival at 401 V1 nodes); the CP6 defect put
    the gap at 7.1e-3 (survival over by the KO probability times 1 - e^{-r lag}, cumulated)."""
    from quantark.asset.equity.engine.quad.v2.engine import SnowballQuadEngineV2

    product, market = lagged_snowball(ki_barrier=85.0), env(r=0.10, q=0.0)
    v1 = sb()().calculate_event_stats(product, market)
    v2 = SnowballQuadEngineV2().calculate_event_stats(product, market)
    assert v1.knock_in.probability > 0.05
    np.testing.assert_allclose(v1.ki_table.times, v2.ki_table.times)
    np.testing.assert_allclose(v1.ki_table.probability, v2.ki_table.probability, atol=1e-3)
    np.testing.assert_allclose(v1.ki_table.survival, v2.ki_table.survival, atol=1e-3)


def test_continuous_rows_follow_the_first_passage_law_within_an_interval():
    """One year-long bridge interval spreads its crossing mass over the days by the Brownian first-passage law: the
    mass up to each day matches the closed form when no knock-out can intervene (to the lattice's resolution)."""
    from scipy.special import ndtr

    product = snowball("cki", ko_dates=(1.0,), ko=1e9)
    market = env()
    stats = SnowballQuadEngine(QuadParams(grid_points=2001)).calculate_event_stats(product, market)
    cumulative = np.cumsum(stats.ki_table.probability)
    m = (0.03 - 0.01 - 0.5 * 0.22 ** 2)

    def first_hit(t):
        d, v = math.log(100.0 / 75.0), 0.22 ** 2 * t
        return float(ndtr((-d - m * t) / math.sqrt(v)) + math.exp(-2 * m * t * d / v) * ndtr((m * t - d) / math.sqrt(v)))

    # the law starts from the readout's dual -- the interpolation weights on the two nodes around the spot -- so the
    # split matches the closed form to the lattice's resolution there (measured 2.5e-5 at 2001 nodes)
    total = first_hit(1.0)
    assert cumulative[-1] == pytest.approx(total, abs=1e-4)
    for k in (10, 63, 126, 200):
        assert cumulative[k] / cumulative[-1] == pytest.approx(first_hit(stats.ki_table.times[k]) / total, abs=1e-4)


# --- AC9 ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset", "snowball_cki"])
def test_no_knock_in_after_a_certain_knock_out(family):
    """Every path knocks out at 0.25 (the KO barrier below any reachable spot); the KI observations come after it and
    cross often -- none of it counts (the legacy KO-independent monitor counted it)."""
    if family == "snowball":
        product, engine = snowball("dki", ko=1e-3, ki_dates=(0.5, 0.75, 1.0)), sb()()
    elif family == "snowball_cki":
        # the barrier at 30 is out of reach before the certain knock-out at 0.05, reachable (~5%) afterwards
        product, engine = snowball("cki", ko=1e-3, ko_dates=(0.05, *OBS), ki_barrier=30.0), sb()()
    elif family == "phoenix":
        product = phoenix(False, CouponPayType.INSTANT, "dki", ko=1e-3, ki_dates=(0.5, 0.75, 1.0))
        engine = ph()()
    else:
        product, engine = ko_reset("dki", ko=1e-3, ki_dates=(0.5, 0.75, 1.0)), kr()()
    stats = engine.calculate_event_stats(product, env(vol=0.6))
    assert stats.observations.probability[0] == pytest.approx(1.0, abs=1e-6)
    assert stats.knock_in.probability == pytest.approx(0.0, abs=1e-6)
    assert stats.knock_in.expected_loss == pytest.approx(0.0, abs=1e-6)


# --- AC10 -----------------------------------------------------------------------------------------------------------


def _spy(monkeypatch):
    calls = []
    original = v1_statistics.forward_pass

    def spy(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    monkeypatch.setattr(v1_statistics, "forward_pass", spy)
    return calls


@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
def test_knocked_out_at_valuation(family, monkeypatch):
    calls = _spy(monkeypatch)
    if family == "snowball":
        product, engine = snowball("cki", ko_dates=(0.0, *OBS)), sb()()
    elif family == "phoenix":
        product, engine = phoenix(True, CouponPayType.INSTANT, ko_dates=(0.0, *OBS), barriers=[80.0] * 5), ph()()
        product.coupon_config = replace(product.coupon_config, fixed_coupon_year_fraction=0.25)
        product.validate()
    else:
        product, engine = ko_reset("dki", ko=90.0), kr()()
        product.barrier_config = replace(product.barrier_config,
                                         ko_observation_schedule=_schedule((0.0, *OBS), 90.0))
        product.validate()
    stats = engine.calculate_event_stats(product, env(spot=110.0))
    assert stats.observations.probability[0] == 1.0 and stats.observations.times[0] == 0.0
    assert stats.maturity.probability == 0.0 and stats.ki_settlement.probability == 0.0
    assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0
    assert stats.knock_in.expected_loss == 0.0 and stats.expected_lifetime == 0.0
    assert stats.unavailable == {"knock_in.expected_time": stats.unavailable["knock_in.expected_time"]}
    assert stats.pv == engine.price(product, env(spot=110.0))
    assert abs(stats.reconciliation_error) <= 1e-12 and stats.probability_residual == 0.0
    if family == "phoenix":
        # the coupon decided with the knock-out is coupon cash, its own stream (cause h: base emitted none)
        assert stats.coupons.probability[0] == 1.0 and stats.coupons.discounted_cash[0] > 0.0
        result = engine.price_with_events(product, env(spot=110.0))
        assert EventType.COUPON in result.event_distribution.stream_times
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
    stats = sb()().calculate_event_stats(product, env(spot=70.0))
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    assert stats.maturity.probability == pytest.approx(0.0, abs=1e-12)
    assert not calls


def test_no_knock_in_barrier(monkeypatch):
    calls = _spy(monkeypatch)
    stats = sb()().calculate_event_stats(snowball("none"), env())
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


# --- refusals (never approximated) ----------------------------------------------------------------------------------


def test_bgk_approximation_refuses_the_knock_in_fields():
    product = snowball("dki", ki_dates=[k / 252 for k in range(1, 253)])
    engine = SnowballQuadEngine(QuadParams(grid_points=401, ki_monitoring_mode="bgk_approximation"))
    stats = engine.calculate_event_stats(product, env())
    for path in ("knock_in.probability", "knock_in.expected_time", "ki_table"):
        assert stats.unavailable[path] == v1_statistics.BGK_REASON
    assert stats.pv == SnowballQuadEngine(engine.params).price(product, env())
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND


# --- layouts --------------------------------------------------------------------------------------------------------


def test_ko_reset_layout():
    stats = kr()().calculate_event_stats(ko_reset("dki"), env())
    obs = stats.observations
    assert obs.schedule == ("pre", "post") * 4 + ("post",) * 4
    np.testing.assert_allclose(obs.times, [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.25, 1.5, 1.75, 2.0])
    assert stats.maturity.time == 1.0 and stats.ki_settlement.time == 2.0
    cumulative = 1.0 - np.cumsum(obs.probability)
    np.testing.assert_allclose(obs.survival, cumulative - np.where(obs.times >= 1.0, stats.maturity.probability, 0.0),
                               atol=1e-15)
    assert stats.coupons is None


@pytest.mark.parametrize("memory", [False, True])
def test_expiry_coupon_cash_to_pay_is_the_coupon_earned_at_the_row(memory):
    """Zero-ish volatility, flat 10% carry: row 1 meets its barrier (3, or 6 with the missed row 0), row 2 knocks out
    with its coupon (3). Paid at the knock-out settlement, so the row's cash to pay is the coupon earned there,
    computed directly (memory: its own undiscounted columns); the ledger buckets hold it at the knock-out."""
    from test_asset.equity.lifecycle.test_quad_v2_coupon_event_stats import environment, phoenix as fixed_phoenix

    p = fixed_phoenix(times=(0.2, 0.4, 0.6, 0.8), ko=[200.0, 200.0, 90.0, 200.0], coupon=[110.0, 90.0, 90.0, 90.0],
                      memory=memory, pay_type=CouponPayType.EXPIRY)
    stats = PhoenixQuadEngine(QuadParams(grid_points=1001)).calculate_event_stats(
        p, environment(vol=0.01, r=0.1, q=0.1))
    first = 6.0 if memory else 3.0
    np.testing.assert_allclose(stats.coupons.cash_to_pay, [0.0, first, 3.0, 0.0], atol=1e-6)
    assert stats.coupons.payment_is_path_dependent.all()
    np.testing.assert_allclose(stats.coupons.discounted_cash, np.array([0.0, first, 3.0, 0.0]) * np.exp(-0.06),
                               atol=1e-6)
    # the ledger books every coupon at the knock-out settlement (row 2), none at maturity
    ledger = stats.ledger
    assert float(np.sum(ledger.discounted_cash)) == pytest.approx(stats.total_discounted_cash, rel=1e-12)


# --- forward_density mode (opt-in; provenance and its own numbers kept) ---------------------------------------------


def test_forward_density_keeps_its_provenance_and_reports_no_knock_in_path():
    product = snowball("cki")
    stacked = sb()().calculate_event_stats(product, env())
    forward = sb(event_stats_mode="forward_density")().calculate_event_stats(product, env())
    assert forward.provenance.statistics_model == "quad_v1_forward_density"
    assert forward.pv == stacked.pv
    for path in ("knock_in.probability", "knock_in.expected_time", "ki_table"):
        assert forward.unavailable[path] == v1_statistics.FORWARD_DENSITY_KI_REASON
    np.testing.assert_allclose(forward.observations.probability, stacked.observations.probability, atol=6e-4)
    assert forward.ki_settlement.probability == pytest.approx(stacked.ki_settlement.probability, abs=1e-2)


# --- reentrancy ------------------------------------------------------------------------------------------------------


def test_coupon_cash_rows_are_restored_after_a_nested_call():
    """The per-call coupon-cash layout is restored (not cleared) when a nested statistics call returns."""
    engine = ph()()
    sentinel = object()
    engine._coupon_cash_rows = sentinel
    engine.calculate_event_stats(phoenix(True, CouponPayType.INSTANT), env())
    assert engine._coupon_cash_rows is sentinel


def test_mc_stats_recorder_is_restored_after_a_nested_call():
    """The plan's reentrancy minor, MC side (T3's recorder): a statistics call restores the caller's recorder rather
    than clearing it."""
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.param import MCParams

    engine = SnowballMCEngine(params=MCParams(num_paths=256, seed=3))
    sentinel = object()
    engine._stats_recorder = sentinel
    with engine._recording_event_stats():
        assert engine._stats_recorder is not sentinel
    assert engine._stats_recorder is sentinel


def test_pruned_phoenix_knocked_out_at_valuation_names_its_pruned_coupon_probability():
    product = phoenix(True, CouponPayType.INSTANT, ko_dates=(0.0, *OBS), barriers=[80.0] * 5)
    stats = ph()().calculate_event_stats(product, env(spot=110.0), streams=frozenset({EventType.KO}))
    assert stats.coupons.probability is None
    assert stats.unavailable["coupons.probability"] == "not requested"

"""MC DCN event statistics on the unified ``EventStatistics`` (spec T11).

Over the DCN part of the T3 route grid (``mc_price_path_routes``: Sobol single / batched / threaded, pseudo,
antithetic, seller, local vol, Heston, QE) and constructed cases (a ``ko_lock_months > lock_months`` contract, a
non-SSE run day, ACT/360 and BUSINESS_DAYS environments, valuation-date rows, lifecycle states):

- AC7: ``EventStatistics`` with a coupon table and nothing unavailable but the mathematically undefined values;
- AC8: both residuals at MC round-off (every outcome summed directly over the price's own paths);
- AC10: valuation-date rules (knock-out at valuation, carried knock-in, knock-in at the valuation close, the final
  observation decided at the valuation);
- AC17: receivables are ledger rows, never events; a terminated note with unpaid cash reports its receivables only;
  a terminated, settled one has no statistics;
- AC18 (0.14.0; D12 retired): the environment's axis with exact dates on every environment, seller cash negative
  with the identities intact, the event route and the fold route on the same axis;
- AC21: standard errors from independent batch means;
- price(), ``price_detailed(...).event_stats.pv`` and ``price_with_events(...).npv`` bitwise equal.

The references re-run the engine's own price with a spy on ``_simulate`` (the price's paths) and decide every outcome
here, path by path (``mc_same_path_reference.dcn_paths``).
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.dcn_valuation import dcn_fold_slots
from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine
from quantark.asset.equity.lifecycle import DCNLifecycleState, ValuationPoint
from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, RealizedCashflow
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.cashleg import (
    DeterministicLeg,
    EventDistribution,
    EventType,
    FixedPayoffLeg,
    LegDirection,
    PaymentTrigger,
    fold_cash_legs,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.priceenv import CurveQuotingBasis
from quantark.priceenv.date_axis import env_time, env_times
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.day_counter import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, DCN_B, FLAT, make_dcn, calendar_env
from test_asset.equity.engine_mc import mc_same_path_reference as reference
from test_asset.equity.engine_mc.mc_price_path_routes import ROUTES

warnings.filterwarnings("ignore", category=DeprecationWarning)

SSE = create_calendar(CalendarType.CHINA_SSE)
START = DCN_A["valuation_date"]
SAT = datetime(2024, 6, 8)           # SSE closed: an unmonitored valuation instant (a non-SSE run day)
HOLIDAY = datetime(2024, 6, 10)      # Dragon Boat Festival: SSE closed on a Monday
DAY = datetime(2024, 5, 15)          # a trading day that is not a monthly row
OBS = datetime(2024, 12, 3)          # DCN_A month 23: a coupon + knock-out row on the valuation date
FINAL = datetime(2025, 1, 3)         # DCN_A's final observation
ROW, ROW_PAY, AGED, SETTLED = datetime(2024, 6, 3), datetime(2024, 6, 5), datetime(2024, 6, 4), datetime(2024, 6, 6)
CASH = 10_000.0
HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
N = 2048
#: round-off bound of the MC partition and cash identities (spec AC8, D16: direct path sums)
ROUND_OFF = 1e-12


def env_on(d, spot=6000.0, *, day_count=DayCountConvention.ACT_365, calendar=None, grid=False):
    if grid:
        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        from quantark.param.vol.vol_surface import GridVolSurface

        base = FLAT["sigma"] + 0.05 * (6000.0 - np.array(strikes)) / 6000.0
        vol = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    else:
        vol = FlatVolSurface(volatility=FLAT["sigma"])
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=FLAT["r"]),
        div_yield=ContinuousDividendYield(div_yield=FLAT["q"]), valuation_date=d, day_count_convention=day_count,
        calendar=calendar,
        **({"rate_curve_quoting": CurveQuotingBasis.TRADING_YEAR, "div_yield_quoting": CurveQuotingBasis.TRADING_YEAR}
           if day_count == DayCountConvention.BUSINESS_DAYS else {}),
    )


def flow(kind, determined, paid, amount=CASH):
    event = {"coupon": LifecycleEventType.COUPON, "ko": LifecycleEventType.KNOCK_OUT,
             "loss": LifecycleEventType.MATURITY}[kind]
    return RealizedCashflow(cashflow_id=f"dcn:{kind}:{determined:%Y-%m-%d}", event_type=event, amount=amount,
                            determination_date=determined, payment_date=paid)


def state_on(d, *flows, **flags) -> DCNLifecycleState:
    return DCNLifecycleState(valuation_point=ValuationPoint(date=d), ledger=LifecycleCashflowLedger(flows), **flags)


def _gbm(**kw):
    return lambda: DCNMCEngine(**{"num_paths": N, "seed": 7, **kw})


#: name -> (engine factory, product factory, env factory, lifecycle-state factory or None)
CASES = {
    **{name: (route.engine, route.product, route.env, None) for name, route in ROUTES.items()
       if route.family == "dcn"},
    "ko_lock_after_lock": (_gbm(use_sobol=False, num_batches=2), lambda: make_dcn(DCN_B), lambda: env_on(START),
                           None),
    "act_360": (_gbm(use_sobol=False, num_batches=2), lambda: make_dcn(DCN_A),
                lambda: env_on(START, day_count=DayCountConvention.ACT_360), None),
    "business_days": (_gbm(num_batches=2), lambda: make_dcn(DCN_A),
                      lambda: env_on(START, day_count=DayCountConvention.BUSINESS_DAYS, calendar=SSE), None),
    "closed_saturday": (_gbm(), lambda: make_dcn(DCN_A, valuation_date=SAT), lambda: env_on(SAT, 5500.0), None),
    "closed_holiday": (_gbm(use_sobol=False, num_batches=2), lambda: make_dcn(DCN_A, valuation_date=HOLIDAY),
                       lambda: env_on(HOLIDAY, 5500.0), None),
    "ko_at_valuation": (_gbm(), lambda: make_dcn(DCN_A, valuation_date=OBS), lambda: env_on(OBS, 6300.0), None),
    "ki_at_valuation": (_gbm(), lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY, 4400.0), None),
    "carried": (_gbm(use_sobol=False, num_batches=2),
                lambda: make_dcn(DCN_A, valuation_date=DAY, knocked_in_at_valuation=True),
                lambda: env_on(DAY, 5000.0), None),
    "carried_ko_at_valuation": (_gbm(), lambda: make_dcn(DCN_A, valuation_date=OBS, knocked_in_at_valuation=True),
                                lambda: env_on(OBS, 6300.0), None),
    "final_survives": (_gbm(), lambda: make_dcn(DCN_A, valuation_date=FINAL), lambda: env_on(FINAL, 5500.0), None),
    "final_knocks_out": (_gbm(), lambda: make_dcn(DCN_A, valuation_date=FINAL), lambda: env_on(FINAL, 6300.0),
                         None),
    "final_knocked_in": (_gbm(), lambda: make_dcn(DCN_A, valuation_date=FINAL), lambda: env_on(FINAL, 4000.0),
                         None),
    "receivable_alive": (_gbm(use_sobol=False, num_batches=2), lambda: make_dcn(DCN_A, valuation_date=AGED),
                         lambda: env_on(AGED), lambda: state_on(AGED, flow("coupon", ROW, ROW_PAY))),
    "receivable_alive_seller": (
        _gbm(), lambda: make_dcn(DCN_A, valuation_date=AGED, direction=DCNDirection.SELLER),
        lambda: env_on(AGED), lambda: state_on(AGED, flow("coupon", ROW, ROW_PAY, -CASH))),
    "lv_closed_holiday": (lambda: LocalVolDCNMCEngine(num_paths=N, seed=7),
                          lambda: make_dcn(DCN_A, valuation_date=HOLIDAY),
                          lambda: env_on(HOLIDAY, 5500.0, grid=True), None),
    "heston_carried": (lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=N, seed=7, num_batches=2,
                                                 use_sobol=False),
                       lambda: make_dcn(DCN_A, valuation_date=DAY, knocked_in_at_valuation=True),
                       lambda: env_on(DAY, 5000.0), None),
}
TERMINATED = {
    "ko_unpaid": (lambda: make_dcn(DCN_A, valuation_date=AGED), lambda: env_on(AGED),
                  lambda: state_on(AGED, flow("ko", ROW, ROW_PAY), knocked_out=True, ko_date=ROW)),
    "ko_unpaid_seller": (lambda: make_dcn(DCN_A, valuation_date=AGED, direction=DCNDirection.SELLER),
                         lambda: env_on(AGED),
                         lambda: state_on(AGED, flow("ko", ROW, ROW_PAY, -CASH), knocked_out=True, ko_date=ROW)),
    "matured_loss_unpaid": (
        lambda: make_dcn(DCN_A, valuation_date=FINAL, knocked_in_at_valuation=True),
        lambda: env_on(FINAL, 4000.0),
        lambda: state_on(FINAL, flow("loss", FINAL, datetime(2025, 1, 7), -55_000.0),
                         knocked_in=True, matured=True)),
}
SETTLED_STATE = (lambda: make_dcn(DCN_A, valuation_date=SETTLED), lambda: env_on(SETTLED),
                 lambda: state_on(SETTLED, flow("ko", ROW, ROW_PAY), knocked_out=True, ko_date=ROW))

#: cases simulated in one Python-loop reference (the reference re-decides every path)
SAME_PATH = ["dcn_sobol", "dcn_sobol_batches", "dcn_pseudo", "dcn_seller", "lv_dcn", "heston_dcn",
             "qe_dcn_batches", "ko_lock_after_lock", "act_360", "closed_saturday", "ki_at_valuation", "carried",
             "receivable_alive", "lv_closed_holiday", "heston_carried"]
DECIDED_NOW = ["final_survives", "final_knocks_out", "final_knocked_in"]


def _kwargs(name):
    state = CASES[name][3]
    return {} if state is None else {"lifecycle_state": state()}


def _fresh(name):
    make, product, env, _ = CASES[name]
    return make(), product(), env()


_STATS: dict = {}


def _stats(name) -> EventStatistics:
    if name not in _STATS:
        engine, product, env = _fresh(name)
        _STATS[name] = engine.calculate_event_stats(product, env, **_kwargs(name))
    return _STATS[name]


def _price(name) -> float:
    engine, product, env = _fresh(name)
    return engine.price(product, env, **_kwargs(name))


def _scale(stats) -> float:
    cash = [stats.observations.discounted_cash, stats.coupons.discounted_cash, stats.receivables.discounted_cash,
            [stats.maturity.discounted_cash, stats.ki_settlement.discounted_cash]]
    return max(1.0, float(sum(np.sum(np.abs(c)) for c in cash)))


# --- AC7 / AC8: the type, nothing unavailable but the undefined, round-off residuals -------------------------------

@pytest.mark.parametrize("name", sorted(CASES))
def test_full_request_is_an_event_statistics_with_only_undefined_values_unavailable(name):
    stats = _stats(name)
    assert isinstance(stats, EventStatistics) and stats.product_family == "dcn"
    assert stats.coupons is not None and stats.coupons.size == stats.observations.size
    assert stats.ki_table is not None and stats.termination is None
    for key, reason in stats.unavailable.items():
        assert key in ("knock_in.expected_time", "standard_errors", "standard_errors.knock_in.expected_time"), key
        assert reason.startswith("undefined:"), (key, reason)
    assert stats.provenance.statistics_model == stats.provenance.price_model == type(_fresh(name)[0]).__name__


@pytest.mark.parametrize("name", sorted(CASES))
def test_partition_and_reconciliation_are_round_off(name):
    stats = _stats(name)
    assert abs(stats.probability_residual) <= ROUND_OFF
    assert abs(stats.reconciliation_error) <= ROUND_OFF * _scale(stats)
    assert stats.ledger.total_discounted_cash == pytest.approx(stats.total_discounted_cash, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("name", sorted(CASES))
def test_pv_is_the_price_bitwise_on_every_entry_point(name):
    price = _price(name)
    engine, product, env = _fresh(name)
    assert engine.calculate_event_stats(product, env, **_kwargs(name)).pv == price
    engine, product, env = _fresh(name)
    detailed = engine.price_detailed(product, env, **_kwargs(name))
    assert detailed.pv == price and detailed.event_stats.pv == price
    engine, product, env = _fresh(name)
    assert engine.price_with_events(product, env, **_kwargs(name)).npv == price


def test_a_plain_price_records_nothing():
    engine, product, env = _fresh("dcn_sobol_batches")
    engine.price(product, env)
    assert engine._last_stats_records == () and engine._last_result.event_stats is None
    engine.price_detailed(product, env)
    assert len(engine._last_stats_records) == 4


# --- the price's own paths ------------------------------------------------------------------------------------------

def _reference(name):
    engine, product, env = _fresh(name)
    return reference.dcn_paths(engine, product, env, **_kwargs(name)), product, env


@pytest.mark.parametrize("name", SAME_PATH)
def test_every_outcome_is_the_same_path_count(name):
    ref, product, env = _reference(name)
    stats = _stats(name)
    n, ctx = ref["n"], ref["ctx"]
    n_obs = ctx.obs_cols.size
    ko = np.bincount(ref["ko_row"][ref["ko_row"] >= 0], minlength=n_obs) / n
    np.testing.assert_array_equal(stats.observations.probability, ko)
    np.testing.assert_array_equal(stats.observations.survival, 1.0 - np.cumsum(ko))
    np.testing.assert_array_equal(stats.coupons.probability, ref["coupon_paid"].sum(axis=0) / n)
    assert stats.maturity.probability == np.count_nonzero(ref["maturity"]) / n
    assert stats.ki_settlement.probability == np.count_nonzero(ref["ki_settlement"]) / n
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability
    knocked_in = ref["ki_col"] >= 0
    assert stats.knock_in.probability == np.count_nonzero(knocked_in) / n
    if np.any(knocked_in):
        assert stats.knock_in.expected_time == pytest.approx(
            float(np.mean(ctx.times[ref["ki_col"][knocked_in]])), rel=1e-12, abs=1e-15)
    df = make_df_fn(env)
    loss_df = df(ctx.loss_pay_time)
    assert stats.knock_in.expected_loss == pytest.approx(-float(np.sum(ref["loss"])) * loss_df / n, rel=1e-12,
                                                         abs=1e-9)
    sign = product.direction_sign
    assert stats.ki_settlement.discounted_cash == pytest.approx(sign * float(np.sum(ref["loss"])) * loss_df / n,
                                                                rel=1e-12, abs=1e-9)
    ko_amount = product.participation * product.ko_coupon_rate * ctx.ko_accruals * product.notional
    coupon_amount = product.participation * product.coupon_rate * ctx.coupon_accruals * product.notional
    np.testing.assert_allclose(stats.observations.cash_to_pay, sign * ko_amount * ko, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(stats.coupons.cash_to_pay, sign * coupon_amount * stats.coupons.probability,
                               rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("name", SAME_PATH)
def test_knock_in_table_is_the_first_knock_in_per_daily_date(name):
    ref, product, env = _reference(name)
    stats = _stats(name)
    n, ctx, first = ref["n"], ref["ctx"], ref["first_col"]
    table = stats.ki_table
    if ref["carried"] or (name == "ki_at_valuation"):
        # a knock-in decided at the valuation (carried, or the valuation close through the barrier): one row at 0
        np.testing.assert_array_equal(table.times, [0.0])
        np.testing.assert_array_equal(table.probability, [1.0])
        assert table.dates == (product.schedule.valuation_date,)
        assert stats.knock_in.expected_time == 0.0
        return
    np.testing.assert_array_equal(table.times, ctx.times[first:])
    assert table.dates == tuple(product.schedule.daily_ki_dates)
    np.testing.assert_array_equal(table.payment_times, table.times)            # D8
    counts = np.bincount(ref["ki_col"][ref["ki_col"] >= 0], minlength=ctx.times.size)[first:]
    np.testing.assert_array_equal(table.probability, counts / n)
    assert not np.any(table.interpolated) and table.interpolation_scheme is None
    # AC13 identities (discrete: exact)
    assert float(np.sum(table.probability)) == pytest.approx(stats.knock_in.probability, abs=1e-14)
    if stats.knock_in.probability > 0.0:
        assert float(np.sum(table.times * table.probability)) / stats.knock_in.probability == pytest.approx(
            stats.knock_in.expected_time, rel=1e-12)
    # survival: not knocked in and not knocked out after the row
    for k in (0, table.size // 2, table.size - 1):
        col = first + k
        out = (ref["ki_col"] >= 0) & (ref["ki_col"] <= col)
        ko_clean = (ref["ko_row"] >= 0) & (ref["ki_col"] < 0)
        out |= ko_clean & (ctx.obs_cols[np.maximum(ref["ko_row"], 0)] <= col)
        assert table.survival[k] == pytest.approx(1.0 - np.count_nonzero(out) / n, abs=1e-14)


def test_a_closed_valuation_day_has_no_knock_in_row_at_zero():
    """A non-SSE run day (A7'): the valuation instant is no schedule date; the first KI row is the next close."""
    stats = _stats("closed_saturday")
    product = CASES["closed_saturday"][1]()
    ctx = build_dcn_grid_context(product, calendar_env(product))
    assert not ctx.valuation_monitored
    np.testing.assert_array_equal(stats.ki_table.times, ctx.times[1:])
    assert stats.ki_table.times[0] == (datetime(2024, 6, 11) - SAT).days / 365.0


# --- AC18 (0.14.0: D12 retired): the environment's axis, exact dates ----------------------------------------------

@pytest.mark.parametrize("name", ["dcn_sobol", "act_360", "business_days", "closed_saturday", "ko_lock_after_lock",
                                  "receivable_alive"])
def test_times_are_the_dcn_axis_with_exact_dates(name):
    stats = _stats(name)
    engine, product, env = _fresh(name)
    ctx = build_dcn_grid_context(product, env)
    pay = product.payment_dates()
    obs = stats.observations
    np.testing.assert_array_equal(obs.times, ctx.times[ctx.obs_cols])
    np.testing.assert_array_equal(obs.payment_times, ctx.ko_pay_times)
    np.testing.assert_array_equal(stats.coupons.payment_times, ctx.coupon_pay_times)
    assert obs.dates == tuple(m.observation_date for m in product.schedule.monthly)
    assert obs.payment_dates == tuple(pay.ko_payment_dates)
    assert stats.coupons.payment_dates == tuple(pay.coupon_payment_dates)
    np.testing.assert_array_equal(obs.is_ko_obs, ctx.obs_is_ko)
    np.testing.assert_array_equal(obs.is_coupon_obs, ctx.obs_is_coupon)
    for row in (stats.maturity, stats.ki_settlement):
        assert (row.time, row.payment_time) == (float(ctx.times[ctx.obs_cols[-1]]), float(ctx.loss_pay_time))
        assert (row.date, row.payment_date) == (product.schedule.monthly[-1].observation_date, pay.loss_payment_date)
    # every time is on the environment's axis from the schedule's valuation date (0.14.0 retired D12's ACT/365F)
    v = product.schedule.valuation_date
    np.testing.assert_array_equal(obs.times, env_times(env, obs.dates, anchor=v))
    np.testing.assert_array_equal(obs.payment_times, env_times(env, obs.payment_dates, anchor=v))
    assert stats.ki_settlement.payment_time == env_time(env, pay.loss_payment_date, anchor=v)
    # AC12: deterministic payment rows -- discounted = cash x the curve read at the row's payment time (D12)
    df = make_df_fn(env)
    np.testing.assert_allclose(obs.discounted_cash, obs.cash_to_pay * np.array([df(t) for t in obs.payment_times]),
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(stats.coupons.discounted_cash,
                               stats.coupons.cash_to_pay * np.array([df(t) for t in stats.coupons.payment_times]),
                               rtol=1e-12, atol=1e-9)
    assert stats.ki_settlement.discounted_cash == pytest.approx(
        stats.ki_settlement.cash_to_pay * df(stats.ki_settlement.payment_time), rel=1e-12, abs=1e-9)
    # expected lifetime on the same axis: the KO row if knocked out, else the final observation
    assert stats.expected_lifetime == pytest.approx(
        float(np.sum(obs.probability * obs.times))
        + (stats.maturity.probability + stats.ki_settlement.probability) * stats.maturity.time, rel=1e-12)


@pytest.mark.parametrize("name", ["act_360", "business_days"])
def test_the_environment_axis_differs_there(name):
    """The two environments really are on another axis (so the test above pins the DCN axis, not a coincidence)."""
    _, product, env = _fresh(name)
    d = product.schedule.monthly[5].observation_date
    t_env = calculate_year_fraction(env.valuation_date, d, env.day_count_convention, env.bus_days_in_year,
                                    calendar=env.calendar)
    assert abs(t_env - (d - env.valuation_date).days / 365.0) > 1e-3


def test_coupon_only_rows_carry_no_knock_out():
    stats = _stats("ko_lock_after_lock")
    obs = stats.observations
    lock = ~obs.is_ko_obs
    assert np.any(lock) and np.all(obs.is_coupon_obs)
    assert np.all(obs.probability[lock] == 0.0) and np.all(obs.discounted_cash[lock] == 0.0)
    assert np.any(stats.coupons.probability[lock] > 0.0)


@pytest.mark.parametrize("name", ["dcn_sobol", "act_360", "business_days", "closed_saturday"])
def test_fold_slots_are_the_statistics_axis(name):
    stats = _stats(name)
    _, product, env = _fresh(name)
    slots = dcn_fold_slots(build_dcn_grid_context(product, env))
    assert [s[1] for s in slots.ko] == stats.observations.times.tolist()
    assert [s[2] for s in slots.ko] == stats.observations.payment_times.tolist()
    assert (slots.determination, slots.payment) == (stats.maturity.time, stats.maturity.payment_time)


def _legs(stats):
    return [
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=7_000.0, trigger=PaymentTrigger.AT_KO,
                       payment_offset_days=0),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=3_000.0, trigger=PaymentTrigger.AT_MATURITY_ANY,
                       payment_offset_days=0),
        FixedPayoffLeg(direction=LegDirection.BUYER_PAYS, amount=4_000.0,
                       trigger=PaymentTrigger.AT_MATURITY_WITH_KI, payment_offset_days=0),
    ]


@pytest.mark.parametrize("name", ["dcn_sobol", "act_360", "business_days", "closed_saturday", "dcn_seller"])
@pytest.mark.parametrize("quantity", [1.0, -2.0])
def test_event_route_and_fold_route_agree(name, quantity):
    """AC18: the event route reads the statistics' axis, the fold route binds the same rows (``bind_dcn_fold``):
    on the same paths a folded leg equals its event-route value to round-off, on every environment."""
    engine, product, env = _fresh(name)
    result = engine.price_with_events(product, env)
    dist = result.event_distribution
    legs = _legs(None)
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    folded = quantity * _fresh(name)[0].price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * result.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    assert abs(folded - expected) <= 1e-9 * product.notional * abs(quantity)


# --- AC18: direction sign --------------------------------------------------------------------------------------------

def test_seller_reports_negative_cash_with_identities_intact():
    buyer, seller = _stats("dcn_sobol"), _stats("dcn_seller")
    assert seller.pv == -buyer.pv and seller.pv == _price("dcn_seller")
    for field in ("observations", "coupons"):
        b, s = getattr(buyer, field), getattr(seller, field)
        np.testing.assert_array_equal(s.probability, b.probability)
        np.testing.assert_array_equal(s.discounted_cash, -b.discounted_cash)
        np.testing.assert_array_equal(s.cash_to_pay, -b.cash_to_pay)
    for row in ("maturity", "ki_settlement"):
        assert getattr(seller, row).probability == getattr(buyer, row).probability
        assert getattr(seller, row).discounted_cash == -getattr(buyer, row).discounted_cash
    assert seller.ki_settlement.discounted_cash > 0.0 > buyer.ki_settlement.discounted_cash
    assert np.all(seller.coupons.discounted_cash <= 0.0) and np.any(seller.coupons.discounted_cash < 0.0)
    assert seller.knock_in == buyer.knock_in                       # unsigned path statistics
    assert seller.knock_in.expected_loss > 0.0
    np.testing.assert_array_equal(seller.ki_table.probability, buyer.ki_table.probability)
    assert seller.expected_lifetime == buyer.expected_lifetime
    assert abs(seller.reconciliation_error) <= ROUND_OFF * _scale(seller)
    assert seller.probability_residual == buyer.probability_residual


# --- AC10: valuation-date rules ---------------------------------------------------------------------------------------

def test_knock_out_at_valuation():
    stats = _stats("ko_at_valuation")
    assert stats.observations.times[0] == 0.0 and stats.observations.probability[0] == 1.0
    assert np.all(stats.observations.probability[1:] == 0.0)
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_table.size == 0
    assert stats.knock_in.expected_time is None and stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert stats.coupons.probability[0] == 0.0                     # no fixed coupon on a knock-out
    assert stats.expected_lifetime == 0.0


def test_carried_knock_in_knocked_out_at_valuation_keeps_its_knock_in():
    stats = _stats("carried_ko_at_valuation")
    assert stats.observations.probability[0] == 1.0
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])


@pytest.mark.parametrize("name", ["carried", "heston_carried", "ki_at_valuation"])
def test_a_knock_in_decided_at_valuation_is_one_row_at_zero(name):
    stats = _stats(name)
    _, product, _ = _fresh(name)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    assert stats.ki_table.dates == (product.schedule.valuation_date,)
    assert np.all(stats.coupons.probability == 0.0)               # never a fixed coupon once knocked in


@pytest.mark.parametrize("name, outcome", [("final_survives", "maturity"), ("final_knocks_out", "ko"),
                                           ("final_knocked_in", "ki_settlement")])
def test_the_final_observation_decided_at_valuation(name, outcome):
    stats = _stats(name)
    assert stats.observations.size == 1 and stats.observations.times[0] == 0.0
    probabilities = {"ko": float(stats.observations.probability[0]), "maturity": stats.maturity.probability,
                     "ki_settlement": stats.ki_settlement.probability}
    assert probabilities == {k: (1.0 if k == outcome else 0.0) for k in probabilities}
    assert stats.unavailable["standard_errors"].startswith("undefined: nothing simulated")
    assert stats.standard_errors is None
    assert stats.knock_in.probability == (1.0 if outcome == "ki_settlement" else 0.0)
    if outcome == "ki_settlement":
        _, product, env = _fresh(name)
        loss = (product.notional / product.initial_price) * product.participation * max(product.k_loss - 4000.0, 0)
        df = make_df_fn(env)(stats.ki_settlement.payment_time)
        assert stats.knock_in.expected_loss == pytest.approx(loss * df, rel=1e-12)
        assert stats.ki_settlement.discounted_cash == pytest.approx(-loss * df, rel=1e-12)


# --- AC17: receivables and terminated states ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["receivable_alive", "receivable_alive_seller"])
def test_receivables_are_ledger_rows_not_events(name):
    stats = _stats(name)
    engine, product, env = _fresh(name)
    amount = CASH * product.direction_sign
    rec = stats.receivables
    assert rec.size == 1 and rec.source == ("coupon",)
    assert rec.decision_dates == (ROW,) and rec.payment_dates == (ROW_PAY,)
    assert rec.decision_times[0] == (ROW - AGED).days / 365.0 < 0.0
    t_pay = calculate_year_fraction(AGED, ROW_PAY, env.day_count_convention, env.bus_days_in_year)
    assert rec.payment_times[0] == pytest.approx(t_pay, abs=0.0)
    assert rec.cash_to_pay[0] == amount
    assert rec.discounted_cash[0] == amount * env.get_discount_factor(rec.payment_times[0])
    # inside the cash identity: pv = outcomes + receivables; the ledger carries the row
    assert stats.pv == _price(name)
    assert abs(stats.reconciliation_error) <= ROUND_OFF * _scale(stats)
    assert ROW in stats.ledger.determination_dates
    # never events: the distribution is the one of the same paths without the receivable
    with_state = engine.price_with_events(product, env, **_kwargs(name))
    without = _fresh(name)[0].price_with_events(product, env)
    assert with_state.npv - without.npv == pytest.approx(rec.discounted_cash[0], rel=1e-12)
    a, b = with_state.event_distribution, without.event_distribution
    assert set(a.probabilities) == set(b.probabilities)
    for event_type in a.probabilities:
        np.testing.assert_array_equal(np.asarray(a.probabilities[event_type]), np.asarray(b.probabilities[event_type]))
        np.testing.assert_array_equal(np.asarray(a.stream_times.get(event_type, [])),
                                      np.asarray(b.stream_times.get(event_type, [])))


@pytest.mark.parametrize("name", sorted(TERMINATED))
def test_a_terminated_note_with_unpaid_cash_reports_its_receivables_only(name):
    product_f, env_f, state_f = TERMINATED[name]
    product, env = product_f(), env_f()
    engine = DCNMCEngine(num_paths=N, seed=7)
    price = engine.price(product, env, lifecycle_state=state_f())
    stats = DCNMCEngine(num_paths=N, seed=7).calculate_event_stats(product, env, lifecycle_state=state_f())
    assert stats.terminated and stats.pv == price
    assert stats.observations.size == 0 and stats.coupons.size == 0
    assert stats.maturity.probability == stats.ki_settlement.probability == 0.0
    assert stats.probability_residual == 0.0
    flows = state_f().ledger.cashflows
    assert stats.receivables.size == len(flows)
    assert stats.receivables.cash_to_pay.tolist() == [f.amount for f in flows]
    assert abs(stats.reconciliation_error) <= ROUND_OFF * max(1.0, abs(price))
    assert set(stats.unavailable) <= {"standard_errors", "knock_in.expected_time"}
    assert stats.unavailable["standard_errors"].startswith("undefined: nothing simulated")
    carried = bool(product.knocked_in_at_valuation)
    assert stats.knock_in.probability == (1.0 if carried else 0.0)
    assert stats.ki_table.size == (1 if carried else 0)
    if name == "matured_loss_unpaid":
        assert stats.receivables.source == ("dcn_loss",)
    # T6 maps it to the ended distribution: event legs refuse it, a deterministic leg still values
    result = DCNMCEngine(num_paths=N, seed=7).price_with_events(product, env, lifecycle_state=state_f())
    assert result.npv == price and result.event_distribution.terminated
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=7_000.0, trigger=PaymentTrigger.AT_KO,
                         payment_offset_days=0)
    with pytest.raises(ValidationError, match="terminated note"):
        leg.value(result.event_distribution, env, 1.0)
    assert DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5.0, payment_time=0.1).value(
        result.event_distribution, env, 1.0) == pytest.approx(-5.0 * env.get_discount_factor(0.1))


def test_a_terminated_settled_note_has_no_statistics_and_no_event():
    product_f, env_f, state_f = SETTLED_STATE
    product, env = product_f(), env_f()
    assert DCNMCEngine(num_paths=N, seed=7).calculate_event_stats(product, env, lifecycle_state=state_f()) is None
    detailed = DCNMCEngine(num_paths=N, seed=7).price_detailed(product, env, lifecycle_state=state_f())
    assert detailed.event_stats is None and detailed.pv == 0.0
    result = DCNMCEngine(num_paths=N, seed=7).price_with_events(product, env, lifecycle_state=state_f())
    assert result.npv == 0.0 and result.event_distribution.terminated
    assert result.event_distribution.probabilities == {}


def test_a_ledger_flow_of_another_kind_is_refused_by_the_statistics_only():
    product, env = make_dcn(DCN_A, valuation_date=AGED), env_on(AGED)
    odd = RealizedCashflow(cashflow_id="dcn:odd:2024-06-03", event_type=LifecycleEventType.EXPIRY, amount=1.0,
                           determination_date=ROW, payment_date=ROW_PAY)
    state = state_on(AGED, odd)
    price = DCNMCEngine(num_paths=N, seed=7).price(product, env, lifecycle_state=state)
    assert np.isfinite(price)
    with pytest.raises(ValidationError, match="knock-out, coupon and loss"):
        DCNMCEngine(num_paths=N, seed=7).calculate_event_stats(product, env, lifecycle_state=state_on(AGED, odd))


# --- AC21: standard errors -------------------------------------------------------------------------------------------

SINGLE = ["dcn_sobol", "dcn_antithetic", "dcn_seller", "lv_dcn", "heston_dcn", "closed_saturday"]
BATCHED = ["dcn_sobol_batches", "dcn_sobol_batches_threads", "dcn_pseudo", "qe_dcn_batches", "act_360",
           "business_days", "receivable_alive", "closed_holiday"]


@pytest.mark.parametrize("name", SINGLE)
def test_single_batch_has_no_standard_errors(name):
    stats = _stats(name)
    assert stats.standard_errors is None and stats.unavailable["standard_errors"] == "undefined: single batch"


@pytest.mark.parametrize("name", BATCHED)
def test_batch_mean_standard_errors(name):
    engine, product, env = _fresh(name)
    stats = engine.calculate_event_stats(product, env, **_kwargs(name))
    records = engine._last_stats_records
    se = stats.standard_errors
    assert se is not None and "standard_errors" not in stats.unavailable
    for key in ("observations.probability", "coupons.probability", "coupons.discounted_cash",
                "maturity.probability", "ki_settlement.probability", "ki_settlement.discounted_cash",
                "knock_in.probability", "knock_in.expected_loss", "ki_table.probability", "expected_lifetime",
                "total_discounted_cash", "knock_in.expected_time"):
        assert key in se, key
    weights = np.array([float(r["n"]) for r in records])
    share = weights / weights.sum()
    means = np.array([float(r["ki_count"]) / float(r["n"]) for r in records])
    b = len(records)
    expected = np.sqrt(b / (b - 1) * np.sum(share ** 2 * (means - stats.knock_in.probability) ** 2))
    assert se["knock_in.probability"] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    seller_like = product.direction_sign
    cash_means = np.array([seller_like * float(r["loss_disc_cash"]) / float(r["n"]) for r in records])
    expected_cash = np.sqrt(b / (b - 1) * np.sum(share ** 2 * (cash_means - stats.ki_settlement.discounted_cash) ** 2))
    assert se["ki_settlement.discounted_cash"] == pytest.approx(expected_cash, rel=1e-10, abs=1e-9)


# --- execution framework, serialisation, legacy converter -----------------------------------------------------------

@pytest.mark.parametrize("workers, backend", [(1, "serial"), (3, "threads")])
def test_execution_kernel_price_detailed_carries_the_same_statistics(workers, backend):
    """The framework's batch adapter records the same batches and builds the same statistics (PRICE_DETAILED)."""
    from execution.test_dcn_batch_adapter import dispatch, make_batch_context

    engine, product, env = _fresh("dcn_sobol_batches")
    direct = engine.price_detailed(product, env)
    framed = dispatch(_fresh("dcn_sobol_batches")[0], product, env,
                      make_batch_context(workers=workers, backend=backend)).value
    assert framed.pv == direct.pv and framed.event_stats.pv == framed.pv
    for field in ("observations.probability", "coupons.discounted_cash", "ki_table.probability",
                  "ki_settlement.discounted_cash", "knock_in.expected_time"):
        np.testing.assert_array_equal(np.asarray(framed.event_stats.require(field)),
                                      np.asarray(direct.event_stats.require(field)))
    assert framed.event_stats.standard_errors["knock_in.probability"] == \
        direct.event_stats.standard_errors["knock_in.probability"]


def test_result_to_dict_is_json():
    engine, product, env = _fresh("receivable_alive")
    payload = engine.price_detailed(product, env, **_kwargs("receivable_alive")).to_dict()
    text = json.dumps(payload)
    assert '"product_family": "dcn"' in text and payload["event_stats"]["pv"] == payload["pv"]


@pytest.mark.parametrize("name", ["dcn_sobol", "ko_lock_after_lock"])
def test_price_with_events_carries_coupon_and_knock_in_streams(name):
    engine, product, env = _fresh(name)
    dist = engine.price_with_events(product, env).event_distribution
    stats = _stats(name)
    np.testing.assert_array_equal(dist.probabilities[EventType.COUPON], stats.coupons.probability)
    np.testing.assert_array_equal(dist.payment_times[EventType.COUPON], stats.coupons.payment_times)
    np.testing.assert_array_equal(dist.stream_times[EventType.KI], stats.ki_table.times)
    assert dist.stream_dates[EventType.KI] == stats.ki_table.dates
    assert not dist.flagged_streams
    assert abs(dist.probability_residual) <= ROUND_OFF

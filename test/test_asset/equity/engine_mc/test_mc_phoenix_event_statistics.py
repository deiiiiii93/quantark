"""MC Phoenix event statistics on the unified ``EventStatistics`` (spec T10).

Over the Phoenix part of the T3 route grid (``mc_price_path_routes``: GBM PSEUDO / QUASI / Dask / RQMC through
``method=``, local vol, Heston incl. antithetic and QE RQMC, SLV, SLV-QE) and constructed cases:

- AC7: ``EventStatistics`` with a coupon table and nothing unavailable but the mathematically undefined values;
- AC8: both residuals at MC round-off (every outcome summed directly over its own paths);
- AC1 / AC12: coupon cash per row equals an independent per-path loop on the same paths (memory arrears incl.
  ``initial_coupon_arrears`` and the knock-out release on a missed coupon; INSTANT at the row's settlement, EXPIRY
  at the note's actual end, bucketed in the cash ledger at that event, undiscounted cash summed per path);
- AC11: a coupon paid with a knock-out sits in the coupon table; KO cash is the redemption only;
- AC9 / AC10: no knock-in after a knock-out, valuation-date rules;
- AC13 (D18): the per-date KI table, discrete exact, continuous within one trading day + one MC step;
- AC21: standard errors from independent batch means.

The references re-run the engine's own price with a spy on the payoff kernel (same paths, grid and bridge seed)
and decide every outcome here (``mc_same_path_reference.phoenix_paths``).
"""
from __future__ import annotations

import math
import warnings
from copy import deepcopy

import numpy as np
import pytest

from quantark.priceenv import CurveQuotingBasis
from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.cashleg import EventDistribution, EventType
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod

from test_asset.equity.engine_mc import mc_price_path_routes as grid
from test_asset.equity.engine_mc import mc_same_path_reference as reference
from test_asset.equity.engine_mc.mc_price_path_routes import ROUTES

warnings.filterwarnings("ignore", category=DeprecationWarning)

LIVE = [n for n, r in ROUTES.items() if r.family == "phoenix" and not r.expired]
#: single-batch routes: one kernel call, so the same-path reference sees every path the price priced
SAME_PATH = ["ph_mem_instant_pseudo", "ph_nomem_instant_pseudo", "ph_mem_expiry_pseudo", "ph_nomem_expiry_quasi",
             "ph_mem_instant_dki_quasi", "ph_mem_arrears_instant_pseudo", "ph_mem_arrears_expiry_pseudo",
             "lv_ph_pseudo", "heston_ph_pseudo", "heston_ph_quasi", "slv_ph_pseudo", "heston_ph_antithetic",
             "slvqe_ph_pseudo", "ph_knocked_in"]
EXPIRY = [n for n in SAME_PATH if ROUTES[n].product().coupon_config.coupon_pay_type == CouponPayType.EXPIRY]
INSTANT = [n for n in SAME_PATH if n not in EXPIRY]
ROUND_OFF = 1e-12
UNDEFINED_KEYS = {"knock_in.expected_time", "standard_errors", "standard_errors.knock_in.expected_time"}


def _stats(name):
    route = ROUTES[name]
    engine, product, env = route.engine(), route.product(), route.env()
    return engine, product, env, engine.calculate_event_stats(product, env)


def _ref(name):
    route = ROUTES[name]
    return reference.phoenix_paths(route.engine(), route.product(), route.env())


def _phoenix(ko_dates=grid.KO_DATES, *, ko_barrier=105.0, ki="continuous", ki_barrier=70.0, ki_dates=None,
             knocked_in=False, has_ki=True, pay=CouponPayType.INSTANT) -> PhoenixOption:
    if not has_ki:
        ki_kw = {}
    elif ki == "continuous":
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_dates=list(ki_dates if ki_dates is not None else ko_dates))
    product = PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=ko_barrier, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_dates), **ki_kw),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * len(ko_dates), coupon_rate=0.02, coupon_pay_type=pay,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=True),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    return product


# --- AC7 / AC11 ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", LIVE)
def test_full_request_is_an_event_statistics_with_only_undefined_values_unavailable(name):
    _, _, _, stats = _stats(name)
    assert isinstance(stats, EventStatistics)
    assert stats.product_family == "phoenix"
    assert stats.coupons is not None and stats.coupons.size == stats.observations.size       # AC11, shared grid
    assert stats.coupons.probability is not None
    assert set(stats.unavailable) <= UNDEFINED_KEYS, dict(stats.unavailable)
    assert all(reason.startswith("undefined:") for reason in stats.unavailable.values()), dict(stats.unavailable)
    assert ("knock_in.expected_time" in stats.unavailable) == (stats.knock_in.probability == 0.0)
    assert stats.ki_table is not None and stats.knock_in.expected_loss is not None
    assert stats.knock_in.expected_loss >= 0.0
    assert stats.provenance.statistics_model == stats.provenance.price_model == type(ROUTES[name].engine()).__name__


# --- AC8 -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", LIVE)
def test_partition_and_reconciliation_are_round_off(name):
    _, _, _, stats = _stats(name)
    assert abs(stats.probability_residual) <= ROUND_OFF
    assert abs(stats.reconciliation_error) <= ROUND_OFF * max(1.0, abs(stats.pv))
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability


@pytest.mark.parametrize("name", ["ph_mem_instant_pseudo", "ph_mem_expiry_pseudo", "ph_mem_arrears_expiry_pseudo"])
def test_ledger_sums_to_the_outcome_cash(name):
    _, _, _, stats = _stats(name)
    assert math.isclose(stats.ledger.total_discounted_cash, stats.total_discounted_cash, rel_tol=1e-12)
    assert math.isclose(stats.ledger.total_cash_to_pay, stats.total_cash_to_pay, rel_tol=1e-12)


# --- AC1 / AC11 / AC12: the coupon table against a per-path loop on the same paths --------------------------------

@pytest.mark.parametrize("name", SAME_PATH)
def test_coupon_rows_equal_the_per_path_loop(name):
    _, product, _, stats = _stats(name)
    ref = _ref(name)
    n = ref["n"]
    coupons = stats.coupons
    np.testing.assert_array_equal(coupons.probability, ref["coupon_hit"].sum(axis=0) / n)
    np.testing.assert_allclose(coupons.cash_to_pay, ref["coupon_paid"].sum(axis=0) / n, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(coupons.discounted_cash, ref["coupon_disc"].sum(axis=0) / n, rtol=1e-12, atol=1e-15)
    expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    np.testing.assert_array_equal(coupons.payment_is_path_dependent, np.full(coupons.size, expiry))
    alive_in = np.concatenate([[1.0], stats.observations.survival[:-1]])
    np.testing.assert_allclose(coupons.survival, alive_in, rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", SAME_PATH)
def test_knock_out_cash_is_the_redemption_only_and_maturity_rows_are_direct(name):
    """AC11: the knock-out row holds the redemption (coupons paid with it are the coupon table's); the maturity and
    knocked-in settlement rows hold their redemption only, computed on the paths here."""
    _, _, _, stats = _stats(name)
    ref = _ref(name)
    n = ref["n"]
    rows = stats.observations.size
    ko = ref["is_ko"]
    np.testing.assert_array_equal(stats.observations.probability, np.bincount(ref["ko_row"][ko], minlength=rows) / n)
    np.testing.assert_allclose(stats.observations.cash_to_pay,
                               np.bincount(ref["ko_row"][ko], weights=ref["ko_redemption"][ko], minlength=rows) / n,
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        stats.observations.discounted_cash,
        np.bincount(ref["ko_row"][ko], weights=(ref["ko_redemption"] * ref["ko_settle_df"])[ko], minlength=rows) / n,
        rtol=1e-12, atol=1e-15)
    for row, mask in ((stats.maturity, ref["v0"]), (stats.ki_settlement, ref["v1"])):
        assert row.probability == mask.sum() / n
        assert math.isclose(row.cash_to_pay, ref["maturity_cash"][mask].sum() / n, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(row.discounted_cash, ref["maturity_cash"][mask].sum() * ref["terminal_df"] / n,
                            rel_tol=1e-12, abs_tol=1e-15)
    # coupons paid with a knock-out exist on these paths and are counted in the coupon table, not the KO row
    if not ref["carried"] or ko.any():
        paid_at_ko = ref["coupon_paid"][ko, ref["ko_row"][ko]]
        assert paid_at_ko.sum() > 0.0


@pytest.mark.parametrize("name", ["ph_mem_arrears_instant_pseudo", "ph_mem_arrears_expiry_pseudo"])
def test_arrears_owed_and_released_by_a_knock_out_on_a_missed_coupon(name):
    """The two memory rules the reference loop implements must actually occur on these paths: the arrears owed at
    the pricing date ride the first coupon hit, and a knock-out on a missed coupon (row 1: coupon level above the
    knock-out level) releases the arrears without the current coupon."""
    _, product, env, stats = _stats(name)
    ref = _ref(name)
    first_amount = product.get_coupon_payoff(0, year_fraction=product.get_coupon_period_year_fractions(env)[0])
    hit0 = ref["coupon_hit"][:, 0]
    np.testing.assert_allclose(ref["coupon_paid"][hit0, 0], first_amount + 0.03, rtol=0, atol=1e-15)
    released = ref["is_ko"] & (ref["ko_row"] == 1) & ~ref["coupon_hit"][:, 1]
    assert released.sum() > 20
    assert np.all(ref["coupon_paid"][released, 1] > 0.0)
    # the coupon row's cash is not probability x coupon on a memory row
    amount1 = product.get_coupon_payoff(1, year_fraction=product.get_coupon_period_year_fractions(env)[1])
    assert stats.coupons.cash_to_pay[1] > stats.coupons.probability[1] * amount1 * (1.0 + 1e-6)


@pytest.mark.parametrize("name", EXPIRY)
def test_expiry_coupons_are_bucketed_at_the_ending_event(name):
    """EXPIRY: the ledger holds the rolled-up coupons at each knock-out row's settlement and the terminal payment,
    discounted there and summed undiscounted per path; no single DF recovers a row's cash."""
    _, _, env, stats = _stats(name)
    ref = _ref(name)
    n = ref["n"]
    rows = stats.observations.size
    k = int(np.count_nonzero(stats.observations.is_ko_obs))
    ledger = stats.ledger
    pd = slice(k, k + rows + 1)                                 # KO rows, then the path-dependent coupon buckets
    expected_det = np.concatenate([stats.observations.times, [stats.maturity.time]])
    expected_pay = np.concatenate([stats.observations.payment_times, [stats.maturity.payment_time]])
    np.testing.assert_array_equal(ledger.determination_times[pd], expected_det)
    np.testing.assert_array_equal(ledger.payment_times[pd], expected_pay)
    cash = np.bincount(ref["end_row"], weights=ref["path_coupons"], minlength=rows + 1) / n
    disc = np.bincount(ref["end_row"], weights=ref["path_coupons"] * ref["end_df"], minlength=rows + 1) / n
    np.testing.assert_allclose(ledger.cash_to_pay[pd], cash, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ledger.discounted_cash[pd], disc, rtol=1e-12, atol=1e-15)
    placeholder = env.get_discount_factor(float(stats.coupons.payment_times[0]))
    assert not np.allclose(stats.coupons.cash_to_pay * placeholder, stats.coupons.discounted_cash, rtol=1e-9)


@pytest.mark.parametrize("name", INSTANT)
def test_deterministic_payment_rows_are_df_consistent(name):
    _, _, env, stats = _stats(name)
    for table in (stats.observations, stats.coupons):
        dfs = np.array([env.get_discount_factor(float(t)) for t in table.payment_times])
        np.testing.assert_allclose(table.discounted_cash, table.cash_to_pay * dfs, rtol=1e-12, atol=1e-14)
    for row in (stats.maturity, stats.ki_settlement):
        assert math.isclose(row.discounted_cash, row.cash_to_pay * env.get_discount_factor(row.payment_time),
                            rel_tol=1e-12, abs_tol=1e-14)


# --- AC9 / AC10 ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ki", ["discrete", "continuous"])
def test_a_knock_in_after_a_certain_knock_out_is_no_knock_in(ki):
    if ki == "discrete":
        product = _phoenix(ko_barrier=[1.0, 105.0, 105.0, 105.0], ki="discrete", ki_barrier=99.0,
                           ki_dates=(0.5, 0.75, 1.0))
        env = grid.flat_env()
    else:
        product = _phoenix((0.0, *grid.KO_DATES), ki_barrier=108.0)
        env = grid.flat_env(110.0)
    stats = PhoenixMCEngine(params=grid.mc()).calculate_event_stats(product, env)
    ref = reference.phoenix_paths(PhoenixMCEngine(params=grid.mc()), product, env)
    assert ref["is_ko"].all() and ref["ki_ever"].mean() > 0.2          # KI-ever would be large
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_settlement.probability == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert np.all(stats.ki_table.probability == 0.0)
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")


def test_knock_out_at_valuation_has_no_knock_in():
    _, _, _, stats = _stats("ph_ko_at_valuation")
    assert stats.observations.times[0] == 0.0 and stats.observations.probability[0] == 1.0
    # the coupon observed with the knock-out (its period is empty at t = 0) and nothing after it
    assert stats.coupons.probability[0] == 1.0 and np.all(stats.coupons.probability[1:] == 0.0)
    assert np.all(stats.coupons.cash_to_pay[1:] == 0.0)
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_table.size == 0 and stats.knock_in.expected_time is None
    assert stats.expected_lifetime == 0.0


def test_carried_knock_in_is_a_knock_in_at_zero():
    _, _, _, stats = _stats("ph_knocked_in")
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    assert stats.maturity.probability == 0.0


def test_carried_knock_in_knocked_out_at_valuation_keeps_its_knock_in():
    product = _phoenix((0.0, *grid.KO_DATES), knocked_in=True)
    stats = PhoenixMCEngine(params=grid.mc()).calculate_event_stats(product, grid.flat_env(110.0))
    assert stats.observations.probability[0] == 1.0
    assert stats.knock_in.probability == 1.0
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])


@pytest.mark.parametrize("ki", ["continuous", "discrete"])
def test_knock_in_observed_at_valuation_is_a_knock_in_at_zero(ki):
    product = _phoenix(ki=ki, ki_dates=(0.0, *grid.KO_DATES))
    stats = PhoenixMCEngine(params=grid.mc()).calculate_event_stats(product, grid.flat_env(65.0))
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    assert stats.ki_table.times[0] == 0.0 and stats.ki_table.probability[0] == 1.0
    assert np.all(stats.ki_table.probability[1:] == 0.0)


def test_no_knock_in_barrier_reports_computed_zeros():
    stats = PhoenixMCEngine(params=grid.mc()).calculate_event_stats(_phoenix(has_ki=False), grid.flat_env())
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_settlement.probability == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert stats.ki_table.size == 0
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")


# --- AC13 (D18) ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [n for n in LIVE if "knocked_in" not in n and "ko_at_valuation" not in n])
def test_knock_in_table_identities(name):
    _, product, env, stats = _stats(name)
    table, ki = stats.ki_table, stats.knock_in
    assert math.isclose(float(table.probability.sum()), ki.probability, rel_tol=1e-12, abs_tol=1e-14)
    np.testing.assert_array_equal(table.payment_times, table.times)              # D8
    if ki.probability == 0.0:
        return
    mean = float(np.sum(table.times * table.probability)) / ki.probability
    if product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS:
        horizon = float(table.times[-1])
        mc_step = horizon / (int(env.bus_days_in_year * horizon) + 1)
        assert abs(mean - ki.expected_time) <= 1.0 / env.bus_days_in_year + mc_step       # D18
        assert table.interpolation_scheme == "step_overlap" and table.interpolated.any()
    else:
        assert math.isclose(mean, ki.expected_time, rel_tol=1e-12)
        np.testing.assert_array_equal(table.times, product.get_ki_observation_profile(env)["observation_times"])
        assert not table.interpolated.any() and table.interpolation_scheme is None


@pytest.mark.parametrize("name", [n for n in SAME_PATH if n != "ph_knocked_in"])
def test_knock_in_counts_only_before_the_knock_out(name):
    _, _, env, stats = _stats(name)
    ref = _ref(name)
    n = ref["n"]
    assert stats.knock_in.probability == ref["ki_first"].sum() / n
    assert ref["ki_ever"].sum() > ref["ki_first"].sum()
    first = ref["ki_index"][ref["ki_first"]]
    if ref["continuous"]:
        days = reference.numeric_trading_days(float(ref["step_ends"][-1]), env.bus_days_in_year)
        np.testing.assert_array_equal(stats.ki_table.times, days)
        expected = reference.day_overlap(ref["step_ends"], np.bincount(first, minlength=ref["step_ends"].size) / n,
                                         days)
    else:
        expected = np.bincount(first, minlength=stats.ki_table.size) / n
    np.testing.assert_allclose(stats.ki_table.probability, expected, rtol=0, atol=1e-14)
    assert math.isclose(stats.knock_in.expected_time, float(ref["ki_time"][ref["ki_first"]].mean()), rel_tol=1e-12)
    loss = np.where(ref["v1"], 100.0 - ref["maturity_cash"], 0.0) * ref["terminal_df"]
    assert math.isclose(stats.knock_in.expected_loss, loss.sum() / n, rel_tol=1e-12, abs_tol=1e-15)


def test_continuous_rows_are_business_days_on_a_dated_contract():
    """A dated Phoenix (ProductClock.DATE) on a BUSINESS_DAYS calendar: one KI row per business day after the
    valuation, the last on the window's end; observation and payment dates carried on every table."""
    from datetime import datetime

    from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
    from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import CalendarType, create_calendar

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    valuation = datetime(2026, 3, 2)
    dates = [datetime(2026, 6, 1), datetime(2026, 9, 1), datetime(2026, 12, 1), datetime(2027, 3, 1)]
    product = PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=valuation, exercise_date=dates[-1],
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=105.0)
                                                                 for d in dates]),
            ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * 4, coupon_rate=0.02,
                                          coupon_pay_type=CouponPayType.EXPIRY,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=True),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.25),
                             rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                             valuation_date=valuation, day_count_convention=DayCountConvention.BUSINESS_DAYS,
                             calendar=cal, rate_curve_quoting=CurveQuotingBasis.TRADING_YEAR,
                             div_yield_quoting=CurveQuotingBasis.TRADING_YEAR)
    stats = PhoenixMCEngine(params=grid.mc()).calculate_event_stats(product, env)
    table = stats.ki_table
    assert table.dates is not None and len(table.dates) == table.size
    assert all(cal.is_business_day(d) for d in table.dates) and table.dates[-1] == dates[-1]
    assert table.dates[0] > valuation
    assert math.isclose(float(table.probability.sum()), stats.knock_in.probability, rel_tol=1e-12)
    assert stats.observations.dates == tuple(dates) and stats.ledger.payment_dates is not None
    assert abs(stats.reconciliation_error) <= ROUND_OFF * stats.pv and abs(stats.probability_residual) <= ROUND_OFF


# --- AC21 ------------------------------------------------------------------------------------------------------

SINGLE = ["ph_mem_instant_pseudo", "ph_mem_instant_dki_quasi", "lv_ph_pseudo", "heston_ph_antithetic"]
BATCHED = ["ph_mem_instant_rqmc_fixed", "ph_mem_expiry_rqmc_adaptive", "ph_mem_instant_dask", "ph_nomem_dki_dask",
           "lv_ph_rqmc", "qe_ph_rqmc", "lv_ph_cki_dask"]


@pytest.mark.parametrize("name", SINGLE)
def test_single_batch_has_no_standard_errors(name):
    _, _, _, stats = _stats(name)
    assert stats.standard_errors is None
    assert stats.unavailable["standard_errors"] == "undefined: single batch"


def test_deterministic_qmc_batches_have_no_standard_errors():
    _, _, _, stats = _stats("ph_nomem_expiry_dask_quasi")
    assert stats.standard_errors is None
    assert stats.unavailable["standard_errors"].startswith("undefined:")


@pytest.mark.parametrize("name", BATCHED)
def test_batch_mean_standard_errors(name):
    engine, _, _, stats = _stats(name)
    se = stats.standard_errors
    records = engine._last_stats_records
    assert se is not None and len(records) >= 2
    for key in ("observations.probability", "observations.discounted_cash", "coupons.probability",
                "coupons.discounted_cash", "coupons.cash_to_pay", "maturity.probability", "maturity.discounted_cash",
                "ki_settlement.probability", "knock_in.probability", "knock_in.expected_loss", "ki_table.probability",
                "expected_lifetime", "total_discounted_cash", "total_cash_to_pay"):
        assert key in se, key
        assert np.all(np.isfinite(se[key])) and np.all(np.asarray(se[key]) >= 0.0)
    n = np.array([float(r["n"]) for r in records])
    w = n / n.sum()
    for key, field in (("coupon_cash", "coupons.cash_to_pay"), ("coupon_count", "coupons.probability")):
        x = np.array([np.asarray(r[key], dtype=float) for r in records]) / n[:, None]
        mean = np.sum(w[:, None] * x, axis=0)
        np.testing.assert_allclose(mean, getattr(stats.coupons, field.split(".")[1]), rtol=1e-12, atol=1e-15)
        by_hand = np.sqrt(len(n) / (len(n) - 1) * np.sum((w * w)[:, None] * (x - mean) ** 2, axis=0))
        np.testing.assert_allclose(se[field], by_hand, rtol=1e-12, atol=1e-15)
    x = np.array([float(r["ki_before_ko_count"]) for r in records]) / n
    mean = float(np.sum(w * x))
    by_hand = math.sqrt(len(n) / (len(n) - 1) * float(np.sum(w * w * (x - mean) ** 2)))
    assert math.isclose(se["knock_in.probability"], by_hand, rel_tol=1e-12, abs_tol=1e-15)
    if stats.knock_in.expected_time is not None and sum(r["ki_before_ko_count"] for r in records) >= 2:
        num = np.array([float(r["ki_time_sum"]) for r in records])
        den = np.array([float(r["ki_before_ko_count"]) for r in records])
        ratio = num.sum() / den.sum()
        delta = math.sqrt(len(n) / (len(n) - 1) * float(np.sum((num - ratio * den) ** 2))) / den.sum()
        assert math.isclose(se["knock_in.expected_time"], delta, rel_tol=1e-12)


# --- the cash-leg route, method= routes, streams= ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["ph_mem_instant_pseudo", "ph_mem_expiry_pseudo", "ph_mem_instant_rqmc_fixed"])
def test_price_with_events_carries_the_coupon_stream(name):
    route = ROUTES[name]
    price = route.engine().price(route.product(), route.env())
    engine, product, env = route.engine(), route.product(), route.env()
    result = engine.price_with_events(product, env)
    assert result.npv == price
    stats = route.engine().calculate_event_stats(route.product(), route.env())
    dist = result.event_distribution
    np.testing.assert_array_equal(dist.probabilities[EventType.COUPON], stats.coupons.probability)
    assert not dist.flagged_streams, dist.flagged_streams
    expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
    assert (EventType.COUPON in dist.path_dependent_payment_events) == expiry
    assert isinstance(EventDistribution.from_statistics(stats), EventDistribution)


@pytest.mark.parametrize("method", [MonteCarloMethod.PSEUDO, MonteCarloMethod.QUASI])
def test_method_selects_the_draws_and_use_qmc_is_not_read(method):
    """``PhoenixMCEngine`` takes QMC through ``method=``; ``MCParams.use_qmc`` does not change the price or the
    statistics (run-state minor: behaviour kept, price bitwise)."""
    product, env = grid.phoenix(), grid.flat_env()
    a = PhoenixMCEngine(params=MCParams(num_paths=grid.N, seed=7, use_qmc=False), method=method)
    b = PhoenixMCEngine(params=MCParams(num_paths=grid.N, seed=7, use_qmc=True), method=method)
    sa, sb = a.calculate_event_stats(product, env), b.calculate_event_stats(product, env)
    assert sa.pv == sb.pv == a.price(product, env)
    np.testing.assert_array_equal(sa.coupons.discounted_cash, sb.coupons.discounted_cash)
    other = PhoenixMCEngine(params=MCParams(num_paths=grid.N, seed=7),
                            method=(MonteCarloMethod.QUASI if method == MonteCarloMethod.PSEUDO
                                    else MonteCarloMethod.PSEUDO))
    assert other.price(product, env) != sa.pv


def test_streams_are_accepted_and_nothing_is_pruned():
    from quantark.cashleg import EventType as E

    route = ROUTES["ph_mem_instant_pseudo"]
    full = route.engine().calculate_event_stats(route.product(), route.env())
    pruned = route.engine().calculate_event_stats(route.product(), route.env(), streams=frozenset({E.KO}))
    assert pruned.pv == full.pv and not pruned.unavailable.keys() - full.unavailable.keys()
    np.testing.assert_array_equal(pruned.coupons.probability, full.coupons.probability)


def test_a_fresh_engine_per_call_reproduces_the_statistics():
    route = ROUTES["ph_mem_expiry_rqmc_adaptive"]
    a = route.engine().calculate_event_stats(route.product(), route.env())
    b = deepcopy(route.engine()).calculate_event_stats(route.product(), route.env())
    assert a.pv == b.pv
    np.testing.assert_array_equal(a.coupons.discounted_cash, b.coupons.discounted_cash)

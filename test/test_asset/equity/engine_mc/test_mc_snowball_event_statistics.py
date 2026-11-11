"""MC Snowball and KO-reset event statistics on the unified ``EventStatistics`` (spec T9).

Over the Snowball / KO-reset part of the T3 route grid (``mc_price_path_routes``) and a few constructed cases:

- AC7: ``EventStatistics`` with nothing unavailable but the mathematically undefined values;
- AC8: both residuals at MC round-off (every outcome summed directly over its own paths);
- AC9: a knock-in after a certain knock-out is no knock-in (P(KI) = 0, expected loss 0);
- AC10: valuation-date rules (KO at valuation, carried knock-in, knock-in at t = 0, no KI barrier);
- AC12: cash to pay summed undiscounted over the paths (path-dependent REBASED rows), DF-consistent where the
  payment is deterministic;
- AC13 (D18): the per-date KI table follows the product's KI schedule; Σ p = P(KI) and Σ t p / P(KI) = E[KI
  time], exact for discrete KI, within one trading day + one MC step for continuous KI (step overlap);
- AC15: REBASED offset table + absolute termination distribution on the deterministic candidate axis;
- AC21: standard errors from independent batch means (single batch / deterministic QMC: undefined);
- REBASED end to end through ``price_with_events`` and a cash leg, against a same-path valuation.

The independent references re-run the engine's own path generator on its own grid (same seed, same bridge seed)
and decide every outcome here, with plain NumPy, from the definitions.
"""
from __future__ import annotations

import math
import warnings
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import EventDistribution, EventType, FixedPayoffLeg, LegDirection, PaymentTrigger
from quantark.montecarlo.conditional_snowball import conditional_standard_snowball_event_record
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode

from test_asset.equity.engine_mc import mc_price_path_routes as grid
from test_asset.equity.engine_mc import mc_same_path_reference as reference
from test_asset.equity.engine_mc.mc_price_path_routes import ROUTES

warnings.filterwarnings("ignore", category=DeprecationWarning)

FAMILIES = ("snowball", "ko_reset")
LIVE = [n for n, r in ROUTES.items() if r.family in FAMILIES and not r.expired]
ROUND_OFF = 1e-12
UNDEFINED_KEYS = {"knock_in.expected_time", "standard_errors", "standard_errors.knock_in.expected_time"}


def _stats(name):
    route = ROUTES[name]
    engine, product, env = route.engine(), route.product(), route.env()
    return engine, product, env, engine.calculate_event_stats(product, env)


def _snowball(ko_dates=grid.KO_DATES, *, ko_barrier=103.0, ki="continuous", ki_barrier=75.0, ki_dates=None,
              knocked_in=False, has_ki=True) -> SnowballOption:
    if not has_ki:
        ki_kw = {}
    elif ki == "continuous":
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_dates=list(ki_dates if ki_dates is not None else ko_dates))
    product = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=ko_barrier, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_dates), **ki_kw),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    return product


# --- independent same-path references (mc_same_path_reference) ---------------------------------------------------

def _first(hit):
    any_hit = hit.any(axis=1)
    return any_hit, np.where(any_hit, np.argmax(hit, axis=1), -1)


# --- AC7 -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", LIVE)
def test_full_request_is_an_event_statistics_with_only_undefined_values_unavailable(name):
    _, product, _, stats = _stats(name)
    assert isinstance(stats, EventStatistics)
    assert stats.product_family == ROUTES[name].family
    assert stats.coupons is None                                    # AC11: no coupon table on these families
    assert set(stats.unavailable) <= UNDEFINED_KEYS, dict(stats.unavailable)
    assert all(reason.startswith("undefined:") for reason in stats.unavailable.values()), dict(stats.unavailable)
    assert ("knock_in.expected_time" in stats.unavailable) == (stats.knock_in.probability == 0.0)
    assert stats.ki_table is not None and stats.knock_in.expected_loss is not None
    assert stats.provenance.statistics_model == stats.provenance.price_model == type(ROUTES[name].engine()).__name__


# --- AC8 -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", LIVE)      # the conditional control included: control-corrected outcomes (T10)
def test_partition_and_reconciliation_are_round_off(name):
    _, _, _, stats = _stats(name)
    assert abs(stats.probability_residual) <= ROUND_OFF
    assert abs(stats.reconciliation_error) <= ROUND_OFF * max(1.0, abs(stats.pv))
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability


def test_ledger_sums_to_the_outcome_cash():
    for name in ("sb_cki_pseudo", "kr_abs_pseudo", "kr_rebased_expiry"):
        _, _, _, stats = _stats(name)
        assert math.isclose(stats.ledger.total_discounted_cash, stats.total_discounted_cash, rel_tol=1e-12)
        assert math.isclose(stats.ledger.total_cash_to_pay, stats.total_cash_to_pay, rel_tol=1e-12)


# --- AC9 -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ki", ["discrete", "continuous"])
def test_a_knock_in_after_a_certain_knock_out_is_no_knock_in(ki):
    """Discrete: every path knocks out at 0.25 (barrier 1) before the first KI date; continuous: knocked out at
    valuation (spot 110 over 103 on a t = 0 date). The barrier is crossed afterwards on many paths."""
    if ki == "discrete":
        product = _snowball(ko_barrier=[1.0, 103.0, 103.0, 103.0], ki="discrete", ki_barrier=99.0,
                            ki_dates=(0.5, 0.75, 1.0))
        env = grid.flat_env()
    else:
        product = _snowball((0.0, *grid.KO_DATES), ki_barrier=108.0)
        env = grid.flat_env(110.0)
    stats = SnowballMCEngine(params=grid.mc()).calculate_event_stats(product, env)
    ref = reference.snowball_paths(SnowballMCEngine(params=grid.mc()), product, env)
    assert ref["is_ko"].all() and ref["ki_ever"].mean() > 0.2         # KI-ever would be large
    assert stats.knock_in.probability == 0.0
    assert stats.knock_in.expected_loss == 0.0
    assert stats.ki_settlement.probability == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert np.all(stats.ki_table.probability == 0.0)
    assert stats.knock_in.expected_time is None and stats.unavailable["knock_in.expected_time"].startswith("undefined:")


SAME_PATH_SNOWBALL = ["sb_cki_pseudo", "sb_dki_pseudo", "sb_cki_quasi", "sb_reverse", "sb_cki_no_ko_after_ki",
                      "heston_sb_cki_pseudo", "lv_sb_cki_pseudo", "heston_sb_dki_antithetic", "slv_sb_dki_pseudo"]


@pytest.mark.parametrize("name", SAME_PATH_SNOWBALL)
def test_knock_in_counts_only_before_the_knock_out(name):
    """The same paths decided here: P(KI before KO), the per-row first-KI table (continuous: trading days by
    brute-force step overlap) and E[KI time]; KI-ever strictly larger."""
    engine, product, env, stats = _stats(name)
    ref = reference.snowball_paths(ROUTES[name].engine(), product, env)
    n = ref["n"]
    assert stats.knock_in.probability == ref["ki_first"].sum() / n
    assert ref["ki_ever"].sum() > ref["ki_first"].sum()
    first = ref["ki_index"][ref["ki_first"]]
    if ref["continuous"]:
        days = reference.numeric_trading_days(float(ref["step_ends"][-1]), env.bus_days_in_year)
        np.testing.assert_array_equal(stats.ki_table.times, days)
        mass = np.bincount(first, minlength=ref["step_ends"].size) / n
        expected = reference.day_overlap(ref["step_ends"], mass, days)
    else:
        expected = np.bincount(first, minlength=stats.ki_table.size) / n
    np.testing.assert_allclose(stats.ki_table.probability, expected, rtol=0, atol=1e-14)
    assert math.isclose(stats.knock_in.expected_time, float(ref["ki_time"][ref["ki_first"]].mean()), rel_tol=1e-12)
    if not product.barrier_config.disable_ko_after_ki:
        assert stats.ki_settlement.probability == (ref["ki_first"] & ~ref["is_ko"]).sum() / n


# --- AC10 ------------------------------------------------------------------------------------------------------

def test_knock_out_at_valuation_has_no_knock_in():
    _, _, _, stats = _stats("sb_ko_at_valuation")
    assert stats.observations.probability[0] == 1.0 and stats.observations.times[0] == 0.0
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_table.size == 0
    assert stats.knock_in.expected_time is None
    assert stats.expected_lifetime == 0.0


@pytest.mark.parametrize("name", ["sb_cki_knocked_in", "sb_cki_knocked_in_no_ko_after_ki", "kr_abs_knocked_in"])
def test_carried_knock_in_is_a_knock_in_at_zero(name):
    _, _, _, stats = _stats(name)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    assert not np.any(stats.ki_table.interpolated)


def test_carried_knock_in_knocked_out_at_valuation_keeps_its_knock_in():
    product = _snowball((0.0, *grid.KO_DATES), knocked_in=True)
    stats = SnowballMCEngine(params=grid.mc()).calculate_event_stats(product, grid.flat_env(110.0))
    assert stats.observations.probability[0] == 1.0
    assert stats.knock_in.probability == 1.0
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])


@pytest.mark.parametrize("ki", ["continuous", "discrete"])
def test_knock_in_observed_at_valuation_is_a_knock_in_at_zero(ki):
    ki_dates = (0.0, *grid.KO_DATES)
    product = _snowball(ki=ki, ki_dates=ki_dates)
    stats = SnowballMCEngine(params=grid.mc()).calculate_event_stats(product, grid.flat_env(70.0))
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    assert stats.ki_table.times[0] == 0.0 and stats.ki_table.probability[0] == 1.0
    assert np.all(stats.ki_table.probability[1:] == 0.0)


def test_no_knock_in_barrier_reports_computed_zeros():
    product = _snowball(has_ki=False)
    stats = SnowballMCEngine(params=grid.mc()).calculate_event_stats(product, grid.flat_env())
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_settlement.probability == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert stats.ki_table.size == 0
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")


def test_ko_reset_knock_out_at_valuation_has_no_knock_in():
    from dataclasses import replace

    product = grid.ko_reset()
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=103.0) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]))
    product.validate()
    stats = SnowballMCEngine(params=grid.mc()).calculate_event_stats(product, grid.flat_env(110.0))
    assert stats.observations.probability[0] == 1.0
    assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0


# --- AC12 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["sb_cki_pseudo", "sb_cki_expiry_pseudo", "kr_abs_pseudo", "kr_abs_expiry"])
def test_deterministic_payment_rows_are_df_consistent(name):
    _, _, env, stats = _stats(name)
    obs = stats.observations
    dfs = np.array([env.get_discount_factor(float(t)) for t in obs.payment_times])
    np.testing.assert_allclose(obs.discounted_cash, obs.cash_to_pay * dfs, rtol=1e-12, atol=1e-13)
    for row in (stats.maturity, stats.ki_settlement):
        assert math.isclose(row.discounted_cash, row.cash_to_pay * env.get_discount_factor(row.payment_time),
                            rel_tol=1e-12, abs_tol=1e-13)


@pytest.mark.parametrize("name", ["kr_rebased_pseudo", "kr_rebased_expiry"])
def test_rebased_cash_to_pay_is_the_per_path_undiscounted_sum(name):
    """Same paths through the kernel (its own payoffs and settlements): each termination row's cash to pay is the
    undiscounted path sum of its (event, determination, payment); offsets total the same per event type."""
    _, product, env, stats = _stats(name)
    ref = reference.ko_reset_paths(ROUTES[name].engine(), product, env)
    n = ref["n"]
    term = stats.termination
    expected = np.zeros(term.size)
    for i, (event, t, pay) in enumerate(zip(term.event, term.times, term.payment_times)):
        mask = (ref["event"] == event) & (ref["det"] == t) & (ref["settle"] == pay)
        expected[i] = ref["payoffs"][mask].sum() / n
    assert sum(np.count_nonzero((ref["event"] == e) & (ref["det"] == t) & (ref["settle"] == p))
               for e, t, p in zip(term.event, term.times, term.payment_times)) == n       # every path on the axis
    np.testing.assert_allclose(term.cash_to_pay, expected, rtol=1e-12, atol=1e-13)
    dfs = np.array([env.get_discount_factor(float(t)) for t in term.payment_times])
    np.testing.assert_allclose(term.discounted_cash, term.cash_to_pay * dfs, rtol=1e-12, atol=1e-13)
    # the offset rows are path-dependent: no single payment DF recovers them from the discounted cash
    offsets = stats.post_ki_offsets
    post = np.asarray(offsets.kind) == "ko_post"
    assert offsets.cash_to_pay[post].sum() > offsets.discounted_cash[post].sum()


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
    continuous = product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
    if continuous:
        days = np.diff(np.concatenate([[0.0], table.times]))
        horizon = float(table.times[-1])
        mc_step = horizon / (int(env.bus_days_in_year * horizon) + 1)
        tolerance = 1.0 / env.bus_days_in_year + mc_step                         # D18
        assert abs(mean - ki.expected_time) <= tolerance
        assert np.all(days <= 1.0 / env.bus_days_in_year * (1.0 + 1e-12))
        assert table.interpolation_scheme == "step_overlap" and table.interpolated.any()
    else:
        assert math.isclose(mean, ki.expected_time, rel_tol=1e-12)
        rows = (product.get_ki_observation_profile(env)["observation_times"]
                if not hasattr(product, "post_ko_mode") else
                [t for t in product.get_ki_observation_profile(env)["observation_times"]
                 if t <= product.get_pre_maturity_time(env) + 1e-12])
        np.testing.assert_array_equal(table.times, np.asarray(rows, dtype=float))
        assert not table.interpolated.any() and table.interpolation_scheme is None


def test_continuous_rows_are_trading_days_on_a_numeric_clock():
    _, product, env, stats = _stats("sb_cki_pseudo")
    times = stats.ki_table.times
    step = 1.0 / env.bus_days_in_year
    np.testing.assert_allclose(times[:-1], step * np.arange(1, times.size), rtol=0, atol=1e-15)
    assert times[-1] == product.get_maturity(env) and stats.ki_table.dates is None


def test_continuous_rows_are_business_days_on_a_dated_contract():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    valuation = datetime(2026, 3, 2)
    dates = [datetime(2026, 6, 1), datetime(2026, 9, 1), datetime(2026, 12, 1), datetime(2027, 3, 1)]
    product = SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=valuation, exercise_date=dates[-1],
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=103.0)
                                                                 for d in dates]),
            ki_barrier=80.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.25),
                             rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                             valuation_date=valuation, day_count_convention=DayCountConvention.BUSINESS_DAYS,
                             calendar=cal)
    stats = SnowballMCEngine(params=grid.mc()).calculate_event_stats(product, env)
    table = stats.ki_table
    assert table.dates is not None and len(table.dates) == table.size
    assert all(cal.is_business_day(d) for d in table.dates) and table.dates[-1] == dates[-1]
    assert table.dates[0] > valuation
    assert math.isclose(float(table.probability.sum()), stats.knock_in.probability, rel_tol=1e-12)


# --- AC15 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["kr_rebased_pseudo", "kr_rebased_quasi", "kr_rebased_rqmc", "kr_rebased_dask",
                                  "kr_rebased_expiry"])
def test_rebased_offsets_and_termination(name):
    _, _, _, stats = _stats(name)
    assert stats.rebased and stats.post_ki_offsets is not None and stats.termination is not None
    term, offsets = stats.termination, stats.post_ki_offsets
    for kind in ("ko_post", "ki_settlement"):
        in_term = np.asarray(term.event) == kind
        in_offsets = np.asarray(offsets.kind) == kind
        for column in ("probability", "discounted_cash", "cash_to_pay"):
            assert math.isclose(float(getattr(term, column)[in_term].sum()),
                                float(getattr(offsets, column)[in_offsets].sum()), rel_tol=1e-12, abs_tol=1e-14)
    assert set(stats.observations.schedule) == {"pre"}
    assert math.isclose(stats.ki_settlement.probability,
                        float(term.probability[np.asarray(term.event) == "ki_settlement"].sum()), rel_tol=1e-12)
    dist = EventDistribution.from_statistics(stats)
    assert dist.rebased and isinstance(dist.probabilities[EventType.MATURITY_WITH_KI], np.ndarray)


def test_rebased_candidate_axis_does_not_depend_on_seed_or_spot():
    axes = []
    for seed, spot in ((7, 100.0), (8, 100.0), (7, 101.0)):
        engine = SnowballMCEngine(params=grid.mc(seed=seed))
        stats = engine.calculate_event_stats(grid.ko_reset(PostKOScheduleMode.REBASED), grid.flat_env(spot))
        term = stats.termination
        axes.append((tuple(term.times), tuple(term.event), tuple(term.payment_times)))
        assert np.any(term.probability == 0.0) or term.size > 0
    assert axes[0] == axes[1] == axes[2]


def test_rebased_end_to_end_cash_leg_equals_the_same_path_valuation():
    """``price_with_events`` on REBASED: npv is price() bitwise, and a FixedPayoffLeg paying 1 at the knock-out
    (+ 2 days) equals Σ_paths 1{KO} DF(settlement + offset) / n on the kernel's own paths and settlements."""
    route = ROUTES["kr_rebased_pseudo"]
    engine, product, env = route.engine(), route.product(), route.env()
    price = route.engine().price(route.product(), route.env())
    result = engine.price_with_events(product, env)
    assert result.npv == price
    ref = reference.ko_reset_paths(route.engine(), product, env)
    ko = ref["is_pre"] | ref["is_post"]
    for trigger, paid in ((PaymentTrigger.AT_KO, ko), (PaymentTrigger.AT_MATURITY_ANY, ~ko)):
        leg = FixedPayoffLeg(leg_id="fee", direction=LegDirection.BUYER_RECEIVES, amount=1.0, trigger=trigger,
                             payment_offset_days=2)
        value = leg.value(result.event_distribution, env, 1.0)
        direct = float(np.sum([env.get_discount_factor(float(t) + 2.0 / 365.0) for t in ref["settle"][paid]])) / ref["n"]
        assert math.isclose(value, direct, rel_tol=1e-12), trigger


# --- AC21 ------------------------------------------------------------------------------------------------------

SINGLE = ["sb_cki_pseudo", "sb_dki_quasi", "kr_abs_pseudo", "heston_sb_cki_pseudo"]
BATCHED = ["sb_cki_rqmc_fixed", "sb_cki_rqmc_adaptive", "sb_cki_dask", "lv_sb_dki_rqmc", "qe_sb_affine_rqmc",
           "qe_sb_affine_adaptive", "kr_abs_rqmc_adaptive", "kr_abs_dask", "kr_rebased_rqmc", "kr_rebased_dask"]


@pytest.mark.parametrize("name", SINGLE)
def test_single_batch_has_no_standard_errors(name):
    _, _, _, stats = _stats(name)
    assert stats.standard_errors is None
    assert stats.unavailable["standard_errors"] == "undefined: single batch"


def test_deterministic_qmc_batches_have_no_standard_errors():
    _, _, _, stats = _stats("sb_dki_dask_quasi")
    assert stats.standard_errors is None
    assert stats.unavailable["standard_errors"].startswith("undefined:")


@pytest.mark.parametrize("name", BATCHED)
def test_batch_mean_standard_errors(name):
    engine, _, _, stats = _stats(name)
    se = stats.standard_errors
    records = engine._last_stats_records
    assert se is not None and len(records) >= 2
    for key in ("observations.probability", "observations.discounted_cash", "maturity.probability",
                "maturity.discounted_cash", "ki_settlement.probability", "knock_in.probability",
                "knock_in.expected_loss", "ki_table.probability", "expected_lifetime", "total_discounted_cash"):
        assert key in se, key
        assert np.all(np.isfinite(se[key])) and np.all(np.asarray(se[key]) >= 0.0)
    # knock_in.probability by hand: the batch means' spread (path-weighted, unequal batches allowed)
    n = np.array([float(r["n"]) for r in records])
    x = np.array([float(r["ki_before_ko_count"]) for r in records]) / n
    w = n / n.sum()
    mean = float(np.sum(w * x))
    assert math.isclose(mean, stats.knock_in.probability, rel_tol=1e-12, abs_tol=1e-15)
    by_hand = math.sqrt(len(n) / (len(n) - 1) * float(np.sum(w * w * (x - mean) ** 2)))
    assert math.isclose(se["knock_in.probability"], by_hand, rel_tol=1e-12, abs_tol=1e-15)
    if len(set(n)) == 1:
        assert math.isclose(by_hand, float(np.std(x, ddof=1)) / math.sqrt(len(n)), rel_tol=1e-12, abs_tol=1e-15)
    # E[KI time] by the delta method on the ratio of batch sums
    if stats.knock_in.expected_time is not None and sum(r["ki_before_ko_count"] for r in records) >= 2:
        num = np.array([float(r["ki_time_sum"]) for r in records])
        den = np.array([float(r["ki_before_ko_count"]) for r in records])
        ratio = num.sum() / den.sum()
        delta = math.sqrt(len(n) / (len(n) - 1) * float(np.sum((num - ratio * den) ** 2))) / den.sum()
        assert math.isclose(se["knock_in.expected_time"], delta, rel_tol=1e-12)


def test_expected_time_error_is_undefined_under_two_knock_in_paths():
    """A batched run where no path knocks in before a knock-out: the ratio has no error."""
    params = grid.rqmc_fixed()
    engine = SnowballMCEngine(params=deepcopy(params), method=grid.RQMC)
    product = _snowball(ko_barrier=[1.0, 103.0, 103.0, 103.0], ki="discrete", ki_barrier=99.0,
                        ki_dates=(0.5, 0.75, 1.0))
    stats = engine.calculate_event_stats(product, grid.flat_env())
    assert stats.standard_errors is not None
    assert stats.unavailable["standard_errors.knock_in.expected_time"].startswith("undefined:")
    assert "knock_in.expected_time" not in stats.standard_errors


# --- the exact conditional KI-before-KO formula ---------------------------------------------------------------

def test_conditional_knock_in_before_knock_out_matches_sampling_the_factor():
    """The exact-affine record's KI-before-KO, clean KO and first-KI rows against sampling Z on the same base
    paths (KO wins a tie with the discrete KI observed at the same date)."""
    rng = np.random.default_rng(3)
    product = grid.affine_snowball()
    env = grid.flat_env()
    n_rows, cols = 6, 5
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.08, size=(n_rows, cols)), axis=1))
    base[:, 0] = 100.0
    loading = np.concatenate([np.zeros((n_rows, 1)), np.tile(np.sqrt([0.25, 0.5, 0.75, 1.0]) * 0.2, (n_rows, 1))],
                             axis=1)
    idx = np.arange(4)
    record = conditional_standard_snowball_event_record(
        product=product, pricing_env=env, base_paths=base, log_spot_factor_loadings=loading, ko_indices=idx,
        ki_indices=idx, maturity=1.0, discount_factors=lambda t: np.exp(-0.03 * np.asarray(t, dtype=float)))
    z = rng.standard_normal(400_000)
    ko_barrier, ki_barrier = 105.0, 75.0
    ki_first = np.zeros(4)
    clean = np.zeros(4)
    for row in range(n_rows):
        spots = base[row, 1:][None, :] * np.exp(loading[row, 1:][None, :] * z[:, None])
        ko_hit, ki_hit = spots >= ko_barrier, spots <= ki_barrier
        is_ko, ko_row = _first(ko_hit)
        is_ki, ki_row = _first(ki_hit)
        before = is_ki & (~is_ko | (ki_row < ko_row))
        ki_first += np.bincount(ki_row[before], minlength=4) / z.size
        clean_ko = is_ko & ~(is_ki & (ki_row < ko_row))
        clean += np.bincount(ko_row[clean_ko], minlength=4) / z.size
    se = math.sqrt(0.25 / z.size) * math.sqrt(n_rows)
    np.testing.assert_allclose(record["ki_first_count"], ki_first, rtol=0, atol=5 * se)
    np.testing.assert_allclose(record["ko_clean_count"], clean, rtol=0, atol=5 * se)
    assert math.isclose(record["ki_before_ko_count"], float(record["ki_first_count"].sum()), rel_tol=1e-12)
    assert record["ki_before_ko_count"] >= record["ki_no_ko_count"] - 1e-12


@pytest.mark.parametrize("name", ["qe_sb_affine_rqmc", "slvqe_sb_control_only"])
def test_conditional_routes_carry_a_knock_in_before_knock_out(name):
    _, _, _, stats = _stats(name)
    assert stats.knock_in.probability is not None and stats.knock_in.probability >= stats.ki_settlement.probability
    assert "knock_in.probability" not in stats.unavailable

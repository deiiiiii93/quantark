"""QUAD V2 Phoenix pays the memory arrears owed at the pricing date (unified event statistics spec D24, AC28).

``CouponBarrierConfig.initial_coupon_arrears`` is the cash of coupon periods missed before the contract's first
remaining observation and still owed. QUAD V1, the PDE solver and Monte Carlo pay it by one rule: the arrears join the
memory accrual at the first remaining coupon observation -- released with that period's coupon on a hit, released at a
knock-out there whatever the coupon barrier says, carried in the memory otherwise; under EXPIRY pay they roll into the
note's end payment, discounted at the path's own end (its knock-out settlement or the terminal payment). QUAD V2 never
read the field: it priced the no-arrears contract.

Gates (AC28):

* QUAD V1 on the same contract: price and per-row coupon cash within an allowance measured by refining V1's grid
  (``V1_PV_ALLOWANCE`` / ``V1_COUPON_ALLOWANCE``; the refinement table is below and in the T4b report);
* a PSEUDO Monte Carlo run of ``N_SEEDS`` independent seeds: within 4 standard errors of the seed mean plus that
  allowance;
* V2's own closing identities: the ledger reconciles the price to round-off, and the coupon cash sums to
  ``price(c, arrears) - price(c = 0, arrears = 0)`` (the price is linear in the coupon amounts and the arrears).

Contracts: the T4 fixture (knock-out 103, a 0.1-year settlement lag) with the first remaining row's coupon barrier at
85 (a coupon hit there releases the arrears with the coupon) or at 108, above the knock-out barrier (a knock-out there
misses the coupon and releases the arrears alone); the same with an observation at the valuation instant; and a dated
contract aged past its first observation.
"""

from datetime import datetime

import numpy as np
import pytest

from test_asset.equity.engine_quad.test_phoenix_quad_coupon_cash import OBS, _env, _phoenix
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.quad import PhoenixQuadEngineV2
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option import PhoenixOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.exceptions import ValidationError

#: two quarterly coupons (0.03 x 0.25 x 100 = 0.75 each) owed at the pricing date
ARREARS = 1.5

# V1 grid the gates run at, and the V1-V2 allowances banked there. Refinement (T4b report; max |V2 - V1| over the
# ten arrears contracts below; V2 is the converged side: cells_per_sd 2 -> 4 or continuous_steps_per_year 48 -> 96
# moves its price and coupon cash < 5e-11):
#   grid         1001      2001      4001      8001     16001
#   PV       8.71e-04  3.04e-04  7.07e-05  1.90e-05  3.31e-06
#   coupon   2.19e-04  4.78e-05  1.52e-05  2.86e-06  5.42e-07   (max over rows)
# Banked: the 4001 gap times 2 (deterministic; the factor covers FFT round-off across machines and library
# versions, orders of magnitude below it, without hiding a regression of the gap's size -- the defect was ~1.0).
GRID_POINTS = 4001
V1_PV_ALLOWANCE = 1.5e-4
V1_COUPON_ALLOWANCE = 3.0e-5
#: V2 ledger reconciliation and linear closing: round-off on prices of order 100
ROUND_OFF = 1.0e-10

N_SEEDS = 8
PATHS_PER_SEED = 20_000
SE_MULTIPLE = 4.0

PAY_TYPES = [pytest.param(CouponPayType.INSTANT, id="instant"), pytest.param(CouponPayType.EXPIRY, id="expiry")]
#: the first remaining row's coupon barrier: below the knock-out barrier (a coupon hit pays there with or without a
#: knock-out) or above it (a knock-out there misses the coupon and releases the arrears alone)
FIRST_ROW = [pytest.param(85.0, id="coupon-hit-first"), pytest.param(108.0, id="ko-on-missed-coupon-first")]


def _fresh(pay_type, first_barrier, arrears=ARREARS, coupon_rate=0.03):
    return _phoenix(memory=True, pay_type=pay_type, coupon_rate=coupon_rate,
                    coupon_barriers=[first_barrier, 108.0, 85.0, 85.0], initial_arrears=arrears)


#: an observation AT the valuation instant (an empty period) and the fixture's four
OBS_NOW = [0.0, *OBS]


def _now(pay_type, first_barrier, arrears=ARREARS, coupon_rate=0.03):
    return _phoenix(memory=True, pay_type=pay_type, coupon_rate=coupon_rate, obs=OBS_NOW,
                    coupon_barriers=[first_barrier, 108.0, 108.0, 85.0, 85.0], initial_arrears=arrears)


#: a dated contract aged past its first observation: the valuation (2026-01-01) sits between 2025-10-01 and 2026-04-01
DATES = [datetime(2025, 10, 1), datetime(2026, 4, 1), datetime(2026, 7, 1), datetime(2026, 10, 1), datetime(2027, 1, 4)]
#: the first REMAINING row knocks out at 100 under the coupon barrier 105 (a knock-out there can miss the coupon);
#: the later rows knock out at 110 above it (a coupon can pay without a knock-out)
DATED_KO = [103.0, 100.0, 110.0, 110.0, 110.0]


def _dated(pay_type, arrears=ARREARS, coupon_rate=0.03):
    records = [ObservationRecord(observation_date=d, barrier=b) for d, b in zip(DATES, DATED_KO)]
    return PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0,
        initial_date=datetime(2025, 7, 1), exercise_date=DATES[-1],
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=records),
            ki_barrier=70.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=105.0, coupon_rate=coupon_rate, coupon_pay_type=pay_type,
            day_count_convention=DayCountConvention.ACT_365, memory_coupon=True, initial_coupon_arrears=arrears),
        payoff_config=PayoffConfig(include_principal=True),
        settlement_convention=SettlementConvention(lag=0.1, lag_unit=SettlementLagUnit.YEAR_FRACTION),
    )


def _contracts():
    """(id, builder(arrears, coupon_rate), number of remaining rows) for every AC28 contract."""
    out = []
    for pay in (CouponPayType.INSTANT, CouponPayType.EXPIRY):
        tag = pay.name.lower()
        for first in (85.0, 108.0):
            out.append((f"fresh-{tag}-{first:g}",
                        lambda a, c, pay=pay, first=first: _fresh(pay, first, a, c), len(OBS)))
            out.append((f"now-{tag}-{first:g}",
                        lambda a, c, pay=pay, first=first: _now(pay, first, a, c), len(OBS_NOW)))
        out.append((f"dated-{tag}", lambda a, c, pay=pay: _dated(pay, a, c), len(DATES) - 1))
    return out


CONTRACTS = _contracts()
CONTRACT_IDS = [c[0] for c in CONTRACTS]


def _v1(grid_points=GRID_POINTS):
    return PhoenixQuadEngine(params=QuadParams(grid_points=grid_points))


def _coupon_cash(stats, n_rows):
    cash = np.asarray(stats.expected_discounted_coupon_cashflow, dtype=float)
    assert cash.shape == (n_rows,), "no per-row coupon cash"
    return cash


# --- the defect itself -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("contract", CONTRACTS, ids=CONTRACT_IDS)
def test_v2_arrears_are_worth_something(contract):
    """Arrears owed now are released with positive probability on every contract here: the price must move."""
    _, build, _ = contract
    env = _env()
    owed = PhoenixQuadEngineV2().price(build(ARREARS, 0.03), env)
    clean = PhoenixQuadEngineV2().price(build(0.0, 0.03), env)
    assert owed - clean > 0.1 * ARREARS, f"V2 arrears worth {owed - clean:+.6e}"


# --- gate 1: QUAD V1 ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("contract", CONTRACTS, ids=CONTRACT_IDS)
def test_v2_matches_v1_with_arrears(contract):
    _, build, n_rows = contract
    env = _env()
    product = build(ARREARS, 0.03)
    v1, v2 = _v1(), PhoenixQuadEngineV2()
    pv_gap = v2.price(product, env) - v1.price(product, env)
    assert abs(pv_gap) <= V1_PV_ALLOWANCE, f"V2 - V1 price {pv_gap:+.3e}"
    gap = _coupon_cash(v2.calculate_event_stats(product, env), n_rows) - _coupon_cash(
        v1.calculate_event_stats(product, env), n_rows)
    assert np.max(np.abs(gap)) <= V1_COUPON_ALLOWANCE, f"V2 - V1 per row: {gap}"


@pytest.mark.parametrize("first_barrier, coupon_now", [(85.0, True), (108.0, False)], ids=["paid-now", "missed-now"])
@pytest.mark.parametrize("pay_type", PAY_TYPES)
def test_v2_knock_out_at_the_valuation_instant_releases_the_arrears(pay_type, first_barrier, coupon_now):
    """Spot 105 above the knock-out barrier at an observation on the valuation instant: the note ends now and pays its
    principal, the (empty-period, zero) coupon where triggered and the arrears, at the knock-out settlement."""
    env = _env()
    env.spot_quote.spot = 105.0
    product = _now(pay_type, first_barrier)
    decision = product.decide_observations_at_valuation(105.0, env)
    assert decision.knocked_out and decision.coupon == pytest.approx(ARREARS, abs=1e-12)
    expected = (100.0 + ARREARS) * env.get_discount_factor(0.1)
    assert PhoenixQuadEngineV2().price(product, env) == pytest.approx(expected, abs=1e-12)
    assert _v1().price(product, env) == pytest.approx(expected, abs=1e-12)


# --- gate 2: PSEUDO Monte Carlo ----------------------------------------------------------------------------------------

MC_CONTRACTS = [c for c in CONTRACTS if not c[0].startswith("now-")]


@pytest.fixture(scope="module")
def mc_seeds():
    """Price and per-row coupon cash of ``N_SEEDS`` independent PSEUDO runs (distinct seeds), per contract."""
    env = _env()
    out = {}
    for name, build, n_rows in MC_CONTRACTS:
        product = build(ARREARS, 0.03)
        prices, rows = [], []
        for seed in range(N_SEEDS):
            engine = PhoenixMCEngine(
                params=MCParams(seed=2027 + 7919 * seed, num_paths=PATHS_PER_SEED, use_qmc=False))
            stats = engine.calculate_event_stats(product, env)
            prices.append(stats.pv)
            rows.append(_coupon_cash(stats, n_rows))
        out[name] = (np.asarray(prices), np.asarray(rows))
    return out


@pytest.mark.parametrize("contract", MC_CONTRACTS, ids=[c[0] for c in MC_CONTRACTS])
def test_v2_matches_pseudo_mc_seeds_with_arrears(contract, mc_seeds):
    name, build, n_rows = contract
    env = _env()
    product = build(ARREARS, 0.03)
    engine = PhoenixQuadEngineV2()
    prices, rows = mc_seeds[name]

    def within(v2, seeds, allowance, what):
        mean = seeds.mean(axis=0)
        se = seeds.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
        assert np.all(np.abs(v2 - mean) <= SE_MULTIPLE * se + allowance), (
            f"{what}: V2 {v2}\nMC {mean}\nSE {se}\n|V2-MC|/SE {np.abs(v2 - mean) / se}")

    within(engine.price(product, env), prices, V1_PV_ALLOWANCE, "price")
    within(_coupon_cash(engine.calculate_event_stats(product, env), n_rows), rows, V1_COUPON_ALLOWANCE, "coupon cash")


# --- V2's own identities -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("contract", CONTRACTS, ids=CONTRACT_IDS)
def test_v2_coupon_cash_closes_the_price_with_arrears(contract):
    _, build, n_rows = contract
    env = _env()
    engine = PhoenixQuadEngineV2()
    product = build(ARREARS, 0.03)
    stats = engine.calculate_event_stats(product, env)
    assert abs(stats.reconciliation_error) <= ROUND_OFF, f"ledger reconciliation {stats.reconciliation_error:+.3e}"
    cash = float(np.sum(_coupon_cash(stats, n_rows)))
    components = engine.price_components(product, env)
    assert cash == pytest.approx(components["coupon"], abs=ROUND_OFF)
    implied = engine.price(product, env) - engine.price(build(0.0, 0.0), env)
    assert abs(cash - implied) <= ROUND_OFF, f"closing residual {cash - implied:+.3e}"


# --- the two carriers of the arrears -----------------------------------------------------------------------------------


def _aged_fixed(arrears=0.0):
    """One observation left, 3.0 a period at a fixed quarter fraction (``test_quad_v2_terms_lifecycle``'s contract)."""
    return PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=0.5,
        barrier_config=BarrierConfig(ko_barrier=200.0, ko_rate=0.0, ko_observation_dates=[0.5], ki_barrier=None),
        coupon_config=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, memory_coupon=True,
                                          fixed_coupon_year_fraction=0.25, initial_coupon_arrears=arrears),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )


def test_the_amount_alone_pays_what_the_count_alone_pays():
    """Two periods missed: the lifecycle state's count (2) and the contract's amount (6.0) are the same debt."""
    env = _env()
    by_count = PhoenixQuadEngineV2().price(_aged_fixed(), env,
                                           lifecycle_state=AutocallableLifecycleState(coupon_memory_count=2))
    by_amount = PhoenixQuadEngineV2().price(_aged_fixed(6.0), env)
    assert by_amount == pytest.approx(by_count, abs=ROUND_OFF)


def test_a_count_and_the_same_amount_are_one_debt_not_two():
    """A route carrying both (the intraday twin hands QUAD V2 its lifecycle state too) prices the count, bitwise."""
    env = _env()
    state = AutocallableLifecycleState(coupon_memory_count=2)
    by_count = PhoenixQuadEngineV2().price(_aged_fixed(), env, lifecycle_state=state)
    both = PhoenixQuadEngineV2().price(_aged_fixed(6.0), env, lifecycle_state=state)
    assert both == by_count


def test_a_count_and_a_different_amount_refuse():
    env = _env()
    state = AutocallableLifecycleState(coupon_memory_count=2)
    with pytest.raises(ValidationError, match="initial_coupon_arrears"):
        PhoenixQuadEngineV2().price(_aged_fixed(3.0), env, lifecycle_state=state)

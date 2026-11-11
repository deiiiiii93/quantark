"""QUAD V1 Phoenix coupon cash: memory arrears counted, EXPIRY coupons discounted at the note's end.

Two defects of ``PhoenixQuadEngine.calculate_event_stats`` (unified event statistics spec, DESIGN §3, AC3):

* under memory no coupon cash was computed at all (the amount paid depends on the arrears, and a
  knock-out releases the arrears even when the current coupon barrier is missed);
* under EXPIRY pay every coupon was discounted at the terminal payment, even on paths that knock out
  early and pay the rolled-up coupons at the knock-out settlement.

Gates (AC3):

* QUAD V2, an independent deterministic producer that already computes memory and deferred coupons:
  per-row coupon cash within an allowance measured by refining V1's grid (the V1-V2 gap shrinks
  under refinement; the refinement table is in the T4 report);
* a PSEUDO Monte Carlo run of ``N_BATCHES`` independent seeds: per row within 4 batch standard
  errors plus that allowance;
* V1's own closing identity: the coupon cash sums to ``price(c) - price(c = 0)`` (the price is linear
  in the coupon amounts, and the zero-coupon note has the same grid and the same events) -- to
  round-off under INSTANT, to a measured second-order residual under EXPIRY (``CLOSING_RESIDUAL``).

The fixture is the T2 MC one: the second observation's coupon barrier sits ABOVE the knock-out
barrier, so a path can knock out there while missing the coupon (memory releases arrears only), and a
0.1-year settlement lag separates every knock-out settlement from the terminal payment.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.quad import PhoenixQuadEngineV2
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option import PhoenixOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType

OBS = [0.25, 0.5, 0.75, 1.0]
KO_BARRIER = 103.0
COUPON_BARRIERS = [85.0, 108.0, 85.0, 85.0]

# V1 grid the gates run at, and the V1-V2 per-row allowance banked at that grid. Refinement table
# (T4 report; max |V1 - V2| over the four products x four rows; V2's own refinement moves it < 2e-13):
#   grid  1001 7.29e-5 | 2001 1.59e-5 | 4001 5.07e-6 | 8001 9.55e-7 | 16001 1.81e-7   (second order)
# Banked: the 4001 gap times 2 (deterministic; the factor covers FFT round-off across machines and
# library versions, orders of magnitude below it, without hiding a regression of the gap's size).
GRID_POINTS = 4001
V2_ALLOWANCE = 1.0e-5
# V1 closing residual sum(coupon cash) - (price(c) - price(c=0)) at GRID_POINTS. INSTANT: round-off.
# EXPIRY: price() moves its termination-value surface W at a knock-out by the breached-fraction
# weight, while the statistics project W jointly with every other column (as the value surfaces are
# projected); the difference is second order (1001 -2.16e-7 | 2001 -5.40e-8 | 4001 -1.35e-8 |
# 8001 -3.37e-9 | 16001 -8.43e-10) and vanishes to round-off when the statistics use the price's rule
# (T4 report). Banked: the 4001 residual times 2.
CLOSING_RESIDUAL = {CouponPayType.INSTANT: 1.0e-12, CouponPayType.EXPIRY: 2.7e-8}

N_BATCHES = 8
PATHS_PER_BATCH = 20_000
SE_MULTIPLE = 4.0

CASES = [
    pytest.param(False, CouponPayType.INSTANT, id="plain-instant"),
    pytest.param(False, CouponPayType.EXPIRY, id="plain-expiry"),
    pytest.param(True, CouponPayType.INSTANT, id="memory-instant"),
    pytest.param(True, CouponPayType.EXPIRY, id="memory-expiry"),
]


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


def _phoenix(
    *,
    memory: bool,
    pay_type: CouponPayType,
    coupon_rate: float = 0.03,
    obs=OBS,
    coupon_barriers=COUPON_BARRIERS,
    initial_arrears: float = 0.0,
) -> PhoenixOption:
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=KO_BARRIER,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=list(obs),
            ki_barrier=70.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=list(coupon_barriers),
            coupon_rate=coupon_rate,
            coupon_pay_type=pay_type,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=memory,
            initial_coupon_arrears=initial_arrears,
        ),
        payoff_config=PayoffConfig(include_principal=True),
        maturity=1.0,
        settlement_convention=SettlementConvention(
            lag=0.1, lag_unit=SettlementLagUnit.YEAR_FRACTION
        ),
    )


def _v1(grid_points: int = GRID_POINTS) -> PhoenixQuadEngine:
    return PhoenixQuadEngine(params=QuadParams(grid_points=grid_points))


def _coupon_cash(stats, n_rows: int = len(OBS)) -> np.ndarray:
    cash = np.asarray(stats.expected_discounted_coupon_cashflow, dtype=float)
    assert cash.shape == (n_rows,), "no per-row coupon cash"
    return cash


@pytest.mark.parametrize("memory, pay_type", CASES)
def test_v1_coupon_cash_matches_quad_v2(memory, pay_type):
    env = _env()
    product = _phoenix(memory=memory, pay_type=pay_type)
    v1 = _v1().calculate_event_stats(product, env)
    v2 = PhoenixQuadEngineV2().calculate_event_stats(product, env)
    assert isinstance(v1, PhoenixEventStats)
    gap = _coupon_cash(v1) - _coupon_cash(v2)
    assert np.max(np.abs(gap)) <= V2_ALLOWANCE, f"V1 - V2 per row: {gap}"


@pytest.fixture(scope="module")
def mc_batches():
    """Per-row coupon cash of ``N_BATCHES`` independent PSEUDO runs (distinct seeds), per case."""
    env = _env()
    out = {}
    for memory in (False, True):
        for pay_type in (CouponPayType.INSTANT, CouponPayType.EXPIRY):
            product = _phoenix(memory=memory, pay_type=pay_type)
            rows = []
            for seed in range(N_BATCHES):
                engine = PhoenixMCEngine(
                    params=MCParams(seed=1009 + 7919 * seed, num_paths=PATHS_PER_BATCH, use_qmc=False)
                )
                rows.append(_coupon_cash(engine.calculate_event_stats(product, env)))
            out[(memory, pay_type)] = np.asarray(rows)
    return out


@pytest.mark.parametrize("memory, pay_type", CASES)
def test_v1_coupon_cash_matches_pseudo_mc_batches(memory, pay_type, mc_batches):
    env = _env()
    product = _phoenix(memory=memory, pay_type=pay_type)
    v1 = _coupon_cash(_v1().calculate_event_stats(product, env))
    batches = mc_batches[(memory, pay_type)]
    mean = batches.mean(axis=0)
    se = batches.std(axis=0, ddof=1) / np.sqrt(N_BATCHES)
    bound = SE_MULTIPLE * se + V2_ALLOWANCE
    assert np.all(np.abs(v1 - mean) <= bound), (
        f"V1 {v1}\nMC {mean}\nSE {se}\n|V1-MC|/SE {np.abs(v1 - mean) / se}"
    )


@pytest.mark.parametrize("memory, pay_type", CASES)
def test_v1_coupon_cash_closes_the_price(memory, pay_type):
    """sum(coupon cash) == price(c) - price(c=0): the price is linear in the coupon amounts."""
    env = _env()
    engine = _v1()
    product = _phoenix(memory=memory, pay_type=pay_type)
    zero = _phoenix(memory=memory, pay_type=pay_type, coupon_rate=0.0)
    stats = engine.calculate_event_stats(product, env)
    implied = engine.price(product, env) - engine.price(zero, env)
    residual = float(np.sum(_coupon_cash(stats))) - implied
    assert abs(residual) <= CLOSING_RESIDUAL[pay_type], f"closing residual {residual:+.3e}"


def test_memory_knock_out_on_a_missed_coupon_releases_the_arrears():
    """Row 1 (coupon barrier above the knock-out barrier) carries released arrears under memory."""
    env = _env()
    plain = _coupon_cash(_v1().calculate_event_stats(
        _phoenix(memory=False, pay_type=CouponPayType.INSTANT), env))
    memory = _coupon_cash(_v1().calculate_event_stats(
        _phoenix(memory=True, pay_type=CouponPayType.INSTANT), env))
    # the first row has no arrears to release; later rows pay strictly more under memory
    assert memory[0] == pytest.approx(plain[0], rel=1e-12)
    assert np.all(memory[1:] > plain[1:])


def test_expiry_coupon_paid_with_an_early_knock_out_is_discounted_at_its_settlement():
    """EXPIRY coupons on knocking-out paths settle with the knock-out: worth more than at maturity."""
    env = _env()
    instant = _coupon_cash(_v1().calculate_event_stats(
        _phoenix(memory=False, pay_type=CouponPayType.INSTANT), env))
    product = _phoenix(memory=False, pay_type=CouponPayType.EXPIRY)
    stats = _v1().calculate_event_stats(product, env)
    expiry = _coupon_cash(stats)
    fractions = product.get_coupon_period_year_fractions(env)
    amounts = np.array([product.get_coupon_payoff(k, year_fraction=f) for k, f in enumerate(fractions)])
    # every coupon paid at the terminal payment (maturity 1.0 + the 0.1-year lag)
    at_terminal = np.asarray(stats.coupon_probability) * amounts * env.get_discount_factor(1.1)
    # Rows 0 and 2: deferred, so worth less than paid at the observation; settled early on knock-out
    # paths, so worth more than paid at the terminal payment.
    for k in (0, 2):
        assert at_terminal[k] + 1e-6 < expiry[k] < instant[k] - 1e-6, f"row {k}"
    # Row 1's coupon barrier is above the knock-out barrier: it is only ever paid with a knock-out, at
    # that settlement under either pay type. The last row settles at the terminal payment either way.
    assert expiry[1] == pytest.approx(instant[1], rel=1e-9)
    assert expiry[-1] == pytest.approx(at_terminal[-1], rel=1e-9)


# An observation AT the valuation instant (decided on the spot, below the knock-out barrier), then the
# fixture's four. Its coupon barrier decides whether the coupon is paid now (85) or missed (108).
OBS_NOW = [0.0, *OBS]


@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY], ids=["instant", "expiry"])
@pytest.mark.parametrize("first_barrier", [85.0, 108.0], ids=["paid-now", "missed-now"])
def test_valuation_instant_observation_matches_quad_v2(first_barrier, pay_type):
    env = _env()
    product = _phoenix(memory=False, pay_type=pay_type, obs=OBS_NOW,
                       coupon_barriers=[first_barrier, *COUPON_BARRIERS])
    v1 = _v1().calculate_event_stats(product, env)
    v2 = PhoenixQuadEngineV2().calculate_event_stats(product, env)
    gap = _coupon_cash(v1, 5) - _coupon_cash(v2, 5)
    assert np.max(np.abs(gap)) <= V2_ALLOWANCE, f"V1 - V2 per row: {gap}"
    # the coupon decided now is paid with certainty (its amount is zero here: an empty period)
    assert v1.coupon_probability[0] == v2.coupon_probability[0] == (1.0 if first_barrier == 85.0 else 0.0)


@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY], ids=["instant", "expiry"])
@pytest.mark.parametrize("first_barrier", [85.0, 108.0], ids=["paid-now", "missed-now"])
def test_memory_arrears_carried_into_the_valuation_instant(first_barrier, pay_type):
    """Initial arrears: paid now with the decided coupon, else carried into the later rows' release.

    Gated by V1's own closing identity and, since QUAD V2 reads ``initial_coupon_arrears`` (T4b, spec D24), by V2
    per row (``test_phoenix_quad_v2_arrears.py`` gates V2's price on arrears contracts against V1 and MC).
    """
    env = _env()
    arrears = 0.02
    product = _phoenix(memory=True, pay_type=pay_type, obs=OBS_NOW,
                       coupon_barriers=[first_barrier, *COUPON_BARRIERS], initial_arrears=arrears)
    zero = _phoenix(memory=True, pay_type=pay_type, obs=OBS_NOW, coupon_rate=0.0,
                    coupon_barriers=[first_barrier, *COUPON_BARRIERS])
    engine = _v1()
    stats = engine.calculate_event_stats(product, env)
    cash = _coupon_cash(stats, 5)
    residual = float(np.sum(cash)) - (engine.price(product, env) - engine.price(zero, env))
    assert abs(residual) <= CLOSING_RESIDUAL[pay_type], f"closing residual {residual:+.3e}"
    gap = cash - _coupon_cash(PhoenixQuadEngineV2().calculate_event_stats(product, env), 5)
    assert np.max(np.abs(gap)) <= V2_ALLOWANCE, f"V1 - V2 per row: {gap}"
    if first_barrier == 85.0:
        # released now with the (empty-period) coupon: settled 0.1 year later, or at the note's end
        assert stats.coupon_probability[0] == 1.0
        if pay_type == CouponPayType.INSTANT:
            assert cash[0] == pytest.approx(arrears * env.get_discount_factor(0.1), rel=1e-12)
        else:
            assert arrears * env.get_discount_factor(1.1) < cash[0] < arrears * env.get_discount_factor(0.35)
    else:
        assert cash[0] == 0.0 and stats.coupon_probability[0] == 0.0

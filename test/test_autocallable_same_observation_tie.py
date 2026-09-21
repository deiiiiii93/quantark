"""A knock-out and a knock-in observed at the SAME instant: the knock-out always wins (desk ruling, 2026-09-21).

The knock-in observed with it neither disables that knock-out (``disable_ko_after_ki``) nor, on a knock-out-reset
snowball, replaces the schedule it belongs to. One spot breaches both only when the knock-out level is at or below the
knock-in level, so the contracts here put it at 60 against a knock-in at 75: at EVERY observation the contract either
knocks out (S >= 60) or knocks in (S < 60), and a spot in [60, 75] is the tie.

Before the ruling the engines disagreed (spot 70, monthly observations): the snowball under ``disable_ko_after_ki`` was
-22.4 on Monte Carlo, PDE and QUAD V1 against 14.0 on QUAD V2; the Phoenix 82.1 on Monte Carlo and both QUAD engines
against 100.2 on PDE; the knock-out-reset snowball -21.6 against QUAD V2's 14.0. With knock-outs left enabled every
engine already let the knock-out win.

The reference is independent of all of them: two observations (0.5y and 1y), so the value is one integral over the
first fixing of Black-Scholes closed forms for the second.
"""
import math
from dataclasses import replace
from datetime import datetime

import pytest
from scipy import integrate
from scipy.stats import norm

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import ObservationType

SPOT, RATE, DIV, VOL = 70.0, 0.03, 0.01, 0.25
T1, T2 = 0.5, 1.0
KO, KI, POST_KO, COUPON_LEVEL = 60.0, 75.0, 95.0, 50.0
KO_CASH, POST_CASH, COUPON, STRIKE = 15.0, 3.0, 1.0, 100.0


def env():
    return PricingEnvironment(rate_curve=FlatRateCurve(RATE), valuation_date=datetime(2026, 3, 16), spot_quote=SpotQuote(SPOT),
                              vol_surface=FlatVolSurface(VOL), div_yield=ContinuousDividendYield(DIV))


def _schedule(level, times):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def _barrier(disable, ko_rate, times):
    return BarrierConfig(ko_barrier=KO, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
                         ko_observation_schedule=_schedule(KO, times), ki_barrier=KI, ki_observation_type=ObservationType.DISCRETE,
                         ki_observation_schedule=_schedule(KI, times), disable_ko_after_ki=disable)


def snowball(disable, times=(T1, T2)):
    return SnowballOption(initial_price=100.0, strike=STRIKE, maturity=T2, barrier_config=_barrier(disable, 0.15, times),
                          accrual_config=AccrualConfig(is_annualized=False),
                          payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))


def phoenix(disable, times=(T1, T2)):
    return PhoenixOption(initial_price=100.0, strike=STRIKE, maturity=T2, barrier_config=_barrier(disable, 0.0, times),
                         coupon_config=CouponBarrierConfig(coupon_barrier=COUPON_LEVEL, coupon_rate=0.01, memory_coupon=False),
                         accrual_config=AccrualConfig(is_annualized=False, is_annualized_coupon=False),
                         payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


def ko_reset(disable, times=(T1, T2)):
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=STRIKE, maturity=T2, barrier_config=_barrier(disable, 0.15, times),
        post_barrier_config=BarrierConfig(ko_barrier=POST_KO, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(POST_KO, times)),
        accrual_config=AccrualConfig(is_annualized=False), payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))


# ---------------------------------------------------------------------------------------------------------------------
# the independent reference
def _second_fixing(s1):
    """(P(S2 >= k), E[S2 1{S2 < k}]) given the first fixing, as a function of the level k."""
    forward, sd = s1 * math.exp((RATE - DIV) * (T2 - T1)), VOL * math.sqrt(T2 - T1)

    def terms(k):
        d1 = (math.log(forward / k) + 0.5 * sd * sd) / sd
        return norm.cdf(d1 - sd), forward * norm.cdf(-d1)
    return terms


def _after_a_knock_in(product, disable):
    """Undiscounted value at the second fixing of a contract knocked in (and not out) at the first, given S1."""
    def value(s1):
        terms = _second_fixing(s1)

        def below(k):                                   # E[(S2 - 100) 1{S2 < k}], the knocked-in loss below k <= 100
            alive, partial = terms(k)
            return partial - STRIKE * (1.0 - alive)
        if product == "phoenix":
            coupon = COUPON * terms(COUPON_LEVEL)[0]
            if disable:
                return coupon + 100.0 + below(STRIKE)
            out = terms(KO)[0]                          # knocked out at 60: principal + coupon; below it: S2 (+ coupon)
            return (100.0 + COUPON) * out + (100.0 * (1.0 - out) + below(KO)) + COUPON * (terms(COUPON_LEVEL)[0] - out)
        if disable:
            return below(STRIKE)                        # no knock-out is left, on either product
        level, cash = (POST_KO, POST_CASH) if product == "ko_reset" else (KO, KO_CASH)
        return cash * terms(level)[0] + below(level)    # the schedule in force after the knock-in
    return value


def reference(product, disable, knock_out_from=KO):
    """The knock-out wins the tie: S1 >= 60 knocks out, whatever the knock-in level says about [60, 75].
    ``knock_out_from=KI`` is the other rule (the knock-in wins, so only S1 > 75 knocks out), kept to show what the
    engines used to price."""
    df1, df2 = math.exp(-RATE * T1), math.exp(-RATE * T2)
    mean, sd = math.log(SPOT) + (RATE - DIV - 0.5 * VOL * VOL) * T1, VOL * math.sqrt(T1)
    p_out = 1.0 - norm.cdf((math.log(knock_out_from) - mean) / sd)
    knocked_out = (100.0 + COUPON if product == "phoenix" else KO_CASH) * df1 * p_out
    inner = _after_a_knock_in(product, disable)

    def integrand(x):
        s1 = math.exp(x)
        first_coupon = COUPON * df1 if product == "phoenix" and s1 >= COUPON_LEVEL else 0.0
        return norm.pdf(x, mean, sd) * (first_coupon + df2 * inner(s1))
    knocked_in, _ = integrate.quad(integrand, mean - 12.0 * sd, math.log(knock_out_from),
                                  points=[math.log(COUPON_LEVEL)], limit=400)
    return knocked_out + knocked_in


# ---------------------------------------------------------------------------------------------------------------------
_mc = lambda cls: (lambda: cls(params=MCParams(seed=7, num_paths=131072)))                                    # noqa: E731
_quad = lambda cls: (lambda: cls(params=QuadParams(grid_points=2001)))                                        # noqa: E731
FAMILIES = {
    "snowball": (snowball, {"mc": _mc(SnowballMCEngine), "pde": lambda: SnowballPDESolver(PDEParams()),
                            "quad_v1": _quad(SnowballQuadEngine), "quad_v2": lambda: SnowballQuadEngineV2()}),
    "phoenix": (phoenix, {"mc": _mc(PhoenixMCEngine), "pde": lambda: PhoenixPDESolver(PDEParams()),
                          "quad_v1": _quad(PhoenixQuadEngine), "quad_v2": lambda: PhoenixQuadEngineV2()}),
    "ko_reset": (ko_reset, {"mc": _mc(SnowballMCEngine), "pde": lambda: KOResetSnowballPDESolver(PDEParams()),
                            "quad_v1": _quad(KOResetSnowballQuadEngine), "quad_v2": lambda: KOResetSnowballQuadEngineV2()}),
}


@pytest.mark.parametrize("engine", ["mc", "pde", "quad_v1", "quad_v2"])
@pytest.mark.parametrize("disable", [False, True], ids=["ko_stays_enabled", "disable_ko_after_ki"])
@pytest.mark.parametrize("product", sorted(FAMILIES))
def test_the_knock_out_wins_a_tie_with_a_knock_in(product, disable, engine):
    build, engines = FAMILIES[product]
    solver = engines[engine]()
    value = float(solver.price(build(disable), env()))
    tolerance = 4.0 * float(solver.get_last_std_error()) + 5e-3 if engine == "mc" else 1e-2
    assert value == pytest.approx(reference(product, disable), abs=tolerance)


def terminal_reference(product):
    """One observation, at maturity: the tie sits in the terminal condition. S >= 60 knocks out; below it the contract
    knocks in and redeems at the loss (a Phoenix pays its coupon down to 50 on the way)."""
    df = math.exp(-RATE * T2)
    forward, sd = SPOT * math.exp((RATE - DIV) * T2), VOL * math.sqrt(T2)

    def terms(k):
        d1 = (math.log(forward / k) + 0.5 * sd * sd) / sd
        return norm.cdf(d1 - sd), forward * norm.cdf(-d1)
    out, partial = terms(KO)
    if product == "phoenix":
        return df * ((100.0 + COUPON) * out + partial + COUPON * (terms(COUPON_LEVEL)[0] - out))
    return df * (KO_CASH * out + partial - STRIKE * (1.0 - out))


@pytest.mark.parametrize("engine", ["mc", "pde", "quad_v1", "quad_v2"])
@pytest.mark.parametrize("disable", [False, True], ids=["ko_stays_enabled", "disable_ko_after_ki"])
@pytest.mark.parametrize("product", sorted(FAMILIES))
def test_the_knock_out_wins_a_tie_at_maturity(product, disable, engine):
    build, engines = FAMILIES[product]
    solver = engines[engine]()
    value = float(solver.price(build(disable, times=(T2,)), env()))
    tolerance = 4.0 * float(solver.get_last_std_error()) + 5e-3 if engine == "mc" else 1e-2
    assert value == pytest.approx(terminal_reference(product), abs=tolerance)


@pytest.mark.parametrize("engine", ["mc", "pde", "quad_v1", "quad_v2"])
@pytest.mark.parametrize("disable", [False, True], ids=["ko_stays_enabled", "disable_ko_after_ki"])
@pytest.mark.parametrize("product", sorted(FAMILIES))
def test_the_knock_out_wins_a_tie_at_the_valuation_instant(product, disable, engine):
    """An observation now, on a known spot of 70: knocked out, for the knock-out cash, with nothing to simulate."""
    build, engines = FAMILIES[product]
    value = float(engines[engine]().price(build(disable, times=(0.0, T2)), env()))
    assert value == pytest.approx(100.0 + COUPON if product == "phoenix" else KO_CASH, abs=1e-9)


@pytest.mark.parametrize("product, disable", [("snowball", True), ("phoenix", True), ("ko_reset", False), ("ko_reset", True)])
def test_the_reference_tells_the_two_rules_apart(product, disable):
    """Where the engines were split, the two rules are far apart: a first fixing in [60, 75] either collects the
    knock-out cash or carries the knocked-in loss."""
    assert reference(product, disable) - reference(product, disable, knock_out_from=KI) > 5.0

"""DoubleBarrierOptionAnalyticalEngine's rebate terms and EXPIRY branch (spec 3C §5.8, §7.2 rows 'analytic double
barrier, rebate' and 'EXPIRY')."""
import json
import math
from pathlib import Path

import pytest

from pde_routes.barrier_reference import knock_out
from pde_routes.cases import CONST, black_scholes, env_const, obs_dates
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.product.option import DoubleBarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.util.enum import DoubleBarrierType, ObservationType, OptionType

HERE = Path(__file__).parent
NT = {tuple(c["args"]): c["reference"] for g in ("engine", "reviewer")
      for c in json.loads((HERE / "no_touch_reference.json").read_text())["cells"][g]}
CAPTURE = json.loads((HERE / "analytic_capture.json").read_text())
VOL, R, Q = CONST["vol"], CONST["rate"], CONST["div"]
KO, KI = DoubleBarrierType.KNOCK_OUT, DoubleBarrierType.KNOCK_IN


def product(kind, call, rebate, obs="cont"):
    kw = {}
    if obs == "cont":
        kw["observation_type"] = ObservationType.CONTINUOUS
    elif obs == "expiry":
        kw["observation_type"] = ObservationType.EXPIRY
    else:
        kw.update(observation_type=ObservationType.DISCRETE, observation_schedule=ObservationSchedule.from_legacy(
            observation_dates=obs_dates("monthly"), default_barrier=None, default_payoff=rebate,
            upper_barrier=130.0, lower_barrier=80.0))
    return DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL if call else OptionType.PUT,
                               upper_barrier=130.0, lower_barrier=80.0, barrier_type=kind, maturity=1.0,
                               rebate=rebate, **kw)


def price(p, env):
    return float(DoubleBarrierOptionAnalyticalEngine().price(p, env))


@pytest.mark.parametrize("kind", [KO, KI])
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("spot", [100.0, 90.0, 120.0])
def test_continuous_rebate_terms(kind, call, spot):
    env = env_const(spot)
    free = price(product(kind, call, 0.0), env)
    p_inside = NT[(spot, 1.0, R, Q, VOL, 80.0, 130.0)]
    leg = 5.0 * math.exp(-R) * ((1.0 - p_inside) if kind is KO else p_inside)
    assert abs(price(product(kind, call, 5.0), env) - (free + leg)) <= 1e-10        # measured 5e-15


def test_review_cells():
    """vol 1 %, one day, spot 129.9, zero log drift (review 2); vol 5e-10 with the forward 8 deviations inside
    130 (review 3): the rebate terms read the resolved probability."""
    for (S, T, r, q, vol, L, U), ref in NT.items():
        if S in (100.0, 90.0, 120.0) and vol == VOL:
            continue
        env = env_const(S, vol=vol, rate=r, div=q)
        for kind in (KO, KI):
            p5 = DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=U, lower_barrier=L,
                                     barrier_type=kind, maturity=T, rebate=5.0,
                                     observation_type=ObservationType.CONTINUOUS)
            p0 = DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=U, lower_barrier=L,
                                     barrier_type=kind, maturity=T, rebate=0.0,
                                     observation_type=ObservationType.CONTINUOUS)
            leg = 5.0 * math.exp(-r * T) * ((1.0 - ref) if kind is KO else ref)
            assert abs(price(p5, env) - (price(p0, env) + leg)) <= 1e-10, (S, vol, kind)


@pytest.mark.parametrize("kind", [KO, KI])
def test_monthly_rebate_against_the_exact_recursion(kind):
    """Bound 0.2 separates a right rebate term (barrier shift error 0.093) from today's (1.8 or more)."""
    obs = obs_dates("monthly")
    for call in (True, False):
        for spot in (100.0, 90.0, 120.0):
            env = env_const(spot)
            payoff = "call" if call else "put"
            ko = knock_out(payoff=payoff, strike=100.0, lower=80.0, upper=130.0, cash=5.0 if kind is KO else 0.0,
                           at_hit=False, obs=obs, spots=[spot], vol=VOL, r=R, q=Q)[0][spot][0]
            if kind is KO:
                exact = ko
            else:
                surv = knock_out(payoff="one", strike=100.0, lower=80.0, upper=130.0, cash=0.0, at_hit=False,
                                 obs=obs, spots=[spot], vol=VOL, r=R, q=Q)[0][spot][0]
                exact = black_scholes(call, spot, 100.0, 1.0)[0] - ko + 5.0 * surv
            assert abs(price(product(kind, call, 5.0, "monthly"), env) - exact) <= 0.2


def test_rebate_free_and_expiry_inside_are_bit_identical():
    for tag, value in CAPTURE.items():
        if not tag.startswith("dbl|"):
            continue
        _, kind, call, rebate, obs, spot = tag.split("|")
        p = product(KO if kind == "KO" else KI, call == "call", float(rebate), obs)
        assert price(p, env_const(float(spot))).hex() == value, tag


@pytest.mark.parametrize("kind", [KO, KI])
@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("spot", [100.0, 135.0, 75.0])
def test_expiry_from_any_spot(kind, call, spot):
    env = env_const(spot)
    payoff = "call" if call else "put"
    ko = knock_out(payoff=payoff, strike=100.0, lower=80.0, upper=130.0, cash=5.0 if kind is KO else 0.0,
                   at_hit=False, obs=[1.0], spots=[spot], vol=VOL, r=R, q=Q)[0][spot][0]
    if kind is KO:
        exact = ko
    else:
        inside = knock_out(payoff="one", strike=100.0, lower=80.0, upper=130.0, cash=0.0, at_hit=False, obs=[1.0],
                           spots=[spot], vol=VOL, r=R, q=Q)[0][spot][0]
        exact = black_scholes(call, spot, 100.0, 1.0)[0] - ko + 5.0 * inside
    assert abs(price(product(kind, call, 5.0, "expiry"), env) - exact) <= 1e-10     # measured 1.1e-14

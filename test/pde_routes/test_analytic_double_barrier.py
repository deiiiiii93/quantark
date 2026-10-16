"""DoubleBarrierOptionAnalyticalEngine's rebate terms and EXPIRY branch (spec 3C §5.8, §7.2 rows 'analytic double
barrier, rebate' and 'EXPIRY')."""
import json
import math
from pathlib import Path

import pytest

from golden_compare import assert_close
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
#: the monthly (DISCRETE, from inception) cells of the capture moved on 2026-10-15: with no record today the
#: engine prices through the first observation, the BGK-shifted formula only from it on (the capture holds
#: the BGK formula at today's spot); before -> after, QUAD (401 points) and the BS PDE for reference
MOVED = {
    # 5.976168698863631 -> (QUAD 6.1165, PDE 6.1157)
    "dbl|KI|call|0.0|monthly|100.0": 5.976253433879049,
    # 21.20169885696715 -> (QUAD 21.3178, PDE 21.3171)
    "dbl|KI|call|0.0|monthly|120.0": 21.223257708846234,
    # 2.42257067267494 -> (QUAD 2.5088, PDE 2.5089)
    "dbl|KI|call|0.0|monthly|90.0": 2.428538627067247,
    # 5.321582431146439 -> (QUAD 5.4269, PDE 5.4260)
    "dbl|KI|put|0.0|monthly|100.0": 5.321671865054488,
    # 1.1916870485536144 -> (QUAD 1.2376, PDE 1.2370)
    "dbl|KI|put|0.0|monthly|120.0": 1.1980049731971216,
    # 10.683824195445997 -> (QUAD 10.7789, PDE 10.7788)
    "dbl|KI|put|0.0|monthly|90.0": 10.69093882625982,
    # 3.6254079367833754 -> (QUAD 3.4863, PDE 3.4861)
    "dbl|KO|call|0.0|monthly|100.0": 3.625323201767981,
    # 2.8394367314048807 -> (QUAD 2.7240, PDE 2.7242)
    "dbl|KO|call|0.0|monthly|120.0": 2.817877879525789,
    # 2.3630275291479954 -> (QUAD 2.2770, PDE 2.2769)
    "dbl|KO|call|0.0|monthly|90.0": 2.3570595747556777,
    # 2.319564184434576 -> (QUAD 2.2154, PDE 2.2152)
    "dbl|KO|put|0.0|monthly|100.0": 2.319474750526517,
    # 1.0880218447690702 -> (QUAD 1.0428, PDE 1.0427)
    "dbl|KO|put|0.0|monthly|120.0": 1.0817039201255791,
    # 2.041842323802623 -> (QUAD 1.9470, PDE 1.9470)
    "dbl|KO|put|0.0|monthly|90.0": 2.0347276929888185,
}
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
        if tag in MOVED:
            # the vectorised projection: bitwise on the ARM64 machine that took them, the golden tolerance across
            # architectures (NumPy's SIMD exp / log / pow on x86)
            assert_close(price(p, env_const(float(spot))), MOVED[tag], msg=tag)
            continue
        assert_close(price(p, env_const(float(spot))), float.fromhex(value), msg=tag)


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

"""The 3A evidence's products and markets, for the tests and the byte book (spec 3A §10 step 0).

Copied verbatim from the evidence directory
``docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-evidence``:

  ``lab.py`` lines 48-60                       ENV, SNOWBALL, PHOENIX, KO_RESET, env
  ``e13_base_family_greeks.py`` lines 51, 56-125  HORIZONS, dates, _spots, _disc, PRODUCTS
  ``e18_phoenix_ko_reset_damping.py`` lines 33-70  CRASH, CALM, PHOENIX_VARIANTS, phoenix
                                                (every ``lab.`` prefix dropped)

The evidence ran its candidates by patching the engine at runtime; nothing of that machinery is
copied here. These tests price production code only.
"""
from __future__ import annotations

from pathlib import Path

from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption,
                                                  DoubleOneTouchOption, OneTouchOption)
from quantark.modelvalidation.builders import equity_phoenix as ph
from quantark.modelvalidation.builders.equity_snowball import make_environment
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.enum.option_enums import BarrierDirection, DoubleBarrierType, TouchType

REPO = Path(__file__).resolve().parents[2]

# --- lab.py: the certified studies' base specs ------------------------------------

ENV = dict(spot=100.0, vol=0.22, rate=0.025, div_yield=0.03)
SNOWBALL = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0,
                ko_rate=0.15, rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)
PHOENIX = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ko_rate=0.0, ki_barrier=75.0,
               coupon_barrier=85.0, coupon_rate=0.02, num_observations=12, memory_coupon=False,
               maturity=1.0, contract_multiplier=1.0)
KO_RESET = dict(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=2.0,
                pre_ko_barrier=103.0, pre_ko_rate=0.15, post_ko_barrier=95.0, post_ko_rate=0.03,
                ki_barrier=80.0, ki_continuous=True, contract_multiplier=1.0)


def env(**over):
    return make_environment(dict(ENV, **over))


# --- e13: the base-family discrete products ----------------------------------------

HORIZONS = {"1d": 1.0 / 252.0, "5d": 5.0 / 252.0, "1m": 1.0 / 12.0}


def dates(t1):
    return [t1 + k / 12.0 for k in range(12)]


# name -> (reference spec, product factory(dates), spots)
#   spec: payoff of a survivor at maturity, lower / upper absorbing levels, the cash paid on a
#   breach and whether it is paid at the breach observation or at maturity; ``knock_in`` = the
#   product is vanilla minus this knock-out.
def _spots(*thresholds):
    out = [100.0]
    for b, up in thresholds:
        out += [b * (0.97 if up else 1.03), b * (0.985 if up else 1.015)]
    return out


def _disc(d):
    return dict(observation_type=ObservationType.DISCRETE, observation_dates=d, maturity=d[-1])


PRODUCTS = {
    "uo_call": (
        dict(payoff="call", strike=100.0, upper=130.0),
        lambda d: BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                barrier_type=BarrierType.UP_OUT, **_disc(d)),
        _spots((130.0, True))),
    "uo_call_rebate_at_hit": (
        dict(payoff="call", strike=100.0, upper=130.0, cash=5.0, at_hit=True),
        lambda d: BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                barrier_type=BarrierType.UP_OUT, rebate=5.0, pay_at_hit=True, **_disc(d)),
        _spots((130.0, True))),
    "uo_call_rebate_at_expiry": (
        dict(payoff="call", strike=100.0, upper=130.0, cash=5.0, at_hit=False),
        lambda d: BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                barrier_type=BarrierType.UP_OUT, rebate=5.0, pay_at_hit=False, **_disc(d)),
        _spots((130.0, True))),
    "do_put_rebate_at_expiry": (
        dict(payoff="put", strike=100.0, lower=85.0, cash=3.0, at_hit=False),
        lambda d: BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=85.0,
                                barrier_type=BarrierType.DOWN_OUT, rebate=3.0, pay_at_hit=False, **_disc(d)),
        _spots((85.0, False))),
    "di_put": (
        dict(payoff="put", strike=100.0, lower=85.0, knock_in=True),
        lambda d: BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=85.0,
                                barrier_type=BarrierType.DOWN_IN, **_disc(d)),
        _spots((85.0, False))),
    "double_ko_call": (
        dict(payoff="call", strike=100.0, lower=80.0, upper=120.0),
        lambda d: DoubleBarrierOption(100.0, OptionType.CALL, 120.0, 80.0, DoubleBarrierType.KNOCK_OUT, **_disc(d)),
        _spots((120.0, True), (80.0, False))),
    "one_touch_up_at_hit": (
        dict(payoff="zero", upper=115.0, cash=1.0, at_hit=True),
        lambda d: OneTouchOption(barrier=115.0, barrier_direction=BarrierDirection.UP, rebate=1.0,
                                 payment_at_hit=True, touch_type=TouchType.ONE_TOUCH, **_disc(d)),
        _spots((115.0, True))),
    "one_touch_up_at_expiry": (
        dict(payoff="zero", upper=115.0, cash=1.0, at_hit=False),
        lambda d: OneTouchOption(barrier=115.0, barrier_direction=BarrierDirection.UP, rebate=1.0,
                                 payment_at_hit=False, touch_type=TouchType.ONE_TOUCH, **_disc(d)),
        _spots((115.0, True))),
    "no_touch_up": (
        dict(payoff="one", upper=115.0),
        lambda d: OneTouchOption(barrier=115.0, barrier_direction=BarrierDirection.UP, rebate=1.0,
                                 payment_at_hit=False, touch_type=TouchType.NO_TOUCH, **_disc(d)),
        _spots((115.0, True))),
    "double_one_touch_at_hit": (
        dict(payoff="zero", lower=85.0, upper=115.0, cash=1.0, at_hit=True),
        lambda d: DoubleOneTouchOption(upper_barrier=115.0, lower_barrier=85.0, rebate=1.0, payment_at_hit=True,
                                       touch_type=TouchType.DOUBLE_ONE_TOUCH, **_disc(d)),
        _spots((115.0, True), (85.0, False))),
}


# --- e18: discrete-KI Phoenix variants and the local-vol markets ---------------------

CRASH = dict(surface="example/modelvalidation/data/iv_surface_20240208.json", rate=0.02)
CALM = dict(CRASH, surface="example/modelvalidation/data/iv_surface_20231115.json")

#: variant -> phoenix spec overrides (levels as fractions of the initial price), KI monitoring
PHOENIX_VARIANTS = {
    "monthly_ki": (dict(), "monthly"),
    "daily_ki": (dict(), "daily"),
    "monthly_ki_memory": (dict(memory_coupon=True), "monthly"),
    "daily_ki_memory": (dict(memory_coupon=True), "daily"),
    "coupon_at_ko": (dict(coupon_barrier=1.03), "monthly"),
    "coupon_near_ko": (dict(coupon_barrier=1.029), "monthly"),
    "coupon_near_ko_memory": (dict(coupon_barrier=1.029, memory_coupon=True), "monthly"),
}


def phoenix(s0, overrides, ki):
    spec = dict(PHOENIX)
    for key in ("initial_price", "strike", "ko_barrier", "ki_barrier", "coupon_barrier"):
        spec[key] = spec[key] / 100.0 * s0
    spec["contract_multiplier"] = 100.0 / s0
    for key, value in overrides.items():
        spec[key] = value * s0 if key == "coupon_barrier" else value
    spec["ki_monitoring"] = "discrete"
    product = ph.make_phoenix(spec)
    if ki == "monthly":
        return product
    daily = [k / 252.0 for k in range(1, 253)]
    return ph.create_standard_phoenix(
        initial_price=float(spec["initial_price"]), strike=float(spec["strike"]), maturity=float(spec["maturity"]),
        contract_multiplier=float(spec["contract_multiplier"]), ko_barrier=ph._stepdown_schedule(spec, "ko_barrier", "ko_stepdown"),
        ko_rate=float(spec.get("ko_rate", 0.0)), ki_barrier=ph._ki_schedule(dict(spec, ki_monitoring="continuous")),
        coupon_barrier=ph._stepdown_schedule(spec, "coupon_barrier", "coupon_stepdown"), coupon_rate=float(spec["coupon_rate"]),
        num_observations=int(spec["num_observations"]), memory_coupon=bool(spec.get("memory_coupon", False)),
        coupon_pay_type=ph._coupon_pay_type(spec), is_reverse=False, disable_ko_after_ki=False,
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=daily)


# --- additions --------------------------------------------------------------------------

def ko_reset_discrete():
    """e18's flat KO-reset with the builder's daily knock-in schedule."""
    from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
    return make_ko_reset(dict(KO_RESET, ki_monitoring="discrete"))


def localvol_phoenix_case(market, moneyness=1.0, variant="daily_ki"):
    """(product, env, local_vol_surface) as e18's ``task`` builds a local-vol Phoenix cell.

    The surface paths are relative to the repository root; e18 resolved them against its working
    directory, these resolve them against this checkout.
    """
    from quantark.modelvalidation.builders.equity_snowball_localvol import (
        load_surface, make_localvol_environment)
    base = CRASH if market == "crash" else CALM
    spec = dict(base, surface=str(REPO / base["surface"]), spot_moneyness=moneyness)
    surf = load_surface(str(spec["surface"]), float(spec["rate"]))
    s0 = float(surf.artifact.s0)
    return phoenix(s0, *PHOENIX_VARIANTS[variant]), make_localvol_environment(spec), surf.local_vol

"""Frozen reference of the memory-Phoenix PDE before the arrears-linear memory fan.

Until base 1e32c27c the Phoenix PDE marched one value surface per memory state, 2(M+1) in all. The
arrears-linear fan keeps two (state 0 and state 0 plus one unit of arrears) and reproduces every readout to
round-off. ``phoenix_memory_fan_reference.json`` holds that old fan's numbers on the case matrix below;
``test_phoenix_memory_fan.py`` compares the live code against it.

Captured on the banking machine (main ``.venv``, py3.11, ARM64) from the base commit. Regenerate only from
a commit that still has the 2(M+1) fan, which no commit after the change does:

    env PYTHONPATH=<base tree>:<base tree>/test <venv>/bin/python test/pde_routes/phoenix_memory_fan_capture.py --out <json>

Not collected by pytest (no ``test_`` prefix).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parents[1]
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from quantark.asset.equity.engine.pde import PhoenixPDESolver  # noqa: E402
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver  # noqa: E402
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.option import PhoenixOption  # noqa: E402
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import (  # noqa: E402
    AccrualConfig, BarrierConfig, PayoffConfig,
)
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator  # noqa: E402
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.param.div.dividend_yield import TermStructureDividendYield  # noqa: E402
from quantark.param.rrf.rate_curve import LinearRateCurve  # noqa: E402
from quantark.param.vol.vol_surface import TermStructureVolSurface  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import CouponPayType, ObservationType, ProtectionType  # noqa: E402
from quantark.util.enum.engine_enums import EventProjectionMode  # noqa: E402

DEFAULT_OUT = Path(__file__).with_name("phoenix_memory_fan_reference.json")
VAL_DATE = datetime(2026, 1, 1)
GREEKS = ("delta", "gamma", "vega", "theta", "rho", "rhoq")


# --- environments -------------------------------------------------------------------------------------

def env_flat(spot=100.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot), rate_curve=FlatRateCurve(0.03),
                              div_yield=ContinuousDividendYield(0.01), vol_surface=FlatVolSurface(0.20),
                              valuation_date=VAL_DATE)


def env_term():
    times = [0.25, 0.5, 1.0, 2.0, 3.0]
    return PricingEnvironment(
        spot_quote=SpotQuote(100.0),
        rate_curve=LinearRateCurve(list(zip(times, [0.02, 0.035, 0.025, 0.03, 0.032]))),
        div_yield=TermStructureDividendYield(times, [0.005, 0.015, 0.01, 0.012, 0.011]),
        vol_surface=TermStructureVolSurface(times, [0.22, 0.19, 0.21, 0.2, 0.195]),
        valuation_date=VAL_DATE)


def env_local_vol():
    from pde_bitwise.capture_book import env_grid_vol
    return env_grid_vol()


# --- products -----------------------------------------------------------------------------------------

def phoenix(num_obs=12, memory=True, include_principal=False, coupon_barrier=85.0, ko_barrier=103.0,
            pay_type=CouponPayType.INSTANT, disable_ko_after_ki=False, ki_continuous=False,
            initial_arrears=0.0, ki_barrier=80.0):
    """A monthly Phoenix with a KO rate, a maturity rebate and a coupon barrier below KO (the study's)."""
    T = num_obs / 12.0
    dates = [i / 12.0 for i in range(1, num_obs + 1)]
    kos = list(ko_barrier) if isinstance(ko_barrier, (list, tuple)) else [ko_barrier] * num_obs
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=T, tenor=T, contract_multiplier=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=kos, ko_rate=0.05, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=dates, ki_barrier=ki_barrier,
            ki_observation_type=ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE,
            ki_observation_dates=None if ki_continuous else dates, ki_continuous=ki_continuous,
            disable_ko_after_ki=disable_ko_after_ki),
        coupon_config=CouponBarrierConfig(coupon_barrier=coupon_barrier, coupon_rate=0.12,
                                          coupon_pay_type=pay_type, memory_coupon=memory,
                                          initial_coupon_arrears=initial_arrears),
        payoff_config=PayoffConfig(rebate_rate=0.04, include_principal=include_principal, participation_rate=1.0,
                                   protection_type=ProtectionType.NONE),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ko=True, is_annualized_rebate=True,
                                     is_annualized_ki=False, is_annualized_coupon=True),
    )


def carried_ki():
    p = phoenix()
    p._otc_lifecycle_knocked_in = True
    return p


def reverse():
    p = phoenix(ko_barrier=97.0, coupon_barrier=115.0, ki_barrier=120.0)
    p.is_reverse = True
    p.validate()
    return p


def partial_protection():
    p = phoenix(include_principal=True)
    p.payoff_config = dataclasses.replace(p.payoff_config, protection_type=ProtectionType.PARTIAL,
                                          protection_rate=0.9)
    p.validate()
    return p


def step_down():
    p = phoenix(ko_barrier=[103.0 - 0.5 * i for i in range(12)])
    p.coupon_config = dataclasses.replace(p.coupon_config, coupon_barrier=[85.0 + 0.5 * i for i in range(12)])
    p.validate()
    return p


def dated_first_observation(spot, pay_type, coupon_rate=None, arrears=0.0):
    """The intraday fixture's dated memory Phoenix valued ON its first coupon date (coupon barrier 80).

    Its coupon is 0.12 * 100 / 12 = 1.0, so a miss at valuation owes exactly one unit; ``coupon_rate`` and
    ``arrears`` (carried from before the pricing date) make it owe something else."""
    from pde_bitwise.capture_book import _dated, _dated_env, _first_obs, _phoenix_expiry
    p, _ = _dated("phoenix", fixed_fraction=1.0 / 12.0)
    if coupon_rate is not None or arrears:
        p.coupon_config = dataclasses.replace(
            p.coupon_config, coupon_rate=p.coupon_config.coupon_rate if coupon_rate is None else coupon_rate,
            initial_coupon_arrears=arrears)
    if pay_type == "expiry":
        p = _phoenix_expiry(p)
    return p, _dated_env(p, _first_obs(p), spot)


# --- the matrix ---------------------------------------------------------------------------------------

def _plain(build, env=env_flat, params=None, greeks=GREEKS, session=False, solver=PhoenixPDESolver):
    def make():
        return build(), env(), solver, params or PDEParams(accuracy="standard")
    return {"make": make, "greeks": greeks, "session": session}


def _dated_case(spot, pay_type, coupon_rate=None, arrears=0.0, greeks=None):
    def make():
        p, env = dated_first_observation(spot, pay_type, coupon_rate, arrears)
        return p, env, PhoenixPDESolver, PDEParams(accuracy="standard")
    # theta refuses an EXPIRY coupon decided at the as-of date: no state carries it into the bumped contract
    if greeks is None:
        greeks = tuple(g for g in GREEKS if g != "theta") if pay_type == "expiry" else GREEKS
    return {"make": make, "greeks": greeks, "session": False}


CASES = {
    # AC1: core memory matrix
    "memory": _plain(phoenix, session=True),
    "memory_principal": _plain(lambda: phoenix(include_principal=True)),
    "memory_carried_ki": _plain(carried_ki),
    "memory_coupon_above_ko": _plain(lambda: phoenix(coupon_barrier=105.0)),
    "memory_expiry": _plain(lambda: phoenix(pay_type=CouponPayType.EXPIRY), session=True),
    "memory_disable_ko_after_ki": _plain(lambda: phoenix(disable_ko_after_ki=True)),
    "memory_continuous_ki": _plain(lambda: phoenix(ki_continuous=True)),
    "memory_initial_arrears": _plain(lambda: phoenix(initial_arrears=3.0)),
    "memory_reverse": _plain(reverse),
    "memory_partial_protection": _plain(partial_protection),
    "memory_step_down": _plain(step_down),
    "memory_term": _plain(phoenix, env=env_term),
    "memory_nodal": _plain(phoenix, params=PDEParams(accuracy="standard",
                                                     event_projection=EventProjectionMode.NODAL)),
    "memory_m36": _plain(lambda: phoenix(num_obs=36)),
    "dated_pay_instant": _dated_case(95.0, "instant"),
    "dated_miss_instant": _dated_case(78.0, "instant"),
    "dated_pay_expiry": _dated_case(95.0, "expiry"),
    "dated_miss_expiry": _dated_case(78.0, "expiry"),
    # a miss at valuation owing 1.5 (18% coupon) and 3.0 (2.0 carried): the old fan's state 1 held exactly that
    "dated_miss_instant_coupon_1.5": _dated_case(78.0, "instant", coupon_rate=0.18, greeks=()),
    "dated_miss_expiry_coupon_1.5": _dated_case(78.0, "expiry", coupon_rate=0.18, greeks=()),
    "dated_miss_instant_arrears_2": _dated_case(78.0, "instant", arrears=2.0, greeks=()),
    "dated_miss_expiry_arrears_2": _dated_case(78.0, "expiry", arrears=2.0, greeks=()),
    # AC2: local vol
    "lv_memory": _plain(phoenix, env=env_local_vol, session=True, solver=LocalVolPhoenixPDESolver),
    "lv_memory_continuous_ki": _plain(lambda: phoenix(ki_continuous=True), env=env_local_vol, session=True,
                                      solver=LocalVolPhoenixPDESolver),
    # AC3: non-memory stays where it was
    "plain": _plain(lambda: phoenix(memory=False), session=True),
    "plain_expiry": _plain(lambda: phoenix(memory=False, pay_type=CouponPayType.EXPIRY)),
}


# --- outputs ------------------------------------------------------------------------------------------

def _number(val):
    """A JSON-safe output: floats as floats, numeric arrays as lists, booleans and None as is, else repr."""
    if val is None or isinstance(val, (bool, np.bool_)):
        return None if val is None else bool(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in np.asarray(val, dtype=float).ravel()]
        except (TypeError, ValueError):
            return repr(val)
    return repr(val)


def stats_fields(stats) -> dict:
    """Every field of an event-statistics dataclass, by name."""
    return {f.name: _number(getattr(stats, f.name)) for f in dataclasses.fields(stats)}


def compute(name: str) -> dict:
    """The case's outputs: price, event statistics, BUMP Greeks and (when asked) session outputs."""
    case = CASES[name]
    product, env, solver_cls, params = case["make"]()
    out = {"price": float(solver_cls(params).price(product, env)),
           "stats": stats_fields(solver_cls(params).calculate_event_stats(product, env))}
    if case["greeks"]:
        greeks = GreeksCalculator().calculate(product, env, solver_cls(params), greeks=list(case["greeks"]))
        out["greeks"] = {k: float(v) for k, v in greeks.items() if isinstance(v, (int, float))}
    if case["session"]:
        session = solver_cls(params)._session_outputs(product, env, want_grid=True)
        out["session"] = {"npv": float(session.npv),
                          "solution": [float(x) for x in np.asarray(session.solution.solution_vec).ravel()],
                          "readout": _number(session.solution.readout_vec)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    names = args.only or list(CASES)
    payload = {name: compute(name) for name in names}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(payload)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

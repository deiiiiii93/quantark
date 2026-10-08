"""Item 7: which PDE prices the V0 boundary-rebate fix moves, and by how much.

Prints, for each case, the PDE price (full precision) next to QUAD (and the closed form where V0 is certain). Run it on
the parent commit (8753894f) and on the fix to get old -> new; the findings document lists both runs.

Cases (all annualized rebate, rebate_rate 10%, two-year tenor unless stated):
- certain-V0 snowball (KO 5x, discrete KI 1%) at T = 2 and 3: the closed form is exact;
- a desk-like snowball: KO 103 monthly, KI 75 discrete daily-ish (monthly) and continuous;
- a desk-like Phoenix: KO 103, coupon barrier 80, KI 75 discrete;
- a KO-reset snowball (ABSOLUTE), KI 75 discrete;
- the non-annualized twins of the snowball cases: must not move (control).

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
import math
from datetime import datetime

from quantark.asset.equity.engine.pde import KOResetSnowballPDESolver, PhoenixPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngine, PhoenixQuadEngine, SnowballQuadEngine
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, PostKOScheduleMode

S0, R = 100.0, 0.03


def sched(ts, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in ts])


def monthly(T):
    n = int(round(12 * T))
    return [(i + 1) / n * T for i in range(n)]


def barrier(T, ko, ki, continuous=False):
    obs = monthly(T)
    kw = dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if continuous else dict(
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False, ki_observation_schedule=sched(obs, ki))
    return BarrierConfig(ko_barrier=ko, ko_rate=0.15, ko_observation_schedule=sched(obs, ko), ki_barrier=ki, **kw)


def snowball(T, ko, ki, annual, continuous=False):
    return SnowballOption(initial_price=S0, strike=S0, barrier_config=barrier(T, ko, ki, continuous),
                          payoff_config=PayoffConfig(rebate_rate=0.10, include_principal=True),
                          accrual_config=AccrualConfig(is_annualized_rebate=annual), maturity=T, tenor=T)


def phoenix(T, annual):
    return PhoenixOption(initial_price=S0, strike=S0, barrier_config=barrier(T, 103.0, 75.0),
                         coupon_config=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12),
                         payoff_config=PayoffConfig(rebate_rate=0.10, include_principal=True),
                         accrual_config=AccrualConfig(is_annualized_rebate=annual), maturity=T, tenor=T)


def ko_reset(annual):
    pre = monthly(1.0)
    post = monthly(2.0)
    return KnockOutResetSnowballOption(
        initial_price=S0, strike=S0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_schedule=sched(pre, 103.0),
                                     ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
                                     ki_continuous=False, ki_observation_schedule=sched(pre, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=100.0, ko_rate=0.05, ko_observation_schedule=sched(post, 100.0)),
        payoff_config=PayoffConfig(rebate_rate=0.10, include_principal=True),
        accrual_config=AccrualConfig(is_annualized_rebate=annual), maturity=2.0, tenor=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)


def main():
    e = PricingEnvironment(spot_quote=SpotQuote(spot=S0, asset_name="X"), vol_surface=FlatVolSurface(volatility=0.2),
                           rate_curve=FlatRateCurve(rate=R), valuation_date=datetime(2024, 1, 2))
    cases = []
    for annual in (True, False):
        tag = "annual" if annual else "flat  "
        for T in (2.0, 3.0):
            cf = (S0 + 10.0 * (T if annual else 1.0)) * math.exp(-R * T)
            cases.append((f"snowball certain-V0 T={T:g} {tag}", snowball(T, 500.0, 1.0, annual),
                          SnowballPDESolver, SnowballQuadEngine, cf))
        cases.append((f"snowball KO103/KI75 discrete T=2 {tag}", snowball(2.0, 103.0, 75.0, annual),
                      SnowballPDESolver, SnowballQuadEngine, None))
        cases.append((f"snowball KO103/KI75 continuous T=2 {tag}", snowball(2.0, 103.0, 75.0, annual, True),
                      SnowballPDESolver, SnowballQuadEngine, None))
        cases.append((f"phoenix KO103/KI75 discrete T=2 {tag}", phoenix(2.0, annual),
                      PhoenixPDESolver, PhoenixQuadEngine, None))
        cases.append((f"ko-reset ABSOLUTE KI75 discrete {tag}", ko_reset(annual),
                      KOResetSnowballPDESolver, KOResetSnowballQuadEngine, None))
    print(f"{'case':<44} {'PDE':>20} {'QUAD':>14} {'closed':>14}")
    for name, p, pde_cls, quad_cls, cf in cases:
        pde = pde_cls().price(p, e)
        try:
            quad = quad_cls().price(p, e)
        except Exception as exc:  # report, do not hide
            quad = float("nan")
            name += f" (QUAD: {type(exc).__name__})"
        print(f"{name:<44} {pde:>20.12f} {quad:>14.6f} {cf if cf is not None else float('nan'):>14.6f}")


if __name__ == "__main__":
    main()

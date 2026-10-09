"""Item 7 follow-up (D7): the V0 rows by edge. Certain-V0 controls priced by PDE and QUAD against the closed form.

Standard (KO 500, KI 1) and reverse (KO 20, KI 10000) orientations, discrete and continuous KI, flat and annualized
rebate, fixed rebate 10% and call rebate (K 100, participation 0.2) with rebate_rate 0 and 0.10. The closed form is
(100 + rebate x tenor^a) e^{-rT}, or 100 e^{-rT} + 0.2 x tenor^a x BS call for the call rebate. The last column prints
the valuation step's V0 rows (branch=value).

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
import math
import sys
from datetime import datetime

from scipy.stats import norm

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

S0, R, VOL, T = 100.0, 0.03, 0.2, 2.0


def env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=S0, asset_name="X"), vol_surface=FlatVolSurface(volatility=VOL),
                              rate_curve=FlatRateCurve(rate=R), valuation_date=datetime(2024, 1, 2))


def product(rebate_rate, annual, reverse=False, call=False, cont=False):
    obs = [(i + 1) / 24 * T for i in range(24)]
    ko, ki = (20.0, 10000.0) if reverse else (500.0, 1.0)
    barrier = BarrierConfig(
        ko_barrier=ko, ko_rate=0.15,
        ko_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=ko) for t in obs]),
        ki_barrier=ki,
        ki_observation_type=ObservationType.CONTINUOUS if cont else ObservationType.DISCRETE, ki_continuous=cont,
        ki_observation_schedule=None if cont else ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ki) for t in obs]))
    extra = dict(call_rebate_enabled=True, call_strike=100.0, call_participation_rate=0.2) if call else {}
    return SnowballOption(initial_price=S0, strike=S0, barrier_config=barrier,
                          payoff_config=PayoffConfig(rebate_rate=rebate_rate, include_principal=True, **extra),
                          accrual_config=AccrualConfig(is_annualized_rebate=annual), maturity=T, tenor=T,
                          is_reverse=reverse)


def closed_form(p, e, rebate_rate, annual, call):
    ten = p.get_contract_tenor(e) if annual else 1.0
    df = math.exp(-R * T)
    if call:
        d1 = (math.log(S0 / 100.0) + (R + VOL ** 2 / 2) * T) / (VOL * math.sqrt(T))
        c = S0 * norm.cdf(d1) - 100.0 * df * norm.cdf(d1 - VOL * math.sqrt(T))
        return 100.0 * df + 0.2 * ten * c
    return (100.0 + rebate_rate * 100.0 * ten) * df


def main():
    e = env()
    print(f"{'case':<34} {'closed':>13} {'PDE-cf':>10} {'QUAD-cf':>10} {'PDE-QUAD':>10}  rows(lo|up)")
    for reverse in (False, True):
        for call, rr in ((False, 0.10), (True, 0.0), (True, 0.10)):
            for cont in (False, True):
                for annual in (False, True):
                    try:
                        p = product(rr, annual, reverse, call, cont)
                        cf = closed_form(p, e, rr, annual, call)
                        s = SnowballPDESolver()
                        pde = s.price(p, e)
                        q = SnowballQuadEngine().price(p, e)
                        st = s._rows_v0_stash
                        name = f"{'rev' if reverse else 'std'} {'call' if call else 'fix'} rr={rr:.2f} {'cont' if cont else 'disc'} {'ann' if annual else 'flat'}"
                        print(f"{name:<34} {cf:>13.8f} {pde - cf:>10.2e} {q - cf:>10.2e} {pde - q:>10.2e}  "
                              f"{st[4].branch}={st[4].value:.4f} | {st[5].branch}={st[5].value:.4f}")
                    except Exception as exc:  # noqa: BLE001 - probe
                        print(reverse, call, rr, cont, annual, "ERR", type(exc).__name__, exc)


if __name__ == "__main__":
    main()

"""Item 7, call-rebate branch: the V0 LOWER row of a call-rebate snowball.

With ``call_rebate_enabled`` the contract's V0 payoff is principal + participation x max(S - K_c, 0) (x tenor when
annualized) and NO fixed rebate (get_maturity_payoff_v0). The PDE's lower row (S -> 0) nevertheless adds
rebate_rate x S0 -- a fixed rebate the contract does not pay. The factory default sets rebate_rate = ko_rate, so a
call-rebate snowball built by create_standard_snowball carries one. This prices a certain-V0 call-rebate contract
(KO 5x, discrete KI at 1%) with PDE and QUAD for rebate_rate 0 and 0.10, both rebate flags.

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
from datetime import datetime

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

S0, T = 100.0, 2.0


def product(rebate_rate, annualized):
    obs = [(i + 1) / 24 * T for i in range(24)]
    barrier = BarrierConfig(
        ko_barrier=500.0, ko_rate=0.15, ko_observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=500.0) for t in obs]),
        ki_barrier=1.0, ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
        ki_observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=1.0) for t in obs]))
    return SnowballOption(
        initial_price=S0, strike=S0, barrier_config=barrier,
        payoff_config=PayoffConfig(rebate_rate=rebate_rate, include_principal=True, call_rebate_enabled=True,
                                   call_strike=100.0, call_participation_rate=0.2),
        accrual_config=AccrualConfig(is_annualized_rebate=annualized), maturity=T, tenor=T)


def main():
    e = PricingEnvironment(spot_quote=SpotQuote(spot=S0, asset_name="X"), vol_surface=FlatVolSurface(volatility=0.2),
                           rate_curve=FlatRateCurve(rate=0.03), valuation_date=datetime(2024, 1, 2))
    print(f"{'rebate_rate':>11} {'annual':>6} {'QUAD':>12} {'PDE':>12} {'PDE-QUAD':>10} {'lower row':>10}")
    for rr in (0.0, 0.10):
        for annual in (False, True):
            p = product(rr, annual)
            quad = SnowballQuadEngine().price(p, e)
            s = SnowballPDESolver()
            pde = s.price(p, e)
            print(f"{rr:>11.2f} {str(annual):>6} {quad:>12.6f} {pde:>12.6f} {pde - quad:>10.2e} "
                  f"{s._rows_v0_stash[4].value:>10.4f}")


if __name__ == "__main__":
    main()

"""Item 7: the PDE V0 Dirichlet boundary rebate is never tenor-scaled.

A snowball that can neither knock out nor knock in in practice (KO barrier 5x spot, discrete KI at 1% of spot) is certain
to mature in V0, so its price is the closed form (principal + rebate_rate x S0 x tenor^a) x DF(T), a = 1 when the rebate
is annualized. Every engine's TERMINAL payoff reads the product (get_maturity_payoff_v0); the PDE's Dirichlet rows
(snowball_pde_solver._boundary_rows_v0) build principal + rebate_rate x S0 themselves. This prices the contract with
PDE, QUAD and MC against the closed form for tenors 1, 2, 3 and both rebate flags, and prints the PDE boundary rows.

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
import math
from datetime import datetime

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod

S0, R, VOL, RATE = 100.0, 0.03, 0.2, 0.10


def env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=S0, asset_name="X"), vol_surface=FlatVolSurface(volatility=VOL),
                              rate_curve=FlatRateCurve(rate=R), valuation_date=datetime(2024, 1, 2))


def product(T, annualized, ki_level=0.01, ko_level=5.0, reverse=False):
    """Monthly KO and KI on the quoted clock; ``tenor=T`` is the contract tenor an annualized rebate multiplies."""
    n = int(round(12 * T))
    obs = [(i + 1) / n * T for i in range(n)]
    if reverse:   # reverse: KO below (1/ko_level), KI above (1/ki_level)
        ko_level, ki_level = 1.0 / ko_level, 1.0 / ki_level
    barrier = BarrierConfig(
        ko_barrier=ko_level * S0, ko_rate=0.15, ko_observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ko_level * S0) for t in obs]),
        ki_barrier=ki_level * S0, ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
        ki_observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ki_level * S0) for t in obs]))
    return SnowballOption(
        initial_price=S0, strike=S0, barrier_config=barrier,
        payoff_config=PayoffConfig(rebate_rate=RATE, include_principal=True),
        accrual_config=AccrualConfig(is_annualized_rebate=annualized), maturity=T, tenor=T, is_reverse=reverse)


def main():
    e = env()
    print(f"{'T':>4} {'annual':>6} {'closed':>12} {'PDE':>12} {'PDE-cf':>10} {'QUAD':>12} {'QUAD-cf':>10} "
          f"{'MC':>12} {'MC-cf':>9} {'bnd_lo':>9} {'bnd_up':>9}")
    for T in (1.0, 2.0, 3.0):
        for annual in (False, True):
            p = product(T, annual)
            tenor = p.get_contract_tenor(e) if annual else 1.0
            cf = (S0 + RATE * S0 * tenor) * math.exp(-R * T)
            solver = SnowballPDESolver()
            pde = solver.price(p, e)
            quad = SnowballQuadEngine().price(p, e)
            mc = SnowballMCEngine(params=MCParams(num_paths=20000, seed=7), method=MonteCarloMethod.QUASI).price(p, e)
            # the solver's last-set boundary rows (the t = 0 step): the Dirichlet values the grid edges carry
            stash = getattr(solver, "_rows_v0_stash", None)
            lo = up = float("nan")
            if stash is not None:
                lo, up = stash[4].value, stash[5].value
            print(f"{T:>4.1f} {str(annual):>6} {cf:>12.6f} {pde:>12.6f} {pde - cf:>10.2e} {quad:>12.6f} "
                  f"{quad - cf:>10.2e} {mc:>12.6f} {mc - cf:>9.2e} {lo:>9.4f} {up:>9.4f}")

    # The boundary's reach grows as the spot moves towards it: T = 2, the same contract priced at lower spots
    # (the closed form does not depend on the spot: V0 is certain and its payoff is a constant).
    print("\nspot sweep, T=2 (closed form independent of spot)")
    print(f"{'spot':>6} {'annual':>6} {'closed':>12} {'PDE':>12} {'PDE-cf':>10} {'QUAD-cf':>10}")
    for spot in (100.0, 50.0, 25.0, 10.0):
        for annual in (False, True):
            p = product(2.0, annual)
            e = PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="X"),
                                   vol_surface=FlatVolSurface(volatility=VOL), rate_curve=FlatRateCurve(rate=R),
                                   valuation_date=datetime(2024, 1, 2))
            tenor = p.get_contract_tenor(e) if annual else 1.0
            cf = (S0 + RATE * S0 * tenor) * math.exp(-R * 2.0)
            pde = SnowballPDESolver().price(p, e)
            quad = SnowballQuadEngine().price(p, e)
            print(f"{spot:>6.1f} {str(annual):>6} {cf:>12.6f} {pde:>12.6f} {pde - cf:>10.2e} {quad - cf:>10.2e}")


if __name__ == "__main__":
    main()

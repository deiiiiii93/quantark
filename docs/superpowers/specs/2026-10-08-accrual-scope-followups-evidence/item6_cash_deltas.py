"""Item 6: the cash and price of factory-default autocallables, under the tree as it stands.

Run it twice: on the clean tree (today: is_annualized_rebate / is_annualized_ki = None are NOT annualized), and after
``item6_effective_flags_patch.py apply`` (None follows is_annualized = True, as knock-out accrual already does).
Every contract here is built by the public factories with their defaults: is_annualized=True, the rebate and KI flags
None, rebate_rate = ko_rate, include_principal=False, contract multiplier 1.

Printed per contract: contract tenor, V0 terminal payoff at S=100 (the rebate), V1 terminal payoff at S=60 (the
knock-in loss), and the QUAD price.

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
from datetime import datetime

from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngine, PhoenixQuadEngine, SnowballQuadEngine
from quantark.asset.equity.product.option import (
    create_ko_reset_snowball, create_standard_phoenix, create_standard_snowball,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import PostKOScheduleMode


def main():
    env = PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                             rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2024, 1, 2))
    cases = []
    for T in (0.5, 1.0, 2.0, 3.0):
        cases.append((f"snowball default T={T:g}",
                      create_standard_snowball(initial_price=100.0, strike=100.0, maturity=T,
                                               num_observations=int(12 * T)), SnowballQuadEngine))
    for T in (1.0, 2.0):
        cases.append((f"phoenix default T={T:g}",
                      create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=T,
                                              num_observations=int(12 * T)), PhoenixQuadEngine))
    cases.append(("ko-reset default pre=1 post=2 ABS",
                  create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=2.0,
                                           post_ko_mode=PostKOScheduleMode.ABSOLUTE), KOResetSnowballQuadEngine))
    cases.append(("ko-reset default pre=1 post=2 ABS, rebate 15%",
                  create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=2.0,
                                           post_ko_mode=PostKOScheduleMode.ABSOLUTE, rebate_rate=0.15),
                  KOResetSnowballQuadEngine))
    print(f"{'contract':<46} {'tenor':>6} {'V0(100)':>10} {'V1(60)':>10} {'QUAD':>12}")
    for name, p, engine in cases:
        tenor = p.get_contract_tenor(env)
        v0 = p.get_maturity_payoff_v0(100.0, pricing_env=env)
        v1 = p.get_maturity_payoff_v1(60.0, pricing_env=env)
        price = engine().price(p, env)
        print(f"{name:<46} {tenor:>6.3f} {v0:>10.4f} {v1:>10.4f} {price:>12.6f}")


if __name__ == "__main__":
    main()

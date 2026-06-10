"""Quick MC comparison for validation report."""
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.engine.pde import BarrierPDESolver
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.param import PDEParams, MCParams
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.param.quote.spot_quote import SpotQuote
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from datetime import datetime
from quantark.util.enum import BarrierType, OptionType, ObservationType

def make_env(spot=100, rate=0.05, vol=0.20):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot),
        rate_curve=FlatRateCurve(rate),
        vol_surface=FlatVolSurface(vol),
        valuation_date=datetime(2024, 1, 1)
    )

pde = BarrierPDESolver(PDEParams(grid_size=400, time_steps=200))
analytical = BarrierAnalyticalEngine()
mc = BarrierOptionMCEngine(params=MCParams(num_paths=50000, seed=42),
                          method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI))

cases = [
    ('ATM Call D0O barrier=90', 100, 100, 90, BarrierType.DOWN_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
    ('OTM Call D0O barrier=95', 100, 105, 95, BarrierType.DOWN_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
    ('ATM Call U0O barrier=110', 100, 100, 110, BarrierType.UP_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
]

print('Case                           PDE          Analytical   MC          PDE vs MC')
print('-' * 85)

for name, spot, strike, barrier, btype, otype, T, r, sigma in cases:
    env = make_env(spot, r, sigma)
    option = BarrierOption(strike=strike, option_type=otype, barrier=barrier,
                          barrier_type=btype, maturity=T, rebate=0.0,
                          observation_type=ObservationType.CONTINUOUS)

    pde_price = pde.price(option, env)
    analytical_price = analytical.price(option, env)
    mc_price = mc.price(option, env)
    if mc_price != 0:
        error_vs_mc = abs(pde_price - mc_price) / mc_price
    else:
        error_vs_mc = abs(pde_price - mc_price)

    print(f'{name:<30} {pde_price:>8.4f}    {analytical_price:>8.4f}   {mc_price:>8.4f}   {error_vs_mc:>6.2%}')

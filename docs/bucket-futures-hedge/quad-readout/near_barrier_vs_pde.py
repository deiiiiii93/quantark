"""Independent check of the peer's near-barrier delta bias claim.

A peer session reports that both QUAD readouts carry a signed delta lobe of
order 14 reference hands within a few percent of the KI barrier, which grid
refinement does not remove, and that my detrended sub-cell metric cannot see
it because the bias is shared by every cell.

Checked here WITHOUT their reference: the PDE engine is an independent
family in this repo. If PDE and QUAD disagree near the barrier by the
reported magnitude, the claim stands on repo tools alone.
"""
from __future__ import annotations
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.param import FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

S0, KI, KO = 100.0, 75.0, 103.0
VOL, RATE, DIV = 0.22, 0.02, 0.03
M_REF, NOTIONAL = 200.0, 50_000_000.0
# study hands: currency delta / m_ref, on a book scaled to the notional
SCALE = NOTIONAL / S0 / M_REF


def env(spot):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot)),
        vol_surface=FlatVolSurface(volatility=VOL),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIV),
        valuation_date=datetime(2026, 1, 2),
    )


def product():
    return create_standard_snowball(
        initial_price=S0, strike=S0, maturity=1.0,
        ko_barrier=KO, ko_rate=0.10, ki_barrier=KI,
        num_observations=10, include_principal=False,
        ko_observation_dates=[(i + 3) / 12.0 for i in range(10)],
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[(i + 1) / 261.0 for i in range(261)],
    )


def delta_of(engine, p, spot):
    row = engine.calculate_spot_greeks_curve(p, env(spot), [spot])[0]
    return float(row["delta"]) * SCALE


def main():
    p = product()
    pde = SnowballPDESolver(PDEParams(accuracy="high"))
    spots = [KI * m for m in (0.985, 0.995, 1.0084, 1.02, 1.05)]
    print(f"# delta in study hands (x{SCALE:.2f}); KI = {KI}")
    print(f"{'spot':>9} {'% vs KI':>9} {'PDE':>12} "
          f"{'legacy':>12} {'transition':>12} {'leg-PDE':>9} {'tr-PDE':>9}")
    for s in spots:
        ref = delta_of(pde, p, s)
        row = [ref]
        for mode in ("legacy_linear", "transition"):
            eng = SnowballQuadEngine(QuadParams(grid_points=401, readout=mode))
            row.append(delta_of(eng, p, s))
        print(f"{s:9.3f} {100 * (s / KI - 1):+9.2f} {row[0]:12.3f} "
              f"{row[1]:12.3f} {row[2]:12.3f} "
              f"{row[1] - row[0]:9.3f} {row[2] - row[0]:9.3f}")


if __name__ == "__main__":
    main()

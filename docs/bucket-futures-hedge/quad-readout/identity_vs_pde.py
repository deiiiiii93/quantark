"""Is the carry identity's residual QUAD's discrete knock-in barrier?

VERDICT: THIS REPRODUCTION FAILS ITS OWN CONTROL. Do not read its numbers as
evidence about the study's residual. Kept, with the control that condemns
it, so the construction is not tried again.

`bump_control()` shows the residual here scaling ~100x between a 1% and a
0.1% spot bump -- textbook O(h^2) truncation in the central differences.
The study's residual is bump-INVARIANT, 0.042613 / 0.043244 / 0.046811 hands
over the same three bumps. Different error structure, so a different
quantity. The substitution below is what breaks it: the study measures
bucket deltas DIRECTLY over a real listed curve, while this routes the same
chain rule through one flat yield analytically, and that analytic step is
what the truncation lives in.

The one thing it does show robustly, because it survives the control, is
that the readout is irrelevant to the residual: QUAD legacy and QUAD
transition agree closely in every cell of the survey.

Everything below this line is the original intent, preserved for context.
--------------------------------------------------------------------------

The study audit's residual collapses 71x the moment the product knocks in
(`residual_by_ki_state.py`), which says the live daily-monitored knock-in
barrier is the whole of it.  That is a claim about the ENGINE, so it should
reproduce with no study, no futures curve and no replay -- and it should not
reproduce on an independent engine family.

The audit's identity is a chain rule.  With one futures tenor `T` and a flat
yield, pinning `F = S*exp((r-q)T)` while spot moves means `q(S) = q0 +
log(S/S0)/T`, and

    dV/dS|_q  =  d/dS V(S, q(S))  -  (1/(S*T)) * dV/dq

so the residual below is the same quantity the study reports, stripped to
three price bumps.  Three arms, one variable each:

  * QUAD legacy vs QUAD transition   -- is it the readout?  (expected: no)
  * QUAD vs PDE                      -- is it QUAD specifically?
  * knock-in live vs no knock-in     -- is it the barrier?

Run:  PYTHONPATH=. python docs/bucket-futures-hedge/quad-readout/identity_vs_pde.py
"""
from __future__ import annotations

from datetime import datetime

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
SCALE = NOTIONAL / S0 / M_REF          # unit delta -> study hands
TENOR = 0.5                            # the pinned contract's tenor
BUMP = 0.01                            # the study's own relative spot bump
DQ = 1e-4


def env(spot: float, div: float) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot)),
        vol_surface=FlatVolSurface(volatility=VOL),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=float(div)),
        valuation_date=datetime(2026, 1, 2),
    )


def product(*, knock_in: bool):
    """The study's shape: daily discrete KI, ten monthly KO observations.

    `knock_in=False` drops the barrier to 1e-6 of spot rather than removing
    the feature, so the two arms differ ONLY in whether the barrier is ever
    within reach -- same payoff assembly, same observation schedule, same
    code path.
    """
    return create_standard_snowball(
        initial_price=S0, strike=S0, maturity=1.0,
        ko_barrier=KO, ko_rate=0.10,
        ki_barrier=KI if knock_in else 1e-6,
        num_observations=10, include_principal=False,
        ko_observation_dates=[(i + 3) / 12.0 for i in range(10)],
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[(i + 1) / 261.0 for i in range(261)],
    )


def residual(engine, p, spot: float, bump: float = BUMP):
    """(residual, dV/dS at frozen q, pinned-path dV/dS, dV/dq) in study hands."""
    import math

    h = bump * spot

    def price(s, q):
        return float(engine.price(p, env(s, q)))

    # spot moves, the yield is frozen: the carry curve moves with spot
    frozen = (price(spot + h, DIV) - price(spot - h, DIV)) / (2 * h)

    # spot moves, the yield absorbs it so the listed forward is pinned
    q_up = DIV + math.log((spot + h) / spot) / TENOR
    q_dn = DIV + math.log((spot - h) / spot) / TENOR
    pinned = (price(spot + h, q_up) - price(spot - h, q_dn)) / (2 * h)

    dv_dq = (price(spot, DIV + DQ) - price(spot, DIV - DQ)) / (2 * DQ)

    resid = frozen - pinned + dv_dq / (spot * TENOR)
    return (resid * SCALE, frozen * SCALE, pinned * SCALE, dv_dq * SCALE)


ARMS = {
    "QUAD legacy_linear": lambda: SnowballQuadEngine(
        params=QuadParams(grid_points=401, readout="legacy_linear")
    ),
    "QUAD transition   ": lambda: SnowballQuadEngine(
        params=QuadParams(grid_points=401, readout="transition")
    ),
    "PDE               ": lambda: SnowballPDESolver(params=PDEParams()),
}


def survey() -> None:
    for knock_in in (True, False):
        label = "knock-in barrier LIVE at 75" if knock_in else "no knock-in barrier"
        print(f"\n=== {label} ===")
        print(f"{'engine':20s} {'spot':>6s} {'residual':>11s} "
              f"{'frozen-q D':>12s} {'pinned-F D':>12s} {'dV/dq':>12s}")
        p = product(knock_in=knock_in)
        for name, make in ARMS.items():
            engine = make()
            for spot in (100.0, 90.0, 80.0):
                r, fr, pi, dq = residual(engine, p, spot)
                print(f"{name:20s} {spot:6.1f} {r:11.4f} "
                      f"{fr:12.4f} {pi:12.4f} {dq:12.2f}")
    print("\nresiduals are study hands; the audit's budget is 0.01")


def bump_control() -> None:
    """The control that decides whether this reproduction is faithful at all.

    The study's residual is bump-INVARIANT: 0.0426 / 0.0432 / 0.0468 hands at
    1%, 0.25% and 0.1% under `transition`.  If the residual here shrinks with
    the bump instead, this standalone chain rule is measuring its own
    truncation error and says nothing about the study's floor.
    """
    print("\n=== bump control: does this residual shrink with the step? ===")
    print(f"{'engine':20s} {'KI':>4s} {'spot':>6s} "
          + "".join(f"{b:>12}" for b in ("1%", "0.25%", "0.1%")) + f"{'ratio':>9s}")
    for knock_in in (True, False):
        p = product(knock_in=knock_in)
        for name, make in ARMS.items():
            engine = make()
            for spot in (100.0, 80.0):
                vals = [residual(engine, p, spot, b)[0]
                        for b in (0.01, 0.0025, 0.001)]
                ratio = abs(vals[0]) / max(abs(vals[2]), 1e-12)
                print(f"{name:20s} {str(knock_in):>4s} {spot:6.1f} "
                      + "".join(f"{v:12.4f}" for v in vals) + f"{ratio:9.2f}")
    print("\nratio = |1% residual| / |0.1% residual|. Pure truncation in the")
    print("central differences would be ~100; the study's ladder gives ~0.9.")


def main() -> None:
    survey()
    bump_control()


if __name__ == "__main__":
    main()

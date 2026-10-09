"""G1: a fixing dated exactly on the valuation date (t = 0, today's unprinted fixing) under w5b's freeze.

User decision 2026-10-09 (late): t = 0 is LIVE - it moves with spot bumps; only t < 0 is frozen. w5b's
``missing_fixing_fills`` selects ``resolve_time <= 0``, so ``freeze_missing_fixings`` records the t = 0 fixing at
the base spot and every bump keeps it. Dated range accrual on day 4 (fixing #0 dated day 4), spot 109.9 under the
110 barrier.
"""
import warnings

import _fixtures as F
from quantark.asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.util.fixings import freeze_missing_fixings

ENGINE = RangeAccrualAnalyticalEngine()


def price(spot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ENGINE.price(F.ra_dated(), F.eq_env(F.FIX_DAY, spot=spot))


if __name__ == "__main__":
    base = 109.9
    env = F.eq_env(F.FIX_DAY, spot=base)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frozen = freeze_missing_fixings(F.ra_dated(), env)
        greeks = GreeksCalculator().calculate(F.ra_dated(), env, ENGINE)
    print(f"freeze_missing_fixings records the t=0 fixing: {frozen.observation_records[0].observed_in_range!r} "
          f"(MissingFixingWarning x{sum('MissingFixing' in w.category.__name__ for w in caught)})")
    print(f"GreeksCalculator delta (t=0 fixing frozen at the base spot): {greeks['delta']:.6f}")
    for h in (0.05, 0.2, 1.0, base * 0.01):
        print(f"live central difference, h={h}: delta {(price(base + h) - price(base - h)) / (2 * h):.6f}")

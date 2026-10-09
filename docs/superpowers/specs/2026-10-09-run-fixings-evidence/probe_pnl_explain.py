"""P1-P4: what PnL explain does with fixings around its two snapshots (dated range accrual, waterfall + Taylor).

P1  t0 = day 4 (fixing #0 dated t0: t = 0, live), t1 = day 10, t1's contract does NOT record #0.
P2  same pair, t1's contract RECORDS #0 (in range, at day 4's spot 105) - the data a desk would book.
P2b same pair, t1 records #0 OUT of range (a real fixing that differs from the t0 fill).
P3  t0 = day 3, t1 = day 5: #0 falls strictly inside the step; t1's contract does not record it.
P3b same step, t1 records #0 (in range).
P4  a fixing already missing at t0 (strictly past): the shocked states keep the t0 fill; pv_t1 vs t1's own value.
"""
import warnings
from datetime import timedelta

import _fixtures as F
from quantark.asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.asset.equity.product.option import RangeAccrualConfig, RangeAccrualObservationRecord, RangeAccrualOption

ENGINE = RangeAccrualAnalyticalEngine()
CONFIG = PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR))


def snap(product, day, spot=None):
    env = F.eq_env(day, spot=spot)
    return ValuationSnapshot(product, ENGINE, env, date=env.valuation_date, quantity=1.0)


def show(title, s0, s1):
    print(f"\n--- {title}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = explain(s0, s1, config=CONFIG)
        except Exception as exc:  # noqa: BLE001 - the probe reports what the library does
            print(f"RAISES {type(exc).__name__}: {exc}")
            return
    own_t1 = ENGINE.price(s1.product, s1.pricing_env)
    print(f"pv_t0 {res.pv_t0.total:.6f}  pv_alive_t1 {res.pv_alive_t1.total:.6f}  pv_t1 {res.pv_t1.total:.6f}  "
          f"t1 snapshot's own value {own_t1:.6f}  total_pnl {res.total_pnl:.6f}  "
          f"MissingFixingWarning x{sum('MissingFixing' in w.category.__name__ for w in caught)}")
    for r in res.rows:
        print(f"   {r.method.value:>9} {r.term:<22} {r.pnl:>12.6f}")


def ra_missing_at_t0():
    """A strictly past fixing (day -3) without its outcome plus the shared future schedule."""
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10),
        exercise_date=F.START + timedelta(days=365), initial_date=F.START - timedelta(days=10),
        observation_records=[RangeAccrualObservationRecord(observation_date=F.START - timedelta(days=3))]
        + [RangeAccrualObservationRecord(observation_date=d) for d in F.fixing_dates()[2:]],
    )


if __name__ == "__main__":
    show("P1 t0 = day 4 (#0 live), t1 = day 10, t1 unrecorded", snap(F.ra_dated(), 4), snap(F.ra_dated(), 10))
    show("P2 t0 = day 4, t1 = day 10, t1 records #0 = in range", snap(F.ra_dated(), 4),
         snap(F.ra_dated({0: True}), 10))
    show("P2b t0 = day 4, t1 = day 10, t1 records #0 = OUT of range (a real fixing that differs from t0's spot)",
         snap(F.ra_dated(), 4), snap(F.ra_dated({0: False}), 10))
    show("P3 t0 = day 3, t1 = day 5 (#0 strictly inside), t1 unrecorded", snap(F.ra_dated(), 3),
         snap(F.ra_dated(), 5))
    show("P3b t0 = day 3, t1 = day 5, t1 records #0 = in range", snap(F.ra_dated(), 3),
         snap(F.ra_dated({0: True}), 5))
    show("P4 fixing missing at t0 (strictly past), t0 spot 105, t1 = day 6 at spot 111",
         snap(ra_missing_at_t0(), 0, spot=105.0), snap(ra_missing_at_t0(), 6, spot=111.0))

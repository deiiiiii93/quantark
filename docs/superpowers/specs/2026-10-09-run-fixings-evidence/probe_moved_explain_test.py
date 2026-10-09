"""M1: the w5b explain test's fixture (test_missing_fixings_scenarios.py::test_pnl_explain_keeps_the_t0_fill_across_the_pair)
under the user's 2026-10-09 explain decision (pv_t1 = the t1 snapshot's OWN value; a change of a filled missing
fixing is its own row). Today pv_t1 carries the t0 fill; the t1 snapshot's own value fills at t1's spot."""
import warnings
from datetime import timedelta

from test_missing_fixings_scenarios import EQ_SPOT, VAL, _eq_env, _ra_dated

from quantark.asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

warnings.simplefilter("ignore")
t1 = VAL + timedelta(days=3)
engine = RangeAccrualAnalyticalEngine()
s0 = ValuationSnapshot(_ra_dated(None), engine, _eq_env(EQ_SPOT, VAL), date=VAL, quantity=2.0)
s1 = ValuationSnapshot(_ra_dated(None), engine, _eq_env(111.0, t1), date=t1, quantity=2.0)
res = explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)))
own = 2.0 * engine.price(_ra_dated(None), _eq_env(111.0, t1))
print(f"today: pv_t0 {res.pv_t0.total:.6f} pv_t1 {res.pv_t1.total:.6f} total_pnl {res.total_pnl:.6f}")
print(f"t1 snapshot's own value {own:.6f}; proposed total_pnl {own - res.pv_t0.total:.6f}; "
      f"proposed fixing-fill row {own - res.pv_t1.total:.6f}")

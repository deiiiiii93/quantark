"""Spec §12 support matrix + test 6b (MC common random numbers) + test 10 (frame determinism)."""
import os
import subprocess
import sys
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import (
    AmericanOptionAnalyticalEngine, DeltaOneEngine, DigitalOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.asset.equity.product.option.american_option import AmericanOption
from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
import quantark.pnlexplain as pnlexplain_pkg
from quantark.pnlexplain import (
    BookSnapshot, ExplainMethod, ExplainRow, ExplainTrade, Factor, FactorCoordinate, FactorMoves,
    LifecycleTransition, MARKET_FACTORS, PnLExplainConfig, PnLExplainResult,
    PortfolioExplainResult, PositionExplainResult, PositionSnapshot, RowKind, ValuationSnapshot,
    ValueBreakdown, contract_fingerprint, explain, explain_portfolio, explain_position,
    lifecycle_fingerprint, value,
)
# `PnLExplainRecorder` and the equity re-exports are looked up INSIDE the export tests via
# `pnlexplain_pkg` so this module collects independently of the final export step.
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.deltaone_enums import DeltaOneType
from quantark.util.enum.engine_enums import AmericanAnalyticalMethod

sys.path.insert(0, os.path.dirname(__file__))
from test_pnlexplain_lifecycle_days import _snowball  # noqa: E402

T0, T1 = datetime(2026, 6, 26), datetime(2026, 6, 29)
_USED = (BookSnapshot, ExplainRow, ExplainTrade, FactorCoordinate, FactorMoves, LifecycleTransition,
         MARKET_FACTORS, PnLExplainResult, PortfolioExplainResult, PositionExplainResult, PositionSnapshot,
         RowKind, ValueBreakdown, contract_fingerprint, explain_portfolio, explain_position,
         lifecycle_fingerprint, value)     # the public surface imports at module scope


def _env(spot, date, vol=0.2):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=date)


def _tol(x):
    return 1e-10 * max(1.0, abs(x))


@pytest.mark.parametrize("family,make", [
    ("american", lambda m: (AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=m),
                            AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93))),
    ("digital", lambda m: (CashOrNothingDigitalOption(strike=100.0, option_type=OptionType.CALL, maturity=m, payout=10.0),
                           DigitalOptionAnalyticalEngine())),
    ("futures", lambda m: (Futures(underlying="X", multiplier=300.0, maturity=m), DeltaOneEngine())),
])
def test_family_reconciles_both_methods(family, make):
    p0, eng = make(1.0)
    p1, _ = make(1.0 - 3 / 365)
    s0 = ValuationSnapshot(p0, eng, _env(100.0, T0), date=T0)
    s1 = ValuationSnapshot(p1, eng, _env(102.0, T1, vol=0.21), date=T1)
    res = explain(s0, s1)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=_tol(res.total_pnl)), family
    if family == "futures":
        assert res.by_factor(ExplainMethod.WATERFALL)["vol"] == 0.0
        assert [r for r in res.rows if r.term == "vega"][0].greek is None


def test_spot_instrument_is_delta_only():
    spot = SpotInstrument(underlying="X", deltaone_type=DeltaOneType.STOCK)
    s0 = ValuationSnapshot(spot, DeltaOneEngine(), _env(100.0, T0), date=T0, quantity=7.0)
    s1 = ValuationSnapshot(spot, DeltaOneEngine(), _env(102.0, T1), date=T1, quantity=7.0)
    res = explain(s0, s1)
    assert res.by_factor(ExplainMethod.WATERFALL)["spot"] == pytest.approx(14.0)
    assert res.by_factor(ExplainMethod.TAYLOR)["spot"] == pytest.approx(14.0)
    assert res.unexplained == pytest.approx(0.0, abs=1e-12)


def test_mc_common_random_numbers_make_the_waterfall_exact():
    import pandas as pd
    from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker
    eng = SnowballMCEngine(MCParams(num_paths=4000, time_steps=32, seed=7))
    p0 = _snowball()
    # a snowball's time grid is tied to its observation schedule: roll it the way the trackers
    # do (schedule and maturity together), never by assigning `maturity`
    tracker = AutocallableLifecycleTracker(product=_snowball(), quantity=-2.0, start_date=pd.Timestamp(T0))
    e1 = _env(101.0, T1, vol=0.23)
    p1 = tracker.product_for_pricing(pd.Timestamp(T1), e1)
    s0 = ValuationSnapshot(p0, eng, _env(100.0, T0, vol=0.22), date=T0, quantity=-2.0)
    s1 = ValuationSnapshot(p1, eng, e1, date=T1, quantity=-2.0)
    cfg = PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR), stencil="first_order")
    res = explain(s0, s1, config=cfg)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    # No lifecycle event: the t1 endpoint and state(all) are the same market priced on the same
    # seed-frozen context, so the event row is zero. Independent MC noise would show up here.
    event = [r for r in res.rows if r.factor is Factor.LIFECYCLE_EVENT][0]
    assert event.pnl == pytest.approx(0.0, abs=_tol(res.total_pnl))
    assert res.unexplained is not None          # reported, not asserted small
    again = explain(s0, s1, config=cfg)          # seeded: row for row identical
    assert [(r.term, r.pnl) for r in again.rows] == [(r.term, r.pnl) for r in res.rows]


def test_frame_order_is_hash_seed_independent(tmp_path):
    script = tmp_path / "order.py"
    script.write_text(
        "import json\n"
        "from datetime import datetime\n"
        "from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine\n"
        "from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption\n"
        "from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote\n"
        "from quantark.param.div import ContinuousDividendYield\n"
        "from quantark.pnlexplain import ValuationSnapshot, explain\n"
        "from quantark.priceenv import PricingEnvironment\n"
        "from quantark.util.enum import OptionType\n"
        "def env(s, d, v=0.2):\n"
        "    return PricingEnvironment(spot_quote=SpotQuote(spot=s), vol_surface=FlatVolSurface(volatility=v),\n"
        "        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=d)\n"
        "eng = BlackScholesEngine()\n"
        "s0 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0), eng, env(100.0, datetime(2026,6,26)), date=datetime(2026,6,26))\n"
        "s1 = ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0-3/365), eng, env(103.0, datetime(2026,6,29), 0.22), date=datetime(2026,6,29))\n"
        "res = explain(s0, s1)\n"
        "print(json.dumps(res.to_dict(), default=str))\n"
        "from quantark.pnlexplain import BookSnapshot, PositionSnapshot, explain_portfolio\n"
        "b0 = BookSnapshot(datetime(2026,6,26), {'a': PositionSnapshot('a', 'X', s0)}, {'X': s0.pricing_env})\n"
        "b1 = BookSnapshot(datetime(2026,6,29), {'a': PositionSnapshot('a', 'X', s1)}, {'X': s1.pricing_env})\n"
        "print(explain_portfolio(b0, b1).to_frame().to_csv(index=False))\n"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True,
                             env={**os.environ, "PYTHONHASHSEED": seed})
        outs.add(out.stdout.strip())
    assert len(outs) == 1


EXPECTED_ALL = [
    "BookSnapshot", "ExplainMethod", "ExplainRow", "ExplainTrade", "FRAME_COLUMNS", "Factor",
    "FactorCoordinate", "FactorMoves", "LifecycleTransition", "MARKET_FACTORS", "MOVE_KEYS",
    "PnLExplainConfig", "PnLExplainRecorder", "PnLExplainResult", "PortfolioExplainResult",
    "PositionExplainResult", "PositionSnapshot", "QuotedLegSnapshot", "RECON_COLUMNS",
    "ReplayPnLExplainRecorder", "RowKind", "ValuationSnapshot", "ValueBreakdown", "component_sum",
    "contract_fingerprint", "engines_equivalent", "explain", "explain_portfolio", "explain_position",
    "explain_quoted_leg", "lifecycle_fingerprint", "make_total_row", "rows_to_frame", "value",
]


def test_public_exports_match_spec():
    assert list(pnlexplain_pkg.__all__) == EXPECTED_ALL          # exact, ordered surface
    for name in EXPECTED_ALL:
        assert getattr(pnlexplain_pkg, name) is not None, name


def test_equity_subpackage_exports():
    import importlib
    eq = importlib.import_module("quantark.pnlexplain.equity")
    for name in ("BookSnapshot", "ExplainTrade", "LifecycleTransition", "PnLExplainRecorder",
                 "PositionSnapshot", "QuotedLegSnapshot", "ReplayPnLExplainRecorder", "ValuationSnapshot",
                 "explain", "explain_portfolio", "explain_position", "explain_quoted_leg", "value"):
        assert hasattr(eq, name), name

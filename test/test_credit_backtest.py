"""Tests for the credit backtest engine and spread-neutral strategy."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.backtest import CreditBacktestConfig, CreditBacktestEngine
from quantark.backtest.strategy import CreditSpreadNeutralStrategy
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment


def _portfolio():
    env = CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curve=FlatHazardCurve(hazard_rate=0.02),
    )
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": env})
    pf.add_position(
        product=CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
                    coupon_spread=0.01, side=ProtectionSide.SELL),
        quantity=1.0, entry_price=0.0, reference_entity="ACME",
        engine=CDSReducedFormEngine(),
    )
    return pf


def _path(days=40, seed=2):
    rng = np.random.default_rng(seed)
    hazard = np.clip(0.02 + np.cumsum(rng.normal(0, 0.0006, days)), 0.001, None)
    rate = 0.03 + np.cumsum(rng.normal(0, 0.0002, days))
    idx = pd.date_range("2026-06-15", periods=days, freq="B")
    return pd.DataFrame({"ACME_hazard": hazard, "ACME_rate": rate}, index=idx)


def test_config_positional_construction_preserves_metadata_slot():
    # Legacy positional signature: hedge_recovery was appended after metadata so
    # the 6th positional argument still binds to metadata (not hedge_recovery).
    meta = {"book": "macro"}
    cfg = CreditBacktestConfig(
        _portfolio(), _path(), CreditSpreadNeutralStrategy(cs01_threshold=500.0),
        None, True, meta,
    )
    assert cfg.metadata == meta
    assert cfg.hedge_recovery == pytest.approx(0.4)  # ISDA standard default


def test_backtest_runs_and_records_steps():
    cfg = CreditBacktestConfig(
        portfolio=_portfolio(), market_path=_path(),
        strategy=CreditSpreadNeutralStrategy(cs01_threshold=1_000.0),
    )
    results = CreditBacktestEngine(cfg).run()
    assert len(results.rows) == len(cfg.market_path)
    assert results.num_hedges > 0


def test_hedging_reduces_pnl_volatility():
    cfg = CreditBacktestConfig(
        portfolio=_portfolio(), market_path=_path(),
        strategy=CreditSpreadNeutralStrategy(cs01_threshold=500.0),
    )
    results = CreditBacktestEngine(cfg).run()
    eff = results.get_hedge_effectiveness()
    # The CS01 hedge should cut spread-driven P&L volatility.
    assert eff["vol_reduction_pct"] > 0


def test_config_rejects_non_credit_portfolio():
    import pytest
    with pytest.raises(Exception):
        CreditBacktestConfig(portfolio=object(), market_path=_path(),
                             strategy=CreditSpreadNeutralStrategy())

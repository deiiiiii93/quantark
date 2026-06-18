"""Smoke tests for the SA-CCR HTML dashboard (quantark.saccr.dashboard)."""

import pytest

from quantark.saccr import (
    SACCRCalculator,
    SACCRDashboard,
    SACCRTrade,
    SACCRNettingSet,
    AssetClass,
    Position,
    OptionType,
    CommodityType,
)

pytest.importorskip("plotly")


def _mixed_netting_set() -> SACCRNettingSet:
    trades = [
        SACCRTrade(
            "IR_SWAP_10Y", AssetClass.INTEREST_RATE, notional=10_000, market_value=30,
            currency="USD", start_date=0, end_date=10, maturity=10,
            position=Position.LONG,
        ),
        SACCRTrade(
            "EUR_SWAPTION", AssetClass.INTEREST_RATE, notional=5_000, market_value=50,
            currency="EUR", start_date=1, end_date=11, maturity=5.5,
            position=Position.LONG, is_option=True, option_type=OptionType.PUT,
            underlying_price=0.06, strike_price=0.05, exercise_date=1.0,
        ),
        SACCRTrade(
            "EQ_FWD", AssetClass.EQUITY, notional=3_000, market_value=40,
            reference_entity="AAPL", maturity=2.0, position=Position.LONG,
        ),
        SACCRTrade(
            "COMM_OIL", AssetClass.COMMODITY, notional=6_000, market_value=10,
            commodity_type=CommodityType.CRUDE_OIL, maturity=1.0, position=Position.LONG,
        ),
    ]
    return SACCRNettingSet("TEST_NS", trades, is_margined=False)


def test_dashboard_writes_html_with_core_content(tmp_path):
    ns = _mixed_netting_set()
    result = SACCRCalculator().calculate(ns)

    out = tmp_path / "saccr_dashboard.html"
    returned = SACCRDashboard(result, netting_set=ns, currency="USD").generate(out)

    assert returned == str(out)
    html = out.read_text(encoding="utf-8")

    # Self-contained HTML document referencing the Plotly bundle.
    assert html.startswith("<!DOCTYPE html>")
    assert "plotly" in html.lower()

    # Core SA-CCR vocabulary and section headers are present.
    for token in ["SA-CCR", "EAD", "Replacement Cost", "PFE", "Add-On",
                  "EAD COMPOSITION", "HEDGING SET DETAIL", "NETTING SET CONTEXT"]:
        assert token in html

    # Asset-class labels for the trades we supplied.
    for label in ["Interest Rate", "Equity", "Commodity"]:
        assert label in html

    # The netting-set id and trade count surface in the context section.
    assert "TEST_NS" in html
    assert "4 trades" in html


def test_dashboard_requires_plotly_flag(monkeypatch):
    """Constructing without Plotly raises a clear ImportError."""
    import quantark.saccr.dashboard as dash

    monkeypatch.setattr(dash, "HAS_PLOTLY", False)
    with pytest.raises(ImportError):
        dash.SACCRDashboard(object())


def test_capped_badge_renders(tmp_path):
    """A margined set whose EAD is capped shows the para-129 cap badge."""
    ns = _mixed_netting_set()
    result = SACCRCalculator().calculate(ns)
    # Force the capped presentation path without re-deriving the calculation.
    result.ead_capped = True
    result.ead_uncapped = result.ead + 100.0
    result.is_margined = True

    out = tmp_path / "capped.html"
    SACCRDashboard(result, netting_set=ns).generate(out)
    html = out.read_text(encoding="utf-8")
    assert "para 129" in html
    assert "α (capped)" in html

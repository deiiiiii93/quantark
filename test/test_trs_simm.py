"""Gate 5 tests: equity TRS in the ISDA SIMM engine.

A TRS is delta-one equity exposure. EquitySwapPosition implements
SIMMSensitivityProvider; the equity sensitivity engine derives a single
EquityDelta from the position's quantity-scaled delta (no vega), which flows
through SIMMPortfolioAdapter -> SIMMCalculator to an initial-margin number.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.util.calendar.business_calendar import Calendar
from quantark.asset.equity.product.swap import (
    AssetParams,
    FixLegParams,
    FloatLegParams,
    PricingParams,
    TRSParams,
    OneAssetTotalReturnSwap,
)
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.portfolio import EquityPortfolio, EquitySwapPosition
from quantark.simm import SIMMConfig
from quantark.simm.engines.aggregation import SIMMCalculator
from quantark.simm.engines.base import SIMMSensitivityProvider
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.util.exceptions import ValidationError

START, END, VALUATION = "2024-01-02", "2024-01-31", "2024-01-30"
INITIAL, VAL_SPOT, NOTIONAL = 100.0, 110.0, 1_000_000.0


def _make_trs() -> OneAssetTotalReturnSwap:
    days = pd.date_range(START, END, freq="D")
    idx = [d.strftime("%Y-%m-%d") for d in days]
    val_i = idx.index(VALUATION)
    path = pd.Series(
        [INITIAL + (VAL_SPOT - INITIAL) * min(i, val_i) / val_i for i in range(len(idx))],
        index=idx,
    ).round(4)
    calendar = Calendar(name="DemoCalendar")
    return OneAssetTotalReturnSwap(
        TRSParams(
            contract_id="TRS_SIMM",
            asset=AssetParams("IDX", INITIAL, path),
            fix_leg=FixLegParams(
                rate=0.048, notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar, direction=-1,
            ),
            float_leg=FloatLegParams(
                notional=NOTIONAL, initial_notional=NOTIONAL,
                start_date=START, end_date=END, payment_calendar=calendar, direction=1,
            ),
            pricing=PricingParams(valuation_date=VALUATION, output_mode="spot"),
        )
    )


def _portfolio(quantity=1.0) -> EquityPortfolio:
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=VAL_SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.048),
        valuation_date=datetime(2024, 1, 30),
    )
    pf = EquityPortfolio(portfolio_name="eq", pricing_environments={"IDX": env})
    pf.add_swap_position(product=_make_trs(), quantity=quantity)
    return pf


def test_swap_position_is_simm_provider():
    assert isinstance(_make_trs(), OneAssetTotalReturnSwap)
    pos = EquitySwapPosition(product=_make_trs(), quantity=1.0)
    assert isinstance(pos, SIMMSensitivityProvider)


def test_swap_generates_equity_delta_and_margin():
    pf = _portfolio(quantity=1.0)
    config = SIMMConfig(
        calculation_currency="USD", calculate_delta=True, calculate_vega=True
    )
    sens = SIMMPortfolioAdapter(config).portfolio_to_sensitivities(pf)

    risk_types = {type(s).__name__ for s in sens.sensitivities}
    assert "EquityDeltaSensitivity" in risk_types
    assert "EquityVegaSensitivity" not in risk_types  # a TRS has no vega

    # SIMM equity delta = 0.01 * spot * delta_$, delta_$ = qty * notional/initial.
    eq_delta = next(
        s for s in sens.sensitivities if type(s).__name__ == "EquityDeltaSensitivity"
    )
    expected = 0.01 * VAL_SPOT * (NOTIONAL / INITIAL)
    assert float(eq_delta.amount) == pytest.approx(expected, rel=1e-6)

    result = SIMMCalculator(config).calculate(sens)
    assert result.total_margin > 0


def test_simm_rejects_non_usd_calc_currency():
    pf = _portfolio()
    config = SIMMConfig(calculation_currency="EUR", calculate_delta=True)
    with pytest.raises(ValidationError):
        SIMMPortfolioAdapter(config).portfolio_to_sensitivities(pf)

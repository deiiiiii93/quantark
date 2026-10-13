"""A DCN held through multi-day runs (W14b, T13): one booking, one spot path, three run consumers.

``DCN_A`` booked on 2023-03-27 (after its first rows are locked). The path holds 5500 (a coupon on the 2023-04-03 row,
paid 2023-04-06 after the Qingming holiday), dips to 4400 on 2023-04-11/12 (a daily knock-in), recovers to 5500 and
jumps to 6100 for the 2023-05-04 row (a knock-out, paid 2023-05-08). Equity spots do not move on closed days.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dcn_fixtures import DCN_A, FLAT, SSE, make_dcn

BOOKED = datetime(2023, 3, 27)
RUN_END = datetime(2023, 5, 12)
UNDERLYING = "CSI500"
COUPON_ROW, COUPON_PAY = datetime(2023, 4, 3), datetime(2023, 4, 6)
KI_DAY = datetime(2023, 4, 11)
KO_ROW, KO_PAY = datetime(2023, 5, 4), datetime(2023, 5, 8)


def booked_dcn(**overrides):
    return make_dcn(DCN_A, valuation_date=BOOKED, **overrides)


def spot_on(d) -> float:
    d = pd.Timestamp(d).to_pydatetime()
    if d < KI_DAY:
        return 5500.0
    if d < datetime(2023, 4, 13):
        return 4400.0
    if d < KO_ROW:
        return 5500.0
    return 6100.0


def calendar_days(start=BOOKED, end=RUN_END):
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def sse_days(start=BOOKED, end=RUN_END):
    return [d for d in calendar_days(start, end) if SSE.is_business_day(d)]


def nyse_like_days(start=BOOKED, end=RUN_END, missing=(COUPON_ROW,)):
    """SSE trading days with ``missing`` closed (another market's holiday): those observations become gap days."""
    return [d for d in sse_days(start, end) if d not in set(missing)]


def flat_env_on(d, spot=None):
    from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import DayCountConvention

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot_on(d) if spot is None else spot, asset_name=UNDERLYING),
        vol_surface=FlatVolSurface(volatility=FLAT["sigma"]), rate_curve=FlatRateCurve(rate=FLAT["r"]),
        div_yield=ContinuousDividendYield(div_yield=FLAT["q"]), valuation_date=d,
        day_count_convention=DayCountConvention.ACT_365)


# --- dynamic scenario (calendar days from the path's start date) -------------------------------------------------

def dynamic_run(engine, product=None, days=None, quantity=1.0, handle=True):
    """A dynamic scenario stepping ``days`` (calendar days from BOOKED by default); spot set from ``spot_on``."""
    from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine
    from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
    from quantark.portfolio import Portfolio
    from quantark.stresstest.stress.stress_types import StressType

    days = calendar_days() if days is None else days
    if any((b - a).days != 1 for a, b in zip(days, days[1:])):
        raise ValueError("a DayPath steps calendar days")
    steps = [DayStep(i, [ParameterChange("spot", StressType.VALUE, spot_on(d))]) for i, d in enumerate(days)]
    path = DayPath(name="dcn-run", steps=steps, start_date=days[0])
    portfolio = Portfolio(portfolio_name="dcn", pricing_environments={UNDERLYING: flat_env_on(days[0])})
    portfolio.add_position(product=product or booked_dcn(), quantity=quantity, entry_price=0.0,
                           underlying=UNDERLYING, engine=engine)
    config = DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=handle)
    return DynamicScenarioEngine(config).run(portfolio, path)


# --- equity backtest -----------------------------------------------------------------------------------------------

class PathAdapter:
    """Market-data adapter over explicit run dates; flat vol / rate / dividend."""

    def __init__(self, dates):
        self.dates = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])

    def get_market_data_set(self, asset_name, start_date, end_date, currency, frequency):
        from quantark.util.marketdata.models import MarketDataSet, TimeSeriesData

        idx = self.dates

        def series(name, values):
            return TimeSeriesData(pd.DataFrame({name: values}, index=idx), asset_name, name)

        return MarketDataSet(
            spot_data=series("spot", [spot_on(d) for d in idx]),
            vol_data=series("volatility", [FLAT["sigma"]] * len(idx)),
            rate_data=series("rate", [FLAT["r"]] * len(idx)),
            div_yield_data=series("div_yield", [FLAT["q"]] * len(idx)),
        )


def backtest_run(engine, product=None, days=None, quantity=1.0, handle=True, pnl_explain=None):
    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.portfolio import Position

    days = sse_days() if days is None else days
    adapter = PathAdapter(days)
    position = Position(product=product or booked_dcn(), quantity=quantity, entry_price=0.0,
                        underlying=UNDERLYING, engine=engine, entry_timestamp=days[0])
    config = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=days[0], end_date=days[-1],
        underlying=UNDERLYING, initial_positions=[position], market_data_adapter=adapter,
        transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=handle, calculate_greeks=False,
        logging_level="ERROR", pnl_explain=pnl_explain)
    return BacktestEngine(config).run()


# --- replay (spot hedge) -------------------------------------------------------------------------------------------

def replay_market(days=None):
    from quantark.backtest.replay import AutocallableMarketDataSet

    days = sse_days() if days is None else days
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    # one listed contract, always live through the run (>= 7 days to expiry for a futures_curve carry)
    expiry = max(pd.Timestamp("2023-12-15"), idx[-1] + pd.Timedelta(days=200))
    spots = np.array([spot_on(d) for d in idx])
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": idx, "spot": spots}),
        vol_data=pd.DataFrame({"date": idx, "volatility": np.full(len(idx), FLAT["sigma"])}),
        rate_data=pd.DataFrame({"date": idx, "rate": np.full(len(idx), FLAT["r"])}),
        futures_data=pd.DataFrame({"date": idx, "contract": ["IC2312"] * len(idx), "futures_price": spots,
                                   "expiry_date": [expiry] * len(idx),
                                   "multiplier": [200.0] * len(idx)}),
    )


def replay_run(engine_config, product=None, days=None, quantity=1.0, fixed_dividend_yield=FLAT["q"], pnl_explain=None):
    from quantark.backtest.replay import (
        BookAutocallableBacktestConfig, BookAutocallableBacktestEngine, BookProduct, HedgeSpec,
    )

    days = sse_days() if days is None else days
    cfg = BookAutocallableBacktestConfig(
        products=[BookProduct(product=product or booked_dcn(), quantity=quantity, position_id=1,
                              has_lifecycle=True)],
        market_data=replay_market(days), hedge=HedgeSpec(kind="spot", multiplier=1.0),
        engine_config=engine_config, underlying=UNDERLYING, start_date=days[0], end_date=days[-1],
        fixed_dividend_yield=fixed_dividend_yield, calculate_event_probabilities=False,
        terminate_on_lifecycle_end=False, pnl_explain=pnl_explain,
    )
    return BookAutocallableBacktestEngine(cfg).run()

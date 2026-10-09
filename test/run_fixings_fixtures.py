"""Shared fixture for the run-fixings tests (spec docs/superpowers/specs/2026-10-09-run-fixings-design.md §2.1).

Daily path from START: spot 105 (inside [90, 110]) through day 4, then +1 a day (out of the band from day 11).
Fixings on day 4, day 18, day 120, day 240, day 365. Numeric twins: the same records with time = days / 365.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from quantark.asset.equity.engine.analytical import AsianOptionAnalyticalEngine, RangeAccrualAnalyticalEngine
from quantark.asset.equity.product.option import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity.portfolio import EquityPortfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AsianStrikeType, AveragingType, OptionType
from quantark.util.fixings import MissingFixingWarning
from quantark.util.marketdata.models import MarketDataSet, TimeSeriesData

START = datetime(2024, 1, 1)          # a Monday
DAYS = 25
FIX_DAY = 4
FIX2_DAY = 18
VOL, RATE, DIV = 0.25, 0.05, 0.02
UNDERLYING = "EQUITY_INDEX"
ENGINES = {"ra": RangeAccrualAnalyticalEngine, "asian": AsianOptionAnalyticalEngine}


def spot_on(day: int) -> float:
    return 105.0 if day <= FIX_DAY else 105.0 + (day - FIX_DAY)


def fixing_dates() -> List[datetime]:
    return [START + timedelta(days=d) for d in (FIX_DAY, FIX2_DAY, 120, 240, 365)]


def day_of(d: datetime) -> int:
    return (d - START).days


def _range_config() -> RangeAccrualConfig:
    return RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10)


def ra_dated(outcomes: Optional[Dict[int, bool]] = None, dates: Optional[List[datetime]] = None,
             initial_date: datetime = START) -> RangeAccrualOption:
    outcomes = outcomes or {}
    return RangeAccrualOption(
        initial_price=100.0, range_config=_range_config(), exercise_date=START + timedelta(days=365),
        initial_date=initial_date,
        observation_records=[RangeAccrualObservationRecord(observation_date=d, observed_in_range=outcomes.get(i))
                             for i, d in enumerate(dates or fixing_dates())],
    )


def ra_numeric(outcomes: Optional[Dict[int, bool]] = None, times: Optional[List[float]] = None,
               maturity: float = 1.0) -> RangeAccrualOption:
    outcomes = outcomes or {}
    times = times if times is not None else [day_of(d) / 365.0 for d in fixing_dates()]
    return RangeAccrualOption(
        initial_price=100.0, range_config=_range_config(), maturity=maturity,
        observation_records=[RangeAccrualObservationRecord(observation_time=t, observed_in_range=outcomes.get(i))
                             for i, t in enumerate(times)],
    )


def asian_dated(prices: Optional[Dict[int, float]] = None) -> AsianOption:
    prices = prices or {}
    return AsianOption(
        strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC, exercise_date=START + timedelta(days=365),
        observation_records=[AsianObservationRecord(observation_date=d, observed_price=prices.get(i))
                             for i, d in enumerate(fixing_dates())],
    )


def asian_numeric(prices: Optional[Dict[int, float]] = None) -> AsianOption:
    prices = prices or {}
    return AsianOption(
        strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC, maturity=1.0,
        observation_records=[AsianObservationRecord(observation_time=day_of(d) / 365.0, observed_price=prices.get(i))
                             for i, d in enumerate(fixing_dates())],
    )


MAKERS = {("ra", "dated"): ra_dated, ("ra", "numeric"): ra_numeric,
          ("asian", "dated"): asian_dated, ("asian", "numeric"): asian_numeric}


def eq_env(day: int, *, spot: Optional[float] = None, date: Optional[datetime] = None, **kw) -> PricingEnvironment:
    return PricingEnvironment(
        valuation_date=date if date is not None else START + timedelta(days=day),
        spot_quote=SpotQuote(spot=spot if spot is not None else spot_on(day)),
        vol_surface=FlatVolSurface(volatility=VOL), rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIV), **kw,
    )


def hand_outcomes(kind: str, last_day: int, spot_of: Callable[[int], float] = spot_on) -> Dict[int, object]:
    """Outcomes of every fixing dated on/before run day ``last_day``, decided at ITS OWN day's spot."""
    passed = {i: day_of(d) for i, d in enumerate(fixing_dates()) if day_of(d) <= last_day}
    if kind == "ra":
        prod = ra_dated()
        recs = prod.get_observation_records()
        return {i: prod.is_in_range(spot_of(k), i, recs[i]) for i, k in passed.items()}
    return {i: spot_of(k) for i, k in passed.items()}


def by_hand(kind: str, last_day: int, form: str = "dated", spot_of: Callable[[int], float] = spot_on):
    """The booked contract with every fixing dated on/before ``last_day`` recorded by hand at its own day's spot."""
    return MAKERS[(kind, form)](hand_outcomes(kind, last_day, spot_of))


def quiet_price(engine, product, env) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        return float(engine.price(product, env))


def count_missing(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, sum(issubclass(w.category, MissingFixingWarning) for w in caught)


def eq_book(*products, underlying: str = UNDERLYING, env: Optional[PricingEnvironment] = None) -> EquityPortfolio:
    pf = EquityPortfolio(portfolio_name="w6", pricing_environments={underlying: env or eq_env(0)},
                         creation_date=START)
    for p in products:
        kind = "asian" if isinstance(p, AsianOption) else "ra"
        pf.add_position(product=p, quantity=1.0, entry_price=0.0, underlying=underlying, engine=ENGINES[kind](),
                        entry_timestamp=START)
    return pf


def move(pf, day: int, spot: Optional[float] = None, underlying: str = UNDERLYING,
         date: Optional[datetime] = None) -> None:
    """Apply run day ``day``'s market to ``pf`` in place, as the run engines do (env mutated, date then spot)."""
    env = pf.pricing_environments[underlying]
    env.valuation_date = date if date is not None else START + timedelta(days=day)
    env.spot_quote = SpotQuote(spot=spot if spot is not None else spot_on(day))


class DailyAdapter:
    """Market data adapter over START .. at ``spot_of`` (``freq`` any pandas frequency, e.g. ``W-MON``, ``h``)."""

    def __init__(self, n: int, freq: str = "D", spot_of: Callable[[int], float] = spot_on):
        self.dates = pd.date_range(START, periods=n, freq=freq)
        self.days = [(d.to_pydatetime() - START).days for d in self.dates]
        self.spot_of = spot_of

    def get_market_data_set(self, asset_name, start_date, end_date, currency, frequency):
        idx = self.dates
        frame = lambda col, vals: TimeSeriesData(pd.DataFrame({col: vals}, index=idx), asset_name, col)  # noqa: E731
        return MarketDataSet(
            spot_data=frame("spot", [self.spot_of(k) for k in self.days]),
            vol_data=frame("volatility", [VOL] * len(idx)), rate_data=frame("rate", [RATE] * len(idx)),
            div_yield_data=frame("div_yield", [DIV] * len(idx)),
        )


# ---------------------------------------------------------------- FX (EURUSD range [1.10, 1.30], notional 1e6)
import numpy as np  # noqa: E402

from quantark.asset.fx.engine.analytical import (  # noqa: E402
    FxQuantoRangeAccrualAnalyticalEngine, FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.engine.analytical.fx_foreign_range_accrual_engine import (  # noqa: E402
    FxForeignRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product import CurrencyPair  # noqa: E402
from quantark.asset.fx.product.option import (  # noqa: E402
    FxQuantoRangeAccrualOption, FxRangeAccrualConfig, FxRangeAccrualObservationRecord, FxRangeAccrualOption,
)
from quantark.asset.fx.product.option.fx_foreign_range_accrual_option import FxForeignRangeAccrualOption  # noqa: E402
from quantark.portfolio.fx import FXPortfolio  # noqa: E402
from quantark.priceenv import FxPricingEnvironment  # noqa: E402
from quantark.priceenv.fx_pricing_environment import FxQuantoMarketData, QuantoConversionOrientation  # noqa: E402

PAIR = CurrencyPair("EUR", "USD")
FX_CLASSES = {
    "domestic": (FxRangeAccrualOption, FxRangeAccrualAnalyticalEngine),
    "foreign": (FxForeignRangeAccrualOption, FxForeignRangeAccrualAnalyticalEngine),
    "quanto": (FxQuantoRangeAccrualOption, FxQuantoRangeAccrualAnalyticalEngine),
}


def fx_spot(day: int) -> float:
    return 1.25 if day <= FIX_DAY else 1.25 + 0.01 * (day - FIX_DAY)


def fx_env(day: int, *, spot: Optional[float] = None, quanto: bool = False, date: Optional[datetime] = None,
           **kw) -> FxPricingEnvironment:
    q = FxQuantoMarketData(settlement_curve=FlatRateCurve(rate=0.01), quanto_vol=0.08, correlation=0.3,
                           conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC)
    return FxPricingEnvironment(
        valuation_date=date if date is not None else START + timedelta(days=day),
        spot_quote=SpotQuote(spot=fx_spot(day) if spot is None else spot),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10), quanto=q if quanto else None, **kw,
    )


def fx_ra(kind: str, form: str, outcomes: Optional[Dict[int, bool]] = None,
          dates: Optional[List[datetime]] = None, times: Optional[List[float]] = None):
    outcomes = outcomes or {}
    cls = FX_CLASSES[kind][0]
    extra = {"quanto_fx_rate": 110.0} if kind == "quanto" else {}
    cfg = FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04)
    if form == "dated":
        recs = [FxRangeAccrualObservationRecord(observation_date=d, observed_in_range=outcomes.get(i))
                for i, d in enumerate(dates or fixing_dates())]
        return cls(notional=1_000_000.0, range_config=cfg, currency_pair=PAIR, expiry_date=START + timedelta(days=365),
                   initial_date=START, observation_records=recs, **extra)
    times = times if times is not None else [day_of(d) / 365.0 for d in fixing_dates()]
    recs = [FxRangeAccrualObservationRecord(observation_time=t, observed_in_range=outcomes.get(i))
            for i, t in enumerate(times)]
    return cls(notional=1_000_000.0, range_config=cfg, currency_pair=PAIR, maturity=1.0, observation_records=recs,
               **extra)


def fx_hand_outcomes(kind: str, last_day: int, spot_of: Callable[[int], float] = fx_spot) -> Dict[int, bool]:
    prod = fx_ra(kind, "dated")
    recs = prod.get_observation_records()
    return {i: prod.is_in_range(spot_of(day_of(d)), i, recs[i])
            for i, d in enumerate(fixing_dates()) if day_of(d) <= last_day}


def fx_book(*products, env: Optional[FxPricingEnvironment] = None) -> FXPortfolio:
    pf = FXPortfolio(portfolio_name="w6fx", pricing_environments={"EURUSD": env or fx_env(0)})
    for p in products:
        kind = next(k for k, (cls, _) in FX_CLASSES.items() if type(p) is cls)
        pf.add_position(product=p, quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=FX_CLASSES[kind][1]())
    return pf


def fx_path(index, spot_of: Callable[[int], float] = fx_spot) -> pd.DataFrame:
    days = [(pd.Timestamp(t).to_pydatetime() - START).days for t in index]
    n = len(index)
    return pd.DataFrame({"EURUSD_spot": [spot_of(k) for k in days], "EURUSD_vol": np.full(n, 0.10),
                         "EURUSD_dom_rate": np.full(n, 0.05), "EURUSD_for_rate": np.full(n, 0.03)}, index=index)

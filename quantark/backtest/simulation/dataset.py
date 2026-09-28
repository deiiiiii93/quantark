"""One simulated path as the replay engine's ``AutocallableMarketDataSet`` (spec 6, 10)."""
from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from quantark.backtest.replay import AutocallableMarketDataSet
from quantark.util.exceptions import ValidationError

from .carry import FUTURES_MULTIPLIER, day_chain
from .dividends import CurveTailPillars
from .paths.market_path import MarketPath


@dataclass(frozen=True)
class SimulatedCarryTail:
    """One day's IV-artifact stand-in for the replay: the path's carry curve as the tail.

    The replay's ``surface_forward_carry`` dividend reads only
    ``implied_q_pillars`` from the day's artifact and records the provenance
    fields below.  The simulation hands ``dividend_yield_for_day`` a
    ``CurveTailPillars`` over the same tenors and carry row, so both engines
    build the dividend object from identical inputs.
    """

    trade_date: _dt.date
    pillars: CurveTailPillars

    def implied_q_pillars(self, rate: float):
        return self.pillars.implied_q_pillars(rate)

    @property
    def sha256(self) -> str:
        tenors = np.ascontiguousarray(self.pillars.tenors, dtype=np.float64)
        carry = np.ascontiguousarray(self.pillars.carry, dtype=np.float64)
        return hashlib.sha256(tenors.tobytes() + carry.tobytes()).hexdigest()

    @property
    def extrapolation_policy(self) -> Dict[str, Any]:
        return {"beyond_last_listed_expiry": "simulated_carry_curve"}

    @property
    def max_listed_T(self) -> float:
        return float(self.pillars.tenors[-1])


class SimulatedCarryTailHistory:
    """``surface_history`` for one simulated path: ``surface_for`` on its own calendar days only.

    It stands in for the dividend tail and nothing else; a replay setting
    that needs real IV surfaces (``vol_source='surface'``, a vol model)
    still fails where it reads a surface this object does not have.
    """

    def __init__(self, path: MarketPath) -> None:
        if path.n_paths != 1:
            raise ValidationError(f"a carry-tail history describes one path, got {path.n_paths}")
        self._tenors = np.asarray(path.tenor_grid, dtype=float)
        self._carry = path.carry[0]
        self._rows = {pd.Timestamp(d).normalize(): i for i, d in enumerate(path.dates)}

    def surface_for(self, d: Any) -> SimulatedCarryTail:
        key = pd.Timestamp(d).normalize()
        row = self._rows.get(key)
        if row is None:
            raise ValidationError(f"no simulated carry curve on {key.date()}; the path's calendar does not list it")
        return SimulatedCarryTail(trade_date=key.date(),
                                  pillars=CurveTailPillars(tenors=self._tenors, carry=self._carry[row, :]))


def to_market_dataset(
    path: MarketPath, path_index: int, *, multiplier: float = FUTURES_MULTIPLIER, carry_tail: bool = False
) -> AutocallableMarketDataSet:
    """The replay engine's dataset for one path of a batch.

    ``carry_tail`` attaches a ``SimulatedCarryTailHistory``: the replay's
    ``surface_forward_carry`` dividend then takes its tail from the path's
    carry curve, exactly as the simulation does.
    """
    single = path.path(path_index)
    dates = list(single.dates)
    spot = pd.DataFrame({"date": dates, "spot": single.spot[0]})
    vol = pd.DataFrame({"date": dates, "volatility": single.atm_vol[0]})
    rate = pd.DataFrame({"date": dates, "rate": single.rate[0]})
    futures = pd.concat([day_chain(single, d, multiplier=multiplier).frame(0) for d in range(single.n_days)],
                        ignore_index=True)
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=spot, vol_data=vol, rate_data=rate, futures_data=futures,
        surface_history=SimulatedCarryTailHistory(single) if carry_tail else None,
    )

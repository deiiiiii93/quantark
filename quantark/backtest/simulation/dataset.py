"""One simulated path as the replay engine's ``AutocallableMarketDataSet`` (spec 6, 10)."""
from __future__ import annotations

import pandas as pd

from quantark.backtest.replay import AutocallableMarketDataSet

from .carry import FUTURES_MULTIPLIER, day_chain
from .paths.market_path import MarketPath


def to_market_dataset(
    path: MarketPath, path_index: int, *, multiplier: float = FUTURES_MULTIPLIER
) -> AutocallableMarketDataSet:
    """The replay engine's dataset for one path of a batch."""
    single = path.path(path_index)
    dates = list(single.dates)
    spot = pd.DataFrame({"date": dates, "spot": single.spot[0]})
    vol = pd.DataFrame({"date": dates, "volatility": single.atm_vol[0]})
    rate = pd.DataFrame({"date": dates, "rate": single.rate[0]})
    futures = pd.concat([day_chain(single, d, multiplier=multiplier).frame(0) for d in range(single.n_days)],
                        ignore_index=True)
    return AutocallableMarketDataSet.from_dataframes(spot_data=spot, vol_data=vol, rate_data=rate, futures_data=futures)

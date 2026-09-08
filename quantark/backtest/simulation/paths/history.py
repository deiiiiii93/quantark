"""``PathHistory``: the realised joint market history as a daily state series (spec 5.1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from ..carry import curve_from_chain
from .market_path import StartState, _validate_tenor_grid


@dataclass(frozen=True)
class PathHistory:
    """The realised market as the same joint state the generators evolve."""

    dates: pd.DatetimeIndex
    spot: np.ndarray      # (n_days,)
    atm_vol: np.ndarray   # (n_days,)
    rate: np.ndarray      # (n_days,)
    carry: np.ndarray     # (n_days, n_tenors)
    tenor_grid: np.ndarray
    source_fingerprint: str

    @property
    def n_days(self) -> int:
        return int(self.spot.size)

    @classmethod
    def from_frames(
        cls,
        *,
        spot: pd.DataFrame,
        vol: pd.DataFrame,
        futures: pd.DataFrame,
        rate: Union[float, pd.DataFrame],
        tenor_grid: np.ndarray,
    ) -> "PathHistory":
        """Build from the study's market frames; days present in all inputs are kept."""
        grid = _validate_tenor_grid(tenor_grid)
        s = spot[["date", "spot"]].assign(date=pd.to_datetime(spot["date"]).dt.normalize()).set_index("date")["spot"]
        v = vol[["date", "volatility"]].assign(date=pd.to_datetime(vol["date"]).dt.normalize()).set_index("date")["volatility"]
        if isinstance(rate, pd.DataFrame):
            r = rate[["date", "rate"]].assign(date=pd.to_datetime(rate["date"]).dt.normalize()).set_index("date")["rate"]
        else:
            r = pd.Series(float(rate), index=s.index)
        fut = futures.assign(date=pd.to_datetime(futures["date"]).dt.normalize(),
                             expiry_date=pd.to_datetime(futures["expiry_date"]).dt.normalize())
        common = s.index.intersection(v.index).intersection(r.index).intersection(pd.DatetimeIndex(fut["date"].unique()))
        common = common.sort_values()
        if len(common) < 2:
            raise ValidationError("PathHistory needs at least two common days across spot, vol, rate and futures")
        carry = np.empty((len(common), grid.size))
        by_day = {d: g for d, g in fut.groupby("date")}
        for k, d in enumerate(common):
            g = by_day[d]
            tenors = ((g["expiry_date"] - d).dt.days / 365.0).to_numpy()
            try:
                carry[k] = curve_from_chain(float(s[d]), tenors, g["futures_price"].to_numpy(), grid)
            except ValidationError as exc:
                raise ValidationError(f"{d.date()}: {exc}") from exc
        h = hashlib.sha256()
        for arr in (common.asi8, s[common].to_numpy(float), v[common].to_numpy(float), r[common].to_numpy(float), carry):
            h.update(np.ascontiguousarray(arr).tobytes())
        return cls(
            dates=pd.DatetimeIndex(common), spot=s[common].to_numpy(float), atm_vol=v[common].to_numpy(float),
            rate=r[common].to_numpy(float), carry=carry, tenor_grid=grid, source_fingerprint=h.hexdigest(),
        )

    def levels(self) -> np.ndarray:
        """``[ln S, vol, rate, B_1 ... B_K]`` per day."""
        return np.column_stack([np.log(self.spot), self.atm_vol, self.rate, self.carry])

    def changes(self) -> np.ndarray:
        """Day-over-day differences of ``levels()``."""
        return np.diff(self.levels(), axis=0)

    def snapshot(self, day_index: int = -1) -> StartState:
        """The day's state as a generator start point (last day by default)."""
        return StartState(spot=float(self.spot[day_index]), atm_vol=float(self.atm_vol[day_index]),
                          rate=float(self.rate[day_index]), carry=self.carry[day_index].copy())

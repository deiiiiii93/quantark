"""
Market data helpers for OTC autocallable backtests.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math

import pandas as pd

from quantark.util.exceptions import ValidationError

from quantark.param.vol.surface_history import VolSurfaceHistory


@dataclass(frozen=True)
class SignedDividendYield:
    """
    Pricing-environment compatible dividend/carry yield adapter.

    The OTC backtest derives a non-negative implied ``q`` by default. This
    adapter only enforces finiteness so callers can still run explicit signed
    sensitivity scenarios when needed.
    """

    div_yield: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.div_yield)):
            raise ValidationError(f"div_yield must be finite, got {self.div_yield}")

    def get_yield(self, time_to_maturity: float) -> float:
        return float(self.div_yield)


@dataclass(frozen=True)
class ImpliedBasisYield:
    """Annualized futures basis yield compatible with PricingEnvironment."""

    basis_yield: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.basis_yield)):
            raise ValidationError(f"basis_yield must be finite, got {self.basis_yield}")

    def get_basis_yield(self, time_to_maturity: float) -> float:
        return float(self.basis_yield)


def calculate_basis_yield(
    spot: float, futures_price: float, time_to_maturity: float
) -> float:
    """
    Calculate annualized simple futures basis yield.

    ``time_to_maturity`` is expressed in years, so this is equivalent to
    ``(F - S) / S * 365 / days_to_maturity`` when using calendar-day year
    fractions.
    """
    if spot <= 0:
        raise ValidationError(f"spot must be positive, got {spot}")
    if futures_price <= 0:
        raise ValidationError(f"futures_price must be positive, got {futures_price}")
    if time_to_maturity <= 0:
        raise ValidationError(
            f"time_to_maturity must be positive, got {time_to_maturity}"
        )
    return (float(futures_price) - float(spot)) / float(spot) / float(
        time_to_maturity
    )


def derive_implied_dividend_yield(
    rate: float, spot: float, futures_price: float, time_to_maturity: float
) -> tuple[float, float]:
    """
    Return ``(basis_yield, implied_q)`` where ``q = max(0, r - basis_yield)``.
    """
    basis_yield = calculate_basis_yield(spot, futures_price, time_to_maturity)
    return basis_yield, max(0.0, float(rate) - basis_yield)


def _normalize_dates(series: pd.Series, column: str) -> pd.Series:
    try:
        return pd.to_datetime(series).dt.normalize()
    except Exception as exc:
        raise ValidationError(f"Could not parse {column} as dates") from exc


def normalize_time_series(
    df: pd.DataFrame, required_columns: list[str], date_column: str = "date"
) -> pd.DataFrame:
    """
    Normalize a dated market-data DataFrame to a sorted DatetimeIndex.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    out = df.copy()
    out[date_column] = _normalize_dates(out[date_column], date_column)
    out = out.sort_values(date_column).set_index(date_column)
    if out.index.has_duplicates:
        raise ValidationError(f"{date_column} values must be unique")
    return out


def normalize_futures_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize futures-chain data.

    Required columns: date, contract, futures_price, expiry_date, multiplier.
    """
    required = ["date", "contract", "futures_price", "expiry_date", "multiplier"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required futures columns: {missing}")

    out = df.copy()
    out["date"] = _normalize_dates(out["date"], "date")
    out["expiry_date"] = _normalize_dates(out["expiry_date"], "expiry_date")
    out["futures_price"] = out["futures_price"].astype(float)
    out["multiplier"] = out["multiplier"].astype(float)
    if (out["futures_price"] <= 0).any():
        raise ValidationError("futures_price must be positive")
    if (out["multiplier"] <= 0).any():
        raise ValidationError("multiplier must be positive")
    return out.sort_values(["date", "expiry_date", "contract"]).reset_index(drop=True)


@dataclass
class AutocallableMarketDataSet:
    """
    Normalized historical market data for OTC autocallable backtests.

    ``surface_history`` is an optional per-day IV-surface channel.  It does
    not participate in the ``dates`` calendar intersection: surfaces attach
    to the existing calendar dates via the manifest carry-forward gap policy
    (see :class:`quantark.param.vol.surface_history.VolSurfaceHistory`).
    """

    spot_data: pd.DataFrame
    vol_data: pd.DataFrame
    rate_data: pd.DataFrame
    futures_data: pd.DataFrame
    metadata: Dict[str, Any] | None = None
    surface_history: Optional[VolSurfaceHistory] = None

    @classmethod
    def from_dataframes(
        cls,
        *,
        spot_data: pd.DataFrame,
        vol_data: pd.DataFrame,
        rate_data: pd.DataFrame,
        futures_data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        surface_history: Optional[VolSurfaceHistory] = None,
    ) -> "AutocallableMarketDataSet":
        return cls(
            spot_data=normalize_time_series(spot_data, ["date", "spot"]),
            vol_data=normalize_time_series(vol_data, ["date", "volatility"]),
            rate_data=normalize_time_series(rate_data, ["date", "rate"]),
            futures_data=normalize_futures_chain(futures_data),
            metadata=metadata or {},
            surface_history=surface_history,
        )

    @property
    def dates(self) -> pd.DatetimeIndex:
        common = self.spot_data.index.intersection(self.vol_data.index)
        common = common.intersection(self.rate_data.index)
        futures_dates = pd.DatetimeIndex(self.futures_data["date"].unique())
        return common.intersection(futures_dates).sort_values()

    def get_market_row(self, date: pd.Timestamp) -> Dict[str, float]:
        date = pd.Timestamp(date).normalize()
        return {
            "spot": float(self.spot_data.loc[date, "spot"]),
            "volatility": float(self.vol_data.loc[date, "volatility"]),
            "rate": float(self.rate_data.loc[date, "rate"]),
        }

    def get_futures_slice(self, date: pd.Timestamp) -> pd.DataFrame:
        date = pd.Timestamp(date).normalize()
        by_date = getattr(self, "_futures_by_date", None)
        if by_date is None:
            # Grouped once (lazily): the per-day boolean filter over the full
            # chain was O(total_rows) per replay day.
            by_date = {
                key: frame for key, frame in self.futures_data.groupby("date")
            }
            object.__setattr__(self, "_futures_by_date", by_date)
        rows = by_date.get(date)
        if rows is None or rows.empty:
            raise ValidationError(f"No futures data for {date.date()}")
        return rows.copy()


class AKShareAutocallableDataAdapter:
    """
    Optional AKShare adapter boundary for OTC backtests.

    The normalizers are intentionally usable without importing AKShare so tests
    and offline workflows do not need network access or the optional dependency.
    """

    source_name = "AKShare"

    @staticmethod
    def normalize_spot(raw: pd.DataFrame) -> pd.DataFrame:
        return normalize_time_series(raw, ["date", "spot"]).reset_index()

    @staticmethod
    def normalize_vol(raw: pd.DataFrame) -> pd.DataFrame:
        return normalize_time_series(raw, ["date", "volatility"]).reset_index()

    @staticmethod
    def normalize_rate(raw: pd.DataFrame) -> pd.DataFrame:
        return normalize_time_series(raw, ["date", "rate"]).reset_index()

    @staticmethod
    def normalize_futures(raw: pd.DataFrame) -> pd.DataFrame:
        return normalize_futures_chain(raw)

    def _load_akshare(self):
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            raise ValidationError(
                "AKShare is not installed. Install akshare or pass normalized "
                "DataFrames directly to AutocallableMarketDataSet."
            ) from exc
        return ak

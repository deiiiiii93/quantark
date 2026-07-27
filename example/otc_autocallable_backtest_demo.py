"""
Real-data OTC autocallable backtest demo.

The demo builds a CSI500 Snowball and hedges it with listed IC futures using the
new ``backtest.otc`` module. AKShare is optional and loaded lazily; normalized
spot/futures data is cached so the demo can be rerun without another network
fetch.

Usage:
    python example/otc_autocallable_backtest_demo.py
    python example/otc_autocallable_backtest_demo.py --engine pde --refresh-data
    python example/otc_autocallable_backtest_demo.py --engine mc --mc-paths 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from calendar import monthcalendar, FRIDAY
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableBacktestDashboard,
    AutocallableDashboardConfig,
    AutocallableBacktestEngine,
    AutocallableDeltaHedgeStrategy,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    SurfaceGridConfig,
)
from quantark.util.enum.engine_enums import EngineType, PDEMethod


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "example" / "output" / "otc_autocallable_backtest"
DEFAULT_START_DATE = "2024-01-02"
DEFAULT_END_DATE = "2024-04-30"
CSI500_INDEX_SYMBOL = "000905"
CSI500_FUTURES_PREFIX = "IC"
IC_MULTIPLIER = 200.0


class DemoDataError(RuntimeError):
    """Raised when demo market data cannot be loaded."""


@dataclass(frozen=True)
class CachedMarketFrames:
    """Normalized frames persisted by the demo cache."""

    spot_data: pd.DataFrame
    futures_data: pd.DataFrame


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CSI500 Snowball OTC backtest hedged with IC futures."
    )
    parser.add_argument(
        "--engine",
        choices=("quad", "pde", "mc"),
        default="quad",
        help="Pricing engine for daily MTM/event stats (default: quad).",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--realized-vol-window", type=int, default=20)
    parser.add_argument("--quad-grid", type=int, default=301)
    parser.add_argument("--quad-std-devs", type=float, default=6.0)
    parser.add_argument("--pde-grid", type=int, default=160)
    parser.add_argument("--mc-paths", type=int, default=2000)
    parser.add_argument("--mc-steps", type=int, default=252)
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip static dashboard.html generation.",
    )
    return parser.parse_args(argv)


def load_akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise DemoDataError(
            "AKShare is not installed. Install akshare, or rerun with an existing "
            "cache and --cache-only."
        ) from exc
    return ak


def third_friday(year: int, month: int) -> pd.Timestamp:
    fridays = [
        week[FRIDAY]
        for week in monthcalendar(year, month)
        if week[FRIDAY] != 0
    ]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def contract_expiry(contract: str) -> pd.Timestamp:
    suffix = contract.replace(CSI500_FUTURES_PREFIX, "")
    if len(suffix) != 4 or not suffix.isdigit():
        raise DemoDataError(f"Cannot infer expiry from futures contract {contract!r}")
    year = 2000 + int(suffix[:2])
    month = int(suffix[2:])
    if not 1 <= month <= 12:
        raise DemoDataError(f"Invalid futures contract month in {contract!r}")
    return third_friday(year, month)


def add_months(date: pd.Timestamp, months: int) -> pd.Timestamp:
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def ic_contract_symbols(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    extra_months: int = 2,
) -> list[str]:
    start = pd.Timestamp(start_date).normalize().replace(day=1)
    end = add_months(pd.Timestamp(end_date).normalize().replace(day=1), extra_months)
    symbols = []
    cursor = start
    while cursor <= end:
        symbols.append(f"{CSI500_FUTURES_PREFIX}{cursor:%y%m}")
        cursor = add_months(cursor, 1)
    return symbols


def _pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise DemoDataError(f"Missing {label} column. Available columns: {list(df.columns)}")


def normalize_csi500_spot(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_column(raw, ["date", "日期"], "spot date")
    close_col = _pick_column(raw, ["close", "收盘", "收盘价"], "spot close")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]).dt.normalize(),
            "spot": pd.to_numeric(raw[close_col], errors="coerce"),
        }
    )
    out = out.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise DemoDataError("CSI500 spot data is empty after normalization")
    return out.reset_index(drop=True)


def normalize_ic_futures(raw: pd.DataFrame, contract: str) -> pd.DataFrame:
    date_col = _pick_column(raw, ["date", "日期"], "futures date")
    close_col = _pick_column(raw, ["close", "收盘", "收盘价"], "futures close")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]).dt.normalize(),
            "contract": contract,
            "futures_price": pd.to_numeric(raw[close_col], errors="coerce"),
            "expiry_date": contract_expiry(contract),
            "multiplier": IC_MULTIPLIER,
        }
    )
    out = out.dropna().sort_values("date").drop_duplicates(
        ["date", "contract"], keep="last"
    )
    return out[out["futures_price"] > 0].reset_index(drop=True)


def fetch_csi500_spot(ak, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    raw = ak.index_zh_a_hist(
        symbol=CSI500_INDEX_SYMBOL,
        period="daily",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    spot = normalize_csi500_spot(raw)
    mask = (spot["date"] >= start_date) & (spot["date"] <= end_date)
    return spot.loc[mask].reset_index(drop=True)


def fetch_ic_futures(
    ak,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for contract in ic_contract_symbols(start_date, end_date):
        try:
            raw = ak.futures_zh_daily_sina(symbol=contract)
            normalized = normalize_ic_futures(raw, contract)
        except Exception as exc:
            print(f"[warn] Skipping {contract}: {exc}")
            continue
        mask = (normalized["date"] >= start_date) & (normalized["date"] <= end_date)
        if mask.any():
            frames.append(normalized.loc[mask])
    if not frames:
        raise DemoDataError("No IC futures data was fetched for the requested window")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "expiry_date", "contract"]
    )


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "spot": cache_dir / "csi500_spot.csv",
        "futures": cache_dir / "ic_futures.csv",
    }


def read_cache(cache_dir: Path) -> CachedMarketFrames:
    paths = cache_paths(cache_dir)
    if not paths["spot"].exists() or not paths["futures"].exists():
        raise DemoDataError(f"Cache is missing under {cache_dir}")
    spot = pd.read_csv(paths["spot"], parse_dates=["date"])
    futures = pd.read_csv(paths["futures"], parse_dates=["date", "expiry_date"])
    return CachedMarketFrames(spot_data=spot, futures_data=futures)


def write_cache(cache_dir: Path, frames: CachedMarketFrames) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    frames.spot_data.to_csv(paths["spot"], index=False)
    frames.futures_data.to_csv(paths["futures"], index=False)


def load_market_cache_or_fetch(
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
    refresh_data: bool,
    cache_only: bool,
) -> CachedMarketFrames:
    if not refresh_data:
        try:
            return read_cache(cache_dir)
        except DemoDataError:
            if cache_only:
                raise

    if cache_only:
        raise DemoDataError(f"Cache is missing under {cache_dir}")

    try:
        ak = load_akshare()
        frames = CachedMarketFrames(
            spot_data=fetch_csi500_spot(ak, start_date, end_date),
            futures_data=fetch_ic_futures(ak, start_date, end_date),
        )
        write_cache(cache_dir, frames)
        return frames
    except Exception as exc:
        try:
            cached = read_cache(cache_dir)
            print(f"[warn] Live AKShare fetch failed; using cache. Reason: {exc}")
            return cached
        except DemoDataError:
            raise DemoDataError(
                "Could not load AKShare data and no usable cache exists. "
                f"Original error: {exc}"
            ) from exc


def restrict_window(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    mask = (out["date"] >= start_date) & (out["date"] <= end_date)
    return out.loc[mask].reset_index(drop=True)


def realized_vol_data(spot_data: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        raise DemoDataError("realized-vol-window must be greater than 1")
    spot = spot_data.sort_values("date").copy()
    returns = np.log(spot["spot"].astype(float) / spot["spot"].astype(float).shift(1))
    vol = returns.rolling(window=window, min_periods=2).std() * np.sqrt(252.0)
    vol = vol.bfill().ffill().fillna(0.20).clip(lower=0.01)
    return pd.DataFrame({"date": spot["date"], "volatility": vol})


def rate_data(dates: pd.Series, rate: float) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize(), "rate": rate})


def build_market_dataset(
    frames: CachedMarketFrames,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    realized_vol_window: int,
    rate: float,
) -> AutocallableMarketDataSet:
    spot = restrict_window(frames.spot_data, start_date, end_date)
    futures = restrict_window(frames.futures_data, start_date, end_date)
    if spot.empty:
        raise DemoDataError("No CSI500 spot data in requested window")
    if futures.empty:
        raise DemoDataError("No IC futures data in requested window")
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=spot,
        vol_data=realized_vol_data(spot, realized_vol_window),
        rate_data=rate_data(spot["date"], rate),
        futures_data=futures,
        metadata={
            "spot_symbol": CSI500_INDEX_SYMBOL,
            "futures_prefix": CSI500_FUTURES_PREFIX,
            "vol_source": f"{realized_vol_window}d_realized",
            "rate_source": "flat",
        },
    )


def create_demo_product(initial_spot: float, issue_date: pd.Timestamp):
    product = create_standard_snowball(
        initial_price=float(initial_spot),
        strike=float(initial_spot),
        maturity=1.0,
        contract_multiplier=100.0,
        ko_barrier=float(initial_spot) * 1.03,
        ko_rate=0.15,
        ki_barrier=float(initial_spot) * 0.75,
        num_observations=12,
        include_principal=True,
    )
    product.initial_date = pd.Timestamp(issue_date).to_pydatetime()
    return product


def create_engine_config(args: argparse.Namespace) -> AutocallableEngineConfig:
    quad_params = QuadParams(
        grid_points=args.quad_grid,
        num_std_devs=args.quad_std_devs,
    )
    if args.engine == "quad":
        return AutocallableEngineConfig(
            pricing_engine_type=EngineType.QUADRATURE,
            quad_params=quad_params,
        )
    if args.engine == "pde":
        return AutocallableEngineConfig(
            pricing_engine_type=EngineType.PDE,
            method=EngineType.PDE(PDEMethod.CRANK_NICOLSON),
            pde_params=PDEParams(
                grid=GridConfig(
                    points=args.pde_grid,
                    max_points=max(2000, args.pde_grid),
                ),
            ),
            quad_params=quad_params,
            surface_engine_type=EngineType.QUADRATURE,
        )
    return AutocallableEngineConfig(
        pricing_engine_type=EngineType.MONTE_CARLO,
        mc_params=MCParams(
            num_paths=args.mc_paths,
            time_steps=args.mc_steps,
            seed=args.mc_seed,
        ),
        quad_params=quad_params,
        surface_engine_type=EngineType.QUADRATURE,
    )


def build_backtest_config(
    args: argparse.Namespace,
    market_data: AutocallableMarketDataSet,
) -> AutocallableBacktestConfig:
    first_date = market_data.dates[0]
    initial_spot = float(market_data.spot_data.loc[first_date, "spot"])
    return AutocallableBacktestConfig(
        product=create_demo_product(initial_spot, first_date),
        market_data=market_data,
        engine_config=create_engine_config(args),
        strategy=AutocallableDeltaHedgeStrategy(
            delta_threshold=0.25,
            hedge_ratio=1.0,
            round_contracts=True,
        ),
        product_quantity=-1.0,
        underlying="CSI500",
        start_date=pd.Timestamp(args.start_date).to_pydatetime(),
        end_date=pd.Timestamp(args.end_date).to_pydatetime(),
        surface_config=SurfaceGridConfig(
            spot_nodes=5,
            spot_width=0.05,
            q_nodes=3,
            q_width=0.005,
        ),
        calculate_surfaces=True,
        calculate_event_probabilities=True,
        metadata={"demo": "otc_autocallable_backtest", "engine": args.engine},
    )


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)


def _excel_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "name") and not isinstance(value, str):
        return value.name
    if isinstance(value, np.ndarray):
        return json.dumps([_excel_value(item) for item in value.tolist()], default=str)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return value


def _first_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _key_value_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["term", "value"])


def _append_param_rows(
    rows: list[dict[str, Any]], prefix: str, params: Any
) -> None:
    if params is None:
        return
    values = getattr(params, "__dict__", {})
    for key, value in sorted(values.items()):
        if key.startswith("_"):
            continue
        rows.append({"term": f"{prefix}.{key}", "value": _excel_value(value)})


def _otc_contract_terms_frame(results) -> pd.DataFrame:
    config = results.config
    product = config.product
    barrier_config = getattr(product, "barrier_config", None)
    payoff_config = getattr(product, "payoff_config", None)
    accrual_config = getattr(product, "accrual_config", None)
    issue_date = getattr(product, "initial_date", None) or config.start_date
    initial_price = _first_numeric(getattr(product, "initial_price", None))
    contract_multiplier = _first_numeric(getattr(product, "contract_multiplier", None))
    note_notional = None
    if initial_price is not None and contract_multiplier is not None:
        note_notional = initial_price * contract_multiplier
    maturity_years = getattr(product, "maturity", None)
    maturity_date = None
    if issue_date is not None and maturity_years is not None:
        maturity_date = pd.Timestamp(issue_date) + pd.Timedelta(
            days=int(round(float(maturity_years) * 365))
        )

    rows: list[dict[str, Any]] = []

    def add(term: str, value: Any) -> None:
        rows.append({"term": term, "value": _excel_value(value)})

    add("product_type", type(product).__name__)
    add("underlying_reference", config.underlying)
    add("issue_date", issue_date)
    add("maturity_years", maturity_years)
    add("maturity_date", maturity_date)
    add("initial_price", initial_price)
    add("strike", getattr(product, "strike", None))
    add("contract_multiplier", contract_multiplier)
    add("note_notional_cny", note_notional)
    add("is_reverse", getattr(product, "is_reverse", None))
    if barrier_config is not None:
        ko_barrier = getattr(barrier_config, "ko_barrier", None)
        ki_barrier = getattr(barrier_config, "ki_barrier", None)
        add("ko_barrier", ko_barrier)
        add(
            "ko_barrier_pct_initial",
            None
            if initial_price in (None, 0)
            else _first_numeric(ko_barrier) / initial_price,
        )
        add("ko_rate", getattr(barrier_config, "ko_rate", None))
        add("ko_observation_type", getattr(barrier_config, "ko_observation_type", None))
        add("ki_barrier", ki_barrier)
        add(
            "ki_barrier_pct_initial",
            None
            if initial_price in (None, 0)
            else _first_numeric(ki_barrier) / initial_price,
        )
        add("ki_observation_type", getattr(barrier_config, "ki_observation_type", None))
        add("ki_continuous", getattr(barrier_config, "ki_continuous", None))
        add("disable_ko_after_ki", getattr(barrier_config, "disable_ko_after_ki", None))
    add("num_ko_observations", getattr(product, "num_ko_observations", None))
    add("num_ki_observations", getattr(product, "num_ki_observations", None))
    if payoff_config is not None:
        add("include_principal", getattr(payoff_config, "include_principal", None))
        add("rebate_rate", getattr(payoff_config, "rebate_rate", None))
        add("participation_rate", getattr(payoff_config, "participation_rate", None))
        add("protection_type", getattr(payoff_config, "protection_type", None))
        add("protection_rate", getattr(payoff_config, "protection_rate", None))
    if accrual_config is not None:
        add("coupon_pay_type", getattr(accrual_config, "coupon_pay_type", None))
        add("is_annualized", getattr(accrual_config, "is_annualized", None))
    return _key_value_frame(rows)


def _trade_position_frame(results) -> pd.DataFrame:
    config = results.config
    product = config.product
    initial_price = _first_numeric(getattr(product, "initial_price", None))
    contract_multiplier = _first_numeric(getattr(product, "contract_multiplier", None))
    quantity = float(getattr(config, "product_quantity", 0.0))
    trade_notional = None
    if initial_price is not None and contract_multiplier is not None:
        trade_notional = abs(quantity) * initial_price * contract_multiplier
    rows = [
        {"term": "desk_perspective", "value": "short_note" if quantity < 0 else "long_note"},
        {"term": "product_quantity", "value": quantity},
        {"term": "trade_notional_cny", "value": trade_notional},
        {"term": "initial_product_price_override", "value": config.initial_product_price},
        {"term": "backtest_start_date", "value": _excel_value(config.start_date)},
        {"term": "backtest_end_date", "value": _excel_value(config.end_date)},
    ]
    return _key_value_frame(rows)


def _pricing_model_config_frame(results, args: argparse.Namespace) -> pd.DataFrame:
    engine_config = results.config.engine_config
    rows = [
        {"term": "selected_cli_engine", "value": args.engine},
        {
            "term": "pricing_engine_type",
            "value": _excel_value(engine_config.pricing_engine_type),
        },
        {"term": "method", "value": _excel_value(engine_config.method)},
        {
            "term": "surface_engine_type",
            "value": _excel_value(engine_config.resolve_surface_engine_type()),
        },
        {
            "term": "event_stats_engine_type",
            "value": _excel_value(engine_config.resolve_event_stats_engine_type()),
        },
        {
            "term": "calculate_surfaces",
            "value": bool(results.config.calculate_surfaces),
        },
        {
            "term": "calculate_event_probabilities",
            "value": bool(results.config.calculate_event_probabilities),
        },
        {
            "term": "surface.spot_nodes",
            "value": results.config.surface_config.spot_nodes,
        },
        {
            "term": "surface.spot_width",
            "value": results.config.surface_config.spot_width,
        },
        {"term": "surface.q_nodes", "value": results.config.surface_config.q_nodes},
        {"term": "surface.q_width", "value": results.config.surface_config.q_width},
    ]
    _append_param_rows(rows, "pde_params", engine_config.pde_params)
    _append_param_rows(rows, "mc_params", engine_config.mc_params)
    _append_param_rows(rows, "quad_params", engine_config.quad_params)
    return _key_value_frame(rows)


def _market_data_assumptions_frame(results, args: argparse.Namespace) -> pd.DataFrame:
    market_data = results.config.market_data
    metadata = market_data.metadata or {}
    dates = market_data.dates
    rows = [
        {"term": "spot_symbol", "value": metadata.get("spot_symbol")},
        {"term": "spot_source", "value": "AKShare index_zh_a_hist"},
        {"term": "futures_prefix", "value": metadata.get("futures_prefix")},
        {"term": "futures_source", "value": "AKShare futures_zh_daily_sina"},
        {"term": "volatility_source", "value": metadata.get("vol_source")},
        {"term": "realized_vol_window", "value": args.realized_vol_window},
        {"term": "rate_source", "value": metadata.get("rate_source")},
        {"term": "flat_rate", "value": args.rate},
        {"term": "market_data_start_date", "value": _excel_value(dates.min() if len(dates) else None)},
        {"term": "market_data_end_date", "value": _excel_value(dates.max() if len(dates) else None)},
        {"term": "basis_yield_formula", "value": "(F - S) / S / T_fut"},
        {"term": "implied_q_formula", "value": "max(0, r - basis_yield)"},
        {"term": "basis_time_basis", "value": "calendar-day year fraction"},
        {"term": "continuous_ki_proxy", "value": "daily close"},
    ]
    return _key_value_frame(rows)


def _hedge_futures_config_frame(results) -> pd.DataFrame:
    config = results.config
    strategy = config.strategy
    strategy_params = (
        strategy.get_parameters()
        if strategy is not None and hasattr(strategy, "get_parameters")
        else {}
    )
    rows = [
        {"term": "hedging_instrument", "value": "CFFEX IC equity-index futures"},
        {"term": "underlying_index", "value": "CSI500"},
        {"term": "futures_prefix", "value": CSI500_FUTURES_PREFIX},
        {"term": "futures_multiplier", "value": IC_MULTIPLIER},
        {
            "term": "futures_roll_days_before_expiry",
            "value": getattr(config.roll_policy, "roll_days_before_expiry", None),
        },
        {
            "term": "futures_roll_rule",
            "value": (
                "roll active futures hedge when current contract is within the "
                "configured calendar-day threshold before expiry"
            ),
        },
        {
            "term": "transaction_cost_model",
            "value": type(config.transaction_cost_model).__name__,
        },
    ]
    for key, value in sorted(strategy_params.items()):
        rows.append({"term": f"delta_hedge.{key}", "value": _excel_value(value)})
    return _key_value_frame(rows)


def _term_rows(results, args: argparse.Namespace) -> pd.DataFrame:
    """Backward-compatible alias for callers that imported the old helper."""
    return _otc_contract_terms_frame(results)
    add("basis_yield_formula", "(F - S) / S / T_fut")
    add("implied_q_formula", "max(0, r - basis_yield)")
    add("volatility_source", config.market_data.metadata.get("vol_source"))
    add("rate_source", config.market_data.metadata.get("rate_source"))
    return pd.DataFrame(rows)


def _value_at_index(value: Any, index: int) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return value[min(index, len(value) - 1)]
    return value


def _observation_schedule_frame(results) -> pd.DataFrame:
    product = results.config.product
    barrier_config = getattr(product, "barrier_config", None)
    initial_price = _first_numeric(getattr(product, "initial_price", None))
    issue_date = pd.Timestamp(
        getattr(product, "initial_date", None) or results.config.start_date
    )
    if barrier_config is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    def add_timed_rows(
        event_type: str,
        times: list[float],
        barrier: Any,
        return_rate: Any = None,
    ) -> None:
        for idx, obs_time in enumerate(times):
            barrier_value = _first_numeric(_value_at_index(barrier, idx))
            rows.append(
                {
                    "event_type": event_type,
                    "observation_index": idx,
                    "observation_time_years": float(obs_time),
                    "scheduled_date": (
                        issue_date
                        + pd.Timedelta(days=int(round(float(obs_time) * 365)))
                    ).normalize(),
                    "barrier": barrier_value,
                    "barrier_pct_initial": None
                    if initial_price in (None, 0) or barrier_value is None
                    else barrier_value / initial_price,
                    "return_rate": _excel_value(_value_at_index(return_rate, idx)),
                    "monitoring": "scheduled",
                }
            )

    ko_schedule = getattr(barrier_config, "ko_observation_schedule", None)
    if ko_schedule is not None:
        ko_times = [
            float(rec.observation_time)
            for rec in ko_schedule.records
            if rec.observation_time is not None
        ]
    else:
        ko_times = [float(t) for t in (getattr(barrier_config, "ko_observation_dates", None) or [])]
    add_timed_rows(
        "KO",
        ko_times,
        getattr(barrier_config, "ko_barrier", None),
        getattr(barrier_config, "ko_rate", None),
    )

    ki_barrier = getattr(barrier_config, "ki_barrier", None)
    if ki_barrier is not None:
        if getattr(barrier_config, "ki_continuous", False):
            barrier_value = _first_numeric(ki_barrier)
            rows.append(
                {
                    "event_type": "KI",
                    "observation_index": None,
                    "observation_time_years": None,
                    "scheduled_date": None,
                    "barrier": barrier_value,
                    "barrier_pct_initial": None
                    if initial_price in (None, 0) or barrier_value is None
                    else barrier_value / initial_price,
                    "return_rate": None,
                    "monitoring": "daily_close_continuous_approximation",
                }
            )
        else:
            ki_schedule = getattr(barrier_config, "ki_observation_schedule", None)
            if ki_schedule is not None:
                ki_times = [
                    float(rec.observation_time)
                    for rec in ki_schedule.records
                    if rec.observation_time is not None
                ]
            else:
                ki_times = [
                    float(t)
                    for t in (getattr(barrier_config, "ki_observation_dates", None) or [])
                ]
            add_timed_rows("KI", ki_times, ki_barrier)

    return pd.DataFrame(rows)


def _listed_futures_info_frame(results) -> pd.DataFrame:
    futures = results.config.market_data.futures_data.copy()
    if futures.empty:
        return pd.DataFrame()
    active_contracts = set()
    states = results.states_df
    if not states.empty and "active_contract" in states.columns:
        active_contracts = set(states["active_contract"].astype(str))

    rows: list[dict[str, Any]] = []
    for contract, group in futures.sort_values("date").groupby("contract", sort=True):
        contract_text = str(contract)
        suffix = contract_text.replace(CSI500_FUTURES_PREFIX, "")
        contract_month = None
        if len(suffix) == 4 and suffix.isdigit():
            contract_month = pd.Timestamp(
                year=2000 + int(suffix[:2]), month=int(suffix[2:]), day=1
            )
        rows.append(
            {
                "contract": contract_text,
                "underlying_index": "CSI500",
                "exchange": "CFFEX",
                "data_source": "AKShare futures_zh_daily_sina",
                "contract_month": contract_month,
                "expiry_date": pd.Timestamp(group["expiry_date"].iloc[0]),
                "multiplier": float(group["multiplier"].iloc[0]),
                "first_price_date": pd.Timestamp(group["date"].min()),
                "last_price_date": pd.Timestamp(group["date"].max()),
                "first_futures_price": float(group["futures_price"].iloc[0]),
                "last_futures_price": float(group["futures_price"].iloc[-1]),
                "min_futures_price": float(group["futures_price"].min()),
                "max_futures_price": float(group["futures_price"].max()),
                "num_price_observations": int(len(group)),
                "used_as_active_hedge": contract_text in active_contracts,
            }
        )
    return pd.DataFrame(rows)


def _active_futures_frame(results) -> pd.DataFrame:
    states = results.states_df
    if states.empty:
        return pd.DataFrame()
    columns = [
        "active_contract",
        "futures_price",
        "futures_ttm",
        "basis_yield",
        "implied_q",
        "futures_contracts",
    ]
    available = [column for column in columns if column in states.columns]
    return states[available].reset_index()


def append_workbook_metadata(results, workbook_path: Path, args: argparse.Namespace) -> None:
    with pd.ExcelWriter(
        workbook_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        _otc_contract_terms_frame(results).to_excel(
            writer, sheet_name="OTC_Contract_Terms", index=False
        )
        observations = _observation_schedule_frame(results)
        if not observations.empty:
            observations.to_excel(
                writer, sheet_name="OTC_Observation_Schedule", index=False
            )
        _trade_position_frame(results).to_excel(
            writer, sheet_name="Trade_Position", index=False
        )
        _pricing_model_config_frame(results, args).to_excel(
            writer, sheet_name="Pricing_Model_Config", index=False
        )
        _market_data_assumptions_frame(results, args).to_excel(
            writer, sheet_name="Market_Data_Assumptions", index=False
        )
        _hedge_futures_config_frame(results).to_excel(
            writer, sheet_name="Hedge_Futures_Config", index=False
        )
        _listed_futures_info_frame(results).to_excel(
            writer, sheet_name="Listed_Futures", index=False
        )
        active_futures = _active_futures_frame(results)
        if not active_futures.empty:
            active_futures.to_excel(
                writer, sheet_name="Active_Futures", index=False
            )


def write_outputs(results, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(results.states_df, output_dir / "states.csv")
    _write_frame(results.greeks_df, output_dir / "greeks.csv")
    _write_frame(results.rebalance_df, output_dir / "rebalance.csv")
    _write_frame(results.trades_df, output_dir / "trades.csv")
    _write_frame(results.actions_df, output_dir / "actions.csv")
    _write_frame(results.daily_event_summary_df, output_dir / "daily_event_summary.csv")
    _write_frame(results.event_probability_df, output_dir / "event_probability.csv")
    results.export_surfaces_to_parquet(str(output_dir / "surfaces.parquet"))
    workbook_path = output_dir / "results.xlsx"
    results.export_to_excel(str(workbook_path))
    append_workbook_metadata(results, workbook_path, args)
    dashboard_path = None
    if not bool(getattr(args, "no_dashboard", False)):
        dashboard_path = AutocallableBacktestDashboard(
            results,
            AutocallableDashboardConfig(),
        ).write_html(output_dir / "dashboard.html")

    states = results.states_df
    actions = results.actions_df
    summary = results.get_summary()
    summary.update(
        {
            "engine": args.engine,
            "output_dir": str(output_dir),
            "trade_count": int(len(results.trades_df)),
            "action_count": int(len(actions)),
            "lifecycle_actions": []
            if actions.empty
            else sorted(actions["action_type"].astype(str).unique().tolist()),
            "final_hedge_contracts": 0.0
            if states.empty
            else float(states["futures_contracts"].iloc[-1]),
            "first_basis_yield": None
            if states.empty
            else float(states["basis_yield"].iloc[0]),
            "last_basis_yield": None
            if states.empty
            else float(states["basis_yield"].iloc[-1]),
            "first_implied_q": None
            if states.empty
            else float(states["implied_q"].iloc[0]),
            "last_implied_q": None
            if states.empty
            else float(states["implied_q"].iloc[-1]),
            "dashboard_html": None if dashboard_path is None else str(dashboard_path),
        }
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\nOTC Autocallable Backtest Demo")
    print("=" * 72)
    print(f"Date range:      {summary.get('start_date')} -> {summary.get('end_date')}")
    print(f"Engine:          {summary.get('engine')}")
    print(f"Final PnL:       {summary.get('total_pnl', 0.0):,.2f}")
    print(f"Trades:          {summary.get('trade_count', 0)}")
    print(f"Actions:         {', '.join(summary.get('lifecycle_actions') or ['none'])}")
    print(f"Final hedge:     {summary.get('final_hedge_contracts', 0.0):,.0f} contracts")
    print(
        "Basis yield:     "
        f"{summary.get('first_basis_yield', 0.0):.4%} -> "
        f"{summary.get('last_basis_yield', 0.0):.4%}"
    )
    print(
        "Implied q:       "
        f"{summary.get('first_implied_q', 0.0):.4%} -> "
        f"{summary.get('last_implied_q', 0.0):.4%}"
    )
    print(f"Outputs:         {summary.get('output_dir')}")


def run_demo(args: argparse.Namespace):
    start_date = pd.Timestamp(args.start_date).normalize()
    end_date = pd.Timestamp(args.end_date).normalize()
    if end_date < start_date:
        raise DemoDataError("end-date must be on or after start-date")
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cached = load_market_cache_or_fetch(
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        refresh_data=bool(args.refresh_data),
        cache_only=bool(args.cache_only),
    )
    market_data = build_market_dataset(
        cached,
        start_date=start_date,
        end_date=end_date,
        realized_vol_window=int(args.realized_vol_window),
        rate=float(args.rate),
    )
    config = build_backtest_config(args, market_data)
    results = AutocallableBacktestEngine(config).run()
    summary = write_outputs(results, output_dir, args)
    print_summary(summary)
    return results, summary


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run_demo(args)
    except DemoDataError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

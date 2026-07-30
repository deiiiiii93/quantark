"""Deterministic fixture configs for the replay-consolidation goldens.

Three configurations freeze pre-consolidation behavior (spec §9.1):

- ``make_scalar_bsm_config``   — single snowball, scalar vol, PDE defaults
- ``make_book_config``         — snowball + phoenix book, futures hedge
- ``make_localvol_config``     — single snowball, per-day localvol calibration
                                 (the only golden that exercises
                                 ``VolModelCalibrator`` / ``create_vol_model_engine``)

Imports go through ``quantark.backtest.otc`` deliberately: after the
consolidation these same fixtures exercise the deprecation shims, so the
goldens double as behavioral compat coverage.

The localvol surface history is written into the golden data directory (not a
tmp dir) so artifact SHAs — and therefore calibration cache keys and recorded
provenance — are bit-stable between capture time and every later test run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quantark.asset.equity.product.option import (
    create_standard_phoenix,
    create_standard_snowball,
)
from quantark.backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    BookAutocallableBacktestConfig,
    BookProduct,
    HedgeSpec,
    VolModelCalibrationConfig,
)
from quantark.backtest.otc.vol_history import VolSurfaceHistory
from quantark.backtest.transaction_costs import ZeroCostModel
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType

GOLDEN_DIR = Path(__file__).parent / "data"

RATE = 0.02
STRIKES = [90.0, 100.0, 110.0]
MATURITIES = [0.25, 0.5, 1.0]
ATM_VOLS_A = [0.18, 0.19, 0.21]
ATM_VOLS_B = [0.22, 0.23, 0.25]
Q_PILLARS = [0.01, 0.015, 0.02]
DATE_A = date(2024, 1, 2)
DATE_EXCLUDED = date(2024, 1, 3)
DATE_B = date(2024, 1, 4)

# Timing fields are wall-clock and excluded from all comparisons (spec §9.2).
TIMING_FIELDS = ("pricing_seconds", "calibration_seconds")


def _snowball_product():
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=6.0 / 365.0,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ki_barrier=97.0,
        ko_rate=0.02,
        num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[1.0 / 365.0, 3.0 / 365.0],
        include_principal=True,
    )


def _phoenix_product():
    return create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=0.25,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.02,
        coupon_rate=0.01,
        num_observations=2,
        ko_observation_dates=[0.125, 0.25],
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[0.125, 0.25],
        memory_coupon=True,
        include_principal=True,
    )


def _market_frames(num_days: int = 5):
    dates = pd.date_range("2024-01-02", periods=num_days, freq="D")
    spots = [100.0, 96.0, 104.0, 105.0, 106.0][:num_days]
    spot_data = pd.DataFrame({"date": dates, "spot": spots})
    vol_data = pd.DataFrame({"date": dates, "volatility": [0.22] * num_days})
    rate_data = pd.DataFrame({"date": dates, "rate": [RATE] * num_days})
    futures_rows = []
    for d, spot in zip(dates, spots):
        futures_rows.extend(
            [
                {
                    "date": d,
                    "contract": "IF2401",
                    "futures_price": spot * 1.004,
                    "expiry_date": pd.Timestamp("2024-01-07"),
                    "multiplier": 300.0,
                },
                {
                    "date": d,
                    "contract": "IF2402",
                    "futures_price": spot * 1.01,
                    "expiry_date": pd.Timestamp("2024-02-16"),
                    "multiplier": 300.0,
                },
            ]
        )
    return spot_data, vol_data, rate_data, pd.DataFrame(futures_rows)


def _market_data(num_days: int = 5, surface_history=None) -> AutocallableMarketDataSet:
    spot_data, vol_data, rate_data, futures_data = _market_frames(num_days)
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=spot_data,
        vol_data=vol_data,
        rate_data=rate_data,
        futures_data=futures_data,
        surface_history=surface_history,
    )


def _artifact_payload(trade_date: date, *, s0: float, atm_vols) -> dict:
    iv_grid = [[atm + 0.05 * (k / s0 - 1.0) for k in STRIKES] for atm in atm_vols]
    atm_pillars = [
        {
            "T": t,
            "expiry_date": (trade_date + timedelta(days=round(t * 365))).isoformat(),
            "atm_vol": atm,
        }
        for t, atm in zip(MATURITIES, atm_vols)
    ]
    per_expiry = [
        {
            "T": t,
            "expiry_date": (trade_date + timedelta(days=round(t * 365))).isoformat(),
            "r": RATE,
            "q": q,
            "forward": s0 * math.exp((RATE - q) * t),
            "df": math.exp(-RATE * t),
        }
        for t, q in zip(MATURITIES, Q_PILLARS)
    ]
    return {
        "trade_date": trade_date.isoformat(),
        "s0": s0,
        "strikes": list(STRIKES),
        "maturities": list(MATURITIES),
        "iv_grid": iv_grid,
        "atm_pillars": atm_pillars,
        "per_expiry": per_expiry,
        "extrapolation_policy": {
            "beyond_last_listed_expiry": "flat_total_variance",
            "max_listed_T": max(MATURITIES),
        },
        "admission": {"status": "ok"},
    }


def _write_if_changed(path: Path, raw: bytes) -> None:
    """Byte-stable, xdist-race-safe write: skip identical content, replace
    atomically otherwise (parallel golden workers share this directory)."""
    if path.is_file() and path.read_bytes() == raw:
        return
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def write_localvol_history(root: Path) -> Path:
    """Write (idempotently, byte-stable) the synthetic surface history."""
    history_dir = root / "history_localvol"
    surface_dir = history_dir / "iv_surface"
    surface_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for d, payload in (
        (DATE_A, _artifact_payload(DATE_A, s0=100.0, atm_vols=ATM_VOLS_A)),
        (DATE_B, _artifact_payload(DATE_B, s0=104.0, atm_vols=ATM_VOLS_B)),
    ):
        raw = json.dumps(payload).encode()
        _write_if_changed(surface_dir / f"mo_iv_surface_{d:%Y%m%d}.json", raw)
        records.append(
            {
                "date": f"{d:%Y%m%d}",
                "status": "ok",
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "reason": None,
                "detail": None,
            }
        )
    records.append(
        {
            "date": f"{DATE_EXCLUDED:%Y%m%d}",
            "status": "excluded",
            "artifact_sha256": None,
            "reason": "static_arbitrage",
            "detail": "synthetic exclusion",
        }
    )
    records.sort(key=lambda rec: rec["date"])
    manifest = {
        "schema_version": 1,
        "gap_policy": "consumers carry forward previous admitted surface",
        "records": records,
    }
    _write_if_changed(
        history_dir / "surface_manifest.json", json.dumps(manifest).encode()
    )
    return history_dir


def make_scalar_bsm_config() -> AutocallableBacktestConfig:
    return AutocallableBacktestConfig(
        product=_snowball_product(),
        market_data=_market_data(),
        engine_config=AutocallableEngineConfig(
            pricing_engine_type=EngineType.PDE,
            # Legacy behavior: the generic PDE engine returns no event stats
            # for this product and the old code silently fell back to MC.
            # The fallback is opt-in now; the goldens freeze that behavior
            # and the provenance column discloses it.
            event_stats_fallback="mc",
        ),
        transaction_cost_model=ZeroCostModel(),
        product_quantity=-1.0,
        calculate_surfaces=True,
        calculate_event_probabilities=True,
        # Goldens freeze pre-consolidation semantics: no early termination.
        terminate_on_lifecycle_end=False,
    )


def make_book_config() -> BookAutocallableBacktestConfig:
    return BookAutocallableBacktestConfig(
        # Two snowballs, not snowball+phoenix: the standard phoenix helper
        # builds no date-based KI schedule, which every pricing route requires
        # (mirrors test_book_backtest, which books snowballs only).
        products=[
            BookProduct(product=_snowball_product(), quantity=-1.0, position_id=1,
                        has_lifecycle=True),
            BookProduct(product=_snowball_product(), quantity=-2.0, position_id=2,
                        has_lifecycle=True),
        ],
        market_data=_market_data(),
        hedge=HedgeSpec(kind="futures", multiplier=300.0),
        # QUADRATURE mirrors test_book_backtest: the PDE route needs a
        # date-based KI schedule the standard phoenix helper doesn't build.
        engine_config=AutocallableEngineConfig(
            pricing_engine_type=EngineType.QUADRATURE,
            event_stats_fallback="mc",
        ),
        transaction_cost_model=ZeroCostModel(),
        calculate_surfaces=False,
        calculate_event_probabilities=True,
        terminate_on_lifecycle_end=False,
    )


def make_localvol_config(history_root: Path = GOLDEN_DIR) -> AutocallableBacktestConfig:
    history_dir = write_localvol_history(history_root)
    engine_config = AutocallableEngineConfig(
        pricing_engine_type=EngineType.PDE,
        event_stats_fallback="mc",
        vol_source="surface",
        vol_model="localvol",
        vol_model_solver="pde",
        vol_model_calibration=VolModelCalibrationConfig(cache_dir=None),
    )
    return AutocallableBacktestConfig(
        product=_snowball_product(),
        market_data=_market_data(
            num_days=3, surface_history=VolSurfaceHistory(history_dir)
        ),
        engine_config=engine_config,
        transaction_cost_model=ZeroCostModel(),
        product_quantity=-1.0,
        calculate_surfaces=False,
        calculate_event_probabilities=True,
        terminate_on_lifecycle_end=False,
    )


FRAME_NAMES = {
    "states": ("states_df",),
    "greeks": ("greeks_df",),
    "rebalances": ("rebalance_df", "rebalances_df"),
    "trades": ("trades_df",),
    "actions": ("actions_df",),
    "surfaces": ("surfaces_df",),
    "daily_event_summary": ("daily_event_summary_df",),
    "event_probabilities": ("event_probability_df",),
}


def result_frames(results) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, candidates in FRAME_NAMES.items():
        for attr in candidates:
            if hasattr(results, attr):
                value = getattr(results, attr)
                df = value() if callable(value) else value
                break
        else:
            raise AttributeError(
                f"no accessor among {candidates} on {type(results).__name__}"
            )
        df = df.reset_index()
        # A RangeIndex leaves a literal "index" artifact column; only real
        # index columns (date) are part of the contract.
        if "index" in df.columns:
            df = df.drop(columns=["index"])
        df = df.drop(columns=[c for c in TIMING_FIELDS if c in df.columns])
        frames[name] = df
    return frames


def result_summary(results) -> dict:
    summary = dict(results.get_summary())
    for key in TIMING_FIELDS:
        summary.pop(key, None)
    return summary


def calibration_records(results) -> list[dict]:
    records = [dict(r) for r in getattr(results, "calibration_records", [])]
    for record in records:
        for key in TIMING_FIELDS:
            record.pop(key, None)
    return records

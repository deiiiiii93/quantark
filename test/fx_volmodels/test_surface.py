"""Tests for the observed/prepared split in the CFETS FX surface stage."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "fx_volmodels"
sys.path.insert(0, str(EXAMPLE))

import _fx_common as fx  # noqa: E402


def _load_surface_module():
    spec = importlib.util.spec_from_file_location(
        "fx_build_surface_contract", EXAMPLE / "02_build_fx_surface.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surface_stage = _load_surface_module()


def _quotes(iv: float) -> list[dict]:
    strikes = (6.80, 6.90, 7.00, 7.10, 7.20)
    return [
        {
            "pillar": pillar,
            "delta": fx.PILLAR_DELTA[pillar],
            "strike": strike,
            "bid_iv": iv - 0.001,
            "mid_iv": iv,
            "ask_iv": iv + 0.001,
        }
        for pillar, strike in zip(fx.PILLAR_ORDER, strikes)
    ]


def _snapshot() -> dict:
    return {
        "schema_version": 1,
        "trade_date": "2026-07-20",
        "quote_time": "16:00",
        "currency_pair": "USD.CNY",
        "spot": 7.0,
        "limitations": ["public composite"],
        "slices": [
            {
                "tenor": "1M",
                "maturity": 0.25,
                "expiry_date": "2026-10-20",
                "domestic_rate": 0.03,
                "foreign_rate": 0.02,
                "pricing_foreign_rate": 0.02,
                "forward": 7.0 * math.exp(0.01 * 0.25),
                "quotes": _quotes(0.20),
            },
            {
                "tenor": "2M",
                "maturity": 0.50,
                "expiry_date": "2027-01-20",
                "domestic_rate": 0.03,
                "foreign_rate": 0.02,
                "pricing_foreign_rate": 0.02,
                "forward": 7.0 * math.exp(0.01 * 0.50),
                "quotes": _quotes(0.10),
            },
        ],
    }


def test_surface_separates_raw_nodes_from_smoothing_and_projects_calendar(monkeypatch) -> None:
    snapshot = _snapshot()

    def fake_calibrate(*, market_vols, beta, **_kwargs):
        return {
            "alpha": float(np.asarray(market_vols)[2]),
            "beta": float(beta),
            "rho": 0.0,
            "nu": 0.1,
            "shift": 0.0,
            "mse": 0.0,
        }

    def fake_sabr(_forward, strikes, _maturity, alpha, *_args, **_kwargs):
        return np.full_like(np.asarray(strikes, dtype=float), float(alpha), dtype=float)

    monkeypatch.setattr(surface_stage, "calibrate_sabr_slice", fake_calibrate)
    monkeypatch.setattr(surface_stage, "sabr_implied_vol_black", fake_sabr)

    surface = surface_stage.build_surface(snapshot, ["1M", "2M"], grid_size=9, beta=1.0)

    assert surface["observed_node_count"] == 10
    assert surface["slices"][0]["raw_quotes"] == snapshot["slices"][0]["quotes"]
    assert surface["slices"][1]["raw_quotes"] == snapshot["slices"][1]["quotes"]
    assert surface["surface_preparation"]["calendar_adjusted_nodes"] == 9
    assert "not additional observed liquidity" in surface["surface_preparation"]["purpose"]

    maturities = np.asarray(surface["maturities"], dtype=float)
    total_variance = np.square(np.asarray(surface["iv_grid"], dtype=float)) * maturities[:, None]
    assert np.all(np.diff(total_variance, axis=0) >= -1e-14)
    # The second raw smile was 10%; calendar projection raises the prepared
    # grid.  The observed quote remains unchanged in raw_quotes above.
    assert np.all(np.asarray(surface["iv_grid"])[1] > 0.10)


def test_pricing_environment_reproduces_each_published_forward() -> None:
    snapshot = _snapshot()
    strikes = [6.8, 7.0, 7.2]
    surface = {
        "trade_date": snapshot["trade_date"],
        "spot": snapshot["spot"],
        "strikes": strikes,
        "maturities": [row["maturity"] for row in snapshot["slices"]],
        "iv_grid": [[0.04] * len(strikes) for _ in snapshot["slices"]],
        "slices": [
            {
                "maturity": row["maturity"],
                "domestic_rate": row["domestic_rate"],
                # Stage 02 uses this zero rate implied by CFETS' published F.
                "foreign_rate": row["domestic_rate"]
                - math.log(row["forward"] / snapshot["spot"]) / row["maturity"],
                "forward": row["forward"],
            }
            for row in snapshot["slices"]
        ],
    }
    environment, _ = fx.build_fx_environment(surface)
    for row in surface["slices"]:
        assert environment.get_forward(row["maturity"]) == pytest.approx(
            row["forward"], rel=0.0, abs=2e-14
        )

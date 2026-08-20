"""Contracts for the CFETS Dupire and Heston-SLV example stages.

The fixtures are deliberately synthetic and in memory: these tests verify the
observed/prepared split and numerical artifact contracts without depending on
the CFETS network endpoint or writing into the example data directory.
"""

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

from quantark.volmodels.black_scholes import bs_call_price  # noqa: E402


def _load_numbered_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage03 = _load_numbered_module("03_dupire_localvol.py", "fx_dupire_contract")
stage05 = _load_numbered_module("05_slv_calibration.py", "fx_slv_contract")


def _surface() -> dict:
    """Flat, calendar-admissible IV surface with some raw wing extrapolation."""
    spot = 7.0
    domestic_rate = 0.02
    foreign_rate = 0.01
    maturities = (0.25, 0.50, 1.00)
    tenors = ("3M", "6M", "1Y")
    strikes = np.linspace(6.75, 7.25, 9)
    flat_iv = 0.08
    slices = []
    for tenor, maturity in zip(tenors, maturities):
        forward = spot * math.exp((domestic_rate - foreign_rate) * maturity)
        raw_strikes = (6.65, 6.85, forward, 7.15, 7.35)
        raw_quotes = [
            {
                "pillar": pillar,
                "delta": delta,
                "strike": float(strike),
                "bid_iv": flat_iv - 0.001,
                "mid_iv": flat_iv,
                "ask_iv": flat_iv + 0.001,
            }
            for pillar, delta, strike in zip(
                ("10P", "25P", "ATM", "25C", "10C"),
                (-0.10, -0.25, None, 0.25, 0.10),
                raw_strikes,
            )
        ]
        slices.append(
            {
                "tenor": tenor,
                "role": "calibration_target",
                "maturity": maturity,
                "expiry_date": "2027-07-20",
                "domestic_rate": domestic_rate,
                "foreign_rate": foreign_rate,
                "published_foreign_rate": foreign_rate,
                "published_forward_basis_bps": 0.0,
                "forward": forward,
                "raw_quotes": raw_quotes,
            }
        )
    return {
        "schema_version": 1,
        "trade_date": "2026-07-20",
        "quote_time": "16:00",
        "currency_pair": "USD.CNY",
        "spot": spot,
        "tenor_set": "synthetic",
        "strikes": strikes.tolist(),
        "maturities": list(maturities),
        "iv_grid": np.full((len(maturities), len(strikes)), flat_iv).tolist(),
        "slices": slices,
        "surface_preparation": {
            "method": "synthetic flat-vol fixture",
            "grid_domain": "intersection",
            "shared_strike_interval": [float(strikes[0]), float(strikes[-1])],
            "grid_size": len(strikes),
            "purpose": "differentiable input for Dupire/SLV; not observed liquidity",
        },
        "limitations": ["synthetic test fixture"],
    }


def _heston_report() -> dict:
    free = {"v0": 0.0064, "kappa": 2.0, "theta": 0.0064, "sigma": 0.02, "rho": -0.20}
    hard = dict(free, sigma=math.sqrt(2.0 * free["kappa"] * free["theta"]))
    return {
        "schema_version": 1,
        "universes": {
            "core": {
                "free": {"best": {"success": True, "params": free}},
                "hard_feller": {"best": {"success": True, "params": hard}},
            }
        },
    }


def test_strict_dupire_accepts_admissible_surface_reproducibly() -> None:
    surface = _surface()
    _, first, first_regularization = stage03.build_local_vol(surface)
    _, second, second_regularization = stage03.build_local_vol(surface)

    assert first_regularization == {
        "enabled": False,
        "validate_arbitrage": True,
        "vol_floor": None,
    }
    assert second_regularization == first_regularization
    assert first.lv_grid.shape == (3, 9)
    assert np.all(np.isfinite(first.lv_grid))
    assert np.all(first.lv_grid > 0.0)
    assert first.lv_grid == pytest.approx(0.08, abs=2e-12)
    assert second.lv_grid == pytest.approx(first.lv_grid, rel=0.0, abs=0.0)


def test_stage03_keeps_prepared_fit_separate_from_raw_domain_metrics(monkeypatch) -> None:
    surface = _surface()

    class ClosedFormSolver:
        def __init__(self, *, grid_size, time_steps, local_vol_surface):
            assert grid_size == 31
            assert time_steps == 12
            assert np.all(np.asarray(local_vol_surface.lv_grid) > 0.0)

        @staticmethod
        def price(option, environment):
            maturity = float(option.maturity)
            return bs_call_price(
                environment.spot,
                float(option.strike),
                maturity,
                0.08,
                environment.get_domestic_rate(maturity),
                environment.get_foreign_rate(maturity),
            )

    monkeypatch.setattr(stage03, "FxLocalVolPDESolver", ClosedFormSolver)
    result = stage03.run(surface, grid_size=31, time_steps=12, target_stride=2)

    prepared = result["prepared_target_fit"]
    raw = result["raw_composite_fit"]
    assert result["input_contract"]["observed"] == "raw five-delta CFETS composite nodes"
    assert prepared["node_count"] == 15  # three maturities x five sampled prepared strikes
    assert raw["node_count"] == 15
    assert raw["in_prepared_domain"]["node_count"] == 9
    assert raw["outside_prepared_domain_count"] == 6
    assert raw["acceptance_metric"] == "in_prepared_domain"
    assert sum(row["inside_prepared_strike_domain"] for row in raw["rows"]) == 9
    assert all(np.isfinite(row["model_iv"]) for row in prepared["rows"] + raw["rows"])
    # The closed-form stub makes the intended IV exactly 8%, but the production
    # implied-vol inversion deliberately uses a finite root-finding tolerance.
    # Keep this a contract/finite-value check, not a brittle solver benchmark.
    assert prepared["rmse_iv"] < 1e-5
    assert raw["in_prepared_domain"]["rmse_iv"] < 1e-5


def test_stage05_selects_the_requested_successful_heston_variant_exactly() -> None:
    report = _heston_report()
    free = stage05.select_heston_params(report, "core", "free")
    hard = stage05.select_heston_params(report, "core", "hard_feller")

    assert free.sigma == pytest.approx(0.02)
    assert hard.sigma == pytest.approx(math.sqrt(2.0 * hard.kappa * hard.theta))
    assert free != hard

    unsuccessful = _heston_report()
    unsuccessful["universes"]["core"]["free"]["best"]["success"] = False
    with pytest.raises(ValueError, match="not successful"):
        stage05.select_heston_params(unsuccessful, "core", "free")

    missing = _heston_report()
    del missing["universes"]["core"]["free"]
    with pytest.raises(ValueError, match="no successful core/free best fit"):
        stage05.select_heston_params(missing, "core", "free")


def test_fast_forward_fp_leverage_is_finite_positive_and_diagnostic() -> None:
    surface = _surface()
    params = stage05.select_heston_params(_heston_report(), "core", "free")
    _, local_vol, leverage, time_grid, config = stage05.calibrate_slv(
        surface, params, fast=True
    )

    assert local_vol.lv_grid == pytest.approx(0.08, abs=2e-12)
    assert time_grid.size == 25
    assert config.n_x == 61
    assert config.n_z == 31
    assert config.n_strike_nodes == 21
    assert leverage.leverage_grid.shape == (24, 21)
    assert np.all(np.isfinite(leverage.leverage_grid))
    assert np.all(leverage.leverage_grid > 0.0)
    assert leverage.diagnostics["method"] == "forward_fokker_planck"
    residual = np.asarray(leverage.diagnostics["mass_residual"], dtype=float)
    assert residual.shape == (24,)
    assert np.all(np.isfinite(residual))
    assert np.max(np.abs(residual)) <= config.mass_tol
    assert np.isfinite(float(leverage.diagnostics["max_negative_mass"]))

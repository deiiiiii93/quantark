"""
Tests for per-day vol-model calibration (``vol_calibrators``) and vol-model
engine injection into the OTC autocallable backtest (Tasks 2.3 + 2.4).

Covers:
- Calibrator determinism and kernel-call avoidance via the sha-keyed cache
  (in-memory and persistent round-trip).
- ``create_vol_model_engine`` dispatch for every (vol_model, solver) combo.
- Fail-closed behavior: kernel failures name date + sha, no flat-vol fallback.
- Calibration record schema (JSON-safe, required keys).
- Config validation (vol_model requires vol_source='surface', full_grid
  implication).
- End-to-end smoke: 5-day mini backtest per variant (localvol / heston /
  heston_slv) against a REAL artifact copied into a tmp history, with a
  second run proving the cross-run cache hit (spy-verified, zero kernels).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
)
from quantark.backtest.otc.config import VolModelCalibrationConfig
from quantark.backtest.otc.engine_factory import create_vol_model_engine
from quantark.backtest.otc.vol_calibrators import (
    CalibratedVolModel,
    VolModelCalibrator,
)
from quantark.backtest.otc.vol_history import VolSurfaceHistory
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv.leverage import LeverageSurface

# Kernel spies must patch the CANONICAL module — the otc path is a
# re-export shim since the relocation and patching it intercepts nothing.
import quantark.volmodels.calibration as vol_calibrators


REAL_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "mo_volmodels"
    / "data"
    / "history"
    / "iv_surface"
    / "mo_iv_surface_20230504.json"
)
ARTIFACT_DATE = date(2023, 5, 4)

pytestmark = pytest.mark.skipif(
    not REAL_ARTIFACT.exists(), reason="real IV surface artifact not present"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_history(root: Path) -> Path:
    """Copy the REAL artifact into a tmp history with a correct manifest."""
    history_dir = root / "history"
    surface_dir = history_dir / "iv_surface"
    surface_dir.mkdir(parents=True)
    target = surface_dir / REAL_ARTIFACT.name
    shutil.copyfile(REAL_ARTIFACT, target)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "gap_policy": "consumers carry forward previous admitted surface",
        "records": [
            {
                "date": f"{ARTIFACT_DATE:%Y%m%d}",
                "status": "ok",
                "artifact_sha256": sha,
                "reason": None,
                "detail": None,
            }
        ],
    }
    (history_dir / "surface_manifest.json").write_text(json.dumps(manifest))
    return history_dir


@pytest.fixture()
def history_dir(tmp_path) -> Path:
    return _make_history(tmp_path)


@pytest.fixture()
def artifact(history_dir):
    return VolSurfaceHistory(history_dir).surface_for(ARTIFACT_DATE)


def _fast_calibration_config(**overrides) -> VolModelCalibrationConfig:
    # heston_max_nfev is a *major-iteration* budget on the hard-Feller path:
    # mo_frozen enforces Feller, so calibrate_heston runs SLSQP, where
    # max_nfev maps to maxiter and each major iteration costs ~6 function
    # evaluations for the 5-parameter gradient.  The old value of 15 was tuned
    # to the unconstrained least_squares branch and fails closed under SLSQP
    # ("Iteration limit reached").  Real MO surfaces converge in 8-24 major
    # iterations; 40 clears the observed maximum with margin and costs nothing
    # when the solve converges earlier.
    kwargs = dict(
        heston_max_nfev=40,
        slv_n_steps=8,
        slv_n_x=61,
        slv_n_z=31,
    )
    kwargs.update(overrides)
    return VolModelCalibrationConfig(**kwargs)


@pytest.fixture()
def calibrator(tmp_path) -> VolModelCalibrator:
    return VolModelCalibrator(
        _fast_calibration_config(cache_dir=str(tmp_path / "calib_cache"))
    )


class _KernelSpy:
    """Monkeypatch counter wrapper for a calibration kernel function."""

    def __init__(self, monkeypatch, module, name):
        self.calls = 0
        original = getattr(module, name)
        def counting(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)
        monkeypatch.setattr(module, name, counting)


# ---------------------------------------------------------------------------
# Calibrator determinism + caching
# ---------------------------------------------------------------------------
class TestLocalVolCalibrator:
    def test_determinism_and_memory_cache_hit(self, monkeypatch, calibrator, artifact):
        spy = _KernelSpy(monkeypatch, vol_calibrators, "build_dupire_local_vol")

        first = calibrator.calibrate("localvol", artifact)
        second = calibrator.calibrate("localvol", artifact)

        assert spy.calls == 1
        assert first.record["cache_hit"] is False
        assert second.record["cache_hit"] is True
        assert isinstance(first.local_vol_surface, LocalVolSurface)
        np.testing.assert_array_equal(
            first.local_vol_surface.lv_grid, second.local_vol_surface.lv_grid
        )

    def test_persistent_cache_roundtrip(self, monkeypatch, tmp_path, artifact):
        cache_dir = str(tmp_path / "calib_cache")
        config = _fast_calibration_config(cache_dir=cache_dir)
        first = VolModelCalibrator(config).calibrate("localvol", artifact)
        assert first.record["cache_hit"] is False
        # A fresh calibrator (empty memory) must rebuild from the JSON cache
        # without calling the kernel.
        spy = _KernelSpy(monkeypatch, vol_calibrators, "build_dupire_local_vol")
        second = VolModelCalibrator(config).calibrate("localvol", artifact)
        assert spy.calls == 0
        assert second.record["cache_hit"] is True
        np.testing.assert_array_equal(
            first.local_vol_surface.lv_grid, second.local_vol_surface.lv_grid
        )

    def test_record_schema(self, calibrator, artifact):
        model = calibrator.calibrate("localvol", artifact)
        record = model.record
        for key in (
            "variant",
            "surface_date",
            "surface_sha",
            "cache_hit",
            "calibration_seconds",
            "lv_min",
            "lv_max",
            "grid_shape",
            "n_maturities",
            "n_strikes",
        ):
            assert key in record, f"missing record key {key}"
        assert record["variant"] == "localvol"
        assert record["surface_date"] == ARTIFACT_DATE.isoformat()
        assert record["surface_sha"] == artifact.sha256
        assert record["lv_min"] > 0.0
        assert record["lv_max"] >= record["lv_min"]
        json.dumps(record)  # JSON-safe


class TestHestonCalibrator:
    def test_determinism_and_memory_cache_hit(self, monkeypatch, calibrator, artifact):
        spy = _KernelSpy(monkeypatch, vol_calibrators, "calibrate_heston")

        first = calibrator.calibrate("heston", artifact)
        second = calibrator.calibrate("heston", artifact)

        assert spy.calls == 1
        assert second.record["cache_hit"] is True
        assert isinstance(first.heston_params, HestonParams)
        assert first.heston_params == second.heston_params

    def test_persistent_cache_roundtrip(self, monkeypatch, tmp_path, artifact):
        cache_dir = str(tmp_path / "calib_cache")
        config = _fast_calibration_config(cache_dir=cache_dir)
        first = VolModelCalibrator(config).calibrate("heston", artifact)
        spy = _KernelSpy(monkeypatch, vol_calibrators, "calibrate_heston")
        second = VolModelCalibrator(config).calibrate("heston", artifact)
        assert spy.calls == 0
        assert second.heston_params == first.heston_params

    def test_record_schema(self, calibrator, artifact):
        model = calibrator.calibrate("heston", artifact)
        record = model.record
        for key in (
            "variant",
            "surface_date",
            "surface_sha",
            "cache_hit",
            "calibration_seconds",
            "v0",
            "kappa",
            "theta",
            "sigma",
            "rho",
            "cost",
            "overall_rmse_iv",
            "per_expiry_rmse",
            "bound_hits",
            "feller_satisfied",
            "nfev",
        ):
            assert key in record, f"missing record key {key}"
        assert record["variant"] == "heston"
        assert record["v0"] > 0.0
        assert record["nfev"] >= 1
        assert isinstance(record["per_expiry_rmse"], list)
        assert len(record["per_expiry_rmse"]) == len(artifact.maturities)
        json.dumps(record)

    def test_config_fingerprint_changes_cache_key(
        self, monkeypatch, tmp_path, artifact
    ):
        cache_dir = str(tmp_path / "calib_cache")
        spy = _KernelSpy(monkeypatch, vol_calibrators, "calibrate_heston")
        VolModelCalibrator(
            _fast_calibration_config(cache_dir=cache_dir, heston_max_nfev=40)
        ).calibrate("heston", artifact)
        VolModelCalibrator(
            _fast_calibration_config(cache_dir=cache_dir, heston_max_nfev=41)
        ).calibrate("heston", artifact)
        # Different calibration config -> different cache entry -> recalibrated.
        assert spy.calls == 2

    def test_temporal_reference_changes_cache_key_and_record(
        self, tmp_path, artifact
    ):
        reference = (0.04, 2.0, 0.04, 0.3, -0.5)
        config = _fast_calibration_config(
            cache_dir=str(tmp_path / "calib_cache"),
            heston_temporal_reference=reference,
            heston_temporal_regularization=0.1,
        )
        calibrator = VolModelCalibrator(config)
        legacy_fingerprint = VolModelCalibrator(
            _fast_calibration_config()
        )._config_fingerprint("heston")
        temporal_fingerprint = calibrator._config_fingerprint("heston")
        assert temporal_fingerprint != legacy_fingerprint

        model = calibrator.calibrate("heston", artifact)
        assert model.record["temporal_reference"]["v0"] == pytest.approx(0.04)
        assert model.record["temporal_regularization"] == pytest.approx(0.1)
        assert model.record["temporal_penalty_cost"] >= 0.0


class TestSlvCalibrator:
    def test_slv_reuses_heston_cache_entry(self, monkeypatch, calibrator, artifact):
        heston_spy = _KernelSpy(monkeypatch, vol_calibrators, "calibrate_heston")
        lv_spy = _KernelSpy(monkeypatch, vol_calibrators, "build_dupire_local_vol")
        fp_spy = _KernelSpy(
            monkeypatch, vol_calibrators, "calibrate_leverage_surface_fp"
        )

        heston = calibrator.calibrate("heston", artifact)
        assert heston_spy.calls == 1
        slv = calibrator.calibrate("heston_slv", artifact)

        # Heston kernel NOT recalibrated for SLV (same cache entry).
        assert heston_spy.calls == 1
        assert lv_spy.calls == 1
        assert fp_spy.calls == 1
        assert slv.heston_params == heston.heston_params
        assert isinstance(slv.leverage_surface, LeverageSurface)
        assert isinstance(slv.local_vol_surface, LocalVolSurface)

        again = calibrator.calibrate("heston_slv", artifact)
        assert fp_spy.calls == 1
        assert again.record["cache_hit"] is True
        np.testing.assert_array_equal(
            slv.leverage_surface.leverage_grid, again.leverage_surface.leverage_grid
        )

    def test_record_schema(self, calibrator, artifact):
        model = calibrator.calibrate("heston_slv", artifact)
        record = model.record
        for key in (
            "variant",
            "surface_date",
            "surface_sha",
            "cache_hit",
            "calibration_seconds",
            "leverage_min",
            "leverage_max",
            "leverage_mean",
            "eta",
            "n_steps",
            "n_x",
            "n_z",
            "heston",
        ):
            assert key in record, f"missing record key {key}"
        assert record["variant"] == "heston_slv"
        assert record["leverage_min"] > 0.0
        assert record["leverage_max"] >= record["leverage_min"]
        assert record["heston"]["v0"] > 0.0
        json.dumps(record)

    def test_slv_override_skips_heston_kernel(
        self, monkeypatch, tmp_path, artifact
    ):
        override = (0.04, 2.0, 0.04, 0.3, -0.5)
        config = _fast_calibration_config(
            cache_dir=str(tmp_path / "calib_cache"),
            slv_heston_override=override,
        )
        heston_spy = _KernelSpy(
            monkeypatch, vol_calibrators, "calibrate_heston"
        )
        model = VolModelCalibrator(config).calibrate("heston_slv", artifact)
        assert heston_spy.calls == 0
        assert model.record["heston_source"] == "config_override"
        assert tuple(model.record["heston"][name] for name in (
            "v0", "kappa", "theta", "sigma", "rho"
        )) == pytest.approx(override)


class TestCalibratorFailClosed:
    def test_kernel_failure_names_date_and_sha(self, monkeypatch, calibrator, artifact):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic kernel failure")

        monkeypatch.setattr(vol_calibrators, "build_dupire_local_vol", boom)
        with pytest.raises(ValidationError) as excinfo:
            calibrator.calibrate("localvol", artifact)
        message = str(excinfo.value)
        assert ARTIFACT_DATE.isoformat() in message
        assert artifact.sha256 in message
        assert "synthetic kernel failure" in message

    def test_unknown_variant_fails(self, calibrator, artifact):
        with pytest.raises(ValidationError, match="variant"):
            calibrator.calibrate("svi", artifact)

    def test_corrupt_cache_file_fails_closed(self, monkeypatch, tmp_path, artifact):
        cache_dir = Path(tmp_path / "calib_cache")
        config = _fast_calibration_config(cache_dir=str(cache_dir))
        VolModelCalibrator(config).calibrate("localvol", artifact)
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1
        cache_files[0].write_text("{not json")
        # Fresh calibrator: corrupt persistent entry must not be silently used.
        with pytest.raises(ValidationError, match="cache"):
            VolModelCalibrator(config).calibrate("localvol", artifact)

    def test_sha_mismatch_at_current_schema_fails_closed(self, tmp_path, artifact):
        cache_dir = Path(tmp_path / "calib_cache")
        config = _fast_calibration_config(cache_dir=str(cache_dir))
        VolModelCalibrator(config).calibrate("localvol", artifact)
        cache_file = next(cache_dir.glob("*.json"))
        payload = json.loads(cache_file.read_text())
        payload["surface_sha"] = "0" * 64  # tampered identity, current schema
        cache_file.write_text(json.dumps(payload))
        # Fresh calibrator: identity mismatch at the current schema version
        # is corruption, never a silent miss.
        with pytest.raises(ValidationError, match="cache"):
            VolModelCalibrator(config).calibrate("localvol", artifact)

    def test_stale_schema_version_is_cache_miss_and_recalibrates(
        self, monkeypatch, tmp_path, artifact
    ):
        cache_dir = Path(tmp_path / "calib_cache")
        config = _fast_calibration_config(cache_dir=str(cache_dir))
        first = VolModelCalibrator(config).calibrate("localvol", artifact)
        cache_file = next(cache_dir.glob("*.json"))
        payload = json.loads(cache_file.read_text())
        payload["schema_version"] = -1  # stale schema, e.g. after a version bump
        cache_file.write_text(json.dumps(payload))

        # Fresh calibrator: stale-schema entry is a MISS (recalibrate +
        # atomic overwrite), not a corruption error.
        spy = _KernelSpy(monkeypatch, vol_calibrators, "build_dupire_local_vol")
        second = VolModelCalibrator(config).calibrate("localvol", artifact)
        assert spy.calls == 1
        assert second.record["cache_hit"] is False
        np.testing.assert_array_equal(
            first.local_vol_surface.lv_grid, second.local_vol_surface.lv_grid
        )
        rewritten = json.loads(cache_file.read_text())
        assert rewritten["schema_version"] == vol_calibrators._CACHE_SCHEMA_VERSION

        # The rewritten entry is warm again for the next fresh calibrator.
        third = VolModelCalibrator(config).calibrate("localvol", artifact)
        assert spy.calls == 1
        assert third.record["cache_hit"] is True

    def test_cache_write_failure_carries_context(self, tmp_path, artifact):
        # cache_dir exists as a FILE: mkdir fails when persisting the entry.
        cache_dir = tmp_path / "calib_cache"
        cache_dir.write_text("not a directory")
        config = _fast_calibration_config(cache_dir=str(cache_dir))
        with pytest.raises(ValidationError) as excinfo:
            VolModelCalibrator(config).calibrate("localvol", artifact)
        message = str(excinfo.value)
        assert "cache write" in message
        assert ARTIFACT_DATE.isoformat() in message
        assert artifact.sha256 in message

    def test_missing_parity_points_fail_closed(self, tmp_path):
        # Artifact schema without per-expiry "points": calibration must fail
        # closed, never fall back to a flat vol.
        payload = json.loads(REAL_ARTIFACT.read_text())
        for pillar in payload["per_expiry"]:
            pillar.pop("points", None)
        history_dir = tmp_path / "history"
        surface_dir = history_dir / "iv_surface"
        surface_dir.mkdir(parents=True)
        target = surface_dir / REAL_ARTIFACT.name
        target.write_text(json.dumps(payload))
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = {
            "records": [
                {
                    "date": f"{ARTIFACT_DATE:%Y%m%d}",
                    "status": "ok",
                    "artifact_sha256": sha,
                }
            ]
        }
        (history_dir / "surface_manifest.json").write_text(json.dumps(manifest))
        artifact = VolSurfaceHistory(history_dir).surface_for(ARTIFACT_DATE)
        calibrator = VolModelCalibrator(
            _fast_calibration_config(cache_dir=str(tmp_path / "cache"))
        )
        with pytest.raises(ValidationError):
            calibrator.calibrate("heston", artifact)


# ---------------------------------------------------------------------------
# Engine dispatch
# ---------------------------------------------------------------------------
@pytest.fixture()
def calibrated_models(calibrator, artifact):
    return {
        "localvol": calibrator.calibrate("localvol", artifact),
        "heston": calibrator.calibrate("heston", artifact),
        "heston_slv": calibrator.calibrate("heston_slv", artifact),
    }


class TestCreateVolModelEngine:
    def test_localvol_pde(self, calibrated_models):
        calibrated = calibrated_models["localvol"]
        engine = create_vol_model_engine(
            vol_model="localvol",
            solver="pde",
            calibrated=calibrated,
            pde_params=PDEParams(grid_size=60, time_steps=30),
        )
        assert isinstance(engine, LocalVolSnowballPDESolver)
        assert engine._prebuilt is calibrated.local_vol_surface

    def test_localvol_mc(self, calibrated_models):
        calibrated = calibrated_models["localvol"]
        engine = create_vol_model_engine(
            vol_model="localvol",
            solver="mc",
            calibrated=calibrated,
            mc_params=MCParams(num_paths=128, time_steps=16, seed=5),
        )
        assert isinstance(engine, LocalVolSnowballMCEngine)
        assert engine._prebuilt is calibrated.local_vol_surface

    def test_heston_pde_with_grid_options(self, calibrated_models):
        calibrated = calibrated_models["heston"]
        engine = create_vol_model_engine(
            vol_model="heston",
            solver="pde",
            calibrated=calibrated,
            pde_params=PDEParams(),
            engine_options={"n_x": 41, "n_v": 21, "n_t": 25},
        )
        assert isinstance(engine, HestonSnowballPDESolver)
        assert engine.model_params is calibrated.heston_params
        assert (engine.n_x, engine.n_v, engine.n_t) == (41, 21, 25)

    def test_heston_mc(self, calibrated_models):
        calibrated = calibrated_models["heston"]
        engine = create_vol_model_engine(
            vol_model="heston",
            solver="mc",
            calibrated=calibrated,
            mc_params=MCParams(num_paths=128, time_steps=16, seed=5),
        )
        assert isinstance(engine, HestonSnowballMCEngine)
        assert engine.model_params is calibrated.heston_params

    def test_heston_slv_pde(self, calibrated_models):
        calibrated = calibrated_models["heston_slv"]
        engine = create_vol_model_engine(
            vol_model="heston_slv",
            solver="pde",
            calibrated=calibrated,
            pde_params=PDEParams(),
            engine_options={"n_x": 41, "n_v": 21, "n_t": 25},
        )
        assert isinstance(engine, HestonSLVSnowballPDESolver)
        assert engine.model_params is calibrated.heston_params
        assert engine.leverage_surface is calibrated.leverage_surface
        assert engine.eta == pytest.approx(calibrated.slv_eta)

    def test_heston_slv_mc(self, calibrated_models):
        calibrated = calibrated_models["heston_slv"]
        engine = create_vol_model_engine(
            vol_model="heston_slv",
            solver="mc",
            calibrated=calibrated,
            mc_params=MCParams(num_paths=128, time_steps=16, seed=5),
        )
        assert isinstance(engine, HestonSLVQESnowballMCEngine)
        assert engine.model_params is calibrated.heston_params
        assert engine.leverage_surface is calibrated.leverage_surface
        assert engine.eta == pytest.approx(calibrated.slv_eta)

    def test_variant_mismatch_fails_closed(self, calibrated_models):
        with pytest.raises(ValidationError, match="variant"):
            create_vol_model_engine(
                vol_model="heston",
                solver="pde",
                calibrated=calibrated_models["localvol"],
            )

    def test_unknown_combo_fails_closed(self, calibrated_models):
        with pytest.raises(ValidationError):
            create_vol_model_engine(
                vol_model="bsm",
                solver="pde",
                calibrated=calibrated_models["localvol"],
            )
        with pytest.raises(ValidationError):
            create_vol_model_engine(
                vol_model="localvol",
                solver="quad",
                calibrated=calibrated_models["localvol"],
            )


# ---------------------------------------------------------------------------
# Engine-config validation
# ---------------------------------------------------------------------------
class TestEngineConfigVolModel:
    def test_vol_model_requires_surface_source(self):
        with pytest.raises(ValidationError, match="vol_model"):
            AutocallableEngineConfig(vol_model="heston")

    def test_vol_model_implies_full_grid(self):
        config = AutocallableEngineConfig(vol_source="surface", vol_model="localvol")
        assert config.surface_vol_mode == "full_grid"

    def test_invalid_vol_model_and_solver(self):
        with pytest.raises(ValidationError, match="vol_model"):
            AutocallableEngineConfig(vol_source="surface", vol_model="svi")
        with pytest.raises(ValidationError, match="vol_model_solver"):
            AutocallableEngineConfig(
                vol_source="surface", vol_model="heston", vol_model_solver="fd"
            )

    def test_default_is_bsm_and_calibration_config_materialized(self):
        config = AutocallableEngineConfig()
        assert config.vol_model == "bsm"
        assert config.vol_model_solver == "pde"
        assert isinstance(config.vol_model_calibration, VolModelCalibrationConfig)

    def test_calibration_config_validation(self):
        with pytest.raises(ValidationError, match="heston_max_nfev"):
            VolModelCalibrationConfig(heston_max_nfev=0)
        with pytest.raises(ValidationError, match="slv_eta"):
            VolModelCalibrationConfig(slv_eta=-0.5)
        with pytest.raises(ValidationError, match="slv_n_steps"):
            VolModelCalibrationConfig(slv_n_steps=0)
        with pytest.raises(ValidationError, match="heston_preset"):
            VolModelCalibrationConfig(heston_preset="unknown_preset")
        with pytest.raises(
            ValidationError, match="heston_temporal_reference is required"
        ):
            VolModelCalibrationConfig(heston_temporal_regularization=0.1)
        with pytest.raises(ValidationError, match="five finite parameters"):
            VolModelCalibrationConfig(
                heston_temporal_reference=(0.04, 2.0)
            )
        with pytest.raises(ValidationError, match="must lie within"):
            VolModelCalibrationConfig(
                slv_heston_override=(0.04, 2.0, 0.04, 0.3, 0.5)
            )


# ---------------------------------------------------------------------------
# End-to-end smoke: 5-day mini backtest per vol-model variant
# ---------------------------------------------------------------------------
SPOT = 6700.0


def _market_frames(dates):
    spot_data = pd.DataFrame({"date": dates, "spot": [SPOT] * len(dates)})
    vol_data = pd.DataFrame({"date": dates, "volatility": [0.20] * len(dates)})
    rate_data = pd.DataFrame({"date": dates, "rate": [0.02] * len(dates)})
    futures_rows = [
        {
            "date": d,
            "contract": "IM2306",
            "futures_price": SPOT * 1.001,
            "expiry_date": pd.Timestamp("2023-06-16"),
            "multiplier": 200.0,
        }
        for d in dates
    ]
    return spot_data, vol_data, rate_data, pd.DataFrame(futures_rows)


def _product():
    return create_standard_snowball(
        initial_price=SPOT,
        strike=SPOT,
        maturity=0.5,
        contract_multiplier=1.0,
        ko_barrier=SPOT * 1.05,
        ko_rate=0.05,
        ki_barrier=SPOT * 0.75,
        num_observations=6,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )


def _run_smoke(history_dir, vol_model, solver, calibration_config):
    dates = pd.date_range(ARTIFACT_DATE, periods=5, freq="D")
    spot_data, vol_data, rate_data, futures_data = _market_frames(dates)
    market_data = AutocallableMarketDataSet.from_dataframes(
        spot_data=spot_data,
        vol_data=vol_data,
        rate_data=rate_data,
        futures_data=futures_data,
        surface_history=VolSurfaceHistory(history_dir),
    )
    if solver == "pde":
        # n_x/n_v/n_t are 2D Heston-family PDE solver knobs; the 1D LV solver
        # takes PDEParams only, and unknown options raise (fail-closed).
        engine_options = (
            {"n_x": 61, "n_v": 21, "n_t": 25}
            if vol_model in ("heston", "heston_slv")
            else {}
        )
        engine_config = AutocallableEngineConfig(
            pricing_engine_type=EngineType.PDE,
            # 0.4.0: snowball-family PDE solvers reject legacy
            # grid_size/time_steps; "fast" keeps the smoke cheap.
            pde_params=PDEParams(accuracy="fast"),
            vol_source="surface",
            vol_model=vol_model,
            vol_model_solver="pde",
            vol_model_calibration=calibration_config,
            vol_model_engine_options=engine_options,
        )
    else:
        engine_config = AutocallableEngineConfig(
            pricing_engine_type=EngineType.MONTE_CARLO,
            mc_params=MCParams(num_paths=256, time_steps=16, seed=13),
            vol_source="surface",
            vol_model=vol_model,
            vol_model_solver="mc",
            vol_model_calibration=calibration_config,
        )
    config = AutocallableBacktestConfig(
        product=_product(),
        market_data=market_data,
        engine_config=engine_config,
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        product_quantity=-1.0,
        underlying="CSI1000",
    )
    started = time.perf_counter()
    results = AutocallableBacktestEngine(config).run()
    elapsed = time.perf_counter() - started
    return results, elapsed


def _assert_smoke_results(results, expected_days=5):
    states = results.states_df
    assert len(states) == expected_days
    assert np.all(np.isfinite(states["product_mtm"]))
    greeks = results.greeks_df
    assert np.all(np.isfinite(greeks["price"]))
    assert np.all(np.isfinite(greeks["delta"]))
    # An alive snowball has nonzero delta; the native greek path must have run.
    assert (greeks["delta"].abs() > 1e-12).all()
    records = results.calibration_records
    assert len(records) == expected_days
    for record in records:
        assert record["surface_sha"]
        assert record["surface_date"] == ARTIFACT_DATE.isoformat()
        json.dumps(record)


class TestEndToEndSmoke:
    def test_localvol_pde(self, tmp_path, history_dir):
        calibration_config = _fast_calibration_config(
            cache_dir=str(tmp_path / "cache"),
            records_path=str(tmp_path / "records.json"),
        )
        results, elapsed = _run_smoke(
            history_dir, "localvol", "pde", calibration_config
        )
        _assert_smoke_results(results)
        for record in results.calibration_records:
            assert record["variant"] == "localvol"
            assert "lv_min" in record
        persisted = json.loads((tmp_path / "records.json").read_text())
        assert len(persisted) == 5
        print(f"\nlocalvol/pde smoke: {elapsed:.2f}s total ({elapsed / 5:.2f}s/day)")

    def test_heston_pde(self, tmp_path, history_dir):
        calibration_config = _fast_calibration_config(
            cache_dir=str(tmp_path / "cache")
        )
        results, elapsed = _run_smoke(history_dir, "heston", "pde", calibration_config)
        _assert_smoke_results(results)
        for record in results.calibration_records:
            assert record["variant"] == "heston"
            assert "v0" in record
        print(f"\nheston/pde smoke: {elapsed:.2f}s total ({elapsed / 5:.2f}s/day)")

    def test_heston_slv_pde(self, tmp_path, history_dir):
        calibration_config = _fast_calibration_config(
            cache_dir=str(tmp_path / "cache")
        )
        results, elapsed = _run_smoke(
            history_dir, "heston_slv", "pde", calibration_config
        )
        _assert_smoke_results(results)
        for record in results.calibration_records:
            assert record["variant"] == "heston_slv"
            assert "leverage_min" in record
        print(f"\nheston_slv/pde smoke: {elapsed:.2f}s total ({elapsed / 5:.2f}s/day)")

    def test_localvol_mc(self, tmp_path, history_dir):
        calibration_config = _fast_calibration_config(
            cache_dir=str(tmp_path / "cache")
        )
        results, elapsed = _run_smoke(history_dir, "localvol", "mc", calibration_config)
        _assert_smoke_results(results)
        print(f"\nlocalvol/mc smoke: {elapsed:.2f}s total ({elapsed / 5:.2f}s/day)")

    def test_second_run_hits_persistent_cache(
        self, monkeypatch, tmp_path, history_dir
    ):
        cache_dir = str(tmp_path / "cache")
        calibration_config = _fast_calibration_config(cache_dir=cache_dir)
        lv_spy = _KernelSpy(monkeypatch, vol_calibrators, "build_dupire_local_vol")

        first, first_elapsed = _run_smoke(
            history_dir, "localvol", "pde", calibration_config
        )
        assert lv_spy.calls == 1  # one surface -> one calibration for 5 days
        _assert_smoke_results(first)

        # Second run, fresh engine, same on-disk cache: zero kernel calls and
        # every record flags a cache hit (spy-verified; wall-clock printed
        # for information only, not asserted).
        second, second_elapsed = _run_smoke(
            history_dir, "localvol", "pde", calibration_config
        )
        assert lv_spy.calls == 1
        _assert_smoke_results(second)
        assert all(r["cache_hit"] for r in second.calibration_records)
        assert not all(r["cache_hit"] for r in first.calibration_records)
        print(
            f"\ncache rerun: first={first_elapsed:.2f}s second={second_elapsed:.2f}s"
        )

    def test_heston_calibration_failure_fails_run_closed(
        self, monkeypatch, tmp_path, history_dir
    ):
        def boom(*args, **kwargs):
            raise RuntimeError("synthetic heston failure")

        monkeypatch.setattr(vol_calibrators, "calibrate_heston", boom)
        calibration_config = _fast_calibration_config(
            cache_dir=str(tmp_path / "cache")
        )
        with pytest.raises(ValidationError) as excinfo:
            _run_smoke(history_dir, "heston", "pde", calibration_config)
        message = str(excinfo.value)
        assert ARTIFACT_DATE.isoformat() in message
        assert "synthetic heston failure" in message


class TestVolModelMcMethodSlot:
    """``vol_model_mc_method`` separates two methods that used to collide.

    ``method`` belongs to ``pricing_engine_type`` and is what the surface and
    event-stats engines receive.  A vol-model MC route needs its own method
    (``MonteCarloMethod.RANDOMIZED_QUASI``), which a PDE engine rejects
    outright ("Invalid method type"), so a Heston-on-RQMC run that keeps the
    deterministic PDE reporting engines could not express both through one
    field.
    """

    def test_defaults_to_none_and_changes_nothing(self):
        config = AutocallableEngineConfig()
        assert config.vol_model_mc_method is None
        assert config.resolve_vol_model_mc_method() is None

    def test_falls_back_to_method_for_existing_configurations(self):
        from quantark.util.enum.engine_enums import MonteCarloMethod

        config = AutocallableEngineConfig(
            pricing_engine_type=EngineType.MONTE_CARLO,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
        )
        assert config.resolve_vol_model_mc_method() == MonteCarloMethod.RANDOMIZED_QUASI

    def test_dedicated_slot_wins_over_method(self):
        from quantark.util.enum.engine_enums import MonteCarloMethod

        config = AutocallableEngineConfig(
            method=None,
            vol_model_mc_method=MonteCarloMethod.RANDOMIZED_QUASI,
        )
        assert config.resolve_vol_model_mc_method() == MonteCarloMethod.RANDOMIZED_QUASI

    def test_pde_reporting_engines_stay_constructible(self):
        """The regression: an RQMC vol-model route + PDE reporting engines."""
        from quantark.backtest.otc.engine_factory import (
            create_event_stats_engine,
            create_pricing_engine,
            create_surface_engine,
        )
        from quantark.util.enum.engine_enums import MonteCarloMethod

        product = create_standard_snowball(
            initial_price=100.0, strike=100.0, maturity=1.0,
            ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
            num_observations=12,
        )
        config = AutocallableEngineConfig(
            pricing_engine_type=EngineType.PDE,
            pde_params=PDEParams(),
            mc_params=MCParams(num_paths=128, seed=1),
            vol_source="surface",
            vol_model="heston",
            vol_model_solver="mc",
            vol_model_mc_method=MonteCarloMethod.RANDOMIZED_QUASI,
        )
        # None of these may raise "Invalid method type".
        assert create_pricing_engine(product, config) is not None
        assert create_surface_engine(product, config) is not None
        assert create_event_stats_engine(product, config) is not None
        assert config.resolve_event_stats_engine_type() == EngineType.PDE

    def test_putting_the_mc_method_in_method_still_breaks_pde(self):
        """Guards the reason the slot exists - don't 'simplify' it away."""
        from quantark.backtest.otc.engine_factory import create_pricing_engine
        from quantark.util.enum.engine_enums import MonteCarloMethod

        product = create_standard_snowball(
            initial_price=100.0, strike=100.0, maturity=1.0,
            ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
            num_observations=12,
        )
        config = AutocallableEngineConfig(
            pricing_engine_type=EngineType.PDE,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
        )
        with pytest.raises(ValidationError, match="Invalid method type"):
            create_pricing_engine(product, config)

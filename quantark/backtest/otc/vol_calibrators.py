"""
Per-day vol-model calibration for OTC autocallable backtests.

Calibrates one vol model per admitted IV-surface artifact and variant
(``localvol`` / ``heston`` / ``heston_slv``) so the backtest can reprice
daily with model-consistent snowball engines.  Calibration inputs come
ONLY from the artifact itself (its SABR-smoothed IV grid and its own
parity ``r``/``q`` pillars), never from the backtest's daily market row:
the cache is keyed by surface sha256 and must be reusable across runs
(monthly inceptions) that share the same calendar-date surface.

Cache layout: ``cache_dir / f"{variant}-{key}.json"`` where
``key = sha256(f"{surface_sha}|{variant}|{config_fingerprint}")`` and the
fingerprint is canonical JSON of the frozen calibration configuration.
Files store params (grids / Heston vector), not live objects; cache hits
rebuild the surface/param objects deterministically without calling the
calibration kernels.  Writes are atomic (tmp file + ``os.replace``).

Fail-closed: any calibration failure raises ``ValidationError`` naming the
surface date and sha.  There is NO fallback to flat vol.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quantark.param import TermStructureDividendYield
from quantark.param.rrf import LinearRateCurve
from quantark.param.term_sampling import forward_carry_on_grid, forward_rates_on_grid
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.io import atomic_write_json
from quantark.volmodels.black_scholes import implied_vol_call
from quantark.volmodels.heston import (
    HestonParams,
    MarketOption,
    calibrate_heston,
    heston_call_prices_vectorized,
)
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.slv.fokkerplanck import (
    FpCalibrationConfig,
    calibrate_leverage_surface_fp,
)
from quantark.volmodels.slv.leverage import LeverageSurface

from quantark.param.vol.surface_history import IvSurfaceArtifact

VOL_MODEL_LOCALVOL = "localvol"
VOL_MODEL_HESTON = "heston"
VOL_MODEL_HESTON_SLV = "heston_slv"
VOL_MODEL_VARIANTS = (VOL_MODEL_LOCALVOL, VOL_MODEL_HESTON, VOL_MODEL_HESTON_SLV)

HESTON_PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")

# Frozen Heston calibration preset "mo_frozen".
# Provenance: values copied from the mo_volmodels suite —
# example/mo_volmodels/04_heston_calibration.py (HESTON_BOUNDS,
# REGULARIZE_FELLER, SOLVER_TOLERANCES, target="iv", method="lewis",
# enforce_feller=False, single deterministic shortest-expiry ATM-variance
# start) and example/mo_volmodels/10_calibration_diagnostics.py (same
# bounds and soft-Feller policy, frozen 2026-07 for the CFFEX MO cohort).
# example/ scripts are NOT importable from quantark (canonical-import
# rule), so the frozen configuration is re-declared here; keep it in sync
# with the suite.
HESTON_PRESETS: Dict[str, Dict[str, Any]] = {
    "mo_frozen": {
        "bounds": (
            (1e-6, 1e-3, 1e-4, 1e-3, -0.95),
            (0.5, 3.0, 0.5, 0.7, 0.0),
        ),
        "regularize_feller": 0.05,
        "solver_tolerances": {"xtol": 1e-6, "ftol": 1e-6, "gtol": 1e-6},
        "target": "iv",
        "method": "lewis",
        "enforce_feller": False,
    }
}

# SLV leverage-calibration grid defaults mirror the mo suite
# (_mo_common.calibrate_leverage_for: n_steps=40, n_x=161, n_z=81); the
# values themselves live on VolModelCalibrationConfig (config.py) so runs
# can coarsen them for smoke tests.
#
# OBLIGATION: bump _CACHE_SCHEMA_VERSION when any kernel default that
# affects calibration output changes — including FpCalibrationConfig
# internals and the Dupire local-vol kernel internals.  Those fixed kernel
# defaults are NOT part of the cache fingerprint, so the version bump is
# the only mechanism that invalidates warm entries after a kernel upgrade
# (stale-schema entries are treated as cache misses and recalibrated).
_CACHE_SCHEMA_VERSION = 1


@dataclass
class CalibratedVolModel:
    """One calibrated vol model for one surface artifact and variant.

    Attributes:
        variant: "localvol" | "heston" | "heston_slv".
        surface_date: artifact trade date.
        surface_sha: artifact sha256 (cache key component).
        local_vol_surface: Dupire LV surface (localvol; also carried by
            heston_slv for provenance/on-the-fly consumers).
        heston_params: calibrated Heston vector (heston / heston_slv).
        leverage_surface: calibrated SLV leverage (heston_slv).
        slv_eta: SLV vol-of-vol mixer used at calibration time (heston_slv);
            engines must reuse the same eta.
        record: JSON-safe calibration record (see VolModelCalibrator).
    """

    variant: str
    surface_date: date
    surface_sha: str
    local_vol_surface: Optional[LocalVolSurface] = None
    heston_params: Optional[HestonParams] = None
    leverage_surface: Optional[LeverageSurface] = None
    slv_eta: Optional[float] = None
    record: Dict[str, Any] = field(default_factory=dict)


# Canonical home: quantark.util.io (kept as an alias for existing importers).
_atomic_write_json = atomic_write_json


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class VolModelCalibrator:
    """Per-surface vol-model calibration with a sha-keyed persistent cache.

    Args:
        config: a ``VolModelCalibrationConfig`` (duck-typed: cache_dir,
            heston_preset, heston_max_nfev, slv_eta, slv_n_steps, slv_n_x,
            slv_n_z, shared_store).  ``cache_dir=None`` means in-memory
            caching only; ``shared_store`` lets a fleet of backtest runs
            share one in-memory cache.
    """

    def __init__(self, config: Any) -> None:
        preset = getattr(config, "heston_preset", "mo_frozen")
        if preset not in HESTON_PRESETS:
            raise ValidationError(
                f"heston_preset must be one of {sorted(HESTON_PRESETS)}, "
                f"got {preset!r}"
            )
        self._config = config
        self._memory: Dict[str, Dict[str, Any]] = (
            config.shared_store if config.shared_store is not None else {}
        )
        self._cache_dir = (
            Path(config.cache_dir) if config.cache_dir is not None else None
        )

    @property
    def config(self) -> Any:
        return self._config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def calibrate(
        self, variant: str, artifact: IvSurfaceArtifact
    ) -> CalibratedVolModel:
        """Return the calibrated model for (variant, artifact), cached by sha.

        Cache hits (memory first, then disk) rebuild the model objects from
        stored params without calling the calibration kernels.  Any kernel
        failure raises ``ValidationError`` naming the surface date and sha
        (fail-closed; never falls back to flat vol).
        """
        if variant not in VOL_MODEL_VARIANTS:
            raise ValidationError(
                f"Unknown vol-model variant {variant!r}; "
                f"expected one of {VOL_MODEL_VARIANTS}"
            )
        started = time.perf_counter()
        fingerprint = self._config_fingerprint(variant)
        key = hashlib.sha256(
            f"{artifact.sha256}|{variant}|{fingerprint}".encode("utf-8")
        ).hexdigest()

        payload = self._memory.get(key)
        if payload is None:
            payload = self._read_disk(variant, key, artifact, fingerprint)
        if payload is not None:
            model = self._reconstruct(variant, artifact, payload)
            model.record = {
                **payload["record"],
                "cache_hit": True,
                "calibration_seconds": time.perf_counter() - started,
            }
            return model

        payload = self._calibrate_fresh(variant, artifact, started)
        self._memory[key] = payload
        self._write_disk(variant, key, payload, artifact)
        model = self._reconstruct(variant, artifact, payload)
        model.record = {**payload["record"], "cache_hit": False}
        return model

    # ------------------------------------------------------------------
    # Fresh calibration (kernel calls; fail-closed wrapper)
    # ------------------------------------------------------------------
    def _calibrate_fresh(
        self, variant: str, artifact: IvSurfaceArtifact, started: float
    ) -> Dict[str, Any]:
        try:
            if variant == VOL_MODEL_LOCALVOL:
                params, record = self._calibrate_localvol(artifact)
            elif variant == VOL_MODEL_HESTON:
                params, record = self._calibrate_heston(artifact)
            else:
                params, record = self._calibrate_slv(artifact)
        except Exception as exc:
            raise ValidationError(
                "Vol-model calibration failed "
                f"(variant={variant}, surface_date="
                f"{artifact.trade_date.isoformat()}, "
                f"surface_sha={artifact.sha256}): {exc}"
            ) from exc
        record.update(
            {
                "variant": variant,
                "surface_date": artifact.trade_date.isoformat(),
                "surface_sha": artifact.sha256,
                "cache_hit": False,
                "calibration_seconds": time.perf_counter() - started,
            }
        )
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "variant": variant,
            "surface_sha": artifact.sha256,
            "surface_date": artifact.trade_date.isoformat(),
            "config_fingerprint": self._config_fingerprint(variant),
            "params": params,
            "record": record,
        }

    def _calibrate_localvol(
        self, artifact: IvSurfaceArtifact
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rate_curve, div_curve = self._artifact_curves(artifact)
        # The artifacts already passed validate_arbitrage=True at build time
        # (manifest admission policy), so the strict default stays on.
        lv = build_dupire_local_vol(
            artifact.grid_vol_surface(),
            spot=float(artifact.s0),
            rate_curve=rate_curve,
            div_yield=div_curve.get_yield,
        )
        lv_grid = np.asarray(lv.lv_grid, dtype=float)
        params = {
            "strike_grid": [float(x) for x in lv.strike_grid],
            "time_grid": [float(x) for x in lv.time_grid],
            "lv_grid": lv_grid.tolist(),
        }
        record = {
            "lv_min": float(np.min(lv_grid)),
            "lv_max": float(np.max(lv_grid)),
            "grid_shape": [int(lv_grid.shape[0]), int(lv_grid.shape[1])],
            "n_maturities": len(artifact.maturities),
            "n_strikes": len(artifact.strikes),
        }
        return params, record

    def _calibrate_heston(
        self, artifact: IvSurfaceArtifact
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self._config
        preset = HESTON_PRESETS[cfg.heston_preset]
        rate_curve, div_curve = self._artifact_curves(artifact)
        s0 = float(artifact.s0)
        nodes = self._heston_nodes(artifact)
        initial = self._initial_heston_guess(artifact)
        options = [
            MarketOption(K=node["K"], T=node["T"], iv=node["iv"], weight=1.0)
            for node in nodes
        ]
        result = calibrate_heston(
            s0=s0,
            options=options,
            r=rate_curve.get_rate,
            carry=div_curve.get_yield,
            initial=initial,
            bounds=preset["bounds"],
            target=preset["target"],
            method=preset["method"],
            regularize_feller=preset["regularize_feller"],
            enforce_feller=preset["enforce_feller"],
            max_nfev=int(cfg.heston_max_nfev),
            **preset["solver_tolerances"],
        )
        if result.success is not True:
            raise ValidationError(
                f"Heston optimizer failed: {result.message} "
                f"(nfev={result.nfev}, cost={result.cost:.6g})"
            )
        params = result.params
        metrics = self._heston_fit_metrics(s0, nodes, params)
        params_payload = {
            name: float(getattr(params, name)) for name in HESTON_PARAMETER_NAMES
        }
        record = {
            **params_payload,
            "cost": float(result.cost),
            "data_cost": float(result.data_cost),
            "feller_penalty_cost": float(result.feller_penalty_cost),
            "feller_margin": float(result.feller_margin),
            "feller_satisfied": bool(params.feller_satisfied()),
            "nfev": int(result.nfev),
            "optimizer": str(result.optimizer),
            "message": str(result.message),
            "overall_rmse_iv": metrics["overall_rmse_iv"],
            "per_expiry_rmse": metrics["per_expiry_rmse"],
            "bound_hits": self._bound_hits(params, preset["bounds"]),
            "n_nodes": len(nodes),
            "heston_preset": str(cfg.heston_preset),
        }
        return params_payload, record

    def _calibrate_slv(
        self, artifact: IvSurfaceArtifact
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self._config
        eta = float(cfg.slv_eta)
        n_steps = int(cfg.slv_n_steps)
        # Reuse the SAME cache entries: no recalibration of LV or Heston.
        lv_model = self.calibrate(VOL_MODEL_LOCALVOL, artifact)
        heston_model = self.calibrate(VOL_MODEL_HESTON, artifact)
        rate_curve, div_curve = self._artifact_curves(artifact)
        # Suite grid (05_slv_calibration.py): uniform time grid to the last
        # listed expiry, forward rate/carry per interval from the artifact
        # parity curves.
        t_grid = np.linspace(0.0, float(max(artifact.maturities)), n_steps + 1)
        r_fwd = forward_rates_on_grid(rate_curve, t_grid)
        carry_fwd = forward_carry_on_grid(div_curve.get_yield, t_grid)
        leverage = calibrate_leverage_surface_fp(
            float(artifact.s0),
            heston_model.heston_params,
            lv_model.local_vol_surface,
            np.diff(t_grid),
            r_fwd,
            carry_fwd,
            eta=eta,
            config=FpCalibrationConfig(n_x=int(cfg.slv_n_x), n_z=int(cfg.slv_n_z)),
        )
        lg = np.asarray(leverage.leverage_grid, dtype=float)
        diagnostics = dict(leverage.diagnostics or {})
        heston_payload = {
            name: float(getattr(heston_model.heston_params, name))
            for name in HESTON_PARAMETER_NAMES
        }
        params = {
            "heston": heston_payload,
            "time_grid": [float(x) for x in leverage.time_grid],
            "strike_grid": [float(x) for x in leverage.strike_grid],
            "leverage_grid": lg.tolist(),
            "eta": eta,
        }
        record = {
            "leverage_min": float(np.min(lg)),
            "leverage_max": float(np.max(lg)),
            "leverage_mean": float(np.mean(lg)),
            "eta": eta,
            "n_steps": n_steps,
            "n_x": int(cfg.slv_n_x),
            "n_z": int(cfg.slv_n_z),
            "heston": heston_payload,
            "n_clipped": int(diagnostics.get("n_clipped", 0)),
            "max_negative_mass": float(diagnostics.get("max_negative_mass", 0.0)),
        }
        return params, record

    # ------------------------------------------------------------------
    # Artifact-derived helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _artifact_curves(
        artifact: IvSurfaceArtifact,
    ) -> Tuple[LinearRateCurve, TermStructureDividendYield]:
        """Rate/carry curves from the artifact's own parity pillars.

        Calibration deliberately uses the artifact's recorded ``r``/``q``
        (not the backtest's daily market rate) so the sha-keyed cache entry
        is a pure function of the artifact.
        """
        times: List[float] = []
        rates: List[float] = []
        carries: List[float] = []
        for pillar in artifact.per_expiry:
            if "r" not in pillar or "q" not in pillar:
                raise ValidationError(
                    f"IV surface artifact {artifact.trade_date.isoformat()} "
                    "per_expiry pillar missing 'r'/'q' — required for "
                    "vol-model calibration"
                )
            times.append(float(pillar["T"]))
            rates.append(float(pillar["r"]))
            carries.append(float(pillar["q"]))
        return (
            LinearRateCurve(list(zip(times, rates))),
            TermStructureDividendYield(times=times, yields=carries),
        )

    @staticmethod
    def _heston_nodes(artifact: IvSurfaceArtifact) -> List[Dict[str, float]]:
        """Flatten the artifact's prepared (SABR-smoothed) IV target nodes."""
        nodes: List[Dict[str, float]] = []
        for pillar in artifact.per_expiry:
            points = pillar.get("points")
            if not points:
                raise ValidationError(
                    f"IV surface artifact {artifact.trade_date.isoformat()} "
                    f"per_expiry pillar at T={pillar.get('T')} has no "
                    "'points' — required for Heston calibration"
                )
            for point in points:
                nodes.append(
                    {
                        "K": float(point[0]),
                        "T": float(pillar["T"]),
                        "iv": float(point[1]),
                        "r": float(pillar["r"]),
                        "q": float(pillar["q"]),
                    }
                )
        if not nodes:
            raise ValidationError(
                f"IV surface artifact {artifact.trade_date.isoformat()} "
                "yields no Heston calibration nodes"
            )
        return nodes

    @staticmethod
    def _initial_heston_guess(artifact: IvSurfaceArtifact) -> HestonParams:
        """Shortest-expiry ATM-variance anchor (suite stage-04 start policy)."""
        atm0 = min(artifact.atm_pillars, key=lambda p: p["T"])
        v_level = float(np.clip(float(atm0["atm_vol"]) ** 2, 0.01, 0.2))
        return HestonParams(v0=v_level, kappa=2.0, theta=v_level, sigma=0.6, rho=-0.5)

    @staticmethod
    def _heston_fit_metrics(
        s0: float, nodes: List[Dict[str, float]], params: HestonParams
    ) -> Dict[str, Any]:
        """Model-vs-target IV RMSE, overall and per expiry (one Lewis sweep each)."""
        maturities = np.array([node["T"] for node in nodes], dtype=float)
        model = np.empty(len(nodes), dtype=float)
        for maturity in np.unique(maturities):
            idx = np.flatnonzero(maturities == maturity)
            ks = np.array([nodes[int(i)]["K"] for i in idx], dtype=float)
            rate = float(nodes[int(idx[0])]["r"])
            carry = float(nodes[int(idx[0])]["q"])
            prices = heston_call_prices_vectorized(
                s0, ks, float(maturity), params, rate, carry
            )
            model[idx] = [
                implied_vol_call(s0, float(k), float(maturity), float(p), rate, carry)
                for k, p in zip(ks, prices)
            ]
        if not np.all(np.isfinite(model)):
            raise NumericalError(
                "Heston model produced non-finite implied vols at the fitted parameters"
            )
        target = np.array([node["iv"] for node in nodes], dtype=float)
        errors = model - target
        per_expiry = [
            {
                "T": float(maturity),
                "rmse_iv": float(
                    np.sqrt(np.mean(errors[np.flatnonzero(maturities == maturity)] ** 2))
                ),
            }
            for maturity in np.unique(maturities)
        ]
        return {
            "overall_rmse_iv": float(np.sqrt(np.mean(errors**2))),
            "per_expiry_rmse": per_expiry,
        }

    @staticmethod
    def _bound_hits(params: HestonParams, bounds) -> Dict[str, Dict[str, bool]]:
        """Suite stage-04 bound-hit diagnostic (relative tolerance 1e-7)."""
        values = np.array(
            [getattr(params, name) for name in HESTON_PARAMETER_NAMES], dtype=float
        )
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        tolerance = np.maximum(1e-10, 1e-7 * (upper - lower))
        return {
            name: {
                "lower": bool(abs(value - lo) <= tol),
                "upper": bool(abs(hi - value) <= tol),
            }
            for name, value, lo, hi, tol in zip(
                HESTON_PARAMETER_NAMES, values, lower, upper, tolerance
            )
        }

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------
    def _config_fingerprint(self, variant: str) -> str:
        """Canonical JSON fingerprint of the variant's frozen calibration config."""
        cfg = self._config
        if variant == VOL_MODEL_LOCALVOL:
            payload = {
                "kernel": "build_dupire_local_vol",
                "validate_arbitrage": True,
                "schema": _CACHE_SCHEMA_VERSION,
            }
        elif variant == VOL_MODEL_HESTON:
            payload = {
                "kernel": "calibrate_heston",
                "preset_name": str(cfg.heston_preset),
                "preset": _json_safe(HESTON_PRESETS[cfg.heston_preset]),
                "max_nfev": int(cfg.heston_max_nfev),
                "schema": _CACHE_SCHEMA_VERSION,
            }
        else:
            payload = {
                "kernel": "calibrate_leverage_surface_fp",
                "heston": self._config_fingerprint(VOL_MODEL_HESTON),
                "localvol": self._config_fingerprint(VOL_MODEL_LOCALVOL),
                "eta": float(cfg.slv_eta),
                "n_steps": int(cfg.slv_n_steps),
                "n_x": int(cfg.slv_n_x),
                "n_z": int(cfg.slv_n_z),
                "schema": _CACHE_SCHEMA_VERSION,
            }
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def _reconstruct(
        self, variant: str, artifact: IvSurfaceArtifact, payload: Dict[str, Any]
    ) -> CalibratedVolModel:
        """Rebuild model objects from stored params (no kernel calls)."""
        params = payload["params"]
        try:
            if variant == VOL_MODEL_LOCALVOL:
                lv = LocalVolSurface(
                    strike_grid=params["strike_grid"],
                    time_grid=params["time_grid"],
                    lv_grid=params["lv_grid"],
                )
                return CalibratedVolModel(
                    variant=variant,
                    surface_date=artifact.trade_date,
                    surface_sha=artifact.sha256,
                    local_vol_surface=lv,
                )
            if variant == VOL_MODEL_HESTON:
                hp = HestonParams(
                    **{
                        name: float(params[name])
                        for name in HESTON_PARAMETER_NAMES
                    }
                )
                return CalibratedVolModel(
                    variant=variant,
                    surface_date=artifact.trade_date,
                    surface_sha=artifact.sha256,
                    heston_params=hp,
                )
            hp = HestonParams(
                **{
                    name: float(params["heston"][name])
                    for name in HESTON_PARAMETER_NAMES
                }
            )
            leverage = LeverageSurface(
                time_grid=params["time_grid"],
                strike_grid=params["strike_grid"],
                leverage_grid=params["leverage_grid"],
            )
            # The LV surface rides along via its own cache entry (hit).
            lv = self.calibrate(VOL_MODEL_LOCALVOL, artifact).local_vol_surface
            return CalibratedVolModel(
                variant=variant,
                surface_date=artifact.trade_date,
                surface_sha=artifact.sha256,
                local_vol_surface=lv,
                heston_params=hp,
                leverage_surface=leverage,
                slv_eta=float(params["eta"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(
                f"Calibration cache entry for variant {variant!r} on surface "
                f"{artifact.trade_date.isoformat()} ({artifact.sha256}) is "
                f"corrupt: {exc}"
            ) from exc

    def _cache_path(self, variant: str, key: str) -> Path:
        return self._cache_dir / f"{variant}-{key}.json"

    def _read_disk(
        self,
        variant: str,
        key: str,
        artifact: IvSurfaceArtifact,
        fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        if self._cache_dir is None:
            return None
        path = self._cache_path(variant, key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(
                f"Calibration cache entry not readable: {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValidationError(f"Calibration cache entry {path} is corrupt")
        if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
            # Stale schema (e.g. after a _CACHE_SCHEMA_VERSION bump on a
            # kernel upgrade): treat as a cache MISS — the caller
            # recalibrates and atomically overwrites this entry.  Only
            # entries at the CURRENT schema version are validated (and can
            # fail closed) below.
            return None
        if not isinstance(payload.get("params"), dict) or not isinstance(
            payload.get("record"), dict
        ):
            raise ValidationError(f"Calibration cache entry {path} is corrupt")
        for field_name, expected in (
            ("variant", variant),
            ("surface_sha", artifact.sha256),
            ("config_fingerprint", fingerprint),
        ):
            if payload.get(field_name) != expected:
                raise ValidationError(
                    f"Calibration cache entry {path} mismatch on "
                    f"{field_name!r}: expected {expected!r}, found "
                    f"{payload.get(field_name)!r}"
                )
        self._memory[key] = payload
        return payload

    def _write_disk(
        self,
        variant: str,
        key: str,
        payload: Dict[str, Any],
        artifact: IvSurfaceArtifact,
    ) -> None:
        if self._cache_dir is None:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self._cache_path(variant, key), payload)
        except Exception as exc:
            raise ValidationError(
                "Calibration cache write failed "
                f"(variant={variant}, surface_date="
                f"{artifact.trade_date.isoformat()}, "
                f"surface_sha={artifact.sha256}): {exc}"
            ) from exc

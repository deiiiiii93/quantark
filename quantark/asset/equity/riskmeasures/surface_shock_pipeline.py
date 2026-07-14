"""Quote-level surface shock -> rebuild -> (re)calibrate -> reprice (WP4.5).

Shock layer (normative): shocks apply to CLEANED MARKET IV NODES — the
per-quote implied vols output by WP4.1 — not to raw bid/ask prices. A shock
adds dsigma to the targeted bucket, then the SVI fit re-runs from the
shocked IVs with the no-arb re-check.

FROZEN vs RECALIBRATE convention matrix (per model; DCN repriced through the
WP1.5 engine in every cell):

| model     | FROZEN                                  | RECALIBRATE            |
|-----------|------------------------------------------|------------------------|
| local_vol | keep the base Dupire local-vol function  | refit SVI -> re-run    |
|           | (vol shock transmits ~0 by construction; | Dupire -> reprice      |
|           | the note records this — it IS the point) |                        |
| heston    | keep HestonParams (PV unchanged; 0 with  | recalibrate on shocked |
|           | the explanation note)                    | cleaned quotes ->      |
|           |                                          | reprice                |

The market-vega headline is the RECALIBRATE mode; FROZEN rows attribute how
much of the shock the model transmits without recalibration (problem §7.5).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

from quantark.util.exceptions import NumericalError, ValidationError

SHOCK_LAYER = "cleaned_market_iv_nodes"
MAX_CALIBRATION_RMSE_IV = 0.02  # 2 vol pts: reject unusable Heston fits


class SurfaceShockMode(Enum):
    FROZEN = "frozen"
    RECALIBRATE = "recalibrate"


@dataclass(frozen=True)
class SurfaceShockResult:
    shock: dict
    mode: str
    model: str
    pv_base: float
    pv_shocked: float
    pnl: float
    no_arb_passed: bool
    notes: Tuple[str, ...]
    calibration: Optional[dict] = None   # heston: base/shocked convergence
    artifact_diagnostics: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "shock": dict(self.shock),
            "mode": self.mode,
            "model": self.model,
            "pv_base": self.pv_base,
            "pv_shocked": self.pv_shocked,
            "pnl": self.pnl,
            "no_arb_passed": self.no_arb_passed,
            "notes": list(self.notes),
            "calibration": (
                dict(self.calibration) if self.calibration else None
            ),
            "artifact_diagnostics": (
                dict(self.artifact_diagnostics)
                if self.artifact_diagnostics else None
            ),
        }


def _in_bucket(value: float, bucket: Optional[Tuple[float, float]]) -> bool:
    return bucket is None or (bucket[0] <= value <= bucket[1])


def shock_cleaned_ivs(
    cleaned,
    dsigma: float,
    tenor_bucket: Optional[Tuple[float, float]] = None,
    moneyness_bucket: Optional[Tuple[float, float]] = None,
):
    """New CleanedQuoteSet with +dsigma on the targeted cleaned IVs only."""
    new_slices = {}
    for expiry_t, quotes in cleaned.slices.items():
        hit_slice = _in_bucket(expiry_t, tenor_bucket)
        new_slices[expiry_t] = tuple(
            dataclasses.replace(q, iv=q.iv + float(dsigma))
            if hit_slice and _in_bucket(q.log_moneyness, moneyness_bucket)
            else q
            for q in quotes
        )
    return dataclasses.replace(cleaned, slices=new_slices)


def _fit_surface(cleaned, carry_curve, spot):
    from quantark.param.vol.svi import SVIVolSurface
    from quantark.volmodels.diagnostics import static_no_arb_report

    surface = SVIVolSurface.fit_from_quotes(cleaned, carry_curve, spot)
    report = static_no_arb_report(surface, sorted(cleaned.slices))
    return surface, report.passed


def _checked_calibration(
    cleaned,
    rate_curve,
    carry_curve,
    label: str,
    max_rmse_iv: float,
    calibration_config: Optional[dict] = None,
):
    """Calibrate and fail CLOSED on non-convergence or poor fit (§7.5:
    silently pricing unconverged parameters is worse than failing)."""
    from quantark.volmodels.heston import calibrate_heston_from_quotes

    calib = calibrate_heston_from_quotes(
        cleaned,
        rate_curve,
        carry_curve,
        config=dict(calibration_config or {}),
    )
    if not calib.result.success:
        raise NumericalError(
            f"{label} Heston calibration did not converge: "
            f"{calib.result.message}"
        )
    if calib.residual_report.rmse_iv > max_rmse_iv:
        raise NumericalError(
            f"{label} Heston calibration residual rmse_iv="
            f"{calib.residual_report.rmse_iv:.4f} exceeds the "
            f"{max_rmse_iv:.4f} gate"
        )
    return calib


def run_surface_shock_pipeline(
    product,
    base_env_builder: Callable[[object], object],  # surface -> PricingEnvironment
    cleaned,
    spot: float,
    rate_curve,
    carry_curve,
    model: str,                                    # "local_vol" | "heston"
    mode: SurfaceShockMode,
    engine_factory: Callable[[str, object], object],
    dsigma: float = 0.01,
    tenor_bucket: Optional[Tuple[float, float]] = None,
    moneyness_bucket: Optional[Tuple[float, float]] = None,
    max_calibration_rmse_iv: float = MAX_CALIBRATION_RMSE_IV,
    heston_calibration_config: Optional[dict] = None,
) -> SurfaceShockResult:
    """One auditable shock cell: base PV, shocked PV, PnL, no-arb, notes."""
    from quantark.volmodels.localvol import build_dupire_local_vol

    if model not in ("local_vol", "heston"):
        raise ValidationError(f"model must be local_vol or heston, got {model}")
    if not isinstance(mode, SurfaceShockMode):
        raise ValidationError(f"mode must be SurfaceShockMode, got {mode!r}")
    if heston_calibration_config is not None and not isinstance(
        heston_calibration_config, dict
    ):
        raise ValidationError("heston_calibration_config must be a dict or None")
    notes = []
    artifact_diagnostics = None
    shock = {
        "layer": SHOCK_LAYER,
        "dsigma": float(dsigma),
        "tenor_bucket": list(tenor_bucket) if tenor_bucket else None,
        "moneyness_bucket": (
            list(moneyness_bucket) if moneyness_bucket else None
        ),
    }

    base_surface, base_ok = _fit_surface(cleaned, carry_curve, spot)
    if not base_ok:
        raise ValidationError("base surface fails the no-arb diagnostics")
    shocked = shock_cleaned_ivs(cleaned, dsigma, tenor_bucket, moneyness_bucket)
    shocked_surface, no_arb_passed = _fit_surface(shocked, carry_curve, spot)

    if model == "local_vol":
        base_env = base_env_builder(base_surface)
        base_lv = build_dupire_local_vol(
            base_surface, spot=spot, rate_curve=rate_curve,
            div_yield=base_env.get_div_yield,
        )
        base_engine = engine_factory(model, base_lv)
        pv_base = float(base_engine.price(product, base_env))
        if mode is SurfaceShockMode.FROZEN:
            # rebuild the surface but KEEP the base local-vol function
            shocked_env = base_env_builder(shocked_surface)
            engine = engine_factory(model, base_lv)
            notes.append(
                "frozen Dupire local-vol function: the vol shock transmits "
                "only through non-vol inputs (~0 by construction — this is "
                "the attribution point of the FROZEN row)"
            )
        else:
            shocked_env = base_env_builder(shocked_surface)
            shocked_lv = build_dupire_local_vol(
                shocked_surface, spot=spot, rate_curve=rate_curve,
                div_yield=shocked_env.get_div_yield,
            )
            engine = engine_factory(model, shocked_lv)
        artifact_diagnostics = {
            "base_local_vol": {
                "min": float(base_lv.lv_grid.min()),
                "max": float(base_lv.lv_grid.max()),
                "n_times": int(base_lv.time_grid.size),
                "n_strikes": int(base_lv.strike_grid.size),
                "time_range": [
                    float(base_lv.time_grid[0]),
                    float(base_lv.time_grid[-1]),
                ],
                "strike_range": [
                    float(base_lv.strike_grid[0]),
                    float(base_lv.strike_grid[-1]),
                ],
            },
            "shocked_local_vol": {
                "min": float(
                    (base_lv if mode is SurfaceShockMode.FROZEN else shocked_lv)
                    .lv_grid.min()
                ),
                "max": float(
                    (base_lv if mode is SurfaceShockMode.FROZEN else shocked_lv)
                    .lv_grid.max()
                ),
            },
        }
        pv_shocked = float(engine.price(product, shocked_env))
    else:  # heston
        calibration_info = {}
        base_calib = _checked_calibration(
            cleaned,
            rate_curve,
            carry_curve,
            "base",
            max_calibration_rmse_iv,
            heston_calibration_config,
        )
        calibration_info["base"] = {
            "success": base_calib.result.success,
            "cost": base_calib.result.cost,
            "nfev": base_calib.result.nfev,
            "rmse_iv": base_calib.residual_report.rmse_iv,
            "feller_policy": dict(base_calib.config["feller_policy"]),
        }
        base_env = base_env_builder(base_surface)
        base_engine = engine_factory(model, base_calib.params)
        pv_base = float(base_engine.price(product, base_env))
        if mode is SurfaceShockMode.FROZEN:
            pv_shocked = pv_base
            notes.append(
                "frozen params: shock not transmitted (the model-implied "
                "vanilla surface is unchanged, so the DCN PV is unchanged; "
                "the model absorbs quote shocks only via recalibration)"
            )
        else:
            shocked_calib = _checked_calibration(
                shocked, rate_curve, carry_curve, "shocked",
                max_calibration_rmse_iv,
                heston_calibration_config,
            )
            calibration_info["shocked"] = {
                "success": shocked_calib.result.success,
                "cost": shocked_calib.result.cost,
                "nfev": shocked_calib.result.nfev,
                "rmse_iv": shocked_calib.residual_report.rmse_iv,
                "feller_policy": dict(
                    shocked_calib.config["feller_policy"]
                ),
            }
            engine = engine_factory(model, shocked_calib.params)
            pv_shocked = float(
                engine.price(product, base_env_builder(shocked_surface))
            )

    return SurfaceShockResult(
        shock=shock,
        mode=mode.value,
        model=model,
        pv_base=pv_base,
        pv_shocked=pv_shocked,
        pnl=pv_shocked - pv_base,
        no_arb_passed=bool(no_arb_passed),
        notes=tuple(notes),
        calibration=(calibration_info if model == "heston" else None),
        artifact_diagnostics=artifact_diagnostics,
    )

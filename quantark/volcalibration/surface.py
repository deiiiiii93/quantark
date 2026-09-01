"""IV-surface construction: rectangular grid assembly and SABR smoothing.

The dict this module produces and consumes is the stage-02 surface schema the
artifact writer already serializes.  It is kept verbatim because artifact bytes
must not change (spec 5.3).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np

from quantark.param.vol.sabr.calibration import calibrate_sabr_slice
from quantark.param.vol.sabr.hagan import sabr_implied_vol_black
from quantark.util.exceptions import ValidationError
from quantark.volcalibration.quotes import QuoteSet

MIN_COMMON_STRIKES = 5
DEFAULT_SABR_BETA = 1.0
DEFAULT_MONEYNESS_WIDTH = 0.35
EXTRAPOLATION_POLICY = "flat_total_variance"
_STRIKE_MATCH_ATOL = 1e-6


def build_raw_surface(quotes: QuoteSet) -> Dict[str, Any]:
    """Assemble a rectangular strike x maturity IV grid from a QuoteSet.

    The common strike grid holds strikes quoted in at least two expiries; each
    expiry's row is linear interpolation of its own smile, flat past its wings.
    """
    expiries = list(quotes.expiries)
    if len(expiries) < 2:
        raise ValidationError("a surface needs at least 2 expiries")

    all_strikes = sorted({n.strike for e in expiries for n in e.nodes})

    def _count(k: float) -> int:
        return sum(
            any(abs(k - n.strike) < _STRIKE_MATCH_ATOL for n in e.nodes)
            for e in expiries
        )

    strikes = [k for k in all_strikes if _count(k) >= 2]
    if len(strikes) < MIN_COMMON_STRIKES:
        raise ValidationError(
            f"only {len(strikes)} common strikes across expiries "
            f"(need >= {MIN_COMMON_STRIKES})"
        )

    maturities = [e.T for e in expiries]
    grid = np.empty((len(maturities), len(strikes)), dtype=float)
    per_expiry: List[Dict[str, Any]] = []
    for i, e in enumerate(expiries):
        ks = np.asarray([n.strike for n in e.nodes], dtype=float)
        vs = np.asarray([n.iv for n in e.nodes], dtype=float)
        grid[i] = np.interp(strikes, ks, vs)
        rmse = e.diagnostics.get("parity_rmse_over_forward")
        per_expiry.append(
            {
                "expiry_date": e.expiry_date if e.expiry_date else e.expiry_label,
                "T": e.T,
                "r": e.r,
                "q": e.q,
                "forward": e.forward,
                "df": e.discount_factor,
                "pair_count": int(e.diagnostics.get("n_pairs", 0)),
                "parity_rmse_points": (
                    float(rmse) * e.forward if rmse is not None else None
                ),
                "points": [(n.strike, n.iv) for n in e.nodes],
            }
        )

    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValidationError("grid assembly produced non-positive or non-finite IVs")

    return {
        "s0": quotes.spot,
        "strikes": strikes,
        "maturities": maturities,
        "iv_grid": grid.tolist(),
        "per_expiry": per_expiry,
    }


def sabr_smoothed_surface(
    surface_json: dict,
    *,
    beta: float = 1.0,
    moneyness_width: float = 0.35,
    grid_size: int = 25,
    min_calendar_slope: float = 1e-8,
) -> dict:
    """Return a SABR-smoothed, calendar-projected copy of a stage-02 IV surface.

    Stage 02 deliberately records the raw call-equivalent IVs. Dupire, however,
    differentiates total variance twice in strike and once in maturity, so it needs a
    smooth static-arbitrage-controlled target. This helper fits one Hagan SABR slice per
    expiry, evaluates those slices on the rectangular stage-02 strike grid, then projects
    total variance to be non-decreasing in maturity at each strike.

    The returned JSON keeps ``raw_points`` beside each expiry's model ``points`` so later
    stages can fit all models to the same smooth target without hiding the raw-market
    discrepancy.
    """

    if not 0.0 <= beta <= 1.0:
        raise ValidationError(f"beta must be in [0, 1], got {beta}")
    if moneyness_width <= 0.0:
        raise ValidationError("moneyness_width must be positive")
    if grid_size < 5:
        raise ValidationError("grid_size must be at least 5")
    if min_calendar_slope < 0.0:
        raise ValidationError("min_calendar_slope must be non-negative")

    out = copy.deepcopy(surface_json)
    strikes = np.asarray(out["strikes"], dtype=float)
    maturities = np.asarray(out["maturities"], dtype=float)
    raw_grid = np.asarray(out["iv_grid"], dtype=float)
    per_expiry = out["per_expiry"]

    pe_by_t = {round(float(p["T"]), 12): p for p in per_expiry}
    sabr_rows = []
    slice_meta = []
    for t in maturities:
        pe = pe_by_t.get(round(float(t), 12))
        if pe is None:
            raise ValidationError(f"missing per_expiry slice for maturity {t}")
        raw_points = [(float(k), float(v)) for k, v in pe["points"]]
        ks = np.asarray([k for k, _ in raw_points], dtype=float)
        vols = np.asarray([v for _, v in raw_points], dtype=float)
        fwd = float(pe["forward"])
        weights = np.exp(-0.5 * (np.log(ks / fwd) / moneyness_width) ** 2)
        alpha_bounds = (1e-4, 5.0) if beta > 0.9 else (1e-4, 100.0)
        params = calibrate_sabr_slice(
            F=fwd,
            strikes=ks,
            T=float(t),
            market_vols=vols,
            beta=beta,
            weights=weights,
            alpha_bounds=alpha_bounds,
            grid_size=grid_size,
            refine=True,
        )
        row = sabr_implied_vol_black(
            fwd,
            strikes,
            np.full_like(strikes, float(t), dtype=float),
            params["alpha"],
            params["beta"],
            params["rho"],
            params["nu"],
            shift=params["shift"],
        )
        row = np.asarray(row, dtype=float)
        if not np.all(np.isfinite(row)) or np.any(row <= 0.0):
            raise ValidationError(f"SABR smoothing produced invalid vols at T={t}")
        sabr_rows.append(row)
        slice_meta.append({
            "T": float(t),
            "alpha": float(params["alpha"]),
            "beta": float(params["beta"]),
            "rho": float(params["rho"]),
            "nu": float(params["nu"]),
            "shift": float(params["shift"]),
            "mse": float(params["mse"]),
        })

    smooth_grid = np.asarray(sabr_rows, dtype=float)
    total_var = smooth_grid * smooth_grid * maturities[:, None]
    adjusted_nodes = 0
    for i in range(1, total_var.shape[0]):
        floor = total_var[i - 1] + min_calendar_slope * (maturities[i] - maturities[i - 1])
        before = total_var[i].copy()
        total_var[i] = np.maximum(total_var[i], floor)
        adjusted_nodes += int(np.count_nonzero(total_var[i] > before))
    smooth_grid = np.sqrt(total_var / maturities[:, None])

    raw_point_sq = []
    for i, pe in enumerate(per_expiry):
        raw_points = [(float(k), float(v)) for k, v in pe["points"]]
        smoothed_points = []
        for k, raw_v in raw_points:
            smooth_v = float(np.interp(k, strikes, smooth_grid[i]))
            smoothed_points.append((float(k), smooth_v))
            raw_point_sq.append((smooth_v - raw_v) ** 2)
        pe["raw_points"] = raw_points
        pe["points"] = smoothed_points
        pe["sabr_params"] = slice_meta[i]

    out["iv_grid"] = smooth_grid.tolist()
    out["target_smoothing"] = {
        "method": "sabr_calendar_projected",
        "beta": float(beta),
        "moneyness_width": float(moneyness_width),
        "grid_size": int(grid_size),
        "min_calendar_slope": float(min_calendar_slope),
        "calendar_adjusted_nodes": int(adjusted_nodes),
        "raw_grid_rmse_iv": float(np.sqrt(np.mean((smooth_grid - raw_grid) ** 2))),
        "raw_points_rmse_iv": float(np.sqrt(np.mean(raw_point_sq))) if raw_point_sq else 0.0,
        "slice_mean_mse": float(np.mean([m["mse"] for m in slice_meta])) if slice_meta else 0.0,
        "slices": slice_meta,
    }
    return out


def prepare_model_surface(
    surface_json: dict,
    *,
    iv_smoothing: str = "sabr",
    sabr_beta: float = 1.0,
) -> dict:
    """Prepare the IV target used by model-calibration stages."""
    if iv_smoothing in ("none", "raw"):
        out = copy.deepcopy(surface_json)
        out["target_smoothing"] = {"method": "none"}
        return out
    if iv_smoothing == "sabr":
        return sabr_smoothed_surface(surface_json, beta=sabr_beta)
    raise ValidationError("iv_smoothing must be one of: sabr, none")


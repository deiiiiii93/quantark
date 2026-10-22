"""Shared helpers for the MO vol-model suite.

DEPRECATED: the calibration procedure now lives in ``quantark.volcalibration``.
This module is a thin compatibility shim preserving the suite's legacy
signatures; new code should import from the library directly.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize import listed as _listed
from quantark.volcalibration.normalize.listed import (  # noqa: F401
    OtmQuote,
    otm_implied_vol,
    select_otm,
)
from quantark.volcalibration.snapshot import QuoteSnapshot

_REQUIRED = {"fetched_at", "underlying", "expiries"}
_LEGACY_TRADE_DATE = date(2026, 7, 6)  # the suite's fixed valuation date


def load_snapshot(path) -> dict:
    """Load and lightly validate an MO snapshot JSON (legacy: returns the dict)."""
    snap = json.loads(Path(path).read_text())
    missing = _REQUIRED - set(snap)
    if missing:
        raise ValueError(f"snapshot missing keys: {sorted(missing)}")
    if "spot" not in snap.get("underlying", {}):
        raise ValueError("snapshot underlying missing 'spot'")
    return snap


@dataclass
class ExpirySlice:
    """Legacy five-field slice; delegates the maths to the library."""

    expiry_date: str
    T: float
    calls: Dict[float, float]
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)

    def to_library(self) -> "_listed.ExpirySlice":
        return _listed.ExpirySlice(
            expiry_label=self.expiry_date,
            expiry_date=self.expiry_date,
            T=self.T,
            calls=self.calls,
            puts=self.puts,
            volume=self.volume,
        )


@dataclass
class ParityResult:
    """Legacy five-field parity result."""

    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int


def iter_expiries(snapshot: dict) -> List[ExpirySlice]:
    """Reshape a legacy snapshot dict into legacy slices via the library."""
    lifted = QuoteSnapshot.from_legacy_live(
        snapshot, trade_date=_LEGACY_TRADE_DATE, symbol="000852.SH"
    )
    return [
        ExpirySlice(
            expiry_date=sl.expiry_date,
            T=sl.T,
            calls=sl.calls,
            puts=sl.puts,
            volume=sl.volume,
        )
        for sl in _listed.iter_expiries(lifted)
    ]


def imply_forward_and_rate(sl: ExpirySlice, s0: float) -> ParityResult:
    """Recover (r, F, DF, q) for one expiry (legacy wrapper; raises ValueError).

    The library raises ``ValidationError``, which derives from
    ``QuantArkException`` and NOT from ``ValueError``.  The suite's tests assert
    ``pytest.raises(ValueError)``, so this re-raise is load-bearing -- do not
    remove it.
    """
    try:
        res = _listed.imply_forward_and_rate(sl.to_library(), s0)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return ParityResult(
        r=res.r,
        forward=res.forward,
        discount_factor=res.discount_factor,
        q=res.q,
        n_pairs=res.n_pairs,
    )


def build_env(surface_json: dict):
    """Reconstruct (PricingEnvironment, GridVolSurface, s0) from stage-02 output.

    The rate curve and dividend/carry curve are built from the per-expiry parity
    pillars, so the pricing environment carries the exact term structure the surface
    was calibrated against. Both curve types require >= 2 pillars (guaranteed by the
    stage-02 >=2-expiry check).
    """
    from datetime import datetime

    from quantark.param import GridVolSurface, SpotQuote
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.param.div import TermStructureDividendYield
    from quantark.priceenv import PricingEnvironment

    s0 = float(surface_json["s0"])
    surf = GridVolSurface(surface_json["strikes"], surface_json["maturities"],
                          np.array(surface_json["iv_grid"]))
    pe = surface_json["per_expiry"]
    ts = [p["T"] for p in pe]
    rate = LinearRateCurve([(p["T"], p["r"]) for p in pe])
    div = TermStructureDividendYield(times=ts, yields=[p["q"] for p in pe])
    env = PricingEnvironment(rate_curve=rate, valuation_date=datetime(2026, 7, 6),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=div)
    return env, surf, s0


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
    from quantark.param.vol.sabr.calibration import calibrate_sabr_slice
    from quantark.param.vol.sabr.hagan import sabr_implied_vol_black

    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if moneyness_width <= 0.0:
        raise ValueError("moneyness_width must be positive")
    if grid_size < 5:
        raise ValueError("grid_size must be at least 5")
    if min_calendar_slope < 0.0:
        raise ValueError("min_calendar_slope must be non-negative")

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
            raise ValueError(f"missing per_expiry slice for maturity {t}")
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
            raise ValueError(f"SABR smoothing produced invalid vols at T={t}")
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
    raise ValueError("iv_smoothing must be one of: sabr, none")


def calibrate_leverage_for(env, hp, lv, t_max, n_steps=40, n_x=161, n_z=81):
    """Calibrate an SLV leverage surface up to t_max (forward Fokker-Planck), as in stage 05."""
    from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig
    from quantark.volmodels.curves import forward_carry_on_grid
    s0 = float(env.spot)
    t_grid = np.linspace(0.0, float(t_max), n_steps + 1)
    r_fwd = np.array([env.rate_curve.get_forward_rate(t_grid[i], t_grid[i + 1]) for i in range(n_steps)])
    carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
    return calibrate_leverage_surface(s0, hp, lv, np.diff(t_grid), r_fwd, carry_fwd,
                                      eta=1.0, fp_config=FpCalibrationConfig(n_x=n_x, n_z=n_z))


def plot_smiles(rows, path, title="MO implied-vol smiles"):
    """rows = list of (label, strikes, ivs). Saves a PNG; returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, strikes, ivs in rows:
        ax.plot(strikes, np.array(ivs) * 100, marker="o", ms=3, label=label)
    ax.set_xlabel("strike")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

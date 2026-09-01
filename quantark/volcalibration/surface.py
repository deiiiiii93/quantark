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
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volcalibration.admission import (
    AdmissionError,
    AdmissionReason,
    validate_static_arbitrage,
)
from quantark.volcalibration.quotes import QuoteSet
from quantark.volcalibration.snapshot import QuoteSnapshot

# The parity gates are the normalizer's, restated in the artifact's admission
# block so a stored surface records the thresholds it was admitted under.
MAX_ABS_PARITY_IMPLIED_RATE = 0.10
MAX_PARITY_RMSE_FORWARD_RATIO = 0.01

MIN_EXPIRIES = 2
MIN_STRIKES_PER_EXPIRY = 5
MIN_COMMON_STRIKES = 3  # the butterfly check needs >= 3 grid strikes
DEFAULT_SABR_BETA = 1.0
DEFAULT_MONEYNESS_WIDTH = 0.35
EXTRAPOLATION_POLICY = "flat_total_variance"
ARTIFACT_SCHEMA_VERSION = 1
_STRIKE_MATCH_ATOL = 1e-6

# How the shared strike grid is built.  This is not a free choice: it follows
# from whether the book's expiries share a strike ladder at all.  A listed
# ladder does, so its grid is the strikes the market actually quoted.  A
# delta-quoted book does not -- every tenor's 25-delta strike is its own -- so
# there is no shared observed strike to build a grid from, and the grid has to
# be laid down over the interval all tenors cover.  The normalizer declares
# which rule its book obeys in ``QuoteSet.universe``; nothing is inferred here,
# because a silent switch would hide an empty intersection as a design choice.
STRIKE_GRID_SHARED_OBSERVED = "shared_observed_within_quoted_range_overlap"
STRIKE_GRID_UNIFORM_OVER_OVERLAP = "uniform_over_quoted_range_overlap"
STRIKE_GRID_RULES = (STRIKE_GRID_SHARED_OBSERVED, STRIKE_GRID_UNIFORM_OVER_OVERLAP)
DEFAULT_UNIFORM_GRID_SIZE = 31

_STRIKE_GRID_LABELS = {
    STRIKE_GRID_SHARED_OBSERVED: "shared_by_>=2_expiries_within_quoted_range_overlap",
    STRIKE_GRID_UNIFORM_OVER_OVERLAP: (
        "uniform_within_quoted_range_overlap_of_all_expiries"
    ),
}


def build_raw_surface(quotes: QuoteSet) -> Dict[str, Any]:
    """Assemble a rectangular strike x maturity IV grid from a QuoteSet.

    The surface domain is the OVERLAP of the per-expiry quoted strike ranges,
    not their union.  Listed ladders differ by contract month, so a union-range
    grid would force SABR wing evaluation where the market never quoted, and a
    single expiry's deep-wing settlement ticks (tick-floor IV inflation) would
    pull its vol-of-vol and distort the whole slice.  Each expiry's nodes are
    trimmed to that shared domain; off-grid wing nodes are counted for audit,
    never silently dropped.  Dropping an expiry can widen the overlap, so the
    trim iterates to its fixed point.

    Under ``STRIKE_GRID_UNIFORM_OVER_OVERLAP`` the overlap is still the domain,
    but nodes are neither trimmed nor dropped: a five-pillar delta-quoted slice
    keeps all five, and the grid is laid uniformly across the interval every
    expiry covers.  ``sabr_smoothed_surface`` then fits each slice to its own
    observed nodes and evaluates it on that grid, so the artifact's
    ``raw_points`` remain the observed quotes and only ``points`` are model
    values.
    """
    per_expiry: List[Dict[str, Any]] = []
    for e in quotes.expiries:
        rmse_points = e.diagnostics.get("parity_rmse_points")
        if rmse_points is None:
            ratio = e.diagnostics.get("parity_rmse_over_forward")
            rmse_points = float(ratio) * e.forward if ratio is not None else None
        per_expiry.append(
            {
                "expiry_date": e.expiry_date if e.expiry_date else e.expiry_label,
                "T": e.T,
                "r": e.r,
                "q": e.q,
                "forward": e.forward,
                "df": e.discount_factor,
                "pair_count": int(e.diagnostics.get("n_pairs", 0)),
                "parity_rmse_points": rmse_points,
                "points": [(n.strike, n.iv) for n in e.nodes],
            }
        )
    if len(per_expiry) < MIN_EXPIRIES:
        raise AdmissionError(
            AdmissionReason.INSUFFICIENT_EXPIRIES,
            f"{len(per_expiry)} expiries (need >= {MIN_EXPIRIES})",
        )
    for i in range(len(per_expiry) - 1):
        if per_expiry[i + 1]["T"] <= per_expiry[i]["T"]:
            raise AdmissionError(
                AdmissionReason.DUPLICATE_MATURITY,
                f"{per_expiry[i]['expiry_date']} and "
                f"{per_expiry[i + 1]['expiry_date']} share T={per_expiry[i]['T']}",
            )

    universe_in = dict(quotes.universe or {})
    grid_rule = str(universe_in.get("strike_grid_rule", STRIKE_GRID_SHARED_OBSERVED))
    if grid_rule not in STRIKE_GRID_RULES:
        raise ValidationError(
            f"unknown strike_grid_rule {grid_rule!r}; known rules are "
            f"{list(STRIKE_GRID_RULES)}"
        )

    if grid_rule == STRIKE_GRID_UNIFORM_OVER_OVERLAP:
        return _uniform_grid_surface(quotes, per_expiry, universe_in)

    dropped_trimmed: List[Dict[str, Any]] = []
    grid_lo = grid_hi = 0.0
    for _ in range(len(per_expiry)):
        grid_lo = max(min(k for k, _ in pe["points"]) for pe in per_expiry)
        grid_hi = min(max(k for k, _ in pe["points"]) for pe in per_expiry)
        kept = []
        dropped_trim = []
        for pe in per_expiry:
            in_grid = [
                (k, v)
                for k, v in pe["points"]
                if grid_lo - 1e-9 <= k <= grid_hi + 1e-9
            ]
            if len(in_grid) >= MIN_STRIKES_PER_EXPIRY:
                pe["off_grid_node_count"] = len(pe["points"]) - len(in_grid)
                pe["points"] = in_grid
                kept.append(pe)
            else:
                dropped_trim.append(pe["expiry_date"])
                dropped_trimmed.append(
                    {
                        "expiry_date": pe["expiry_date"],
                        "reason": "fewer_than_min_strikes_inside_quoted_range_overlap",
                        "node_count": len(pe["points"]),
                        "in_domain_node_count": len(in_grid),
                        "min_strikes_per_expiry": MIN_STRIKES_PER_EXPIRY,
                        "quoted_range_overlap": [grid_lo, grid_hi],
                    }
                )
        if not dropped_trim:
            break
        per_expiry = kept
        if len(per_expiry) < MIN_EXPIRIES:
            raise AdmissionError(
                AdmissionReason.INSUFFICIENT_EXPIRIES,
                f"< {MIN_EXPIRIES} expiries with >= {MIN_STRIKES_PER_EXPIRY} "
                f"nodes inside the quoted-range overlap; dropped={dropped_trim}",
            )

    all_strikes = sorted({k for pe in per_expiry for k, _ in pe["points"]})

    def _count(k: float) -> int:
        return sum(
            any(abs(k - kk) < _STRIKE_MATCH_ATOL for kk, _ in pe["points"])
            for pe in per_expiry
        )

    strikes = [k for k in all_strikes if _count(k) >= 2 and grid_lo <= k <= grid_hi]
    if len(strikes) < MIN_COMMON_STRIKES:
        raise AdmissionError(
            AdmissionReason.INSUFFICIENT_COMMON_STRIKES,
            f"{len(strikes)} shared strikes inside quoted-range overlap "
            f"[{grid_lo}, {grid_hi}] (< {MIN_COMMON_STRIKES})",
        )
    maturities = [pe["T"] for pe in per_expiry]
    grid = np.empty((len(maturities), len(strikes)))
    for i, pe in enumerate(per_expiry):
        ks = np.array([k for k, _ in pe["points"]])
        vs = np.array([v for _, v in pe["points"]])
        grid[i] = np.interp(strikes, ks, vs)  # flat past this expiry's wings

    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValidationError("grid assembly produced non-positive or non-finite IVs")

    universe = dict(quotes.universe or {})
    return {
        "s0": quotes.spot,
        "strikes": strikes,
        "maturities": maturities,
        "iv_grid": grid.tolist(),
        "per_expiry": per_expiry,
        "node_universe": {
            "node_count": int(universe.get("node_count", 0)),
            "expiry_count": int(universe.get("expiry_count", 0)),
            "filtered_quote_counts": universe.get("filtered_quote_counts", {}),
            "excluded_expiries": list(universe.get("excluded_expiries", []))
            + dropped_trimmed,
        },
    }


def _uniform_grid_surface(
    quotes: QuoteSet,
    per_expiry: List[Dict[str, Any]],
    universe: Dict[str, Any],
) -> Dict[str, Any]:
    """Grid assembly for books whose expiries share no observed strike.

    Every expiry keeps all its nodes; the grid spans the interval all expiries
    cover, so no slice is evaluated outside its own quoted range.
    """
    grid_size = int(universe.get("strike_grid_size", DEFAULT_UNIFORM_GRID_SIZE))
    if grid_size < MIN_COMMON_STRIKES:
        raise ValidationError(
            f"strike_grid_size {grid_size} is below min_common_strikes "
            f"{MIN_COMMON_STRIKES}"
        )
    for pe in per_expiry:
        if len(pe["points"]) < MIN_STRIKES_PER_EXPIRY:
            raise AdmissionError(
                AdmissionReason.INSUFFICIENT_EXPIRIES,
                f"expiry {pe['expiry_date']} has {len(pe['points'])} nodes "
                f"(need >= {MIN_STRIKES_PER_EXPIRY})",
            )
        pe["off_grid_node_count"] = 0

    grid_lo = max(min(k for k, _ in pe["points"]) for pe in per_expiry)
    grid_hi = min(max(k for k, _ in pe["points"]) for pe in per_expiry)
    if not grid_hi > grid_lo:
        raise AdmissionError(
            AdmissionReason.INSUFFICIENT_COMMON_STRIKES,
            f"quoted strike ranges do not overlap: [{grid_lo}, {grid_hi}]",
        )
    strikes = np.linspace(grid_lo, grid_hi, grid_size).tolist()

    maturities = [pe["T"] for pe in per_expiry]
    grid = np.empty((len(maturities), len(strikes)))
    for i, pe in enumerate(per_expiry):
        ks = np.array([k for k, _ in pe["points"]])
        vs = np.array([v for _, v in pe["points"]])
        # Placeholder values only: sabr_smoothed_surface refits each slice to
        # pe["points"] and overwrites this row.
        grid[i] = np.interp(strikes, ks, vs)

    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValidationError("grid assembly produced non-positive or non-finite IVs")

    return {
        "s0": quotes.spot,
        "strikes": strikes,
        "maturities": maturities,
        "iv_grid": grid.tolist(),
        "per_expiry": per_expiry,
        "node_universe": {
            "node_count": int(universe.get("node_count", 0)),
            "expiry_count": int(universe.get("expiry_count", 0)),
            "filtered_quote_counts": universe.get("filtered_quote_counts", {}),
            "excluded_expiries": list(universe.get("excluded_expiries", [])),
            # Every grid value is a model value; the observed quotes survive as
            # per_expiry raw_points.  Saying so keeps a reader from mistaking
            # grid width for liquidity.
            "grid_values_are_model_values": True,
            "quoted_range_overlap": [grid_lo, grid_hi],
        },
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


def build_artifact(
    quotes: QuoteSet,
    snapshot: QuoteSnapshot,
    *,
    sabr_beta: float = DEFAULT_SABR_BETA,
) -> Dict[str, Any]:
    """QuoteSet -> complete, admission-checked IV-surface artifact dict.

    The key set and field names are those the existing artifacts already use:
    the artifact's sha256 feeds the calibration cache key, so changing them
    invalidates every warm cache entry and cohort pin (spec 5.3).
    """
    raw = build_raw_surface(quotes)
    try:
        smoothed = sabr_smoothed_surface(raw, beta=sabr_beta)
    except (ValidationError, NumericalError) as exc:
        raise AdmissionError(
            AdmissionReason.SABR_SMOOTHING_FAILED, f"{type(exc).__name__}: {exc}"
        ) from exc

    atm_pillars = []
    for pe in smoothed["per_expiry"]:
        params = pe["sabr_params"]
        atm_vol = float(
            sabr_implied_vol_black(
                float(pe["forward"]),
                [float(pe["forward"])],
                [float(pe["T"])],
                params["alpha"],
                params["beta"],
                params["rho"],
                params["nu"],
                shift=params["shift"],
            )[0]
        )
        if not (np.isfinite(atm_vol) and atm_vol > 0.0):
            raise AdmissionError(
                AdmissionReason.INVALID_ATM_PILLAR,
                f"expiry {pe['expiry_date']}: SABR ATM vol {atm_vol}",
            )
        atm_pillars.append(
            {
                "T": float(pe["T"]),
                "expiry_date": pe["expiry_date"],
                "atm_vol": atm_vol,
            }
        )

    try:
        validation_method = validate_static_arbitrage(smoothed)
    except (NumericalError, ValidationError) as exc:
        raise AdmissionError(
            AdmissionReason.STATIC_ARBITRAGE, f"{type(exc).__name__}: {exc}"
        ) from exc

    source = dict(snapshot.source or {})
    smoothed["schema_version"] = ARTIFACT_SCHEMA_VERSION
    smoothed["trade_date"] = snapshot.trade_date.isoformat()
    smoothed["source_class"] = source.get("vendor")
    smoothed["price_field"] = snapshot.price_field
    smoothed["source_url"] = source.get("source_url")
    smoothed["source_sha256"] = source.get("sha256")
    smoothed["atm_pillars"] = atm_pillars
    smoothed["extrapolation_policy"] = {
        "beyond_last_listed_expiry": EXTRAPOLATION_POLICY,
        "max_listed_T": max(float(t) for t in smoothed["maturities"]),
    }
    smoothed["admission"] = {
        "min_expiries": MIN_EXPIRIES,
        "min_strikes_per_expiry": MIN_STRIKES_PER_EXPIRY,
        "min_common_strikes": MIN_COMMON_STRIKES,
        "sabr_beta": float(sabr_beta),
        "parity_quality_gate": {
            "maximum_absolute_annualized_implied_rate": MAX_ABS_PARITY_IMPLIED_RATE,
            "maximum_rmse_divided_by_forward": MAX_PARITY_RMSE_FORWARD_RATIO,
        },
        "static_arbitrage_validation": validation_method,
        "strike_grid": _STRIKE_GRID_LABELS[
            str(
                (quotes.universe or {}).get(
                    "strike_grid_rule", STRIKE_GRID_SHARED_OBSERVED
                )
            )
        ],
        "sabr_fit_domain": "nodes_inside_strike_grid_only",
    }
    return smoothed


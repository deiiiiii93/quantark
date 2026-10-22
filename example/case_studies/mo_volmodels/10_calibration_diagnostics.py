"""Cross-date Heston diagnostics for official CFFEX MO settlements.

This stage is intentionally separate from the live Sina midpoint calibration.  It
consumes only ``mo_settlement_snapshot_YYYYMMDD.json`` artifacts produced by
``01_fetch_mo_settlement_history.py`` and fits the same normalized, raw-settlement
Heston objective on every admitted date.  No surface interpolation, extrapolation,
or quote smoothing is used.

The normalization is expiry-local: put-call parity supplies ``DF`` and ``F``;
strikes become ``K/F`` and call-equivalent prices become ``C/(DF*F)``.  Heston is
then calibrated with spot=forward=1 and zero rates.  This makes parameter vectors
comparable across dates without pretending that settlement prices are executable
midpoints.

Example::

    .venv/bin/python example/mo_volmodels/10_calibration_diagnostics.py \
      --tags 20260430 20260515 20260615 20260630 20260706 20260715 20260720
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _heston_diagnostics as hd  # noqa: E402

from quantark.util.exceptions import NumericalError  # noqa: E402
from quantark.volmodels.black_scholes import implied_vol_call  # noqa: E402
from quantark.volmodels.heston import (  # noqa: E402
    HestonParams,
    MarketOption,
    calibrate_heston,
    heston_call_prices_vectorized,
)


SOURCE_CLASS = "official_cffex_eod_settlement"
PRICE_FIELD = "settlement"
PARAMETER_NAMES = hd.PARAMETER_NAMES
HESTON_BOUNDS = (
    (1e-6, 1e-3, 1e-4, 1e-3, -0.95),
    (0.5, 3.0, 0.5, 0.7, 0.0),
)
REGULARIZE_FELLER = 0.05
START_COUNT = 3
MIN_CALENDAR_DAYS = 7
MAX_CALENDAR_DAYS = 365
MIN_EXPIRIES = 5
MIN_NODES = 80
BOUND_HIT_RELATIVE_TOLERANCE = 1e-6
MAX_ABS_PARITY_IMPLIED_RATE = 0.10
MAX_PARITY_RMSE_FORWARD_RATIO = 0.01
NEAR_ATM_PARITY_PAIR_COUNT = 9
STATIC_ARBITRAGE_ABSOLUTE_TOLERANCE = 1e-10
OPTIMIZER_XTOL = 1e-6
OPTIMIZER_FTOL = 1e-6
OPTIMIZER_GTOL = 1e-6
ENFORCE_FELLER = False
BEST_START_SELECTION_POLICY = "minimum_weighted_rmse_among_optimizer_success_true"
# CFFEX holiday roll required by the frozen 2026 cohort.  Keep this explicit so
# historical maturity is never recomputed from today's holiday calendar.
EXPIRY_DATE_OVERRIDES = {"2606": date(2026, 6, 22)}


class CoverageError(ValueError):
    """Raised when a date cannot support the frozen cross-date universe."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


def _parse_iso_date(value: Any, *, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field} {value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} must use canonical YYYY-MM-DD form")
    return parsed


def _third_friday(year_month: str) -> date:
    if not re.fullmatch(r"\d{4}", str(year_month)):
        raise ValueError(f"contract_month must be YYMM, got {year_month!r}")
    year = 2000 + int(year_month[:2])
    month = int(year_month[2:])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid contract_month {year_month!r}")
    cursor = date(year, month, 1)
    fridays: list[date] = []
    while cursor.month == month:
        if cursor.weekday() == 4:
            fridays.append(cursor)
        cursor += timedelta(days=1)
    return EXPIRY_DATE_OVERRIDES.get(year_month, fridays[2])


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _validate_snapshot_identity(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("source_class") != SOURCE_CLASS:
        raise ValueError(
            f"source_class must be {SOURCE_CLASS!r}; midpoint and settlement cohorts cannot mix"
        )
    if snapshot.get("price_field") != PRICE_FIELD:
        raise ValueError(f"price_field must be frozen as {PRICE_FIELD!r}")
    _parse_iso_date(snapshot.get("trade_date"), field="trade_date")
    digest = snapshot.get("source_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(snapshot.get("expiries"), list):
        raise ValueError("snapshot expiries must be a list")


def _near_atm_parity_sensitivity(
    strikes: np.ndarray,
    differences: np.ndarray,
    *,
    primary_forward: float,
    primary_rate: float,
    maturity: float,
) -> dict:
    """Refit parity on the nearest strikes without changing the primary OLS."""
    subset_size = min(NEAR_ATM_PARITY_PAIR_COUNT, strikes.size)
    indices = np.argsort(np.abs(strikes - primary_forward))[:subset_size]
    subset_strikes = strikes[indices]
    subset_differences = differences[indices]
    slope, intercept = np.polyfit(subset_strikes, subset_differences, 1)
    discount_factor = -float(slope)
    if not math.isfinite(discount_factor) or discount_factor <= 0.0:
        return {
            "method": "OLS_on_nearest_strikes_to_primary_forward",
            "subset_pair_count": int(subset_size),
            "status": "invalid_non_positive_discount_factor",
            "discount_factor": discount_factor,
        }
    forward = float(intercept / discount_factor)
    implied_rate = -math.log(discount_factor) / maturity
    residuals = subset_differences - (
        -discount_factor * subset_strikes + discount_factor * forward
    )
    return {
        "method": "OLS_on_nearest_strikes_to_primary_forward",
        "subset_pair_count": int(subset_size),
        "status": "measured",
        "discount_factor": discount_factor,
        "forward": forward,
        "implied_rate": implied_rate,
        "parity_rmse_points": float(np.sqrt(np.mean(np.square(residuals)))),
        "forward_relative_difference_vs_full_ols": forward / primary_forward - 1.0,
        "implied_rate_difference_vs_full_ols": implied_rate - primary_rate,
    }


def _static_arbitrage_diagnostics(expiry_nodes: Sequence[Mapping[str, Any]]) -> dict:
    """Test raw normalized call-equivalent nodes for strike monotonicity/convexity."""
    ordered = sorted(expiry_nodes, key=lambda node: node["normalized_strike"])
    strikes = np.asarray([node["normalized_strike"] for node in ordered], dtype=float)
    calls = np.asarray([node["normalized_call_price"] for node in ordered], dtype=float)
    monotonicity: list[dict] = []
    for index, change in enumerate(np.diff(calls)):
        if change > STATIC_ARBITRAGE_ABSOLUTE_TOLERANCE:
            monotonicity.append(
                {
                    "left_contract": ordered[index]["contract"],
                    "right_contract": ordered[index + 1]["contract"],
                    "left_normalized_strike": float(strikes[index]),
                    "right_normalized_strike": float(strikes[index + 1]),
                    "call_price_increase": float(change),
                }
            )
    strike_steps = np.diff(strikes)
    if np.any(strike_steps <= 0.0):
        raise ValueError("retained normalized strikes must be strictly increasing")
    slopes = np.diff(calls) / strike_steps
    convexity: list[dict] = []
    for index, slope_change in enumerate(np.diff(slopes)):
        if slope_change < -STATIC_ARBITRAGE_ABSOLUTE_TOLERANCE:
            convexity.append(
                {
                    "left_contract": ordered[index]["contract"],
                    "center_contract": ordered[index + 1]["contract"],
                    "right_contract": ordered[index + 2]["contract"],
                    "left_slope": float(slopes[index]),
                    "right_slope": float(slopes[index + 1]),
                    "slope_decrease": float(-slope_change),
                }
            )
    return {
        "method": "raw_normalized_call_monotonicity_and_irregular_grid_slope_convexity",
        "repair_applied": False,
        "absolute_tolerance": STATIC_ARBITRAGE_ABSOLUTE_TOLERANCE,
        "node_count": len(ordered),
        "non_increasing_call_violations": len(monotonicity),
        "convex_slope_violations": len(convexity),
        "monotonicity_details": monotonicity,
        "convexity_details": convexity,
    }


def build_calibration_nodes(
    snapshot: Mapping[str, Any],
    *,
    min_expiries: int | None = None,
    min_nodes: int | None = None,
) -> tuple[list[dict], dict]:
    """Build raw settlement-IV nodes and auditable parity diagnostics.

    Each expiry receives total objective weight one.  The resulting fit is not
    dominated by the front contract merely because it has a denser strike ladder.

    ``min_expiries`` / ``min_nodes`` default to the frozen cross-date gates
    (``MIN_EXPIRIES`` / ``MIN_NODES``); the surface-history stage relaxes them
    because it admits any date with >= 2 SABR-fittable expiries.
    """
    _validate_snapshot_identity(snapshot)
    trade_date = _parse_iso_date(snapshot["trade_date"], field="trade_date")
    nodes: list[dict] = []
    parity_rows: list[dict] = []
    evaluated_parity_rows: list[dict] = []
    excluded_expiries: list[dict] = []
    filtered_quotes = Counter()

    for expiry in snapshot["expiries"]:
        contract_month = str(expiry.get("contract_month", ""))
        expected_expiry = _third_friday(contract_month)
        supplied_expiry = _parse_iso_date(expiry.get("expiry_date"), field="expiry_date")
        if supplied_expiry != expected_expiry:
            raise ValueError(
                f"{contract_month}: expiry_date {supplied_expiry} is not third Friday "
                f"{expected_expiry}"
            )
        calendar_days = (expected_expiry - trade_date).days
        if not MIN_CALENDAR_DAYS <= calendar_days <= MAX_CALENDAR_DAYS:
            excluded_expiries.append(
                {
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "reason": "outside_maturity_window",
                }
            )
            continue
        maturity = calendar_days / 365.0
        by_strike: dict[float, dict[str, dict]] = {}
        for quote in expiry.get("quotes", []):
            option_type = str(quote.get("type", ""))
            if option_type not in {"C", "P"}:
                raise ValueError(f"{contract_month}: quote type must be C or P")
            strike = _finite_positive(quote.get("strike"), name="strike")
            if option_type in by_strike.setdefault(strike, {}):
                raise ValueError(
                    f"{contract_month}: duplicate {option_type} quote at strike {strike}"
                )
            by_strike[strike][option_type] = dict(quote)

        paired_strikes = sorted(
            strike for strike, quotes in by_strike.items() if set(quotes) >= {"C", "P"}
        )
        if len(paired_strikes) < 3:
            excluded_expiries.append(
                {
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "reason": "fewer_than_three_settlement_pairs",
                }
            )
            continue
        strikes_for_parity: list[float] = []
        differences: list[float] = []
        for strike in paired_strikes:
            call = by_strike[strike]["C"].get(PRICE_FIELD)
            put = by_strike[strike]["P"].get(PRICE_FIELD)
            try:
                call_price = _finite_positive(call, name=f"{contract_month} call settlement")
                put_price = _finite_positive(put, name=f"{contract_month} put settlement")
            except ValueError:
                filtered_quotes["invalid_parity_settlement"] += 1
                continue
            strikes_for_parity.append(strike)
            differences.append(call_price - put_price)
        if len(strikes_for_parity) < 3:
            excluded_expiries.append(
                {
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "reason": "fewer_than_three_valid_settlement_pairs",
                }
            )
            continue

        strike_array = np.asarray(strikes_for_parity, dtype=float)
        difference_array = np.asarray(differences, dtype=float)
        slope, intercept = np.polyfit(strike_array, difference_array, 1)
        discount_factor = -float(slope)
        if not math.isfinite(discount_factor) or discount_factor <= 0.0:
            excluded_expiries.append(
                {
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "reason": "non_positive_parity_discount_factor",
                    "discount_factor": discount_factor,
                }
            )
            continue
        forward = float(intercept / discount_factor)
        if not math.isfinite(forward) or forward <= 0.0:
            excluded_expiries.append(
                {
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "reason": "non_positive_parity_forward",
                    "forward": forward,
                }
            )
            continue
        parity_residuals = difference_array - (
            -discount_factor * strike_array + discount_factor * forward
        )
        implied_rate = -math.log(discount_factor) / maturity
        parity_rmse_points = float(np.sqrt(np.mean(np.square(parity_residuals))))
        parity_rmse_forward_ratio = parity_rmse_points / forward
        parity_evaluation = {
            "contract_month": contract_month,
            "expiry_date": expected_expiry.isoformat(),
            "calendar_days": calendar_days,
            "T": maturity,
            "pair_count": len(strikes_for_parity),
            "forward": forward,
            "discount_factor": discount_factor,
            "implied_rate": implied_rate,
            "parity_rmse_points": parity_rmse_points,
            "parity_rmse_forward_ratio": parity_rmse_forward_ratio,
            "quality_gate_passed": bool(
                abs(implied_rate) <= MAX_ABS_PARITY_IMPLIED_RATE
                and parity_rmse_forward_ratio <= MAX_PARITY_RMSE_FORWARD_RATIO
            ),
            "near_atm_sensitivity": _near_atm_parity_sensitivity(
                strike_array,
                difference_array,
                primary_forward=forward,
                primary_rate=implied_rate,
                maturity=maturity,
            ),
        }
        evaluated_parity_rows.append(parity_evaluation)
        if not parity_evaluation["quality_gate_passed"]:
            excluded_expiries.append(
                {
                    **parity_evaluation,
                    "reason": "parity_quality_gate_failed",
                    "maximum_absolute_implied_rate": MAX_ABS_PARITY_IMPLIED_RATE,
                    "maximum_rmse_forward_ratio": MAX_PARITY_RMSE_FORWARD_RATIO,
                }
            )
            continue

        expiry_nodes: list[dict] = []
        for strike in paired_strikes:
            option_type = "P" if strike < forward else "C"
            quote = by_strike[strike][option_type]
            settlement = quote.get(PRICE_FIELD)
            try:
                option_price = _finite_positive(
                    settlement, name=f"{quote.get('contract')} settlement"
                )
            except ValueError:
                filtered_quotes["invalid_selected_settlement"] += 1
                continue
            try:
                volume = int(quote.get("volume", 0))
                open_interest = int(quote.get("oi", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{quote.get('contract')}: invalid volume/OI") from exc
            if volume <= 0:
                filtered_quotes["zero_selected_volume"] += 1
                continue
            if open_interest <= 0:
                filtered_quotes["zero_selected_open_interest"] += 1
                continue
            call_equivalent = (
                option_price + discount_factor * (forward - strike)
                if option_type == "P"
                else option_price
            )
            normalized_strike = strike / forward
            normalized_price = call_equivalent / (discount_factor * forward)
            if not (
                math.isfinite(normalized_price)
                and max(1.0 - normalized_strike, 0.0) < normalized_price < 1.0
            ):
                filtered_quotes["normalized_price_outside_no_arbitrage_bounds"] += 1
                continue
            try:
                implied_vol = implied_vol_call(
                    1.0,
                    normalized_strike,
                    maturity,
                    normalized_price,
                    0.0,
                    0.0,
                )
            except (NumericalError, ValueError, OverflowError):
                filtered_quotes["iv_inversion_failure"] += 1
                continue
            if not math.isfinite(implied_vol) or not 0.0 < implied_vol < 2.0:
                filtered_quotes["iv_outside_sanity_range"] += 1
                continue
            expiry_nodes.append(
                {
                    "contract": quote.get("contract"),
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "T": maturity,
                    "option_type": option_type,
                    "strike": strike,
                    "forward": forward,
                    "discount_factor": discount_factor,
                    "normalized_strike": normalized_strike,
                    "normalized_call_price": normalized_price,
                    "log_moneyness": math.log(normalized_strike),
                    "market_iv": implied_vol,
                    "volume": volume,
                    "open_interest": open_interest,
                }
            )

        if not expiry_nodes:
            excluded_expiries.append(
                {
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "reason": "no_usable_otm_nodes",
                }
            )
            continue
        wing_counts = Counter(node["option_type"] for node in expiry_nodes)
        if wing_counts["P"] == 0 or wing_counts["C"] == 0:
            excluded_expiries.append(
                {
                    "contract_month": contract_month,
                    "expiry_date": expected_expiry.isoformat(),
                    "calendar_days": calendar_days,
                    "reason": "missing_liquid_otm_wing",
                    "put_node_count": wing_counts["P"],
                    "call_node_count": wing_counts["C"],
                }
            )
            continue
        expiry_weight = 1.0 / len(expiry_nodes)
        for node in expiry_nodes:
            node["weight"] = expiry_weight
        nodes.extend(expiry_nodes)
        moneyness = [node["log_moneyness"] for node in expiry_nodes]
        static_arbitrage = _static_arbitrage_diagnostics(expiry_nodes)
        parity_rows.append(
            {
                "contract_month": contract_month,
                "expiry_date": expected_expiry.isoformat(),
                "calendar_days": calendar_days,
                "T": maturity,
                "pair_count": len(strikes_for_parity),
                "node_count": len(expiry_nodes),
                "put_node_count": wing_counts["P"],
                "call_node_count": wing_counts["C"],
                "two_sided_otm_wings": True,
                "forward": forward,
                "discount_factor": discount_factor,
                "implied_rate": implied_rate,
                "parity_rmse_points": parity_rmse_points,
                "parity_rmse_forward_ratio": parity_rmse_forward_ratio,
                "min_log_moneyness": min(moneyness),
                "max_log_moneyness": max(moneyness),
                "static_arbitrage": static_arbitrage,
            }
        )

    nodes.sort(key=lambda row: (row["T"], row["normalized_strike"]))
    expiry_count = len({node["expiry_date"] for node in nodes})
    if not evaluated_parity_rows:
        raise CoverageError(
            f"{snapshot['trade_date']}: no expiry passed basic parity construction",
            details={"excluded_expiries": excluded_expiries},
        )
    metadata = {
        "node_count": len(nodes),
        "expiry_count": expiry_count,
        "node_keys": [[node["expiry_date"], node["contract"]] for node in nodes],
        "per_expiry": parity_rows,
        "excluded_expiries": excluded_expiries,
        "filtered_quote_counts": dict(sorted(filtered_quotes.items())),
        "total_objective_weight": float(sum(node["weight"] for node in nodes)),
        "static_arbitrage": {
            "method": "raw_normalized_call_monotonicity_and_irregular_grid_slope_convexity",
            "repair_applied": False,
            "non_increasing_call_violations": sum(
                row["static_arbitrage"]["non_increasing_call_violations"]
                for row in parity_rows
            ),
            "convex_slope_violations": sum(
                row["static_arbitrage"]["convex_slope_violations"]
                for row in parity_rows
            ),
            "affected_contract_months": [
                row["contract_month"]
                for row in parity_rows
                if row["static_arbitrage"]["non_increasing_call_violations"]
                or row["static_arbitrage"]["convex_slope_violations"]
            ],
        },
        "parity_quality": {
            "rmse_points_range": [
                min(row["parity_rmse_points"] for row in evaluated_parity_rows),
                max(row["parity_rmse_points"] for row in evaluated_parity_rows),
            ],
            "rmse_forward_ratio_range": [
                min(row["parity_rmse_forward_ratio"] for row in evaluated_parity_rows),
                max(row["parity_rmse_forward_ratio"] for row in evaluated_parity_rows),
            ],
            "implied_rate_range": [
                min(row["implied_rate"] for row in evaluated_parity_rows),
                max(row["implied_rate"] for row in evaluated_parity_rows),
            ],
            "discount_factor_range": [
                min(row["discount_factor"] for row in evaluated_parity_rows),
                max(row["discount_factor"] for row in evaluated_parity_rows),
            ],
            "quality_gate": {
                "maximum_absolute_implied_rate": MAX_ABS_PARITY_IMPLIED_RATE,
                "maximum_rmse_forward_ratio": MAX_PARITY_RMSE_FORWARD_RATIO,
            },
            "failed_contract_months": [
                row["contract_month"]
                for row in evaluated_parity_rows
                if not row["quality_gate_passed"]
            ],
            "evaluated_expiries": evaluated_parity_rows,
        },
    }
    if min_expiries is None:
        min_expiries = MIN_EXPIRIES
    if min_nodes is None:
        min_nodes = MIN_NODES
    if expiry_count < min_expiries or len(nodes) < min_nodes:
        raise CoverageError(
            f"{snapshot['trade_date']}: need >= {min_expiries} expiries and >= {min_nodes} "
            f"nodes after parity quality gating, got {expiry_count} expiries and "
            f"{len(nodes)} nodes",
            details=metadata,
        )
    return nodes, metadata


def _params_to_dict(params: HestonParams) -> dict[str, float]:
    return {name: float(getattr(params, name)) for name in PARAMETER_NAMES}


def _params_from_vector(values: Sequence[float]) -> HestonParams:
    return HestonParams(**dict(zip(PARAMETER_NAMES, map(float, values))))


def _model_ivs(nodes: Sequence[Mapping[str, Any]], params: HestonParams) -> np.ndarray:
    """Evaluate Lewis Heston IVs on the normalized cross-date nodes."""
    strikes = np.asarray([node["normalized_strike"] for node in nodes], dtype=float)
    maturities = np.asarray([node["T"] for node in nodes], dtype=float)
    model = np.empty(len(nodes), dtype=float)
    for maturity in np.unique(maturities):
        indices = np.flatnonzero(maturities == maturity)
        prices = heston_call_prices_vectorized(
            1.0, strikes[indices], float(maturity), params, 0.0, 0.0
        )
        model[indices] = [
            implied_vol_call(1.0, float(strike), float(maturity), float(price), 0.0, 0.0)
            for strike, price in zip(strikes[indices], prices)
        ]
    if not np.all(np.isfinite(model)):
        raise NumericalError("Heston model produced non-finite implied vols")
    return model


def _bound_hits(params: HestonParams) -> dict[str, dict[str, bool]]:
    values = np.asarray([getattr(params, name) for name in PARAMETER_NAMES], dtype=float)
    lower = np.asarray(HESTON_BOUNDS[0], dtype=float)
    upper = np.asarray(HESTON_BOUNDS[1], dtype=float)
    tolerance = BOUND_HIT_RELATIVE_TOLERANCE * (upper - lower)
    return {
        name: {
            "lower": bool(abs(value - lo) <= tol),
            "upper": bool(abs(hi - value) <= tol),
        }
        for name, value, lo, hi, tol in zip(
            PARAMETER_NAMES, values, lower, upper, tolerance
        )
    }


def _fit_metrics(
    nodes: Sequence[Mapping[str, Any]], params: HestonParams
) -> tuple[dict, list[dict], np.ndarray]:
    model = _model_ivs(nodes, params)
    market = np.asarray([node["market_iv"] for node in nodes], dtype=float)
    weights = np.asarray([node["weight"] for node in nodes], dtype=float)
    errors = model - market
    weighted_rmse = float(np.sqrt(np.sum(weights * errors**2) / np.sum(weights)))
    rows = []
    per_expiry = []
    for node, model_iv, error in zip(nodes, model, errors):
        rows.append(
            {
                "contract": node["contract"],
                "expiry_date": node["expiry_date"],
                "T": node["T"],
                "normalized_strike": node["normalized_strike"],
                "market_iv": node["market_iv"],
                "model_iv": float(model_iv),
                "error_iv": float(error),
                "weight": node["weight"],
            }
        )
    for expiry_date in sorted({node["expiry_date"] for node in nodes}):
        indices = [i for i, node in enumerate(nodes) if node["expiry_date"] == expiry_date]
        expiry_errors = errors[indices]
        per_expiry.append(
            {
                "expiry_date": expiry_date,
                "T": nodes[indices[0]]["T"],
                "node_count": len(indices),
                "rmse_iv": float(np.sqrt(np.mean(expiry_errors**2))),
            }
        )
    return (
        {
            "rmse_iv": float(np.sqrt(np.mean(errors**2))),
            "weighted_rmse_iv": weighted_rmse,
            "mae_iv": float(np.mean(np.abs(errors))),
            "max_abs_iv": float(np.max(np.abs(errors))),
            "per_expiry": per_expiry,
        },
        rows,
        model,
    )


def _initial_starts(nodes: Sequence[Mapping[str, Any]]) -> list[HestonParams]:
    near_atm = sorted(nodes, key=lambda node: (abs(node["log_moneyness"]), node["T"]))
    anchor = float(np.median([node["market_iv"] for node in near_atm[:12]]))
    level = float(np.clip(anchor**2, 0.01, 0.2))
    return [
        HestonParams(v0=level, kappa=2.0, theta=level, sigma=0.6, rho=-0.5),
        HestonParams(
            v0=float(np.clip(0.8 * level, 0.01, 0.2)),
            kappa=1.0,
            theta=float(np.clip(1.2 * level, 0.01, 0.2)),
            sigma=0.35,
            rho=-0.3,
        ),
        HestonParams(
            v0=float(np.clip(1.2 * level, 0.01, 0.2)),
            kappa=2.8,
            theta=float(np.clip(0.8 * level, 0.01, 0.2)),
            sigma=0.68,
            rho=-0.7,
        ),
    ]


def calibration_config(*, max_nfev: int) -> dict:
    if not isinstance(max_nfev, int) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    return {
        "calibration_target": "raw_official_CFFEX_settlement_implied_vols",
        "source_class": SOURCE_CLASS,
        "price_field": PRICE_FIELD,
        "normalization": "expiry_parity_forward_equals_1_and_rates_equal_0",
        "parity_method": "OLS_of_C_minus_P_on_strike",
        "parity_sensitivity": {
            "method": "OLS_on_nearest_strikes_to_primary_forward",
            "maximum_pairs": NEAR_ATM_PARITY_PAIR_COUNT,
            "diagnostic_only": True,
        },
        "parity_quality_gate": {
            "maximum_absolute_annualized_implied_rate": MAX_ABS_PARITY_IMPLIED_RATE,
            "maximum_rmse_divided_by_forward": MAX_PARITY_RMSE_FORWARD_RATIO,
        },
        "surface_preparation": "none_no_interpolation_no_extrapolation_no_smoothing",
        "static_arbitrage_diagnostic": {
            "method": "raw_normalized_call_monotonicity_and_irregular_grid_slope_convexity",
            "absolute_tolerance": STATIC_ARBITRAGE_ABSOLUTE_TOLERANCE,
            "diagnostic_only_no_repair_no_gate": True,
        },
        "maturity_day_count": "ACT/365",
        "expiry_calendar": {
            "rule": "third_Friday_rolled_to_next_trading_day",
            "frozen_overrides": {
                key: value.isoformat() for key, value in EXPIRY_DATE_OVERRIDES.items()
            },
        },
        "maturity_window_calendar_days": [MIN_CALENDAR_DAYS, MAX_CALENDAR_DAYS],
        "liquidity_filter": "selected_OTM_side_volume_gt_0_and_open_interest_gt_0",
        "two_sided_otm_wings_required_per_expiry": True,
        "weighting": "equal_total_weight_per_expiry",
        "method": "lewis",
        "target": "iv",
        "bounds": [list(HESTON_BOUNDS[0]), list(HESTON_BOUNDS[1])],
        "regularize_feller": REGULARIZE_FELLER,
        "enforce_feller": ENFORCE_FELLER,
        "optimizer_tolerances": {
            "xtol": OPTIMIZER_XTOL,
            "ftol": OPTIMIZER_FTOL,
            "gtol": OPTIMIZER_GTOL,
        },
        "starts": START_COUNT,
        "best_start_selection_policy": BEST_START_SELECTION_POLICY,
        "max_nfev": max_nfev,
        "minimum_expiries": MIN_EXPIRIES,
        "minimum_nodes": MIN_NODES,
        "jacobian_scale_for_cross_date": "fixed_economic",
    }


def _calibrate_nodes(nodes: Sequence[Mapping[str, Any]], *, max_nfev: int) -> dict:
    options = [
        MarketOption(
            K=float(node["normalized_strike"]),
            T=float(node["T"]),
            iv=float(node["market_iv"]),
            weight=float(node["weight"]),
        )
        for node in nodes
    ]
    attempts: list[dict] = []
    params_by_attempt: dict[int, HestonParams] = {}
    for index, initial in enumerate(_initial_starts(nodes)):
        initial_dict = _params_to_dict(initial)
        try:
            result = calibrate_heston(
                s0=1.0,
                options=options,
                r=0.0,
                carry=0.0,
                initial=initial,
                bounds=HESTON_BOUNDS,
                target="iv",
                method="lewis",
                regularize_feller=REGULARIZE_FELLER,
                max_nfev=max_nfev,
                xtol=OPTIMIZER_XTOL,
                ftol=OPTIMIZER_FTOL,
                gtol=OPTIMIZER_GTOL,
                enforce_feller=ENFORCE_FELLER,
            )
            params = result.params
            metrics, _rows, _model = _fit_metrics(nodes, params)
            feller_margin = 2.0 * params.kappa * params.theta - params.sigma**2
            attempt = {
                "start_index": index,
                "initial": initial_dict,
                "success": bool(result.success),
                "message": str(result.message),
                "optimizer": str(result.optimizer),
                "nfev": int(result.nfev),
                "params": _params_to_dict(params),
                "cost": float(result.cost),
                "data_cost": float(result.data_cost),
                "feller_penalty_cost": float(result.feller_penalty_cost),
                "feller_margin": float(feller_margin),
                "feller_ratio": float(
                    2.0 * params.kappa * params.theta / params.sigma**2
                ),
                "feller_satisfied": bool(feller_margin >= 0.0),
                "bound_hits": _bound_hits(params),
                **metrics,
            }
            attempts.append(attempt)
            params_by_attempt[index] = params
        except Exception as exc:  # preserve every failed deterministic start
            attempts.append(
                {
                    "start_index": index,
                    "initial": initial_dict,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    successful = [
        attempt
        for attempt in attempts
        if attempt.get("success") is True
        and math.isfinite(float(attempt.get("weighted_rmse_iv", math.inf)))
    ]
    if not successful:
        raise RuntimeError(f"all {START_COUNT} Heston starts failed: {attempts}")
    best = min(successful, key=lambda attempt: float(attempt["weighted_rmse_iv"]))
    best_params = params_by_attempt[int(best["start_index"])]
    best_metrics, best_rows, _model = _fit_metrics(nodes, best_params)
    best = {**best, **best_metrics, "rows": best_rows}

    values = np.asarray(
        [[attempt["params"][name] for name in PARAMETER_NAMES] for attempt in successful],
        dtype=float,
    )
    threshold = float(best["weighted_rmse_iv"]) * 1.05 + 1e-12
    near_best = [
        attempt for attempt in successful if float(attempt["weighted_rmse_iv"]) <= threshold
    ]
    multistart = {
        "requested": START_COUNT,
        "successful": len(successful),
        "failed": START_COUNT - len(successful),
        "near_best_within_5pct": len(near_best),
        "best_start_selection_policy": BEST_START_SELECTION_POLICY,
        "weighted_rmse_range_iv": [
            min(float(attempt["weighted_rmse_iv"]) for attempt in successful),
            max(float(attempt["weighted_rmse_iv"]) for attempt in successful),
        ],
        "parameter_ranges": {
            name: [float(np.min(values[:, index])), float(np.max(values[:, index]))]
            for index, name in enumerate(PARAMETER_NAMES)
        },
    }
    parameter_vector = np.asarray(
        [best["params"][name] for name in PARAMETER_NAMES], dtype=float
    )
    jacobian = _identification_jacobian(nodes, parameter_vector)
    return {
        "best": best,
        "multistart": multistart,
        "fits": attempts,
        "jacobian": jacobian,
    }


def _identification_jacobian(
    nodes: Sequence[Mapping[str, Any]], parameter_vector: Sequence[float]
) -> dict:
    """Differentiate the actual equal-per-expiry weighted IV objective.

    The SVD must see the same ``sqrt(weight)`` row scaling as the optimizer.
    The unweighted IV Jacobian matrix is recovered algebraically and retained
    for audit, but cross-date condition numbers use the objective-weighted view.
    """
    square_root_weights = np.sqrt(
        np.asarray([float(node["weight"]) for node in nodes], dtype=float)
    )
    if np.any(~np.isfinite(square_root_weights)) or np.any(square_root_weights <= 0.0):
        raise ValueError("Jacobian node weights must be finite and positive")
    jacobian = hd.finite_difference_model_jacobian(
        lambda values_: _model_ivs(nodes, _params_from_vector(values_))
        * square_root_weights,
        parameter_vector,
        HESTON_BOUNDS[0],
        HESTON_BOUNDS[1],
    )
    weighted_matrix = np.asarray(jacobian["matrix"], dtype=float)
    jacobian["row_weighting"] = {
        "policy": "sqrt_equal_total_weight_per_expiry",
        "weights": [float(value * value) for value in square_root_weights],
        "cross_date_svd_uses_weighted_rows": True,
    }
    jacobian["unweighted_market_iv_matrix"] = (
        weighted_matrix / square_root_weights[:, None]
    ).tolist()
    return jacobian


def calibrate_snapshot(snapshot: Mapping[str, Any], *, max_nfev: int = 250) -> dict:
    """Build nodes, run deterministic multistart, and return one date artifact."""
    nodes, universe = build_calibration_nodes(snapshot)
    fit = _calibrate_nodes(nodes, max_nfev=max_nfev)
    return {
        "schema_version": 1,
        "trade_date": snapshot["trade_date"],
        "source_class": snapshot["source_class"],
        "source_url": snapshot.get("source_url"),
        "source_sha256": snapshot["source_sha256"],
        "price_field": snapshot["price_field"],
        "config": calibration_config(max_nfev=max_nfev),
        "node_universe": universe,
        **fit,
    }


def _distribution(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if len(array) >= 2 else 0.0
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "range": float(np.max(array) - np.min(array)),
        "mean": mean,
        "std": standard_deviation,
        "cv_abs_mean": (
            standard_deviation / abs(mean) if abs(mean) > 1e-14 else None
        ),
    }


def _config_signature(evidence: Mapping[str, Any]) -> dict:
    config = evidence.get("config")
    if not isinstance(config, dict):
        raise ValueError("date evidence missing config")
    return dict(config)


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a canonical, finite JSON fingerprint for configuration voting."""
    return json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_config(
    evidences: Sequence[Mapping[str, Any]],
    expected_config: Mapping[str, Any] | None,
) -> dict | None:
    """Choose an explicit config or a deterministic majority fingerprint."""
    if expected_config is not None:
        selected = dict(expected_config)
        _config_fingerprint(selected)
        return selected
    by_fingerprint: dict[str, dict] = {}
    counts: Counter[str] = Counter()
    for evidence in evidences:
        if (
            evidence.get("source_class") != SOURCE_CLASS
            or evidence.get("price_field") != PRICE_FIELD
            or evidence.get("best", {}).get("success") is not True
        ):
            continue
        try:
            signature = _config_signature(evidence)
            fingerprint = _config_fingerprint(signature)
        except (TypeError, ValueError):
            continue
        by_fingerprint[fingerprint] = signature
        counts[fingerprint] += 1
    if not counts:
        return None
    largest_count = max(counts.values())
    # Lexical tie-break makes a tied vote independent of input order.
    selected_fingerprint = min(
        fingerprint
        for fingerprint, count in counts.items()
        if count == largest_count
    )
    return by_fingerprint[selected_fingerprint]


def aggregate_evidence(
    evidences: Sequence[Mapping[str, Any]],
    *,
    extra_exclusions: Sequence[Mapping[str, Any]] = (),
    expected_config: Mapping[str, Any] | None = None,
) -> dict:
    """Apply the strict source/config/identity gate and summarize date stability."""
    included: list[dict] = []
    exclusions = [dict(row) for row in extra_exclusions]
    seen_dates: set[str] = set()
    seen_hashes: set[str] = set()
    baseline_config = _canonical_config(evidences, expected_config)

    # Identity duplicates keep the established first-observation policy.  The
    # configuration baseline itself is selected independently of this order by
    # ``_canonical_config`` above.
    for original in evidences:
        evidence = dict(original)
        trade_date = evidence.get("trade_date")
        if evidence.get("source_class") != SOURCE_CLASS:
            exclusions.append(
                {"trade_date": trade_date, "reason": "source_class_mismatch"}
            )
            continue
        if evidence.get("price_field") != PRICE_FIELD:
            exclusions.append(
                {"trade_date": trade_date, "reason": "price_field_mismatch"}
            )
            continue
        try:
            _parse_iso_date(trade_date, field="trade_date")
        except ValueError as exc:
            exclusions.append({"trade_date": trade_date, "reason": str(exc)})
            continue
        if trade_date in seen_dates:
            exclusions.append({"trade_date": trade_date, "reason": "duplicate_trade_date"})
            continue
        digest = evidence.get("source_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            exclusions.append({"trade_date": trade_date, "reason": "invalid_source_sha256"})
            continue
        if digest in seen_hashes:
            exclusions.append(
                {
                    "trade_date": trade_date,
                    "reason": "duplicate_source_sha256",
                    "source_sha256": digest,
                }
            )
            continue
        if evidence.get("best", {}).get("success") is not True:
            exclusions.append({"trade_date": trade_date, "reason": "no_successful_best_fit"})
            continue
        try:
            signature = _config_signature(evidence)
        except ValueError as exc:
            exclusions.append({"trade_date": trade_date, "reason": str(exc)})
            continue
        if baseline_config is None or signature != baseline_config:
            exclusions.append(
                {
                    "trade_date": trade_date,
                    "reason": "calibration_config_mismatch",
                    "expected_config": baseline_config,
                    "actual_config": signature,
                }
            )
            continue
        seen_dates.add(trade_date)
        seen_hashes.add(digest)
        included.append(evidence)

    included.sort(key=lambda row: row["trade_date"])
    exclusions.sort(
        key=lambda row: (
            str(row.get("trade_date", row.get("tag", ""))),
            str(row.get("reason", "")),
            str(row.get("source_sha256", "")),
        )
    )
    stability: dict[str, Any] = {}
    if included:
        stability["parameters"] = {
            name: _distribution([row["best"]["params"][name] for row in included])
            for name in PARAMETER_NAMES
        }
        stability["weighted_rmse_iv"] = _distribution(
            [row["best"]["weighted_rmse_iv"] for row in included]
        )
        stability["feller_ratio"] = _distribution(
            [row["best"]["feller_ratio"] for row in included]
        )
        stability["node_count"] = _distribution(
            [row["node_universe"]["node_count"] for row in included]
        )
        stability["bound_hit_frequency"] = {}
        for name in PARAMETER_NAMES:
            hits = [
                row["best"]["bound_hits"][name]["lower"]
                or row["best"]["bound_hits"][name]["upper"]
                for row in included
            ]
            stability["bound_hit_frequency"][name] = float(np.mean(hits))
        fixed_svd = [row["jacobian"]["svd"]["fixed_economic"] for row in included]
        stability["identification"] = {
            "minimum_policy_effective_rank": min(
                int(row["policy_effective_rank"]) for row in fixed_svd
            ),
            "condition_number_range": [
                min(
                    float(row["condition_number"])
                    for row in fixed_svd
                    if row["condition_number"] is not None
                ),
                max(
                    float(row["condition_number"])
                    for row in fixed_svd
                    if row["condition_number"] is not None
                ),
            ]
            if any(row["condition_number"] is not None for row in fixed_svd)
            else None,
            "scale_policy": "fixed_economic",
            "row_weighting": "sqrt_equal_total_weight_per_expiry",
        }
    candidate_universes = [
        (row["trade_date"], row["node_universe"]) for row in included
    ] + [
        (exclusion.get("trade_date"), exclusion["details"])
        for exclusion in exclusions
        if isinstance(exclusion.get("details"), dict)
    ]
    candidate_parity_rows: list[dict] = []
    for trade_date, universe in candidate_universes:
        for expiry in universe.get("parity_quality", {}).get("evaluated_expiries", []):
            candidate_parity_rows.append({"trade_date": trade_date, **expiry})
    if candidate_parity_rows:
        maximum_parity = max(
            candidate_parity_rows, key=lambda row: row["parity_rmse_points"]
        )
        failed_pillars = [
            {
                "trade_date": row["trade_date"],
                "contract_month": row["contract_month"],
                "implied_rate": row["implied_rate"],
                "parity_rmse_points": row["parity_rmse_points"],
                "parity_rmse_forward_ratio": row["parity_rmse_forward_ratio"],
            }
            for row in candidate_parity_rows
            if not row["quality_gate_passed"]
        ]
        sensitivity_rows = [
            {"trade_date": row["trade_date"], "contract_month": row["contract_month"], **sensitivity}
            for row in candidate_parity_rows
            if (sensitivity := row.get("near_atm_sensitivity", {})).get("status")
            == "measured"
        ]
        max_forward_sensitivity = (
            max(
                sensitivity_rows,
                key=lambda row: abs(row["forward_relative_difference_vs_full_ols"]),
            )
            if sensitivity_rows
            else None
        )
        max_rate_sensitivity = (
            max(
                sensitivity_rows,
                key=lambda row: abs(row["implied_rate_difference_vs_full_ols"]),
            )
            if sensitivity_rows
            else None
        )
        stability["parity_quality"] = {
            "maximum_rmse_points": maximum_parity["parity_rmse_points"],
            "maximum_rmse_forward_ratio": maximum_parity[
                "parity_rmse_forward_ratio"
            ],
            "maximum_rmse_trade_date": maximum_parity["trade_date"],
            "maximum_rmse_contract_month": maximum_parity["contract_month"],
            "implied_rate_range": [
                min(row["implied_rate"] for row in candidate_parity_rows),
                max(row["implied_rate"] for row in candidate_parity_rows),
            ],
            "discount_factor_range": [
                min(row["discount_factor"] for row in candidate_parity_rows),
                max(row["discount_factor"] for row in candidate_parity_rows),
            ],
            "quality_gate": {
                "maximum_absolute_implied_rate": MAX_ABS_PARITY_IMPLIED_RATE,
                "maximum_rmse_forward_ratio": MAX_PARITY_RMSE_FORWARD_RATIO,
            },
            "failed_pillars": failed_pillars,
            "dates_with_failed_parity_pillars": sorted(
                {row["trade_date"] for row in failed_pillars}
            ),
            "near_atm_sensitivity": {
                "method": "OLS_on_nearest_strikes_to_primary_forward",
                "maximum_pairs": NEAR_ATM_PARITY_PAIR_COUNT,
                "measured_pillars": len(sensitivity_rows),
                "maximum_absolute_forward_relative_difference": (
                    abs(max_forward_sensitivity["forward_relative_difference_vs_full_ols"])
                    if max_forward_sensitivity is not None
                    else None
                ),
                "maximum_forward_difference_trade_date": (
                    max_forward_sensitivity["trade_date"]
                    if max_forward_sensitivity is not None
                    else None
                ),
                "maximum_forward_difference_contract_month": (
                    max_forward_sensitivity["contract_month"]
                    if max_forward_sensitivity is not None
                    else None
                ),
                "maximum_absolute_implied_rate_difference": (
                    abs(max_rate_sensitivity["implied_rate_difference_vs_full_ols"])
                    if max_rate_sensitivity is not None
                    else None
                ),
                "maximum_rate_difference_trade_date": (
                    max_rate_sensitivity["trade_date"]
                    if max_rate_sensitivity is not None
                    else None
                ),
                "maximum_rate_difference_contract_month": (
                    max_rate_sensitivity["contract_month"]
                    if max_rate_sensitivity is not None
                    else None
                ),
            },
        }
    static_rows = [
        {"trade_date": trade_date, **universe.get("static_arbitrage", {})}
        for trade_date, universe in candidate_universes
        if universe.get("static_arbitrage")
    ]
    if static_rows:
        affected_expiries = [
            {"trade_date": row["trade_date"], "contract_month": contract_month}
            for row in static_rows
            for contract_month in row.get("affected_contract_months", [])
        ]
        stability["static_arbitrage"] = {
            "repair_applied": False,
            "non_increasing_call_violations": sum(
                int(row.get("non_increasing_call_violations", 0)) for row in static_rows
            ),
            "convex_slope_violations": sum(
                int(row.get("convex_slope_violations", 0)) for row in static_rows
            ),
            "affected_dates": sorted(
                {
                    row["trade_date"]
                    for row in static_rows
                    if row.get("non_increasing_call_violations", 0)
                    or row.get("convex_slope_violations", 0)
                }
            ),
            "affected_expiry_count": len(affected_expiries),
            "affected_expiries": affected_expiries,
            "per_date": static_rows,
        }

    date_count = len(included)
    high_cv = []
    frequent_bounds = []
    if included:
        high_cv = [
            name
            for name, row in stability["parameters"].items()
            if row["cv_abs_mean"] is not None and row["cv_abs_mean"] > 0.5
        ]
        frequent_bounds = [
            name
            for name, frequency in stability["bound_hit_frequency"].items()
            if frequency >= 0.5
        ]
    identification = stability.get("identification", {})
    rank = identification.get("minimum_policy_effective_rank")
    verdicts = [
        {
            "name": "genuine_cross_date_coverage",
            "status": "measured" if date_count >= 5 else "insufficient",
            "evidence": {
                "included_dates": date_count,
                "minimum_dates_for_panel": 5,
                "excluded_candidates": len(exclusions),
            },
            "interpretation": (
                "The panel contains independently dated official settlement cross sections."
                if date_count >= 5
                else "Fewer than five comparable genuine dates remain after strict gating."
            ),
        },
        {
            "name": "parameter_stability",
            "status": (
                "insufficient"
                if date_count < 2
                else ("warning" if high_cv or frequent_bounds else "measured")
            ),
            "evidence": {
                "parameter_cv_above_0_5": high_cv,
                "parameters_at_bounds_on_at_least_half_dates": frequent_bounds,
            },
            "interpretation": "Cross-date movement is evidence, not a universal pass/fail test.",
        },
        {
            "name": "local_identification",
            "status": (
                "insufficient"
                if rank is None
                else ("warning" if rank < len(PARAMETER_NAMES) else "measured")
            ),
            "evidence": identification,
            "interpretation": (
                "The fixed-economic-scale Jacobian is comparable across dates; it tests local, "
                "not global, identification."
            ),
        },
        {
            "name": "source_scope",
            "status": "limitation",
            "evidence": {"source_class": SOURCE_CLASS, "price_field": PRICE_FIELD},
            "interpretation": (
                "Official settlement history is genuine EOD market evidence but is not "
                "executable bid/ask history."
            ),
        },
        {
            "name": "settlement_parity_quality",
            "status": (
                "warning"
                if stability.get("parity_quality", {}).get("failed_pillars")
                else "measured"
            ),
            "evidence": stability.get("parity_quality", {}),
            "interpretation": (
                "Settlement put-call parity is measured, not assumed clean; large residuals "
                "or implausible implied rates can contaminate normalized IVs."
            ),
        },
        {
            "name": "raw_settlement_static_arbitrage",
            "status": (
                "warning"
                if stability.get("static_arbitrage", {}).get(
                    "non_increasing_call_violations", 0
                )
                or stability.get("static_arbitrage", {}).get(
                    "convex_slope_violations", 0
                )
                else "measured"
            ),
            "evidence": stability.get("static_arbitrage", {}),
            "interpretation": (
                "Raw settlement call-equivalent nodes are diagnosed without smoothing, "
                "repair, or exclusion from the Heston objective."
            ),
        },
    ]
    return {
        "schema_version": 1,
        "source_class": SOURCE_CLASS,
        "price_field": PRICE_FIELD,
        "strict_comparability_gate": {
            "required_source_class": SOURCE_CLASS,
            "required_price_field": PRICE_FIELD,
            "unique_trade_dates": True,
            "unique_source_sha256": True,
            "required_config": baseline_config,
        },
        "included": included,
        "exclusions": exclusions,
        "stability": stability,
        "verdicts": verdicts,
    }


def build_cross_date_report(
    tags: Sequence[str],
    data_dir: Path,
    *,
    max_nfev: int = 250,
    calibrator: Callable[..., dict] = calibrate_snapshot,
) -> dict:
    """Load, calibrate and strictly aggregate the requested frozen tags."""
    if not tags:
        raise ValueError("at least one settlement snapshot tag is required")
    evidences: list[dict] = []
    exclusions: list[dict] = []
    for tag in tags:
        snapshot: dict | None = None
        if re.fullmatch(r"\d{8}", str(tag)) is None:
            exclusions.append({"tag": tag, "reason": "invalid_tag"})
            continue
        path = data_dir / f"mo_settlement_snapshot_{tag}.json"
        if not path.is_file():
            exclusions.append({"tag": tag, "reason": "missing_snapshot", "path": str(path)})
            continue
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            evidence = calibrator(snapshot, max_nfev=max_nfev)
        except CoverageError as exc:
            exclusions.append(
                {
                    "tag": tag,
                    "trade_date": snapshot.get("trade_date") if snapshot is not None else None,
                    "reason": "coverage_gate_failed",
                    "message": str(exc),
                    "details": exc.details,
                }
            )
            continue
        except Exception as exc:
            exclusions.append(
                {
                    "tag": tag,
                    "trade_date": snapshot.get("trade_date") if snapshot is not None else None,
                    "reason": f"calibration_error: {type(exc).__name__}: {exc}",
                }
            )
            continue
        evidences.append(evidence)
    report = aggregate_evidence(
        evidences,
        extra_exclusions=exclusions,
        expected_config=calibration_config(max_nfev=max_nfev),
    )
    report["requested_tags"] = list(tags)
    report["max_nfev"] = max_nfev
    return report


def _fixed_svd_summary(row: Mapping[str, Any]) -> tuple[Any, Any]:
    summary = row.get("jacobian", {}).get("svd", {}).get("fixed_economic", {})
    return summary.get("condition_number"), summary.get("policy_effective_rank")


def write_artifacts(report: dict, output_dir: Path, output_tag: str) -> dict[str, Path | None]:
    """Write aggregate JSON/CSV and a compact cross-date parameter plot."""
    if re.fullmatch(r"[A-Za-z0-9_-]+", output_tag) is None:
        raise ValueError("output_tag may contain only letters, digits, underscore and hyphen")
    # Fail before creating partial artifacts when a numerical stage leaked NaN/Inf.
    json.dumps(report, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"mo_calibration_diagnostics_{output_tag}.csv"
    columns = [
        "trade_date",
        "node_count",
        "expiry_count",
        *PARAMETER_NAMES,
        "weighted_rmse_iv",
        "feller_ratio",
        "jacobian_condition_fixed_economic",
        "jacobian_effective_rank_fixed_economic",
        *[f"{name}_bound_hit" for name in PARAMETER_NAMES],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["included"]:
            condition, rank = _fixed_svd_summary(row)
            writer.writerow(
                {
                    "trade_date": row["trade_date"],
                    "node_count": row["node_universe"]["node_count"],
                    "expiry_count": row["node_universe"]["expiry_count"],
                    **row["best"]["params"],
                    "weighted_rmse_iv": row["best"]["weighted_rmse_iv"],
                    "feller_ratio": row["best"]["feller_ratio"],
                    "jacobian_condition_fixed_economic": condition,
                    "jacobian_effective_rank_fixed_economic": rank,
                    **{
                        f"{name}_bound_hit": int(
                            row["best"]["bound_hits"][name]["lower"]
                            or row["best"]["bound_hits"][name]["upper"]
                        )
                        for name in PARAMETER_NAMES
                    },
                }
            )

    plot_path: Path | None = None
    if report["included"]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = report["included"]
        x = np.arange(len(rows))
        labels = [row["trade_date"] for row in rows]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        for axis, field in zip(axes.flat, (*PARAMETER_NAMES, "weighted_rmse_iv")):
            values = [
                row["best"]["params"][field]
                if field in PARAMETER_NAMES
                else 100.0 * row["best"][field]
                for row in rows
            ]
            axis.plot(x, values, "o-", color="#a33a2b", linewidth=1.5)
            axis.set_title(field if field in PARAMETER_NAMES else "weighted RMSE (vol pts)")
            axis.set_xticks(x, labels, rotation=35, ha="right")
            axis.grid(alpha=0.25)
        fig.suptitle("MO Heston cross-date diagnostics — official CFFEX settlements")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plot_path = plot_dir / f"10_heston_cross_date_{output_tag}.png"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

    report["artifacts"] = {
        "csv": str(csv_path),
        "plot": str(plot_path) if plot_path is not None else None,
    }
    json_path = output_dir / f"mo_calibration_diagnostics_{output_tag}.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "csv": csv_path, "plot": plot_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", required=True, help="settlement YYYYMMDD tags")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-tag", default="latest")
    parser.add_argument("--max-nfev", type=int, default=250)
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir
    report = build_cross_date_report(
        args.tags, args.data_dir, max_nfev=args.max_nfev
    )
    paths = write_artifacts(report, output_dir, args.output_tag)
    print(
        f"included {len(report['included'])}/{len(args.tags)} date(s), "
        f"excluded {len(report['exclusions'])} -> {paths['json']}"
    )
    for exclusion in report["exclusions"]:
        print(f"  excluded {exclusion.get('tag', exclusion.get('trade_date'))}: "
              f"{exclusion['reason']}")


if __name__ == "__main__":
    main()

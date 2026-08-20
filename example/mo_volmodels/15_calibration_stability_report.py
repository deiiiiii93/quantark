"""Build an auditable HTML/JSON/CSV stability report for daily MO calibrations.

The report reads the persisted manifests produced by
``14_daily_calibration_pipeline.py``. It never recalibrates, fills missing
dates, or drops excluded surfaces from the evidence set.

Example::

    .venv/bin/python example/mo_volmodels/15_calibration_stability_report.py \
      --start 2025-07-31 --end 2026-07-30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_SURFACE_MANIFEST = HERE / "data" / "history" / "surface_manifest.json"
DEFAULT_CALIBRATION_MANIFEST = (
    PROJECT_ROOT / "output" / "mo_daily_calibration" / "calibration_manifest.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "mo_calibration_stability"

PARAMETERS = ("v0", "kappa", "theta", "sigma", "rho")
BOUNDS = {
    "v0": (1e-6, 0.5),
    "kappa": (1e-3, 3.0),
    "theta": (1e-4, 0.5),
    "sigma": (1e-3, 0.7),
    "rho": (-0.95, 0.0),
}
NEGATIVE_MASS_TOLERANCE = 0.05


def parse_date(value: str) -> str:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"invalid date {value!r}; expected YYYY-MM-DD or YYYYMMDD"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def by_date(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise SystemExit("manifest records must be a list")
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("date"):
            raise SystemExit("manifest record is missing a date")
        output[str(record["date"])] = dict(record)
    return output


def percentile(values: Sequence[float], p: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    location = p * (len(clean) - 1)
    lower, upper = math.floor(location), math.ceil(location)
    weight = location - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(clean),
        "min": min(clean) if clean else None,
        "median": statistics.median(clean) if clean else None,
        "mean": statistics.fmean(clean) if clean else None,
        "p95": percentile(clean, 0.95),
        "max": max(clean) if clean else None,
        "stddev": (
            statistics.pstdev(clean)
            if len(clean) > 1
            else 0.0
            if clean
            else None
        ),
    }


def variant_record(
    calibration: Mapping[str, Any], variant: str
) -> Mapping[str, Any]:
    item = calibration.get("variants", {}).get(variant, {})
    record = item.get("record", {}) if isinstance(item, Mapping) else {}
    return record if isinstance(record, Mapping) else {}


def flatten(
    trade_date: str,
    surface: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    calibration = calibration or {}
    variants = calibration.get("variants", {})
    variants = variants if isinstance(variants, Mapping) else {}
    heston = variant_record(calibration, "heston")
    slv = variant_record(calibration, "heston_slv")
    localvol = variant_record(calibration, "localvol")
    temporal = calibration.get("temporal_scheme", {})
    temporal = temporal if isinstance(temporal, Mapping) else {}
    raw_heston = temporal.get("raw_heston", {})
    raw_heston = raw_heston if isinstance(raw_heston, Mapping) else {}
    row: dict[str, Any] = {
        "date": trade_date,
        "surface_status": surface.get("status"),
        "surface_reason": surface.get("reason"),
        "surface_detail": surface.get("detail"),
        "surface_n_expiries": surface.get("n_expiries"),
        "surface_sha": surface.get("artifact_sha256"),
        "calibration_status": calibration.get("status", "missing"),
        "calibration_elapsed_seconds": calibration.get("elapsed_seconds"),
    }
    for variant in ("localvol", "heston", "heston_slv"):
        item = variants.get(variant, {})
        item = item if isinstance(item, Mapping) else {}
        row[f"{variant}_status"] = item.get("status", "missing")
        row[f"{variant}_elapsed_seconds"] = item.get("elapsed_seconds")
        row[f"{variant}_error"] = item.get("error")
    for parameter in PARAMETERS:
        row[f"heston_{parameter}"] = heston.get(parameter)
        row[f"temporal_raw_heston_{parameter}"] = raw_heston.get(parameter)
        hit = heston.get("bound_hits", {}).get(parameter, {})
        row[f"heston_{parameter}_lower_hit"] = bool(hit.get("lower", False))
        row[f"heston_{parameter}_upper_hit"] = bool(hit.get("upper", False))
    structural_moves = [
        abs(float(row[f"heston_{parameter}"]) - float(row[f"temporal_raw_heston_{parameter}"]))
        / (BOUNDS[parameter][1] - BOUNDS[parameter][0])
        for parameter in ("kappa", "theta", "sigma", "rho")
        if isinstance(row.get(f"heston_{parameter}"), (int, float))
        and isinstance(row.get(f"temporal_raw_heston_{parameter}"), (int, float))
    ]
    row.update(
        {
            "heston_rmse_iv": heston.get("overall_rmse_iv"),
            "heston_feller_ratio": heston.get("feller_ratio"),
            "heston_feller_margin": heston.get("feller_margin"),
            "heston_feller_satisfied": heston.get("feller_satisfied"),
            "heston_nfev": heston.get("nfev"),
            "slv_leverage_min": slv.get("leverage_min"),
            "slv_leverage_mean": slv.get("leverage_mean"),
            "slv_leverage_max": slv.get("leverage_max"),
            "slv_max_negative_mass": slv.get("max_negative_mass"),
            "slv_n_clipped": slv.get("n_clipped"),
            "temporal_scheme": temporal.get("name"),
            "temporal_heston_raw_rmse_iv": raw_heston.get("overall_rmse_iv"),
            "temporal_raw_to_regularized_structural_move": (
                max(structural_moves) if structural_moves else None
            ),
            "temporal_heston_penalty_cost": heston.get(
                "temporal_penalty_cost"
            ),
            "temporal_slv_feller_ratio": temporal.get(
                "slv_heston_feller_ratio"
            ),
            "temporal_slv_feller_satisfied": temporal.get(
                "slv_heston_feller_satisfied"
            ),
            "localvol_min": localvol.get("lv_min"),
            "localvol_max": localvol.get("lv_max"),
        }
    )
    return row


def values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and math.isfinite(float(row[key]))
    ]


def parameter_jumps(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for previous, current in zip(rows, rows[1:]):
        changes = {
            parameter: abs(
                float(current[f"heston_{parameter}"])
                - float(previous[f"heston_{parameter}"])
            )
            / (BOUNDS[parameter][1] - BOUNDS[parameter][0])
            for parameter in PARAMETERS
            if isinstance(current.get(f"heston_{parameter}"), (int, float))
            and isinstance(previous.get(f"heston_{parameter}"), (int, float))
        }
        if changes:
            largest = max(changes, key=changes.get)
            output.append(
                {
                    "previous_date": previous["date"],
                    "date": current["date"],
                    "largest_parameter": largest,
                    "max_normalized_change": changes[largest],
                    "normalized_changes": changes,
                }
            )
    return output


def make_gate(
    name: str, observed: str, status: str, criterion: str, meaning: str
) -> dict[str, str]:
    return {
        "name": name,
        "observed": observed,
        "status": status,
        "criterion": criterion,
        "meaning": meaning,
    }


def worst_status(*statuses: str) -> str:
    severity = {"PASS": 0, "WATCH": 1, "FAIL": 2}
    return max(statuses, key=severity.__getitem__)


def build_report(
    surface_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    *,
    start: str,
    end: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    surfaces = {
        tag: record
        for tag, record in sorted(by_date(surface_payload).items())
        if start <= tag <= end
    }
    if not surfaces:
        raise SystemExit(f"no surface decisions in {start}..{end}")
    calibrations = by_date(calibration_payload)
    daily = [
        flatten(tag, surface, calibrations.get(tag))
        for tag, surface in surfaces.items()
    ]
    admitted = [row for row in daily if row["surface_status"] == "ok"]
    excluded = [row for row in daily if row["surface_status"] != "ok"]
    successful = [
        row for row in admitted if row["calibration_status"] == "ok"
    ]
    failed = [
        row for row in admitted if row["calibration_status"] == "failed"
    ]
    missing = [
        row
        for row in admitted
        if row["calibration_status"] not in ("ok", "failed")
    ]

    coverage = len(successful) / len(admitted) if admitted else 0.0
    feller = values(successful, "heston_feller_ratio")
    n_feller = sum(
        row.get("heston_feller_satisfied") is True for row in successful
    )
    feller_compliance = n_feller / len(successful) if successful else 0.0
    near_feller = sum(item <= 1.01 for item in feller)
    near_feller_rate = near_feller / len(feller) if feller else 0.0
    rmse = distribution(values(successful, "heston_rmse_iv"))
    negative_mass = distribution(values(successful, "slv_max_negative_mass"))
    clipped = sum(int(item) for item in values(successful, "slv_n_clipped"))
    jumps = parameter_jumps(successful)
    jump_stats = distribution(item["max_normalized_change"] for item in jumps)
    temporal_config = calibration_payload.get("config", {})
    temporal_config = (
        temporal_config.get("temporal_scheme")
        if isinstance(temporal_config, Mapping)
        else None
    )
    temporal_enabled = isinstance(temporal_config, Mapping)

    boundary_hits: dict[str, dict[str, Any]] = {}
    for parameter in PARAMETERS:
        lower = sum(
            bool(row[f"heston_{parameter}_lower_hit"]) for row in successful
        )
        upper = sum(
            bool(row[f"heston_{parameter}_upper_hit"]) for row in successful
        )
        either = sum(
            bool(row[f"heston_{parameter}_lower_hit"])
            or bool(row[f"heston_{parameter}_upper_hit"])
            for row in successful
        )
        boundary_hits[parameter] = {
            "lower": lower,
            "upper": upper,
            "either": either,
            "rate": either / len(successful) if successful else 0.0,
        }
    worst_boundary_parameter, worst_boundary = max(
        boundary_hits.items(), key=lambda item: item[1]["rate"]
    )

    coverage_status = (
        "PASS" if coverage == 1.0 else "WATCH" if coverage >= 0.98 else "FAIL"
    )
    feller_status = (
        "PASS"
        if feller_compliance == 1.0
        else "WATCH"
        if feller_compliance >= 0.99
        else "FAIL"
    )
    p95_rmse = float(rmse["p95"]) if rmse["p95"] is not None else math.inf
    rmse_status = (
        "PASS" if p95_rmse <= 0.02 else "WATCH" if p95_rmse <= 0.03 else "FAIL"
    )
    max_negative = (
        float(negative_mass["max"])
        if negative_mass["max"] is not None
        else math.inf
    )
    negative_status = (
        "PASS"
        if max_negative <= 0.005
        else "WATCH"
        if max_negative <= NEGATIVE_MASS_TOLERANCE
        else "FAIL"
    )
    p95_jump = (
        float(jump_stats["p95"]) if jump_stats["p95"] is not None else math.inf
    )
    jump_status = (
        "PASS" if p95_jump <= 0.20 else "WATCH" if p95_jump <= 0.35 else "FAIL"
    )
    gates = [
        make_gate(
            "Calibration coverage",
            f"{len(successful)}/{len(admitted)} ({coverage:.1%})",
            coverage_status,
            "PASS = 100%; WATCH >= 98%",
            "All three governed variants must complete on every admitted surface.",
        ),
        make_gate(
            "Hard-Feller compliance",
            f"{n_feller}/{len(successful)} ({feller_compliance:.1%})",
            feller_status,
            "PASS = 100%; WATCH >= 99%",
            "This is enforced feasibility, not independent fit evidence.",
        ),
        make_gate(
            "Heston fit RMSE p95",
            f"{p95_rmse * 100:.3f} vol points",
            rmse_status,
            "PASS <= 2; WATCH <= 3 vol points",
            "Tail fit quality after imposing hard Feller.",
        ),
        make_gate(
            "Feller-boundary concentration",
            f"{near_feller}/{len(feller)} ({near_feller_rate:.1%})",
            "WATCH" if near_feller_rate > 0.25 else "PASS",
            "WATCH when > 25% have ratio <= 1.01",
            "Frequent boundary solutions indicate constraint-driven parameters.",
        ),
        make_gate(
            "Parameter-bound concentration",
            f"{worst_boundary_parameter}: {worst_boundary['rate']:.1%}",
            "WATCH" if worst_boundary["rate"] > 0.25 else "PASS",
            "WATCH when any box bound is hit on > 25% of dates",
            "Repeated bounds indicate parameter-identifiability pressure.",
        ),
        make_gate(
            "Daily parameter movement p95",
            f"{p95_jump:.1%} of frozen bound span",
            jump_status,
            "PASS <= 20%; WATCH <= 35%",
            "Largest normalized Heston move between adjacent admitted sessions.",
        ),
        make_gate(
            "SLV leverage clipping",
            f"{clipped} clipped nodes",
            "PASS" if clipped == 0 else "WATCH",
            "PASS = 0; otherwise WATCH",
            "Clipping can hide unstable leverage extrapolation.",
        ),
        make_gate(
            "SLV FP negative mass",
            f"max {max_negative:.3e}",
            negative_status,
            "PASS <= 0.005; WATCH <= fail-closed tolerance 0.05",
            "Empirical tripwire; not a positivity proof.",
        ),
    ]
    gate_states = {item["status"] for item in gates}
    overall = (
        "FAIL"
        if "FAIL" in gate_states
        else "WATCH"
        if "WATCH" in gate_states
        else "PASS"
    )
    domain_assessments = {
        "pipeline_reliability": worst_status(coverage_status, feller_status),
        "heston_fit_quality": rmse_status,
        "heston_parameter_stability": worst_status(
            jump_status,
            "WATCH" if near_feller_rate > 0.25 else "PASS",
            "WATCH" if worst_boundary["rate"] > 0.25 else "PASS",
        ),
        "slv_numerical_health": worst_status(
            negative_status,
            "PASS" if clipped == 0 else "WATCH",
        ),
    }

    metrics = {
        "heston_rmse_iv": rmse,
        "heston_feller_ratio": distribution(feller),
        "heston_feller_margin": distribution(
            values(successful, "heston_feller_margin")
        ),
        "heston_nfev": distribution(values(successful, "heston_nfev")),
        "heston_parameters": {
            parameter: distribution(values(successful, f"heston_{parameter}"))
            for parameter in PARAMETERS
        },
        "heston_boundary_hits": boundary_hits,
        "heston_normalized_daily_jump": jump_stats,
        "slv_leverage_min": distribution(
            values(successful, "slv_leverage_min")
        ),
        "slv_leverage_mean": distribution(
            values(successful, "slv_leverage_mean")
        ),
        "slv_leverage_max": distribution(
            values(successful, "slv_leverage_max")
        ),
        "slv_max_negative_mass": negative_mass,
        "slv_total_clipped": clipped,
        "localvol_min": distribution(values(successful, "localvol_min")),
        "localvol_max": distribution(values(successful, "localvol_max")),
        "calibration_elapsed_seconds": distribution(
            values(successful, "calibration_elapsed_seconds")
        ),
    }
    if temporal_enabled:
        metrics["temporal_heston_raw_rmse_iv"] = distribution(
            values(successful, "temporal_heston_raw_rmse_iv")
        )
        metrics["temporal_heston_penalty_cost"] = distribution(
            values(successful, "temporal_heston_penalty_cost")
        )
        metrics["temporal_raw_to_regularized_structural_move"] = distribution(
            values(successful, "temporal_raw_to_regularized_structural_move")
        )
        metrics["temporal_slv_feller_ratio"] = distribution(
            values(successful, "temporal_slv_feller_ratio")
        )
    worst_fits = sorted(
        (
            {
                "date": row["date"],
                "rmse_iv": row["heston_rmse_iv"],
                "feller_ratio": row["heston_feller_ratio"],
                "kappa": row["heston_kappa"],
                "sigma": row["heston_sigma"],
            }
            for row in successful
        ),
        key=lambda item: float(item["rmse_iv"]),
        reverse=True,
    )[:10]
    return {
        "schema_version": 1,
        "report": "mo_daily_calibration_stability",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start, "end": end},
        "overall_assessment": overall,
        "domain_assessments": domain_assessments,
        "coverage": {
            "surface_decisions": len(daily),
            "surface_admitted": len(admitted),
            "surface_excluded": len(excluded),
            "calibration_ok": len(successful),
            "calibration_failed": len(failed),
            "calibration_missing": len(missing),
            "coverage_ratio": coverage,
        },
        "configuration": calibration_payload.get("config"),
        "temporal": {
            "enabled": temporal_enabled,
            "configuration": dict(temporal_config)
            if temporal_enabled
            else None,
            "raw_heston_rmse_iv": metrics.get(
                "temporal_heston_raw_rmse_iv"
            ),
            "heston_penalty_cost": metrics.get(
                "temporal_heston_penalty_cost"
            ),
            "raw_to_regularized_structural_move": metrics.get(
                "temporal_raw_to_regularized_structural_move"
            ),
            "slv_feller_ratio": metrics.get("temporal_slv_feller_ratio"),
        },
        "methodology": {
            "surface": "SABR-smoothed settlement IV with static-arbitrage admission gates",
            "heston": (
                "mo_frozen IV fit with an SLSQP hard-Feller constraint and "
                "structural temporal regularization"
                if temporal_enabled
                else "mo_frozen IV fit with an SLSQP hard-Feller constraint"
            ),
            "heston_bounds": BOUNDS,
            "slv": "forward Fokker-Planck leverage calibration",
            "slv_negative_mass_fail_closed_tolerance": NEGATIVE_MASS_TOLERANCE,
            "caveat": (
                "Operational gates are diagnostics, not statistical proof of "
                "out-of-sample pricing or hedging validity."
            ),
        },
        "gates": gates,
        "metrics": metrics,
        "surface_exclusions": [
            {
                "date": row["date"],
                "reason": row["surface_reason"],
                "detail": row["surface_detail"],
            }
            for row in excluded
        ],
        "calibration_failures": [
            {
                "date": row["date"],
                "status": row["calibration_status"],
                "localvol": row["localvol_status"],
                "heston": row["heston_status"],
                "heston_error": row["heston_error"],
                "heston_slv": row["heston_slv_status"],
                "heston_slv_error": row["heston_slv_error"],
            }
            for row in failed + missing
        ],
        "worst_heston_fit_dates": worst_fits,
        "largest_parameter_jumps": sorted(
            jumps,
            key=lambda item: float(item["max_normalized_change"]),
            reverse=True,
        )[:10],
        "daily_rows": daily,
        "source_hashes": dict(source_hashes),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}g}" if isinstance(value, (int, float)) else str(value)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def line_chart(
    title: str,
    dates: Sequence[str],
    series: Sequence[tuple[str, Sequence[float | None], str]],
    *,
    scale: float = 1.0,
    suffix: str = "",
    thresholds: Sequence[tuple[float, str, str]] = (),
    fixed_range: tuple[float, float] | None = None,
    floor_zero: bool = False,
) -> str:
    width, height = 1120, 280
    left, right, top, bottom = 76, 28, 42, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    observed = [
        float(value) * scale
        for _, points, _ in series
        for value in points
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    observed += [value * scale for value, _, _ in thresholds]
    if not observed:
        return "<p>No successful observations.</p>"
    if fixed_range is None:
        low, high = min(observed), max(observed)
        pad = (high - low) * 0.08 or max(abs(high) * 0.05, 1e-6)
        low, high = low - pad, high + pad
        if floor_zero:
            low = max(0.0, low)
    else:
        low, high = fixed_range

    def x(index: int) -> float:
        return left + plot_w * index / max(1, len(dates) - 1)

    def y(value: float) -> float:
        return top + plot_h * (high - value) / (high - low)

    output = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>',
    ]
    for index in range(5):
        value = low + (high - low) * index / 4
        py = y(value)
        output += [
            f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" class="grid"/>',
            f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" class="axis">{value:.3g}{html.escape(suffix)}</text>',
        ]
    for value, label, color in thresholds:
        py = y(value * scale)
        output += [
            f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="{color}" stroke-dasharray="6 5"/>',
            f'<text x="{width-right-4}" y="{py-5:.2f}" text-anchor="end" fill="{color}" class="axis">{html.escape(label)}</text>',
        ]
    for name, points, color in series:
        coordinates = [
            f"{x(index):.2f},{y(float(value)*scale):.2f}"
            for index, value in enumerate(points)
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        output.append(
            f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="1.7"/>'
        )
    ticks = sorted(
        {0, len(dates) - 1, *[round(i * (len(dates) - 1) / 4) for i in range(1, 4)]}
    )
    for index in ticks:
        label = dates[index]
        output.append(
            f'<text x="{x(index):.2f}" y="{height-16}" text-anchor="middle" class="axis">{label[:4]}-{label[4:6]}-{label[6:]}</text>'
        )
    legend_x = left
    for name, _, color in series:
        output += [
            f'<line x1="{legend_x}" y1="{height-31}" x2="{legend_x+18}" y2="{height-31}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x+23}" y="{height-27}" class="axis">{html.escape(name)}</text>',
        ]
        legend_x += max(90, len(name) * 8 + 35)
    output.append("</svg>")
    return "".join(output)


def render_html(report: Mapping[str, Any], json_name: str, csv_name: str) -> str:
    successful = [
        row
        for row in report["daily_rows"]
        if row["calibration_status"] == "ok"
    ]
    dates = [row["date"] for row in successful]
    metrics, coverage = report["metrics"], report["coverage"]
    colors = {
        "v0": "#55d6be",
        "kappa": "#8ea7ff",
        "theta": "#f5c451",
        "sigma": "#ef8ed4",
        "rho": "#f16f6f",
    }
    normalized_parameters = [
        (
            parameter,
            [
                (
                    (float(row[f"heston_{parameter}"]) - BOUNDS[parameter][0])
                    / (BOUNDS[parameter][1] - BOUNDS[parameter][0])
                    if isinstance(row.get(f"heston_{parameter}"), (int, float))
                    else None
                )
                for row in successful
            ],
            colors[parameter],
        )
        for parameter in PARAMETERS
    ]
    charts = [
        line_chart(
            "Heston IV Fit RMSE",
            dates,
            [("RMSE", [row["heston_rmse_iv"] for row in successful], "#55d6be")],
            scale=100,
            suffix=" vol pts",
            thresholds=[(0.02, "2 vol-point gate", "#f5c451")],
            floor_zero=True,
        ),
        line_chart(
            "Hard-Feller Ratio (log10 scale)",
            dates,
            [
                (
                    "log10(2κθ / σ²)",
                    [
                        math.log10(max(float(row["heston_feller_ratio"]), 1e-12))
                        for row in successful
                    ],
                    "#8ea7ff",
                )
            ],
            thresholds=[(0.0, "ratio = 1 constraint", "#f16f6f")],
        ),
        line_chart(
            "Heston Parameters — Normalized Position in Frozen Bounds",
            dates,
            normalized_parameters,
            fixed_range=(-0.02, 1.02),
            thresholds=[(0.0, "lower", "#778099"), (1.0, "upper", "#778099")],
        ),
        line_chart(
            "SLV Leverage Surface Range",
            dates,
            [
                ("minimum", [row["slv_leverage_min"] for row in successful], "#55d6be"),
                ("mean", [row["slv_leverage_mean"] for row in successful], "#8ea7ff"),
                ("maximum", [row["slv_leverage_max"] for row in successful], "#ef8ed4"),
            ],
            floor_zero=True,
        ),
        line_chart(
            "SLV Forward-FP Negative Mass (log10)",
            dates,
            [
                (
                    "log10 mass",
                    [
                        math.log10(max(float(row["slv_max_negative_mass"]), 1e-12))
                        for row in successful
                    ],
                    "#f5c451",
                )
            ],
            thresholds=[
                (
                    math.log10(NEGATIVE_MASS_TOLERANCE),
                    "fail-closed tolerance",
                    "#f16f6f",
                )
            ],
        ),
    ]
    temporal = report.get("temporal", {})
    temporal_enabled = bool(temporal.get("enabled"))
    if temporal_enabled:
        charts.insert(
            1,
            line_chart(
                "Raw vs Temporally Regularized Heston RMSE",
                dates,
                [
                    (
                        "raw daily fit",
                        [
                            row.get("temporal_heston_raw_rmse_iv")
                            for row in successful
                        ],
                        "#f5c451",
                    ),
                    (
                        "regularized fit",
                        [row.get("heston_rmse_iv") for row in successful],
                        "#55d6be",
                    ),
                ],
                scale=100,
                suffix=" vol pts",
                thresholds=[(0.02, "2 vol-point gate", "#f16f6f")],
                floor_zero=True,
            ),
        )
    gate_rows = [
        (
            item["name"],
            item["observed"],
            item["status"],
            item["criterion"],
            item["meaning"],
        )
        for item in report["gates"]
    ]
    parameter_rows = [
        (
            parameter,
            fmt(item["min"]),
            fmt(item["median"]),
            fmt(item["p95"]),
            fmt(item["max"]),
            fmt(item["stddev"]),
        )
        for parameter, item in metrics["heston_parameters"].items()
    ]
    boundary_rows = [
        (parameter, item["lower"], item["upper"], f"{item['rate']:.1%}")
        for parameter, item in metrics["heston_boundary_hits"].items()
    ]
    jump_rows = [
        (
            item["previous_date"],
            item["date"],
            item["largest_parameter"],
            f"{item['max_normalized_change']:.1%}",
        )
        for item in report["largest_parameter_jumps"]
    ]
    fit_rows = [
        (
            item["date"],
            f"{item['rmse_iv']*100:.3f}",
            f"{item['feller_ratio']:.4f}",
            f"{item['kappa']:.4f}",
            f"{item['sigma']:.4f}",
        )
        for item in report["worst_heston_fit_dates"]
    ]
    exclusion_rows = [
        (item["date"], item.get("reason") or "", item.get("detail") or "")
        for item in report["surface_exclusions"]
    ]
    config = report.get("configuration") or {}
    verdict = report["overall_assessment"]
    domains = report["domain_assessments"]
    domain_summary = " · ".join(
        f"{label}: <span class='{status.lower()}'>{status}</span>"
        for label, status in (
            ("Reliability", domains["pipeline_reliability"]),
            ("Fit", domains["heston_fit_quality"]),
            ("Parameters", domains["heston_parameter_stability"]),
            ("SLV", domains["slv_numerical_health"]),
        )
    )
    temporal_summary = ""
    temporal_section = ""
    if temporal_enabled:
        temporal_config = temporal.get("configuration") or {}
        temporal_summary = (
            f" · Temporal: EWMA span {temporal_config.get('structural_ewma_span', 'n/a')} "
            f"(α={fmt(temporal_config.get('structural_ewma_alpha'), 3)}), "
            f"λ={fmt(temporal_config.get('heston_temporal_regularization'), 3)}"
        )
        raw_rmse = temporal.get("raw_heston_rmse_iv") or {}
        move = temporal.get("raw_to_regularized_structural_move") or {}
        slv_feller = temporal.get("slv_feller_ratio") or {}
        temporal_section = (
            "<h2>Temporal calibration evidence</h2>"
            "<div class='panel'><p>SLV uses today's raw <code>v0</code> with the "
            "updated structural EWMA. Pure Heston applies the penalty only to "
            "<code>κ, θ, σ, ρ</code>; hard Feller remains enforced.</p>"
            + table(
                ["Diagnostic", "Median", "P95", "Maximum"],
                [
                    (
                        "Raw Heston RMSE (vol pts)",
                        fmt((raw_rmse.get("median") or 0.0) * 100, 4),
                        fmt((raw_rmse.get("p95") or 0.0) * 100, 4),
                        fmt((raw_rmse.get("max") or 0.0) * 100, 4),
                    ),
                    (
                        "Raw → regularized structural move",
                        fmt((move.get("median") or 0.0) * 100, 4) + "%",
                        fmt((move.get("p95") or 0.0) * 100, 4) + "%",
                        fmt((move.get("max") or 0.0) * 100, 4) + "%",
                    ),
                    (
                        "SLV EWMA Heston Feller ratio",
                        fmt(slv_feller.get("median"), 5),
                        fmt(slv_feller.get("p95"), 5),
                        fmt(slv_feller.get("max"), 5),
                    ),
                ],
            )
            + "</div>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MO Daily Calibration Stability</title>
<style>
:root{{--bg:#0a0f18;--panel:#111927;--line:#243149;--text:#edf3ff;--muted:#93a1b8;--pass:#55d6be;--watch:#f5c451;--fail:#f16f6f}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(160deg,#08101d,var(--bg));color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}}
main{{width:min(1220px,calc(100% - 36px));margin:auto;padding:38px 0 64px}}h1{{font-size:32px;margin:0}}h2{{margin:30px 0 12px}}p,.small{{color:var(--muted)}}
.eyebrow{{color:#8ea7ff;text-transform:uppercase;letter-spacing:.12em;font-size:12px}}.banner,.card,.panel{{border:1px solid var(--line);background:rgba(17,25,39,.9);border-radius:12px}}
.banner{{display:flex;gap:24px;align-items:center;padding:20px;margin:22px 0}}.verdict{{font-size:25px;font-weight:800}}.pass{{color:var(--pass)}}.watch{{color:var(--watch)}}.fail{{color:var(--fail)}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{padding:16px}}.label{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}.value{{font-size:23px;font-weight:750}}
.panel{{padding:12px;margin:12px 0;overflow:hidden}}svg{{display:block;width:100%;height:auto}}.chart-title{{fill:var(--text);font-size:15px;font-weight:650}}.axis{{fill:var(--muted);font-size:10px}}.grid{{stroke:var(--line)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;min-width:720px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#151f30;color:#b9c7dc;font-size:12px}}tr:last-child td{{border:0}}code,a{{color:#b8c7ff}}
@media(max-width:850px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<div class="eyebrow">QuantArk · CFFEX MO · Daily EOD Calibration</div><h1>One-Year Calibration Stability</h1>
<p>{report['window']['start']} — {report['window']['end']} · generated {html.escape(report['generated_at'])}{temporal_summary}</p>
<section class="banner"><div class="verdict {verdict.lower()}">{verdict}</div><div><strong>{domain_summary}</strong><br><span class="small">The overall verdict is driven by the worst domain. WATCH identifies governance attention; it is not model rejection.</span></div></section>
<section class="cards">
<div class="card"><div class="label">Surface decisions</div><div class="value">{coverage['surface_decisions']}</div><div class="small">{coverage['surface_admitted']} admitted · {coverage['surface_excluded']} excluded</div></div>
<div class="card"><div class="label">Calibration coverage</div><div class="value">{coverage['coverage_ratio']:.1%}</div><div class="small">{coverage['calibration_ok']} OK · {coverage['calibration_failed']} failed · {coverage['calibration_missing']} missing</div></div>
<div class="card"><div class="label">Heston RMSE p95</div><div class="value">{metrics['heston_rmse_iv']['p95']*100:.3f}</div><div class="small">volatility points</div></div>
<div class="card"><div class="label">SLV max negative mass</div><div class="value">{metrics['slv_max_negative_mass']['max']:.2e}</div><div class="small">fail-closed tolerance {NEGATIVE_MASS_TOLERANCE:.2f}</div></div>
</section>
<h2>Stability gates</h2>{table(["Diagnostic","Observed","Status","Gate","Interpretation"],gate_rows)}
<h2>Time-series evidence</h2>{''.join(f'<div class="panel">{chart}</div>' for chart in charts)}
{temporal_section}
<h2>Heston parameter distributions</h2>{table(["Parameter","Min","Median","P95","Max","Std dev"],parameter_rows)}
<h2>Frozen-bound hits</h2>{table(["Parameter","Lower hits","Upper hits","Either rate"],boundary_rows)}
<h2>Largest adjacent-admitted-session movements</h2>{table(["Previous","Date","Parameter","Normalized change"],jump_rows)}
<h2>Worst Heston fits</h2>{table(["Date","RMSE vol pts","Feller ratio","Kappa","Sigma"],fit_rows)}
<h2>Surface exclusions</h2><p>Explicit data-quality decisions; not calibration failures.</p>{table(["Date","Reason","Detail"],exclusion_rows) if exclusion_rows else '<p>No exclusions.</p>'}
<h2>Method and provenance</h2><div class="panel"><p>SABR-smoothed settlement IV with static-arbitrage admission gates. Heston <code>mo_frozen</code>, hard Feller, max evaluations {config.get('heston_max_nfev','n/a')}. SLV η={config.get('slv_eta','n/a')}, steps={config.get('slv_n_steps','n/a')}, x={config.get('slv_n_x','n/a')}, z={config.get('slv_n_z','n/a')}.</p>
<p><a href="{html.escape(json_name)}">Machine-readable JSON evidence</a> · <a href="{html.escape(csv_name)}">Daily CSV</a></p>
<p class="small">Surface SHA-256 <code>{report['source_hashes']['surface_manifest']}</code><br>Calibration SHA-256 <code>{report['source_hashes']['calibration_manifest']}</code></p></div>
<p class="small">Hard-Feller success is partly mechanical because the constraint is enforced. Read it with fit error, boundary concentration, parameter movement, and SLV numerical diagnostics. Historical stability is not proof of out-of-sample pricing or hedging performance.</p>
</main></body></html>"""


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--surface-manifest", type=Path, default=DEFAULT_SURFACE_MANIFEST)
    parser.add_argument("--calibration-manifest", type=Path, default=DEFAULT_CALIBRATION_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.start > args.end:
        raise SystemExit("--start must be <= --end")
    surface_path = args.surface_manifest.resolve()
    calibration_path = args.calibration_manifest.resolve()
    report = build_report(
        read_json(surface_path),
        read_json(calibration_path),
        start=args.start,
        end=args.end,
        source_hashes={
            "surface_manifest": file_sha256(surface_path),
            "calibration_manifest": file_sha256(calibration_path),
        },
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mo_calibration_stability_{args.start}_{args.end}"
    json_path, csv_path, html_path = (
        output_dir / f"{stem}.json",
        output_dir / f"{stem}.csv",
        output_dir / f"{stem}.html",
    )
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, report["daily_rows"])
    html_path.write_text(
        render_html(report, json_path.name, csv_path.name), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "overall_assessment": report["overall_assessment"],
                "coverage": report["coverage"],
                "html": str(html_path),
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )
    return 0 if report["coverage"]["calibration_missing"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

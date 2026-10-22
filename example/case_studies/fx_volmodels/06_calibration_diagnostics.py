"""Stage 06 — compare Heston calibration diagnostics across CFETS dates.

This stage consumes the frozen snapshots and stage-04 calibration artifacts; it
does not recalibrate.  A date enters the stability panel only when its raw node
keys exactly match the requested tenor/pillar universe and its calibration
configuration matches the other included dates.  Every rejected tag is written
to ``exclusions`` with concrete missing/extra nodes or configuration differences.

Example::

    .venv/bin/python example/fx_volmodels/06_calibration_diagnostics.py \
        --tags 20260430 20260515 20260630 20260720 --output-tag study
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _fx_common as fx  # noqa: E402


PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
MODES = ("free", "hard_feller")


def expected_node_keys(universe: str) -> list[list[str]]:
    if universe not in fx.TENOR_SETS:
        raise ValueError(f"unknown universe {universe!r}; choose {sorted(fx.TENOR_SETS)}")
    return [
        [tenor, pillar]
        for tenor in fx.TENOR_SETS[universe]
        for pillar in fx.PILLAR_ORDER
    ]


def _raw_node_keys(snapshot: dict, universe: str) -> list[list[str]]:
    """Extract keys without assuming the snapshot passes full schema validation."""
    allowed = set(fx.TENOR_SETS[universe])
    keys: list[list[str]] = []
    for row in snapshot.get("slices", []):
        tenor = fx.normalise_tenor(row.get("tenor", ""))
        if tenor not in allowed:
            continue
        for quote in row.get("quotes", []):
            pillar = str(quote.get("pillar", ""))
            if pillar:
                keys.append([tenor, pillar])
    order = {
        (tenor, pillar): tenor_index * len(fx.PILLAR_ORDER) + pillar_index
        for tenor_index, tenor in enumerate(fx.TENOR_SETS[universe])
        for pillar_index, pillar in enumerate(fx.PILLAR_ORDER)
    }
    return sorted(keys, key=lambda key: order.get(tuple(key), len(order)))


def _node_difference(actual: Sequence[Sequence[str]], expected: Sequence[Sequence[str]]) -> dict:
    actual_tuples = [tuple(key) for key in actual]
    expected_tuples = [tuple(key) for key in expected]
    actual_set = set(actual_tuples)
    expected_set = set(expected_tuples)
    duplicates = sorted({key for key in actual_tuples if actual_tuples.count(key) > 1})
    return {
        "missing_nodes": [list(key) for key in expected_tuples if key not in actual_set],
        "extra_nodes": [list(key) for key in actual_tuples if key not in expected_set],
        "duplicate_nodes": [list(key) for key in duplicates],
        "ordered_keys_match": actual_tuples == expected_tuples,
    }


def _config_signature(report: dict) -> dict:
    config = report.get("config", {})
    keys = (
        "calibration_target",
        "normalization",
        "method",
        "weight_mode",
        "starts",
        "max_nfev",
        "bounds",
        "feller_modes",
    )
    return {key: config.get(key) for key in keys}


def _mode_summary(entry: dict, mode: str) -> dict:
    mode_entry = entry.get(mode, {})
    best = mode_entry.get("best")
    if not isinstance(best, dict) or best.get("success") is not True:
        raise ValueError(f"{mode} does not contain a successful best fit")
    jacobian = mode_entry.get("jacobian", {})
    scaled = jacobian.get("scaled", {})
    return {
        "params": dict(best["params"]),
        "rmse_vol_points": float(best["rmse_vol_points"]),
        "mae_vol_points": float(best["mae_vol_points"]),
        "max_abs_vol_points": float(best["max_abs_vol_points"]),
        "inside_nonzero_public_band_pct": best.get("inside_nonzero_public_band_pct"),
        "feller_ratio": float(best["feller_ratio"]),
        "feller_margin": float(best["feller_margin"]),
        "optimizer": best.get("optimizer"),
        "nfev": int(best["nfev"]),
        "jacobian_condition": scaled.get("condition_number"),
        "jacobian_rank": scaled.get("numerical_rank"),
        "multistart": dict(mode_entry.get("multistart", {})),
    }


def _load_candidate(tag: str, data_dir: Path, universe: str) -> tuple[dict | None, dict | None]:
    snapshot_path = data_dir / f"cfets_usdcny_snapshot_{tag}.json"
    calibration_path = data_dir / f"cfets_usdcny_heston_{tag}.json"
    exclusion = {
        "tag": tag,
        "snapshot": str(snapshot_path),
        "calibration": str(calibration_path),
    }
    if not snapshot_path.exists():
        return None, {**exclusion, "reason": "missing_snapshot"}
    if not calibration_path.exists():
        return None, {**exclusion, "reason": "missing_calibration_artifact"}

    try:
        raw_snapshot = fx.load_json(snapshot_path)
    except Exception as exc:
        return None, {**exclusion, "reason": f"snapshot_json_error: {type(exc).__name__}: {exc}"}

    expected = expected_node_keys(universe)
    actual = _raw_node_keys(raw_snapshot, universe)
    difference = _node_difference(actual, expected)
    if not difference["ordered_keys_match"] or difference["duplicate_nodes"]:
        return None, {
            **exclusion,
            "trade_date": raw_snapshot.get("trade_date"),
            "reason": "non_comparable_snapshot_nodes",
            **difference,
        }

    try:
        snapshot = fx.load_snapshot(snapshot_path)
    except Exception as exc:
        return None, {
            **exclusion,
            "trade_date": raw_snapshot.get("trade_date"),
            "reason": f"snapshot_schema_error: {type(exc).__name__}: {exc}",
            **difference,
        }
    try:
        calibration = fx.load_json(calibration_path)
    except Exception as exc:
        return None, {
            **exclusion,
            "trade_date": snapshot.get("trade_date"),
            "reason": f"calibration_json_error: {type(exc).__name__}: {exc}",
        }

    entry = calibration.get("universes", {}).get(universe)
    if not isinstance(entry, dict):
        return None, {
            **exclusion,
            "trade_date": snapshot["trade_date"],
            "reason": "calibration_missing_universe",
        }
    calibration_keys = entry.get("node_keys", [])
    calibration_difference = _node_difference(calibration_keys, expected)
    if not calibration_difference["ordered_keys_match"] or calibration_difference["duplicate_nodes"]:
        return None, {
            **exclusion,
            "trade_date": snapshot["trade_date"],
            "reason": "calibration_node_keys_do_not_match_snapshot_universe",
            **calibration_difference,
        }
    if calibration.get("trade_date") != snapshot["trade_date"]:
        return None, {
            **exclusion,
            "trade_date": snapshot["trade_date"],
            "reason": "snapshot_calibration_trade_date_mismatch",
            "calibration_trade_date": calibration.get("trade_date"),
        }

    try:
        modes = {mode: _mode_summary(entry, mode) for mode in MODES}
    except (KeyError, TypeError, ValueError) as exc:
        return None, {
            **exclusion,
            "trade_date": snapshot["trade_date"],
            "reason": f"invalid_calibration_modes: {exc}",
        }
    return {
        "tag": tag,
        "trade_date": snapshot["trade_date"],
        "quote_time": snapshot["quote_time"],
        "node_keys": expected,
        "config_signature": _config_signature(calibration),
        "modes": modes,
        "hard_feller_fit_penalty": dict(entry.get("hard_feller_fit_penalty", {})),
    }, None


def _range(values: Sequence[float | None]) -> list[float] | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return [min(finite), max(finite)] if finite else None


def _stability_summary(included: Sequence[dict], mode: str) -> dict:
    parameters = {
        name: np.array([row["modes"][mode]["params"][name] for row in included], dtype=float)
        for name in PARAMETER_NAMES
    }
    ranges = {name: [float(values.min()), float(values.max())] for name, values in parameters.items()}
    means = {name: float(values.mean()) for name, values in parameters.items()}
    stds = {name: float(values.std(ddof=0)) for name, values in parameters.items()}
    cvs = {
        name: (None if abs(means[name]) <= 1e-15 else abs(stds[name] / means[name]))
        for name in PARAMETER_NAMES
    }
    rho_signs = sorted({int(np.sign(value)) for value in parameters["rho"] if abs(value) > 1e-12})
    return {
        "dates": len(included),
        "parameter_ranges": ranges,
        "parameter_means": means,
        "parameter_std": stds,
        "parameter_cv": cvs,
        "rho_signs": rho_signs,
        "rho_sign_change": len(rho_signs) > 1,
        "rmse_vol_points_range": _range(
            [row["modes"][mode]["rmse_vol_points"] for row in included]
        ),
        "feller_ratio_range": _range(
            [row["modes"][mode]["feller_ratio"] for row in included]
        ),
        "jacobian_condition_range": _range(
            [row["modes"][mode].get("jacobian_condition") for row in included]
        ),
        "minimum_jacobian_rank": min(
            (
                int(row["modes"][mode]["jacobian_rank"])
                for row in included
                if row["modes"][mode].get("jacobian_rank") is not None
            ),
            default=None,
        ),
    }


def _verdicts(included: Sequence[dict], exclusions: Sequence[dict], stability: dict) -> list[dict]:
    if len(included) < 2:
        comparability_status = "insufficient"
    elif exclusions:
        comparability_status = "qualified"
    else:
        comparability_status = "pass"

    free = stability.get("free", {})
    free_feller = free.get("feller_ratio_range")
    if free_feller is None:
        feller_status = "insufficient"
    elif free_feller[1] < 1.0:
        feller_status = "warning"
    else:
        feller_status = "mixed" if free_feller[0] < 1.0 else "pass"

    free_cvs = [value for value in free.get("parameter_cv", {}).values() if value is not None]
    unstable = bool(free.get("rho_sign_change")) or any(value > 0.5 for value in free_cvs)
    stability_status = "insufficient" if len(included) < 2 else ("warning" if unstable else "pass")

    rank = free.get("minimum_jacobian_rank")
    condition_range = free.get("jacobian_condition_range")
    weak_identification = (
        (rank is not None and rank < len(PARAMETER_NAMES))
        or (condition_range is not None and condition_range[1] > 1e6)
        or (rank is not None and condition_range is None)
    )
    identification_status = (
        "insufficient"
        if rank is None and condition_range is None
        else ("warning" if weak_identification else "pass")
    )
    return [
        {
            "name": "cross_date_comparability",
            "status": comparability_status,
            "evidence": {"included_dates": len(included), "excluded_tags": len(exclusions)},
            "interpretation": "Only exact tenor/pillar node sets and matching calibration configs are compared.",
        },
        {
            "name": "free_fit_feller",
            "status": feller_status,
            "evidence": {"feller_ratio_range": free_feller},
            "interpretation": "Feller compliance is reported as a model-risk diagnostic, not a fit-quality proxy.",
        },
        {
            "name": "parameter_stability",
            "status": stability_status,
            "evidence": {
                "parameter_cv": free.get("parameter_cv"),
                "rho_sign_change": free.get("rho_sign_change"),
                "warning_threshold_cv": 0.5,
            },
            "interpretation": "Large cross-date moves or a rho sign change weaken the case for stable parameters.",
        },
        {
            "name": "local_identification",
            "status": identification_status,
            "evidence": {
                "minimum_scaled_jacobian_rank": rank,
                "scaled_condition_range": condition_range,
                "warning_threshold_condition": 1e6,
            },
            "interpretation": "The scaled finite-difference IV Jacobian tests local, not global, identification.",
        },
    ]


def build_cross_date_report(tags: Iterable[str], data_dir: Path, universe: str) -> dict:
    requested = list(tags)
    if not requested:
        raise ValueError("at least one input tag is required")
    expected = expected_node_keys(universe)
    included: list[dict] = []
    exclusions: list[dict] = []
    seen_tags: set[str] = set()
    seen_dates: set[str] = set()
    baseline_signature: dict | None = None

    for tag in requested:
        if tag in seen_tags:
            exclusions.append({"tag": tag, "reason": "duplicate_requested_tag"})
            continue
        seen_tags.add(tag)
        candidate, exclusion = _load_candidate(tag, data_dir, universe)
        if exclusion is not None:
            exclusions.append(exclusion)
            continue
        assert candidate is not None
        if candidate["trade_date"] in seen_dates:
            exclusions.append(
                {
                    "tag": tag,
                    "trade_date": candidate["trade_date"],
                    "reason": "duplicate_trade_date",
                }
            )
            continue
        signature = candidate["config_signature"]
        if baseline_signature is None:
            baseline_signature = signature
        elif signature != baseline_signature:
            exclusions.append(
                {
                    "tag": tag,
                    "trade_date": candidate["trade_date"],
                    "reason": "calibration_config_mismatch",
                    "expected_config": baseline_signature,
                    "actual_config": signature,
                }
            )
            continue
        seen_dates.add(candidate["trade_date"])
        included.append(candidate)

    included.sort(key=lambda row: (row["trade_date"], row["tag"]))
    stability = {
        mode: _stability_summary(included, mode) for mode in MODES
    } if included else {}
    return {
        "schema_version": 1,
        "universe": universe,
        "requested_tags": requested,
        "strict_comparable_node_gate": {
            "expected_node_count": len(expected),
            "expected_node_keys": expected,
            "required_config_signature": baseline_signature,
        },
        "included": included,
        "exclusions": exclusions,
        "stability": stability,
        "verdicts": _verdicts(included, exclusions, stability),
    }


def _plot_cross_date(report: dict, output: Path) -> Path | None:
    if not report["included"]:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["included"]
    x = np.arange(len(rows))
    labels = [row["trade_date"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    panels = (*PARAMETER_NAMES, "rmse_vol_points")
    for axis, field in zip(axes.flat, panels):
        for mode, marker in (("free", "o"), ("hard_feller", "s")):
            if field in PARAMETER_NAMES:
                values = [row["modes"][mode]["params"][field] for row in rows]
            else:
                values = [row["modes"][mode][field] for row in rows]
            axis.plot(x, values, marker=marker, label=mode)
        axis.set_title(field)
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(alpha=0.25)
    handles, labels_legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2)
    fig.suptitle(f"CFETS USD/CNY Heston cross-date diagnostics — {report['universe']}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_artifacts(report: dict, data_dir: Path, output_tag: str) -> dict[str, Path | None]:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"cfets_usdcny_diagnostics_{output_tag}.csv"
    columns = (
        "tag",
        "trade_date",
        "universe",
        "mode",
        *PARAMETER_NAMES,
        "rmse_vol_points",
        "feller_ratio",
        "inside_nonzero_public_band_pct",
        "jacobian_condition",
        "jacobian_rank",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["included"]:
            for mode in MODES:
                values = row["modes"][mode]
                writer.writerow(
                    {
                        "tag": row["tag"],
                        "trade_date": row["trade_date"],
                        "universe": report["universe"],
                        "mode": mode,
                        **values["params"],
                        "rmse_vol_points": values["rmse_vol_points"],
                        "feller_ratio": values["feller_ratio"],
                        "inside_nonzero_public_band_pct": values["inside_nonzero_public_band_pct"],
                        "jacobian_condition": values["jacobian_condition"],
                        "jacobian_rank": values["jacobian_rank"],
                    }
                )

    plot_path = _plot_cross_date(
        report,
        data_dir / "plots" / f"06_calibration_diagnostics_{output_tag}_{report['universe']}.png",
    )
    report["artifacts"] = {
        "csv": str(csv_path),
        "plot": None if plot_path is None else str(plot_path),
    }
    json_path = fx.write_json(data_dir / f"cfets_usdcny_diagnostics_{output_tag}.json", report)
    return {"json": json_path, "csv": csv_path, "plot": plot_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", required=True, help="stage-01/stage-04 artifact tags")
    parser.add_argument("--output-tag", default="latest", help="diagnostics output tag")
    parser.add_argument("--universe", choices=sorted(fx.TENOR_SETS), default="core")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "data",
        help="directory containing tagged snapshots/calibrations; also the default output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional diagnostics artifact directory, separate from --data-dir inputs",
    )
    args = parser.parse_args()

    report = build_cross_date_report(args.tags, args.data_dir, args.universe)
    paths = write_artifacts(report, args.output_dir or args.data_dir, args.output_tag)
    print(paths["json"])
    print(
        f"cross-date gate: {len(report['included'])} included, "
        f"{len(report['exclusions'])} excluded"
    )
    for exclusion in report["exclusions"]:
        print(f"  EXCLUDED {exclusion['tag']}: {exclusion['reason']}")


if __name__ == "__main__":
    main()

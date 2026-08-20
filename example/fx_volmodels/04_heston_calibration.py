"""Stage 04 — calibrate Heston to the raw CFETS five-delta nodes.

The calibration universe is deliberately *not* the SABR grid produced by stage
02.  Each expiry is normalized to forward=1 and contributes exactly the five
public CFETS nodes (10P, 25P, ATM, 25C, 10C).  Free and hard-Feller fits are
reported separately so the constraint is a model-risk stress, not a hidden
replacement calibration.

Examples::

    .venv/bin/python example/fx_volmodels/04_heston_calibration.py --tag sample
    .venv/bin/python example/fx_volmodels/04_heston_calibration.py \
        --tag latest --universe core --weight-mode spread
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

from quantark.volmodels.heston import HestonParams  # noqa: E402


PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
_PARAMETER_SCALES_FLOOR = np.array([1e-4, 0.1, 1e-4, 0.01, 0.1], dtype=float)


def _best_successful(fits: Sequence[dict]) -> dict:
    """Return the best converged fit; never rank a failed solve as usable."""
    successful = [
        fit
        for fit in fits
        if fit.get("success") is True and math.isfinite(float(fit.get("rmse_iv", math.inf)))
    ]
    if not successful:
        errors = [fit.get("error", fit.get("message", "unknown failure")) for fit in fits]
        raise RuntimeError(f"all Heston calibration starts failed: {errors}")
    return min(successful, key=lambda fit: float(fit["rmse_iv"]))


def _params_from_vector(values: Sequence[float]) -> HestonParams:
    return HestonParams(**dict(zip(PARAMETER_NAMES, map(float, values))))


def finite_difference_iv_jacobian(
    nodes: Sequence[dict],
    params: HestonParams,
    *,
    relative_step: float = 1e-4,
) -> dict:
    """Finite-difference the Lewis-consistent model-IV residual Jacobian.

    Both raw and parameter-scaled singular values are recorded.  The raw
    condition number depends on parameter units; the scaled version answers the
    more useful question of sensitivity to comparable relative parameter moves.
    """
    if not nodes:
        raise ValueError("Jacobian requires at least one calibration node")
    if not math.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")

    x = np.array([getattr(params, name) for name in PARAMETER_NAMES], dtype=float)
    lower = np.asarray(fx.HESTON_BOUNDS[0], dtype=float)
    upper = np.asarray(fx.HESTON_BOUNDS[1], dtype=float)
    span = upper - lower
    steps = np.maximum(np.abs(x) * relative_step, span * 1e-6)
    base = fx.heston_model_ivs(nodes, params)
    jacobian = np.empty((len(nodes), len(PARAMETER_NAMES)), dtype=float)
    schemes: list[str] = []

    for index, step in enumerate(steps):
        can_down = x[index] - step >= lower[index]
        can_up = x[index] + step <= upper[index]
        if can_down and can_up:
            x_down = x.copy()
            x_up = x.copy()
            x_down[index] -= step
            x_up[index] += step
            down = fx.heston_model_ivs(nodes, _params_from_vector(x_down))
            up = fx.heston_model_ivs(nodes, _params_from_vector(x_up))
            jacobian[:, index] = (up - down) / (2.0 * step)
            schemes.append("central")
        elif can_up:
            x_up = x.copy()
            x_up[index] += step
            up = fx.heston_model_ivs(nodes, _params_from_vector(x_up))
            jacobian[:, index] = (up - base) / step
            schemes.append("forward")
        elif can_down:
            x_down = x.copy()
            x_down[index] -= step
            down = fx.heston_model_ivs(nodes, _params_from_vector(x_down))
            jacobian[:, index] = (base - down) / step
            schemes.append("backward")
        else:  # Only possible with a zero-width custom bound.
            raise ValueError(f"cannot perturb Heston parameter {PARAMETER_NAMES[index]}")

    scales = np.maximum(np.abs(x), _PARAMETER_SCALES_FLOOR)
    scaled_jacobian = jacobian * scales[None, :]

    def svd_summary(matrix: np.ndarray) -> dict:
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        tolerance = float(np.finfo(float).eps * max(matrix.shape) * largest)
        rank = int(np.sum(singular_values > tolerance))
        condition = None if smallest <= tolerance else float(largest / smallest)
        return {
            "singular_values": [float(value) for value in singular_values],
            "condition_number": condition,
            "numerical_rank": rank,
            "rank_tolerance": tolerance,
        }

    return {
        "method": "finite_difference_of_Lewis_implied_vols",
        "shape": [int(value) for value in jacobian.shape],
        "parameter_order": list(PARAMETER_NAMES),
        "steps": {name: float(step) for name, step in zip(PARAMETER_NAMES, steps)},
        "difference_schemes": dict(zip(PARAMETER_NAMES, schemes)),
        "parameter_scales": {name: float(scale) for name, scale in zip(PARAMETER_NAMES, scales)},
        "raw": svd_summary(jacobian),
        "scaled": svd_summary(scaled_jacobian),
    }


def _serialise_attempts(fits: Sequence[dict]) -> list[dict]:
    """Retain every start outcome without duplicating node-level rows five times."""
    return [{key: value for key, value in fit.items() if key != "rows"} for fit in fits]


def _mode_report(
    snapshot: dict,
    universe: str,
    *,
    hard_feller: bool,
    weight_mode: str,
    starts: int,
    max_nfev: int,
) -> dict:
    fits = fx.calibrate_heston_multistart(
        snapshot,
        tenor_set=universe,
        hard_feller=hard_feller,
        weight_mode=weight_mode,
        starts=starts,
        max_nfev=max_nfev,
    )
    best = _best_successful(fits)
    nodes = fx.iter_nodes(snapshot, universe)
    return {
        "best": best,
        "multistart": fx.summarise_multistart(fits),
        "fits": _serialise_attempts(fits),
        "jacobian": finite_difference_iv_jacobian(
            nodes,
            HestonParams(**best["params"]),
        ),
    }


def build_calibration_report(
    snapshot: dict,
    *,
    tag: str,
    universes: Iterable[str] = ("core", "liquid", "full"),
    weight_mode: str = "equal",
    starts: int = 5,
    max_nfev: int = 500,
) -> dict:
    """Calibrate both Feller policies for each requested raw-node universe."""
    selected = tuple(universes)
    unknown = set(selected) - set(fx.TENOR_SETS)
    if not selected or unknown:
        raise ValueError(f"universes must be non-empty members of {sorted(fx.TENOR_SETS)}")
    if not 1 <= starts <= 5:
        raise ValueError("starts must be between 1 and 5")

    output = {
        "schema_version": 1,
        "tag": tag,
        "trade_date": snapshot["trade_date"],
        "quote_time": snapshot["quote_time"],
        "currency_pair": snapshot["currency_pair"],
        "source_class": snapshot.get("source_class"),
        "config": {
            "calibration_target": "raw_CFETS_five_delta_mid_IVs",
            "normalization": "forward=1, domestic_rate=foreign_rate=0",
            "method": "lewis",
            "weight_mode": weight_mode,
            "starts": starts,
            "max_nfev": max_nfev,
            "bounds": [list(fx.HESTON_BOUNDS[0]), list(fx.HESTON_BOUNDS[1])],
            "feller_modes": ["free", "hard_feller"],
        },
        "universes": {},
        "limitations": list(snapshot.get("limitations", [])),
    }

    for universe in selected:
        nodes = fx.iter_nodes(snapshot, universe)
        expected = len(fx.TENOR_SETS[universe]) * len(fx.PILLAR_ORDER)
        if len(nodes) != expected:
            raise ValueError(
                f"{universe} must contain {expected} raw five-delta nodes, got {len(nodes)}"
            )
        free = _mode_report(
            snapshot,
            universe,
            hard_feller=False,
            weight_mode=weight_mode,
            starts=starts,
            max_nfev=max_nfev,
        )
        hard = _mode_report(
            snapshot,
            universe,
            hard_feller=True,
            weight_mode=weight_mode,
            starts=starts,
            max_nfev=max_nfev,
        )
        free_best = free["best"]
        hard_best = hard["best"]
        free_rmse = float(free_best["rmse_vol_points"])
        hard_rmse = float(hard_best["rmse_vol_points"])
        free_coverage = free_best.get("inside_nonzero_public_band_pct")
        hard_coverage = hard_best.get("inside_nonzero_public_band_pct")
        output["universes"][universe] = {
            "node_count": len(nodes),
            "node_keys": [[node["tenor"], node["pillar"]] for node in nodes],
            "free": free,
            "hard_feller": hard,
            "hard_feller_fit_penalty": {
                "rmse_vol_points": hard_rmse - free_rmse,
                "relative_rmse_pct": (
                    100.0 * (hard_rmse / free_rmse - 1.0) if free_rmse > 0.0 else None
                ),
                "inside_nonzero_public_band_pct": (
                    None
                    if free_coverage is None or hard_coverage is None
                    else float(hard_coverage) - float(free_coverage)
                ),
                "hard_constraint_active": abs(float(hard_best["feller_margin"])) <= 1e-7,
            },
        }
    return output


def _plot_universe(universe: str, report: dict, output: Path) -> Path:
    """Render one small model-vs-market panel per tenor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entry = report["universes"][universe]
    free_rows = entry["free"]["best"]["rows"]
    hard_rows = entry["hard_feller"]["best"]["rows"]
    tenors = list(fx.TENOR_SETS[universe])
    columns = 3
    rows_count = int(math.ceil(len(tenors) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(12, 3.2 * rows_count), squeeze=False)
    for axis, tenor in zip(axes.flat, tenors):
        free_slice = [row for row in free_rows if row["tenor"] == tenor]
        hard_by_pillar = {
            row["pillar"]: row for row in hard_rows if row["tenor"] == tenor
        }
        x = np.array([row["strike_over_forward"] for row in free_slice], dtype=float)
        market = np.array([row["market_iv"] for row in free_slice], dtype=float) * 100.0
        free = np.array([row["model_iv"] for row in free_slice], dtype=float) * 100.0
        hard = np.array(
            [hard_by_pillar[row["pillar"]]["model_iv"] for row in free_slice], dtype=float
        ) * 100.0
        axis.plot(x, market, "o-", label="CFETS mid", color="black", ms=4)
        axis.plot(x, free, "s--", label="Heston free", ms=3)
        axis.plot(x, hard, "^--", label="Heston hard Feller", ms=3)
        axis.set_title(tenor)
        axis.set_xlabel("strike / forward")
        axis.set_ylabel("IV (%)")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(tenors) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"CFETS USD/CNY raw five-node Heston fit — {universe}", y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output


def write_artifacts(report: dict, data_dir: Path) -> dict[str, Path]:
    """Write the stable JSON, best-fit residual CSV, and universe plots."""
    tag = report["tag"]
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = data_dir / "plots"
    csv_path = data_dir / f"cfets_usdcny_heston_residuals_{tag}.csv"
    columns = (
        "tag",
        "trade_date",
        "universe",
        "mode",
        "tenor",
        "pillar",
        "maturity",
        "strike_over_forward",
        "bid_iv",
        "market_iv",
        "ask_iv",
        "model_iv",
        "error_iv",
        "inside_public_band",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for universe, entry in report["universes"].items():
            for mode in ("free", "hard_feller"):
                for row in entry[mode]["best"]["rows"]:
                    writer.writerow(
                        {
                            "tag": tag,
                            "trade_date": report["trade_date"],
                            "universe": universe,
                            "mode": mode,
                            **row,
                        }
                    )

    plot_paths = {
        universe: _plot_universe(
            universe,
            report,
            plot_dir / f"04_heston_fit_{tag}_{universe}.png",
        )
        for universe in report["universes"]
    }
    report["artifacts"] = {
        "residual_csv": str(csv_path),
        "plots": {key: str(value) for key, value in plot_paths.items()},
    }
    json_path = fx.write_json(data_dir / f"cfets_usdcny_heston_{tag}.json", report)
    return {"json": json_path, "csv": csv_path, **{f"plot_{k}": v for k, v in plot_paths.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest", help="input/output artifact tag")
    parser.add_argument(
        "--universe",
        choices=("all", *sorted(fx.TENOR_SETS)),
        default="all",
        help="raw-node tenor universe; all writes core, liquid and full",
    )
    parser.add_argument("--weight-mode", choices=("equal", "spread"), default="equal")
    parser.add_argument("--starts", type=int, default=5)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--snapshot", type=Path, help="override the tagged snapshot path")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "data",
        help="directory containing tagged input data; also the default output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional artifact directory, separate from --data-dir inputs",
    )
    args = parser.parse_args()

    snapshot_path = args.snapshot or args.data_dir / f"cfets_usdcny_snapshot_{args.tag}.json"
    snapshot = fx.load_snapshot(snapshot_path)
    universes = tuple(fx.TENOR_SETS) if args.universe == "all" else (args.universe,)
    report = build_calibration_report(
        snapshot,
        tag=args.tag,
        universes=universes,
        weight_mode=args.weight_mode,
        starts=args.starts,
        max_nfev=args.max_nfev,
    )
    report["input_snapshot"] = str(snapshot_path)
    paths = write_artifacts(report, args.output_dir or args.data_dir)
    print(paths["json"])
    for universe, entry in report["universes"].items():
        free = entry["free"]["best"]
        hard = entry["hard_feller"]["best"]
        penalty = entry["hard_feller_fit_penalty"]
        print(
            f"{universe:6s}: {entry['node_count']:2d} raw nodes | "
            f"free RMSE={free['rmse_vol_points']:.4f} vol-pts "
            f"Feller={free['feller_ratio']:.3f} | "
            f"hard RMSE={hard['rmse_vol_points']:.4f} vol-pts "
            f"penalty={penalty['rmse_vol_points']:+.4f}"
        )


if __name__ == "__main__":
    main()

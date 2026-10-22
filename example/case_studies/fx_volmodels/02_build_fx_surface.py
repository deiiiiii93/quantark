"""Stage 02 — reconstruct CFETS strikes and build a differentiable FX IV grid.

Heston uses the raw five-delta nodes written into this artifact.  Dupire and
SLV use the separately labelled SABR-smoothed, calendar-projected rectangular
grid because numerical differentiation of five isolated strikes is not valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _fx_common as fx  # noqa: E402

from quantark.param.vol.sabr.calibration import calibrate_sabr_slice  # noqa: E402
from quantark.param.vol.sabr.hagan import sabr_implied_vol_black  # noqa: E402


def build_surface(
    snapshot: dict,
    tenor_set: str,
    grid_size: int,
    beta: float,
    grid_domain: str = "intersection",
) -> dict:
    target_slices = fx.selected_slices(snapshot, tenor_set)
    snapshot_slices = fx.selected_slices(snapshot, tuple(row["tenor"] for row in snapshot["slices"]))
    target_tenors = {row["tenor"] for row in target_slices}
    target_indices = [
        index for index, row in enumerate(snapshot_slices) if row["tenor"] in target_tenors
    ]
    support_indices = set()
    if min(target_indices) > 0:
        support_indices.add(min(target_indices) - 1)
    if max(target_indices) + 1 < len(snapshot_slices):
        support_indices.add(max(target_indices) + 1)
    slices = [
        row
        for index, row in enumerate(snapshot_slices)
        if index in support_indices or row["tenor"] in target_tenors
    ]
    if grid_size < 9:
        raise ValueError("grid_size must be at least 9")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("SABR beta must be in [0, 1]")

    raw_ranges = []
    for row in target_slices:
        strikes = [float(q["strike"]) for q in row["quotes"]]
        raw_ranges.append((min(strikes), max(strikes)))
    if grid_domain == "union":
        shared_low = min(low for low, _ in raw_ranges)
        shared_high = max(high for _, high in raw_ranges)
    elif grid_domain == "intersection":
        shared_low = max(low for low, _ in raw_ranges)
        shared_high = min(high for _, high in raw_ranges)
    else:
        raise ValueError("grid_domain must be 'union' or 'intersection'")
    if shared_low >= shared_high:
        raise ValueError(
            "selected tenors have no common observed strike interval; use a narrower tenor universe"
        )
    strikes = np.linspace(shared_low, shared_high, grid_size)

    rows = []
    slice_outputs = []
    for row in slices:
        raw_k = np.array([q["strike"] for q in row["quotes"]], dtype=float)
        raw_iv = np.array([q["mid_iv"] for q in row["quotes"]], dtype=float)
        forward = float(row["forward"])
        maturity = float(row["maturity"])
        log_money = np.log(raw_k / forward)
        weights = np.exp(-0.5 * np.square(log_money / 0.10))
        params = calibrate_sabr_slice(
            F=forward,
            strikes=raw_k,
            T=maturity,
            market_vols=raw_iv,
            beta=beta,
            weights=weights,
            alpha_bounds=(1e-5, 1.0),
            grid_size=31,
            refine=True,
        )
        smooth = np.asarray(
            sabr_implied_vol_black(
                forward,
                strikes,
                np.full_like(strikes, maturity),
                params["alpha"],
                params["beta"],
                params["rho"],
                params["nu"],
                shift=params["shift"],
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(smooth)) or np.any(smooth <= 0.0):
            raise ValueError(f"SABR generated invalid volatility at {row['tenor']}")
        rows.append(smooth)
        slice_outputs.append(
            {
                "tenor": row["tenor"],
                "role": "calibration_target" if row["tenor"] in target_tenors else "boundary_support",
                "maturity": maturity,
                "expiry_date": row["expiry_date"],
                "domestic_rate": float(row["domestic_rate"]),
                # Native FX LV/SLV v1 intentionally rejects a market-forward
                # override.  Use the zero rate implied by CFETS' own published
                # forward so pricing and strike reconstruction share F(T).
                "foreign_rate": float(row.get("pricing_foreign_rate", row["foreign_rate"])),
                "published_foreign_rate": float(row["foreign_rate"]),
                "published_forward_basis_bps": float(row.get("published_forward_basis_bps", 0.0)),
                "forward": forward,
                "raw_quotes": row["quotes"],
                "prepared_extrapolated_nodes": int(
                    np.count_nonzero((strikes < raw_k[0]) | (strikes > raw_k[-1]))
                ),
                "sabr": {key: float(params[key]) for key in ("alpha", "beta", "rho", "nu", "shift", "mse")},
            }
        )

    grid = np.asarray(rows, dtype=float)
    maturities = np.array([float(row["maturity"]) for row in slices], dtype=float)
    total_variance = np.square(grid) * maturities[:, None]
    adjusted_nodes = 0
    for index in range(1, len(maturities)):
        floor = total_variance[index - 1] + 1e-10 * (maturities[index] - maturities[index - 1])
        before = total_variance[index].copy()
        total_variance[index] = np.maximum(total_variance[index], floor)
        adjusted_nodes += int(np.count_nonzero(total_variance[index] > before))
    projected = np.sqrt(total_variance / maturities[:, None])

    raw_errors = []
    support_errors = []
    for row, output in zip(slices, slice_outputs):
        maturity = float(row["maturity"])
        params = output["sabr"]
        for quote in row["quotes"]:
            fitted = float(
                sabr_implied_vol_black(
                    output["forward"],
                    quote["strike"],
                    maturity,
                    params["alpha"],
                    params["beta"],
                    params["rho"],
                    params["nu"],
                    shift=params["shift"],
                )
            )
            error = fitted - float(quote["mid_iv"])
            (raw_errors if output["role"] == "calibration_target" else support_errors).append(error)

    return {
        "schema_version": 1,
        "source": snapshot.get("source"),
        "source_class": snapshot.get("source_class"),
        "trade_date": snapshot["trade_date"],
        "quote_time": snapshot["quote_time"],
        "currency_pair": snapshot["currency_pair"],
        "spot": float(snapshot["spot"]),
        "delta_convention": snapshot.get("delta_convention"),
        "provenance": snapshot.get("provenance", {}),
        "tenor_set": tenor_set,
        "observed_node_count": sum(len(row["quotes"]) for row in target_slices),
        "boundary_support_node_count": sum(
            len(row["quotes"]) for row in slices if row["tenor"] not in target_tenors
        ),
        "strikes": strikes.tolist(),
        "maturities": maturities.tolist(),
        "target_maturities": [
            float(row["maturity"])
            for row in slice_outputs
            if row["role"] == "calibration_target"
        ],
        "iv_grid": projected.tolist(),
        "slices": slice_outputs,
        "surface_preparation": {
            "method": "per-tenor lognormal SABR plus fixed-strike calendar total-variance projection",
            "beta": beta,
            "grid_domain": grid_domain,
            "shared_strike_interval": [shared_low, shared_high],
            "grid_size": grid_size,
            "sabr_extrapolated_nodes": int(
                sum(row["prepared_extrapolated_nodes"] for row in slice_outputs)
            ),
            "calendar_adjusted_nodes": adjusted_nodes,
            "raw_five_delta_sabr_rmse_iv": float(np.sqrt(np.mean(np.square(raw_errors)))),
            "boundary_support_sabr_rmse_iv": (
                float(np.sqrt(np.mean(np.square(support_errors)))) if support_errors else None
            ),
            "boundary_support_tenors": [
                row["tenor"] for row in slice_outputs if row["role"] == "boundary_support"
            ],
            "purpose": "differentiable input for Dupire/SLV; not additional observed liquidity",
        },
        "limitations": list(snapshot.get("limitations", []))
        + [
            "The rectangular grid is model-based interpolation; the optional union domain also requires per-tenor extrapolation.",
            "Adjacent maturity slices are included only as finite-difference boundary support and are excluded from calibration scoring.",
            "Local-vol and SLV results are therefore smoothing-sensitive diagnostics, not direct market observations.",
        ],
    }


def plot_surface(surface: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for row, vols in zip(surface["slices"], surface["iv_grid"]):
        forward = float(row["forward"])
        ax.plot(
            np.asarray(surface["strikes"]) / forward,
            np.asarray(vols) * 100.0,
            label=row["tenor"],
        )
        raw = row["raw_quotes"]
        ax.scatter(
            [q["strike"] / forward for q in raw],
            [q["mid_iv"] * 100.0 for q in raw],
            s=14,
        )
    ax.set_xlabel("strike / forward")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("CFETS USD/CNY — observed five-delta nodes and prepared surface")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tenor-set", choices=sorted(fx.TENOR_SETS), default="core")
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--sabr-beta", type=float, default=1.0)
    parser.add_argument("--grid-domain", choices=("union", "intersection"), default="intersection")
    args = parser.parse_args()

    snapshot = fx.load_snapshot(args.data_dir / f"cfets_usdcny_snapshot_{args.tag}.json")
    surface = build_surface(
        snapshot, args.tenor_set, args.grid_size, args.sabr_beta, args.grid_domain
    )
    output_dir = args.output_dir or args.data_dir
    output = fx.write_json(output_dir / f"cfets_usdcny_surface_{args.tag}.json", surface)
    plot_surface(surface, output_dir / f"plots/02_surface_{args.tag}.png")
    prep = surface["surface_preparation"]
    print(output)
    print(
        f"{surface['observed_node_count']} observed nodes -> {len(surface['maturities'])}x{len(surface['strikes'])} grid; "
        f"raw SABR RMSE={prep['raw_five_delta_sabr_rmse_iv'] * 100:.4f} vol points; "
        f"calendar adjustments={prep['calendar_adjusted_nodes']}"
    )


if __name__ == "__main__":
    main()

"""Stage 03 — build Dupire local volatility and reprice the CFETS surface.

The input is the explicitly prepared stage-02 grid, not additional market
liquidity.  Results report errors both to that smooth target and to the raw
five-delta composite nodes from which it was inferred.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _fx_common as fx  # noqa: E402

from quantark.asset.fx.engine.pde.local_vol_pde_solver import FxLocalVolPDESolver  # noqa: E402
from quantark.asset.fx.product.option import FxVanillaOption  # noqa: E402
from quantark.util.enum import OptionType  # noqa: E402
from quantark.volmodels.black_scholes import implied_vol_call  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol  # noqa: E402


def build_local_vol(surface: dict, *, vol_floor: float | None = None):
    """Build Dupire local vol, keeping admissibility checks on by default."""
    env, iv_grid = fx.build_fx_environment(surface)
    if vol_floor is None:
        local_vol = build_dupire_local_vol(
            iv_grid,
            spot=float(surface["spot"]),
            rate_curve=env.domestic_curve,
            div_yield=env.get_foreign_rate,
        )
        regularization = {"enabled": False, "validate_arbitrage": True, "vol_floor": None}
    else:
        local_vol = build_dupire_local_vol(
            iv_grid,
            spot=float(surface["spot"]),
            rate_curve=env.domestic_curve,
            div_yield=env.get_foreign_rate,
            vol_floor=float(vol_floor),
            validate_arbitrage=False,
        )
        regularization = {
            "enabled": True,
            "validate_arbitrage": False,
            "vol_floor": float(vol_floor),
            "warning": "opt-in diagnostic floor; this is not an arbitrage-free repair",
        }
    return env, local_vol, regularization


def _reprice(
    surface: dict,
    env,
    local_vol,
    *,
    grid_size: int,
    time_steps: int,
    target_stride: int = 1,
) -> dict:
    # USD/CNY vol is only a few percent.  The native solver's wide 4*S domain
    # therefore needs a fine grid; expose the resolution in the artifact.
    solver = FxLocalVolPDESolver(
        grid_size=int(grid_size),
        time_steps=int(time_steps),
        local_vol_surface=local_vol,
    )
    spot = float(surface["spot"])

    def price_one(strike: float, maturity: float, r_dom: float, r_for: float) -> tuple[float, float]:
        option = FxVanillaOption(
            strike=float(strike),
            option_type=OptionType.CALL,
            maturity=float(maturity),
            delivery=float(maturity),
            notional_foreign=1.0,
        )
        price = float(solver.price(option, env))
        model_iv = float(
            implied_vol_call(spot, float(strike), float(maturity), price, r_dom, r_for)
        )
        return price, model_iv

    raw_rows: list[dict] = []
    target_rows: list[dict] = []
    prepared_low, prepared_high = map(float, surface["surface_preparation"]["shared_strike_interval"])
    for slice_index, row in enumerate(surface["slices"]):
        if row.get("role", "calibration_target") != "calibration_target":
            continue
        maturity = float(row["maturity"])
        r_dom = float(row["domestic_rate"])
        r_for = float(row["foreign_rate"])
        forward = float(row["forward"])
        for quote in row["raw_quotes"]:
            price, model_iv = price_one(quote["strike"], maturity, r_dom, r_for)
            market_iv = float(quote["mid_iv"])
            raw_rows.append(
                {
                    "tenor": row["tenor"],
                    "pillar": quote["pillar"],
                    "maturity": maturity,
                    "strike": float(quote["strike"]),
                    "strike_over_forward": float(quote["strike"]) / forward,
                    "market_iv": market_iv,
                    "model_iv": model_iv,
                    "error_iv": model_iv - market_iv,
                    "unit_call_price_cny": price,
                    "inside_prepared_strike_domain": bool(
                        prepared_low <= float(quote["strike"]) <= prepared_high
                    ),
                }
            )
        for strike, target_iv in zip(
            surface["strikes"][::target_stride], surface["iv_grid"][slice_index][::target_stride]
        ):
            price, model_iv = price_one(strike, maturity, r_dom, r_for)
            target_rows.append(
                {
                    "tenor": row["tenor"],
                    "maturity": maturity,
                    "strike": float(strike),
                    "strike_over_forward": float(strike) / forward,
                    "target_iv": float(target_iv),
                    "model_iv": model_iv,
                    "error_iv": model_iv - float(target_iv),
                    "unit_call_price_cny": price,
                }
            )

    def metrics(rows: list[dict]) -> dict:
        errors = np.array([row["error_iv"] for row in rows], dtype=float)
        return {
            "node_count": len(rows),
            "rmse_iv": float(np.sqrt(np.mean(np.square(errors)))),
            "rmse_vol_points": float(np.sqrt(np.mean(np.square(errors))) * 100.0),
            "mae_vol_points": float(np.mean(np.abs(errors)) * 100.0),
            "max_abs_vol_points": float(np.max(np.abs(errors)) * 100.0),
        }

    in_domain_rows = [row for row in raw_rows if row["inside_prepared_strike_domain"]]
    return {
        "pde": {
            "engine": "FxLocalVolPDESolver",
            "grid_size": int(grid_size),
            "time_steps": int(time_steps),
            "target_stride": int(target_stride),
        },
        "prepared_target_fit": {**metrics(target_rows), "rows": target_rows},
        "raw_composite_fit": {
            **metrics(raw_rows),
            "scope": "all raw nodes; nodes outside the prepared domain use flat local-vol extrapolation",
            "acceptance_metric": "in_prepared_domain",
            "in_prepared_domain": metrics(in_domain_rows),
            "outside_prepared_domain_count": len(raw_rows) - len(in_domain_rows),
            "rows": raw_rows,
        },
    }


def run(
    surface: dict,
    *,
    grid_size: int = 1400,
    time_steps: int = 220,
    target_stride: int = 1,
    vol_floor: float | None = None,
) -> dict:
    env, local_vol, regularization = build_local_vol(surface, vol_floor=vol_floor)
    repricing = _reprice(
        surface,
        env,
        local_vol,
        grid_size=grid_size,
        time_steps=time_steps,
        target_stride=target_stride,
    )
    return {
        "schema_version": 1,
        "trade_date": surface["trade_date"],
        "quote_time": surface["quote_time"],
        "currency_pair": surface["currency_pair"],
        "input_contract": {
            "observed": "raw five-delta CFETS composite nodes",
            "dupire_target": surface["surface_preparation"],
        },
        "regularization": regularization,
        "local_vol": {
            "strike_grid": local_vol.strike_grid.tolist(),
            "time_grid": local_vol.time_grid.tolist(),
            "lv_grid": local_vol.lv_grid.tolist(),
            "minimum": float(np.min(local_vol.lv_grid)),
            "maximum": float(np.max(local_vol.lv_grid)),
        },
        **repricing,
        "limitations": list(surface.get("limitations", []))
        + [
            "Expiry and delivery are treated as equal in the native FX local-vol v1 diagnostic.",
            "PDE error and SABR/calendar preparation error are both present in raw-node RMSE.",
        ],
    }


def plot_local_vol(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = result["local_vol"]
    strikes = np.asarray(data["strike_grid"])
    maturities = np.asarray(data["time_grid"])
    local_vol = np.asarray(data["lv_grid"]) * 100.0
    figure, axis = plt.subplots(figsize=(9, 5.5))
    contour = axis.contourf(strikes, maturities, local_vol, levels=20, cmap="viridis")
    figure.colorbar(contour, label="local volatility (%)")
    axis.set_xlabel("USD/CNY strike")
    axis.set_ylabel("maturity (years)")
    axis.set_title("CFETS USD/CNY — Dupire local-volatility target")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pde-grid", type=int, default=1400)
    parser.add_argument("--pde-steps", type=int, default=220)
    parser.add_argument("--vol-floor", type=float)
    parser.add_argument("--fast", action="store_true", help="smaller PDE and sampled prepared grid")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    surface = fx.load_json(args.data_dir / f"cfets_usdcny_surface_{args.tag}.json")
    grid_size, time_steps, stride = (
        (500, 80, 4) if args.fast else (args.pde_grid, args.pde_steps, 1)
    )
    result = run(
        surface,
        grid_size=grid_size,
        time_steps=time_steps,
        target_stride=stride,
        vol_floor=args.vol_floor,
    )
    output = fx.write_json(output_dir / f"cfets_usdcny_localvol_{args.tag}.json", result)
    plot_local_vol(result, output_dir / "plots" / f"03_localvol_{args.tag}.png")
    prepared = result["prepared_target_fit"]
    raw = result["raw_composite_fit"]
    print(output)
    print(
        f"Dupire/PDE RMSE: prepared={prepared['rmse_vol_points']:.4f} vol points "
        f"({prepared['node_count']} nodes), raw in-domain="
        f"{raw['in_prepared_domain']['rmse_vol_points']:.4f} vol points "
        f"({raw['in_prepared_domain']['node_count']} nodes); "
        f"{raw['outside_prepared_domain_count']} raw nodes are extrapolation diagnostics"
    )


if __name__ == "__main__":
    main()

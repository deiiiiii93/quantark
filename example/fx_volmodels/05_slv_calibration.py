"""Stage 05 — calibrate a Heston-SLV leverage surface and reprice vanillas.

Forward Fokker-Planck calibration maps the stage-02 Dupire target onto the
stage-04 Heston dynamics.  Repricing uses the asset-neutral SLV Monte Carlo
kernel with common random numbers and antithetic paths.  A separate native FX
SLV-PDE probe is recorded as a numerical diagnostic, never mixed into the
market-fit acceptance metric.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _fx_common as fx  # noqa: E402

from quantark.asset.fx.engine.pde.heston_slv_pde_solver import FxHestonSLVPDESolver  # noqa: E402
from quantark.asset.fx.product.option import FxVanillaOption  # noqa: E402
from quantark.util.enum import OptionType  # noqa: E402
from quantark.volmodels.black_scholes import bs_vega, implied_vol_call  # noqa: E402
from quantark.volmodels.curves import forward_rates_on_grid  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol  # noqa: E402
from quantark.volmodels.slv import FpCalibrationConfig, calibrate_leverage_surface  # noqa: E402
from quantark.volmodels.slv.slv_mc_kernel import price_european_slv_mc  # noqa: E402


def select_heston_params(report: dict, universe: str, variant: str) -> HestonParams:
    try:
        best = report["universes"][universe][variant]["best"]
    except KeyError as exc:
        raise ValueError(
            f"Heston artifact has no successful {universe}/{variant} best fit"
        ) from exc
    if not best.get("success", True):
        raise ValueError(f"Heston {universe}/{variant} best fit is not successful")
    return HestonParams(**best["params"])


def calibrate_slv(surface: dict, params: HestonParams, *, fast: bool = False):
    env, iv_grid = fx.build_fx_environment(surface)
    local_vol = build_dupire_local_vol(
        iv_grid,
        spot=float(surface["spot"]),
        rate_curve=env.domestic_curve,
        div_yield=env.get_foreign_rate,
    )
    steps = 24 if fast else 40
    t_grid = np.linspace(0.0, max(surface["maturities"]), steps + 1)
    r_fwd = forward_rates_on_grid(env.domestic_curve, t_grid)
    carry_fwd = forward_rates_on_grid(env.foreign_curve, t_grid)
    config = (
        FpCalibrationConfig(
            n_x=61,
            n_z=31,
            n_strike_nodes=21,
            mass_tol=5e-3,
            tol_neg=0.10,
        )
        if fast
        else FpCalibrationConfig(n_x=161, n_z=81, n_strike_nodes=41)
    )
    leverage = calibrate_leverage_surface(
        float(surface["spot"]),
        params,
        local_vol,
        np.diff(t_grid),
        r_fwd,
        carry_fwd,
        eta=1.0,
        fp_config=config,
    )
    return env, local_vol, leverage, t_grid, config


def _mc_reprice(
    surface: dict,
    env,
    local_vol,
    leverage,
    params: HestonParams,
    *,
    paths: int,
    time_steps: int,
    target_stride: int,
    seed: int,
) -> dict:
    spot = float(surface["spot"])
    prepared_low, prepared_high = map(
        float, surface["surface_preparation"]["shared_strike_interval"]
    )
    raw_rows: list[dict] = []
    target_rows: list[dict] = []

    def price_one(
        strike: float,
        maturity: float,
        r_dom: float,
        r_for: float,
        node_seed: int,
    ) -> tuple[float, float, float]:
        t_grid = np.linspace(0.0, maturity, time_steps + 1)
        r_fwd = forward_rates_on_grid(env.domestic_curve, t_grid)
        carry_fwd = forward_rates_on_grid(env.foreign_curve, t_grid)
        price, price_stderr = price_european_slv_mc(
            s0=spot,
            strike=float(strike),
            is_call=True,
            params=params,
            lv_surface=local_vol,
            step_dt=np.diff(t_grid),
            r_fwd=r_fwd,
            carry_fwd=carry_fwd,
            disc_factor=float(env.domestic_curve.get_discount_factor(maturity)),
            eta=1.0,
            num_paths=int(paths),
            num_bins=20,
            seed=int(node_seed),
            return_stderr=True,
            leverage_surface=leverage,
            use_antithetic=True,
        )
        model_iv = float(
            implied_vol_call(spot, float(strike), maturity, price, r_dom, r_for)
        )
        vega = float(bs_vega(spot, float(strike), maturity, model_iv, r_dom, r_for))
        iv_stderr = math.inf if vega <= 0.0 else float(price_stderr / vega)
        return float(price), model_iv, iv_stderr

    for surface_index, row in enumerate(surface["slices"]):
        if row.get("role", "calibration_target") != "calibration_target":
            continue
        maturity = float(row["maturity"])
        r_dom = float(row["domestic_rate"])
        r_for = float(row["foreign_rate"])
        forward = float(row["forward"])
        common_seed = seed + 10_000 * surface_index
        for quote in row["raw_quotes"]:
            price, model_iv, iv_stderr = price_one(
                quote["strike"], maturity, r_dom, r_for, common_seed
            )
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
                    "mc_iv_stderr": iv_stderr,
                    "unit_call_price_cny": price,
                    "inside_prepared_strike_domain": bool(
                        prepared_low <= float(quote["strike"]) <= prepared_high
                    ),
                }
            )
        for strike, target_iv in zip(
            surface["strikes"][::target_stride],
            surface["iv_grid"][surface_index][::target_stride],
        ):
            price, model_iv, iv_stderr = price_one(
                strike, maturity, r_dom, r_for, common_seed
            )
            target_rows.append(
                {
                    "tenor": row["tenor"],
                    "maturity": maturity,
                    "strike": float(strike),
                    "strike_over_forward": float(strike) / forward,
                    "target_iv": float(target_iv),
                    "model_iv": model_iv,
                    "error_iv": model_iv - float(target_iv),
                    "mc_iv_stderr": iv_stderr,
                    "unit_call_price_cny": price,
                }
            )

    def metrics(rows: list[dict]) -> dict:
        errors = np.array([row["error_iv"] for row in rows], dtype=float)
        stderr = np.array([row["mc_iv_stderr"] for row in rows], dtype=float)
        return {
            "node_count": len(rows),
            "rmse_iv": float(np.sqrt(np.mean(np.square(errors)))),
            "rmse_vol_points": float(np.sqrt(np.mean(np.square(errors))) * 100.0),
            "mae_vol_points": float(np.mean(np.abs(errors)) * 100.0),
            "max_abs_vol_points": float(np.max(np.abs(errors)) * 100.0),
            "median_mc_stderr_vol_points": float(np.median(stderr) * 100.0),
        }

    in_domain = [row for row in raw_rows if row["inside_prepared_strike_domain"]]
    return {
        "mc": {
            "kernel": "price_european_slv_mc",
            "paths": int(paths),
            "time_steps": int(time_steps),
            "antithetic": True,
            "common_random_numbers_by_maturity": True,
            "seed": int(seed),
        },
        "prepared_target_fit": {**metrics(target_rows), "rows": target_rows},
        "raw_composite_fit": {
            **metrics(raw_rows),
            "acceptance_metric": "in_prepared_domain",
            "in_prepared_domain": metrics(in_domain),
            "outside_prepared_domain_count": len(raw_rows) - len(in_domain),
            "rows": raw_rows,
        },
    }


def _pde_resolution_probe(surface: dict, env, leverage, params: HestonParams, *, fast: bool) -> dict:
    target_rows = [
        row for row in surface["slices"] if row.get("role", "calibration_target") == "calibration_target"
    ]
    row = min(target_rows, key=lambda item: abs(float(item["maturity"]) - 0.5))
    quote = next(item for item in row["raw_quotes"] if item["pillar"] == "ATM")
    maturity = float(row["maturity"])
    option = FxVanillaOption(
        strike=float(quote["strike"]),
        option_type=OptionType.CALL,
        maturity=maturity,
        delivery=maturity,
        notional_foreign=1.0,
    )
    resolutions = ((60, 30, 40), (100, 50, 70)) if fast else (
        (100, 50, 70),
        (160, 80, 100),
        (240, 100, 150),
    )
    rows = []
    for n_x, n_v, n_t in resolutions:
        solver = FxHestonSLVPDESolver(
            params, leverage, eta=1.0, n_x=n_x, n_v=n_v, n_t=n_t
        )
        price = float(solver.price(option, env))
        model_iv = float(
            implied_vol_call(
                float(surface["spot"]),
                float(quote["strike"]),
                maturity,
                price,
                float(row["domestic_rate"]),
                float(row["foreign_rate"]),
            )
        )
        rows.append(
            {
                "n_x": n_x,
                "n_v": n_v,
                "n_t": n_t,
                "model_iv": model_iv,
                "market_iv": float(quote["mid_iv"]),
                "error_vol_points": (model_iv - float(quote["mid_iv"])) * 100.0,
            }
        )
    return {
        "purpose": "numerical convergence probe only; excluded from SLV market-fit acceptance",
        "tenor": row["tenor"],
        "pillar": "ATM",
        "uniform_grid_warning": (
            "The native uniform ADI domain converges slowly in this 2-4% volatility regime; "
            "use the MC fit above for this example's SLV acceptance metric."
        ),
        "rows": rows,
    }


def run(
    surface: dict,
    heston_report: dict,
    *,
    universe: str = "core",
    variant: str = "free",
    fast: bool = False,
    paths: int | None = None,
    time_steps: int | None = None,
    seed: int = 42,
) -> dict:
    params = select_heston_params(heston_report, universe, variant)
    env, local_vol, leverage, calibration_t_grid, config = calibrate_slv(
        surface, params, fast=fast
    )
    paths = int(paths or (20_000 if fast else 60_000))
    time_steps = int(time_steps or (40 if fast else 100))
    if paths <= 0 or time_steps <= 0:
        raise ValueError("paths and time_steps must be positive")
    repricing = _mc_reprice(
        surface,
        env,
        local_vol,
        leverage,
        params,
        paths=paths,
        time_steps=time_steps,
        target_stride=4,
        seed=seed,
    )
    diagnostics = dict(leverage.diagnostics)
    diagnostics["mass_residual"] = [float(value) for value in diagnostics.get("mass_residual", [])]
    return {
        "schema_version": 1,
        "trade_date": surface["trade_date"],
        "quote_time": surface["quote_time"],
        "currency_pair": surface["currency_pair"],
        "heston_input": {
            "universe": universe,
            "variant": variant,
            "params": {
                "v0": params.v0,
                "kappa": params.kappa,
                "theta": params.theta,
                "sigma": params.sigma,
                "rho": params.rho,
            },
        },
        "leverage_calibration": {
            "method": "forward_fokker_planck",
            "target": surface["surface_preparation"],
            "time_grid": calibration_t_grid[:-1].tolist(),
            "strike_grid": leverage.strike_grid.tolist(),
            "leverage_grid": leverage.leverage_grid.tolist(),
            "minimum": float(np.min(leverage.leverage_grid)),
            "maximum": float(np.max(leverage.leverage_grid)),
            "fp_config": {
                "n_x": config.n_x,
                "n_z": config.n_z,
                "n_strike_nodes": config.n_strike_nodes,
                "mass_tol": config.mass_tol,
                "tol_neg": config.tol_neg,
            },
            "diagnostics": diagnostics,
        },
        **repricing,
        "pde_resolution_probe": _pde_resolution_probe(
            surface, env, leverage, params, fast=fast
        ),
        "limitations": list(surface.get("limitations", []))
        + [
            "SLV inherits the stage-02 smile interpolation and boundary-support assumptions.",
            "Monte Carlo standard errors are sampling diagnostics, not executable quote spreads.",
            "The uniform-grid SLV PDE probe diagnoses engine resolution and is not market evidence.",
        ],
    }


def plot_leverage(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = result["leverage_calibration"]
    strikes = np.asarray(data["strike_grid"])
    times = np.asarray(data["time_grid"])
    leverage = np.asarray(data["leverage_grid"])
    figure, axis = plt.subplots(figsize=(9, 5.5))
    contour = axis.contourf(strikes, times, leverage, levels=20, cmap="coolwarm")
    figure.colorbar(contour, label="leverage L(S,t)")
    axis.set_xlabel("USD/CNY spot")
    axis.set_ylabel("time (years)")
    axis.set_title("CFETS USD/CNY — Heston-SLV leverage surface")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--universe", choices=sorted(fx.TENOR_SETS), default="core")
    parser.add_argument("--heston-variant", choices=("free", "hard_feller"), default="free")
    parser.add_argument("--paths", type=int)
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    surface = fx.load_json(args.data_dir / f"cfets_usdcny_surface_{args.tag}.json")
    heston = fx.load_json(args.data_dir / f"cfets_usdcny_heston_{args.tag}.json")
    result = run(
        surface,
        heston,
        universe=args.universe,
        variant=args.heston_variant,
        fast=args.fast,
        paths=args.paths,
        time_steps=args.time_steps,
        seed=args.seed,
    )
    output = fx.write_json(output_dir / f"cfets_usdcny_slv_{args.tag}.json", result)
    plot_leverage(result, output_dir / "plots" / f"05_slv_{args.tag}.png")
    prepared = result["prepared_target_fit"]
    raw = result["raw_composite_fit"]["in_prepared_domain"]
    print(output)
    print(
        f"SLV/MC RMSE: prepared={prepared['rmse_vol_points']:.4f} vol points; "
        f"raw in-domain={raw['rmse_vol_points']:.4f} vol points; "
        f"median MC stderr={prepared['median_mc_stderr_vol_points']:.4f} vol points"
    )


if __name__ == "__main__":
    main()

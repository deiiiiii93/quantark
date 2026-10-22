"""Stage 08 - Snowball priced under BSM / LV / Heston QE / SLV / SLV QE.

Run: .venv/bin/python example/mo_volmodels/08_snowball_exotic.py [--tag latest|sample]

The product is a 2Y standard autocallable Snowball with principal excluded:
monthly discrete KO, continuous KI, and an annualized KO coupon. MC is available
for all rows, including the standalone SLV QE engine. PDE is reported where a
corresponding solver exists; SLV QE has no PDE analogue because the QE choice is
a Monte Carlo variance discretization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

from quantark.asset.equity.engine.mc import (  # noqa: E402
    HestonSLVQESnowballMCEngine,
    HestonSLVSnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (  # noqa: E402
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import (  # noqa: E402
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import (  # noqa: E402
    SnowballOption,
)
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote  # noqa: E402
from quantark.param.div import TermStructureDividendYield  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402
from quantark.util.enum.engine_enums import MonteCarloMethod  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol  # noqa: E402


def _atm_vol(surface: dict, maturity: float) -> float:
    expiry = min(surface["per_expiry"], key=lambda row: abs(row["T"] - maturity))
    strikes = np.asarray([strike for strike, _ in expiry["points"]], dtype=float)
    vols = np.asarray([vol for _, vol in expiry["points"]], dtype=float)
    return float(vols[np.argmin(np.abs(strikes - expiry["forward"]))])


def _flat_env(s0: float, r: float, q: float, vol: float, valdate) -> PricingEnvironment:
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    maturities = list(np.linspace(0.05, 2.0, 7))
    surface = GridVolSurface(
        strikes,
        maturities,
        np.full((len(maturities), len(strikes)), vol),
    )
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=valdate,
        spot_quote=SpotQuote(spot=s0),
        vol_surface=surface,
        div_yield=TermStructureDividendYield(times=[0.05, 2.0], yields=[q, q]),
    )


def _monthly_observation_dates(maturity: float) -> list[float]:
    obs = {
        round(i / 12.0, 8)
        for i in range(1, int(np.floor(maturity * 12.0)) + 1)
        if i / 12.0 < maturity
    }
    obs.add(float(maturity))
    return sorted(obs)


def _parse_mc_method(value: str) -> MonteCarloMethod:
    try:
        return MonteCarloMethod(value)
    except ValueError:
        valid = ", ".join(method.value for method in MonteCarloMethod)
        raise argparse.ArgumentTypeError(f"invalid MC method {value!r}; choose {valid}")


def _snowball_product(s0: float, maturity: float) -> SnowballOption:
    ko_dates = _monthly_observation_dates(maturity)
    return SnowballOption(
        initial_price=float(s0),
        strike=round(float(s0), 2),
        maturity=float(maturity),
        contract_multiplier=1.0,
        is_reverse=False,
        payoff_config=PayoffConfig(include_principal=False),
        barrier_config=BarrierConfig(
            ko_barrier=round(1.03 * float(s0), 2),
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=ko_dates,
            ki_barrier=round(0.80 * float(s0), 2),
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _result_dict(mc_price: float, mc_engine, pde_price: float | None, notional: float):
    result = mc_engine.get_last_result()
    mc_stderr = mc_engine.get_last_std_error()
    mc_se = float(mc_stderr) if mc_stderr is not None else float("nan")
    out = {
        "mc": float(mc_price),
        "mc_stderr": mc_se,
        "mc_pct_initial": 100.0 * float(mc_price) / notional,
        "pde": None if pde_price is None else float(pde_price),
        "pde_pct_initial": None
        if pde_price is None
        else 100.0 * float(pde_price) / notional,
        "gap": None if pde_price is None else abs(float(mc_price) - float(pde_price)),
        "gap_pct": None
        if pde_price is None or not mc_price
        else 100.0 * abs(float(mc_price) - float(pde_price)) / abs(float(mc_price)),
        "cross_check": None,
        "ko_probability": None,
        "v0_probability": None,
        "v1_probability": None,
        "batches_used": None,
        "paths_used": None,
    }
    if pde_price is not None:
        tol = max(0.05 * abs(float(mc_price)), 3.0 * mc_se) if np.isfinite(mc_se) else 0.05 * abs(float(mc_price))
        out["cross_check"] = bool(abs(float(mc_price) - float(pde_price)) < tol)
    if result is not None:
        out.update(
            {
                "ko_probability": float(result.ko_probability),
                "v0_probability": float(result.v0_probability),
                "v1_probability": float(result.v1_probability),
                "batches_used": result.batches_used,
                "paths_used": int(result.num_paths),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="latest")
    parser.add_argument(
        "--iv-smoothing",
        choices=["sabr", "none"],
        default="sabr",
        help="model target preparation: SABR + calendar projection (default) or raw",
    )
    parser.add_argument(
        "--sabr-beta",
        type=float,
        default=1.0,
        help="SABR beta used by --iv-smoothing sabr; beta=1 is lognormal",
    )
    parser.add_argument(
        "--vol-floor",
        type=float,
        default=None,
        help="diagnostic local-vol floor; intended with --iv-smoothing none",
    )
    parser.add_argument(
        "--mc-method",
        type=_parse_mc_method,
        default=MonteCarloMethod.QUASI,
        choices=list(MonteCarloMethod),
        help="Snowball MC sampling method",
    )
    parser.add_argument("--mc-paths", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--adi-x", type=int, default=None)
    parser.add_argument("--adi-v", type=int, default=None)
    parser.add_argument("--leverage-steps", type=int, default=None)
    parser.add_argument("--leverage-x", type=int, default=None)
    parser.add_argument("--leverage-z", type=int, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use small MC/PDE/leverage grids for tests and quick local demos",
    )
    args = parser.parse_args()

    raw_surface = json.loads((HERE / f"data/mo_iv_surface_{args.tag}.json").read_text())
    surface = mc.prepare_model_surface(
        raw_surface,
        iv_smoothing=args.iv_smoothing,
        sabr_beta=args.sabr_beta,
    )
    calib = json.loads((HERE / f"data/mo_calib_heston_{args.tag}.json").read_text())[
        "params"
    ]

    s0 = float(surface["s0"])
    env, surf, _ = mc.build_env(surface)
    hp = HestonParams(**calib)
    smoothing = surface.get("target_smoothing", {"method": "none"})

    maturity = 2.0
    product = _snowball_product(s0, maturity)
    ko_dates = product.barrier_config.ko_observation_dates or []
    r_t, q_t = float(env.get_rate(maturity)), float(env.get_div_yield(maturity))
    notional = float(product.initial_price * product.contract_multiplier)

    if args.fast:
        mc_paths = args.mc_paths or 2_048
        n_steps = args.time_steps or 24
        grid_size = args.grid_size or 90
        # 0.4.0: the 2D S-axis comes from the shared spatial builder
        # (num_std=8), whose fail-closed resolution check needs >=~224
        # nodes at eps_crit 0.003 — pre-0.4.0 coarse axes are rejected.
        adi_x = args.adi_x or 224
        adi_v = args.adi_v or 24
        leverage_steps = args.leverage_steps or 12
        leverage_x = args.leverage_x or 61
        leverage_z = args.leverage_z or 31
        rqmc_batches = 2
    else:
        mc_paths = args.mc_paths or 16_384
        n_steps = args.time_steps or 96
        grid_size = args.grid_size or 240
        adi_x = args.adi_x or 240
        adi_v = args.adi_v or 36
        leverage_steps = args.leverage_steps or 40
        leverage_x = args.leverage_x or 161
        leverage_z = args.leverage_z or 81
        rqmc_batches = 4

    print(
        "Snowball: "
        f"S0={s0:.1f} K={product.strike:.1f} KO={product.barrier_config.ko_barrier:.1f} "
        f"KI={product.barrier_config.ki_barrier:.1f} T={maturity:.3f} "
        f"({len(ko_dates)} monthly KO obs, continuous KI, r={r_t*100:.2f}% q={q_t*100:.2f}%)"
    )
    if smoothing.get("method") == "sabr_calendar_projected":
        print(
            "SABR-smoothed Snowball target: "
            f"raw-grid RMSE={smoothing['raw_grid_rmse_iv']*100:.3f} vol-pts, "
            f"calendar-adjusted nodes={smoothing['calendar_adjusted_nodes']}"
        )

    lv_kwargs = (
        dict(vol_floor=args.vol_floor, validate_arbitrage=False)
        if args.vol_floor and args.vol_floor > 0
        else {}
    )
    lv = build_dupire_local_vol(
        surf,
        spot=s0,
        rate_curve=env.rate_curve,
        div_yield=env.get_div_yield,
        **lv_kwargs,
    )
    leverage = mc.calibrate_leverage_for(
        env,
        hp,
        lv,
        maturity,
        n_steps=leverage_steps,
        n_x=leverage_x,
        n_z=leverage_z,
    )

    mcp = MCParams(
        num_paths=mc_paths,
        time_steps=n_steps,
        seed=17,
        use_antithetic=False,
        rqmc_min_batches=rqmc_batches,
        rqmc_max_batches=rqmc_batches,
        rqmc_target_std=1e-12,
    )
    # 0.4.0 grid layer: legacy grid_size/time_steps are rejected by the
    # snowball-family solvers, and the fail-closed resolution check refuses
    # under-resolved point overrides — the profile owns the spatial axis.
    pdep = PDEParams(accuracy="fast")

    results = {}

    def record(name: str, mc_engine, pde_engine, product_, environment):
        mc_price = float(mc_engine.price(product_, environment))
        pde_price = None if pde_engine is None else float(pde_engine.price(product_, environment))
        results[name] = _result_dict(mc_price, mc_engine, pde_price, notional)

    atm = _atm_vol(surface, maturity)
    flat_env = _flat_env(s0, r_t, q_t, atm, env.valuation_date)
    record("BSM (flat ATM)", SnowballMCEngine(params=mcp, method=args.mc_method), SnowballPDESolver(pdep), product, flat_env)
    record(
        "Local Vol",
        LocalVolSnowballMCEngine(
            params=mcp,
            method=args.mc_method,
            local_vol_surface=lv,
        ),
        LocalVolSnowballPDESolver(pdep, local_vol_surface=lv),
        product,
        env,
    )
    record(
        "Heston QE",
        QESnowballMCEngine(hp, params=mcp, method=args.mc_method),
        HestonSnowballPDESolver(hp, n_x=adi_x, n_v=adi_v, n_t=n_steps),
        product,
        env,
    )
    record(
        "SLV",
        HestonSLVSnowballMCEngine(
            hp,
            params=mcp,
            method=args.mc_method,
            leverage_surface=leverage,
        ),
        HestonSLVSnowballPDESolver(hp, leverage, n_x=adi_x, n_v=adi_v, n_t=n_steps),
        product,
        env,
    )
    record(
        "SLV QE",
        HestonSLVQESnowballMCEngine(
            hp,
            params=mcp,
            method=args.mc_method,
            leverage_surface=leverage,
        ),
        None,
        product,
        env,
    )

    out = {
        "spec": {
            "s0": s0,
            "strike": float(product.strike),
            "ko_barrier": float(product.barrier_config.ko_barrier),
            "ki_barrier": float(product.barrier_config.ki_barrier),
            "T": maturity,
            "type": "standard snowball",
            "principal": "excluded",
            "include_principal": bool(product.payoff_config.include_principal),
            "ko_monitoring": "monthly discrete",
            "ki_monitoring": "continuous",
            "n_ko_obs": len(ko_dates),
            "ko_rate": float(product.barrier_config.ko_rate),
            "notional": notional,
            "mc_method": args.mc_method.value,
            "mc_paths": mc_paths,
            "n_steps": n_steps,
            "pde_grid_focus": "auto",
            "pde_grid_policy": "2D Heston/SLV PDE centers at KI when present; multi-level critical pinning is opt-in diagnostics",
            "vol_floor": args.vol_floor,
            "iv_smoothing": smoothing,
        },
        "models": results,
    }

    (HERE / f"data/mo_snowball_{args.tag}.json").write_text(json.dumps(out, indent=2))
    for name, row in results.items():
        pde = "n/a" if row["pde"] is None else f"{row['pde_pct_initial']:.3f}%"
        se = row["mc_stderr"]
        se_text = "n/a" if not np.isfinite(se) else f"{100.0 * se / notional:.3f}%"
        status = "" if row["cross_check"] is None else (" OK" if row["cross_check"] else " !!")
        print(
            f"  {name:16s} mc={row['mc_pct_initial']:.3f}% +/- {se_text}  "
            f"pde={pde}{status}  KO={row['ko_probability']:.2%}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    mc_pct = [results[name]["mc_pct_initial"] for name in names]
    mc_err = [
        100.0 * results[name]["mc_stderr"] / notional
        if np.isfinite(results[name]["mc_stderr"])
        else np.nan
        for name in names
    ]
    pde_pct = [
        results[name]["pde_pct_initial"]
        if results[name]["pde_pct_initial"] is not None
        else np.nan
        for name in names
    ]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        mc_pct,
        width,
        yerr=mc_err,
        capsize=3,
        label="MC (+/-1 s.e.)",
        color="#1E3A5F",
    )
    ax.bar(x + width / 2, pde_pct, width, label="PDE", color="#b5432f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("PV (% of initial notional)")
    ax.set_title(
        "Snowball price by model - MO (000852.SH), same vanilla smile\n"
        f"KO={float(product.barrier_config.ko_barrier):.0f} "
        f"KI={float(product.barrier_config.ki_barrier):.0f} T={maturity:.2f}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / f"data/plots/08_snowball_{args.tag}.png", dpi=120)
    plt.close(fig)
    print(f"wrote data/mo_snowball_{args.tag}.json")


if __name__ == "__main__":
    main()

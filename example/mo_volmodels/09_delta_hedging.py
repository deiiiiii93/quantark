"""Stage 09 - Delta-neutral hedge behavior under BSM / LV / Heston / SLV.

Run: .venv/bin/python example/mo_volmodels/09_delta_hedging.py [--tag latest|sample]

This stage keeps the calibrated model surfaces fixed and walks a deterministic
spot scenario for one ATM European call. At each rebalance date it reprices the
remaining option, records model delta/gamma, sets the stock/futures hedge to
``-delta``, and accumulates the residual PnL of a long-option, delta-hedged book:

    dV - delta_previous * dS

The goal is educational: same vanilla smile, same realized spot path, different
model-implied hedge ratios and hedge turnover.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

from quantark.asset.equity.engine.analytical import BlackScholesEngine  # noqa: E402
from quantark.asset.equity.engine.pde import (  # noqa: E402
    HestonPDESolver,
    HestonSLVPDESolver,
    LocalVolPDESolver,
)
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.option import EuropeanVanillaOption  # noqa: E402
from quantark.param import FlatVolSurface  # noqa: E402
from quantark.util.enum import OptionType  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402
from quantark.volmodels.localvol import build_dupire_local_vol  # noqa: E402


def _scenario_spots(s0: float, n_points: int) -> list[float]:
    anchors = np.asarray(
        [1.00, 0.97, 0.92, 0.88, 0.91, 0.95, 1.00, 1.05, 1.02, 0.98, 0.94, 0.99],
        dtype=float,
    )
    x_old = np.linspace(0.0, 1.0, anchors.size)
    x_new = np.linspace(0.0, 1.0, int(n_points))
    return (float(s0) * np.interp(x_new, x_old, anchors)).tolist()


def _bumped_env(env, spot: float):
    out = deepcopy(env)
    out.spot_quote.spot = float(spot)
    return out


def _atm_vol(surface: dict, maturity: float) -> float:
    expiry = min(surface["per_expiry"], key=lambda row: abs(row["T"] - maturity))
    strikes = np.asarray([strike for strike, _ in expiry["points"]], dtype=float)
    vols = np.asarray([vol for _, vol in expiry["points"]], dtype=float)
    return float(vols[np.argmin(np.abs(strikes - expiry["forward"]))])


def _flat_vol_env(env, flat_vol: float):
    out = deepcopy(env)
    out.vol_surface = FlatVolSurface(float(flat_vol))
    return out


def _call(strike: float, tau: float) -> EuropeanVanillaOption:
    return EuropeanVanillaOption(
        strike=float(strike),
        option_type=OptionType.CALL,
        maturity=float(tau),
        contract_multiplier=1.0,
    )


def _model_path(name: str, engine, env, strike: float, times, spots):
    rows = []
    prev_price = None
    prev_spot = None
    prev_delta = None
    cum_pnl = 0.0
    turnover = 0.0
    prev_hedge = None
    t_end = float(times[-1])

    for idx, (t, spot) in enumerate(zip(times, spots)):
        tau = max(t_end - float(t), 1.0 / 365.0)
        product = _call(strike, tau)
        row_env = _bumped_env(env, spot)
        greeks = engine.calculate_greeks(product, row_env)
        price = float(greeks["price"])
        delta = float(greeks["delta"])
        gamma = float(greeks.get("gamma", 0.0))
        hedge_units = -delta

        step_pnl = 0.0
        rebalance = abs(hedge_units) if prev_hedge is None else abs(hedge_units - prev_hedge)
        if idx > 0 and prev_price is not None and prev_spot is not None and prev_delta is not None:
            step_pnl = (price - prev_price) - prev_delta * (float(spot) - prev_spot)
        cum_pnl += step_pnl
        turnover += rebalance

        rows.append(
            {
                "t": float(t),
                "tau": float(tau),
                "spot": float(spot),
                "price": price,
                "delta": delta,
                "gamma": gamma,
                "hedge_units": hedge_units,
                "rebalance_units": float(rebalance),
                "step_pnl": float(step_pnl),
                "cum_pnl": float(cum_pnl),
            }
        )
        prev_price, prev_spot, prev_delta, prev_hedge = price, float(spot), delta, hedge_units

    deltas = np.asarray([r["delta"] for r in rows], dtype=float)
    gammas = np.asarray([r["gamma"] for r in rows], dtype=float)
    return {
        "name": name,
        "initial_price": rows[0]["price"],
        "initial_delta": rows[0]["delta"],
        "initial_gamma": rows[0]["gamma"],
        "terminal_price_on_path": rows[-1]["price"],
        "total_residual_pnl": float(cum_pnl),
        "total_rebalance_units": float(turnover),
        "max_abs_hedge_units": float(np.max(np.abs(deltas))),
        "mean_abs_gamma": float(np.mean(np.abs(gammas))),
        "path": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="latest")
    parser.add_argument(
        "--iv-smoothing",
        choices=["sabr", "none"],
        default="sabr",
        help="model target preparation: SABR + calendar projection (default) or raw",
    )
    parser.add_argument("--sabr-beta", type=float, default=1.0)
    parser.add_argument(
        "--vol-floor",
        type=float,
        default=None,
        help="diagnostic local-vol floor; intended with --iv-smoothing none",
    )
    parser.add_argument("--rebalances", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--adi-x", type=int, default=None)
    parser.add_argument("--adi-v", type=int, default=None)
    parser.add_argument("--leverage-steps", type=int, default=None)
    parser.add_argument("--leverage-x", type=int, default=None)
    parser.add_argument("--leverage-z", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
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

    maturity = float(min(surface["maturities"], key=lambda t: abs(t - 0.70)))
    strike = round(s0, 2)
    flat_vol = _atm_vol(surface, maturity)

    if args.fast:
        rebalances = args.rebalances or 7
        grid_size = args.grid_size or 90
        time_steps = args.time_steps or 24
        adi_x = args.adi_x or 50
        adi_v = args.adi_v or 20
        leverage_steps = args.leverage_steps or 8
        leverage_x = args.leverage_x or 41
        leverage_z = args.leverage_z or 21
    else:
        rebalances = args.rebalances or 12
        grid_size = args.grid_size or 220
        time_steps = args.time_steps or 72
        adi_x = args.adi_x or 120
        adi_v = args.adi_v or 56
        leverage_steps = args.leverage_steps or 32
        leverage_x = args.leverage_x or 121
        leverage_z = args.leverage_z or 61

    print(
        "Delta hedge demo: "
        f"S0={s0:.1f} K={strike:.1f} T={maturity:.3f} "
        f"({rebalances} rebalances, deterministic stress path)"
    )
    if smoothing.get("method") == "sabr_calendar_projected":
        print(
            "SABR-smoothed hedge target: "
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

    times = np.linspace(0.0, 0.75 * maturity, rebalances)
    spots = _scenario_spots(s0, rebalances)
    bsm_env = _flat_vol_env(env, flat_vol)

    engines = {
        "BSM (flat vol)": BlackScholesEngine(),
        "Local Vol": LocalVolPDESolver(
            PDEParams(grid_size=grid_size, time_steps=time_steps),
            local_vol_surface=lv,
        ),
        "Heston": HestonPDESolver(hp, n_x=adi_x, n_v=adi_v, n_t=time_steps),
        "SLV": HestonSLVPDESolver(
            hp,
            leverage,
            n_x=adi_x,
            n_v=adi_v,
            n_t=time_steps,
        ),
    }
    models = {
        name: _model_path(
            name,
            engine,
            bsm_env if name == "BSM (flat vol)" else env,
            strike,
            times,
            spots,
        )
        for name, engine in engines.items()
    }

    out = {
        "spec": {
            "type": "ATM European call delta-neutral hedge",
            "s0": s0,
            "strike": strike,
            "T": maturity,
            "path": "deterministic down-up stress",
            "rebalances": rebalances,
            "flat_vol": flat_vol,
            "flat_vol_source": "ATM implied vol at the hedge option maturity",
            "hedge_rule": "long one option; hold -model_delta units of the underlying after each rebalance",
            "residual_pnl_rule": "step PnL = dV - previous_delta * dS; funding and transaction costs excluded",
            "surface_policy": "calibrated LV surface, Heston params, and SLV leverage are held fixed",
            "iv_smoothing": smoothing,
            "grid_size": grid_size,
            "time_steps": time_steps,
            "adi_x": adi_x,
            "adi_v": adi_v,
            "leverage_steps": leverage_steps,
        },
        "models": models,
    }

    (HERE / f"data/mo_hedging_{args.tag}.json").write_text(json.dumps(out, indent=2))
    for name, row in models.items():
        print(
            f"  {name:10s} delta0={row['initial_delta']:+.4f} "
            f"gamma0={row['initial_gamma']:+.6f} "
            f"turnover={row['total_rebalance_units']:.3f} "
            f"residual_pnl={row['total_residual_pnl']:+.4f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(times, spots, marker="o", color="#1E3A5F")
    axes[0].set_ylabel("spot")
    axes[0].set_title(
        "Delta-neutral hedge behavior - MO (000852.SH)\n"
        "same ATM call and spot path, model-specific delta hedge"
    )

    colors = {
        "BSM (flat vol)": "#6b7280",
        "Local Vol": "#1E3A5F",
        "Heston": "#b5432f",
        "SLV": "#2f7d57",
    }
    for name, row in models.items():
        path = row["path"]
        axes[1].plot(
            [p["t"] for p in path],
            [p["hedge_units"] for p in path],
            marker="o",
            label=name,
            color=colors.get(name),
        )
        axes[2].plot(
            [p["t"] for p in path],
            [p["cum_pnl"] for p in path],
            marker="o",
            label=name,
            color=colors.get(name),
        )
    axes[1].axhline(0.0, color="#666", linewidth=0.8)
    axes[1].set_ylabel("hedge units")
    axes[1].legend(fontsize=8)
    axes[2].axhline(0.0, color="#666", linewidth=0.8)
    axes[2].set_ylabel("cum residual PnL")
    axes[2].set_xlabel("years from hedge inception")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / f"data/plots/09_delta_hedging_{args.tag}.png", dpi=120)
    plt.close(fig)
    print(f"wrote data/mo_hedging_{args.tag}.json")


if __name__ == "__main__":
    main()

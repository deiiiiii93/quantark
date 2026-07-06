"""Stage 05 — calibrate the Heston-SLV leverage surface and reprice.

Run: .venv/bin/python example/mo_volmodels/05_slv_calibration.py [--tag latest|sample]

Stochastic-Local Volatility grafts a deterministic leverage function L(S,t) onto the
calibrated Heston process: the SLV instantaneous vol is L(S,t) * sqrt(v_t). L is solved
(here by the forward Fokker-Planck method) so that E[v_t | S_t=K] * L(K,t)^2 reproduces the
Dupire local variance — i.e. SLV matches the market smile while keeping Heston's stochastic
dynamics. That smile-consistent dynamics is what matters for EXOTICS; for European vanillas
Heston already suffices, and the SLV PDE carries a small discretization bias (see README).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig
from quantark.volmodels.curves import forward_carry_on_grid
from quantark.asset.equity.engine.pde import HestonSLVPDESolver
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.volmodels.black_scholes import implied_vol_call
from quantark.util.enum import OptionType
from quantark.util.exceptions import NumericalError

N_STEPS = 40  # leverage-calibration time grid (modest -> runs in seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--iv-smoothing", choices=["sabr", "none"], default="sabr",
                    help="model target preparation: SABR + calendar projection (default) or raw")
    ap.add_argument("--sabr-beta", type=float, default=1.0,
                    help="SABR beta used by --iv-smoothing sabr; beta=1 is lognormal")
    ap.add_argument("--vol-floor", type=float, default=None,
                    help="opt-in local-vol floor for arbitrage-tainted real surfaces; diagnostic raw mode only")
    args = ap.parse_args()
    raw_surface = json.loads((HERE / f"data/mo_iv_surface_{args.tag}.json").read_text())
    surface = mc.prepare_model_surface(raw_surface, iv_smoothing=args.iv_smoothing,
                                       sabr_beta=args.sabr_beta)
    calib = json.loads((HERE / f"data/mo_calib_heston_{args.tag}.json").read_text())["params"]
    s0 = float(surface["s0"])
    pe = surface["per_expiry"]
    env, surf, _ = mc.build_env(surface)
    hp = HestonParams(**calib)
    smoothing = surface.get("target_smoothing", {"method": "none"})
    if smoothing.get("method") == "sabr_calendar_projected":
        print("SABR-smoothed SLV target: "
              f"raw-grid RMSE={smoothing['raw_grid_rmse_iv']*100:.3f} vol-pts, "
              f"calendar-adjusted nodes={smoothing['calendar_adjusted_nodes']}")

    # Dupire local vol is the SLV calibration TARGET. The default path smooths and
    # calendar-projects the IV surface upstream so Dupire validation can remain enabled.
    if args.vol_floor is None:
        try:
            lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve,
                                        div_yield=env.get_div_yield)
        except NumericalError as e:
            if args.iv_smoothing == "none":
                print(f"ARBITRAGE in raw surface: {e}")
                sys.exit(
                    f"\nRe-run with default SABR smoothing, or diagnostic raw mode: "
                    f"--tag {args.tag} --iv-smoothing none --vol-floor 0.05"
                )
            sys.exit(f"SABR-smoothed surface failed Dupire validation: {e}")
    else:
        print(f"OPT-IN regularization: vol_floor={args.vol_floor:.3f}, validate_arbitrage=False")
        lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve,
                                    div_yield=env.get_div_yield,
                                    vol_floor=args.vol_floor, validate_arbitrage=False)

    # Forward Fokker-Planck leverage calibration on a uniform time grid to the last expiry.
    t_grid = np.linspace(0.0, max(surface["maturities"]), N_STEPS + 1)
    r_fwd = np.array([env.rate_curve.get_forward_rate(t_grid[i], t_grid[i + 1]) for i in range(N_STEPS)])
    carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
    leverage = calibrate_leverage_surface(s0, hp, lv, np.diff(t_grid), r_fwd, carry_fwd,
                                          eta=1.0, fp_config=FpCalibrationConfig(n_x=161, n_z=81))
    lg = leverage.leverage_grid
    print(f"leverage surface {lg.shape}: L in [{lg.min():.3f}, {lg.max():.3f}]  "
          f"(L=1 => pure Heston; L!=1 bends vol toward the market local vol)")

    solver = HestonSLVPDESolver(hp, leverage, eta=1.0, n_x=180, n_v=64, n_t=80)
    per_expiry, sq, raw_sq = [], [], []
    for p in pe:
        T, r, q = p["T"], p["r"], p["q"]
        raw_by_k = {float(k): float(v) for k, v in p.get("raw_points", [])}
        errs, raw_errs = [], []
        for k, mkt_iv in p["points"]:
            price = solver.price(EuropeanVanillaOption(strike=k, option_type=OptionType.CALL, maturity=T), env)
            try:
                model_iv = implied_vol_call(s0, k, T, price, r, q)
            except Exception:
                continue
            errs.append(model_iv - mkt_iv)
            if raw_by_k:
                raw_errs.append(model_iv - raw_by_k[float(k)])
        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            row = {"T": T, "rmse_iv": rmse}
            if raw_errs:
                row["raw_rmse_iv"] = float(np.sqrt(np.mean(np.square(raw_errs))))
                raw_sq.extend(raw_errs)
            per_expiry.append(row)
            sq.extend(errs)
            print(f"  T={T:.3f}  SLV RMSE={rmse*100:.3f} vol-pts")
    overall = float(np.sqrt(np.mean(np.square(sq))))
    raw_overall = float(np.sqrt(np.mean(np.square(raw_sq)))) if raw_sq else None
    out = {"overall_rmse_iv": overall, "per_expiry": per_expiry,
           "leverage_min": float(lg.min()), "leverage_max": float(lg.max()),
           "target_smoothing": smoothing}
    if raw_overall is not None:
        out["raw_overall_rmse_iv"] = raw_overall
    (HERE / f"data/mo_reprice_slv_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"overall SLV RMSE = {overall*100:.3f} vol-pts")
    if raw_overall is not None:
        print(f"overall SLV vs raw quotes = {raw_overall*100:.3f} vol-pts")

    # Leverage surface heatmap L(S, t).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ss = leverage.strike_grid
    Ts = leverage.time_grid
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(Ss, Ts, lg, levels=20, cmap="coolwarm")
    fig.colorbar(c, label="leverage L(S, t)")
    ax.set_xlabel("spot S")
    ax.set_ylabel("time t (years)")
    ax.set_title("Heston-SLV leverage surface — MO (000852.SH)")
    fig.tight_layout()
    fig.savefig(HERE / f"data/plots/05_slv_leverage_{args.tag}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Stage 03 — Dupire local vol: build sigma_LV(K,T) from the market surface, reprice, RMSE.

Run: .venv/bin/python example/mo_volmodels/03_dupire_localvol.py [--tag latest|sample]

The Dupire local-vol surface is the unique deterministic sigma(S,t) that, fed to a
one-factor diffusion, reproduces the entire market IV surface. We build it and then
reprice every OTM strike through the local-vol PDE to measure the discretization error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.black_scholes import implied_vol_call
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.pde import LocalVolPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.util.enum import OptionType
from quantark.util.exceptions import NumericalError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--vol-floor", type=float, default=None,
                    help="opt-in local-vol floor for arbitrage-tainted real surfaces "
                         "(e.g. 0.05). Omit to demand a clean, arbitrage-free input.")
    args = ap.parse_args()
    surface = json.loads((HERE / f"data/mo_iv_surface_{args.tag}.json").read_text())
    env, surf, s0 = mc.build_env(surface)

    # Exact path (default): the builder REJECTS a butterfly/calendar-arbitraging surface
    # rather than silently produce a bad local vol. Raw market quotes frequently violate
    # no-arbitrage at the wings and shortest expiry, so the lecture pipeline explicitly
    # opts into a local-vol floor (--vol-floor). A production desk would instead fit an
    # arbitrage-free smile (e.g. SVI/SABR) upstream and keep validation on.
    if args.vol_floor is None:
        try:
            lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve,
                                        div_yield=env.get_div_yield)
        except NumericalError as e:
            print(f"ARBITRAGE in raw surface: {e}")
            sys.exit(
                "\nThe raw market surface is not arbitrage-free. Re-run with an explicit "
                "opt-in floor, e.g.:\n"
                f"  .venv/bin/python example/mo_volmodels/03_dupire_localvol.py --tag {args.tag} --vol-floor 0.05\n"
                "or arbitrage-free the smile upstream and keep validation on."
            )
    else:
        print(f"OPT-IN regularization: validate_arbitrage=False, vol_floor={args.vol_floor:.3f} "
              "(flooring local vol where the raw surface is inadmissible).")
        lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve,
                                    div_yield=env.get_div_yield,
                                    vol_floor=args.vol_floor, validate_arbitrage=False)

    # Pass the surface we just built (floored or exact) as the solver's prebuilt surface,
    # so it prices against exactly that lv and does not re-derive one per option.
    solver = LocalVolPDESolver(PDEParams(grid_size=300, time_steps=150), local_vol_surface=lv)
    per_expiry, sq_err = [], []
    for pe in surface["per_expiry"]:
        T, r, q = pe["T"], pe["r"], pe["q"]
        errs = []
        for k, mkt_iv in pe["points"]:
            opt = EuropeanVanillaOption(strike=k, option_type=OptionType.CALL, maturity=T)
            price = solver.price(opt, env)
            try:
                model_iv = implied_vol_call(s0, k, T, price, r, q)
            except Exception:
                continue  # outside no-arb band -> excluded, never fabricated
            errs.append(model_iv - mkt_iv)
        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            per_expiry.append({"T": T, "rmse_iv": rmse})
            sq_err.extend(errs)
            print(f"  T={T:.3f}  LV RMSE={rmse*100:.3f} vol-pts  ({len(errs)} strikes)")

    overall = float(np.sqrt(np.mean(np.square(sq_err))))
    (HERE / f"data/mo_reprice_localvol_{args.tag}.json").write_text(
        json.dumps({"per_expiry": per_expiry, "overall_rmse_iv": overall}, indent=2))
    print(f"overall LV RMSE = {overall*100:.3f} vol-pts")

    # Local-vol surface heatmap over the traded strike/maturity box.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ks = np.array(surface["strikes"])
    Ts = np.array(surface["maturities"])
    Z = np.array([[float(lv.local_vol(k, t)) for k in Ks] for t in Ts])
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(Ks, Ts, Z * 100, levels=20, cmap="viridis")
    fig.colorbar(c, label="local vol (%)")
    ax.set_xlabel("strike")
    ax.set_ylabel("maturity T (years)")
    ax.set_title("Dupire local volatility surface — MO (000852.SH)")
    fig.tight_layout()
    fig.savefig(HERE / f"data/plots/03_localvol_surface_{args.tag}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

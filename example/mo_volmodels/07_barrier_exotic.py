"""Stage 07 — an up-and-out call exotic priced MC AND PDE under BSM / LV / Heston / SLV.

Run: .venv/bin/python example/mo_volmodels/07_barrier_exotic.py [--tag latest|sample] [--vol-floor 0.05]

All four models are calibrated to the SAME vanilla smile (stages 02-05), yet a barrier
option — which depends on forward-vol DYNAMICS, not just the terminal smile — prices
differently under each. Two independent numerical methods (MC and PDE) cross-check every
cell where both are available (BSM and LV). Writes data/mo_barrier_{tag}.json + a bar chart.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.mc import (
    BarrierOptionMCEngine, LocalVolBarrierMCEngine, HestonBarrierMCEngine, HestonSLVBarrierMCEngine,
)
from quantark.asset.equity.engine.pde import (
    BarrierPDESolver, LocalVolBarrierPDESolver, HestonBarrierPDESolver, HestonSLVBarrierPDESolver,
)
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.util.enum import OptionType, BarrierType, ObservationType
from datetime import datetime


def _atm_vol(surface, T):
    pe = min(surface["per_expiry"], key=lambda p: abs(p["T"] - T))
    ks = np.array([k for k, _ in pe["points"]]); vs = np.array([v for _, v in pe["points"]])
    return float(vs[np.argmin(np.abs(ks - pe["forward"]))])


def _flat_env(s0, r, q, vol, valdate):
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    surf = GridVolSurface(strikes, list(np.linspace(0.05, 2.0, 7)), np.full((7, 9), vol))
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=valdate,
                              spot_quote=SpotQuote(spot=s0), vol_surface=surf,
                              div_yield=ContinuousDividendYield(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--vol-floor", type=float, default=None)
    args = ap.parse_args()
    surface = json.loads((HERE / f"data/mo_iv_surface_{args.tag}.json").read_text())
    calib = json.loads((HERE / f"data/mo_calib_heston_{args.tag}.json").read_text())["params"]
    s0 = float(surface["s0"])
    env, surf, _ = mc.build_env(surface)
    hp = HestonParams(**calib)

    # Exotic: reverse up-and-out call, K = spot, B = 110% spot. WEEKLY discrete monitoring so
    # MC (hard breach at obs dates) and the PDE (KO injected at the same obs steps) use the
    # identical monitoring convention and cross-check cleanly across every model. (Continuous
    # monitoring would compare a Brownian-bridge MC against per-step PDE injection — two
    # different O(sqrt(dt)) approximations that need not agree at demo resolution.)
    T = float(min(surface["maturities"], key=lambda t: abs(t - 0.45)))
    K = round(s0, 2)
    B = round(1.10 * s0, 2)
    obs_dates = list(np.round(np.arange(1, int(T * 52) + 1) / 52.0, 6))
    obs_dates = [d for d in obs_dates if d <= T] or [T]
    r_T, q_T = float(env.get_rate(T)), float(env.get_div_yield(T))
    prod = BarrierOption(strike=K, maturity=T, option_type=OptionType.CALL, barrier=B,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.DISCRETE,
                         observation_dates=obs_dates)
    print(f"Up-and-out call: S0={s0:.1f} K={K:.1f} B={B:.1f} T={T:.3f} "
          f"({len(obs_dates)} weekly obs, r={r_T*100:.2f}% q={q_T*100:.2f}%)")

    kw = dict(vol_floor=args.vol_floor, validate_arbitrage=False) if args.vol_floor else {}
    lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve, div_yield=env.get_div_yield, **kw)
    leverage = mc.calibrate_leverage_for(env, hp, lv, T)

    mcp = MCParams(num_paths=120_000, time_steps=120, seed=7)
    results = {}

    # BSM (flat ATM vol): analytical + MC + PDE on a flat-vol env.
    atm = _atm_vol(surface, T)
    fenv = _flat_env(s0, r_T, q_T, atm, env.valuation_date)
    results["BSM (flat ATM)"] = {
        "mc": BarrierOptionMCEngine(mcp).price(prod, fenv),
        "pde": BarrierPDESolver(PDEParams(grid_size=400, time_steps=200)).price(prod, fenv),
    }
    # Local Vol: MC + PDE (cross-check on the smile model).
    results["Local Vol"] = {
        "mc": LocalVolBarrierMCEngine(mcp, local_vol_surface=lv).price(prod, env),
        "pde": LocalVolBarrierPDESolver(PDEParams(grid_size=500, time_steps=150), local_vol_surface=lv).price(prod, env),
    }
    # Heston: MC + PDE.
    results["Heston"] = {
        "mc": HestonBarrierMCEngine(hp, mcp).price(prod, env),
        "pde": HestonBarrierPDESolver(hp, n_x=200, n_v=80, n_t=120).price(prod, env),
    }
    # SLV: MC + PDE (leverage surface consumed by both).
    results["SLV"] = {
        "mc": HestonSLVBarrierMCEngine(hp, leverage, mcp).price(prod, env),
        "pde": HestonSLVBarrierPDESolver(hp, leverage, n_x=180, n_v=64, n_t=120).price(prod, env),
    }

    out = {"spec": {"s0": s0, "strike": K, "barrier": B, "T": T, "type": "up-and-out call",
                    "monitoring": "continuous"}, "models": results}
    (HERE / f"data/mo_barrier_{args.tag}.json").write_text(json.dumps(out, indent=2))
    for name, d in results.items():
        cells = "  ".join(f"{k}={v:.3f}" for k, v in d.items())
        print(f"  {name:16s} {cells}")

    # bar chart of the (MC) price per model — the divergence story
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    mc_px = [results[n]["mc"] for n in names]
    pde_px = [results[n].get("pde", np.nan) for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, mc_px, w, label="MC", color="#1E3A5F")
    ax.bar(x + w / 2, pde_px, w, label="PDE", color="#b5432f")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("up-and-out call price"); ax.legend()
    ax.set_title(f"Barrier price by model — MO (000852.SH), same vanilla smile\nK={K:.0f} B={B:.0f} T={T:.2f}")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(HERE / f"data/plots/07_barrier_{args.tag}.png", dpi=120); plt.close(fig)
    print(f"wrote data/mo_barrier_{args.tag}.json")


if __name__ == "__main__":
    main()

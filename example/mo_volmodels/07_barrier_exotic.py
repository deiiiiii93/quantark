"""Stage 07 — an up-and-out call exotic priced MC AND PDE under BSM / LV / Heston / SLV.

Run: .venv/bin/python example/mo_volmodels/07_barrier_exotic.py [--tag latest|sample]

All four models are calibrated to the SAME vanilla smile (stages 02-05), yet a barrier
option — which depends on forward-vol DYNAMICS, not just the terminal smile — prices
differently under each. MC is the reference for the model-divergence story; PDE is reported
as an independent diagnostic cross-check and is flagged where the numerical gap is outside
the noise-aware tolerance. Writes data/mo_barrier_{tag}.json + a bar chart.
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
    ap.add_argument("--iv-smoothing", choices=["sabr", "none"], default="sabr",
                    help="model target preparation: SABR + calendar projection (default) or raw")
    ap.add_argument("--sabr-beta", type=float, default=1.0,
                    help="SABR beta used by --iv-smoothing sabr; beta=1 is lognormal")
    ap.add_argument("--vol-floor", type=float, default=None,
                    help="diagnostic local-vol floor; intended with --iv-smoothing none")
    args = ap.parse_args()
    raw_surface = json.loads((HERE / f"data/mo_iv_surface_{args.tag}.json").read_text())
    surface = mc.prepare_model_surface(raw_surface, iv_smoothing=args.iv_smoothing,
                                       sabr_beta=args.sabr_beta)
    calib = json.loads((HERE / f"data/mo_calib_heston_{args.tag}.json").read_text())["params"]
    s0 = float(surface["s0"])
    env, surf, _ = mc.build_env(surface)
    hp = HestonParams(**calib)
    smoothing = surface.get("target_smoothing", {"method": "none"})
    if smoothing.get("method") == "sabr_calendar_projected":
        print("SABR-smoothed barrier target: "
              f"raw-grid RMSE={smoothing['raw_grid_rmse_iv']*100:.3f} vol-pts, "
              f"calendar-adjusted nodes={smoothing['calendar_adjusted_nodes']}")

    # Exotic: reverse up-and-out call, K = spot, B = 110% spot. WEEKLY discrete monitoring keeps
    # the product convention explicit: MC checks hard breaches at observation dates; PDE injects
    # the KO condition at the same snapped observation nodes. Continuous controls must compare a
    # Brownian-bridge MC with a continuous-domain PDE; endpoint-only MC is a different approximation.
    T = float(min(surface["maturities"], key=lambda t: abs(t - 0.45)))
    K = round(s0, 2)
    B = round(1.10 * s0, 2)
    # Weekly fixings PLUS a final fixing at maturity — the standard barrier convention. (Without
    # the maturity fixing the last ~week is unmonitored; MC honours that, and the PDE now does too
    # after the terminal-observation fix, but the conventional product observes at maturity.)
    obs_dates = sorted({round(d, 6) for d in np.arange(1, int(T * 52) + 1) / 52.0 if d < T} | {float(T)})
    r_T, q_T = float(env.get_rate(T)), float(env.get_div_yield(T))
    prod = BarrierOption(strike=K, maturity=T, option_type=OptionType.CALL, barrier=B,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.DISCRETE,
                         observation_dates=obs_dates)
    print(f"Up-and-out call: S0={s0:.1f} K={K:.1f} B={B:.1f} T={T:.3f} "
          f"({len(obs_dates)} weekly obs, r={r_T*100:.2f}% q={q_T*100:.2f}%)")

    kw = dict(vol_floor=args.vol_floor, validate_arbitrage=False) if args.vol_floor and args.vol_floor > 0 else {}
    lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve, div_yield=env.get_div_yield, **kw)
    leverage = mc.calibrate_leverage_for(env, hp, lv, T)

    # ONE shared time-step count across MC and every PDE. Because both engines snap the weekly
    # observation dates to their OWN time grid, only an identical grid guarantees they knock out
    # on the SAME physical dates — otherwise the "cross-check" silently compares two different
    # contracts (the LV row previously used MC=120 vs PDE=150 steps and disagreed for that reason).
    N_STEPS = 260
    mcp = MCParams(num_paths=150_000, time_steps=N_STEPS, seed=7)
    results = {}

    def record(name, mc_engine, pde_engine, product, environment, rel_tol=0.03, k_sigma=3.0):
        """Price MC + PDE, capture MC standard error, and gate the cross-check on
        |MC - PDE| < max(rel_tol * price, k_sigma * stderr) — noise-aware, not asserted.
        MC is the REFERENCE for this reverse knock-out: smile-model PDE gaps are diagnostics
        of finite barrier grid/time resolution, not the model
        divergence measure used in the lecture."""
        mc_px = float(mc_engine.price(product, environment))
        se = mc_engine.get_last_std_error()
        mc_se = float(se) if se is not None else float("nan")
        pde_px = float(pde_engine.price(product, environment))
        gap = abs(mc_px - pde_px)
        gap_pct = 100.0 * gap / mc_px if mc_px else float("nan")
        tol = max(rel_tol * mc_px, k_sigma * mc_se) if mc_se == mc_se else rel_tol * mc_px
        results[name] = {"mc": mc_px, "mc_stderr": mc_se, "pde": pde_px,
                         "gap": gap, "gap_pct": gap_pct, "cross_check": bool(gap < tol)}

    # BSM (flat ATM vol): MC + PDE on a flat-vol env. Grid converged (grid_size=2000) and time
    # steps SHARED with MC so the residual gap is pure MC noise, not grid/monitoring mismatch.
    atm = _atm_vol(surface, T)
    fenv = _flat_env(s0, r_T, q_T, atm, env.valuation_date)
    record("BSM (flat ATM)", BarrierOptionMCEngine(mcp),
           BarrierPDESolver(PDEParams(grid_size=2000, time_steps=N_STEPS)), prod, fenv)
    # Local Vol: MC + PDE on the same SABR-smoothed Dupire target. This is now a real
    # cross-check rather than the old floored-raw-surface diagnostic.
    record("Local Vol", LocalVolBarrierMCEngine(mcp, local_vol_surface=lv),
           LocalVolBarrierPDESolver(PDEParams(grid_size=500, time_steps=N_STEPS), local_vol_surface=lv), prod, env)
    # Heston: MC + 2D ADI PDE. The vanilla Heston PDE converges to the analytical Heston price,
    # but the discrete barrier surface remains sensitive to the finite (spot, variance) grid; MC
    # is therefore the reference for the divergence story.
    record("Heston", HestonBarrierMCEngine(hp, mcp),
           HestonBarrierPDESolver(hp, n_x=260, n_v=120, n_t=N_STEPS), prod, env)
    # SLV: MC + 2D ADI PDE (leverage surface consumed by both).
    record("SLV", HestonSLVBarrierMCEngine(hp, leverage, mcp),
           HestonSLVBarrierPDESolver(hp, leverage, n_x=260, n_v=120, n_t=N_STEPS), prod, env)

    out = {"spec": {"s0": s0, "strike": K, "barrier": B, "T": T, "type": "up-and-out call",
                    "monitoring": "weekly discrete", "n_obs": len(obs_dates), "obs_per_year": 52,
                    "n_steps": N_STEPS, "vol_floor": args.vol_floor,
                    "iv_smoothing": smoothing}, "models": results}
    (HERE / f"data/mo_barrier_{args.tag}.json").write_text(json.dumps(out, indent=2))
    for name, d in results.items():
        tick = "OK" if d["cross_check"] else "!!"
        print(f"  {name:16s} mc={d['mc']:.3f}±{d['mc_stderr']:.3f}  pde={d['pde']:.3f}  "
              f"gap={d['gap']:.3f} ({d['gap_pct']:.1f}%)  cross-check[{tick}]")

    # bar chart of the (MC) price per model — the divergence story
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    mc_px = [results[n]["mc"] for n in names]
    mc_err = [results[n].get("mc_stderr", np.nan) for n in names]
    pde_px = [results[n].get("pde", np.nan) for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, mc_px, w, yerr=mc_err, capsize=3, label="MC (±1 s.e.)", color="#1E3A5F")
    ax.bar(x + w / 2, pde_px, w, label="PDE", color="#b5432f")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("up-and-out call price"); ax.legend()
    ax.set_title(f"Barrier price by model — MO (000852.SH), same vanilla smile\nK={K:.0f} B={B:.0f} T={T:.2f}")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(HERE / f"data/plots/07_barrier_{args.tag}.png", dpi=120); plt.close(fig)
    print(f"wrote data/mo_barrier_{args.tag}.json")


if __name__ == "__main__":
    main()

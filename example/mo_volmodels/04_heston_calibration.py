"""Stage 04 — calibrate Heston (v0, kappa, theta, sigma, rho) to the OTM chain.

Run: .venv/bin/python example/mo_volmodels/04_heston_calibration.py [--tag latest|sample]

Heston is an arbitrage-free stochastic-volatility model: variance follows a CIR process
dv = kappa(theta - v)dt + sigma sqrt(v) dW, correlated (rho) with spot. Its five params
control the whole smile — v0/theta the level, kappa the term structure, sigma the
convexity (smile), rho the skew. We fit them to market prices with the fast Lewis pricer.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

from quantark.volmodels.heston import (
    HestonParams, MarketOption, calibrate_heston, heston_call_prices_vectorized,
)
from quantark.volmodels.black_scholes import bs_call_price, implied_vol_call


def _atm_iv_shortest(per_expiry):
    """ATM implied vol at the shortest expiry — anchors the Heston level guess."""
    pe0 = min(per_expiry, key=lambda p: p["T"])
    ks = np.array([k for k, _ in pe0["points"]])
    vs = np.array([v for _, v in pe0["points"]])
    return float(vs[np.argmin(np.abs(ks - pe0["forward"]))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--iv-smoothing", choices=["sabr", "none"], default="sabr",
                    help="model target preparation: SABR + calendar projection (default) or raw")
    ap.add_argument("--sabr-beta", type=float, default=1.0,
                    help="SABR beta used by --iv-smoothing sabr; beta=1 is lognormal")
    args = ap.parse_args()
    raw_surface = json.loads((HERE / f"data/mo_iv_surface_{args.tag}.json").read_text())
    surface = mc.prepare_model_surface(raw_surface, iv_smoothing=args.iv_smoothing,
                                       sabr_beta=args.sabr_beta)
    s0 = float(surface["s0"])
    pe = surface["per_expiry"]
    smoothing = surface.get("target_smoothing", {"method": "none"})
    if smoothing.get("method") == "sabr_calendar_projected":
        print("SABR-smoothed Heston target: "
              f"raw-grid RMSE={smoothing['raw_grid_rmse_iv']*100:.3f} vol-pts, "
              f"calendar-adjusted nodes={smoothing['calendar_adjusted_nodes']}")

    # Term-structure r(T), q(T) as callables interpolated from the parity pillars.
    Ts = np.array([p["T"] for p in pe])
    Rs = np.array([p["r"] for p in pe])
    Qs = np.array([p["q"] for p in pe])
    r_of = lambda t: float(np.interp(t, Ts, Rs))  # noqa: E731
    q_of = lambda t: float(np.interp(t, Ts, Qs))  # noqa: E731

    # Market targets: convert each OTM implied vol to a call-equivalent price.
    options = []
    for p in pe:
        for k, iv in p["points"]:
            price = bs_call_price(s0, k, p["T"], iv, p["r"], p["q"])
            options.append(MarketOption(K=k, T=p["T"], price=price))

    # --- Initial guess + bounds (the modeling-judgment call) --------------------------
    # Anchor the variance level to the ATM short-dated vol: v0 ~ theta ~ ATM^2. Start the
    # dynamics at typical equity-index values — moderate mean reversion, strong vol-of-vol
    # to bend the smile, and a negative correlation for the downward skew. The initial
    # guess MUST sit strictly inside `bounds`, or calibrate_heston rejects it.
    atm = _atm_iv_shortest(pe)
    v_level = float(np.clip(atm ** 2, 0.01, 0.2))
    initial = HestonParams(v0=v_level, kappa=2.0, theta=v_level, sigma=0.6, rho=-0.5)
    # We deliberately cap vol-of-vol (sigma <= 0.7, kappa <= 3) and add a Feller penalty
    # (regularize_feller). An unconstrained fit drives sigma huge to chase the short-dated
    # smile, which (a) is a weakly-identified overfit and (b) leaves the variance process
    # deeply Feller-violated (v hits 0), which the downstream ADI PDE prices with a
    # systematic bias. Keeping Feller ~ 1 costs ~0.8 vol-pt of smile fit but makes BOTH the
    # calibration and the Heston/SLV PDE trustworthy. target="iv" weights each strike evenly.
    bounds = ((1e-6, 1e-3, 1e-4, 1e-3, -0.95), (0.5, 3.0, 0.5, 0.7, 0.0))
    # ----------------------------------------------------------------------------------

    result = calibrate_heston(s0=s0, options=options, r=r_of, carry=q_of,
                              initial=initial, bounds=bounds, target="iv", method="lewis",
                              regularize_feller=0.05)
    hp = result.params
    feller = 2.0 * hp.kappa * hp.theta / (hp.sigma ** 2)
    print(f"start ATM={atm*100:.1f}%  ->  v0={hp.v0:.4f} kappa={hp.kappa:.3f} theta={hp.theta:.4f} "
          f"sigma={hp.sigma:.3f} rho={hp.rho:.3f}")
    print(f"Feller 2*kappa*theta/sigma^2 = {feller:.2f}  "
          f"({'satisfied' if feller >= 1 else 'VIOLATED (v can hit 0)'})  "
          f"cost={result.cost:.3e}  success={result.success}")

    # Per-expiry IV RMSE via the semi-analytical Heston pricer, plus model-vs-market smiles.
    per_expiry, sq, raw_sq, smile_rows = [], [], [], []
    for p in pe:
        ks = np.array([k for k, _ in p["points"]])
        mkt = np.array([v for _, v in p["points"]])
        raw_by_k = {float(k): float(v) for k, v in p.get("raw_points", [])}
        hprices = heston_call_prices_vectorized(s0, ks, p["T"], hp, r_of(p["T"]), q_of(p["T"]))
        model_iv, errs, raw_errs = [], [], []
        for k, mv, hpx in zip(ks, mkt, hprices):
            try:
                miv = implied_vol_call(s0, float(k), p["T"], float(hpx), p["r"], p["q"])
            except Exception:
                model_iv.append(np.nan)
                continue
            model_iv.append(miv)
            errs.append(miv - mv)
            if raw_by_k:
                raw_errs.append(miv - raw_by_k[float(k)])
        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            row = {"T": p["T"], "rmse_iv": rmse}
            if raw_errs:
                row["raw_rmse_iv"] = float(np.sqrt(np.mean(np.square(raw_errs))))
                raw_sq.extend(raw_errs)
            per_expiry.append(row)
            sq.extend(errs)
            print(f"  T={p['T']:.3f}  Heston RMSE={rmse*100:.3f} vol-pts")
        smile_rows.append((f"target T={p['T']:.2f}", ks.tolist(), mkt.tolist()))
        smile_rows.append((f"Heston T={p['T']:.2f}", ks.tolist(), model_iv))

    overall = float(np.sqrt(np.mean(np.square(sq))))
    raw_overall = float(np.sqrt(np.mean(np.square(raw_sq)))) if raw_sq else None
    out = {
        "params": {"v0": hp.v0, "kappa": hp.kappa, "theta": hp.theta, "sigma": hp.sigma, "rho": hp.rho},
        "feller": feller, "cost": result.cost, "success": result.success,
        "overall_rmse_iv": overall, "per_expiry": per_expiry,
        "target_smoothing": smoothing,
    }
    if raw_overall is not None:
        out["raw_overall_rmse_iv"] = raw_overall
    (HERE / f"data/mo_calib_heston_{args.tag}.json").write_text(json.dumps(out, indent=2))
    mc.plot_smiles(smile_rows, HERE / f"data/plots/04_heston_fit_{args.tag}.png",
                   title="Heston fit vs calibration target — MO (000852.SH)")
    print(f"overall Heston RMSE = {overall*100:.3f} vol-pts")
    if raw_overall is not None:
        print(f"overall Heston vs raw quotes = {raw_overall*100:.3f} vol-pts")


if __name__ == "__main__":
    main()

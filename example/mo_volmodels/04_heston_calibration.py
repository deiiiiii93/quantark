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
    args = ap.parse_args()
    surface = json.loads((HERE / f"data/mo_iv_surface_{args.tag}.json").read_text())
    s0 = float(surface["s0"])
    pe = surface["per_expiry"]

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
    initial = HestonParams(v0=v_level, kappa=3.0, theta=v_level, sigma=1.0, rho=-0.4)
    # kappa is capped at a realistic 10: the smile only weakly identifies kappa vs sigma
    # (raising the cap chases marginal RMSE into implausible mean-reversion), so we
    # regularize rather than overfit. target="iv" weights each strike's smile evenly,
    # which avoids the price-weighted degeneracy that pinned theta at its bound.
    bounds = ((1e-6, 1e-3, 1e-4, 1e-3, -0.95), (0.5, 10.0, 0.5, 3.0, 0.0))
    # ----------------------------------------------------------------------------------

    result = calibrate_heston(s0=s0, options=options, r=r_of, carry=q_of,
                              initial=initial, bounds=bounds, target="iv", method="lewis")
    hp = result.params
    feller = 2.0 * hp.kappa * hp.theta / (hp.sigma ** 2)
    print(f"start ATM={atm*100:.1f}%  ->  v0={hp.v0:.4f} kappa={hp.kappa:.3f} theta={hp.theta:.4f} "
          f"sigma={hp.sigma:.3f} rho={hp.rho:.3f}")
    print(f"Feller 2*kappa*theta/sigma^2 = {feller:.2f}  "
          f"({'satisfied' if feller >= 1 else 'VIOLATED (v can hit 0)'})  "
          f"cost={result.cost:.3e}  success={result.success}")

    # Per-expiry IV RMSE via the semi-analytical Heston pricer, plus model-vs-market smiles.
    per_expiry, sq, smile_rows = [], [], []
    for p in pe:
        ks = np.array([k for k, _ in p["points"]])
        mkt = np.array([v for _, v in p["points"]])
        hprices = heston_call_prices_vectorized(s0, ks, p["T"], hp, r_of(p["T"]), q_of(p["T"]))
        model_iv, errs = [], []
        for k, mv, hpx in zip(ks, mkt, hprices):
            try:
                miv = implied_vol_call(s0, float(k), p["T"], float(hpx), p["r"], p["q"])
            except Exception:
                model_iv.append(np.nan)
                continue
            model_iv.append(miv)
            errs.append(miv - mv)
        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            per_expiry.append({"T": p["T"], "rmse_iv": rmse})
            sq.extend(errs)
            print(f"  T={p['T']:.3f}  Heston RMSE={rmse*100:.3f} vol-pts")
        smile_rows.append((f"mkt T={p['T']:.2f}", ks.tolist(), mkt.tolist()))
        smile_rows.append((f"Heston T={p['T']:.2f}", ks.tolist(), model_iv))

    overall = float(np.sqrt(np.mean(np.square(sq))))
    out = {
        "params": {"v0": hp.v0, "kappa": hp.kappa, "theta": hp.theta, "sigma": hp.sigma, "rho": hp.rho},
        "feller": feller, "cost": result.cost, "success": result.success,
        "overall_rmse_iv": overall, "per_expiry": per_expiry,
    }
    (HERE / f"data/mo_calib_heston_{args.tag}.json").write_text(json.dumps(out, indent=2))
    mc.plot_smiles(smile_rows, HERE / f"data/plots/04_heston_fit_{args.tag}.png",
                   title="Heston fit vs market — MO (000852.SH)")
    print(f"overall Heston RMSE = {overall*100:.3f} vol-pts")


if __name__ == "__main__":
    main()

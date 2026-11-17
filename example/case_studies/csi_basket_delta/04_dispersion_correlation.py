"""
Stage 4 — what drives the gap: vol dispersion and correlation, to the extremes.

Sweep 4: vol dispersion scaled in log space so vols stay positive,
sigma_i(lambda) = geo-mean * exp(lambda (ln sigma_i - mean ln sigma)),
lambda in {0, 0.5, 1, 1.5, 2}.  lambda = 0 leaves only concentration and
yield dispersion.

Sweep 5: correlation, including the extremes.
* Uniform rho on all 5 names over {-0.25 (PSD floor -1/(N-1)), 0, ..., 1}.
  rho = 1 is singular (one shock drives every name); with unequal vols the
  weights still drift, so the gap does not vanish.
* A two-name basket (EQ 30% / HEDGE 15%) over rho in {-1, -0.5, 0, 0.5, 1}:
  rho = -1 is only feasible pairwise.
* At each interior point, the correlation model-risk band: how far the
  reference delta itself moves when rho is shifted by +/-0.1 (same paths).
  A method gap smaller than the band is "within correlation model risk".

All scenarios: rule A (fixed shares), 1Y ATM call.  Writes
results/04_dispersion_correlation.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _common as cm  # noqa: E402

RESULTS = HERE / "results"
LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
RHO_5 = [-0.25, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
RHO_2 = [-1.0, -0.5, 0.0, 0.5, 1.0]
BAND_SHIFT = 0.1
OPTION = cm.CSIOption(1.0, 1.0)


def scaled_vols(lam: float) -> np.ndarray:
    logs = np.log(cm.BASE_VOLS)
    return np.exp(logs.mean() + lam * (logs - logs.mean()))


def sweep_dispersion() -> list:
    rows = []
    for lam in LAMBDAS:
        basket = cm.base_basket("none", vols=scaled_vols(lam))
        rec = cm.evaluate_scenario(
            basket, cm.default_state(basket), OPTION, f"vol dispersion x{lam:.1f}"
        )
        rec["lambda"] = lam
        rows.append(rec)
    return rows


def _feasible(n: int, rho: float) -> bool:
    return -1.0 / (n - 1) - 1e-12 <= rho <= 1.0 + 1e-12


def _with_band(make, n: int, rho: float, label: str) -> dict:
    basket = make(rho)
    state = cm.default_state(basket)
    rec = cm.evaluate_scenario(basket, state, OPTION, label)
    rec["rho"] = rho
    lo, hi = rho - BAND_SHIFT, rho + BAND_SHIFT
    if _feasible(n, lo) and _feasible(n, hi) and n > 1:
        band = cm.ref_corr_band(
            basket, state, OPTION, cm.uniform_corr(n, min(hi, 1.0)), cm.uniform_corr(n, lo)
        )
        rec["corr_band_cash_per_pct"] = band
        for key, m in rec["methods"].items():
            m["within_corr_band"] = [
                bool(abs(g) <= b) for g, b in zip(m["gap_cash_per_pct"], band)
            ]
    else:
        rec["corr_band_cash_per_pct"] = None
    return rec


def sweep_correlation_five() -> list:
    def make(rho):
        return cm.base_basket("none", corr=cm.uniform_corr(5, rho))

    return [_with_band(make, 5, rho, f"5 names, uniform rho={rho:+.2f}") for rho in RHO_5]


def sweep_correlation_two() -> list:
    return [
        _with_band(cm.two_name_basket, 2, rho, f"EQ/HEDGE, rho={rho:+.1f}") for rho in RHO_2
    ]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    out = {
        "stage": "04_dispersion_correlation",
        "band_shift": BAND_SHIFT,
        "sweep4_vol_dispersion": sweep_dispersion(),
        "sweep5_corr_five_names": sweep_correlation_five(),
        "sweep5_corr_two_names": sweep_correlation_two(),
    }
    path = RESULTS / "04_dispersion_correlation.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    for key in ("sweep4_vol_dispersion", "sweep5_corr_five_names", "sweep5_corr_two_names"):
        for rec in out[key]:
            band = rec.get("corr_band_cash_per_pct")
            band_txt = "" if band is None else f"  band max {max(band):8,.0f}"
            print(
                f"  {rec['label']:28s} M1 sigma {rec['methods']['M1']['residual_sigma_cny']:9,.0f}  "
                f"max|M1 gap| {max(abs(g) for g in rec['methods']['M1']['gap_cash_per_pct']):8,.0f}{band_txt}"
            )


if __name__ == "__main__":
    main()

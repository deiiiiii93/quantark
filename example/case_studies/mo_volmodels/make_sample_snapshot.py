"""Generate a deterministic, arbitrage-free MO sample snapshot for offline runs/tests.

Prices a chosen skewed smile with Black-Scholes (r=2%, q=1%, S0=6000) so downstream
stages have a known ground truth. Run: python example/mo_volmodels/make_sample_snapshot.py

This script is intentionally self-contained (its own BS implementation, no quantark
import) so it can run under any interpreter, mirroring how the real stage-01 fetch
produces the same snapshot schema from live data.
"""
import json
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bs(cp: str, s: float, k: float, t: float, sigma: float, r: float, q: float) -> float:
    d1 = (log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    if cp == "C":
        return s * exp(-q * t) * _norm_cdf(d1) - k * exp(-r * t) * _norm_cdf(d2)
    return k * exp(-r * t) * _norm_cdf(-d2) - s * exp(-q * t) * _norm_cdf(-d1)


def _smile_iv(k: float, s: float, t: float) -> float:
    # Downward equity skew + mild convexity/term, expressed in log-moneyness.
    m = log(k / s)
    return 0.22 - 0.35 * m + 0.6 * m * m + 0.02 * t


def main() -> None:
    S0, R, Q = 6000.0, 0.02, 0.01
    expiries = [
        ("2026-08-15", 0.11),
        ("2026-09-19", 0.20),
        ("2026-12-19", 0.45),
        ("2027-03-20", 0.70),
    ]
    strikes = [round(x / 50.0) * 50.0 for x in (S0 * np.exp(np.linspace(-0.35, 0.35, 15)))]
    out = {
        "fetched_at": "2026-07-06T15:00:00",
        "market_open": True,
        "underlying": {"code": "000852.SH", "spot": S0},
        "expiries": [],
    }
    MIN_TICK = 0.2  # index points; deep-OTM wings below this simply are not quoted
    for date, t in expiries:
        quotes = []
        for k in strikes:
            iv = _smile_iv(k, S0, t)
            for cp in ("C", "P"):
                px = _bs(cp, S0, k, t, iv, R, Q)
                if px < MIN_TICK:  # no live quote for a sub-tick deep-OTM option
                    continue
                quotes.append(
                    {
                        "strike": float(k),
                        "type": cp,
                        "last": round(px, 2),
                        "bid": round(px * 0.995, 2),
                        "ask": round(px * 1.005, 2),
                        "volume": 500,
                        "oi": 2000,
                    }
                )
        out["expiries"].append({"expiry_date": date, "T_years": t, "quotes": quotes})

    dest = Path(__file__).resolve().parent / "data/mo_snapshot_sample.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} ({sum(len(e['quotes']) for e in out['expiries'])} quotes)")


if __name__ == "__main__":
    main()

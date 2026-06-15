#!/usr/bin/env python
"""
SABR smile demo.

Shows the Hagan lognormal SABR implied-vol smile, a calibration round-trip, and
the ``SABRVolSurface`` adapter that plugs into the quant-ark vol-surface
interface.

Run:
    python example/sabr_smile_demo.py
"""

from __future__ import annotations

import numpy as np

from quantark.param.vol import SABRVolSurface
from quantark.param.vol.sabr import (
    calibrate_sabr_slice,
    sabr_implied_vol_black,
)


def demo_smile() -> None:
    F, T = 100.0, 1.0
    alpha, beta, rho, nu = 0.20, 0.6, -0.30, 0.80
    strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
    vols = sabr_implied_vol_black(F, strikes, T, alpha, beta, rho, nu)

    print("SABR smile (F=100, T=1y, beta=0.6, rho=-0.3, nu=0.8):")
    for k, v in zip(strikes, vols):
        print(f"  K={k:6.1f}   sigma={v:7.4%}")


def demo_calibration() -> None:
    F, T = 100.0, 1.0
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    true = dict(alpha=0.18, beta=0.6, rho=-0.25, nu=0.9)
    market = sabr_implied_vol_black(F, strikes, T, **true)

    res = calibrate_sabr_slice(F, strikes, T, market, beta=true["beta"])
    print("\nCalibration round-trip (beta fixed at 0.6):")
    print(f"  true : alpha={true['alpha']:.3f} rho={true['rho']:+.3f} nu={true['nu']:.3f}")
    print(f"  fit  : alpha={res['alpha']:.3f} rho={res['rho']:+.3f} nu={res['nu']:.3f}")
    print(f"  mse  : {res['mse']:.2e}")


def demo_surface() -> None:
    surface = SABRVolSurface(
        slices={
            0.5: {"alpha": 0.20, "beta": 0.6, "rho": -0.40, "nu": 0.9},
            2.0: {"alpha": 0.24, "beta": 0.6, "rho": -0.20, "nu": 0.6},
        },
        forward=lambda spot, ttm: spot * np.exp(0.01 * ttm),
    )
    print("\nSABRVolSurface.get_vol(strike, ttm, spot=100):")
    for ttm in (0.5, 1.0, 2.0):
        v_atm = surface.get_vol(strike=100.0, time_to_maturity=ttm, spot=100.0)
        v_otm = surface.get_vol(strike=120.0, time_to_maturity=ttm, spot=100.0)
        print(f"  T={ttm:4.2f}y   atm={v_atm:7.4%}   K=120={v_otm:7.4%}")


if __name__ == "__main__":
    demo_smile()
    demo_calibration()
    demo_surface()

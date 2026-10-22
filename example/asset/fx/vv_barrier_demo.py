#!/usr/bin/env python
"""
Vanna-Volga one-touch barrier demo.

Prices an FX one-touch with the Vanna-Volga correction, comparing the
Black-Scholes baseline (BSTV) against the VV-adjusted price under both the
survival-probability and first-exit-time attenuation measures.

Run:
    python example/vv_barrier_demo.py
"""

from __future__ import annotations

from quantark.asset.fx.engine.analytical.vannavolga import (
    BarrierGamma,
    price_vv_one_touch,
)
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes


def main() -> None:
    env = FXEnv(spot=1.10, rd=0.01, rf=0.005, tau=1.0)
    quotes = SmileQuotes(sigma_atm=0.10, rr25=-0.005, bf25_2vol=0.002)
    barrier = 1.25

    print("USDCHF-like 1y up one-touch, barrier=1.25, spot=1.10, sigma_atm=10%")
    for gamma in (BarrierGamma.SURV, BarrierGamma.FET):
        res = price_vv_one_touch(
            env, quotes, barrier=barrier, is_up=True, gamma_type=gamma
        )
        print(
            f"  gamma={gamma.value:>4}:  BSTV={res.bstv:.4f}  VV={res.vv:.4f}  "
            f"attenuation={res.gamma:.3f}  (p_vanna={res.p_vanna:.3f}, p_volga={res.p_volga:.3f})"
        )


if __name__ == "__main__":
    main()

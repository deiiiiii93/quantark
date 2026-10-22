#!/usr/bin/env python
"""
Vanna-Volga FX smile demo.

Builds a VV-adjusted FX smile from ATM / 25-delta risk-reversal / 25-delta
butterfly quotes and reads implied vols by strike through the
``VannaVolgaVolSurface`` adapter.

Run:
    python example/vanna_volga_smile_demo.py
"""

from __future__ import annotations

from quantark.param.vol import VannaVolgaVolSurface
from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    SmileQuotes,
    compute_omega,
    strikes_25d,
)


def main() -> None:
    env = FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=1.0)
    quotes = SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.0025)
    conv = DeltaConvention.SPOT

    kp, kc = strikes_25d(quotes.sigma_atm, env, conv)
    omega, _ = compute_omega(env, quotes, conv)
    sigma_25p, sigma_25c = quotes.sigma_25d()

    print("EURUSD 1y, sigma_atm=10%, RR25=-1.0%, BF25=0.25%")
    print(f"  25d strikes: K_put={kp:.4f}  K_call={kc:.4f}")
    print(f"  25d vols:    put={sigma_25p:.4%}  call={sigma_25c:.4%}")
    print(f"  Omega [vega, vanna, volga]: {omega.round(4).tolist()}")

    surface = VannaVolgaVolSurface(env, quotes, conv)
    print("\nVannaVolgaVolSurface.get_vol(strike):")
    for k in (1.05, 1.12, 1.20, 1.28, 1.35):
        print(f"  K={k:5.2f}   sigma={surface.get_vol(k, env.tau, env.spot):7.4%}")


if __name__ == "__main__":
    main()

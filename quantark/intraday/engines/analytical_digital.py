"""Analytical cash-or-nothing digital on the intraday clock.

With total variance W to expiry and carry R - Q, the closed form N(d2) only
needs W; the wrapped surface returns sigma_eff = sqrt(W/T), so the legacy
engine prices exactly the profile's variance. W == 0.0 (expiry at valuation,
or a zero-variance interval such as lunch under a sessions-only profile) is
the exact deterministic limit: the contract's own payoff of the forward.
"""
from __future__ import annotations

from math import exp, log

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.intraday.engines.base import EnginePriceOutcome


class AnalyticalDigitalRoute:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        twin, env = ctx.numerical.product, ctx.pricing_env
        T = float(ctx.numerical.maturity_tau)
        S, K = float(env.spot), float(twin.strike)
        df_pay = float(resolve_terminal_timing(twin, env).payment_df)
        if T > 0.0:
            W = float(env.vol_surface.total_variance(K, T, S))
            R = -log(float(env.get_discount_factor(T)))
            Q = float(env.get_div_yield(T)) * T
        else:
            W, R, Q = 0.0, 0.0, 0.0
        if W == 0.0:   # exact plateau value from the clock, not a tolerance
            F = S * exp(R - Q)
            records = ("terminal forward equals the strike: the contract's strict inequality pays 0",) if F == K else ()
            pv = float(twin.get_payoff(F)) * df_pay
            return EnginePriceOutcome(pv, "deterministic_zero_variance",
                                      {"total_variance": 0.0, "forward": F, "carry": R - Q, "payment_df": df_pay}, {}, records)
        pv = float(engine.price(twin, env))
        return EnginePriceOutcome(pv, "analytical_bs_effective_variance",
                                  {"total_variance": W, "effective_vol": (W / T) ** 0.5, "carry": R - Q, "payment_df": df_pay}, {})

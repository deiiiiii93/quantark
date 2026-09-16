"""Analytical cash-or-nothing digital on the intraday clock.

With total variance W to expiry and carry R - Q, the closed form N(d2) only
needs W; the wrapped surface returns sigma_eff = sqrt(W/T), so the legacy
engine prices exactly the profile's variance. W == 0.0 (expiry at valuation,
or a zero-variance interval such as lunch under a sessions-only profile) is
the exact deterministic limit: the contract's own payoff of the forward.
"""
from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.execution.errors import CapabilityError
from quantark.intraday.engines.base import TERMINATED_POINT_GREEKS, EnginePriceOutcome, PointGreeks


def _refuse_daily_expiry_tolerance(engine, T: float, W: float) -> None:
    """Refuse the window where the daily engine abandons the positive-variance law.

    The daily engine switches to intrinsic below ``MIN_MATURITY``, a CALENDAR-TIME
    tolerance. Intraday that window is reachable with variance still on the clock
    (a millisecond of a 20% vol carries W ~ 8e-12, and N(d2) is perfectly
    conditioned there), so the daily price and the intraday point Greeks would
    describe different functions. The route refuses rather than reuse a tolerance
    that was never meant for this axis; the zero-variance limit is exact and is
    handled before this check.
    """
    floor = float(getattr(engine, "MIN_MATURITY", 0.0) or 0.0)
    if W > 0.0 and 0.0 < T < floor:
        raise CapabilityError(
            f"{type(engine).__name__} switches to the intrinsic payoff below MIN_MATURITY={floor:g} years, but this "
            f"valuation still carries total variance W={W:g} at T={T:g}: the daily near-expiry tolerance is not an "
            "intraday limit. Value at or before the expiry instant itself (W == 0 is exact), or move the valuation "
            f"timestamp more than {floor * 365.0 * 86400.0:.3g} s before the fixing.")


class AnalyticalDigitalRoute:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        if ctx.numerical.terminated:
            return EnginePriceOutcome(0.0, "terminated",
                                      {"reason": "the terminal fixing is history; only the ledger remains"}, {}, exact=True)
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
                                      {"total_variance": 0.0, "forward": F, "carry": R - Q, "payment_df": df_pay}, {}, records,
                                      exact=True)
        _refuse_daily_expiry_tolerance(engine, T, W)
        pv = float(engine.price(twin, env))
        return EnginePriceOutcome(pv, "analytical_bs_effective_variance",
                                  {"total_variance": W, "effective_vol": (W / T) ** 0.5, "carry": R - Q, "payment_df": df_pay},
                                  {}, exact=True)

    def point_greeks(self, ctx, engine, *, certify: bool = True) -> PointGreeks:
        """Closed-form delta/gamma of payout*df*N(+-d2); the zero-variance limit is flat except at the strike."""
        if ctx.numerical.terminated:
            return TERMINATED_POINT_GREEKS
        twin, env = ctx.numerical.product, ctx.pricing_env
        T = float(ctx.numerical.maturity_tau)
        S, K = float(env.spot), float(twin.strike)
        W = float(env.vol_surface.total_variance(K, T, S)) if T > 0.0 else 0.0
        R = -log(float(env.get_discount_factor(T))) if T > 0.0 else 0.0
        Q = float(env.get_div_yield(T)) * T if T > 0.0 else 0.0
        if W == 0.0:
            if S * exp(R - Q) == K:
                return PointGreeks(None, None, "undefined", "the deterministic forward sits on the strike: the payoff jumps there",
                                   "closed_form")
            return PointGreeks(0.0, 0.0, "ok", "", "closed_form")
        _refuse_daily_expiry_tolerance(engine, T, W)
        scale = float(twin.payout) * float(twin.contract_multiplier) * float(resolve_terminal_timing(twin, env).payment_df)
        sign = 1.0 if twin.is_call() else -1.0
        sw = sqrt(W)
        d2 = (log(S / K) + (R - Q) - 0.5 * W) / sw
        density = float(norm.pdf(d2))
        delta = sign * scale * density / (S * sw)
        gamma = -sign * scale * density * (1.0 + d2 / sw) / (S * S * sw)
        return PointGreeks(delta, gamma, "ok", "", "closed_form")

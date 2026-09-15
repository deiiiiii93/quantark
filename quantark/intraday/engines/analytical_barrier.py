"""Analytical continuous barrier / one-touch route, gated by exact admissibility.

Discrete monitoring is refused (the BGK shift is an approximation). Expiry-only
monitoring needs only the terminal distribution, so it is exact as priced. For
continuous monitoring the admissibility test decides: a uniform calendar
variance rate with flat carry prices the twin on the clock-wrapped environment;
zero carry prices a proxy in variance time under zero rates.
"""
from __future__ import annotations

from copy import deepcopy
from math import sqrt

from quantark.execution.errors import CapabilityError
from quantark.intraday.admissibility import analytical_barrier_admissibility
from quantark.intraday.engines.base import EnginePriceOutcome
from quantark.intraday.timestamp import calendar_year_fraction
from quantark.util.enum.option_enums import ObservationType

_NUMERICAL_ALTERNATIVES = "BarrierPDESolver, BarrierOptionMCEngine"


class AnalyticalBarrierRoute:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
        from quantark.param import FlatRateCurve, FlatVolSurface, NoDividend
        from quantark.priceenv import PricingEnvironment

        num, env = ctx.numerical, ctx.pricing_env
        if num.terminated:
            return EnginePriceOutcome(0.0, "terminated", {"reason": "the barrier outcome is history; only the ledger remains"}, {})
        twin = num.product
        if isinstance(twin, EuropeanVanillaOption):
            pv = float(BlackScholesEngine().price(twin, env))
            return EnginePriceOutcome(pv, "analytical_vanilla_after_ki", {"total_variance": float(
                env.vol_surface.total_variance(float(twin.strike), num.maturity_tau, float(env.spot)))}, {})
        observation = twin.observation_type
        if observation == ObservationType.DISCRETE:
            raise CapabilityError("discrete barrier monitoring has no exact closed form: BGK is an approximation; discrete "
                                  f"barriers route to PDE/QUAD/MC intraday ({_NUMERICAL_ALTERNATIVES})")
        if observation == ObservationType.EXPIRY:
            return EnginePriceOutcome(float(engine.price(twin, env)), "analytical_expiry_monitoring", {}, {})
        adm = analytical_barrier_admissibility(ctx)
        if not adm.admissible:
            raise CapabilityError(f"analytical {type(twin).__name__} is not exact here: {adm.reason}. "
                                  f"Numerical intraday alternatives: {_NUMERICAL_ALTERNATIVES}")
        if adm.mode == "uniform_calendar_rate":
            pv = float(engine.price(twin, env))
            return EnginePriceOutcome(pv, "analytical_uniform_calendar_rate",
                                      {"sigma_effective": adm.sigma_effective, "total_variance": adm.total_variance}, {})
        u_T = adm.variance_time_maturity
        proxy = deepcopy(twin)
        proxy.maturity = u_T
        t_pay = calendar_year_fraction(ctx.valuation_timestamp, ctx.timeline.terminal().payment_timestamp)
        lag = max(float(ctx.time_map.to_trading(t_pay)) - u_T, 0.0) if t_pay > num.maturity_tau else 0.0
        proxy.settlement_convention = SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION) if lag > 0.0 else None
        sigma = sqrt(adm.total_variance / u_T)
        proxy_env = PricingEnvironment(rate_curve=FlatRateCurve(0.0), valuation_date=ctx.valuation_timestamp,
                                       spot_quote=env.spot_quote, vol_surface=FlatVolSurface(sigma), div_yield=NoDividend())
        pv = float(engine.price(proxy, proxy_env))
        return EnginePriceOutcome(pv, "analytical_zero_carry_time_change",
                                  {"variance_time_maturity": u_T, "sigma_proxy": sigma, "total_variance": adm.total_variance}, {})

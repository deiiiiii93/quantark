"""Smile-consistent SABR Monte-Carlo engine for European vanillas."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from quantark.asset.equity.process.sabr import SABRProcess
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.param import MCParams
from quantark.param.vol import SABRVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import SABRMCScheme
from quantark.util.exceptions import MarketDataError, PricingError


class SABRMCEngine:
    """Prices European vanillas under genuine SABR dynamics (not a flat-vol proxy).

    Requires a SABRVolSurface on the pricing environment. Builds a SABRProcess
    from the maturity slice (forward from the env), simulates F_T under the chosen
    SABRMCScheme, discounts the European payoff. Smile-consistent by construction,
    so the smile-collapse guard does not apply.

    LOG_EULER (default) works for every beta; QUADEXP is the Andersen-style
    conditional lognormal (exact for beta=1, accurate at coarse time grids).
    """

    def __init__(
        self,
        params: Optional[MCParams] = None,
        scheme: SABRMCScheme = SABRMCScheme.LOG_EULER,
    ):
        self.params = params or MCParams()
        self.scheme = scheme

    def price(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> float:
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"SABRMCEngine only supports EuropeanVanillaOption, got {type(product).__name__}"
            )
        surface = pricing_env.vol_surface
        if not isinstance(surface, SABRVolSurface):
            raise MarketDataError(
                f"SABRMCEngine requires a SABRVolSurface, got {type(surface).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        params = surface._params_at(T)
        # Forward consistent with the env (carry); SABR evolves the forward.
        fwd = S * math.exp((r - q) * T)

        proc = SABRProcess(
            f0=fwd, alpha=params["alpha"], beta=params["beta"],
            rho=params["rho"], nu=params["nu"], shift=params["shift"],
        )
        fT = proc.simulate(
            T=T, n_paths=self.params.num_paths, n_steps=self.params.time_steps,
            seed=self.params.seed, antithetic=getattr(self.params, "use_antithetic", True),
            scheme=self.scheme,
        )
        if product.is_call():
            payoff = np.maximum(fT - K, 0.0)
        else:
            payoff = np.maximum(K - fT, 0.0)
        price = math.exp(-r * T) * float(payoff.mean()) * product.contract_multiplier
        if price < 0:
            raise PricingError(f"Negative price computed: {price}")
        return price

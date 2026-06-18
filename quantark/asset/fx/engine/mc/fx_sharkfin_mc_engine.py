"""Monte Carlo engine for FX single sharkfin options.

A sharkfin is a capped knock-out vanilla with rebates. The knock-out survival /
first-hit simulation and the KO payoff assembly are reused from
:class:`FxBarrierMCEngine`; only the surviving-path terminal payoff differs
(participation-scaled capped vanilla + a no-hit bonus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from quantark.asset.fx.engine.mc.fx_barrier_mc_engine import FxBarrierMCEngine
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_sharkfin_option import FxSharkfinOption
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxBarrierType, ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero


@dataclass
class FxSharkfinMCResult:
    """Breakdown of an FX sharkfin Monte Carlo run."""

    price: float
    std_error: Optional[float]
    num_paths: int
    sigma: float
    monitoring: str


class FxSharkfinMCEngine(FxBarrierMCEngine):
    """Constant-vol GK Monte Carlo for FxSharkfinOption."""

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        opt = self._check_product(product)
        if self.mc.method == MonteCarloMethod.RANDOMIZED_QUASI:
            raise NotImplementedError(
                "RANDOMIZED_QUASI is not yet implemented for FxSharkfinMCEngine; "
                "use PSEUDO or QUASI."
            )

        spot = fx_env.effective_spot()
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")
        T = float(opt.get_maturity(fx_env))
        if T < 0:
            raise ValidationError(f"maturity must be non-negative, got {T}")
        T_pay = float(opt.get_delivery(fx_env))
        df_pay = fx_env.get_domestic_df(T_pay)

        if is_zero(T):
            price = self._expiry_value(opt, spot, df_pay)
            self._last_result = FxSharkfinMCResult(price, 0.0, 0, 0.0, "expiry")
            return price

        if self._already_hit(opt, spot):
            # Knocked out at inception: only the KO rebate remains.
            price = opt.ko_rebate if opt.rebate_at_hit else opt.ko_rebate * df_pay
            self._last_result = FxSharkfinMCResult(price, 0.0, 0, 0.0, "inception")
            return price

        sigma = float(fx_env.get_vol(fx_env.get_forward(T), T))
        if sigma <= 0:
            raise ValidationError(f"vol must be positive, got {sigma}")

        sim = self._barrier_sim(opt, fx_env, T, sigma)
        terminal_payoff = self._capped_payoff(opt, sim.terminal) + opt.no_hit_rebate
        payoffs = self._assemble_ko_ki(
            FxBarrierType.KNOCK_OUT, opt.ko_rebate, opt.rebate_at_hit,
            sim, terminal_payoff, df_pay,
        )

        price = float(payoffs.mean())
        if self.mc.method == MonteCarloMethod.QUASI:
            std_error: Optional[float] = None
        else:
            std_error = float(payoffs.std(ddof=1)) / math.sqrt(payoffs.size)
        if price < 0:
            raise PricingError(f"Negative price computed: {price}")
        self._last_result = FxSharkfinMCResult(
            price, std_error, int(payoffs.size), sigma, opt.monitoring.name.lower()
        )
        return price

    def _capped_payoff(self, opt, terminal: np.ndarray) -> np.ndarray:
        """Participation-scaled capped vanilla over a vector of terminal spots."""
        if opt.is_call():
            capped = np.minimum(terminal, opt.barrier) - opt.strike
        else:
            capped = opt.strike - np.maximum(terminal, opt.barrier)
        return opt.participation * np.maximum(capped, 0.0)

    def _expiry_value(self, opt, spot, df_pay) -> float:
        hit = self._already_hit(opt, spot)
        if hit:
            return opt.ko_rebate * df_pay
        return (opt.capped_intrinsic(spot) + opt.no_hit_rebate) * df_pay

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxSharkfinOption:
        if not isinstance(product, FxSharkfinOption):
            raise PricingError(
                f"FxSharkfinMCEngine only supports FxSharkfinOption, "
                f"got {type(product).__name__}"
            )
        return product

    def get_last_result(self) -> Optional[FxSharkfinMCResult]:
        return self._last_result

    def __repr__(self) -> str:
        return (
            f"FxSharkfinMCEngine(method={self.mc.method.name}, "
            f"brownian_bridge={self.use_brownian_bridge})"
        )

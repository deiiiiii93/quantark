"""Monte Carlo engine for FX Target Redemption Forwards (constant-vol GK).

Simulates the FX rate at each fixing on the shared :class:`FxGKPathGenerator`,
then walks the periods pathwise accumulating the favorable amount, truncating the
final favorable amount so the cumulative favorable equals the target exactly, and
discounting each period's geared cashflow on the domestic curve until the deal
redeems.

Validated under a flat surface against closed forms: ``target=inf, gearing=1``
reduces to a plain forward strip, and ``target=inf, gearing!=1`` to a forward
strip minus ``(gearing-1)`` x a GK vanilla put/call strip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.process.fx_gk_path_generator import FxGKPathGenerator
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_target_redemption_forward import (
    FxTargetRedemptionForward,
)
from quantark.montecarlo.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.montecarlo.qmc_variance_reduction import VarianceReductionConfig
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError


@dataclass
class FxTarnMCResult:
    """Breakdown of a target-redemption Monte Carlo run."""

    price: float
    std_error: Optional[float]
    num_paths: int
    sigma: float
    mean_redemption_fraction: float  # fraction of paths that redeemed early


class FxTarnForwardMCEngine(BaseFxEngine):
    """Constant-vol GK Monte Carlo for FxTargetRedemptionForward."""

    engine_type = EngineType.MONTE_CARLO
    _ACC_TOL = 1e-12

    def __init__(
        self,
        params: Optional[FxMCParams] = None,
        greeks_params: Optional[FxEngineParams] = None,
    ):
        super().__init__(greeks_params)
        self.mc = params or FxMCParams()
        self._last_result: Optional[FxTarnMCResult] = None

    # -- entry -----------------------------------------------------------

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        opt = self._check_product(product)
        if self.mc.method == MonteCarloMethod.RANDOMIZED_QUASI:
            raise NotImplementedError(
                "RANDOMIZED_QUASI is not yet implemented for FxTarnForwardMCEngine; "
                "use PSEUDO or QUASI."
            )

        spot = fx_env.effective_spot()
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")
        T = float(opt.get_maturity(fx_env))
        sigma = float(fx_env.get_vol(fx_env.get_forward(T), T))
        if sigma <= 0:
            raise ValidationError(f"vol must be positive, got {sigma}")

        fixings = np.asarray(opt.fixing_times, dtype=float)
        df_pay = np.array(
            [fx_env.get_domestic_df(float(t)) for t in opt.pay_times]
        )
        spots = self._simulate(fx_env, spot, sigma, fixings)  # (P, M)

        pv, redeemed = self._accumulate(opt, spots, df_pay)
        price = float(pv.mean())
        if self.mc.method == MonteCarloMethod.QUASI:
            std_error: Optional[float] = None
        else:
            std_error = float(pv.std(ddof=1)) / math.sqrt(pv.size)
        self._last_result = FxTarnMCResult(
            price, std_error, int(pv.size), sigma, float(redeemed.mean())
        )
        return price

    # -- simulation ------------------------------------------------------

    def _simulate(self, fx_env, spot, sigma, fixings) -> np.ndarray:
        gen = self._make_generator(fx_env, spot, sigma, fixings)
        paths, _ = gen.generate_paths()
        return paths[:, 1:]  # drop S_0 column -> (P, M) at the fixings

    def _make_generator(self, fx_env, spot, sigma, times) -> FxGKPathGenerator:
        if self.mc.method == MonteCarloMethod.PSEUDO:
            stream = PseudoRandomNormalGenerator(seed=self.mc.seed)
            is_qmc = False
            vr = VarianceReductionConfig(antithetic=True) if self.mc.use_antithetic else None
        else:  # QUASI
            stream = SobolNormalGenerator(base_seed=self.mc.seed)
            is_qmc = True
            vr = None
        return FxGKPathGenerator(
            spot=spot, sigma=sigma, forward_fn=fx_env.get_forward,
            times=times, num_paths=self.mc.num_paths, random_stream=stream,
            use_brownian_bridge=False, vr_config=vr, is_qmc=is_qmc,
        )

    # -- pathwise target redemption -------------------------------------

    def _accumulate(self, opt, spots: np.ndarray, df_pay: np.ndarray):
        """Walk periods pathwise; return (pv_per_path, redeemed_early_mask)."""
        n_paths = spots.shape[0]
        sign = 1.0 if opt.is_buy_base else -1.0
        diff = sign * (spots - opt.strike)  # (P, M); >0 favorable
        favorable = opt.notional * np.maximum(diff, 0.0)
        unfavorable = opt.notional * np.maximum(-diff, 0.0)

        acc = np.zeros(n_paths)
        alive = np.ones(n_paths, dtype=bool)
        pv = np.zeros(n_paths)
        target = opt.target

        for i in range(spots.shape[1]):
            fav_i, unf_i = favorable[:, i], unfavorable[:, i]
            remaining = target - acc  # capacity left (inf if unbounded target)
            fav_eff = np.where(alive, np.minimum(fav_i, remaining), 0.0)
            cf = fav_eff - opt.gearing * np.where(alive, unf_i, 0.0)
            pv += alive * cf * df_pay[i]
            acc = acc + fav_eff
            hit = acc >= target - self._ACC_TOL  # never True when target=inf
            alive = alive & ~hit

        redeemed_early = ~alive
        return pv, redeemed_early

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxTargetRedemptionForward:
        if not isinstance(product, FxTargetRedemptionForward):
            raise PricingError(
                f"FxTarnForwardMCEngine only supports FxTargetRedemptionForward, "
                f"got {type(product).__name__}"
            )
        return product

    def calculate_greeks(self, product, fx_env) -> Dict[str, float]:
        self._check_product(product)
        return super().calculate_greeks(product, fx_env)

    def get_last_result(self) -> Optional[FxTarnMCResult]:
        return self._last_result

    def __repr__(self) -> str:
        return f"FxTarnForwardMCEngine(method={self.mc.method.name})"

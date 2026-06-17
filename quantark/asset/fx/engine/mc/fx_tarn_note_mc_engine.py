"""Monte Carlo engine for FX Target Redemption Notes (constant-vol GK).

Simulates the FX rate at each fixing on the shared :class:`FxGKPathGenerator`,
then walks the periods pathwise: pays a digital coupon when the (inclusive)
fixing condition holds, truncates the final coupon so the cumulative coupon
equals the target exactly, redeems on the period the target is hit (or at final
maturity otherwise), and — when ``include_principal`` is set — adds the
principal discounted from the path-dependent redemption pay date.

Validated under a flat surface against closed forms: ``target=inf,
include_principal=False`` reduces to a strip of FX digitals (via
:class:`FxDigitalOptionAnalyticalEngine`), and ``target=inf,
include_principal=True`` to that strip plus ``notional * DF(T_final)``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.engine.mc.fx_tarf_mc_engine import FxTarnMCResult
from quantark.asset.fx.process.fx_gk_path_generator import FxGKPathGenerator
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_target_redemption_note import (
    FxTargetRedemptionNote,
)
from quantark.montecarlo.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.montecarlo.qmc_variance_reduction import VarianceReductionConfig
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError


class FxTargetRedemptionNoteMCEngine(BaseFxEngine):
    """Constant-vol GK Monte Carlo for FxTargetRedemptionNote."""

    engine_type = EngineType.MONTE_CARLO
    _HIT_RTOL = 1e-12

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
                "RANDOMIZED_QUASI is not yet implemented for "
                "FxTargetRedemptionNoteMCEngine; use PSEUDO or QUASI."
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
        if price < 0:
            raise PricingError(f"Negative price computed: {price}")
        self._last_result = FxTarnMCResult(
            price, std_error, int(pv.size), sigma, float(redeemed.mean())
        )
        return price

    # -- simulation ------------------------------------------------------

    def _simulate(self, fx_env, spot, sigma, fixings) -> np.ndarray:
        gen = self._make_generator(fx_env, spot, sigma, fixings)
        paths, _ = gen.generate_paths()
        return paths[:, 1:]

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
        n_paths, n_periods = spots.shape
        cond = self._condition_mask(opt, spots)  # (P, M) bool

        acc = np.zeros(n_paths)
        alive = np.ones(n_paths, dtype=bool)
        pv = np.zeros(n_paths)
        early = np.zeros(n_paths, dtype=bool)
        target = opt.target
        tol = self._hit_tol(target)
        include_principal = opt.include_principal

        for i in range(n_periods):
            full_c = opt.full_coupon(i)
            coupon = np.where(cond[:, i], full_c, 0.0)
            remaining = target - acc
            c_eff = np.where(alive, np.minimum(coupon, remaining), 0.0)
            pv += alive * c_eff * df_pay[i]
            acc = acc + c_eff
            # Relative tolerance so a small target is not falsely hit with no
            # coupon accrual; never fires for target=inf.
            hit = acc >= target - tol
            newly = alive & hit
            if include_principal:
                pv += newly * opt.notional * df_pay[i]
            if i < n_periods - 1:
                early |= newly  # a final-period hit is not an early redemption
            alive = alive & ~hit

        # Paths that never redeemed early repay principal at final maturity.
        if include_principal:
            pv += alive * opt.notional * df_pay[-1]

        return pv, early

    @staticmethod
    def _hit_tol(target: float) -> float:
        """Target-relative hit tolerance; 0 for an unbounded target."""
        return (
            0.0
            if math.isinf(target)
            else abs(target) * FxTargetRedemptionNoteMCEngine._HIT_RTOL
        )

    def _condition_mask(self, opt, spots: np.ndarray) -> np.ndarray:
        if opt.is_band:
            return (spots >= opt.lower) & (spots <= opt.upper)
        if opt.is_above:
            return spots >= opt.strike
        return spots <= opt.strike

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxTargetRedemptionNote:
        if not isinstance(product, FxTargetRedemptionNote):
            raise PricingError(
                f"FxTargetRedemptionNoteMCEngine only supports "
                f"FxTargetRedemptionNote, got {type(product).__name__}"
            )
        return product

    def calculate_greeks(self, product, fx_env) -> Dict[str, float]:
        self._check_product(product)
        return super().calculate_greeks(product, fx_env)

    def get_last_result(self) -> Optional[FxTarnMCResult]:
        return self._last_result

    def __repr__(self) -> str:
        return f"FxTargetRedemptionNoteMCEngine(method={self.mc.method.name})"

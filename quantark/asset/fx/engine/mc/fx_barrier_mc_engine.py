"""Monte Carlo engine for FX single-barrier options (constant-vol GK).

Continuous monitoring uses a Brownian-bridge survival correction; discrete
monitoring samples at the product's observation times. Validated against
VannaVolgaBarrierEngine under a flat surface for zero / expiry-paid rebate (a
continuous pay-at-hit rebate is a documented discretized convention).
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
from quantark.asset.fx.product.option.fx_barrier_option import FxBarrierOption
from quantark.montecarlo.qmc_brownian_bridge import compute_step_crossing_probabilities
from quantark.montecarlo.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.montecarlo.qmc_variance_reduction import VarianceReductionConfig
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxBarrierType, ObservationType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero


@dataclass
class FxBarrierMCResult:
    """Breakdown of an FX barrier Monte Carlo run."""

    price: float
    std_error: Optional[float]
    num_paths: int
    sigma: float
    monitoring: str


class FxBarrierMCEngine(BaseFxEngine):
    """Constant-vol GK Monte Carlo for FxBarrierOption (continuous + discrete)."""

    engine_type = EngineType.MONTE_CARLO
    MIN_VOL = 1e-8

    def __init__(
        self,
        params: Optional[FxMCParams] = None,
        use_brownian_bridge: bool = True,
        greeks_params: Optional[FxEngineParams] = None,
    ):
        super().__init__(greeks_params)
        self.mc = params or FxMCParams()
        self.use_brownian_bridge = bool(use_brownian_bridge)
        self._last_result: Optional[FxBarrierMCResult] = None

    # -- entry -----------------------------------------------------------

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        opt = self._check_product(product)
        if self.mc.method == MonteCarloMethod.RANDOMIZED_QUASI:
            raise NotImplementedError(
                "RANDOMIZED_QUASI is not yet implemented for FxBarrierMCEngine; "
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
            self._last_result = FxBarrierMCResult(price, 0.0, 0, 0.0, "expiry")
            return price

        # Inception state: already through the barrier.
        if self._already_hit(opt, spot):
            price = self._inception_hit_value(opt, fx_env, spot, T, T_pay, df_pay)
            self._last_result = FxBarrierMCResult(price, 0.0, 0, 0.0, "inception")
            return price

        sigma = float(fx_env.get_vol(fx_env.get_forward(T), T))
        if sigma <= 0:
            raise ValidationError(f"vol must be positive, got {sigma}")

        if opt.monitoring == ObservationType.DISCRETE:
            payoffs = self._discrete_payoffs(opt, fx_env, T, T_pay, df_pay, sigma)
        else:
            payoffs = self._continuous_payoffs(opt, fx_env, T, T_pay, df_pay, sigma)

        price = float(payoffs.mean())
        if self.mc.method == MonteCarloMethod.QUASI:
            std_error: Optional[float] = None
        else:
            std_error = float(payoffs.std(ddof=1)) / math.sqrt(payoffs.size)
        if price < 0:
            raise PricingError(f"Negative price computed: {price}")
        self._last_result = FxBarrierMCResult(
            price, std_error, int(payoffs.size), sigma,
            opt.monitoring.name.lower(),
        )
        return price

    # -- generator -------------------------------------------------------

    def _make_generator(
        self,
        fx_env: FxPricingEnvironment,
        spot: float,
        sigma: float,
        times: np.ndarray,
        use_bb: bool,
    ) -> FxGKPathGenerator:
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
            use_brownian_bridge=use_bb, vr_config=vr, is_qmc=is_qmc,
        )

    # -- continuous monitoring ------------------------------------------

    def _continuous_payoffs(self, opt, fx_env, T, T_pay, df_pay, sigma) -> np.ndarray:
        n = int(self.mc.time_steps)
        times = np.linspace(0.0, T, n + 1)[1:]
        use_bb = self.use_brownian_bridge and sigma > self.MIN_VOL
        gen = self._make_generator(fx_env, fx_env.effective_spot(), sigma, times, use_bb)
        paths, _ = gen.generate_paths()

        vanilla = self._vanilla_payoff(opt, paths[:, -1])

        if use_bb:
            p = compute_step_crossing_probabilities(paths, opt.barrier, sigma, times)
            survival = np.prod(1.0 - p, axis=1)
            return self._combine(
                opt, survival, vanilla, df_pay,
                first_hit_disc=self._bb_first_hit_discount(p, times, fx_env),
            )

        # sigma ~ 0 or BB disabled: discrete-grid hit detection on dense grid.
        hit_matrix = (
            paths[:, 1:] >= opt.barrier if opt.is_up else paths[:, 1:] <= opt.barrier
        )
        survival = (~hit_matrix.any(axis=1)).astype(float)
        return self._combine(opt, survival, vanilla, df_pay, first_hit_disc=None)

    def _bb_first_hit_discount(self, p, times, fx_env) -> np.ndarray:
        """Discretized rebate-at-detection: first-hit prob * DF(step time)."""
        survival_before = np.cumprod(1.0 - p, axis=1)
        survival_before = np.concatenate(
            [np.ones((p.shape[0], 1)), survival_before[:, :-1]], axis=1
        )
        first_hit = survival_before * p
        df_steps = np.array([fx_env.get_domestic_df(float(t)) for t in times])
        return np.sum(first_hit * df_steps[None, :], axis=1)

    # -- discrete monitoring (Task 7) -----------------------------------

    def _discrete_payoffs(self, opt, fx_env, T, T_pay, df_pay, sigma) -> np.ndarray:
        obs = np.array(opt.observation_times, dtype=float)
        if np.isclose(obs[-1], T):
            times = obs
        else:
            times = np.concatenate([obs, [T]])
        obs_idx = np.arange(obs.size)
        gen = self._make_generator(
            fx_env, fx_env.effective_spot(), sigma, times, use_bb=False
        )
        paths, _ = gen.generate_paths()

        obs_prices = paths[:, obs_idx + 1]
        hit_matrix = (
            obs_prices >= opt.barrier if opt.is_up else obs_prices <= opt.barrier
        )
        hit_any = hit_matrix.any(axis=1)
        vanilla = self._vanilla_payoff(opt, paths[:, -1])

        is_ko = opt.knock_type == FxBarrierType.KNOCK_OUT
        if is_ko:
            if opt.rebate_at_hit and opt.rebate > 0.0:
                first_idx = np.argmax(hit_matrix, axis=1)
                hit_time = times[obs_idx][first_idx]
                df_hit = np.array([fx_env.get_domestic_df(float(t)) for t in hit_time])
                rebate_leg = np.where(hit_any, opt.rebate * df_hit, 0.0)
                return np.where(hit_any, rebate_leg, vanilla * df_pay)
            return np.where(hit_any, opt.rebate, vanilla) * df_pay
        # KNOCK_IN: vanilla if knocked in, else rebate.
        return np.where(hit_any, vanilla * df_pay, opt.rebate * df_pay)

    # -- payoff assembly -------------------------------------------------

    def _combine(self, opt, survival, vanilla, df_pay, first_hit_disc) -> np.ndarray:
        is_ko = opt.knock_type == FxBarrierType.KNOCK_OUT
        if is_ko:
            if opt.rebate_at_hit and opt.rebate > 0.0 and first_hit_disc is not None:
                return first_hit_disc * opt.rebate + survival * vanilla * df_pay
            return (survival * vanilla + (1.0 - survival) * opt.rebate) * df_pay
        # KNOCK_IN: rebate paid at expiry iff never knocked in.
        return ((1.0 - survival) * vanilla + survival * opt.rebate) * df_pay

    def _vanilla_payoff(self, opt, terminal: np.ndarray) -> np.ndarray:
        if opt.is_call():
            return np.maximum(terminal - opt.strike, 0.0)
        return np.maximum(opt.strike - terminal, 0.0)

    # -- degenerate cases ------------------------------------------------

    def _already_hit(self, opt, spot: float) -> bool:
        return spot >= opt.barrier if opt.is_up else spot <= opt.barrier

    def _inception_hit_value(self, opt, fx_env, spot, T, T_pay, df_pay) -> float:
        is_ko = opt.knock_type == FxBarrierType.KNOCK_OUT
        if is_ko:
            # Dead on touch: only the rebate remains.
            if opt.rebate_at_hit:
                return opt.rebate  # paid now (t=0)
            return opt.rebate * df_pay
        # Already knocked in: a plain vanilla under GK.
        from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
        from quantark.asset.fx.product.option import FxVanillaOption
        vanilla = FxVanillaOption(
            strike=opt.strike, option_type=opt.option_type, maturity=T,
            delivery=T_pay, notional_foreign=1.0,
        )
        return GarmanKohlhagenEngine().price(vanilla, fx_env)

    def _expiry_value(self, opt, spot, df_pay) -> float:
        hit = self._already_hit(opt, spot)
        vanilla = max(spot - opt.strike, 0.0) if opt.is_call() else max(opt.strike - spot, 0.0)
        is_ko = opt.knock_type == FxBarrierType.KNOCK_OUT
        if is_ko:
            value = opt.rebate if hit else vanilla
        else:
            value = vanilla if hit else opt.rebate
        return value * df_pay

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxBarrierOption:
        if not isinstance(product, FxBarrierOption):
            raise PricingError(
                f"FxBarrierMCEngine only supports FxBarrierOption, "
                f"got {type(product).__name__}"
            )
        return product

    def calculate_greeks(self, product, fx_env) -> Dict[str, float]:
        self._check_product(product)
        return super().calculate_greeks(product, fx_env)

    def get_last_result(self) -> Optional[FxBarrierMCResult]:
        return self._last_result

    def __repr__(self) -> str:
        return (
            f"FxBarrierMCEngine(method={self.mc.method.name}, "
            f"brownian_bridge={self.use_brownian_bridge})"
        )

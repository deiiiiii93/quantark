"""
Unified quadrature pricing engine for discretely monitored barrier and one-touch options.

Merges BarrierQuadEngine and OneTouchQuadEngine, eliminating ~285 lines of duplication.
"""

import math
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.quad.european_quad_engine import EuropeanQuadEngine
from quantark.asset.equity.engine.quad.quad_adapters import QuadInputAdapter, resolve_quad_adapter
from quantark.asset.equity.engine.quad.quad_core import QuadratureCore
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError


class DiscreteQuadEngine(BaseEngine):
    """
    Unified quadrature engine for discretely monitored products.

    Uses FFT-based convolution and Simpson integration for discrete monitoring.
    """

    engine_type = EngineType.QUADRATURE
    MIN_MATURITY = 1e-10
    MAX_VOL = 5.0

    def __init__(self, params: Optional[QuadParams] = None) -> None:
        if params is None:
            params = QuadParams()
        if not isinstance(params, QuadParams):
            raise ValidationError(
                f"params must be QuadParams instance, got {type(params).__name__}"
            )
        super().__init__(params)
        self._vanilla_engine = EuropeanQuadEngine(params=params)

    @property
    def vanilla_engine(self) -> EuropeanQuadEngine:
        return self._vanilla_engine

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        adapter = resolve_quad_adapter(product)
        return self._price_discrete_instrument(adapter, product, pricing_env)

    def _price_discrete_instrument(
        self,
        adapter: QuadInputAdapter,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> float:
        context = adapter.build_pricing_context(product, pricing_env, self)
        early_price = adapter.early_price(product, pricing_env, context, self)
        if early_price is not None:
            return early_price

        resolved = adapter.resolve_schedule(product, pricing_env, context)
        inputs = adapter.build_inputs(product, resolved, context)
        observation_times = np.asarray(inputs.observation_times, dtype=float)
        effective_grid_x = self._select_grid_points(
            observation_times, context.maturity, context.vol
        )

        core = QuadratureCore(
            grid_x=effective_grid_x,
            spot=context.spot,
            observation_times=inputs.observation_times,
            rate=context.rate,
            div=context.div,
            vol=context.vol,
        )
        core_price = core.price(inputs)
        return adapter.finalize_price(product, pricing_env, context, core_price, self)

    def _select_grid_points(
        self, observation_times: np.ndarray, maturity: float, vol: float
    ) -> int:
        """Select appropriate grid size for stability."""
        intervals = np.diff(np.concatenate(([0.0], observation_times)))
        dt_min = float(np.min(intervals)) if intervals.size else maturity
        vol_max = vol if np.isscalar(vol) else float(np.max(vol))

        approx_log_c = (
            10.0 * vol_max * math.sqrt(maturity)
            + (1.0 + 0.5 * vol_max**2) * maturity
        )
        range_width = 2.0 * approx_log_c
        target_h = 0.5 * vol_max * math.sqrt(dt_min)
        safe_grid_x = int(range_width / target_h) + 1
        return max(self.params.grid_points, safe_grid_x)

    def __repr__(self) -> str:
        return "DiscreteQuadEngine()"


# Backward compatibility aliases
BarrierQuadEngine = DiscreteQuadEngine
OneTouchQuadEngine = DiscreteQuadEngine

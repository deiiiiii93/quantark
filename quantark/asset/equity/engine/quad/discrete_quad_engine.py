"""
Unified quadrature pricing engine for discretely monitored barrier and one-touch options.

Merges BarrierQuadEngine and OneTouchQuadEngine, eliminating ~285 lines of duplication.
"""

import math
from typing import Optional

import numpy as np

from quantark.asset.equity.engine.barrier_schedule_rules import (
    refuse_knock_in_record_payoff,
    refuse_no_touch_record_payoff,
)
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.quad.european_quad_engine import EuropeanQuadEngine
from quantark.asset.equity.engine.quad.quad_adapters import QuadInputAdapter, resolve_quad_adapter
from quantark.asset.equity.engine.quad.term_inputs import build_quad_term_params
from quantark.asset.equity.engine.quad.quad_core import QuadratureCore
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    DoubleOneTouchOption,
    OneTouchOption,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero


from quantark.asset.equity.engine.monitoring_ended import (ended_without_knock_in, knock_in_carried,
                                                           remaining_replication)


class DiscreteQuadEngine(BaseEngine):
    """
    Unified quadrature engine for discretely monitored products.

    Uses FFT-based convolution and Simpson integration for discrete monitoring.
    """

    engine_type = EngineType.QUADRATURE
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
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
        if isinstance(product, (BarrierOption, DoubleBarrierOption)):
            refuse_knock_in_record_payoff(product)      # one rebate per knock-in, as on every route (W18b)
        if isinstance(product, (OneTouchOption, DoubleOneTouchOption)):
            refuse_no_touch_record_payoff(product)      # one rebate per no-touch, as on every route (2026-10-15)
        if knock_in_carried(product) and product.get_maturity(pricing_env) > 0.0:
            # a knock-in carried as triggered (spec 2026-10-07 R4): the adapter's own triggered value, its vanilla
            # (vanilla + rebate x DF - the knock-out core, which pays the rebate once knocked in)
            return adapter._vanilla_price(product, pricing_env, self) * product.contract_multiplier
        if ended_without_knock_in(product) and product.get_maturity(pricing_env) > 0.0:
            # no observation left (spec 2026-10-07 theta boundary 3.3): the remaining contract, by replication
            return remaining_replication(product).price(pricing_env)
        return self._price_discrete_instrument(adapter, product, pricing_env)

    def static_replication(self, product, pricing_env):
        """An ended schedule's remaining contract (spec 2026-10-07 theta boundary 3.3), else none."""
        if ended_without_knock_in(product):
            return remaining_replication(product)
        return super().static_replication(product, pricing_env)

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
        if product.observation_type == ObservationType.DISCRETE:
            # A record at the valuation instant is decided before the lattice (``early_price``: a crossed record
            # short-circuits, W18b / W18c), so one that reaches here was not crossed: decided "not hit", it leaves the
            # observation set (I5 / A14, as the Black-Scholes barrier MC; a zero first interval is no lattice step).
            # Two records on that instant are refused, as any repeated time, never filtered.
            if sum(1 for rec in resolved if is_zero(rec.observation_time)) > 1:
                raise ValidationError("Observation times must be strictly increasing.")
            ahead = [rec for rec in resolved if not is_zero(rec.observation_time)]
            if not ahead and resolved:
                # the sole record was it: nothing is left to observe -- the never-hit value, the remaining contract
                # (as once monitoring has ended)
                return remaining_replication(product).price(pricing_env)
            resolved = ahead
        inputs = adapter.build_inputs(product, resolved, context)
        # an adapter may split a payoff the core's three regions per step cannot carry into legs on one observation
        # grid (the double knock-out's last-record rebate, T17 R30): the core prices each, linearly summed
        legs = tuple(inputs) if isinstance(inputs, (list, tuple)) else (inputs,)
        inputs = legs[0]
        observation_times = np.asarray(inputs.observation_times, dtype=float)

        # Per-interval forward params on the RESOLVED observation grid.
        # Flat curves collapse back to the context scalars (exact — the
        # forward of a constant curve IS the constant), preserving the
        # pre-term code path bit-for-bit.
        rate_in, div_in, vol_in = context.rate, context.div, context.vol
        if context.ref_strike is not None:
            tp = build_quad_term_params(
                pricing_env, context.ref_strike, observation_times
            )
            if not (
                np.all(np.abs(tp.rate - context.rate) <= 1e-12)
                and np.all(np.abs(tp.div - context.div) <= 1e-12)
                and np.all(np.abs(tp.vol - context.vol) <= 1e-12)
            ):
                rate_in, div_in, vol_in = tp.rate, tp.div, tp.vol

        effective_grid_x = self._select_grid_points(
            observation_times, context.maturity, vol_in
        )

        def leg_price(leg) -> float:
            core = QuadratureCore(
                grid_x=effective_grid_x,
                spot=context.spot,
                observation_times=inputs.observation_times,
                rate=rate_in,
                div=div_in,
                vol=vol_in,
            )
            return core.price(leg)

        core_price = leg_price(inputs)
        for leg in legs[1:]:
            core_price += leg_price(leg)
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

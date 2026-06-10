"""
RFQ quote solving service.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Tuple
from uuid import uuid4

from rfq.builders import (
    build_engine_from_termsheet,
    build_pricing_env_from_market_kwargs,
    build_product_from_termsheet,
)
from rfq.models import (
    RFQInputMode,
    RFQQuote,
    RFQQuoteStatus,
    RFQRequest,
)
from rfq.registry import resolve_unknown_adapter
from util.exceptions import PricingError
from util.numerical import Tolerance, is_close, is_zero


@dataclass(frozen=True)
class _NormalizedRFQContext:
    product: Any
    pricing_env: Any
    engine: Any
    field_label: str
    request_summary: dict[str, Any]


class RFQService:
    """In-memory RFQ solver."""

    def __init__(
        self,
        *,
        price_tolerance: float = 1e-8,
        value_tolerance: float = 1e-10,
        max_iterations: int = 100,
    ) -> None:
        self.price_tolerance = price_tolerance
        self.value_tolerance = value_tolerance
        self.max_iterations = max_iterations

    def quote(self, request: RFQRequest) -> RFQQuote:
        """Solve a single-unknown RFQ request."""
        context = self._normalize_request(request)
        adapter = resolve_unknown_adapter(
            request.unknown, context.product, context.pricing_env
        )
        solved_value, achieved_price = self._solve(
            context.product,
            context.pricing_env,
            context.engine,
            adapter,
            request,
        )
        residual = achieved_price - request.target.value

        return RFQQuote(
            quote_id=f"rfq-{uuid4().hex[:12]}",
            quoted_at=datetime.utcnow(),
            status=RFQQuoteStatus.SUCCESS,
            field_path=adapter.field_path,
            field_label=context.field_label,
            solved_value=solved_value,
            target_label=request.target.label,
            target_value=request.target.value,
            achieved_price=achieved_price,
            residual=residual,
            engine_summary={
                "engine_class": type(context.engine).__name__,
                "engine_type": getattr(
                    getattr(context.engine, "engine_type", None),
                    "name",
                    str(getattr(context.engine, "engine_type", "unknown")),
                ),
            },
            request_summary=context.request_summary,
            valid_until=request.valid_until,
        )

    def _normalize_request(self, request: RFQRequest) -> _NormalizedRFQContext:
        if request.input_mode == RFQInputMode.OBJECT:
            object_input = request.object_input
            assert object_input is not None
            return _NormalizedRFQContext(
                product=object_input.product,
                pricing_env=object_input.pricing_env,
                engine=object_input.engine,
                field_label=request.unknown.display_label or request.unknown.field_path,
                request_summary={
                    "input_mode": request.input_mode.value,
                    "product_type": type(object_input.product).__name__,
                    "engine_class": type(object_input.engine).__name__,
                    "field_path": request.unknown.field_path,
                    "target_label": request.target.label.value,
                },
            )

        termsheet_input = request.termsheet_input
        assert termsheet_input is not None
        product = build_product_from_termsheet(termsheet_input)
        pricing_env = build_pricing_env_from_market_kwargs(
            termsheet_input.market_kwargs
        )
        engine = build_engine_from_termsheet(termsheet_input)
        return _NormalizedRFQContext(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            field_label=request.unknown.display_label or request.unknown.field_path,
            request_summary={
                "input_mode": request.input_mode.value,
                "product_type": type(product).__name__,
                "engine_class": type(engine).__name__,
                "field_path": request.unknown.field_path,
                "target_label": request.target.label.value,
            },
        )

    def _evaluate_candidate(
        self,
        base_product: Any,
        base_pricing_env: Any,
        engine: Any,
        adapter: Any,
        candidate: float,
    ) -> float:
        product = deepcopy(base_product)
        pricing_env = deepcopy(base_pricing_env)
        adapter.set_value(product, pricing_env, candidate)
        return float(engine.price(product, pricing_env))

    def _objective(
        self,
        base_product: Any,
        base_pricing_env: Any,
        engine: Any,
        adapter: Any,
        request: RFQRequest,
        candidate: float,
    ) -> float:
        return (
            self._evaluate_candidate(
                base_product, base_pricing_env, engine, adapter, candidate
            )
            - request.target.value
        )

    def _solve(
        self,
        base_product: Any,
        base_pricing_env: Any,
        engine: Any,
        adapter: Any,
        request: RFQRequest,
    ) -> Tuple[float, float]:
        lower = request.unknown.lower_bound
        upper = request.unknown.upper_bound

        f_lower = self._objective(
            base_product, base_pricing_env, engine, adapter, request, lower
        )
        if is_close(f_lower, 0.0, abs_tol=self.price_tolerance):
            return lower, request.target.value

        f_upper = self._objective(
            base_product, base_pricing_env, engine, adapter, request, upper
        )
        if is_close(f_upper, 0.0, abs_tol=self.price_tolerance):
            return upper, request.target.value

        if f_lower * f_upper > 0:
            raise PricingError(
                "RFQ target is not bracketed by the supplied unknown bounds"
            )

        x_low = lower
        x_high = upper
        y_low = f_lower
        y_high = f_upper
        x_mid = request.unknown.initial_guess

        for _ in range(self.max_iterations):
            if x_mid is None or not (x_low < x_mid < x_high):
                if is_zero(y_high - y_low, tol=Tolerance.ZERO):
                    x_mid = 0.5 * (x_low + x_high)
                else:
                    secant = x_high - y_high * (x_high - x_low) / (y_high - y_low)
                    if x_low < secant < x_high:
                        x_mid = secant
                    else:
                        x_mid = 0.5 * (x_low + x_high)

            y_mid = self._objective(
                base_product, base_pricing_env, engine, adapter, request, x_mid
            )
            if is_close(y_mid, 0.0, abs_tol=self.price_tolerance):
                achieved = self._evaluate_candidate(
                    base_product, base_pricing_env, engine, adapter, x_mid
                )
                return x_mid, achieved

            if y_low * y_mid < 0:
                x_high = x_mid
                y_high = y_mid
            else:
                x_low = x_mid
                y_low = y_mid

            if abs(x_high - x_low) <= self.value_tolerance:
                best = 0.5 * (x_low + x_high)
                achieved = self._evaluate_candidate(
                    base_product, base_pricing_env, engine, adapter, best
                )
                if is_close(
                    achieved, request.target.value, abs_tol=self.price_tolerance
                ):
                    return best, achieved
                break

            x_mid = None

        raise PricingError("RFQ solver did not converge within max_iterations")


def quote_rfq(request: RFQRequest, **service_kwargs: Any) -> RFQQuote:
    """Convenience wrapper around RFQService.quote."""
    return RFQService(**service_kwargs).quote(request)

"""
Unified PDE pricing engine with automatic product-to-solver dispatch.

This module provides a single PDEEngine that implements the BaseEngine interface
and automatically routes pricing requests to the appropriate PDE solver based on
the product type.
"""

from typing import Dict, Optional, Type, Sequence

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.event_stats import AutocallableEventStats
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param.pde_removed_names import reject_removed_kwargs
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import (
    AmericanOption,
    BarrierOption,
    DoubleBarrierOption,
    DoubleOneTouchOption,
    EuropeanVanillaOption,
    OneTouchOption,
    SnowballOption,
    KnockOutResetSnowballOption,
    PhoenixOption,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

from .pde import (
    AmericanPDESolver,
    BarrierPDESolver,
    BasePDESolver,
    DoubleBarrierPDESolver,
    DoubleOneTouchPDESolver,
    EuropeanPDESolver,
    OneTouchPDESolver,
    SnowballPDESolver,
    KOResetSnowballPDESolver,
    PhoenixPDESolver,
)


class PDEEngine(BaseEngine):
    """
    Unified PDE pricing engine with automatic dispatch to product-specific solvers.

    This engine implements the BaseEngine interface and routes pricing requests
    to the appropriate PDE solver based on the product type. This enables:
    - Seamless integration with GreeksCalculator for numerical Greeks
    - Consistent API across all pricing engines (analytical, MC, PDE)
    - Simplified user experience with single engine instantiation

    Supported Products:
        - EuropeanVanillaOption → EuropeanPDESolver
        - AmericanOption → AmericanPDESolver
        - BarrierOption → BarrierPDESolver
        - DoubleBarrierOption → DoubleBarrierPDESolver
        - OneTouchOption → OneTouchPDESolver
        - DoubleOneTouchOption → DoubleOneTouchPDESolver
        - SnowballOption → SnowballPDESolver
        - KnockOutResetSnowballOption → KOResetSnowballPDESolver
        - PhoenixOption → PhoenixPDESolver

    Usage:
        # Basic usage — grid resolution via the accuracy profile
        engine = PDEEngine(PDEParams(accuracy="high"))
        price = engine.price(european_option, pricing_env)

        # Expert grid control (field-by-field GridConfig overlay)
        engine = PDEEngine(PDEParams(grid=GridConfig(points=500)))

        # Scheme weight: 0.5 is Crank-Nicolson (default), 1.0 backward Euler
        engine = PDEEngine(PDEParams(theta=1.0))

        # Integration with GreeksCalculator
        greeks = calculator.calculate_numerical_greeks(
            product=american_option,
            pricing_env=pricing_env,
            engine=engine
        )
    """

    engine_type = EngineType.PDE

    PRODUCT_SOLVER_MAP: Dict[Type[BaseEquityProduct], Type[BasePDESolver]] = {
        EuropeanVanillaOption: EuropeanPDESolver,
        AmericanOption: AmericanPDESolver,
        BarrierOption: BarrierPDESolver,
        DoubleBarrierOption: DoubleBarrierPDESolver,
        OneTouchOption: OneTouchPDESolver,
        DoubleOneTouchOption: DoubleOneTouchPDESolver,
        SnowballOption: SnowballPDESolver,
        KnockOutResetSnowballOption: KOResetSnowballPDESolver,
        PhoenixOption: PhoenixPDESolver,
    }

    def __init__(self, params: Optional[PDEParams] = None, **removed):
        """Initialize the unified PDE engine.

        Args:
            params: PDE configuration (accuracy, grid overlay, scheme settings).
        """
        reject_removed_kwargs("PDEEngine", removed)
        self.params = params if params is not None else PDEParams()
        super().__init__(self.params)
        self._solver_cache: Dict[Type[BaseEquityProduct], BasePDESolver] = {}

    def _get_solver(self, product: BaseEquityProduct) -> BasePDESolver:
        """
        Get the appropriate PDE solver for the given product type.

        Uses caching to avoid recreating solvers for the same product type.

        Args:
            product: The product to price

        Returns:
            PDE solver instance

        Raises:
            ValidationError: If product type is not supported
        """
        product_type = type(product)

        if product_type in self._solver_cache:
            return self._solver_cache[product_type]

        solver_class = self.PRODUCT_SOLVER_MAP.get(product_type)

        if solver_class is None:
            supported_types = [cls.__name__ for cls in self.PRODUCT_SOLVER_MAP.keys()]
            raise ValidationError(
                f"PDEEngine does not support product type {product_type.__name__}. "
                f"Supported types: {', '.join(supported_types)}"
            )

        solver = solver_class(params=self.params)
        self._solver_cache[product_type] = solver
        return solver

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a derivative product using the appropriate PDE solver.

        Automatically dispatches to the correct solver based on product type.

        Args:
            product: The derivative product to price
            pricing_env: Pricing environment with market data

        Returns:
            Product price

        Raises:
            ValidationError: If product is None or unsupported type
        """
        if product is None:
            raise ValidationError("Product cannot be None")

        solver = self._get_solver(product)
        return solver.price(product, pricing_env)

    def create_bump_context(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> "PDEEngine":
        """
        Return a facade engine whose dispatched solver has the base-market
        grid layout frozen.

        The frozen layout lives on the SOLVER INSTANCE (``_frozen_base_layout``,
        spec §4.8), so the facade clone must carry that exact instance in its
        dispatch cache — rebuilding a solver from params would silently drop
        the freeze.
        """
        solver = self._get_solver(product)
        fixed_solver = solver.create_bump_context(product, pricing_env)
        if fixed_solver is solver:
            return self
        clone = type(self)(params=self.params)
        clone._solver_cache[type(product)] = fixed_solver
        return clone

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """
        Forward to the product's solver, which owns the computation.

        The base class returns None for "unsupported"; that default silently
        discarded working solver results (SnowballPDESolver et al. implement
        exact PDE event statistics), and the replay layer's fail-closed
        event-stats policy turned the discard into a hard failure. A product
        type outside PRODUCT_SOLVER_MAP still raises ValidationError from
        _get_solver (same as price()) rather than being swallowed into None:
        that failure means "PDEEngine cannot handle this product at all",
        which is a different condition from a supported product whose solver
        has no event-stats implementation.

        Args:
            product: The derivative product
            pricing_env: Pricing environment with market data

        Returns:
            Event stats, or None if the dispatched solver does not support
            them for this product.

        Raises:
            ValidationError: If product is None or unsupported type
        """
        if product is None:
            raise ValidationError("Product cannot be None")

        solver = self._get_solver(product)
        return solver.calculate_event_stats(product, pricing_env)

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks using the appropriate PDE solver's grid-based method.

        Automatically dispatches to the correct solver based on product type.

        Args:
            product: The derivative product
            pricing_env: Pricing environment with market data

        Returns:
            Dictionary of Greeks (price, delta, gamma)
        """
        if product is None:
            raise ValidationError("Product cannot be None")

        solver = self._get_solver(product)
        return solver.calculate_greeks(product, pricing_env)

    def calculate_spot_greeks_curve(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot_levels: Sequence[float],
    ) -> list[dict[str, float | str]]:
        solver = self._get_solver(product)
        return solver.calculate_spot_greeks_curve(product, pricing_env, spot_levels)

    def explain(self, product, pricing_env, *, lifecycle_state=None):
        """The dispatched solver's plan (``BasePDESolver.explain``)."""
        return self._get_solver(product).explain(product, pricing_env, lifecycle_state=lifecycle_state)

    def solve_life_surface(self, product, pricing_env, *, extra_times):
        """One solve with extra time nodes, both value slabs (life-surface providers)."""
        solver = self._get_solver(product)
        seam = getattr(solver, "solve_life_surface", None)
        if seam is None:
            raise ValidationError(f"{type(solver).__name__} does not expose a life surface")
        return seam(product, pricing_env, extra_times=extra_times)

    def __repr__(self):
        return f"PDEEngine(accuracy={self.params.accuracy!r})"

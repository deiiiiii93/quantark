"""
Unified PDE pricing engine with automatic product-to-solver dispatch.

This module provides a single PDEEngine that implements the BaseEngine interface
and automatically routes pricing requests to the appropriate PDE solver based on
the product type.
"""

from typing import Optional, Union, Dict, Type
from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import (
    EuropeanVanillaOption,
    AmericanOption,
    BarrierOption,
    DoubleBarrierOption,
    OneTouchOption,
    DoubleOneTouchOption,
)
from asset.equity.param import PDEParams
from priceenv import PricingEnvironment
from util.exceptions import ValidationError
from util.enum.engine_enums import PDEMethod, EngineType

from .pde import (
    BasePDESolver,
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
    DoubleOneTouchPDESolver,
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

    Usage:
        # Basic usage
        engine = PDEEngine(PDEParams(num_space_steps=500))
        price = engine.price(european_option, pricing_env)

        # With method selection (two-level enum pattern)
        engine = PDEEngine(
            params=PDEParams(),
            method=EngineType.PDE(PDEMethod.CRANK_NICOLSON)
        )

        # Integration with GreeksCalculator
        greeks = calculator.calculate_numerical_greeks(
            product=american_option,
            pricing_env=pricing_env,
            engine=engine
        )
    """

    PRODUCT_SOLVER_MAP: Dict[Type[BaseEquityProduct], Type[BasePDESolver]] = {
        EuropeanVanillaOption: EuropeanPDESolver,
        AmericanOption: AmericanPDESolver,
        BarrierOption: BarrierPDESolver,
        DoubleBarrierOption: DoubleBarrierPDESolver,
        OneTouchOption: OneTouchPDESolver,
        DoubleOneTouchOption: DoubleOneTouchPDESolver,
    }

    DEFAULT_METHOD = PDEMethod.CRANK_NICOLSON

    def __init__(
        self,
        params: Optional[PDEParams] = None,
        method: Optional[Union[str, PDEMethod, tuple]] = None,
    ):
        """
        Initialize unified PDE engine.

        Args:
            params: PDE configuration parameters (grid size, scheme, smoothing)
            method: PDE method selection, can be:
                - PDEMethod enum (e.g., PDEMethod.CRANK_NICOLSON)
                - String "crank_nicolson"/"explicit_euler"/"implicit_euler"
                - Tuple from EngineType.PDE(PDEMethod.CRANK_NICOLSON)
                - None (defaults to Crank-Nicolson)

        Raises:
            ValidationError: If invalid method is specified
        """
        self.params = params if params is not None else PDEParams()
        super().__init__(self.params)

        if method is None:
            self.method = self.DEFAULT_METHOD
        elif isinstance(method, tuple):
            engine_type, pde_method = method
            if engine_type != EngineType.PDE:
                raise ValidationError(
                    f"Expected EngineType.PDE, got {engine_type}"
                )
            if not isinstance(pde_method, PDEMethod):
                raise ValidationError(
                    f"Expected PDEMethod, got {type(pde_method).__name__}"
                )
            self.method = pde_method
        elif isinstance(method, PDEMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = PDEMethod[method.upper()]
            except KeyError:
                raise ValidationError(
                    f"Invalid PDE method: {method}. "
                    f"Valid methods: {[m.value for m in PDEMethod]}"
                )
        else:
            raise ValidationError(
                f"Invalid method type: {type(method).__name__}. "
                f"Expected PDEMethod, str, or tuple."
            )

        self._apply_method_to_params()
        self._solver_cache: Dict[Type[BaseEquityProduct], BasePDESolver] = {}

    def _apply_method_to_params(self) -> None:
        """
        Apply the selected method to PDEParams scheme.

        Maps PDEMethod enum to the scheme string expected by PDE solvers.
        """
        self.params.scheme = self.method.value

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

    def __repr__(self):
        return f"PDEEngine(method={self.method.value})"

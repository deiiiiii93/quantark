"""
Unified facade engine for convertible bond pricing.

This module provides ConvertibleBondEngine that dispatches to appropriate
underlying engines based on method selection.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Union

from asset.bond.product.convertible.convertible_bond import ConvertibleBond
from asset.bond.engine.tree.convertible import (
    ConvertibleBondTreeParams,
    ConvertibleBondBinomialEngine,
    ConvertibleBondTrinomialEngine,
)
from asset.bond.engine.pde.convertible import (
    ConvertibleBondPDEParams,
    ConvertibleBondJumpDiffusionEngine,
    ConvertibleBondTFEngine,
)
from priceenv import PricingEnvironment
from util.enum.engine_enums import EngineType, ConvertibleBondMethod, PDEMethod
from util.exceptions import ValidationError


@dataclass
class ConvertibleBondResult:
    """
    Comprehensive result container for convertible bond pricing.

    Attributes:
        price: Clean price of the convertible bond
        dirty_price: Dirty price including accrued interest
        delta: Price sensitivity to stock price
        gamma: Second derivative of price with respect to stock
        conversion_probability: Probability of eventual conversion
        equity_component: Equity-like component of value
        bond_component: Bond-like component of value (COCB for TF model)
        default_probability: Probability of default (trinomial model only)
        method: Method used for pricing
    """

    price: float
    dirty_price: float
    delta: float = 0.0
    gamma: float = 0.0
    conversion_probability: float = 0.0
    equity_component: float = 0.0
    bond_component: float = 0.0
    default_probability: float = 0.0
    method: str = ""


class ConvertibleBondEngine:
    """
    Unified facade engine for convertible bond pricing.

    This engine dispatches pricing requests to specialized engines based on
    the selected method. It supports the two-level enum pattern consistent
    with other engines in the library.

    Supported Methods:
        Tree-based:
            - BINOMIAL_GS: Goldman Sachs credit-adjusted binomial model
            - TRINOMIAL_HW: Hull-White trinomial with default

        PDE-based:
            - JUMP_DIFFUSION: Bloomberg OVCV jump-diffusion model
            - TF: Tsiveriotis-Fernandes decomposition

    Usage:
        # Using two-level enum pattern
        engine = ConvertibleBondEngine(
            pricing_env,
            method=EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS)
        )

        # Using single method enum
        engine = ConvertibleBondEngine(
            pricing_env,
            method=ConvertibleBondMethod.JUMP_DIFFUSION
        )

        # Using string
        engine = ConvertibleBondEngine(pricing_env, method="binomial_gs")

        # Pricing
        price = engine.price(convertible_bond)
        result = engine.price_with_details(convertible_bond)
    """

    # Map methods to engine types
    TREE_METHODS = {
        ConvertibleBondMethod.BINOMIAL_GS,
        ConvertibleBondMethod.TRINOMIAL_HW,
    }

    PDE_METHODS = {
        ConvertibleBondMethod.JUMP_DIFFUSION,
        ConvertibleBondMethod.TF,
    }

    DEFAULT_METHOD = ConvertibleBondMethod.BINOMIAL_GS

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        method: Optional[
            Union[str, ConvertibleBondMethod, tuple]
        ] = None,
        tree_params: Optional[ConvertibleBondTreeParams] = None,
        pde_params: Optional[ConvertibleBondPDEParams] = None,
        scheme: Optional[Union[str, PDEMethod]] = None,
    ):
        """
        Initialize the facade engine.

        Args:
            pricing_env: Pricing environment with market data
            method: Pricing method selection, can be:
                - ConvertibleBondMethod enum
                - String (e.g., "binomial_gs", "jump_diffusion")
                - Tuple from EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS)
                - None (defaults to BINOMIAL_GS)
            tree_params: Configuration for tree-based engines (optional)
            pde_params: Configuration for PDE-based engines (optional)
            scheme: PDE numerical scheme (optional, for PDE methods only)

        Raises:
            ValidationError: If invalid method or configuration
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        self.pricing_env = pricing_env
        self.tree_params = tree_params
        self.pde_params = pde_params
        self.scheme = scheme

        # Parse method
        self.method = self._parse_method(method)

        # Create appropriate underlying engine
        self._engine = self._create_engine()

    def _parse_method(
        self, method: Optional[Union[str, ConvertibleBondMethod, tuple]]
    ) -> ConvertibleBondMethod:
        """
        Parse the method argument into a ConvertibleBondMethod enum.

        Args:
            method: Method specification

        Returns:
            ConvertibleBondMethod enum value

        Raises:
            ValidationError: If invalid method
        """
        if method is None:
            return self.DEFAULT_METHOD

        if isinstance(method, ConvertibleBondMethod):
            return method

        if isinstance(method, str):
            try:
                return ConvertibleBondMethod(method.lower())
            except ValueError:
                valid_methods = [m.value for m in ConvertibleBondMethod]
                raise ValidationError(
                    f"Invalid method: {method}. Valid methods: {valid_methods}"
                )

        if isinstance(method, tuple):
            if len(method) != 2:
                raise ValidationError(
                    f"Invalid method tuple: expected (EngineType, Method), got {method}"
                )

            engine_type, cb_method = method

            # Validate engine type matches method
            if engine_type == EngineType.TREE:
                if cb_method not in self.TREE_METHODS:
                    raise ValidationError(
                        f"Method {cb_method} is not a tree method"
                    )
            elif engine_type == EngineType.PDE:
                if cb_method not in self.PDE_METHODS:
                    raise ValidationError(
                        f"Method {cb_method} is not a PDE method"
                    )
            else:
                raise ValidationError(
                    f"Unsupported engine type: {engine_type}. "
                    f"Expected TREE or PDE."
                )

            if not isinstance(cb_method, ConvertibleBondMethod):
                raise ValidationError(
                    f"Expected ConvertibleBondMethod, got {type(cb_method)}"
                )

            return cb_method

        raise ValidationError(
            f"Invalid method type: {type(method)}. "
            f"Expected ConvertibleBondMethod, str, or tuple."
        )

    def _create_engine(self):
        """
        Create the appropriate underlying engine based on method.

        Returns:
            Specialized pricing engine instance
        """
        if self.method == ConvertibleBondMethod.BINOMIAL_GS:
            return ConvertibleBondBinomialEngine(
                self.pricing_env, self.tree_params
            )

        elif self.method == ConvertibleBondMethod.TRINOMIAL_HW:
            return ConvertibleBondTrinomialEngine(
                self.pricing_env, self.tree_params
            )

        elif self.method == ConvertibleBondMethod.JUMP_DIFFUSION:
            params = self.pde_params or ConvertibleBondPDEParams()
            if self.scheme:
                if isinstance(self.scheme, PDEMethod):
                    params.scheme = self.scheme.value
                else:
                    params.scheme = str(self.scheme)
            return ConvertibleBondJumpDiffusionEngine(self.pricing_env, params)

        elif self.method == ConvertibleBondMethod.TF:
            params = self.pde_params or ConvertibleBondPDEParams()
            if self.scheme:
                if isinstance(self.scheme, PDEMethod):
                    params.scheme = self.scheme.value
                else:
                    params.scheme = str(self.scheme)
            return ConvertibleBondTFEngine(self.pricing_env, params)

        else:
            raise ValidationError(f"Unsupported method: {self.method}")

    def price(self, bond: ConvertibleBond) -> float:
        """
        Calculate the clean price of the convertible bond.

        Args:
            bond: Convertible bond to price

        Returns:
            Clean price
        """
        return self._engine.price(bond)

    def price_with_details(self, bond: ConvertibleBond) -> ConvertibleBondResult:
        """
        Calculate price with detailed results.

        Dispatches to the appropriate underlying engine and converts
        the result to a unified ConvertibleBondResult.

        Args:
            bond: Convertible bond to price

        Returns:
            ConvertibleBondResult with full pricing details
        """
        raw_result = self._engine.price_with_details(bond)

        # Convert to unified result format
        result = ConvertibleBondResult(
            price=raw_result.price,
            dirty_price=raw_result.dirty_price,
            delta=getattr(raw_result, "delta", 0.0),
            gamma=getattr(raw_result, "gamma", 0.0),
            method=self.method.value,
        )

        # Extract method-specific fields
        if hasattr(raw_result, "conversion_probability"):
            result.conversion_probability = raw_result.conversion_probability

        if hasattr(raw_result, "equity_component"):
            result.equity_component = raw_result.equity_component

        if hasattr(raw_result, "bond_component"):
            result.bond_component = raw_result.bond_component

        if hasattr(raw_result, "default_probability"):
            result.default_probability = raw_result.default_probability

        return result

    def calculate_delta(self, bond: ConvertibleBond) -> float:
        """
        Calculate delta (price sensitivity to stock price).

        Args:
            bond: Convertible bond

        Returns:
            Delta
        """
        if hasattr(self._engine, "calculate_delta"):
            return self._engine.calculate_delta(bond)
        result = self.price_with_details(bond)
        return result.delta

    def calculate_gamma(self, bond: ConvertibleBond) -> float:
        """
        Calculate gamma (second derivative with respect to stock).

        Args:
            bond: Convertible bond

        Returns:
            Gamma
        """
        if hasattr(self._engine, "calculate_gamma"):
            return self._engine.calculate_gamma(bond)
        result = self.price_with_details(bond)
        return result.gamma

    def get_cocb(self, bond: ConvertibleBond) -> float:
        """
        Get Cash-Only Component of Bond (TF model only).

        Args:
            bond: Convertible bond

        Returns:
            COCB value

        Raises:
            ValidationError: If not using TF model
        """
        if self.method != ConvertibleBondMethod.TF:
            raise ValidationError(
                "COCB is only available with the TF model"
            )
        return self._engine.get_cocb(bond)

    def __repr__(self):
        return f"ConvertibleBondEngine(method={self.method.value})"

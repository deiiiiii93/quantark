"""
Unified facade engine for convertible bond pricing.

This module provides ConvertibleBondEngine that dispatches to appropriate
underlying engines based on method selection.
"""
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Union

from quantark.asset.bond.product.convertible.convertible_bond import ConvertibleBond
from quantark.asset.bond.engine.tree.convertible import (
    ConvertibleBondTreeParams,
    ConvertibleBondBinomialEngine,
    ConvertibleBondTrinomialEngine,
)
from quantark.asset.bond.engine.pde.convertible import (
    ConvertibleBondPDEParams,
    ConvertibleBondJumpDiffusionEngine,
    ConvertibleBondTFEngine,
)
from quantark.priceenv import PricingEnvironment
from quantark.param.rrf import ParallelShiftRateCurve
from quantark.util.enum.engine_enums import EngineType, ConvertibleBondMethod, PDEMethod
from quantark.util.exceptions import ValidationError


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
        floor_bond_price: Straight bond price without conversion/options
        floor_bond_dv01: Floor bond DV01 (price change per bp rate move)
        floor_bond_cs01: Floor bond CS01 (price change per bp spread move)
        floor_bond_duration: Floor bond modified duration
        floor_bond_convexity: Floor bond convexity
        dv01: Convertible DV01 (price change per bp rate move)
        cs01: Convertible CS01 (price change per bp spread move)
        modified_duration: Convertible modified duration
        convexity: Convertible convexity
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
    # Floor bond metrics
    floor_bond_price: float = 0.0
    floor_bond_dv01: float = 0.0
    floor_bond_cs01: float = 0.0
    floor_bond_duration: float = 0.0
    floor_bond_convexity: float = 0.0
    # Convertible risk metrics
    dv01: float = 0.0
    cs01: float = 0.0
    modified_duration: float = 0.0
    convexity: float = 0.0


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

    def price_with_details(
        self, bond: ConvertibleBond, include_risk_metrics: bool = True
    ) -> ConvertibleBondResult:
        """
        Calculate price with detailed results.

        Dispatches to the appropriate underlying engine and converts
        the result to a unified ConvertibleBondResult.

        Args:
            bond: Convertible bond to price
            include_risk_metrics: Whether to compute risk metrics (DV01, CS01,
                duration, convexity). Set to False to skip for performance.

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

        # Compute risk metrics if requested
        if include_risk_metrics:
            # Floor bond metrics (analytical, fast)
            result.floor_bond_price = self.floor_bond_price(bond)
            result.floor_bond_duration = self.floor_bond_duration(bond)
            result.floor_bond_convexity = self.floor_bond_convexity(bond)
            result.floor_bond_dv01 = self.floor_bond_dv01(bond)
            result.floor_bond_cs01 = self.floor_bond_cs01(bond)

            # Convertible risk metrics (numerical, slower)
            result.dv01 = self.dv01(bond)
            result.cs01 = self.cs01(bond)
            result.modified_duration = self.modified_duration(bond)
            result.convexity = self.convexity(bond)

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

    # =========================================================================
    # Floor Bond Methods
    # =========================================================================

    def _floor_bond_price_with_env(
        self, bond: ConvertibleBond, pricing_env: PricingEnvironment
    ) -> float:
        """
        Calculate floor bond (straight bond) price in a given environment.

        The floor bond is the value of the convertible assuming no conversion
        and no exercise of call/put options. It represents the investment
        value floor of the convertible bond.

        Cashflows are discounted at the risky rate (risk-free + credit spread).

        Args:
            bond: Convertible bond product
            pricing_env: Pricing environment to use

        Returns:
            Floor bond dirty price
        """
        valuation_date = pricing_env.valuation_date

        # Check if bond has matured
        if bond.is_expired(valuation_date):
            return 0.0

        # Get future cashflows
        cashflows = bond.get_cashflows(valuation_date)

        if not cashflows:
            return 0.0

        # Get credit spread from bond
        credit_spread = bond.credit_spread if bond.credit_spread else 0.0

        # Discount each cashflow at risky rate
        pv = 0.0
        for cf in cashflows:
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0

            if time_to_payment < 0:
                continue

            # Risk-free discount factor from curve; apply spread as parallel shift
            df = pricing_env.get_discount_factor(time_to_payment)
            risky_df = df * math.exp(-credit_spread * time_to_payment)
            pv += cf.amount * risky_df

        return pv

    def floor_bond_price(self, bond: ConvertibleBond) -> float:
        """
        Calculate the floor bond (straight bond) price.

        Args:
            bond: Convertible bond product

        Returns:
            Floor bond dirty price
        """
        return self._floor_bond_price_with_env(bond, self.pricing_env)

    def floor_bond_dv01(self, bond: ConvertibleBond) -> float:
        """
        Calculate DV01 of the floor bond.

        DV01 is the price change for a 1 basis point parallel increase in rates.
        Uses numerical bumping to remain consistent under non-flat curves.

        Args:
            bond: Convertible bond product

        Returns:
            Floor bond DV01 (positive value, price decreases when rates rise)
        """
        rate_bump = 0.0001
        base_price = self.floor_bond_price(bond)
        if base_price == 0.0:
            return 0.0

        env_up = deepcopy(self.pricing_env)
        env_up.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=rate_bump
        )
        price_up = self._floor_bond_price_with_env(bond, env_up)

        env_down = deepcopy(self.pricing_env)
        env_down.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=-rate_bump
        )
        price_down = self._floor_bond_price_with_env(bond, env_down)

        return (price_down - price_up) / 2.0

    def floor_bond_cs01(self, bond: ConvertibleBond) -> float:
        """
        Calculate CS01 of the floor bond.

        CS01 is the price change for a 1 basis point increase in credit spread.
        For the floor bond, CS01 equals DV01 since both rate and spread
        affect discounting identically (discount at r + s).

        Args:
            bond: Convertible bond product

        Returns:
            Floor bond CS01 (equals DV01)
        """
        # For floor bond, CS01 = DV01 since both enter the discount factor
        return self.floor_bond_dv01(bond)

    def floor_bond_duration(self, bond: ConvertibleBond) -> float:
        """
        Calculate modified duration of the floor bond.

        Derived from DV01: Duration = DV01 / (Price * 0.0001).

        Args:
            bond: Convertible bond product

        Returns:
            Floor bond modified duration
        """
        price = self.floor_bond_price(bond)
        if price == 0.0:
            return 0.0
        dv01_value = self.floor_bond_dv01(bond)
        return dv01_value / (price * 0.0001)

    def floor_bond_convexity(self, bond: ConvertibleBond) -> float:
        """
        Calculate convexity of the floor bond.

        Convexity measures the curvature of the price-yield relationship.

        Args:
            bond: Convertible bond product

        Returns:
            Floor bond convexity
        """
        rate_bump = 0.0001
        base_price = self.floor_bond_price(bond)
        if base_price == 0.0:
            return 0.0

        env_up = deepcopy(self.pricing_env)
        env_up.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=rate_bump
        )
        price_up = self._floor_bond_price_with_env(bond, env_up)

        env_down = deepcopy(self.pricing_env)
        env_down.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=-rate_bump
        )
        price_down = self._floor_bond_price_with_env(bond, env_down)

        return (price_up + price_down - 2.0 * base_price) / (
            base_price * rate_bump * rate_bump
        )

    # =========================================================================
    # Convertible Bond Risk Metrics (Numerical)
    # =========================================================================

    def dv01(self, bond: ConvertibleBond) -> float:
        """
        Calculate DV01 of the convertible bond.

        Uses numerical rate bumping since the convertible has embedded options.
        Only the risk-free rate is bumped, isolating interest rate risk.

        Args:
            bond: Convertible bond product

        Returns:
            Convertible DV01 (positive value, price decreases when rates rise)
        """
        rate_bump = 0.0001  # 1 basis point

        # Price with rate bumped up
        env_up = deepcopy(self.pricing_env)
        env_up.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=rate_bump
        )
        engine_up = self._create_bumped_engine(env_up)
        price_up = engine_up.price(bond)

        # Price with rate bumped down
        env_down = deepcopy(self.pricing_env)
        env_down.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=-rate_bump
        )
        engine_down = self._create_bumped_engine(env_down)
        price_down = engine_down.price(bond)

        # DV01 = (price_down - price_up) / 2 (central difference)
        # Note: price falls when rate rises, so DV01 is positive
        return (price_down - price_up) / 2

    def cs01(self, bond: ConvertibleBond) -> float:
        """
        Calculate CS01 of the convertible bond.

        Uses numerical credit bumping.

        - For BINOMIAL_GS: bumps credit_spread directly (GS credit-adjusted discounting).
        - For TRINOMIAL_HW / JUMP_DIFFUSION / TF: bumps hazard_rate using an
          approximate mapping from spread shift to intensity shift:
              d(lambda) ~= d(spread) / (1 - recovery)

        Args:
            bond: Convertible bond product

        Returns:
            Convertible CS01 (positive value, price decreases when spread rises)
        """
        spread_bump = 0.0001  # 1 basis point

        if self.method == ConvertibleBondMethod.BINOMIAL_GS:
            # Create bond with bumped spread up
            base_spread = bond.credit_spread if bond.credit_spread else 0.0
            bond_up = self._create_spread_bumped_bond(
                bond, base_spread + spread_bump
            )
            price_up = self.price(bond_up)

            # Create bond with bumped spread down
            bond_down = self._create_spread_bumped_bond(
                bond, max(0.0, base_spread - spread_bump)
            )
            price_down = self.price(bond_down)

            # CS01 = (price_down - price_up) / 2 (central difference)
            return (price_down - price_up) / 2

        # Hazard-based models: bump hazard_rate
        recovery = bond.recovery_rate if bond.recovery_rate is not None else 0.0
        denom = max(1e-8, 1.0 - recovery)
        hazard_bump = spread_bump / denom

        base_hazard = bond.hazard_rate if bond.hazard_rate else 0.0
        bond_up = self._create_hazard_bumped_bond(
            bond, base_hazard + hazard_bump
        )
        price_up = self.price(bond_up)

        bond_down = self._create_hazard_bumped_bond(
            bond, max(0.0, base_hazard - hazard_bump)
        )
        price_down = self.price(bond_down)

        # CS01 = (price_down - price_up) / 2 (central difference)
        return (price_down - price_up) / 2

    def modified_duration(self, bond: ConvertibleBond) -> float:
        """
        Calculate modified duration of the convertible bond.

        Derived from DV01: Duration = DV01 / (Price * 0.0001)

        Args:
            bond: Convertible bond product

        Returns:
            Convertible modified duration
        """
        price = self.price(bond)
        if price == 0:
            return 0.0

        dv01_value = self.dv01(bond)
        return dv01_value / (price * 0.0001)

    def convexity(self, bond: ConvertibleBond) -> float:
        """
        Calculate convexity of the convertible bond.

        Uses central difference with rate bumps.

        Args:
            bond: Convertible bond product

        Returns:
            Convertible convexity
        """
        rate_bump = 0.0001  # 1 basis point
        base_price = self.price(bond)

        if base_price == 0:
            return 0.0

        # Price with rate bumped up
        env_up = deepcopy(self.pricing_env)
        env_up.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=rate_bump
        )
        engine_up = self._create_bumped_engine(env_up)
        price_up = engine_up.price(bond)

        # Price with rate bumped down
        env_down = deepcopy(self.pricing_env)
        env_down.rate_curve = ParallelShiftRateCurve(
            self.pricing_env.rate_curve, shift=-rate_bump
        )
        engine_down = self._create_bumped_engine(env_down)
        price_down = engine_down.price(bond)

        # Convexity = (P_up + P_down - 2*P_base) / (P_base * bump^2)
        return (price_up + price_down - 2 * base_price) / (base_price * rate_bump * rate_bump)

    def _create_bumped_engine(self, bumped_env: PricingEnvironment):
        """
        Create a new engine instance with a bumped pricing environment.

        Args:
            bumped_env: Pricing environment with bumped parameters

        Returns:
            New engine instance
        """
        if self.method == ConvertibleBondMethod.BINOMIAL_GS:
            return ConvertibleBondBinomialEngine(bumped_env, self.tree_params)
        elif self.method == ConvertibleBondMethod.TRINOMIAL_HW:
            return ConvertibleBondTrinomialEngine(bumped_env, self.tree_params)
        elif self.method == ConvertibleBondMethod.JUMP_DIFFUSION:
            params = self.pde_params or ConvertibleBondPDEParams()
            return ConvertibleBondJumpDiffusionEngine(bumped_env, params)
        elif self.method == ConvertibleBondMethod.TF:
            params = self.pde_params or ConvertibleBondPDEParams()
            return ConvertibleBondTFEngine(bumped_env, params)
        else:
            raise ValidationError(f"Unsupported method: {self.method}")

    def _create_spread_bumped_bond(
        self, bond: ConvertibleBond, new_spread: float
    ) -> ConvertibleBond:
        """
        Create a copy of the bond with a bumped credit spread.

        Args:
            bond: Original convertible bond
            new_spread: New credit spread value

        Returns:
            New bond with modified credit spread
        """
        return ConvertibleBond(
            issue_date=bond.issue_date,
            maturity_date=bond.maturity_date,
            face_value=bond.face_value,
            coupon_rate=bond.coupon_rate,
            conversion_ratio=bond.conversion_ratio,
            conversion_price=bond.conversion_price,
            payment_frequency=bond.payment_frequency,
            day_count_convention=bond.day_count_convention,
            conversion_start_date=bond.conversion_start_date,
            conversion_end_date=bond.conversion_end_date,
            call_schedule=bond.call_schedule,
            put_schedule=bond.put_schedule,
            credit_spread=new_spread,
            hazard_rate=bond.hazard_rate,
            recovery_rate=bond.recovery_rate,
            stock_jump_on_default=bond.stock_jump_on_default,
            continuous_dividend_yield=bond.continuous_dividend_yield,
            discrete_dividends=bond.discrete_dividends,
        )

    def _create_hazard_bumped_bond(
        self, bond: ConvertibleBond, new_hazard_rate: float
    ) -> ConvertibleBond:
        """
        Create a copy of the bond with a bumped hazard rate.

        Args:
            bond: Original convertible bond
            new_hazard_rate: New hazard rate value

        Returns:
            New bond with modified hazard rate
        """
        return ConvertibleBond(
            issue_date=bond.issue_date,
            maturity_date=bond.maturity_date,
            face_value=bond.face_value,
            coupon_rate=bond.coupon_rate,
            conversion_ratio=bond.conversion_ratio,
            conversion_price=bond.conversion_price,
            payment_frequency=bond.payment_frequency,
            day_count_convention=bond.day_count_convention,
            conversion_start_date=bond.conversion_start_date,
            conversion_end_date=bond.conversion_end_date,
            call_schedule=bond.call_schedule,
            put_schedule=bond.put_schedule,
            credit_spread=bond.credit_spread,
            hazard_rate=new_hazard_rate,
            recovery_rate=bond.recovery_rate,
            stock_jump_on_default=bond.stock_jump_on_default,
            continuous_dividend_yield=bond.continuous_dividend_yield,
            discrete_dividends=bond.discrete_dividends,
        )

    def __repr__(self):
        return f"ConvertibleBondEngine(method={self.method.value})"

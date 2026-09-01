"""
Greeks calculation for equity derivatives.
"""

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.bucketed_coordinates import (
    carry_rhoq,
    futures_delta,
    rate_keyrate,
    vol_model,
    vol_tenor_vega,
)
from quantark.asset.equity.riskmeasures.greeks import (
    analytical,
    bump_envs,
    numerical,
    registry,
    theta_decomposition,
)
from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
    BucketedGreeksRequest,
    BucketedGreeksResult,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import EngineType, GreeksCalculationMode
from quantark.util.exceptions import ValidationError


class GreeksCalculator:
    """
    Calculator for option Greeks using both analytical and numerical methods.

    Supports:
    - Analytical Greeks: Using closed-form Black-Scholes formulas
    - Numerical Greeks: Using finite difference method (FDM)

    The greeks_mode parameter controls delta/gamma calculation for engines that
    implement their own calculate_greeks() method (e.g., PDE engines):
    - GreeksCalculationMode.BUMP: Always use finite difference bump method
    - GreeksCalculationMode.ENGINE: Use engine.calculate_greeks() when overridden
    - GreeksCalculationMode.AUTO: Use engine method for PDE engines, bump otherwise
    """

    def __init__(
        self,
        params: Optional[EngineParams] = None,
        greeks_mode: GreeksCalculationMode = GreeksCalculationMode.BUMP,
    ):
        """
        Initialize Greeks calculator.

        Args:
            params: Engine parameters (for bump sizes in FDM)
            greeks_mode: Mode for delta/gamma calculation when engine has
                        its own calculate_greeks() method (e.g., PDE engines)
        """
        self.params = params if params is not None else EngineParams()
        self._bump_config = self.params.get_effective_bump_config()
        self.greeks_mode = greeks_mode

    def calculate(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        method: str = "auto",
        greeks: Optional[Sequence[object]] = None,
    ) -> Dict[str, float]:
        """Unified entry point for Greeks calculation."""
        method = method.lower()
        if method not in ("auto", "analytical", "numerical"):
            raise ValidationError(f"Unknown greeks method: {method}")

        requested = self._normalize_greeks(greeks)
        analytical_supported = registry.ANALYTICAL_AUTO_SET

        if method in ("auto", "analytical") and isinstance(
            product, EuropeanVanillaOption
        ):
            if requested is None or requested.issubset(analytical_supported):
                greeks_out = self.calculate_analytical_greeks(product, pricing_env)
                if requested is None:
                    return greeks_out
                return {key: greeks_out[key] for key in greeks_out if key in requested}
            if method == "analytical":
                raise ValidationError(
                    "Analytical greeks do not support requested greeks: "
                    f"{sorted(requested - analytical_supported)}"
                )

        return self.calculate_numerical_greeks(
            product, pricing_env, engine, greeks=greeks
        )

    def _normalize_greeks(
        self, greeks: Optional[Sequence[object]]
    ) -> Optional[set[str]]:
        requests = registry.normalize_greeks(greeks)
        if requests is None:
            return None
        return {req.key for req in requests}

    def _has_custom_greeks(self, engine: BaseEngine) -> bool:
        """Return True if engine overrides calculate_greeks()."""
        engine_calculate_greeks = getattr(engine.__class__, "calculate_greeks", None)
        return (
            engine_calculate_greeks is not None
            and engine_calculate_greeks is not BaseEngine.calculate_greeks
        )

    def _should_use_engine_greeks(self, engine: BaseEngine) -> bool:
        """
        Check if engine's calculate_greeks() should be used for delta/gamma.

        Args:
            engine: The pricing engine

        Returns:
            True if engine.calculate_greeks() should be used
        """
        if not self._has_custom_greeks(engine):
            return False

        if self.greeks_mode == GreeksCalculationMode.BUMP:
            return False
        if self.greeks_mode == GreeksCalculationMode.ENGINE:
            return True
        # AUTO mode: use for PDE engines
        return getattr(engine, "engine_type", None) == EngineType.PDE

    @staticmethod
    def _bump_term_vol_node(surface, node_index: int, bump: float):
        """Copy of a term ATM vol curve with one node's vol bumped."""
        return vol_tenor_vega.bump_term_vol_node(surface, node_index, bump)

    def _resolve_bucketed_coordinates(
        self, request: BucketedGreeksRequest
    ) -> Tuple[BucketedGreekCoordinate, ...]:
        if request.coordinates is not None:
            coordinates = tuple(request.coordinates)
        elif request.futures_curve is not None:
            coordinates = (
                BucketedGreekCoordinate.FUTURES_DELTA,
                BucketedGreekCoordinate.CARRY_RHOQ,
            )
        else:
            coordinates = (
                BucketedGreekCoordinate.VOL_TENOR_VEGA,
                BucketedGreekCoordinate.CARRY_RHOQ,
            )

        for coordinate in request.difference_mode_overrides:
            if coordinate not in coordinates:
                raise ValidationError(
                    f"difference_mode override coordinate {coordinate.value} "
                    "is not in the requested coordinate set"
                )
        return coordinates

    @staticmethod
    def _coordinate_default_difference_mode(
        coordinate: BucketedGreekCoordinate,
    ) -> BucketedGreekDifferenceMode:
        if coordinate in (
            BucketedGreekCoordinate.FUTURES_DELTA,
            BucketedGreekCoordinate.CARRY_RHOQ,
        ):
            return BucketedGreekDifferenceMode.ONE_SIDED_UP
        return BucketedGreekDifferenceMode.CENTRAL

    def _resolve_bucketed_difference_mode(
        self,
        coordinate: BucketedGreekCoordinate,
        request: BucketedGreeksRequest,
    ) -> BucketedGreekDifferenceMode:
        mode = request.difference_mode_overrides.get(
            coordinate, request.difference_mode
        )
        if mode == BucketedGreekDifferenceMode.COORDINATE_DEFAULT:
            mode = self._coordinate_default_difference_mode(coordinate)
        if (
            coordinate
            in (
                BucketedGreekCoordinate.MARKET_IV_VEGA,
                BucketedGreekCoordinate.MODEL_ARTIFACT,
            )
            and mode == BucketedGreekDifferenceMode.ONE_SIDED_UP
        ):
            raise ValidationError(
                f"{coordinate.value} does not support one_sided_up in v1; "
                "request central mode or extend VolModelRiskCalculator"
            )
        return mode

    def calculate_bucketed_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        request: Optional[BucketedGreeksRequest] = None,
    ) -> BucketedGreeksResult:
        request = request or BucketedGreeksRequest()
        coordinates = self._resolve_bucketed_coordinates(request)
        if (
            BucketedGreekCoordinate.FUTURES_DELTA in coordinates
            and request.futures_curve is None
        ):
            raise ValidationError("FUTURES_DELTA requires request.futures_curve")

        points: List[BucketedGreekPoint] = []
        result_metadata: dict = {}
        for coordinate in coordinates:
            mode = self._resolve_bucketed_difference_mode(coordinate, request)
            if coordinate == BucketedGreekCoordinate.RATE_KEYRATE:
                keyrate_points = rate_keyrate.calculate_points(
                    self, product, pricing_env, engine, request, mode
                )
                points.extend(keyrate_points)
                for pt in keyrate_points:
                    if pt.name == "rate_keyrate.parallel":
                        result_metadata.update(
                            {
                                "sum_of_buckets": pt.metadata["sum_of_buckets"],
                                "parallel": pt.reported,
                                "reconciles": pt.metadata["reconciles"],
                                "roles_inferred": pt.metadata.get(
                                    "roles_inferred"
                                ),
                            }
                        )
            elif coordinate == BucketedGreekCoordinate.FUTURES_DELTA:
                points.extend(
                    futures_delta.calculate_points(
                        self, product, pricing_env, engine, request, mode
                    )
                )
            elif coordinate == BucketedGreekCoordinate.CARRY_RHOQ:
                points.extend(
                    carry_rhoq.calculate_points(
                        self, product, pricing_env, engine, request, mode
                    )
                )
            elif coordinate == BucketedGreekCoordinate.VOL_TENOR_VEGA:
                points.extend(
                    vol_tenor_vega.calculate_points(
                        self, product, pricing_env, engine, request, mode
                    )
                )
            elif coordinate in (
                BucketedGreekCoordinate.MARKET_IV_VEGA,
                BucketedGreekCoordinate.MODEL_ARTIFACT,
            ):
                points.extend(
                    vol_model.calculate_points(
                        self, product, pricing_env, engine, request, coordinate, mode
                    )
                )
            else:
                raise ValidationError(
                    f"unsupported bucketed Greek coordinate: {coordinate}"
                )

        result_metadata["coordinates"] = tuple(
            coordinate.value for coordinate in coordinates
        )
        return BucketedGreeksResult(
            points=tuple(points),
            metadata=result_metadata,
        )

    def _ensure_base_price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float],
    ) -> float:
        """Return base price, computing it if needed."""
        return bump_envs.ensure_base_price(product, pricing_env, engine, base_price)

    def _resolve_bump_engine(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
    ) -> BaseEngine:
        """Return the engine context used for numerical bump repricing."""
        return bump_envs.resolve_bump_engine(product, pricing_env, engine)

    def _calculate_sensitivity(
        self,
        base_price: float,
        price_up: float,
        price_down: Optional[float] = None,
        bump: float = 1.0,
        scale: float = 1.0,
        mode: str = "central",
    ) -> float:
        """Generic finite-difference sensitivity helper."""
        return bump_envs.calculate_sensitivity(
            base_price, price_up, price_down, bump=bump, scale=scale, mode=mode
        )

    def _get_delta_gamma(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float],
    ) -> Tuple[float, float, float]:
        """Get base price, delta, and gamma via engine or bump method."""
        return numerical.get_delta_gamma(
            self, product, pricing_env, engine, base_price
        )

    def calculate_analytical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        price: Optional[float] = None,
    ) -> Dict[str, float]:
        """Closed-form BS greeks for European vanillas; see greeks.analytical."""
        return analytical.calculate_analytical_greeks(product, pricing_env, price)

    def calculate_numerical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        greeks: Optional[Sequence[object]] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference method (FDM).

        Uses central differences for better accuracy.
        Works for any product and engine combination.

        Bump sizes are configured via BumpConfig in EngineParams:
            - Delta/Gamma: Relative spot bump (default: 1%)
            - Vega: Absolute vol bump (default: 1 vol point)
            - Theta: Time bump in days (default: 1 day)
            - Rho: Absolute rate bump (default: 1bp), scaled to per 1% change
            - Dividend Rho: Absolute div yield bump (default: 1bp), scaled to per 1% change

        For delta and gamma, if greeks_mode is ENGINE or AUTO (with PDE engine),
        the engine's own calculate_greeks() method is used instead of bumping.

        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine to use
            base_price: Pre-calculated base price (optional)

        Returns:
            Dictionary of Greeks for the requested set (or defaults if None).
        """
        requested = self._normalize_greeks(greeks)
        if requested is None:
            requested = set(registry.DEFAULT_SET)

        if product.is_linear:
            base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
            greeks_out = self._greeks_for_linear(product, base_price)
            for extra in requested:
                greeks_out.setdefault(extra, 0.0)
            return {key: greeks_out[key] for key in greeks_out if key in requested}

        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        greeks_out: Dict[str, float] = {}
        if "price" in requested and base_price is not None:
            greeks_out["price"] = base_price

        delta = None
        gamma = None
        if {"delta", "gamma", "delta_q", "vanna"} & requested:
            base_price, delta, gamma = self._get_delta_gamma(
                product, pricing_env, bump_engine, base_price
            )
        if delta is not None and "delta" in requested:
            greeks_out["delta"] = delta
        if gamma is not None and "gamma" in requested:
            greeks_out["gamma"] = gamma

        if base_price is None:
            base_price = self._ensure_base_price(
                product, pricing_env, bump_engine, base_price
            )
        if "price" in requested and "price" not in greeks_out:
            greeks_out["price"] = base_price

        # Other Greeks always use bump method
        if "vega" in requested:
            greeks_out["vega"] = self.calculate_numerical_vega(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                vol_bump=self._bump_config.vol_bump,
            )
        if "volga" in requested:
            greeks_out["volga"] = self.calculate_numerical_volga(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                vol_bump=self._bump_config.vol_bump,
            )
        if "vanna" in requested:
            greeks_out["vanna"] = self.calculate_numerical_vanna(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                vol_bump=self._bump_config.vol_bump,
            )
        if "delta_q" in requested:
            greeks_out["delta_q"] = self.calculate_numerical_delta_q(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                div_bump=self._bump_config.div_bump,
                base_delta=delta,
            )
        if "theta" in requested:
            greeks_out["theta"] = self.calculate_numerical_theta(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                time_bump_days=self._bump_config.time_bump_days,
            )
        if "rho" in requested:
            greeks_out["rho"] = self.calculate_numerical_rho(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                rate_bump=self._bump_config.rate_bump,
            )
        if "dividend_rho" in requested:
            greeks_out["dividend_rho"] = self.calculate_numerical_dividend_rho(
                product,
                pricing_env,
                bump_engine,
                base_price=base_price,
                div_bump=self._bump_config.div_bump,
            )

        # Estimate theta components using fast approximation from existing Greeks
        if {"convexity_theta", "r_theta", "q_theta"} & requested:
            if "theta" not in greeks_out:
                greeks_out["theta"] = self.calculate_numerical_theta(
                    product,
                    pricing_env,
                    bump_engine,
                    base_price=base_price,
                    time_bump_days=self._bump_config.time_bump_days,
                )
            if "rho" not in greeks_out:
                greeks_out["rho"] = self.calculate_numerical_rho(
                    product,
                    pricing_env,
                    bump_engine,
                    base_price=base_price,
                    rate_bump=self._bump_config.rate_bump,
                )
            if "dividend_rho" not in greeks_out:
                greeks_out["dividend_rho"] = self.calculate_numerical_dividend_rho(
                    product,
                    pricing_env,
                    bump_engine,
                    base_price=base_price,
                    div_bump=self._bump_config.div_bump,
                )
            T = product.get_maturity(pricing_env)
            r = pricing_env.get_rate(T)
            q = pricing_env.get_div_yield(T)
            theta_components = self.estimate_theta_components(
                theta=greeks_out["theta"],
                rho=greeks_out["rho"],
                dividend_rho=greeks_out["dividend_rho"],
                r=r,
                q=q,
                T=T,
            )
            for key, value in theta_components.items():
                if key in requested:
                    greeks_out[key] = value

        return {key: greeks_out[key] for key in greeks_out if key in requested}

    def calculate_numerical_delta(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        spot_prices: Optional[Tuple[float, float]] = None,
        bump: Optional[float] = None,
    ) -> float:
        """Numerical delta using central spot bump."""
        return numerical.numerical_delta(
            self,
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=bump,
        )

    def calculate_numerical_gamma(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        spot_prices: Optional[Tuple[float, float]] = None,
        bump: Optional[float] = None,
    ) -> float:
        """Numerical gamma using central spot bump."""
        return numerical.numerical_gamma(
            self,
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=bump,
        )

    def calculate_numerical_vega(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        vol_bump: Optional[float] = None,
    ) -> float:
        """Numerical vega from a vol bump."""
        return numerical.numerical_vega(
            self, product, pricing_env, engine,
            base_price=base_price, vol_bump=vol_bump,
        )

    def calculate_numerical_volga(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        vol_bump: Optional[float] = None,
    ) -> float:
        """Numerical volga (second derivative wrt vol) using vol bumps."""
        return numerical.numerical_volga(
            self, product, pricing_env, engine,
            base_price=base_price, vol_bump=vol_bump,
        )

    def calculate_numerical_vanna(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        vol_bump: Optional[float] = None,
    ) -> float:
        """Numerical vanna (cross derivative wrt spot and vol)."""
        return numerical.numerical_vanna(
            self, product, pricing_env, engine,
            base_price=base_price, vol_bump=vol_bump,
        )

    def calculate_numerical_theta(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        time_bump_days: Optional[int] = None,
        time_bump_mode: Optional[str] = None,
    ) -> float:
        """
        Numerical theta via time bump with observation schedule handling.

        Theta date advancement is controlled by BumpConfig.time_bump_mode:
        "calendar_days" preserves legacy calendar-date bumps, "business_days"
        advances by valid pricing-calendar business days, and "auto" uses
        business days for BUSINESS_DAYS pricing environments with a calendar.
        """
        return numerical.numerical_theta(
            self,
            product,
            pricing_env,
            engine,
            base_price=base_price,
            time_bump_days=time_bump_days,
            time_bump_mode=time_bump_mode,
        )

    def _advance_theta_bump(
        self,
        pricing_env: PricingEnvironment,
        time_bump_days: int,
        time_bump_mode: str,
    ) -> Tuple[datetime, float, str]:
        """Advance the theta valuation date and return date, year fraction, mode."""
        return bump_envs.advance_theta_bump(
            pricing_env, time_bump_days, time_bump_mode
        )

    @staticmethod
    def _resolve_theta_bump_mode(
        pricing_env: PricingEnvironment, time_bump_mode: str
    ) -> str:
        """Resolve auto theta mode against the pricing environment."""
        return bump_envs.resolve_theta_bump_mode(pricing_env, time_bump_mode)

    def calculate_numerical_rho(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        rate_bump: Optional[float] = None,
    ) -> float:
        """Numerical rho from a rate bump (per 1% rate change)."""
        return numerical.numerical_rho(
            self, product, pricing_env, engine,
            base_price=base_price, rate_bump=rate_bump,
        )

    def calculate_numerical_dividend_rho(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        div_bump: Optional[float] = None,
    ) -> float:
        """Numerical dividend_rho (dV/dq, per 1% div_yield change)."""
        return numerical.numerical_dividend_rho(
            self, product, pricing_env, engine,
            base_price=base_price, div_bump=div_bump,
        )

    def calculate_numerical_delta_q(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        div_bump: Optional[float] = None,
        base_delta: Optional[float] = None,
    ) -> float:
        """Numerical dDelta/dq via dividend yield bumps."""
        return numerical.numerical_delta_q(
            self, product, pricing_env, engine,
            base_price=base_price, div_bump=div_bump, base_delta=base_delta,
        )

    def calculate_futures_delta_buckets(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        futures_curve,
        *,
        mode=None,
        price_bump: float = 1.0,
    ) -> List[Dict[str, object]]:
        """Futures-tenor bucket deltas; see bucketed_coordinates.futures_delta."""
        return futures_delta.calculate_futures_delta_buckets(
            self,
            product,
            pricing_env,
            engine,
            futures_curve,
            mode=mode,
            price_bump=price_bump,
        )

    def calculate_futures_rhoq_buckets(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        futures_curve,
        *,
        mode=None,
        div_bump: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        """Bucketed rhoq per futures tenor; see bucketed_coordinates.carry_rhoq."""
        return carry_rhoq.calculate_futures_rhoq_buckets(
            self,
            product,
            pricing_env,
            engine,
            futures_curve,
            mode=mode,
            div_bump=div_bump,
        )

    def estimate_theta_components(
        self,
        theta: float,
        rho: float,
        dividend_rho: float,
        r: float,
        q: float,
        T: float,
        rate_bump: float = 0.01,
        dividend_bump: float = 0.01,
    ) -> Dict[str, float]:
        """Fast theta component estimate; see greeks.theta_decomposition."""
        return theta_decomposition.estimate_theta_components(
            theta,
            rho,
            dividend_rho,
            r,
            q,
            T,
            rate_bump=rate_bump,
            dividend_bump=dividend_bump,
        )

    def _calculate_numerical_theta_components(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        time_bump_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Exact zeroed-r/q theta decomposition; see greeks.theta_decomposition."""
        return theta_decomposition.exact_theta_components(
            self,
            product,
            pricing_env,
            engine,
            base_price=base_price,
            time_bump_days=time_bump_days,
        )

    def _spot_bumped_prices(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        bump: float,
        base_price: Optional[float] = None,
        reuse: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute base, up, and down spot bump prices, optionally reusing bumps.
        """
        return bump_envs.spot_bumped_prices(
            product,
            pricing_env,
            engine,
            bump,
            base_price=base_price,
            reuse=reuse,
        )

    def _build_vol_bumped_env(
        self,
        pricing_env: PricingEnvironment,
        product: BaseEquityProduct,
        current_vol: float,
        vol_bump: float,
        *,
        direction: float,
    ) -> PricingEnvironment:
        return bump_envs.build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=direction
        )

    def _build_div_bumped_env(
        self,
        pricing_env: PricingEnvironment,
        product: BaseEquityProduct,
        current_div: float,
        div_bump: float,
        *,
        direction: float,
    ) -> PricingEnvironment:
        return bump_envs.build_div_bumped_env(
            pricing_env, product, current_div, div_bump, direction=direction
        )

    def _greeks_for_linear(
        self, product: BaseEquityProduct, price: float
    ) -> Dict[str, float]:
        """Greeks for linear (delta-one) products; see greeks.numerical."""
        return numerical.linear_greeks(product, price)

    def _greeks_at_expiry(
        self, product: EuropeanVanillaOption, spot: float
    ) -> Dict[str, float]:
        """Greeks at expiry; see greeks.analytical.greeks_at_expiry."""
        return analytical.greeks_at_expiry(product, spot)

    def compare_greeks(
        self, analytical: Dict[str, float], numerical: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare analytical and numerical Greeks.

        Args:
            analytical: Analytical Greeks
            numerical: Numerical Greeks

        Returns:
            Dictionary with 'analytical', 'numerical', and 'difference' sub-dictionaries
        """
        difference = {}
        for key in analytical:
            if key in numerical:
                diff = analytical[key] - numerical[key]
                rel_diff = diff / analytical[key] if abs(analytical[key]) > 1e-10 else 0
                difference[key] = {"absolute": diff, "relative": rel_diff}

        return {
            "analytical": analytical,
            "numerical": numerical,
            "difference": difference,
        }

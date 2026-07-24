"""
Greeks calculation for equity derivatives.
"""

import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from scipy import stats

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
    BucketedGreeksRequest,
    BucketedGreeksResult,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.util.enum import CommonGreek, EquityGreek
from quantark.util.enum.engine_enums import EngineType, GreeksCalculationMode
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import is_zero


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
        analytical_supported = {
            "price",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
        }

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
        if greeks is None:
            return None
        if len(greeks) == 0:
            return set()
        aliases = {
            "deltaq": "delta_q",
            "deltadq": "delta_q",
            "d_delta_d_q": "delta_q",
            "d_delta_dq": "delta_q",
            "rhoq": "dividend_rho",
            "div_rho": "dividend_rho",
            "dividendrho": "dividend_rho",
        }
        allowed = {
            "price",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "dividend_rho",
            "vanna",
            "volga",
            "delta_q",
            "charm",
            "color",
            "convexity_theta",
            "r_theta",
            "q_theta",
        }
        normalized: set[str] = set()
        for greek in greeks:
            if isinstance(greek, (CommonGreek, EquityGreek)):
                name = greek.value
            elif isinstance(greek, str):
                name = greek.strip().lower()
            else:
                raise ValidationError(
                    f"Unsupported greek identifier type: {type(greek).__name__}"
                )
            name = aliases.get(name, name)
            if name not in allowed:
                raise ValidationError(f"Unknown greek name: {name}")
            normalized.add(name)
        return normalized

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
                keyrate_points = self._calculate_bucketed_rate_keyrate_points(
                    product, pricing_env, engine, request, mode
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
                    self._calculate_bucketed_futures_delta_points(
                        product, pricing_env, engine, request, mode
                    )
                )
            elif coordinate == BucketedGreekCoordinate.CARRY_RHOQ:
                points.extend(
                    self._calculate_bucketed_carry_rhoq_points(
                        product, pricing_env, engine, request, mode
                    )
                )
            elif coordinate == BucketedGreekCoordinate.VOL_TENOR_VEGA:
                points.extend(
                    self._calculate_bucketed_vol_tenor_vega_points(
                        product, pricing_env, engine, request, mode
                    )
                )
            elif coordinate in (
                BucketedGreekCoordinate.MARKET_IV_VEGA,
                BucketedGreekCoordinate.MODEL_ARTIFACT,
            ):
                points.extend(
                    self._calculate_bucketed_vol_model_points(
                        product, pricing_env, engine, request, coordinate, mode
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

    def _calculate_bucketed_rate_keyrate_points(
        self, product, pricing_env, engine, request, mode
    ) -> List[BucketedGreekPoint]:
        """Per-CALIBRATED-pillar zero-rate bumps + a parallel reconciliation
        point (spec WP3.3). Reported per +1bp; central differences."""
        from quantark.param.node_roles import NodeRole, resolve_node_roles
        from quantark.param.rrf import ParallelShiftRateCurve
        from quantark.param.rrf.key_rate import key_rate_bumped_zero_curve

        if mode != BucketedGreekDifferenceMode.CENTRAL:
            raise ValidationError(
                f"RATE_KEYRATE supports central mode only, got {mode.value}"
            )
        curve = pricing_env.rate_curve
        tenors = list(getattr(curve, "tenors", []) or [])
        if not tenors:
            raise ValidationError(
                "RATE_KEYRATE requires an interpolated rate curve with pillars"
            )
        info = resolve_node_roles(
            tenors,
            getattr(curve, "node_roles", None),
            getattr(curve, "last_observable_tenor", None),
        )
        bump = request.rate_bump if request.rate_bump is not None else 1e-4
        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        base_price = bump_engine.price(product, pricing_env)

        def _rate_bumped_env(bumped_curve):
            # desk convention (spec WP3.3): carry B(T) is the invariant, so
            # a discount bump re-derives q pointwise and F(0,T) is unchanged
            # -> the bump is pure discounting. This applies ALSO when
            # div_yield is None (PricingEnvironment treats None as zero
            # yield): wrap an explicit zero-yield base, otherwise the
            # forward would move and the F-unchanged metadata would be false.
            from quantark.param.div import ContinuousDividendYield
            from quantark.param.div.dividend_yield import (
                CarryInvariantDividendYield,
            )

            env = deepcopy(pricing_env)
            env.rate_curve = bumped_curve
            base_div = (
                pricing_env.div_yield
                if pricing_env.div_yield is not None
                else ContinuousDividendYield(0.0)
            )
            env.div_yield = CarryInvariantDividendYield(
                base=base_div,
                base_rate_curve=curve,
                bumped_rate_curve=bumped_curve,
            )
            return env

        points: List[BucketedGreekPoint] = []
        for tenor, role in zip(tenors, info.roles):
            if role is not NodeRole.CALIBRATED:
                continue
            up_env = _rate_bumped_env(
                key_rate_bumped_zero_curve(curve, tenor, +bump)
            )
            down_env = _rate_bumped_env(
                key_rate_bumped_zero_curve(curve, tenor, -bump)
            )
            up_price = bump_engine.price(product, up_env)
            down_price = bump_engine.price(product, down_env)
            derivative = (up_price - down_price) / (2.0 * bump)
            points.append(
                BucketedGreekPoint(
                    coordinate=BucketedGreekCoordinate.RATE_KEYRATE,
                    name=f"rate_keyrate.{tenor:g}y",
                    reported=derivative * 1e-4,
                    derivative=derivative,
                    pnl=(up_price - down_price) / 2.0,
                    bump_size=float(bump),
                    convention_scale=1e-4,
                    base_price=float(base_price),
                    up_price=float(up_price),
                    down_price=float(down_price),
                    difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                    maturity=float(tenor),
                    extrapolated_tail=bool(
                        tenor > info.last_observable_tenor
                    ),
                    metadata={
                        "unit": "per_1bp",
                        "roles_inferred": info.roles_inferred,
                        "rebuild_rule": "zero-rate pillar bump; carry-invariant "
                        "q re-derivation (F unchanged) -> pure discounting",
                    },
                )
            )
        par_up = _rate_bumped_env(ParallelShiftRateCurve(curve, +bump))
        par_down = _rate_bumped_env(ParallelShiftRateCurve(curve, -bump))
        par_derivative = (
            bump_engine.price(product, par_up)
            - bump_engine.price(product, par_down)
        ) / (2.0 * bump)
        parallel_per_1bp = par_derivative * 1e-4
        sum_of_buckets = float(
            sum(pt.reported for pt in points if pt.reported is not None)
        )
        points.append(
            BucketedGreekPoint(
                coordinate=BucketedGreekCoordinate.RATE_KEYRATE,
                name="rate_keyrate.parallel",
                reported=parallel_per_1bp,
                derivative=par_derivative,
                pnl=parallel_per_1bp,
                bump_size=float(bump),
                convention_scale=1e-4,
                base_price=float(base_price),
                difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                metadata={
                    "unit": "per_1bp",
                    "sum_of_buckets": sum_of_buckets,
                    "reconciles": bool(
                        abs(sum_of_buckets - parallel_per_1bp)
                        <= 0.05 * max(abs(parallel_per_1bp), 1e-12)
                    ),
                    "roles_inferred": info.roles_inferred,
                    "rebuild_rule": "ParallelShiftRateCurve; carry-invariant "
                    "q re-derivation (F unchanged)",
                },
            )
        )
        return points

    def _calculate_bucketed_futures_delta_points(
        self, product, pricing_env, engine, request, mode
    ) -> List[BucketedGreekPoint]:
        if request.futures_curve is None:
            raise ValidationError("FUTURES_DELTA requires request.futures_curve")
        if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
            return self._calculate_futures_delta_one_sided_points(
                product, pricing_env, engine, request
            )
        if mode == BucketedGreekDifferenceMode.CENTRAL:
            return self._calculate_futures_delta_central_points(
                product, pricing_env, engine, request
            )
        raise ValidationError(
            f"unsupported FUTURES_DELTA difference mode: {mode.value}"
        )

    def _calculate_futures_delta_one_sided_points(
        self, product, pricing_env, engine, request
    ) -> List[BucketedGreekPoint]:
        rows = self.calculate_futures_delta_buckets(
            product,
            pricing_env,
            engine,
            request.futures_curve,
            mode=request.futures_carry_mode,
            price_bump=request.futures_price_bump,
        )
        points: List[BucketedGreekPoint] = []
        for row in rows:
            derivative = float(row["delta_bucket"])
            bump_size = float(row["price_bump"])
            points.append(
                BucketedGreekPoint(
                    coordinate=BucketedGreekCoordinate.FUTURES_DELTA,
                    name=f"futures_delta.{row['contract']}",
                    reported=derivative,
                    derivative=derivative,
                    pnl=derivative * bump_size,
                    bump_size=bump_size,
                    convention_scale=1.0,
                    base_price=0.0,
                    difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                    contract=str(row["contract"]),
                    maturity=float(row["maturity"]),
                    future_price=float(row["future_price"]),
                    delta_per_hand=float(row["delta_per_hand"]),
                    hedge_hands=float(row["hedge_hands"]),
                    extrapolated_tail=bool(row["extrapolated_tail"]),
                    metadata={"source": "calculate_futures_delta_buckets"},
                )
            )
        return points

    def _calculate_futures_delta_central_points(
        self, product, pricing_env, engine, request
    ) -> List[BucketedGreekPoint]:
        from quantark.asset.equity.market import hedge_hands as _hedge_hands
        from quantark.util.enum import FuturesCarryRiskMode

        futures_curve = request.futures_curve
        resolved_mode = (
            request.futures_carry_mode
            if request.futures_carry_mode is not None
            else futures_curve.mode
        )
        if resolved_mode is not FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
            raise ValidationError(
                "calculate_futures_delta_buckets requires IMPLIED_FUTURES_CARRY "
                f"mode, got {resolved_mode}"
            )

        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        base_env = deepcopy(pricing_env)
        base_env.div_yield = futures_curve.to_dividend_yield_curve(
            pricing_env.rate_curve
        )
        base_price = bump_engine.price(product, base_env)
        maturity = product.get_maturity(pricing_env)
        last_index = len(futures_curve.quotes) - 1
        points: List[BucketedGreekPoint] = []

        for i, quote in enumerate(futures_curve.quotes):
            up_curve = futures_curve.bump_contract(
                quote.contract, request.futures_price_bump
            )
            down_curve = futures_curve.bump_contract(
                quote.contract, -request.futures_price_bump
            )
            up_env = deepcopy(pricing_env)
            up_env.div_yield = up_curve.to_dividend_yield_curve(
                pricing_env.rate_curve
            )
            down_env = deepcopy(pricing_env)
            down_env.div_yield = down_curve.to_dividend_yield_curve(
                pricing_env.rate_curve
            )
            up_price = bump_engine.price(product, up_env)
            down_price = bump_engine.price(product, down_env)
            derivative = (up_price - down_price) / (
                2.0 * request.futures_price_bump
            )
            per_hand = futures_curve.delta_per_hand(quote.contract)
            extrapolated_tail = (
                i == last_index and maturity > quote.maturity
            ) or (i == 0 and maturity < quote.maturity)
            points.append(
                BucketedGreekPoint(
                    coordinate=BucketedGreekCoordinate.FUTURES_DELTA,
                    name=f"futures_delta.{quote.contract}",
                    reported=derivative,
                    derivative=derivative,
                    pnl=(up_price - down_price) / 2.0,
                    bump_size=float(request.futures_price_bump),
                    convention_scale=1.0,
                    base_price=float(base_price),
                    up_price=float(up_price),
                    down_price=float(down_price),
                    difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                    contract=quote.contract,
                    maturity=float(quote.maturity),
                    future_price=float(quote.price),
                    delta_per_hand=float(per_hand),
                    hedge_hands=float(_hedge_hands(derivative, per_hand)),
                    extrapolated_tail=bool(extrapolated_tail),
                    metadata={"source": "central_futures_mark_bump"},
                )
            )
        return points

    def _calculate_bucketed_carry_rhoq_points(
        self, product, pricing_env, engine, request, mode
    ) -> List[BucketedGreekPoint]:
        div_bump = request.carry_bump or self._bump_config.div_bump
        if request.futures_curve is not None:
            if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
                return self._calculate_futures_rhoq_one_sided_points(
                    product, pricing_env, engine, request, div_bump
                )
            if mode == BucketedGreekDifferenceMode.CENTRAL:
                return self._calculate_futures_rhoq_central_points(
                    product, pricing_env, engine, request, div_bump
                )
            raise ValidationError(
                f"unsupported CARRY_RHOQ difference mode: {mode.value}"
            )
        return self._calculate_generic_carry_rhoq_points(
            product, pricing_env, engine, request, mode, div_bump
        )

    def _calculate_futures_rhoq_one_sided_points(
        self, product, pricing_env, engine, request, div_bump
    ) -> List[BucketedGreekPoint]:
        rows = self.calculate_futures_rhoq_buckets(
            product,
            pricing_env,
            engine,
            request.futures_curve,
            mode=request.futures_carry_mode,
            div_bump=div_bump,
        )
        points: List[BucketedGreekPoint] = []
        for row in rows:
            reported = float(row["rhoq_bucket"])
            derivative = reported / 0.01
            points.append(
                BucketedGreekPoint(
                    coordinate=BucketedGreekCoordinate.CARRY_RHOQ,
                    name=f"carry_rhoq.{row['contract']}",
                    reported=reported,
                    derivative=derivative,
                    pnl=derivative * float(row["div_bump"]),
                    bump_size=float(row["div_bump"]),
                    convention_scale=0.01,
                    base_price=0.0,
                    difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                    contract=str(row["contract"]),
                    maturity=float(row["maturity"]),
                    future_price=float(row["future_price"]),
                    extrapolated_tail=bool(row["extrapolated_tail"]),
                    metadata={"source": "calculate_futures_rhoq_buckets"},
                )
            )
        return points

    def _calculate_futures_rhoq_central_points(
        self, product, pricing_env, engine, request, div_bump
    ) -> List[BucketedGreekPoint]:
        from quantark.asset.equity.market import bump_term_yield_node
        from quantark.asset.equity.report.term_structure import BucketedDividendYield
        from quantark.param.div import ContinuousDividendYield
        from quantark.util.enum import FuturesCarryRiskMode

        futures_curve = request.futures_curve
        resolved_mode = (
            request.futures_carry_mode
            if request.futures_carry_mode is not None
            else futures_curve.mode
        )
        if resolved_mode is FuturesCarryRiskMode.MARKET_PRICE:
            raise ValidationError(
                "calculate_futures_rhoq_buckets does not support MARKET_PRICE "
                "mode (it supplies no carry curve for repricing the option)"
            )

        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        maturity = product.get_maturity(pricing_env)
        last_index = len(futures_curve.quotes) - 1

        def _tail_flag(i, quote):
            return (i == last_index and maturity > quote.maturity) or (
                i == 0 and maturity < quote.maturity
            )

        points: List[BucketedGreekPoint] = []
        if resolved_mode is FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
            base_div = futures_curve.to_dividend_yield_curve(pricing_env.rate_curve)
            base_env = deepcopy(pricing_env)
            base_env.div_yield = base_div
            base_price = bump_engine.price(product, base_env)
            for i, quote in enumerate(futures_curve.quotes):
                up_env = deepcopy(pricing_env)
                up_env.div_yield = bump_term_yield_node(base_div, i, div_bump)
                down_env = deepcopy(pricing_env)
                down_env.div_yield = bump_term_yield_node(base_div, i, -div_bump)
                up_price = bump_engine.price(product, up_env)
                down_price = bump_engine.price(product, down_env)
                derivative = (up_price - down_price) / (2.0 * div_bump)
                points.append(
                    self._carry_rhoq_point(
                        name=f"carry_rhoq.{quote.contract}",
                        derivative=derivative,
                        bump_size=div_bump,
                        base_price=base_price,
                        up_price=up_price,
                        down_price=down_price,
                        difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                        contract=quote.contract,
                        maturity=quote.maturity,
                        future_price=quote.price,
                        extrapolated_tail=_tail_flag(i, quote),
                        source="central_futures_implied_carry_node_bump",
                    )
                )
            return points

        base_price = bump_engine.price(product, pricing_env)
        base_div = pricing_env.div_yield or ContinuousDividendYield(0.0)
        edges = [0.0] + [q.maturity for q in futures_curve.quotes]
        if maturity > edges[-1]:
            edges[-1] = maturity
        for i, quote in enumerate(futures_curve.quotes):
            up_env = deepcopy(pricing_env)
            up_env.div_yield = BucketedDividendYield(
                base=base_div,
                bucket_start=edges[i],
                bucket_end=edges[i + 1],
                bump=div_bump,
            )
            down_env = deepcopy(pricing_env)
            down_env.div_yield = BucketedDividendYield(
                base=base_div,
                bucket_start=edges[i],
                bucket_end=edges[i + 1],
                bump=-div_bump,
            )
            up_price = bump_engine.price(product, up_env)
            down_price = bump_engine.price(product, down_env)
            derivative = (up_price - down_price) / (2.0 * div_bump)
            points.append(
                self._carry_rhoq_point(
                    name=f"carry_rhoq.{quote.contract}",
                    derivative=derivative,
                    bump_size=div_bump,
                    base_price=base_price,
                    up_price=up_price,
                    down_price=down_price,
                    difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                    contract=quote.contract,
                    maturity=quote.maturity,
                    future_price=quote.price,
                    extrapolated_tail=_tail_flag(i, quote),
                    source="central_theoretical_carry_bucket_bump",
                )
            )
        return points

    def _calculate_generic_carry_rhoq_points(
        self, product, pricing_env, engine, request, mode, div_bump
    ) -> List[BucketedGreekPoint]:
        from quantark.asset.equity.report.term_structure import (
            BucketedDividendYield,
            default_tenor_buckets,
        )
        from quantark.param.div import ContinuousDividendYield

        if mode not in (
            BucketedGreekDifferenceMode.ONE_SIDED_UP,
            BucketedGreekDifferenceMode.CENTRAL,
        ):
            raise ValidationError(
                f"unsupported CARRY_RHOQ difference mode: {mode.value}"
            )
        # spec WP3.3: with no explicit buckets and a term-structure carry
        # curve, buckets align to the curve's calibrated NODES (node bumps
        # rebuild the curve; no window-step discontinuities)
        if request.tenor_buckets is None and hasattr(
            pricing_env.div_yield, "times"
        ):
            return self._calculate_node_aligned_carry_rhoq_points(
                product, pricing_env, engine, mode, div_bump
            )
        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        maturity = product.get_maturity(pricing_env)
        buckets = tuple(request.tenor_buckets or default_tenor_buckets(maturity))
        base_div = pricing_env.div_yield or ContinuousDividendYield(0.0)
        base_price = bump_engine.price(product, pricing_env)
        points: List[BucketedGreekPoint] = []
        for bucket in buckets:
            up_env = deepcopy(pricing_env)
            up_env.div_yield = BucketedDividendYield(
                base=base_div,
                bucket_start=bucket.start,
                bucket_end=bucket.end,
                bump=div_bump,
            )
            up_price = bump_engine.price(product, up_env)
            if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
                derivative = (up_price - base_price) / div_bump
                points.append(
                    self._carry_rhoq_point(
                        name=f"carry_rhoq.{bucket.label}",
                        derivative=derivative,
                        bump_size=div_bump,
                        base_price=base_price,
                        up_price=up_price,
                        down_price=None,
                        difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                        bucket=self._bucket_label(bucket),
                        source="generic_tenor_carry_bucket_bump",
                    )
                )
                continue

            down_env = deepcopy(pricing_env)
            down_env.div_yield = BucketedDividendYield(
                base=base_div,
                bucket_start=bucket.start,
                bucket_end=bucket.end,
                bump=-div_bump,
            )
            down_price = bump_engine.price(product, down_env)
            derivative = (up_price - down_price) / (2.0 * div_bump)
            points.append(
                self._carry_rhoq_point(
                    name=f"carry_rhoq.{bucket.label}",
                    derivative=derivative,
                    bump_size=div_bump,
                    base_price=base_price,
                    up_price=up_price,
                    down_price=down_price,
                    difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                    bucket=self._bucket_label(bucket),
                    source="generic_tenor_carry_bucket_bump",
                )
            )
        return points

    def _calculate_node_aligned_carry_rhoq_points(
        self, product, pricing_env, engine, mode, div_bump
    ) -> List[BucketedGreekPoint]:
        """Central node bumps on a TermStructureDividendYield, one point per
        CALIBRATED node (spec WP3.3). The term curve's own interpolation
        rebuilds carry between nodes."""
        from quantark.asset.equity.market.index_futures_curve import (
            bump_term_yield_node,
        )
        from quantark.param.node_roles import NodeRole, resolve_node_roles

        term_div = pricing_env.div_yield
        info = resolve_node_roles(
            list(term_div.times),
            getattr(term_div, "node_roles", None),
            getattr(term_div, "last_observable_tenor", None),
        )
        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        base_price = bump_engine.price(product, pricing_env)
        points: List[BucketedGreekPoint] = []
        for i, (tenor, role) in enumerate(zip(term_div.times, info.roles)):
            if role is not NodeRole.CALIBRATED:
                continue
            up_env = deepcopy(pricing_env)
            up_env.div_yield = bump_term_yield_node(term_div, i, +div_bump)
            down_env = deepcopy(pricing_env)
            down_env.div_yield = bump_term_yield_node(term_div, i, -div_bump)
            up_price = bump_engine.price(product, up_env)
            down_price = bump_engine.price(product, down_env)
            derivative = (up_price - down_price) / (2.0 * div_bump)
            point = self._carry_rhoq_point(
                name=f"carry_rhoq.node_{tenor:g}y",
                derivative=derivative,
                bump_size=div_bump,
                base_price=base_price,
                up_price=up_price,
                down_price=down_price,
                difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                maturity=float(tenor),
                extrapolated_tail=bool(tenor > info.last_observable_tenor),
                source="node_aligned_term_yield_bump",
            )
            points.append(point)
        return points

    @staticmethod
    def _bucket_label(bucket) -> str:
        return f"{bucket.label} ({bucket.start:.3g}-{bucket.end:.3g}y)"

    def _carry_rhoq_point(
        self,
        *,
        name,
        derivative,
        bump_size,
        base_price,
        up_price,
        down_price,
        difference_mode,
        source,
        bucket=None,
        contract=None,
        maturity=None,
        future_price=None,
        extrapolated_tail=None,
    ) -> BucketedGreekPoint:
        reported = derivative * 0.01
        if difference_mode == BucketedGreekDifferenceMode.CENTRAL.value:
            pnl = (up_price - down_price) / 2.0
        else:
            pnl = up_price - base_price
        return BucketedGreekPoint(
            coordinate=BucketedGreekCoordinate.CARRY_RHOQ,
            name=name,
            reported=reported,
            derivative=derivative,
            pnl=pnl,
            bump_size=float(bump_size),
            convention_scale=0.01,
            base_price=float(base_price),
            up_price=None if up_price is None else float(up_price),
            down_price=None if down_price is None else float(down_price),
            difference_mode=difference_mode,
            bucket=bucket,
            contract=contract,
            maturity=None if maturity is None else float(maturity),
            future_price=None if future_price is None else float(future_price),
            extrapolated_tail=extrapolated_tail,
            metadata={"source": source},
        )

    def _calculate_bucketed_vol_tenor_vega_points(
        self, product, pricing_env, engine, request, mode
    ) -> List[BucketedGreekPoint]:
        from quantark.asset.equity.report.term_structure import (
            BucketedVolSurface,
            default_tenor_buckets,
        )
        from quantark.volmodels.heston import HestonParams

        if isinstance(getattr(engine, "model_params", None), HestonParams):
            engine_name = type(engine).__name__
            if "SLV" in engine_name:
                raise ValidationError(
                    "VOL_TENOR_VEGA bumps pricing_env.vol_surface directly; "
                    "SLV market vega requires local-vol/leverage recalibration. "
                    "Request MARKET_IV_VEGA instead."
                )
            raise ValidationError(
                "VOL_TENOR_VEGA bumps pricing_env.vol_surface directly; "
                "Heston uses model params calibrated from market IV. "
                "Request MARKET_IV_VEGA instead."
            )
        if pricing_env.vol_surface is None:
            raise ValidationError("vol_surface is required for bucketed vega.")
        if mode not in (
            BucketedGreekDifferenceMode.ONE_SIDED_UP,
            BucketedGreekDifferenceMode.CENTRAL,
        ):
            raise ValidationError(
                f"unsupported VOL_TENOR_VEGA difference mode: {mode.value}"
            )

        # spec WP3.3: with no explicit buckets and a term-structure ATM
        # curve, bump NODE vols and rebuild total variance (no window-step
        # calendar-arb discontinuities on dense grids)
        if request.tenor_buckets is None and hasattr(
            pricing_env.vol_surface, "times"
        ):
            return self._calculate_node_aligned_vol_vega_points(
                product, pricing_env, engine, request
            )

        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        maturity = product.get_maturity(pricing_env)
        buckets = tuple(request.tenor_buckets or default_tenor_buckets(maturity))
        base_price = bump_engine.price(product, pricing_env)
        points: List[BucketedGreekPoint] = []
        for bucket in buckets:
            up_env = deepcopy(pricing_env)
            up_env.vol_surface = BucketedVolSurface(
                base=pricing_env.vol_surface,
                bucket_start=bucket.start,
                bucket_end=bucket.end,
                bump=request.vol_bump,
            )
            up_price = bump_engine.price(product, up_env)
            if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
                derivative = (up_price - base_price) / request.vol_bump
                points.append(
                    self._vol_tenor_vega_point(
                        bucket=bucket,
                        derivative=derivative,
                        bump_size=request.vol_bump,
                        base_price=base_price,
                        up_price=up_price,
                        down_price=None,
                        difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                    )
                )
                continue

            down_env = deepcopy(pricing_env)
            down_env.vol_surface = BucketedVolSurface(
                base=pricing_env.vol_surface,
                bucket_start=bucket.start,
                bucket_end=bucket.end,
                bump=-request.vol_bump,
            )
            down_price = bump_engine.price(product, down_env)
            derivative = (up_price - down_price) / (2.0 * request.vol_bump)
            points.append(
                self._vol_tenor_vega_point(
                    bucket=bucket,
                    derivative=derivative,
                    bump_size=request.vol_bump,
                    base_price=base_price,
                    up_price=up_price,
                    down_price=down_price,
                    difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                )
            )
        return points

    def _calculate_node_aligned_vol_vega_points(
        self, product, pricing_env, engine, request
    ) -> List[BucketedGreekPoint]:
        """Central node-vol bumps on a term ATM curve, one point per
        CALIBRATED node, reported per +1 vol pt (spec WP3.3). If a down bump
        produces negative forward variance (calendar arbitrage on the pricing
        grid), fall back to one-sided-up and record the adjustment."""
        from quantark.param.node_roles import NodeRole, resolve_node_roles

        surface = pricing_env.vol_surface
        info = resolve_node_roles(
            list(surface.times),
            getattr(surface, "node_roles", None),
            getattr(surface, "last_observable_tenor", None),
        )
        vol_bump = float(request.vol_bump)
        bump_engine = self._resolve_bump_engine(product, pricing_env, engine)
        base_price = bump_engine.price(product, pricing_env)
        points: List[BucketedGreekPoint] = []
        for i, (tenor, role) in enumerate(zip(surface.times, info.roles)):
            if role is not NodeRole.CALIBRATED:
                continue
            def _try_price(node_bump: float):
                env = deepcopy(pricing_env)
                env.vol_surface = self._bump_term_vol_node(
                    surface, i, node_bump
                )
                try:
                    return bump_engine.price(product, env), None
                except NumericalError as exc:
                    # negative forward variance: this bump direction is not
                    # representable on the pricing grid (calendar arbitrage)
                    return None, str(exc)

            up_price, up_error = _try_price(+vol_bump)
            down_price, down_error = _try_price(-vol_bump)
            adjusted = up_price is None or down_price is None
            status, error = "ok", None
            if up_price is not None and down_price is not None:
                derivative = (up_price - down_price) / (2.0 * vol_bump)
                pnl = (up_price - down_price) / 2.0
                diff_mode = BucketedGreekDifferenceMode.CENTRAL.value
            elif up_price is not None:
                # one-sided-up is explicitly allowed when recorded (§6.3)
                derivative = (up_price - base_price) / vol_bump
                pnl = up_price - base_price
                diff_mode = BucketedGreekDifferenceMode.ONE_SIDED_UP.value
            elif down_price is not None:
                derivative = (base_price - down_price) / vol_bump
                pnl = base_price - down_price
                diff_mode = "one_sided_down"
            else:
                derivative, pnl = None, None
                diff_mode = BucketedGreekDifferenceMode.CENTRAL.value
                status = "failed"
                error = f"up: {up_error}; down: {down_error}"
            points.append(
                BucketedGreekPoint(
                    coordinate=BucketedGreekCoordinate.VOL_TENOR_VEGA,
                    name=f"vol_tenor_vega.node_{tenor:g}y",
                    reported=(
                        None if derivative is None else derivative * 0.01
                    ),
                    derivative=derivative,
                    pnl=pnl,
                    bump_size=vol_bump,
                    convention_scale=0.01,
                    base_price=float(base_price),
                    up_price=None if up_price is None else float(up_price),
                    down_price=(
                        None if down_price is None else float(down_price)
                    ),
                    difference_mode=diff_mode,
                    status=status,
                    error=error,
                    maturity=float(tenor),
                    extrapolated_tail=bool(tenor > info.last_observable_tenor),
                    metadata={
                        "unit": "per_1volpt",
                        "source": "node_aligned_term_vol_bump",
                        "rebuild_rule": "node vol bump, total variance "
                        "rebuilt by term interpolation",
                        "adjusted_to_one_sided": adjusted,
                        "roles_inferred": info.roles_inferred,
                    },
                )
            )
        return points

    @staticmethod
    def _bump_term_vol_node(surface, node_index: int, bump: float):
        """Copy of a term ATM vol curve with one node's vol bumped."""
        from quantark.param.vol.vol_surface import TermStructureVolSurface

        vols = [float(v) for v in surface.vols]
        vols[node_index] += float(bump)
        return TermStructureVolSurface(
            times=list(surface.times),
            vols=vols,
            node_roles=getattr(surface, "node_roles", None),
            last_observable_tenor=getattr(
                surface, "last_observable_tenor", None
            ),
        )

    def _vol_tenor_vega_point(
        self,
        *,
        bucket,
        derivative,
        bump_size,
        base_price,
        up_price,
        down_price,
        difference_mode,
    ) -> BucketedGreekPoint:
        if difference_mode == BucketedGreekDifferenceMode.CENTRAL.value:
            pnl = (up_price - down_price) / 2.0
        else:
            pnl = up_price - base_price
        return BucketedGreekPoint(
            coordinate=BucketedGreekCoordinate.VOL_TENOR_VEGA,
            name=f"vol_tenor_vega.{bucket.label}",
            reported=derivative * 0.01,
            derivative=derivative,
            pnl=pnl,
            bump_size=float(bump_size),
            convention_scale=0.01,
            base_price=float(base_price),
            up_price=float(up_price),
            down_price=None if down_price is None else float(down_price),
            difference_mode=difference_mode,
            bucket=self._bucket_label(bucket),
            metadata={"source": "generic_tenor_vol_bucket_bump"},
        )

    def _calculate_bucketed_vol_model_points(
        self, product, pricing_env, engine, request, coordinate, mode
    ) -> List[BucketedGreekPoint]:
        from quantark.asset.equity.riskmeasures.vol_model_risk import (
            VolModelRiskCalculator,
        )

        if mode != BucketedGreekDifferenceMode.CENTRAL:
            raise ValidationError(
                f"{coordinate.value} does not support {mode.value} in v1; "
                "request central mode or extend VolModelRiskCalculator"
            )
        risk_calc = VolModelRiskCalculator(
            heston_calibration_spec=request.heston_calibration_spec,
            slv_calibration_spec=request.slv_calibration_spec,
        )
        if coordinate == BucketedGreekCoordinate.MARKET_IV_VEGA:
            vol_result = risk_calc.calculate_market_vega(
                product, pricing_env, engine, request.market_vega_request
            )
            return self._vol_risk_result_to_bucketed_points(
                coordinate=BucketedGreekCoordinate.MARKET_IV_VEGA,
                result=vol_result,
                convention_scale=0.01,
            )
        if request.model_risk_request is None:
            raise ValidationError("MODEL_ARTIFACT requires request.model_risk_request")
        model_result = risk_calc.calculate_model_risk(
            product, pricing_env, engine, request.model_risk_request
        )
        return self._vol_risk_result_to_bucketed_points(
            coordinate=BucketedGreekCoordinate.MODEL_ARTIFACT,
            result=model_result,
            convention_scale=1.0,
        )

    def _vol_risk_result_to_bucketed_points(
        self, *, coordinate, result, convention_scale
    ) -> List[BucketedGreekPoint]:
        model = result.metadata.get("model")
        points: List[BucketedGreekPoint] = []
        for point in result.points:
            reported = None
            if point.derivative is not None:
                reported = point.derivative * convention_scale
            if point.status == "failed":
                points.append(
                    BucketedGreekPoint.failed(
                        coordinate=coordinate,
                        name=point.name,
                        bump_size=point.bump_size,
                        base_price=result.base_price,
                        error=point.error or "vol-model scenario failed",
                        convention_scale=convention_scale,
                        difference_mode=point.difference_mode,
                        up_price=point.up_price,
                        down_price=point.down_price,
                        model=model,
                        metadata=result.metadata,
                    )
                )
                continue
            points.append(
                BucketedGreekPoint(
                    coordinate=coordinate,
                    name=point.name,
                    reported=reported,
                    derivative=point.derivative,
                    pnl=point.pnl,
                    bump_size=point.bump_size,
                    convention_scale=convention_scale,
                    base_price=result.base_price,
                    up_price=point.up_price,
                    down_price=point.down_price,
                    difference_mode=point.difference_mode,
                    status=point.status,
                    error=point.error,
                    model=model,
                    metadata=result.metadata,
                )
            )
        return points

    def _ensure_base_price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float],
    ) -> float:
        """Return base price, computing it if needed."""
        return (
            base_price if base_price is not None else engine.price(product, pricing_env)
        )

    def _resolve_bump_engine(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
    ) -> BaseEngine:
        """Return the engine context used for numerical bump repricing."""
        create_context = getattr(engine, "create_bump_context", None)
        if not callable(create_context):
            return engine
        bump_engine = create_context(product, pricing_env)
        return bump_engine if bump_engine is not None else engine

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
        if mode == "central":
            if price_down is None:
                raise ValidationError("central mode requires price_down")
            return (price_up - price_down) / (2.0 * scale * bump)
        if mode == "second_order":
            if price_down is None:
                raise ValidationError("second_order mode requires price_down")
            return (price_up - 2.0 * base_price + price_down) / (
                scale * bump
            ) ** 2
        if mode == "one_sided":
            return price_up - base_price
        raise ValidationError(f"Unknown sensitivity mode: {mode}")

    def _get_delta_gamma(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float],
    ) -> Tuple[float, float, float]:
        """Get base price, delta, and gamma via engine or bump method."""
        if self._should_use_engine_greeks(engine):
            engine_greeks = engine.calculate_greeks(product, pricing_env)
            if base_price is None:
                base_price = engine_greeks["price"]
            return base_price, engine_greeks["delta"], engine_greeks["gamma"]

        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        spot_prices = self._spot_bumped_prices(
            product, pricing_env, engine, self._bump_config.spot_bump, base_price=base_price
        )[1:]

        delta = self.calculate_numerical_delta(
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=self._bump_config.spot_bump,
        )
        gamma = self.calculate_numerical_gamma(
            product,
            pricing_env,
            engine,
            base_price=base_price,
            spot_prices=spot_prices,
            bump=self._bump_config.spot_bump,
        )

        return base_price, delta, gamma

    def calculate_analytical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        price: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using analytical Black-Scholes formulas.

        Only works for European vanilla options under Black-Scholes model.

        Args:
            product: European vanilla option
            pricing_env: Pricing environment
            price: Pre-calculated price (optional, will calculate if not provided)

        Returns:
            Dictionary of Greeks: delta, gamma, vega, theta, rho

        Raises:
            ValidationError: If product is not a European vanilla option
        """
        if not isinstance(product, EuropeanVanillaOption):
            raise ValidationError(
                f"Analytical Greeks only support EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )

        # Extract parameters
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        # Handle edge case: option at expiry
        if is_zero(T):
            return self._greeks_at_expiry(product, S)

        # Calculate d1 and d2
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Calculate discount factors
        discount_div = math.exp(-q * T)
        discount_rf = math.exp(-r * T)

        # Standard normal PDF and CDF
        n_d1 = stats.norm.pdf(d1)  # phi(d1)
        N_d1 = stats.norm.cdf(d1)  # Phi(d1)
        N_d2 = stats.norm.cdf(d2)  # Phi(d2)

        greeks = {}

        multiplier = product.contract_multiplier

        # Calculate price if not provided (per-unit)
        if price is None:
            if product.is_call():
                price = S * discount_div * N_d1 - K * discount_rf * N_d2
            else:
                price = K * discount_rf * stats.norm.cdf(
                    -d2
                ) - S * discount_div * stats.norm.cdf(-d1)
        else:
            price = price / multiplier
        greeks["price"] = price

        # Delta: ∂V/∂S
        if product.is_call():
            delta = discount_div * N_d1
        else:
            delta = -discount_div * stats.norm.cdf(-d1)
        greeks["delta"] = delta

        # Gamma: ∂²V/∂S²
        gamma = discount_div * n_d1 / (S * sigma * sqrt_T)
        greeks["gamma"] = gamma

        # Vega: ∂V/∂σ (divided by 100 for 1% change)
        vega = S * discount_div * n_d1 * sqrt_T / 100
        greeks["vega"] = vega

        # Theta: ∂V/∂t (per day, divided by 365)
        # Decomposed into three components:
        #   convexity_theta: time decay from gamma/convexity erosion (always negative)
        #   r_theta: time decay from interest rate cost of carry
        #   q_theta: time decay from dividend yield
        term1 = -S * discount_div * n_d1 * sigma / (2 * sqrt_T)
        if product.is_call():
            term2 = -r * K * discount_rf * N_d2
            term3 = q * S * discount_div * N_d1
        else:
            term2 = r * K * discount_rf * stats.norm.cdf(-d2)
            term3 = -q * S * discount_div * stats.norm.cdf(-d1)

        # Store decomposed components (per day)
        convexity_theta = term1 / 365
        r_theta = term2 / 365
        q_theta = term3 / 365
        theta = convexity_theta + r_theta + q_theta

        greeks["theta"] = theta
        greeks["convexity_theta"] = convexity_theta
        greeks["r_theta"] = r_theta
        greeks["q_theta"] = q_theta

        # Rho: ∂V/∂r (divided by 100 for 1% change)
        if product.is_call():
            rho = K * T * discount_rf * N_d2 / 100
        else:
            rho = -K * T * discount_rf * stats.norm.cdf(-d2) / 100
        greeks["rho"] = rho

        # Dividend Rho: ∂V/∂q (divided by 100 for 1% change)
        if product.is_call():
            dividend_rho = -S * T * discount_div * N_d1 / 100
        else:
            dividend_rho = S * T * discount_div * stats.norm.cdf(-d1) / 100
        greeks["dividend_rho"] = dividend_rho

        for key, value in greeks.items():
            greeks[key] = value * multiplier

        return greeks

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
            requested = {
                "price",
                "delta",
                "gamma",
                "vega",
                "theta",
                "rho",
                "dividend_rho",
                "convexity_theta",
                "r_theta",
                "q_theta",
            }

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
        bump = bump if bump is not None else self._bump_config.spot_bump
        base_price, price_up_spot, price_down_spot = self._spot_bumped_prices(
            product,
            pricing_env,
            engine,
            bump,
            base_price=base_price,
            reuse=spot_prices,
        )
        return self._calculate_sensitivity(
            base_price,
            price_up_spot,
            price_down_spot,
            bump=bump,
            scale=pricing_env.spot,
            mode="central",
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
        bump = bump if bump is not None else self._bump_config.spot_bump
        base_price, price_up_spot, price_down_spot = self._spot_bumped_prices(
            product,
            pricing_env,
            engine,
            bump,
            base_price=base_price,
            reuse=spot_prices,
        )
        return self._calculate_sensitivity(
            base_price,
            price_up_spot,
            price_down_spot,
            bump=bump,
            scale=pricing_env.spot,
            mode="second_order",
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
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        vol_bump = vol_bump if vol_bump is not None else self._bump_config.vol_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        current_vol = pricing_env.get_vol(strike, T)
        env_up_vol = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=1.0
        )
        price_up_vol = engine.price(product, env_up_vol)
        return self._calculate_sensitivity(
            base_price, price_up_vol, bump=vol_bump, mode="one_sided"
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
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        vol_bump = vol_bump if vol_bump is not None else self._bump_config.vol_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        current_vol = pricing_env.get_vol(strike, T)

        if current_vol - vol_bump <= 0:
            env_up = self._build_vol_bumped_env(
                pricing_env, product, current_vol, vol_bump, direction=1.0
            )
            vega_base = self.calculate_numerical_vega(
                product, pricing_env, engine, base_price=base_price, vol_bump=vol_bump
            )
            vega_up = self.calculate_numerical_vega(
                product, env_up, engine, base_price=None, vol_bump=vol_bump
            )
            return (vega_up - vega_base) / vol_bump

        env_up = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=1.0
        )
        env_down = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=-1.0
        )
        price_up = engine.price(product, env_up)
        price_down = engine.price(product, env_down)
        return self._calculate_sensitivity(
            base_price, price_up, price_down, bump=vol_bump, mode="second_order"
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
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        vol_bump = vol_bump if vol_bump is not None else self._bump_config.vol_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        current_vol = pricing_env.get_vol(strike, T)

        env_up = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=1.0
        )
        env_down = self._build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=-1.0
        )

        if current_vol - vol_bump <= 0:
            base_delta = self.calculate_numerical_delta(
                product,
                pricing_env,
                engine,
                base_price=base_price,
                bump=self._bump_config.spot_bump,
            )
            delta_up = self.calculate_numerical_delta(
                product,
                env_up,
                engine,
                base_price=base_price,
                bump=self._bump_config.spot_bump,
            )
            return (delta_up - base_delta) / vol_bump

        delta_up = self.calculate_numerical_delta(
            product,
            env_up,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        delta_down = self.calculate_numerical_delta(
            product,
            env_down,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        return (delta_up - delta_down) / (2.0 * vol_bump)

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
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        time_bump_days = (
            time_bump_days
            if time_bump_days is not None
            else self._bump_config.time_bump_days
        )
        time_bump_mode = (
            time_bump_mode
            if time_bump_mode is not None
            else getattr(self._bump_config, "time_bump_mode", "auto")
        )
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        product_theta = deepcopy(product)
        env_theta = deepcopy(pricing_env)
        current_maturity = product.get_maturity(pricing_env)

        bumped_date, time_bump, resolved_mode = self._advance_theta_bump(
            pricing_env, time_bump_days, time_bump_mode
        )

        if time_bump <= 0.0:
            if current_maturity <= 0.0:
                return 0.0
            if resolved_mode == "business_days":
                raise ValidationError(
                    "Business-day theta bump did not advance time: "
                    f"valuation_date={pricing_env.valuation_date}, "
                    f"bumped_date={bumped_date}, time_bump_days={time_bump_days}"
                )
            return 0.0
        if current_maturity <= time_bump:
            return 0.0

        env_theta.valuation_date = bumped_date
        dropped_all_observations = product_theta.time_shift(
            time_bump, bumped_date, env_theta
        )

        if dropped_all_observations:
            return 0.0

        price_theta = engine.price(product_theta, env_theta)
        return price_theta - base_price

    def _advance_theta_bump(
        self,
        pricing_env: PricingEnvironment,
        time_bump_days: int,
        time_bump_mode: str,
    ) -> Tuple[datetime, float, str]:
        """Advance the theta valuation date and return date, year fraction, mode."""
        mode = self._resolve_theta_bump_mode(pricing_env, time_bump_mode)
        if mode == "calendar_days":
            bumped_date = pricing_env.valuation_date + timedelta(days=time_bump_days)
        else:
            calendar = getattr(pricing_env, "calendar", None)
            if calendar is None or not hasattr(calendar, "add_business_days"):
                raise ValidationError(
                    "time_bump_mode='business_days' requires pricing_env.calendar "
                    "with add_business_days()"
                )
            bumped_date = calendar.add_business_days(
                pricing_env.valuation_date, time_bump_days
            )

        time_bump = calculate_year_fraction(
            pricing_env.valuation_date,
            bumped_date,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year,
            calendar=getattr(pricing_env, "calendar", None),
        )
        return bumped_date, time_bump, mode

    @staticmethod
    def _resolve_theta_bump_mode(
        pricing_env: PricingEnvironment, time_bump_mode: str
    ) -> str:
        """Resolve auto theta mode against the pricing environment."""
        mode = time_bump_mode.lower()
        if mode not in {"auto", "calendar_days", "business_days"}:
            raise ValidationError(
                "time_bump_mode must be one of 'auto', 'calendar_days', "
                f"or 'business_days', got {time_bump_mode!r}"
            )
        if mode != "auto":
            return mode

        if (
            pricing_env.day_count_convention == DayCountConvention.BUSINESS_DAYS
            and getattr(pricing_env, "calendar", None) is not None
        ):
            return "business_days"
        return "calendar_days"

    def calculate_numerical_rho(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        rate_bump: Optional[float] = None,
    ) -> float:
        """Numerical rho from a rate bump (per 1% rate change)."""
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        rate_bump = rate_bump if rate_bump is not None else self._bump_config.rate_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        env_up_rate = deepcopy(pricing_env)
        from quantark.param.rrf import FlatRateCurve

        T = product.get_maturity(pricing_env)
        current_rate = pricing_env.get_rate(T)
        env_up_rate.rate_curve = FlatRateCurve(current_rate + rate_bump)
        price_up_rate = engine.price(product, env_up_rate)
        raw = self._calculate_sensitivity(
            base_price, price_up_rate, bump=rate_bump, mode="one_sided"
        )
        return raw * (0.01 / rate_bump)

    def calculate_numerical_dividend_rho(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        div_bump: Optional[float] = None,
    ) -> float:
        """
        Numerical dividend_rho (psi) from dividend yield bump.

        Measures price sensitivity to dividend yield changes:
            dividend_rho = dV/dq

        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine
            base_price: Pre-calculated base price
            div_bump: Absolute dividend yield bump (uses config if None)

        Returns:
            Dividend rho value (price change per 1% div_yield change).
            Negative for call options (higher div = lower call price).
            Positive for put options (higher div = higher put price).
        """
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        div_bump = div_bump if div_bump is not None else self._bump_config.div_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        current_div = pricing_env.get_div_yield(T)
        env_up_div = self._build_div_bumped_env(
            pricing_env, product, current_div, div_bump, direction=1.0
        )
        price_up_div = engine.price(product, env_up_div)
        raw = self._calculate_sensitivity(
            base_price, price_up_div, bump=div_bump, mode="one_sided"
        )
        return raw * (0.01 / div_bump)

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
        engine = self._resolve_bump_engine(product, pricing_env, engine)
        div_bump = div_bump if div_bump is not None else self._bump_config.div_bump
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        T = product.get_maturity(pricing_env)
        current_div = pricing_env.get_div_yield(T)

        if base_delta is None:
            base_delta = self.calculate_numerical_delta(
                product,
                pricing_env,
                engine,
                base_price=base_price,
                bump=self._bump_config.spot_bump,
            )

        env_up = self._build_div_bumped_env(
            pricing_env, product, current_div, div_bump, direction=1.0
        )
        env_down = self._build_div_bumped_env(
            pricing_env, product, current_div, div_bump, direction=-1.0
        )
        delta_up = self.calculate_numerical_delta(
            product,
            env_up,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        delta_down = self.calculate_numerical_delta(
            product,
            env_down,
            engine,
            base_price=base_price,
            bump=self._bump_config.spot_bump,
        )
        return (delta_up - delta_down) / (2.0 * div_bump)

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
        """
        Futures-tenor bucket deltas: delta_bucket_i = dPV / dF_i.

        Bumps one futures mark at a time, rebuilds the implied q(T) curve,
        and reprices (one-sided up bump). The base and bumped legs reuse the
        same engine/params unchanged, so MC engines with a fixed seed price
        with common random numbers. The base PV is always computed internally
        under the implied-carry environment (div_yield rebuilt from
        ``futures_curve``); no ``base_price`` parameter is accepted because a
        caller's PV under ``pricing_env.div_yield`` would silently shift every
        bucket.

        hedge_hands = -delta_bucket / delta_per_hand (fractional, unrounded).
        Rows are flagged ``extrapolated_tail`` when the product maturity lies
        outside the quoted node range (flat-extrapolated carry).
        """
        from quantark.asset.equity.market import hedge_hands as _hedge_hands
        from quantark.util.enum import FuturesCarryRiskMode

        resolved_mode = mode if mode is not None else futures_curve.mode
        if resolved_mode is not FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
            raise ValidationError(
                "calculate_futures_delta_buckets requires IMPLIED_FUTURES_CARRY "
                f"mode, got {resolved_mode}"
            )
        if price_bump <= 0.0:
            raise ValidationError("price_bump must be positive")

        engine = self._resolve_bump_engine(product, pricing_env, engine)
        base_env = deepcopy(pricing_env)
        base_env.div_yield = futures_curve.to_dividend_yield_curve(
            pricing_env.rate_curve
        )
        base_price = engine.price(product, base_env)

        maturity = product.get_maturity(pricing_env)
        last_index = len(futures_curve.quotes) - 1
        rows: List[Dict[str, object]] = []
        for i, quote in enumerate(futures_curve.quotes):
            bumped_curve = futures_curve.bump_contract(quote.contract, price_bump)
            bumped_env = deepcopy(pricing_env)
            bumped_env.div_yield = bumped_curve.to_dividend_yield_curve(
                pricing_env.rate_curve
            )
            bumped_price = engine.price(product, bumped_env)
            delta_bucket = (bumped_price - base_price) / price_bump
            per_hand = futures_curve.delta_per_hand(quote.contract)
            extrapolated_tail = (
                i == last_index and maturity > quote.maturity
            ) or (i == 0 and maturity < quote.maturity)
            rows.append(
                {
                    "contract": quote.contract,
                    "maturity": quote.maturity,
                    "future_price": quote.price,
                    "price_bump": price_bump,
                    "delta_bucket": delta_bucket,
                    "delta_per_hand": per_hand,
                    "hedge_hands": _hedge_hands(delta_bucket, per_hand),
                    "extrapolated_tail": extrapolated_tail,
                }
            )
        return rows

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
        """
        Bucketed rhoq diagnostics per futures tenor (carry coordinate).
        The base PV is always computed internally (per-mode base environment);
        no ``base_price`` parameter — see calculate_futures_delta_buckets.

        One-sided **up** dividend bump, matching the scalar
        ``calculate_numerical_dividend_rho`` convention; output scaled to
        per-1% yield change via ``* (0.01 / div_bump)``.

        IMPLIED_FUTURES_CARRY: bumps one implied q(T_i) node at a time on the
        curve rebuilt from ``futures_curve``. THEORETICAL_CARRY: bumps
        ``pricing_env.div_yield`` on the interval (T_{i-1}, T_i] via
        ``BucketedDividendYield`` (the futures curve supplies metadata only).
        MARKET_PRICE is rejected: a zero bucket table can look like a real
        hedge result.
        """
        from quantark.asset.equity.market import bump_term_yield_node
        from quantark.asset.equity.report.term_structure import (
            BucketedDividendYield,
        )
        from quantark.param.div import ContinuousDividendYield
        from quantark.util.enum import FuturesCarryRiskMode

        resolved_mode = mode if mode is not None else futures_curve.mode
        if resolved_mode is FuturesCarryRiskMode.MARKET_PRICE:
            raise ValidationError(
                "calculate_futures_rhoq_buckets does not support MARKET_PRICE "
                "mode (it supplies no carry curve for repricing the option)"
            )
        div_bump = div_bump if div_bump is not None else self._bump_config.div_bump
        if div_bump <= 0.0:
            raise ValidationError("div_bump must be positive")
        engine = self._resolve_bump_engine(product, pricing_env, engine)

        maturity = product.get_maturity(pricing_env)
        last_index = len(futures_curve.quotes) - 1

        def _tail_flag(i: int, quote) -> bool:
            return (i == last_index and maturity > quote.maturity) or (
                i == 0 and maturity < quote.maturity
            )

        rows: List[Dict[str, object]] = []
        if resolved_mode is FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
            base_div = futures_curve.to_dividend_yield_curve(pricing_env.rate_curve)
            base_env = deepcopy(pricing_env)
            base_env.div_yield = base_div
            base_price = engine.price(product, base_env)
            for i, quote in enumerate(futures_curve.quotes):
                bumped_env = deepcopy(pricing_env)
                bumped_env.div_yield = bump_term_yield_node(base_div, i, div_bump)
                bumped_price = engine.price(product, bumped_env)
                rows.append(
                    self._rhoq_bucket_row(
                        quote, div_bump, base_price, bumped_price, _tail_flag(i, quote)
                    )
                )
        else:  # THEORETICAL_CARRY: pricing_env.div_yield is the carry source
            base_price = engine.price(product, pricing_env)
            base_div = pricing_env.div_yield
            if base_div is None:
                base_div = ContinuousDividendYield(0.0)
            edges = [0.0] + [q.maturity for q in futures_curve.quotes]
            # a product maturing beyond the last futures tenor still carries
            # dividend exposure on (T_last, T*]; attribute that tail to the
            # last contract's bucket (roll-hedge convention, mirrored from the
            # implied mode's flat extrapolation) instead of dropping it
            if maturity > edges[-1]:
                edges[-1] = maturity
            for i, quote in enumerate(futures_curve.quotes):
                bumped_env = deepcopy(pricing_env)
                bumped_env.div_yield = BucketedDividendYield(
                    base=base_div,
                    bucket_start=edges[i],
                    bucket_end=edges[i + 1],
                    bump=div_bump,
                )
                bumped_price = engine.price(product, bumped_env)
                rows.append(
                    self._rhoq_bucket_row(
                        quote, div_bump, base_price, bumped_price, _tail_flag(i, quote)
                    )
                )
        return rows

    @staticmethod
    def _rhoq_bucket_row(quote, div_bump, base_price, bumped_price, extrapolated_tail):
        return {
            "contract": quote.contract,
            "maturity": quote.maturity,
            "future_price": quote.price,
            "div_bump": div_bump,
            "rhoq_bucket": (bumped_price - base_price) * (0.01 / div_bump),
            "extrapolated_tail": extrapolated_tail,
        }

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
        """
        Fast estimation of theta components from existing Greeks.

        Uses the relationships between theta components and other Greeks:
            r_theta ≈ -r/T * rho / rate_bump (corrected for scale and daily conversion)
            q_theta ≈ -q/T * dividend_rho / dividend_bump (corrected for scale and daily conversion)
            convexity_theta ≈ theta - r_theta - q_theta

        This is an approximation that avoids repricing. For exact decomposition,
        use _calculate_numerical_theta_components() instead.

        Args:
            theta: Total theta (per day)
            rho: Rho (sensitivity to rate, per 1% change)
            dividend_rho: Dividend rho (sensitivity to dividend yield, per 1% change)
            r: Interest rate (annual)
            q: Dividend yield (annual)
            T: Time to maturity in years
            rate_bump: Rate scale of the rho input (default: 1% = 0.01)
            dividend_bump: Dividend scale of the dividend_rho input (default: 1% = 0.01)

        Returns:
            Dictionary with convexity_theta, r_theta, q_theta (all per day)
        """
        if is_zero(T):
            return {
                "convexity_theta": 0.0,
                "r_theta": 0.0,
                "q_theta": 0.0,
            }

        # Rho/Dividend Rho are per rate_bump/dividend_bump size, so divide by scale.
        # Divide by 365 to convert annual rate decay to daily theta equivalent.
        # Divide by T to cancel out the T term in Rho (Rho = dV/dr = T * dV/d(rT) approx).
        r_theta = -r / T * (rho / rate_bump) / 365
        q_theta = -q / T * (dividend_rho / dividend_bump) / 365
        convexity_theta = theta - r_theta - q_theta

        return {
            "convexity_theta": convexity_theta,
            "r_theta": r_theta,
            "q_theta": q_theta,
        }

    def _calculate_numerical_theta_components(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None,
        time_bump_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Exact numerical theta decomposition via repricing with zeroed r/q.

        This method computes exact theta components by repricing with different
        rate and dividend yield combinations:
            1. theta_no_rq = theta with r=0, q=0 → convexity_theta
            2. theta_no_q = theta with q=0 → r_theta = theta_no_q - convexity_theta
            3. theta_no_r = theta with r=0 → q_theta = theta_no_r - convexity_theta

        Note: This is computationally expensive (3 extra pricings) and should
        be treated as a slow path. For fast estimation, use
        estimate_theta_components() instead.

        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine
            base_price: Pre-calculated base price
            time_bump_days: Time bump in days

        Returns:
            Dictionary with convexity_theta, r_theta, q_theta (all per day)
        """
        from quantark.param.div import ContinuousDividendYield
        from quantark.param.rrf import FlatRateCurve

        time_bump_days = (
            time_bump_days
            if time_bump_days is not None
            else self._bump_config.time_bump_days
        )

        T = product.get_maturity(pricing_env)
        if is_zero(T):
            return {
                "convexity_theta": 0.0,
                "r_theta": 0.0,
                "q_theta": 0.0,
            }

        # Create environments with zeroed r and/or q
        env_no_r = deepcopy(pricing_env)
        env_no_r.rate_curve = FlatRateCurve(0.0)

        env_no_q = deepcopy(pricing_env)
        env_no_q.div_yield = ContinuousDividendYield(0.0)

        env_no_rq = deepcopy(pricing_env)
        env_no_rq.rate_curve = FlatRateCurve(0.0)
        env_no_rq.div_yield = ContinuousDividendYield(0.0)

        # Calculate theta in each environment
        theta_no_rq = self.calculate_numerical_theta(
            product, env_no_rq, engine, time_bump_days=time_bump_days
        )
        theta_no_q = self.calculate_numerical_theta(
            product, env_no_q, engine, time_bump_days=time_bump_days
        )
        theta_no_r = self.calculate_numerical_theta(
            product, env_no_r, engine, time_bump_days=time_bump_days
        )

        # Decompose
        convexity_theta = theta_no_rq
        r_theta = theta_no_q - convexity_theta
        q_theta = theta_no_r - convexity_theta

        return {
            "convexity_theta": convexity_theta,
            "r_theta": r_theta,
            "q_theta": q_theta,
        }

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
        base_price = self._ensure_base_price(product, pricing_env, engine, base_price)
        if reuse is not None:
            price_up_spot, price_down_spot = reuse
        else:
            env_up = deepcopy(pricing_env)
            env_up.spot_quote.spot *= 1 + bump
            price_up_spot = engine.price(product, env_up)

            env_down = deepcopy(pricing_env)
            env_down.spot_quote.spot *= 1 - bump
            price_down_spot = engine.price(product, env_down)

        return base_price, price_up_spot, price_down_spot

    def _build_vol_bumped_env(
        self,
        pricing_env: PricingEnvironment,
        product: BaseEquityProduct,
        current_vol: float,
        vol_bump: float,
        *,
        direction: float,
    ) -> PricingEnvironment:
        from quantark.param.vol import FlatVolSurface, TermStructureVolSurface

        new_vol = current_vol + direction * vol_bump
        if new_vol <= 0:
            raise ValidationError(
                f"Stressed volatility must be positive, got {new_vol}"
            )

        env = deepcopy(pricing_env)
        if isinstance(pricing_env.vol_surface, TermStructureVolSurface):
            new_vols = [float(v) + direction * vol_bump for v in pricing_env.vol_surface.vols]
            if any(v <= 0 for v in new_vols):
                raise ValidationError("Stressed term-structure vol must be positive.")
            env.vol_surface = TermStructureVolSurface(
                times=list(pricing_env.vol_surface.times), vols=new_vols
            )
        else:
            env.vol_surface = FlatVolSurface(new_vol)
        return env

    def _build_div_bumped_env(
        self,
        pricing_env: PricingEnvironment,
        product: BaseEquityProduct,
        current_div: float,
        div_bump: float,
        *,
        direction: float,
    ) -> PricingEnvironment:
        from quantark.param.div import ContinuousDividendYield, TermStructureDividendYield

        new_div = current_div + direction * div_bump

        env = deepcopy(pricing_env)
        if isinstance(pricing_env.div_yield, TermStructureDividendYield):
            new_yields = [
                float(y) + direction * div_bump for y in pricing_env.div_yield.yields
            ]
            env.div_yield = TermStructureDividendYield(
                times=list(pricing_env.div_yield.times), yields=new_yields
            )
        else:
            env.div_yield = ContinuousDividendYield(new_div)
        return env

    def _greeks_for_linear(
        self, product: BaseEquityProduct, price: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks for linear (delta-one) products.

        Delta-one products have trivial Greeks:
        - Delta = 1.0 (always)
        - Gamma, Vega, Theta, Rho, Dividend Rho = 0.0 (no optionality)

        Args:
            product: Delta-one product
            price: Current price

        Returns:
            Dictionary of Greeks
        """
        return {
            "price": price,
            "delta": 1.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "convexity_theta": 0.0,
            "r_theta": 0.0,
            "q_theta": 0.0,
            "rho": 0.0,
            "dividend_rho": 0.0,
        }

    def _greeks_at_expiry(
        self, product: EuropeanVanillaOption, spot: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks at expiry.

        At expiry:
        - Price = intrinsic value
        - Delta = 1 (ITM call), -1 (ITM put), 0 (OTM)
        - Gamma, Vega, Theta, Rho = 0

        Args:
            product: European vanilla option
            spot: Current spot price

        Returns:
            Dictionary of Greeks
        """
        multiplier = product.contract_multiplier
        price = product.get_payoff(spot) / multiplier

        # Delta at expiry
        if product.is_call():
            delta = 1.0 if spot > product.strike else 0.0
        else:
            delta = -1.0 if spot < product.strike else 0.0

        return {
            "price": price * multiplier,
            "delta": delta * multiplier,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "convexity_theta": 0.0,
            "r_theta": 0.0,
            "q_theta": 0.0,
            "rho": 0.0,
        }

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

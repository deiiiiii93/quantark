"""
Equity position class for tracking individual equity derivative positions.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import (
    CashLeg,
    LegPV,
    TradeGreeks,
    TradeValueBreakdown,
    value_leg,
)
from quantark.util.exceptions import ValidationError


@dataclass
class EquityPosition:
    """
    Represents a single equity derivative position in a portfolio.
    
    A position tracks a product (e.g., option) with quantity, entry details,
    and its own pricing engine.
    
    Attributes:
        product: The derivative product (e.g., EuropeanVanillaOption)
        quantity: Number of contracts/units (positive=long, negative=short)
        entry_price: Price at which position was entered
        underlying: Identifier for the underlying asset
        engine: Pricing engine specific to this position
        entry_timestamp: When the position was opened
        position_id: Unique identifier for this position
        cash_legs: Optional cash terms attached to this position
    """
    product: BaseEquityProduct
    quantity: float
    entry_price: float
    underlying: str
    engine: BaseEngine
    entry_timestamp: datetime
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cash_legs: List[CashLeg] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate position parameters."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate position parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        if self.quantity == 0:
            raise ValidationError("Position quantity cannot be zero")
        if self.entry_price < 0:
            raise ValidationError(f"Entry price must be non-negative, got {self.entry_price}")
        if not self.underlying:
            raise ValidationError("Underlying identifier is required")
        if self.product is None:
            raise ValidationError("Product is required")
        if self.engine is None:
            raise ValidationError("Engine is required")
    
    def get_current_price(self, pricing_env: PricingEnvironment) -> float:
        """
        Get current market price of the product.
        
        Uses the position's specific engine to price the product.
        
        Args:
            pricing_env: Pricing environment for the underlying
            
        Returns:
            Current market price
        """
        return self.engine.price(self.product, pricing_env)
    
    def get_market_value(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate current market value of the position.

        Market value = current_price × quantity
        
        Args:
            pricing_env: Pricing environment for the underlying
            
        Returns:
            Current market value
        """
        current_price = self.get_current_price(pricing_env)
        return current_price * self.quantity

    def _required_streams(self) -> "frozenset":
        """Union of EventDistribution streams the attached legs read [§11.1].

        Passed to ``price_with_events`` so the engine prunes indicator columns
        (e.g. KI) that no leg needs. ``None`` when there are no legs.
        """
        if not self.cash_legs:
            return frozenset()
        streams: frozenset = frozenset()
        for leg in self.cash_legs:
            streams = streams | leg.required_event_types()
        return streams

    def get_trade_value(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate full trade value including product and attached cash legs.

        For positions without cash legs, this is exactly get_market_value().
        """
        if not self.cash_legs:
            return self.get_market_value(pricing_env)

        needs_distribution = any(
            leg.requires_event_distribution() for leg in self.cash_legs
        )
        result = self.engine.price_with_events(
            self.product,
            pricing_env,
            emit_distribution=needs_distribution,
            streams=self._required_streams(),
        )
        unit_notional = self._get_unit_notional(pricing_env)
        leg_pv_total = sum(
            value_leg(leg, result.event_distribution, pricing_env, unit_notional)
            for leg in self.cash_legs
        )
        return (result.npv + leg_pv_total) * self.quantity

    def get_trade_value_breakdown(
        self, pricing_env: PricingEnvironment
    ) -> TradeValueBreakdown:
        """Return product and per-leg PV attribution."""
        if not self.cash_legs:
            return TradeValueBreakdown(
                product_npv=self.get_market_value(pricing_env),
                leg_pvs={},
            )

        needs_distribution = any(
            leg.requires_event_distribution() for leg in self.cash_legs
        )
        result = self.engine.price_with_events(
            self.product,
            pricing_env,
            emit_distribution=needs_distribution,
            streams=self._required_streams(),
        )
        unit_notional = self._get_unit_notional(pricing_env)
        leg_pvs = {}
        for leg in self.cash_legs:
            pv = (
                value_leg(leg, result.event_distribution, pricing_env, unit_notional)
                * self.quantity
            )
            leg_pvs[leg.leg_id] = LegPV(
                name=leg.name,
                direction=leg.direction,
                pv=pv,
            )

        return TradeValueBreakdown(
            product_npv=result.npv * self.quantity,
            leg_pvs=leg_pvs,
        )

    def get_trade_greeks(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: GreeksCalculator,
    ) -> Dict[str, float]:
        """Calculate trade-level Greeks by bumping get_trade_value()."""
        from copy import deepcopy
        from quantark.param import FlatRateCurve

        bump = self.engine.params.bump_size
        base_value = self.get_trade_value(pricing_env)

        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= 1.0 + bump
        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= 1.0 - bump
        value_up = self.get_trade_value(env_up)
        value_down = self.get_trade_value(env_down)

        spot_bump_amount = pricing_env.spot * bump
        delta = (value_up - value_down) / (2.0 * spot_bump_amount)
        gamma = (value_up - 2.0 * base_value + value_down) / (spot_bump_amount**2)

        env_vol_up = deepcopy(pricing_env)
        env_vol_down = deepcopy(pricing_env)
        self._bump_flat_or_term_vol(env_vol_up, bump)
        self._bump_flat_or_term_vol(env_vol_down, -bump)
        vega = (
            self.get_trade_value(env_vol_up) - self.get_trade_value(env_vol_down)
        ) / (2.0 * bump)

        current_rate = pricing_env.get_rate(self.product.get_maturity(pricing_env))
        env_rate_up = deepcopy(pricing_env)
        env_rate_down = deepcopy(pricing_env)
        env_rate_up.rate_curve = FlatRateCurve(current_rate + bump)
        env_rate_down.rate_curve = FlatRateCurve(current_rate - bump)
        rho = (
            self.get_trade_value(env_rate_up) - self.get_trade_value(env_rate_down)
        ) / (2.0 * bump)

        return {
            "price": base_value,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
        }

    def get_trade_risk(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: GreeksCalculator,
        greeks: Optional[List[str]] = None,
    ) -> TradeGreeks:
        """Product + full-trade greeks from ONE frozen-grid bump loop [§11.4, §11.8].

        The frozen bump context is resolved once; every bump reprices the product
        AND its cash legs from a single ``price_with_events`` call (npv + event
        distribution), so product and total greeks come from the same solves
        instead of two independent bump loops. Delta and gamma share the two spot
        bumps. Theta shifts the product and the legs consistently.

        Contract [§11.8]: ``product`` greeks = ``quantity * d(npv)``; ``total``
        greeks = ``quantity * d(npv) + d(Σ leg_pv)`` — legs carry their own
        direction and are not scaled by the note quantity sign.
        """
        from copy import deepcopy

        from quantark.param.rrf import FlatRateCurve

        gc = greeks_calculator
        bc = gc._bump_config
        product = self.product
        q = float(self.quantity)
        streams = self._required_streams()
        requested = set(
            greeks
            if greeks is not None
            else ["delta", "gamma", "vega", "theta", "rho", "dividend_rho"]
        )

        bump_engine = gc._resolve_bump_engine(product, pricing_env, self.engine)

        def reprice(prod, env, legs):
            """Return (npv, Σ leg_pv) from one solve on the frozen bump engine."""
            result = bump_engine.price_with_events(prod, env, streams=streams)
            unit = self._get_unit_notional(env)
            leg_sum = sum(
                value_leg(leg, result.event_distribution, env, unit) for leg in legs
            )
            return result.npv, leg_sum

        # Base solve: npv, per-leg PVs (absolute), and Σ leg_pv.
        base_result = bump_engine.price_with_events(
            product, pricing_env, streams=streams
        )
        unit0 = self._get_unit_notional(pricing_env)
        base_npv = float(base_result.npv)
        leg_pvs: Dict[str, LegPV] = {}
        base_legs = 0.0
        for leg in self.cash_legs:
            pv = float(
                value_leg(leg, base_result.event_distribution, pricing_env, unit0)
            )
            base_legs += pv
            leg_pvs[leg.leg_id] = LegPV(name=leg.name, direction=leg.direction, pv=pv)

        product_g: Dict[str, float] = {"price": q * base_npv}
        total_g: Dict[str, float] = {"price": q * base_npv + base_legs}

        def record(name, d_npv, d_legs):
            product_g[name] = q * d_npv
            total_g[name] = q * d_npv + d_legs

        # Delta / gamma: two shared spot bumps.
        if {"delta", "gamma"} & requested:
            h = pricing_env.spot * bc.spot_bump
            env_up = deepcopy(pricing_env)
            env_up.spot_quote.spot *= 1.0 + bc.spot_bump
            env_dn = deepcopy(pricing_env)
            env_dn.spot_quote.spot *= 1.0 - bc.spot_bump
            nu, lu = reprice(product, env_up, self.cash_legs)
            nd, ld = reprice(product, env_dn, self.cash_legs)
            if "delta" in requested:
                record("delta", (nu - nd) / (2.0 * h), (lu - ld) / (2.0 * h))
            if "gamma" in requested:
                record(
                    "gamma",
                    (nu - 2.0 * base_npv + nd) / h**2,
                    (lu - 2.0 * base_legs + ld) / h**2,
                )

        T = product.get_maturity(pricing_env)

        # Vega: one-sided vol bump. Convention MUST match
        # GreeksCalculator.calculate_numerical_vega — the canonical greek used
        # everywhere else reports the RAW P&L for a `vol_bump` move (NOT a
        # dV/dsigma derivative); do not divide by the bump.
        if "vega" in requested:
            strike = getattr(product, "strike", pricing_env.spot)
            cur_vol = pricing_env.get_vol(strike, T)
            env_v = gc._build_vol_bumped_env(
                pricing_env, product, cur_vol, bc.vol_bump, direction=1.0
            )
            nv, lv = reprice(product, env_v, self.cash_legs)
            record("vega", nv - base_npv, lv - base_legs)

        # Rho: one-sided rate bump, rescaled to per 1% change — exactly matching
        # GreeksCalculator.calculate_numerical_rho (`(price_up - base) * 0.01/rate_bump`,
        # a SINGLE rescale, not a derivative-then-rescale).
        if "rho" in requested:
            cur_rate = pricing_env.get_rate(T)
            env_r = deepcopy(pricing_env)
            env_r.rate_curve = FlatRateCurve(cur_rate + bc.rate_bump)
            nr, lr = reprice(product, env_r, self.cash_legs)
            scale = 0.01 / bc.rate_bump
            record("rho", (nr - base_npv) * scale, (lr - base_legs) * scale)

        # Dividend rho: one-sided div bump, rescaled to per 1% change — matching
        # GreeksCalculator.calculate_numerical_dividend_rho (single rescale).
        if "dividend_rho" in requested:
            cur_div = pricing_env.get_div_yield(T)
            env_d = gc._build_div_bumped_env(
                pricing_env, product, cur_div, bc.div_bump, direction=1.0
            )
            nd_, ld_ = reprice(product, env_d, self.cash_legs)
            scale = 0.01 / bc.div_bump
            record("dividend_rho", (nd_ - base_npv) * scale, (ld_ - base_legs) * scale)

        # Theta: shift product AND legs by the same time bump.
        if "theta" in requested:
            product_g["theta"], total_g["theta"] = self._trade_theta(
                gc, bump_engine, pricing_env, streams, base_npv, base_legs, q
            )

        return TradeGreeks(product=product_g, total=total_g, leg_pvs=leg_pvs)

    def _trade_theta(
        self, gc, bump_engine, pricing_env, streams, base_npv, base_legs, q
    ) -> tuple:
        """Theta for product + total via a time shift of product and legs."""
        from copy import deepcopy

        bc = gc._bump_config
        product = self.product
        bumped_date, time_bump, _mode = gc._advance_theta_bump(
            pricing_env, bc.time_bump_days, getattr(bc, "time_bump_mode", "auto")
        )
        current_maturity = product.get_maturity(pricing_env)
        if time_bump <= 0.0 or current_maturity <= time_bump:
            return 0.0, 0.0

        prod_theta = deepcopy(product)
        env_theta = deepcopy(pricing_env)
        env_theta.valuation_date = bumped_date
        dropped_all = prod_theta.time_shift(time_bump, bumped_date, env_theta)
        if dropped_all:
            return 0.0, 0.0

        shifted_legs = []
        for leg in self.cash_legs:
            shift = getattr(leg, "time_shift", None)
            shifted = shift(time_bump) if callable(shift) else leg
            if shifted is not None:
                shifted_legs.append(shifted)

        result = bump_engine.price_with_events(prod_theta, env_theta, streams=streams)
        unit = self._get_unit_notional(env_theta)
        npv_t = float(result.npv)
        legs_t = sum(
            value_leg(leg, result.event_distribution, env_theta, unit)
            for leg in shifted_legs
        )
        product_theta = q * (npv_t - base_npv)
        total_theta = q * (npv_t - base_npv) + (legs_t - base_legs)
        return product_theta, total_theta

    def get_actual_notional(
        self, pricing_env: Optional[PricingEnvironment] = None
    ) -> float:
        """
        Calculate actual notional based on contract multiplier.

        Uses product.initial_price when available; otherwise falls back to spot
        from pricing_env.

        Args:
            pricing_env: Pricing environment for spot fallback (optional)

        Returns:
            Actual notional amount
        """
        base_price = getattr(self.product, "initial_price", 0.0) or 0.0
        if base_price <= 0:
            if pricing_env is None:
                raise ValidationError(
                    "PricingEnvironment required when product.initial_price is not set"
                )
            base_price = pricing_env.spot

        contract_multiplier = getattr(self.product, "contract_multiplier", 1.0)
        return self.quantity * base_price * contract_multiplier

    def _get_unit_notional(self, pricing_env: PricingEnvironment) -> float:
        """Return per-unit notional so quantity scales product and leg PV once."""
        return self.get_actual_notional(pricing_env) / self.quantity

    @staticmethod
    def _bump_flat_or_term_vol(pricing_env: PricingEnvironment, bump: float) -> None:
        vol_surface = pricing_env.vol_surface
        if hasattr(vol_surface, "volatility"):
            vol_surface.volatility += bump
            return
        if hasattr(vol_surface, "vols"):
            vol_surface.vols = [float(vol) + bump for vol in vol_surface.vols]
            return
        raise ValidationError(
            f"Unsupported volatility surface bump for {type(vol_surface).__name__}"
        )
    
    def get_pnl(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate unrealized P&L of the position.
        
        P&L = (current_price - entry_price) × quantity
        
        Args:
            pricing_env: Pricing environment for the underlying
            
        Returns:
            Unrealized profit/loss
        """
        current_price = self.get_current_price(pricing_env)
        return (current_price - self.entry_price) * self.quantity
    
    def get_greeks(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: GreeksCalculator,
        use_analytical: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Greeks for this position.
        
        Greeks are scaled by position quantity.
        
        Args:
            pricing_env: Pricing environment for the underlying
            greeks_calculator: Greeks calculator instance
            use_analytical: If True, use analytical Greeks when possible
            
        Returns:
            Dictionary of Greeks scaled by quantity
        """
        # Calculate single-contract Greeks
        if use_analytical:
            try:
                greeks = greeks_calculator.calculate_analytical_greeks(
                    self.product,
                    pricing_env
                )
            except (ValidationError, AttributeError):
                # Fall back to numerical if analytical not available
                greeks = greeks_calculator.calculate_numerical_greeks(
                    self.product,
                    pricing_env,
                    self.engine
                )
        else:
            greeks = greeks_calculator.calculate_numerical_greeks(
                self.product,
                pricing_env,
                self.engine
            )
        
        # Scale Greeks by quantity
        scaled_greeks = {}
        for key, value in greeks.items():
            if key == 'price':
                # For price, use market value instead
                scaled_greeks['market_value'] = value * self.quantity
            else:
                scaled_greeks[key] = value * self.quantity
        
        return scaled_greeks
    
    def get_risk_measures(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: Optional[GreeksCalculator] = None,
        use_analytical: bool = True
    ) -> Dict[str, float]:
        """
        Calculate risk measures for this position.
        
        For equity positions, this returns Greeks (delta, gamma, vega, theta, rho).
        
        Args:
            pricing_env: Pricing environment for the underlying
            greeks_calculator: Greeks calculator instance (optional)
            use_analytical: If True, use analytical Greeks when possible
            
        Returns:
            Dictionary of risk measures
        """
        if greeks_calculator is None:
            greeks_calculator = GreeksCalculator()
        return self.get_greeks(pricing_env, greeks_calculator, use_analytical)

    def get_simm_sensitivities(self, config: Any, market_data: Any):
        """Return built-in SIMM equity sensitivities for this position."""
        from quantark.simm.engines.risk_class.equity_engine import EquitySensitivityEngine

        if config.calculation_currency.upper() != "USD":
            raise ValidationError(
                "Built-in equity SIMM generation assumes USD-valued pricing "
                "environments; supply explicit translated sensitivities for "
                f"calculation currency {config.calculation_currency}"
            )
        if not isinstance(market_data, dict):
            raise ValidationError(
                "Equity SIMM sensitivity generation requires pricing environments "
                "mapped by underlying"
            )
        if self.underlying not in market_data:
            raise ValidationError(
                f"Missing equity pricing environment for {self.underlying}"
            )
        return EquitySensitivityEngine(config).calculate_sensitivities(
            [self], market_data, config
        )
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize position to dictionary.
        
        Returns:
            Dictionary representation of position
        """
        return {
            'position_id': self.position_id,
            'underlying': self.underlying,
            'product_type': self.product.__class__.__name__,
            'product_repr': repr(self.product),
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_timestamp': self.entry_timestamp.isoformat(),
            'engine_type': self.engine.__class__.__name__,
            'direction': 'LONG' if self.is_long() else 'SHORT',
            'asset_class': 'equity',
        }
    
    def __repr__(self):
        direction = "LONG" if self.is_long() else "SHORT"
        return (
            f"EquityPosition(id={self.position_id[:8]}..., "
            f"{direction} {abs(self.quantity)} x {self.product.__class__.__name__}, "
            f"entry=${self.entry_price:.2f}, underlying={self.underlying})"
        )


# Backward compatibility alias
Position = EquityPosition

"""Equity Total Return Swap position for the risk/analytics stack.

An :class:`EquitySwapPosition` wraps a realized-cashflow TRS product
(:class:`~quantark.asset.equity.product.swap.base_swap.BaseSwap`) and exposes the
asset-agnostic :class:`~quantark.portfolio.base.BasePosition` interface used by the
portfolio, VaR, stress, dynamic-scenario and backtest layers.

Unlike :class:`EquityPosition` (which wraps a payoff-on-spot ``BaseEquityProduct``
and prices through its engine), the swap is marked to market by the
:class:`~quantark.asset.equity.engine.cashflow.trs_valuation.TRSValuationEngine`,
which re-runs the cashflow engine against a spot/funding override implied by the
supplied :class:`PricingEnvironment`. The position therefore plugs into the same
deepcopy-and-bump revaluation machinery as every other position type.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from quantark.asset.equity.product.swap.base_swap import BaseSwap
from quantark.asset.equity.engine.cashflow.trs_valuation import TRSValuationEngine
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

#: Equity core greeks aggregated across positions by the risk stack.
_CORE_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


@dataclass
class EquitySwapPosition:
    """A single equity Total Return Swap position in a portfolio.

    Attributes:
        product: The TRS product (e.g. ``OneAssetTotalReturnSwap``). Its
            ``params`` carry the observed path, legs, events, margin and
            valuation date.
        quantity: Whole-swap multiplier (positive=long, negative=short). The swap
            carries its own notional, so this is normally ``1`` / ``-1``.
        underlying: Underlying identifier; defaults to the asset id on the product.
        entry_price: Mark-to-market at which the position was entered.
        funding_rate_ref: Reference funding rate for env rate flow-through; see
            :class:`TRSValuationEngine`. ``None`` keeps the contractual fixed
            financing rate (env rate shocks do not move base MtM).
        entry_timestamp: When the position was opened.
        position_id: Unique identifier.
    """

    product: BaseSwap
    quantity: float = 1.0
    underlying: Optional[str] = None
    entry_price: float = 0.0
    funding_rate_ref: Optional[float] = None
    entry_timestamp: datetime = field(default_factory=datetime.now)
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        params = getattr(self.product, "params", None)
        if params is None:
            raise ValidationError(
                "EquitySwapPosition requires a TRS product exposing `params` "
                f"(got {type(self.product).__name__})"
            )
        self._params = params
        if self.underlying is None:
            self.underlying = params.asset.asset_id
        self._valuator = TRSValuationEngine(
            params, funding_rate_ref=self.funding_rate_ref
        )
        self.validate()

    def validate(self) -> None:
        """Validate position parameters."""
        if self.quantity == 0:
            raise ValidationError("Position quantity cannot be zero")
        if self.product is None:
            raise ValidationError("Product is required")
        if not self.underlying:
            raise ValidationError("Underlying identifier is required")

    # ------------------------------------------------------------------ #
    # Risk-module interface (BasePosition)
    # ------------------------------------------------------------------ #
    def get_current_price(self, pricing_env: PricingEnvironment) -> float:
        """Mark-to-market of one swap under the pricing environment."""
        return self._valuator.mark_to_market(pricing_env)

    def get_market_value(self, pricing_env: PricingEnvironment) -> float:
        """Market value = mark-to-market x quantity."""
        return self.get_current_price(pricing_env) * self.quantity

    def get_pnl(self, pricing_env: PricingEnvironment) -> float:
        """Unrealized P&L versus the entry mark."""
        return (self.get_current_price(pricing_env) - self.entry_price) * self.quantity

    def get_greeks(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: Any = None,
        use_analytical: bool = True,
        *,
        spot_bump: float = 0.01,
        rate_bump: float = 1e-4,
    ) -> Dict[str, float]:
        """Quantity-scaled equity core greeks for this swap position.

        ``greeks_calculator`` / ``use_analytical`` are accepted for call-signature
        compatibility with :class:`EquityPortfolio` aggregation but ignored: a TRS
        is marked by re-running its cashflow engine, so greeks are intrinsic
        finite differences (see :class:`TRSValuationEngine`).
        """
        raw = self._valuator.greeks(
            pricing_env, spot_bump=spot_bump, rate_bump=rate_bump
        )
        scaled = {key: raw.get(key, 0.0) * self.quantity for key in _CORE_GREEKS}
        scaled["market_value"] = raw["price"] * self.quantity
        return scaled

    def get_risk_measures(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: Any = None,
        use_analytical: bool = True,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Asset-agnostic risk-measure entry point (returns equity greeks)."""
        return self.get_greeks(
            pricing_env, greeks_calculator, use_analytical, **kwargs
        )

    # ------------------------------------------------------------------ #
    # SIMM interface (SIMMSensitivityProvider)
    # ------------------------------------------------------------------ #
    def get_simm_sensitivities(self, config: Any, market_data: Any):
        """Return ISDA SIMM equity sensitivities for this TRS position.

        A TRS is delta-one equity exposure, so the equity sensitivity engine
        derives a single EquityDelta from the position's (quantity-scaled) delta
        via ``get_greeks`` — the same duck-typed path used for option positions.
        Vega is zero (no volatility dependence). ``market_data`` is the
        per-underlying pricing-environment map.
        """
        from quantark.simm.engines.risk_class.equity_engine import (
            EquitySensitivityEngine,
        )

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

    # ------------------------------------------------------------------ #
    # SA-CCR interface
    # ------------------------------------------------------------------ #
    def to_saccr_trade(
        self, pricing_env: PricingEnvironment, *, is_index: bool = False
    ) -> Any:
        """Map this TRS to a SA-CCR equity-derivative trade (paragraphs 176-178).

        A TRS is a linear (delta-one) equity trade: its SA-CCR adjusted notional
        is ``current spot x number of shares`` and its supervisory delta is +/-1
        by economic direction. The current mark-to-market feeds the netting set's
        replacement cost.

        Args:
            pricing_env: Environment supplying the current spot / MtM.
            is_index: True if the underlying is an equity index (drives the 20%
                index supervisory factor vs 32% single-name).
        """
        from quantark.saccr.models.trade import SACCRTrade, MIN_MATURITY
        from quantark.saccr.models.enums import AssetClass, Position
        from quantark.asset.equity.product.swap.trs_params import SwapState

        # A matured swap (valuation date past contract end) carries no future
        # counterparty exposure; do not fabricate a floored 10-business-day trade
        # for it. remaining_tenor() floors at zero, so check the lifecycle state.
        if getattr(self.product, "state", None) == SwapState.MATURED:
            raise ValidationError(
                f"TRS {self.position_id} has matured (valuation date past contract "
                "end); a matured swap carries no SA-CCR counterparty exposure"
            )

        shares = (
            self._params.float_leg.initial_notional
            / self._params.asset.asset_initial_price
        ) * abs(self.quantity)
        adjusted_notional = pricing_env.spot * shares

        # Economic long/short = quantity sign x the float (total-return) leg
        # direction; the supervisory delta carries this sign, so the notional is
        # the absolute exposure.
        economic_direction = self.quantity * self._params.float_leg.direction
        position = Position.LONG if economic_direction >= 0 else Position.SHORT

        # Floor the remaining tenor at SA-CCR's 10-business-day minimum and use the
        # floored value for both maturity (M_i) and end_date (E_i) so the trade is
        # internally consistent (M_i <= E_i) for near-expiry swaps. SACCRTrade also
        # floors maturity, but not end_date — set both here explicitly.
        maturity = max(self._valuator.remaining_tenor(), MIN_MATURITY)
        return SACCRTrade(
            trade_id=self.position_id,
            asset_class=AssetClass.EQUITY,
            notional=adjusted_notional,
            market_value=self.get_market_value(pricing_env),
            maturity=maturity,
            start_date=0.0,
            end_date=maturity,
            reference_entity=self.underlying,
            is_index=is_index,
            position=position,
        )

    def is_long(self) -> bool:
        return self.quantity > 0

    def is_short(self) -> bool:
        return self.quantity < 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "underlying": self.underlying,
            "product_type": type(self.product).__name__,
            "contract_id": getattr(self._params, "contract_id", None),
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "direction": "LONG" if self.is_long() else "SHORT",
            "asset_class": "equity_swap",
        }

    def __repr__(self) -> str:
        direction = "LONG" if self.is_long() else "SHORT"
        return (
            f"EquitySwapPosition(id={self.position_id[:8]}..., "
            f"{direction} {abs(self.quantity)} x {type(self.product).__name__}, "
            f"underlying={self.underlying})"
        )

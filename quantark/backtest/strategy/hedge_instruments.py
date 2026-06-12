"""
Hedge instrument specifications for multi-instrument hedging strategies.

A hedge instrument is a *specification* of what a strategy trades (spot,
futures, or an option defined by moneyness and tenor), not a live position.
The executor turns specs into concrete products at trade time: an
OptionHedgeInstrument struck at 1.0x moneyness creates a new ATM option at
the prevailing spot whenever a fresh contract is needed, and signals via
requires_roll() when a held contract is too close to expiry (or has
drifted too far from its target moneyness) and must be closed.

Option contracts are created with a date-based expiry so they age
correctly as the backtest valuation date advances.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional

from quantark.asset.equity.engine.analytical import BlackScholesEngine, DeltaOneEngine
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.deltaone_enums import DeltaOneType
from quantark.util.exceptions import ValidationError


class BaseHedgeInstrument(ABC):
    """
    Abstract specification of a tradeable hedge instrument.

    Attributes:
        name: Unique identifier within a strategy's instrument set
        instrument_type: Type label passed to transaction cost models
            (e.g. 'spot', 'futures', 'option')
    """

    def __init__(self, name: str, instrument_type: str):
        """
        Initialize hedge instrument specification.

        Args:
            name: Unique identifier within a strategy's instrument set
            instrument_type: Type label for transaction cost models
        """
        if not name:
            raise ValidationError("Hedge instrument name must be non-empty")
        self.name = name
        self.instrument_type = instrument_type

    @abstractmethod
    def create_product(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
    ) -> BaseEquityProduct:
        """
        Create a concrete tradeable product from this specification.

        Args:
            underlying: Underlying asset identifier
            pricing_env: Current pricing environment (e.g. for strike setting)
            current_time: Trade timestamp (e.g. for expiry date setting)

        Returns:
            Product instance ready to be added to a portfolio
        """

    @abstractmethod
    def create_engine(self) -> BaseEngine:
        """Create the pricing engine used for this instrument's products."""

    def requires_roll(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> bool:
        """
        Whether a held contract must be closed and replaced.

        Linear instruments never roll; option specs roll near expiry or on
        excessive moneyness drift.

        Args:
            product: Currently held contract created from this spec
            pricing_env: Current pricing environment

        Returns:
            True if the contract should be closed before further hedging
        """
        return False

    def unit_greeks(
        self,
        product: BaseEquityProduct,
        engine: BaseEngine,
        pricing_env: PricingEnvironment,
        greeks_calculator: GreeksCalculator,
    ) -> Dict[str, float]:
        """
        Per-unit Greeks of a concrete contract.

        Routed through the same GreeksCalculator used for portfolio
        aggregation so that hedge sizing and portfolio Greeks are computed
        in an identical measure.

        Args:
            product: Concrete contract
            engine: Pricing engine for the contract
            pricing_env: Current pricing environment
            greeks_calculator: Calculator shared with the backtest engine

        Returns:
            Dictionary of per-unit Greeks (delta, gamma, vega, ...)
        """
        return greeks_calculator.calculate(product, pricing_env, engine)

    def unit_price(
        self,
        product: BaseEquityProduct,
        engine: BaseEngine,
        pricing_env: PricingEnvironment,
    ) -> float:
        """Per-unit price of a concrete contract."""
        return engine.price(product, pricing_env)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class SpotHedgeInstrument(BaseHedgeInstrument):
    """Underlying spot as a hedge instrument (delta only)."""

    def __init__(self, name: str = "spot"):
        super().__init__(name=name, instrument_type="spot")

    def create_product(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
    ) -> SpotInstrument:
        return SpotInstrument(underlying=underlying, deltaone_type=DeltaOneType.STOCK)

    def create_engine(self) -> DeltaOneEngine:
        return DeltaOneEngine()


class FuturesHedgeInstrument(BaseHedgeInstrument):
    """
    Futures contract as a hedge instrument (delta only).

    Attributes:
        maturity: Contract maturity in years (static, consistent with the
            existing single-instrument futures hedge)
        multiplier: Contract multiplier
    """

    def __init__(
        self, maturity: float, multiplier: float = 1.0, name: str = "futures"
    ):
        super().__init__(name=name, instrument_type="futures")
        if maturity <= 0:
            raise ValidationError(
                f"Futures maturity must be positive, got {maturity}"
            )
        if multiplier <= 0:
            raise ValidationError(
                f"Futures multiplier must be positive, got {multiplier}"
            )
        self.maturity = maturity
        self.multiplier = multiplier

    def create_product(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
    ) -> Futures:
        return Futures(
            underlying=underlying,
            multiplier=self.multiplier,
            maturity=self.maturity,
        )

    def create_engine(self) -> DeltaOneEngine:
        return DeltaOneEngine()


class OptionHedgeInstrument(BaseHedgeInstrument):
    """
    European vanilla option spec for gamma/vega hedging.

    Contracts are struck at moneyness x spot and expire tenor years from
    the trade date (date-based, so they age with the backtest clock).
    A held contract is rolled when its remaining maturity falls below
    min_time_to_expiry, or when spot drifts so far that the contract's
    moneyness deviates from the spec by more than restrike_band.

    Attributes:
        tenor: Time to expiry in years for freshly created contracts
        moneyness: Strike as a fraction of spot at creation (1.0 = ATM)
        option_type: CALL or PUT
        min_time_to_expiry: Remaining maturity (years) below which a held
            contract is rolled
        restrike_band: Optional absolute moneyness drift (|K/S - moneyness|)
            beyond which a held contract is rolled; None disables
        contract_multiplier: Contract multiplier for created options
    """

    def __init__(
        self,
        tenor: float,
        moneyness: float = 1.0,
        option_type: OptionType = OptionType.CALL,
        name: str = "option",
        min_time_to_expiry: float = 7.0 / 365.0,
        restrike_band: Optional[float] = None,
        contract_multiplier: float = 1.0,
    ):
        super().__init__(name=name, instrument_type="option")
        if tenor <= 0:
            raise ValidationError(f"Option tenor must be positive, got {tenor}")
        if moneyness <= 0:
            raise ValidationError(
                f"Option moneyness must be positive, got {moneyness}"
            )
        if not 0 < min_time_to_expiry < tenor:
            raise ValidationError(
                f"min_time_to_expiry must be in (0, tenor={tenor}), "
                f"got {min_time_to_expiry}"
            )
        if restrike_band is not None and restrike_band <= 0:
            raise ValidationError(
                f"restrike_band must be positive or None, got {restrike_band}"
            )
        self.tenor = tenor
        self.moneyness = moneyness
        self.option_type = option_type
        self.min_time_to_expiry = min_time_to_expiry
        self.restrike_band = restrike_band
        self.contract_multiplier = contract_multiplier

    def create_product(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
    ) -> EuropeanVanillaOption:
        strike = self.moneyness * pricing_env.spot
        exercise_date = current_time + timedelta(days=self.tenor * 365.0)
        return EuropeanVanillaOption(
            strike=strike,
            option_type=self.option_type,
            exercise_date=exercise_date,
            contract_multiplier=self.contract_multiplier,
        )

    def create_engine(self) -> BlackScholesEngine:
        return BlackScholesEngine()

    def requires_roll(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> bool:
        if (
            not isinstance(product, EuropeanVanillaOption)
            or product.exercise_date is None
        ):
            raise ValidationError(
                f"OptionHedgeInstrument '{self.name}' holds an unexpected "
                f"contract: {product!r} (expected a date-based European "
                "vanilla option created by this spec)"
            )
        # get_maturity raises once valuation reaches expiry, so compare
        # dates before asking for the remaining tenor
        if pricing_env.valuation_date >= product.exercise_date:
            return True
        if product.get_maturity(pricing_env) < self.min_time_to_expiry:
            return True
        if self.restrike_band is not None:
            drift = abs(product.strike / pricing_env.spot - self.moneyness)
            if drift > self.restrike_band:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"OptionHedgeInstrument(name={self.name}, tenor={self.tenor}, "
            f"moneyness={self.moneyness}, type={self.option_type.name})"
        )

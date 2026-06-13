"""
Credit position class for tracking individual CDS-type positions.

Mirrors :class:`~quantark.portfolio.fx.position.FXPosition`:

* **Risk-module layer** — prices through a
  :class:`~quantark.priceenv.CreditPricingEnvironment` and reports the credit
  Greeks (``cs01, ir01, rec01``), consumed by the VaR / stress /
  dynamic-scenario / backtest modules.
* **SIMM sensitivity provider** — implements ``get_simm_sensitivities`` so the
  same position feeds the ISDA SIMM v2.6 CreditQ / CreditNQ initial-margin
  calculation via :class:`SIMMPortfolioAdapter`.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

from quantark.asset.credit.engine.base_credit_engine import BaseCreditEngine
from quantark.asset.credit.product.base_credit_product import BaseCreditProduct
from quantark.asset.credit.riskmeasures import CreditGreeksCalculator
from quantark.priceenv import CreditPricingEnvironment
from quantark.simm.engines.classification.bucket_mapper import BucketMapper
from quantark.simm.sensitivity import CreditDeltaSensitivity, SensitivityCollection
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

#: Core credit Greeks aggregated across positions.
_CORE_GREEKS = ("cs01", "ir01", "rec01")

#: Shared bucket mapper (issuer -> SIMM credit bucket).
_BUCKET_MAPPER = BucketMapper()


@dataclass
class CreditPosition:
    """
    A single credit (CDS-type) position in a portfolio.

    Attributes:
        product: The credit product (e.g. a single-name CDS).
        quantity: Number of contracts (positive=long, negative=short). The
            product carries its own notional; quantity is a whole-contract
            multiplier.
        engine: Credit pricing engine for this position.
        reference_entity: Issuer/reference-entity identifier (also the key into
            the portfolio's pricing-environment map and the SIMM issuer).
        entry_price: NPV at which the position was entered.
        is_qualifying: True for SIMM Credit Qualifying (investment grade).
        payment_currency: Trade payment currency (a credit risk-factor key).
        seniority: Reference obligation seniority (informational).
        simm_bucket: Explicit SIMM credit bucket; derived from the issuer when
            omitted.
        entry_timestamp: When the position was opened.
        position_id: Unique identifier.
    """

    product: BaseCreditProduct
    quantity: float
    engine: BaseCreditEngine
    reference_entity: str
    entry_price: float = 0.0
    is_qualifying: bool = True
    payment_currency: str = "USD"
    seniority: str = "SENIOR"
    simm_bucket: Optional[int] = None
    entry_timestamp: datetime = field(default_factory=datetime.now)
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.quantity == 0:
            raise ValidationError("Position quantity cannot be zero")
        if not self.reference_entity:
            raise ValidationError("reference_entity is required")
        if self.product is None:
            raise ValidationError("Product is required")
        if self.engine is None:
            raise ValidationError("Engine is required")

    # ------------------------------------------------------------------ #
    # Risk-module interface
    # ------------------------------------------------------------------ #
    def get_current_price(self, env: CreditPricingEnvironment) -> float:
        """Present value of one contract (position holder's perspective)."""
        return self.engine.price(self.product, env)

    def get_market_value(self, env: CreditPricingEnvironment) -> float:
        return self.get_current_price(env) * self.quantity

    def get_pnl(self, env: CreditPricingEnvironment) -> float:
        return (self.get_current_price(env) - self.entry_price) * self.quantity

    def get_greeks(
        self,
        env: CreditPricingEnvironment,
        greeks_calculator: Optional[CreditGreeksCalculator] = None,
    ) -> Dict[str, float]:
        """Quantity-scaled credit Greeks (cs01, ir01, rec01)."""
        calc = greeks_calculator or CreditGreeksCalculator()
        raw = calc.calculate(self.product, env, self.engine)
        return {key: raw.get(key, 0.0) * self.quantity for key in _CORE_GREEKS}

    def get_risk_measures(
        self, env: CreditPricingEnvironment, **kwargs: Any
    ) -> Dict[str, float]:
        return self.get_greeks(env, kwargs.get("greeks_calculator"))

    def is_long(self) -> bool:
        return self.quantity > 0

    def is_short(self) -> bool:
        return self.quantity < 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "reference_entity": self.reference_entity,
            "product_type": type(self.product).__name__,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "is_qualifying": self.is_qualifying,
            "entry_timestamp": self.entry_timestamp.isoformat(),
        }

    # ------------------------------------------------------------------ #
    # SIMM interface (SIMMSensitivityProvider)
    # ------------------------------------------------------------------ #
    def _bucket(self) -> int:
        if self.simm_bucket is not None:
            return self.simm_bucket
        bucket, _ = _BUCKET_MAPPER.classify_credit_bucket(
            self.reference_entity, self.is_qualifying
        )
        return bucket

    def get_simm_sensitivities(
        self, config: Any, market_data: Any
    ) -> SensitivityCollection:
        """
        Produce the ISDA SIMM credit-spread delta for this position.

        The credit-spread CS01 (PV change per 1bp move of the issuer's CDS
        spread) is computed by chain rule from the hazard sensitivity: under
        the credit triangle ``s = lambda (1 - R)`` a 1bp spread move equals a
        ``1bp / (1 - R)`` hazard move. For a single-maturity CDS on a flat
        hazard curve all spread risk sits at the contract maturity vertex, so
        the sensitivity is reported there (the vertex is snapped by the
        sensitivity class).
        """
        if not isinstance(market_data, dict):
            raise ValidationError(
                "Credit SIMM sensitivity generation requires pricing environments "
                "mapped by reference entity"
            )
        env = market_data.get(self.reference_entity)
        if env is None:
            raise ValidationError(
                f"Missing credit pricing environment for {self.reference_entity}"
            )

        recovery = getattr(self.product, "recovery_rate", 0.4)
        spread_bump = 1e-4 / max(1e-6, 1.0 - recovery)  # hazard move for 1bp spread
        up = self.engine.price(self.product, env.with_hazard_shift(+spread_bump))
        down = self.engine.price(self.product, env.with_hazard_shift(-spread_bump))
        cs01 = (up - down) / 2.0 * self.quantity

        result = SensitivityCollection()
        if not is_zero(cs01):
            result.add(
                CreditDeltaSensitivity(
                    trade_id=self.position_id,
                    amount=cs01,
                    amount_currency=config.calculation_currency.upper(),
                    issuer=self.reference_entity,
                    bucket_number=self._bucket(),
                    tenor=self.product.maturity,
                    is_qualifying=self.is_qualifying,
                    payment_currency=self.payment_currency.upper(),
                )
            )
        return result

"""
FX quanto range accrual product.

Identical accrual mechanics to :class:`FxRangeAccrualOption`, but the coupon is
fixed in the domestic (quote) currency and paid in a third *settlement*
currency at a contractually fixed ``quanto_fx_rate`` (settlement units per
domestic unit). Pricing therefore lives under the settlement-currency measure:
the in-range probabilities use the quanto-adjusted forward and the coupon is
discounted on the settlement curve (see
:class:`FxQuantoRangeAccrualAnalyticalEngine`).
"""

from datetime import datetime
from typing import List, Optional

from quantark.util.exceptions import ValidationError

from ..currency_pair import CurrencyPair
from .fx_range_accrual_config import (
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
)
from .fx_range_accrual_option import FxRangeAccrualOption


class FxQuantoRangeAccrualOption(FxRangeAccrualOption):
    """
    FX range accrual settled in a third currency at a fixed quanto rate.

    Attributes:
        quanto_fx_rate: Fixed conversion rate from the domestic (quote)
            currency to the settlement currency.
        settlement_ccy: Settlement currency code (informational).

    All other attributes match :class:`FxRangeAccrualOption`.
    """

    def __init__(
        self,
        notional: float,
        range_config: FxRangeAccrualConfig,
        quanto_fx_rate: float,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
        observation_records: Optional[List[FxRangeAccrualObservationRecord]] = None,
        observation_times: Optional[List[float]] = None,
        num_observations: Optional[int] = None,
        settlement_ccy: str = "SET",
    ):
        self.quanto_fx_rate = quanto_fx_rate
        self.settlement_ccy = settlement_ccy.upper()
        super().__init__(
            notional=notional,
            range_config=range_config,
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
            observation_records=observation_records,
            observation_times=observation_times,
            num_observations=num_observations,
        )

    def validate(self) -> None:
        """Validate the quanto range accrual specification."""
        super().validate()
        if self.quanto_fx_rate <= 0:
            raise ValidationError(
                f"quanto_fx_rate must be positive, got {self.quanto_fx_rate}"
            )

    def get_payoff(
        self,
        spot: float,
        in_range_weights: Optional[float] = None,
        total_weights: Optional[float] = None,
        fx_env=None,
    ) -> float:
        """Terminal coupon in the settlement currency (converted at the quanto rate)."""
        domestic_coupon = super().get_payoff(
            spot,
            in_range_weights=in_range_weights,
            total_weights=total_weights,
            fx_env=fx_env,
        )
        return domestic_coupon * self.quanto_fx_rate

    def __repr__(self) -> str:
        cfg = self.range_config
        return (
            f"FxQuantoRangeAccrualOption({self.currency_pair}, "
            f"range=[{cfg.lower_barrier}, {cfg.upper_barrier}], "
            f"rate={cfg.accrual_rate:.2%}, notional={self.notional:,.0f}, "
            f"X_q={self.quanto_fx_rate:.6f} -> {self.settlement_ccy})"
        )

"""
FX foreign-currency-coupon range accrual product.

Identical accrual mechanics to :class:`FxRangeAccrualOption`, but the coupon is
paid in the **foreign (base) currency** rather than the domestic (quote)
currency. Receiving one foreign unit at payment is worth ``X_T`` domestic units
at payment, so each in-range fixing is a cash-or-nothing digital *under the
foreign measure* — its in-the-money probability is ``N(d1)`` (not the domestic
``N(d2)``) and the coupon is valued as ``S_eff * DF_foreign`` per unit. The
value is still reported in the domestic (quote) currency. See
:class:`FxForeignRangeAccrualAnalyticalEngine`.

``notional`` here is the coupon notional in the **foreign (base)** currency.
"""

from datetime import datetime
from typing import List, Optional

from ..currency_pair import CurrencyPair
from .fx_range_accrual_config import (
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
)
from .fx_range_accrual_option import FxRangeAccrualOption


class FxForeignRangeAccrualOption(FxRangeAccrualOption):
    """
    FX range accrual paying a foreign (base) currency coupon on in-range fixings.

    Same schedule, barriers, weights, reverse mode and pay-type semantics as
    :class:`FxRangeAccrualOption`; only the coupon currency (and therefore the
    pricing measure) differs. ``notional`` is denominated in the foreign (base)
    currency. Price with :class:`FxForeignRangeAccrualAnalyticalEngine`.
    """

    def __init__(
        self,
        notional: float,
        range_config: FxRangeAccrualConfig,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
        observation_records: Optional[List[FxRangeAccrualObservationRecord]] = None,
        observation_times: Optional[List[float]] = None,
        num_observations: Optional[int] = None,
    ):
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

    def __repr__(self) -> str:
        cfg = self.range_config
        base = self.currency_pair.base_ccy if self.currency_pair else "FOR"
        return (
            f"FxForeignRangeAccrualOption({self.currency_pair}, "
            f"range=[{cfg.lower_barrier}, {cfg.upper_barrier}], "
            f"rate={cfg.accrual_rate:.2%}, notional={self.notional:,.0f} {base})"
        )

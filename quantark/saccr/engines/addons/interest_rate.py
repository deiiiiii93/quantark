"""Interest Rate add-on calculator for SA-CCR.

Reference: Basel Committee SA-CCR document, paragraphs 166-169.

Methodology: group trades by currency (hedging set); within each currency assign
trades to one of three maturity buckets (E < 1y, 1-5y, >= 5y); aggregate the bucket
effective notionals with partial offset; the hedging-set add-on is SF x effective
notional; sum across currencies.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from quantark.saccr.engines.addons.base import BaseAddOn
from quantark.saccr.engines.maths import supervisory_duration, supervisory_delta
from quantark.saccr.models.enums import AssetClass
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.parameters.supervisory import SupervisoryParameters
from quantark.util.numerical import safe_sqrt


class InterestRateAddOn(BaseAddOn):
    """Interest Rate add-on (paragraphs 166-169)."""

    BUCKET_1_MAX = 1.0  # E < 1 year
    BUCKET_2_MAX = 5.0  # 1 <= E < 5 years

    def calculate_with_breakdown(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, Dict[str, float]]:
        ir_trades = [t for t in trades if t.asset_class == AssetClass.INTEREST_RATE]
        breakdown: Dict[str, float] = {}
        if not ir_trades:
            return 0.0, breakdown

        trades_by_currency: Dict[str, List[SACCRTrade]] = defaultdict(list)
        for trade in ir_trades:
            trades_by_currency[trade.currency].append(trade)

        total = 0.0
        for currency, currency_trades in trades_by_currency.items():
            addon = self._calculate_currency_addon(currency_trades, netting_set)
            breakdown[f"IR:{currency}"] = addon
            total += addon
        return total, breakdown

    def _calculate_currency_addon(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> float:
        """Add-on for a single currency hedging set (paragraphs 168-169)."""
        d1, d2, d3 = self._bucket_effective_notionals(trades, netting_set)

        # EffectiveNotional = sqrt(D1^2 + D2^2 + D3^2
        #                          + 1.4 D1 D2 + 1.4 D2 D3 + 0.6 D1 D3)  (paragraph 169)
        effective_notional = safe_sqrt(
            d1 ** 2 + d2 ** 2 + d3 ** 2
            + 1.4 * d1 * d2 + 1.4 * d2 * d3 + 0.6 * d1 * d3
        )
        return SupervisoryParameters.SF_INTEREST_RATE * float(effective_notional)

    def _bucket_effective_notionals(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, float, float]:
        """Effective notional per maturity bucket (paragraph 168).

        ``D_k = sum(delta_i * AdjustedNotional_i * MF_i)`` for trades in bucket k.
        """
        d1 = d2 = d3 = 0.0
        for trade in trades:
            adjusted_notional = trade.notional * supervisory_duration(
                trade.start_date, trade.end_date)
            mf = self.get_maturity_factor(trade, netting_set)
            delta = self._delta(trade)
            contribution = delta * adjusted_notional * mf

            bucket = self._maturity_bucket(trade.end_date)
            if bucket == 1:
                d1 += contribution
            elif bucket == 2:
                d2 += contribution
            else:
                d3 += contribution
        return d1, d2, d3

    @staticmethod
    def _delta(trade: SACCRTrade) -> float:
        return supervisory_delta(
            position=trade.position,
            is_option=trade.is_option,
            option_type=trade.option_type,
            underlying_price=trade.underlying_price,
            strike_price=trade.strike_price,
            exercise_date=trade.exercise_date,
            supervisory_volatility=(
                SupervisoryParameters.get_option_volatility(AssetClass.INTEREST_RATE)
                if trade.is_option else None
            ),
            is_cdo_tranche=trade.is_cdo_tranche,
            attachment_point=trade.attachment_point,
            detachment_point=trade.detachment_point,
        )

    def _maturity_bucket(self, end_date: float) -> int:
        """Maturity bucket from the end date (paragraph 166).

        Bucket 1: E < 1y; bucket 2: 1y <= E <= 5y; bucket 3: E > 5y. Exactly 5
        years belongs to bucket 2 ("between one and five years").
        """
        if end_date < self.BUCKET_1_MAX:
            return 1
        if end_date <= self.BUCKET_2_MAX:
            return 2
        return 3

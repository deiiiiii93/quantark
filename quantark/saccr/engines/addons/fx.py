"""Foreign Exchange add-on calculator for SA-CCR.

Reference: Basel Committee SA-CCR document, paragraphs 170-171.

Methodology: group trades by currency pair (hedging set); full offset within a
pair; AddOn = SF x |effective notional|; sum across pairs (no cross-pair offset).

Hedging-set key normalization (preserved from the validated source): ``EUR/USD``
and ``USD/EUR`` reference the same FX rate, so they belong to the SAME hedging set
and offset. The pair is normalized by sorting the two legs alphabetically; the
position direction is carried by the supervisory delta sign. Inputs are still
required to be in canonical ``"CCY1/CCY2"`` form (enforced by ``SACCRTrade``).
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from quantark.saccr.engines.addons.base import BaseAddOn
from quantark.saccr.engines.maths import supervisory_delta
from quantark.saccr.models.enums import AssetClass
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.parameters.supervisory import SupervisoryParameters


class FXAddOn(BaseAddOn):
    """Foreign Exchange add-on (paragraphs 170-171)."""

    def calculate_with_breakdown(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, Dict[str, float]]:
        fx_trades = [t for t in trades if t.asset_class == AssetClass.FX]
        breakdown: Dict[str, float] = {}
        if not fx_trades:
            return 0.0, breakdown

        trades_by_pair: Dict[str, List[SACCRTrade]] = defaultdict(list)
        for trade in fx_trades:
            pair = self._normalize_currency_pair(trade.currency_pair)
            trades_by_pair[pair].append(trade)

        total = 0.0
        for pair, pair_trades in trades_by_pair.items():
            addon = self._calculate_pair_addon(pair_trades, netting_set)
            breakdown[f"FX:{pair}"] = addon
            total += addon
        return total, breakdown

    def _calculate_pair_addon(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> float:
        """Add-on for a single currency-pair hedging set (paragraph 171)."""
        effective_notional = 0.0
        for trade in trades:
            adjusted_notional = trade.notional  # already in domestic currency (para 171)
            mf = self.get_maturity_factor(trade, netting_set)
            delta = self._delta(trade)
            effective_notional += delta * adjusted_notional * mf
        return SupervisoryParameters.SF_FX * abs(effective_notional)

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
                SupervisoryParameters.get_option_volatility(AssetClass.FX)
                if trade.is_option else None
            ),
        )

    @staticmethod
    def _normalize_currency_pair(pair: str) -> str:
        """Normalize a canonical ``CCY1/CCY2`` pair to a sorted hedging-set key."""
        legs = pair.split("/")
        if len(legs) == 2:
            legs.sort()
            return f"{legs[0]}/{legs[1]}"
        return pair

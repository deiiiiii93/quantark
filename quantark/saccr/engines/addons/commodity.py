"""Commodity derivatives add-on calculator for SA-CCR.

Reference: Basel Committee SA-CCR document, paragraphs 178-182.

Methodology: group trades into the 4 hedging sets (Energy, Metals, Agricultural,
Other); within a hedging set, full offset within each commodity type to a type-level
add-on, then single-factor aggregation across types (rho = 40%); no offset across
hedging sets (simple sum). Add-on is reported per hedging set.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from quantark.saccr.engines.addons.base import BaseAddOn
from quantark.saccr.engines.maths import supervisory_delta
from quantark.saccr.models.enums import (
    AssetClass,
    CommodityHedgingSet,
    CommodityType,
    COMMODITY_TYPE_TO_HEDGING_SET,
)
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.parameters.supervisory import SupervisoryParameters
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_sqrt


class CommodityAddOn(BaseAddOn):
    """Commodity derivatives add-on (paragraphs 178-182)."""

    def calculate_with_breakdown(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, Dict[str, float]]:
        commodity_trades = [t for t in trades if t.asset_class == AssetClass.COMMODITY]
        breakdown: Dict[str, float] = {}
        if not commodity_trades:
            return 0.0, breakdown

        trades_by_hs: Dict[CommodityHedgingSet, List[SACCRTrade]] = defaultdict(list)
        for trade in commodity_trades:
            trades_by_hs[self._hedging_set(trade)].append(trade)

        total = 0.0
        for hedging_set, hs_trades in trades_by_hs.items():
            addon = self._hedging_set_addon(hs_trades, netting_set)
            breakdown[f"COMMODITY:{hedging_set.name}"] = addon
            total += addon
        return total, breakdown

    def _hedging_set_addon(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> float:
        """Single-factor add-on for one hedging set (paragraph 179)."""
        type_addons = self._type_addons(trades, netting_set)
        if not type_addons:
            return 0.0
        rho = SupervisoryParameters.CORRELATION_COMMODITY
        sum_addons = sum(type_addons)
        sum_addons_sq = sum(a ** 2 for a in type_addons)
        systematic = rho * sum_addons
        idiosyncratic = (1 - rho ** 2) * sum_addons_sq
        return float(safe_sqrt(systematic ** 2 + idiosyncratic))

    def _type_addons(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> List[float]:
        """Type-level add-ons within a hedging set (paragraph 179)."""
        trades_by_type: Dict[CommodityType, List[SACCRTrade]] = defaultdict(list)
        for trade in trades:
            trades_by_type[trade.commodity_type].append(trade)

        type_addons: List[float] = []
        for commodity_type, type_trades in trades_by_type.items():
            effective_notional = 0.0
            for trade in type_trades:
                adjusted_notional = trade.notional
                mf = self.get_maturity_factor(trade, netting_set)
                delta = self._delta(trade)
                effective_notional += delta * adjusted_notional * mf
            sf = SupervisoryParameters.get_supervisory_factor(
                AssetClass.COMMODITY, commodity_type=commodity_type)
            type_addons.append(sf * effective_notional)
        return type_addons

    @staticmethod
    def _hedging_set(trade: SACCRTrade) -> CommodityHedgingSet:
        """Resolve the hedging set (explicit override, else mapped; else raise)."""
        if trade.commodity_hedging_set is not None:
            return trade.commodity_hedging_set
        if trade.commodity_type not in COMMODITY_TYPE_TO_HEDGING_SET:
            raise ValidationError(
                f"Unknown/missing commodity type: {trade.commodity_type!r}")
        return COMMODITY_TYPE_TO_HEDGING_SET[trade.commodity_type]

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
                SupervisoryParameters.get_option_volatility(
                    AssetClass.COMMODITY, commodity_type=trade.commodity_type)
                if trade.is_option else None
            ),
        )

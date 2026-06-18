"""Equity derivatives add-on calculator for SA-CCR.

Reference: Basel Committee SA-CCR document, paragraphs 176-178.

Single-factor model (same shape as Credit). SF = 32% single name / 20% index;
correlation = 50% single / 80% index. Adjusted notional = current price x quantity
(carried in ``trade.notional``).
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from quantark.saccr.engines.addons.base import BaseAddOn
from quantark.saccr.engines.maths import supervisory_delta
from quantark.saccr.models.enums import AssetClass
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.parameters.supervisory import SupervisoryParameters
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_sqrt


class EquityAddOn(BaseAddOn):
    """Equity derivatives add-on (paragraphs 176-178)."""

    HEDGING_SET_LABEL = "EQUITY"

    def calculate_with_breakdown(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, Dict[str, float]]:
        equity_trades = [t for t in trades if t.asset_class == AssetClass.EQUITY]
        if not equity_trades:
            return 0.0, {}
        entity_addons = self._entity_addons(equity_trades, netting_set)
        total = self._aggregate_single_factor(entity_addons)
        return total, {self.HEDGING_SET_LABEL: total}

    def _entity_addons(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> List[Tuple[float, float]]:
        """Return ``[(entity_addon, correlation), ...]`` (paragraph 176)."""
        trades_by_entity: Dict[str, List[SACCRTrade]] = defaultdict(list)
        for trade in trades:
            trades_by_entity[trade.reference_entity].append(trade)

        entity_addons: List[Tuple[float, float]] = []
        for entity, entity_trades in trades_by_entity.items():
            rep = entity_trades[0]
            # One SF/correlation per reference entity; reject inconsistent index flags.
            for t in entity_trades:
                if t.is_index != rep.is_index:
                    raise ValidationError(
                        f"Inconsistent equity metadata for reference_entity {entity!r}: "
                        "all trades must share is_index")
            effective_notional = 0.0
            for trade in entity_trades:
                adjusted_notional = trade.notional
                mf = self.get_maturity_factor(trade, netting_set)
                delta = self._delta(trade)
                effective_notional += delta * adjusted_notional * mf

            sf = SupervisoryParameters.get_supervisory_factor(
                AssetClass.EQUITY, is_index=rep.is_index)
            correlation = SupervisoryParameters.get_correlation(
                AssetClass.EQUITY, is_index=rep.is_index)
            entity_addons.append((sf * effective_notional, correlation))
        return entity_addons

    @staticmethod
    def _aggregate_single_factor(entity_addons: List[Tuple[float, float]]) -> float:
        """Single-factor aggregation (paragraph 178)."""
        systematic = 0.0
        idiosyncratic = 0.0
        for addon, rho in entity_addons:
            systematic += rho * addon
            idiosyncratic += (1 - rho ** 2) * addon ** 2
        return float(safe_sqrt(systematic ** 2 + idiosyncratic))

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
                    AssetClass.EQUITY, is_index=trade.is_index)
                if trade.is_option else None
            ),
        )

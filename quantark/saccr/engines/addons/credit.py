"""Credit derivatives add-on calculator for SA-CCR.

Reference: Basel Committee SA-CCR document, paragraphs 172-175.

Single-factor model: full offset within a reference entity to an entity-level
add-on, then aggregate across entities as
``sqrt((sum rho_k AddOn_k)^2 + sum (1 - rho_k^2) AddOn_k^2)``.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from quantark.saccr.engines.addons.base import BaseAddOn
from quantark.saccr.engines.maths import supervisory_duration, supervisory_delta
from quantark.saccr.models.enums import AssetClass
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.trade import SACCRTrade
from quantark.saccr.parameters.supervisory import SupervisoryParameters
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_sqrt


class CreditAddOn(BaseAddOn):
    """Credit derivatives add-on (paragraphs 172-175)."""

    HEDGING_SET_LABEL = "CREDIT"

    def calculate_with_breakdown(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, Dict[str, float]]:
        credit_trades = [t for t in trades if t.asset_class == AssetClass.CREDIT]
        if not credit_trades:
            return 0.0, {}
        entity_addons = self._entity_addons(credit_trades, netting_set)
        total = self._aggregate_single_factor(entity_addons)
        return total, {self.HEDGING_SET_LABEL: total}

    def _entity_addons(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> List[Tuple[float, float]]:
        """Return ``[(entity_addon, correlation), ...]`` (paragraph 172)."""
        trades_by_entity: Dict[str, List[SACCRTrade]] = defaultdict(list)
        for trade in trades:
            trades_by_entity[trade.reference_entity].append(trade)

        entity_addons: List[Tuple[float, float]] = []
        for entity, entity_trades in trades_by_entity.items():
            rep = entity_trades[0]
            # The single-factor model uses one SF/correlation per reference entity;
            # reject inconsistent discriminators rather than silently using the first.
            for t in entity_trades:
                if (t.is_index != rep.is_index
                        or t.credit_rating != rep.credit_rating
                        or t.index_grade != rep.index_grade):
                    raise ValidationError(
                        f"Inconsistent credit metadata for reference_entity {entity!r}: "
                        "all trades must share is_index, credit_rating and index_grade")
            effective_notional = 0.0
            for trade in entity_trades:
                adjusted_notional = trade.notional * supervisory_duration(
                    trade.start_date, trade.end_date)
                mf = self.get_maturity_factor(trade, netting_set)
                delta = self._delta(trade)
                effective_notional += delta * adjusted_notional * mf

            sf = SupervisoryParameters.get_supervisory_factor(
                AssetClass.CREDIT,
                credit_rating=rep.credit_rating,
                index_grade=rep.index_grade,
                is_index=rep.is_index,
            )
            correlation = SupervisoryParameters.get_correlation(
                AssetClass.CREDIT, is_index=rep.is_index)
            entity_addons.append((sf * effective_notional, correlation))
        return entity_addons

    @staticmethod
    def _aggregate_single_factor(entity_addons: List[Tuple[float, float]]) -> float:
        """Single-factor aggregation (paragraph 173)."""
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
                    AssetClass.CREDIT, is_index=trade.is_index)
                if trade.is_option else None
            ),
            is_cdo_tranche=trade.is_cdo_tranche,
            attachment_point=trade.attachment_point,
            detachment_point=trade.detachment_point,
        )

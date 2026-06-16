"""Main SA-CCR Calculator.

Reference: Basel Committee SA-CCR document, paragraphs 128-129.

EAD = alpha x (RC + PFE), where alpha = 1.4 (paragraph 128),
RC is the replacement cost (paragraphs 130-145) and
PFE = multiplier x AddOn_aggregate (paragraphs 146-187).
For margined netting sets, EAD is capped at the EAD computed as if the netting set
were unmargined (paragraph 129).
"""

from typing import Dict, List, Tuple

from quantark.saccr.engines.addons.base import BaseAddOn
from quantark.saccr.engines.addons.commodity import CommodityAddOn
from quantark.saccr.engines.addons.credit import CreditAddOn
from quantark.saccr.engines.addons.equity import EquityAddOn
from quantark.saccr.engines.addons.fx import FXAddOn
from quantark.saccr.engines.addons.interest_rate import InterestRateAddOn
from quantark.saccr.engines.pfe import calculate_multiplier, calculate_pfe
from quantark.saccr.engines.replacement_cost import (
    calculate_rc_margined,
    calculate_rc_unmargined,
)
from quantark.saccr.models.enums import AssetClass
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.parameters.supervisory import SupervisoryParameters
from quantark.saccr.results.result import SACCRResult


class SACCRCalculator:
    """Standardised Approach for Counterparty Credit Risk calculator.

    Usage::

        result = SACCRCalculator().calculate(netting_set)
        print(result.ead)
    """

    def __init__(self):
        self.addon_calculators: Dict[AssetClass, BaseAddOn] = {
            AssetClass.INTEREST_RATE: InterestRateAddOn(),
            AssetClass.FX: FXAddOn(),
            AssetClass.CREDIT: CreditAddOn(),
            AssetClass.EQUITY: EquityAddOn(),
            AssetClass.COMMODITY: CommodityAddOn(),
        }

    def calculate(self, netting_set: SACCRNettingSet) -> SACCRResult:
        """Calculate SA-CCR EAD for a netting set (paragraph 128)."""
        v = netting_set.market_value
        c = netting_set.collateral
        alpha = SupervisoryParameters.ALPHA

        # --- Primary path (margined for margined sets, unmargined otherwise) ---
        if netting_set.is_margined:
            rc = calculate_rc_margined(
                v=v, c=c,
                th=netting_set.threshold,
                mta=netting_set.minimum_transfer_amount,
                nica=netting_set.nica,
            )
        else:
            rc = calculate_rc_unmargined(v=v, c=c)

        addon_aggregate, by_asset_class, by_hedging_set = self._addons(netting_set)
        multiplier = calculate_multiplier(v=v, c=c, addon_aggregate=addon_aggregate)
        pfe = calculate_pfe(addon_aggregate=addon_aggregate, multiplier=multiplier)
        ead_primary = alpha * (rc + pfe)

        # --- Margined EAD cap (paragraph 129) ---
        if netting_set.is_margined:
            ead_unmargined_cap = self._unmargined_ead(netting_set, v, c, alpha)
            ead = min(ead_primary, ead_unmargined_cap)
            ead_capped = ead_unmargined_cap < ead_primary
        else:
            ead = ead_primary
            ead_capped = False

        return SACCRResult(
            ead=ead,
            ead_uncapped=ead_primary,
            ead_capped=ead_capped,
            rc=rc,
            pfe=pfe,
            multiplier=multiplier,
            alpha=alpha,
            addon_aggregate=addon_aggregate,
            addon_by_asset_class=by_asset_class,
            addon_by_hedging_set=by_hedging_set,
            v=v,
            c=c,
            nica=netting_set.nica,
            is_margined=netting_set.is_margined,
        )

    def _addons(
        self,
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Aggregate add-ons across asset classes (no diversification, paragraph 150).

        Returns ``(aggregate, by_asset_class, by_hedging_set)``.
        """
        trades_by_class: Dict[AssetClass, List] = {}
        for trade in netting_set.trades:
            trades_by_class.setdefault(trade.asset_class, []).append(trade)

        aggregate = 0.0
        by_asset_class: Dict[str, float] = {}
        by_hedging_set: Dict[str, float] = {}
        for asset_class, calculator in self.addon_calculators.items():
            trades = trades_by_class.get(asset_class, [])
            if not trades:
                continue
            total, breakdown = calculator.calculate_with_breakdown(trades, netting_set)
            by_asset_class[asset_class.name] = total
            by_hedging_set.update(breakdown)
            aggregate += total
        return aggregate, by_asset_class, by_hedging_set

    def _unmargined_ead(
        self,
        netting_set: SACCRNettingSet,
        v: float,
        c: float,
        alpha: float,
    ) -> float:
        """EAD computed as if the netting set were unmargined (paragraph 129).

        Same V and C; RC = max(V - C, 0); add-ons recomputed with the unmargined
        (trade-maturity) factor via an unmargined twin of the netting set.
        """
        rc = calculate_rc_unmargined(v=v, c=c)
        unmargined_twin = SACCRNettingSet(
            netting_set_id=netting_set.netting_set_id + "_unmargined",
            trades=netting_set.trades,
            is_margined=False,
            net_collateral=c,
        )
        addon_aggregate, _, _ = self._addons(unmargined_twin)
        multiplier = calculate_multiplier(v=v, c=c, addon_aggregate=addon_aggregate)
        pfe = calculate_pfe(addon_aggregate=addon_aggregate, multiplier=multiplier)
        return alpha * (rc + pfe)

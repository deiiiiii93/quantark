"""Strict sensitivity engine backed by position-level SIMM providers."""

from typing import Any, Dict, List

from quantark.simm.engines.base import BaseSensitivityEngine, SIMMSensitivityProvider
from quantark.simm.sensitivity import AnySensitivity, SensitivityCollection
from quantark.simm.taxonomy import MarginType, RiskClass
from quantark.util.exceptions import ValidationError


class ProviderSensitivityEngine(BaseSensitivityEngine):
    """Routes compliant provider-supplied sensitivities for one risk class."""

    def __init__(self, config, risk_class: RiskClass):
        super().__init__(config)
        self._risk_class = risk_class

    @property
    def risk_class(self) -> RiskClass:
        return self._risk_class

    def _provided(
        self, positions: List[Any], market_data: Dict[str, Any]
    ) -> SensitivityCollection:
        result = SensitivityCollection()
        for position in positions:
            if not isinstance(position, SIMMSensitivityProvider):
                raise ValidationError(
                    f"Position {getattr(position, 'position_id', position)!r} does "
                    "not implement SIMMSensitivityProvider"
                )
            supplied = position.get_simm_sensitivities(self.config, market_data)
            result.add_many(supplied.by_risk_class(self.risk_class))
        return result

    def _by_margin(
        self,
        positions: List[Any],
        market_data: Dict[str, Any],
        margin_type: MarginType,
    ) -> List[AnySensitivity]:
        return self._provided(positions, market_data).by_margin_type(margin_type)

    def calculate_delta_sensitivities(self, positions, pricing_environments):
        return self._by_margin(positions, pricing_environments, MarginType.DELTA)

    def calculate_vega_sensitivities(self, positions, pricing_environments):
        return self._by_margin(positions, pricing_environments, MarginType.VEGA)

    def calculate_curvature_sensitivities(self, positions, pricing_environments):
        return self._by_margin(positions, pricing_environments, MarginType.CURVATURE)

    def calculate_base_corr_sensitivities(self, positions, pricing_environments):
        return self._by_margin(positions, pricing_environments, MarginType.BASE_CORR)

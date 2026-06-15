"""Trade-level structured volatility-model risk for FX engines."""

from __future__ import annotations

from typing import Optional

from quantark.asset.vol_model_risk import BaseVolModelRiskCalculator
from quantark.volmodels.risk import HestonCalibrationSpec, SlvCalibrationSpec


class FxVolModelRiskCalculator(BaseVolModelRiskCalculator):
    """Calculate FX model-factor risk and fully recalibrated market-IV vega."""

    def __init__(
        self,
        heston_calibration_spec: Optional[HestonCalibrationSpec] = None,
        slv_calibration_spec: Optional[SlvCalibrationSpec] = None,
    ) -> None:
        super().__init__(
            is_fx=True,
            heston_calibration_spec=heston_calibration_spec,
            slv_calibration_spec=slv_calibration_spec,
        )

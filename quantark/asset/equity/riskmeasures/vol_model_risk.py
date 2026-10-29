"""Trade-level structured volatility-model risk for equity engines."""

from __future__ import annotations

from typing import Optional

from quantark.asset.vol_model_risk import BaseVolModelRiskCalculator
from quantark.volmodels.localvol import AndreasenHugeConfig, LocalVolConstruction
from quantark.volmodels.risk import HestonCalibrationSpec, SlvCalibrationSpec


class VolModelRiskCalculator(BaseVolModelRiskCalculator):
    """Calculate model-factor risk and fully recalibrated market-IV vega."""

    def __init__(
        self,
        heston_calibration_spec: Optional[HestonCalibrationSpec] = None,
        slv_calibration_spec: Optional[SlvCalibrationSpec] = None,
        *,
        local_vol_construction: LocalVolConstruction = LocalVolConstruction.DUPIRE,
        andreasen_huge_config: Optional[AndreasenHugeConfig] = None,
    ) -> None:
        super().__init__(
            is_fx=False,
            heston_calibration_spec=heston_calibration_spec,
            slv_calibration_spec=slv_calibration_spec,
            local_vol_construction=local_vol_construction,
            andreasen_huge_config=andreasen_huge_config,
        )

"""Trade-level structured volatility-model risk for FX engines."""

from __future__ import annotations

from typing import Optional

from quantark.asset.vol_model_risk import BaseVolModelRiskCalculator
from quantark.volmodels.localvol import AndreasenHugeConfig, LocalVolConstruction
from quantark.volmodels.risk import HestonCalibrationSpec, SlvCalibrationSpec


class FxVolModelRiskCalculator(BaseVolModelRiskCalculator):
    """Calculate FX model-factor risk and fully recalibrated market-IV vega.

    Args:
        heston_calibration_spec: Heston recalibration settings for market vega.
        slv_calibration_spec: SLV leverage-calibration settings.
        local_vol_construction: builder for every local-vol surface this calculator constructs
            (market vega, SLV market vega, model risk on engines without a prebuilt surface).
            ``ANDREASEN_HUGE`` (opt-in) reprices the quotes; with it, model-risk ``base_price`` is
            the AH price even though the engine's own ``price()`` still builds Dupire.
        andreasen_huge_config: ``AndreasenHugeConfig``; only with ``ANDREASEN_HUGE``.
    """

    def __init__(
        self,
        heston_calibration_spec: Optional[HestonCalibrationSpec] = None,
        slv_calibration_spec: Optional[SlvCalibrationSpec] = None,
        *,
        local_vol_construction: LocalVolConstruction = LocalVolConstruction.DUPIRE,
        andreasen_huge_config: Optional[AndreasenHugeConfig] = None,
    ) -> None:
        super().__init__(
            is_fx=True,
            heston_calibration_spec=heston_calibration_spec,
            slv_calibration_spec=slv_calibration_spec,
            local_vol_construction=local_vol_construction,
            andreasen_huge_config=andreasen_huge_config,
        )

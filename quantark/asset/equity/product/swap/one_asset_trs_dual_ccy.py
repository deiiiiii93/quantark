"""Single-asset Total Return Swap with a dual-currency settlement overlay.

The underlying TRS is priced in the asset currency; this subclass converts the
present value and margin figures into the settlement currency using either a
fixed FX rate or an observed FX fixing series.
"""

from typing import Dict, Optional

import pandas as pd

from quantark.util.exceptions import ValidationError
from quantark.asset.equity.product.swap.trs_params import TRSParams
from quantark.asset.equity.product.swap.one_asset_trs import OneAssetTotalReturnSwap

_VALID_FX_OBS_TYPES = ("fixed", "spot", "close", "bfix")


class OneAssetTotalReturnSwapDualCcy(OneAssetTotalReturnSwap):
    """A single-asset TRS reported in a settlement currency distinct from the asset."""

    def __init__(
        self,
        params: TRSParams,
        asset_ccy: str = "CNY",
        settle_ccy: str = "CNY",
        fx_rate_obs_type: str = "fixed",
        fx_rate: float = 0.0,
        fx_rate_obs_records: Optional[Dict[str, float]] = None,
    ):
        super().__init__(params)

        if fx_rate_obs_type.lower() not in _VALID_FX_OBS_TYPES:
            raise ValidationError(
                f"fx_rate_obs_type must be one of {_VALID_FX_OBS_TYPES}, "
                f"got {fx_rate_obs_type!r}"
            )

        self.asset_ccy = asset_ccy
        self.settle_ccy = settle_ccy
        self.fx_rate_obs_type = fx_rate_obs_type.lower()
        self.fx_rate = 1 if asset_ccy == settle_ccy else fx_rate
        self.fx_rate_obs_records = fx_rate_obs_records

        if self.fx_rate_obs_type != "fixed" and self.fx_rate_obs_records is None:
            raise ValidationError(
                "fx_rate_obs_records must be provided for a non-fixed "
                f"fx_rate_obs_type ({self.fx_rate_obs_type!r})"
            )

    def price(self, precision: int = 2) -> pd.DataFrame:
        """Price the TRS and add settlement-currency columns."""
        df = super().price(precision=4)

        fx_col = f"fx_rate_{self.settle_ccy}/{self.asset_ccy}"
        if self.fx_rate_obs_type == "fixed" or self.asset_ccy == self.settle_ccy:
            df[fx_col] = self.fx_rate
        else:
            df[fx_col] = df["period_start"].map(
                lambda d: self.fx_rate_obs_records[d]
            )

        fx = pd.to_numeric(df[fx_col], errors="coerce")
        for base in ("present_value", "outstanding_margin", "margin_mtm"):
            values = pd.to_numeric(df[base], errors="coerce")
            df[f"{base}_{self.settle_ccy}"] = (values / fx).round(precision)

        return df

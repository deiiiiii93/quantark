"""Configuration for FX risk factors used by the FX VaR engines."""

from dataclasses import dataclass


@dataclass
class FXRiskFactorConfig:
    """
    Which FX risk factors to include in VaR.

    FX carries two rate factors (domestic and foreign) in addition to spot and
    vol, reflecting the Garman-Kohlhagen carry structure.
    """

    include_spot: bool = True
    include_vol: bool = True
    include_domestic_rate: bool = True
    include_foreign_rate: bool = True

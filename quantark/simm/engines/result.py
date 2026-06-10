"""
SIMM calculation result classes.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from decimal import Decimal

from simm.sensitivity import SensitivityCollection


@dataclass
class SIMMResult:
    """
    Result of a SIMM calculation.

    This class encapsulates the results of a SIMM margin calculation, including
    breakdowns by risk class and margin type.
    """

    # Total SIMM margin
    total_margin: float

    # Breakdown by risk class
    delta_margin: float
    vega_margin: float
    curvature_margin: float
    base_corr_margin: float

    # Additional breakdown by risk class
    ir_delta_margin: float = 0.0
    ir_vega_margin: float = 0.0
    ir_curvature_margin: float = 0.0

    equity_delta_margin: float = 0.0
    equity_vega_margin: float = 0.0
    equity_curvature_margin: float = 0.0

    credit_q_delta_margin: float = 0.0
    credit_q_vega_margin: float = 0.0
    credit_q_curvature_margin: float = 0.0

    credit_nq_delta_margin: float = 0.0
    credit_nq_vega_margin: float = 0.0
    credit_nq_curvature_margin: float = 0.0

    commodity_delta_margin: float = 0.0
    commodity_vega_margin: float = 0.0
    commodity_curvature_margin: float = 0.0

    fx_delta_margin: float = 0.0
    fx_vega_margin: float = 0.0
    fx_curvature_margin: float = 0.0

    # Concentration add-ons
    concentration_add_on: float = 0.0

    # Calculation metadata
    calculation_currency: Optional[str] = None
    calculation_timestamp: Optional[str] = None
    simm_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "total_margin": self.total_margin,
            "delta_margin": self.delta_margin,
            "vega_margin": self.vega_margin,
            "curvature_margin": self.curvature_margin,
            "base_corr_margin": self.base_corr_margin,
            "ir": {
                "delta": self.ir_delta_margin,
                "vega": self.ir_vega_margin,
                "curvature": self.ir_curvature_margin,
            },
            "equity": {
                "delta": self.equity_delta_margin,
                "vega": self.equity_vega_margin,
                "curvature": self.equity_curvature_margin,
            },
            "credit_q": {
                "delta": self.credit_q_delta_margin,
                "vega": self.credit_q_vega_margin,
                "curvature": self.credit_q_curvature_margin,
            },
            "credit_nq": {
                "delta": self.credit_nq_delta_margin,
                "vega": self.credit_nq_vega_margin,
                "curvature": self.credit_nq_curvature_margin,
            },
            "commodity": {
                "delta": self.commodity_delta_margin,
                "vega": self.commodity_vega_margin,
                "curvature": self.commodity_curvature_margin,
            },
            "fx": {
                "delta": self.fx_delta_margin,
                "vega": self.fx_vega_margin,
                "curvature": self.fx_curvature_margin,
            },
            "concentration_add_on": self.concentration_add_on,
            "calculation_currency": self.calculation_currency,
            "calculation_timestamp": self.calculation_timestamp,
            "simm_version": self.simm_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SIMMResult":
        """Create result from dictionary."""
        return cls(
            total_margin=data["total_margin"],
            delta_margin=data["delta_margin"],
            vega_margin=data["vega_margin"],
            curvature_margin=data["curvature_margin"],
            base_corr_margin=data.get("base_corr_margin", 0.0),
            ir_delta_margin=data.get("ir", {}).get("delta", 0.0),
            ir_vega_margin=data.get("ir", {}).get("vega", 0.0),
            ir_curvature_margin=data.get("ir", {}).get("curvature", 0.0),
            equity_delta_margin=data.get("equity", {}).get("delta", 0.0),
            equity_vega_margin=data.get("equity", {}).get("vega", 0.0),
            equity_curvature_margin=data.get("equity", {}).get("curvature", 0.0),
            credit_q_delta_margin=data.get("credit_q", {}).get("delta", 0.0),
            credit_q_vega_margin=data.get("credit_q", {}).get("vega", 0.0),
            credit_q_curvature_margin=data.get("credit_q", {}).get("curvature", 0.0),
            credit_nq_delta_margin=data.get("credit_nq", {}).get("delta", 0.0),
            credit_nq_vega_margin=data.get("credit_nq", {}).get("vega", 0.0),
            credit_nq_curvature_margin=data.get("credit_nq", {}).get("curvature", 0.0),
            commodity_delta_margin=data.get("commodity", {}).get("delta", 0.0),
            commodity_vega_margin=data.get("commodity", {}).get("vega", 0.0),
            commodity_curvature_margin=data.get("commodity", {}).get("curvature", 0.0),
            fx_delta_margin=data.get("fx", {}).get("delta", 0.0),
            fx_vega_margin=data.get("fx", {}).get("vega", 0.0),
            fx_curvature_margin=data.get("fx", {}).get("curvature", 0.0),
            concentration_add_on=data.get("concentration_add_on", 0.0),
            calculation_currency=data.get("calculation_currency"),
            calculation_timestamp=data.get("calculation_timestamp"),
            simm_version=data.get("simm_version"),
        )

    def __str__(self) -> str:
        """String representation of the result."""
        return (
            f"SIMM Result:\n"
            f"  Total Margin: {self.total_margin:,.2f} {self.calculation_currency or ''}\n"
            f"  Delta Margin: {self.delta_margin:,.2f}\n"
            f"  Vega Margin: {self.vega_margin:,.2f}\n"
            f"  Curvature Margin: {self.curvature_margin:,.2f}\n"
            f"  Base Corr Margin: {self.base_corr_margin:,.2f}\n"
        )


@dataclass
class SensitivityCalculationResult:
    """
    Result of a sensitivity calculation.

    This class represents the output of a single engine's sensitivity calculation.
    """

    # The sensitivity collection
    sensitivities: SensitivityCollection

    # Calculation metadata
    calculation_time_ms: float
    positions_processed: int
    engine_name: str

    # Any warnings or errors
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

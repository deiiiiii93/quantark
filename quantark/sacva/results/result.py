"""SA-CVA result with capital + audit decomposition (MAR50.42-50.53)."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SACVAResult:
    """SA-CVA capital result.

    ``total_capital``/``delta_capital``/``vega_capital`` and ``by_risk_class`` are
    m_CVA-scaled capital figures. ``by_bucket`` (K_b), ``bucket_s_b`` (S_b),
    ``bucket_sum_ws`` (sum WS) and ``hedge_disallowance`` (R*sum WS_hdg^2) are RAW,
    unscaled building blocks (multiply by ``m_cva`` for the scaled view).
    """

    total_capital: float
    delta_capital: float
    vega_capital: float
    by_risk_class: Dict[str, float] = field(default_factory=dict)
    by_bucket: Dict[str, float] = field(default_factory=dict)
    bucket_s_b: Dict[str, float] = field(default_factory=dict)
    bucket_sum_ws: Dict[str, float] = field(default_factory=dict)
    hedge_disallowance: Dict[str, float] = field(default_factory=dict)
    m_cva: float = 1.0
    version: str = ""

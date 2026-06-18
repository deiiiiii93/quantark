"""Potential Future Exposure (PFE) calculation for SA-CCR.

Reference: Basel Committee SA-CCR document, paragraphs 146-150.
"""

from quantark.saccr.parameters.supervisory import SupervisoryParameters
from quantark.util.numerical import safe_exp, safe_divide


def calculate_multiplier(
    v: float,
    c: float,
    addon_aggregate: float,
    floor: float = SupervisoryParameters.FLOOR,
) -> float:
    """PFE multiplier (paragraph 149).

    ``multiplier = min(1, Floor + (1 - Floor) * exp((V - C) / (2 (1 - Floor) AddOn)))``

    The multiplier is 1 when the netting set is under-collateralised (V - C >= 0)
    and decreases (floored at ``Floor``) as excess collateral grows. These guards
    are the regulatory definition, not numerical fallbacks.
    """
    if addon_aggregate <= 0:
        return 1.0

    v_minus_c = v - c
    if v_minus_c >= 0:
        return 1.0

    exponent = safe_divide(v_minus_c, 2 * (1 - floor) * addon_aggregate)
    multiplier = floor + (1 - floor) * float(safe_exp(exponent))
    return min(1.0, multiplier)


def calculate_pfe(addon_aggregate: float, multiplier: float) -> float:
    """PFE = ``multiplier * AddOn_aggregate`` (paragraph 146)."""
    return multiplier * addon_aggregate

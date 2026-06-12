"""
SIMM input template (pricing / stress results only).

Generates and loads an Excel workbook whose only inputs are pricing and
stress results: a base PV per trade and one shifted PV per risk-factor
shock, following the sensitivity definitions of ISDA SIMM v2.6
Sections C.2 and C.3.

Example:
    >>> from quantark.simm.template import (
    ...     create_simm_input_template, calculate_simm_from_xlsx)
    >>> create_simm_input_template("simm_input.xlsx")
    >>> # ... fill PricingResults and StressResults ...
    >>> template_input, result = calculate_simm_from_xlsx("simm_input.xlsx")
    >>> print(result.total_margin)
"""

from quantark.simm.template.xlsx_template import (
    PRICING_COLUMNS,
    PRICING_SHEET,
    SHIFT_CONVENTIONS,
    STRESS_COLUMNS,
    STRESS_SHEET,
    create_simm_input_template,
)
from quantark.simm.template.xlsx_loader import (
    SIMMTemplateInput,
    TradePricing,
    calculate_simm_from_xlsx,
    load_simm_input,
)

__all__ = [
    "PRICING_COLUMNS",
    "PRICING_SHEET",
    "SHIFT_CONVENTIONS",
    "STRESS_COLUMNS",
    "STRESS_SHEET",
    "create_simm_input_template",
    "SIMMTemplateInput",
    "TradePricing",
    "calculate_simm_from_xlsx",
    "load_simm_input",
]

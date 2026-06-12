"""
Enumeration types for FX products.
"""

from enum import Enum


class FxPayoutCurrency(Enum):
    """
    Currency in which an FX digital option pays out.

    - DOMESTIC: cash-or-nothing — pays a fixed amount of the domestic
      (quote) currency when in the money.
    - FOREIGN: asset-or-nothing — pays a fixed amount of the foreign
      (base) currency when in the money, worth payout * S_T in domestic
      currency.
    """

    DOMESTIC = "domestic"
    FOREIGN = "foreign"

    def __str__(self):
        return self.value

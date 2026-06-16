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


class FxBarrierType(Enum):
    """Knock direction for a single-barrier FX option.

    - KNOCK_OUT: the option ceases to exist if the barrier is touched.
    - KNOCK_IN: the option only comes into existence if the barrier is touched.
    """

    KNOCK_OUT = "knock_out"
    KNOCK_IN = "knock_in"

    def __str__(self):
        return self.value

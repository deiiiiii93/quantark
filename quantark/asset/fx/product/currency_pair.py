"""
Currency pair specification for FX products.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CurrencyPair:
    """
    A currency pair quoted as units of quote currency per one unit of base currency.

    Follows the standard FX market convention: for EUR/USD the base currency
    is EUR and the quote currency is USD, and a rate of 1.20 means
    1 EUR = 1.20 USD.

    In Garman-Kohlhagen terminology the *quote* currency is the domestic
    (pricing) currency and the *base* currency is the foreign (asset) currency.

    Attributes:
        base_ccy: Base (foreign/asset) currency code, e.g. "EUR"
        quote_ccy: Quote (domestic/pricing) currency code, e.g. "USD"
    """

    base_ccy: str = "FOR"
    quote_ccy: str = "DOM"

    def __post_init__(self):
        object.__setattr__(self, "base_ccy", self.base_ccy.upper())
        object.__setattr__(self, "quote_ccy", self.quote_ccy.upper())

    @property
    def foreign(self) -> str:
        """Foreign (asset) currency — alias for base_ccy."""
        return self.base_ccy

    @property
    def domestic(self) -> str:
        """Domestic (pricing) currency — alias for quote_ccy."""
        return self.quote_ccy

    def __str__(self) -> str:
        return f"{self.base_ccy}/{self.quote_ccy}"

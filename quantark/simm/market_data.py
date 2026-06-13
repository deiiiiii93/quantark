"""Market data contracts used by SIMM sensitivity generation and aggregation."""

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

from quantark.util.exceptions import MarketDataError


@runtime_checkable
class FXRateProvider(Protocol):
    """Provides conversion rates as units of ``to_currency`` per unit of ``from_currency``."""

    def rate(self, from_currency: str, to_currency: str) -> float:
        ...


@dataclass
class MappingFXRateProvider:
    """FX-rate provider backed by direct currency-pair mappings."""

    rates: Dict[Tuple[str, str], float]

    def rate(self, from_currency: str, to_currency: str) -> float:
        source = from_currency.upper()
        target = to_currency.upper()
        if source == target:
            return 1.0
        direct = self.rates.get((source, target))
        if direct is not None and direct > 0:
            return direct
        inverse = self.rates.get((target, source))
        if inverse is not None and inverse > 0:
            return 1.0 / inverse
        raise MarketDataError(f"Missing FX conversion rate from {source} to {target}")


@dataclass
class SIMMMarketData:
    """Market data required outside the pre-computed sensitivity records."""

    fx_rate_provider: Optional[FXRateProvider] = None

    def fx_rate(self, from_currency: str, to_currency: str) -> float:
        source = from_currency.upper()
        target = to_currency.upper()
        if source == target:
            return 1.0
        if self.fx_rate_provider is None:
            raise MarketDataError(
                f"FXRateProvider required to convert {source} to {target}"
            )
        rate = self.fx_rate_provider.rate(source, target)
        if rate <= 0:
            raise MarketDataError(
                f"FX conversion rate from {source} to {target} must be positive"
            )
        return rate

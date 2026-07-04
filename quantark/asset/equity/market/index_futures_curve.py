"""
Index futures marks as an implied dividend/carry curve.

Continuous-compounding inversion of F = S * exp((r - q) * T):

    q(T_i) = r(T_i) - ln(F_i / S) / T_i

Observed basis is intentionally folded into implied carry (tradable futures
risk). Interpolation between nodes is linear in nodal q ("linear_q");
extrapolation outside [T_first, T_last] is flat in nodal q (the
TermStructureDividendYield convention).
"""
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional, Sequence

from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.enum import FuturesCarryRiskMode
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class IndexFuturesQuote:
    contract: str
    maturity: float
    price: float
    multiplier: float
    beta: float = 1.0
    expiry_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.contract:
            raise ValidationError("contract must be non-empty")
        if self.maturity <= 0.0:
            raise ValidationError("maturity must be positive")
        if self.price <= 0.0:
            raise ValidationError("price must be positive")
        if self.multiplier <= 0.0:
            raise ValidationError("multiplier must be positive")
        if self.beta != 1.0:
            raise ValidationError(
                "beta must be 1.0 (cross-hedge beta is not supported in v1; "
                "dPV/dF_i already carries the hedge relationship)"
            )


def hedge_hands(delta_bucket: float, delta_per_hand: float) -> float:
    """Fractional hedge hands: -delta_bucket / delta_per_hand. No rounding."""
    if delta_per_hand <= 0.0:
        raise ValidationError("delta_per_hand must be positive")
    return -delta_bucket / delta_per_hand


def bump_term_yield_node(
    term_div: TermStructureDividendYield, node_index: int, bump: float
) -> TermStructureDividendYield:
    """Return a copy of a term dividend curve with one node bumped."""
    if not 0 <= node_index < len(term_div.times):
        raise ValidationError(
            f"node_index {node_index} out of range for {len(term_div.times)} nodes"
        )
    yields = [float(y) for y in term_div.yields]
    yields[node_index] += float(bump)
    return TermStructureDividendYield(times=list(term_div.times), yields=yields)


@dataclass(frozen=True)
class IndexFuturesCurve:
    underlying: str
    spot: float
    quotes: Sequence[IndexFuturesQuote]
    mode: FuturesCarryRiskMode = FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY
    interpolation: str = "linear_q"

    def __post_init__(self) -> None:
        # snapshot the quotes: a frozen curve must not be mutable through the
        # caller's original list after validation
        object.__setattr__(self, "quotes", tuple(self.quotes))
        if not self.underlying:
            raise ValidationError("underlying must be non-empty")
        if self.spot <= 0.0:
            raise ValidationError("spot must be positive")
        if len(self.quotes) < 2:
            raise ValidationError(
                "at least 2 futures quotes are required "
                "(TermStructureDividendYield needs >= 2 nodes; single-tenor "
                "markets are out of scope in v1)"
            )
        contracts = [q.contract for q in self.quotes]
        if len(set(contracts)) != len(contracts):
            raise ValidationError("futures contracts must be unique")
        maturities = [q.maturity for q in self.quotes]
        if any(
            maturities[i] >= maturities[i + 1] for i in range(len(maturities) - 1)
        ):
            raise ValidationError("quotes must have strictly increasing maturities")
        if self.interpolation != "linear_q":
            raise ValidationError(
                f"unsupported interpolation: {self.interpolation!r} "
                "(only 'linear_q' in v1)"
            )
        if not isinstance(self.mode, FuturesCarryRiskMode):
            raise ValidationError("mode must be a FuturesCarryRiskMode")

    def get_quote(self, contract: str) -> IndexFuturesQuote:
        for quote in self.quotes:
            if quote.contract == contract:
                return quote
        raise ValidationError(f"unknown futures contract: {contract!r}")

    def implied_yields(self, rate_curve: RateCurve) -> list:
        """q(T_i) = r(T_i) - ln(F_i / S) / T_i for each quote, in order."""
        return [
            rate_curve.get_rate(q.maturity)
            - math.log(q.price / self.spot) / q.maturity
            for q in self.quotes
        ]

    def to_dividend_yield_curve(
        self, rate_curve: RateCurve
    ) -> TermStructureDividendYield:
        return TermStructureDividendYield(
            times=[q.maturity for q in self.quotes],
            yields=self.implied_yields(rate_curve),
        )

    def bump_contract(self, contract: str, price_bump: float) -> "IndexFuturesCurve":
        quote = self.get_quote(contract)
        bumped_price = quote.price + price_bump
        if bumped_price <= 0.0:
            raise ValidationError(
                f"bumped price must be positive for {contract}: "
                f"{quote.price} + {price_bump} = {bumped_price}"
            )
        new_quotes = tuple(
            replace(q, price=bumped_price) if q.contract == contract else q
            for q in self.quotes
        )
        return replace(self, quotes=new_quotes)

    def delta_per_hand(self, contract: str) -> float:
        """PnL per hand per futures index point = multiplier (beta == 1.0)."""
        return self.get_quote(contract).multiplier

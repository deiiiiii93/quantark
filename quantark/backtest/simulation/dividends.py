"""The pricer's dividend input on a simulated path (spec 6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.replay.config import AutocallableEngineConfig
from quantark.backtest.replay.dividend_source import term_dividend_yield
from quantark.backtest.replay.market import SignedDividendYield, derive_implied_dividend_yield
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError

from .carry import DayChain


def legacy_implied_q(spot: float, futures_price: float, tenor: float, rate: float) -> float:
    """The engine's historical active-contract channel: simple compounding, floored at zero."""
    _, implied_q = derive_implied_dividend_yield(
        rate=float(rate), spot=float(spot), futures_price=float(futures_price), time_to_maturity=float(tenor)
    )
    return float(implied_q)


@dataclass(frozen=True)
class CurveTailPillars:
    """A simulated carry curve standing in for the IV artifact's parity forwards.

    ``implied_q_pillars(rate)`` returns ``q_k = rate - B(T_k)/T_k`` at the
    curve's tenors, which is all ``surface_tail_carry_yield`` reads; the
    level cancels there and only the forward carry beyond the last listed
    contract survives (spec 6).
    """

    tenors: np.ndarray
    carry: np.ndarray

    def implied_q_pillars(self, rate: float) -> Tuple[List[float], List[float]]:
        t = np.asarray(self.tenors, dtype=float)
        b = np.asarray(self.carry, dtype=float)
        return [float(x) for x in t], [float(rate) - float(bk) / float(tk) for tk, bk in zip(t, b)]


def dividend_yield_for_day(
    chain: DayChain,
    path_index: int,
    *,
    spot: float,
    rate: float,
    engine_config: AutocallableEngineConfig,
    active_contract: str,
    curve_tenors: np.ndarray,
    curve_carry: np.ndarray,
):
    """The dividend object the pricer receives under ``engine_config.dividend_source``.

    ``None`` / ``"active_contract"``: the floored simple-compounded yield of
    ``active_contract`` (``SignedDividendYield``).  ``"futures_curve"``:
    every listed contract with at least ``futures_curve_min_tenor_days`` to
    expiry through the shared ``term_dividend_yield``; for
    ``surface_forward_carry`` the path's own curve supplies the tail.
    ``"surface_forwards"`` has no meaning on a simulated path and raises.
    """
    source = engine_config.dividend_source
    prices = chain.prices[path_index]
    if source in (None, "active_contract"):
        if active_contract not in chain.contracts:
            raise ValidationError(f"active contract {active_contract!r} is not listed on {chain.date.date()}")
        j = chain.contracts.index(active_contract)
        return SignedDividendYield(legacy_implied_q(spot, prices[j], chain.tenors[j], rate))
    if source == "futures_curve":
        min_days = int(engine_config.futures_curve_min_tenor_days)
        quotes = [
            IndexFuturesQuote(
                contract=c, maturity=float(t), price=float(p), multiplier=float(chain.multiplier),
                expiry_date=e.to_pydatetime(),
            )
            for c, e, t, p in zip(chain.contracts, chain.expiries, chain.tenors, prices)
            if round(float(t) * 365.0) >= min_days
        ]
        if not quotes:
            raise ValidationError(
                f"dividend_source='futures_curve' found no contract with at least "
                f"{min_days} days to expiry on {chain.date.date()}"
            )
        return term_dividend_yield(
            quotes,
            spot=float(spot),
            rate_curve=FlatRateCurve(rate=float(rate)),
            extrapolation=engine_config.futures_curve_extrapolation,
            underlying="index",
            artifact=CurveTailPillars(tenors=curve_tenors, carry=curve_carry),
        )
    raise ValidationError(f"dividend_source {source!r} is not available on a simulated path")

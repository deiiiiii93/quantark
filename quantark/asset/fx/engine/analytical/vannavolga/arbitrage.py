"""
No-arbitrage clamps for Vanna-Volga barrier prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BarrierPrices:
    """Container for related barrier prices used by the arbitrage clamps.

    Attributes:
        vanilla: Vanilla price (upper bound for knock-outs).
        ko: Single knock-out price.
        dko: Double knock-out price.
        wko: Window / partial knock-out price.
    """

    vanilla: float
    ko: Optional[float] = None
    dko: Optional[float] = None
    wko: Optional[float] = None


def clamp_basic(p: float) -> float:
    return max(p, 0.0)


def enforce_single_barrier_arbitrage(pr: BarrierPrices) -> BarrierPrices:
    """Enforce 0 <= ko <= wko <= vanilla (where defined)."""
    pr.vanilla = clamp_basic(pr.vanilla)
    if pr.ko is not None:
        pr.ko = min(clamp_basic(pr.ko), pr.vanilla)
    if pr.wko is not None:
        pr.wko = min(clamp_basic(pr.wko), pr.vanilla)
        if pr.ko is not None:
            pr.wko = max(pr.wko, pr.ko)
    return pr


def enforce_double_barrier_arbitrage(
    pr: BarrierPrices, ko1: Optional[float], ko2: Optional[float]
) -> BarrierPrices:
    """Enforce single-barrier clamps plus dko <= min(ko1, ko2)."""
    pr = enforce_single_barrier_arbitrage(pr)
    if pr.dko is not None:
        # A double knock-out cannot be worth more than the vanilla, nor more
        # than either single-barrier knock-out bound (when supplied). Supplied
        # bounds are clamped to be non-negative first so a negative raw VV price
        # cannot drag the DKO below zero.
        pr.dko = min(clamp_basic(pr.dko), pr.vanilla)
        if ko1 is not None:
            pr.dko = min(pr.dko, clamp_basic(ko1))
        if ko2 is not None:
            pr.dko = min(pr.dko, clamp_basic(ko2))
    return pr


__all__ = [
    "BarrierPrices",
    "clamp_basic",
    "enforce_single_barrier_arbitrage",
    "enforce_double_barrier_arbitrage",
]

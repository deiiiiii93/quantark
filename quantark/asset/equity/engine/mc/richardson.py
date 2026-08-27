"""Richardson (Talay-Tubaro) pair extrapolation harness for MC reference prices.

The Euler-family schemes carry a weak error c1*h + O(h^2) in the SDE step size, so
the pair combination 2*P(h/2) - P(h) cancels the leading term. This module runs the
pair at the harness level -- two independently seeded engine prices at substep
factors n and 2n -- so no engine internals change. Intended for certification
references, where the residual O(h^2) bias buys ~an order of magnitude fewer
substeps at equal bias (see docs/lv-mc-scheme-demos/RESULTS.md).

The two legs are treated as statistically independent when combining standard
errors (exact for engines whose draw streams differ, e.g. different Sobol
dimensions or seeds; conservative otherwise is NOT guaranteed -- if the legs are
positively coupled the combined stderr overstates, negatively coupled understates.
The engine factory contract keeps legs independent by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class RichardsonPairResult:
    """Extrapolated price with its legs.

    Attributes:
        price: 2 * fine_price - coarse_price (weak-order-2 combination).
        coarse_price / fine_price: the two legs.
        coarse_substeps / fine_substeps: substeps-per-interval of each leg.
        coarse_std_error / fine_std_error: per-leg MC standard errors (None if
            the engine does not report one).
        std_error: sqrt(4 * fine_se^2 + coarse_se^2) under leg independence,
            or None when either leg lacks a standard error.
    """

    price: float
    coarse_price: float
    fine_price: float
    coarse_substeps: int
    fine_substeps: int
    coarse_std_error: Optional[float]
    fine_std_error: Optional[float]
    std_error: Optional[float]


def richardson_pair_price(
    engine_factory: Callable[[int], object],
    product,
    pricing_env,
    substeps: int = 1,
) -> RichardsonPairResult:
    """Price ``product`` with the Richardson pair 2*P(2n) - P(n).

    Args:
        engine_factory: callable mapping a substeps-per-interval factor to a
            ready engine exposing ``price(product, pricing_env)`` (and
            optionally ``get_last_std_error()``). Called with ``substeps`` and
            ``2 * substeps``; each call must return a FRESH engine.
        product, pricing_env: forwarded to both legs unchanged.
        substeps: the coarse leg's substeps-per-interval (>= 1).
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValidationError(
            f"substeps must be a positive integer, got {substeps!r}"
        )
    coarse_engine = engine_factory(substeps)
    fine_engine = engine_factory(2 * substeps)
    coarse_price = float(coarse_engine.price(product, pricing_env))
    fine_price = float(fine_engine.price(product, pricing_env))

    def _se(engine) -> Optional[float]:
        getter = getattr(engine, "get_last_std_error", None)
        if getter is None:
            return None
        value = getter()
        return None if value is None else float(value)

    coarse_se = _se(coarse_engine)
    fine_se = _se(fine_engine)
    std_error = (
        math.sqrt(4.0 * fine_se * fine_se + coarse_se * coarse_se)
        if coarse_se is not None and fine_se is not None
        else None
    )
    return RichardsonPairResult(
        price=2.0 * fine_price - coarse_price,
        coarse_price=coarse_price,
        fine_price=fine_price,
        coarse_substeps=substeps,
        fine_substeps=2 * substeps,
        coarse_std_error=coarse_se,
        fine_std_error=fine_se,
        std_error=std_error,
    )

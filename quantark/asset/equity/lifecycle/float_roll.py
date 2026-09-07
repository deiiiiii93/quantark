"""Daily roll of schedule-free float-maturity contracts (patch spec 2026-09-03 §4).

A product whose only time coordinate is the scalar ``maturity`` ages by
``days / 365`` from the day its position enters the working portfolio, exactly
as the lifecycle trackers roll their products. Contracts with observation
schedules need a tracker (their timing must shift too), and a float-maturity
``Futures`` hedge is a constant-maturity proxy by design
(``FuturesHedgeInstrument``), so neither is rolled here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from quantark.asset.equity.product.deltaone.base_deltaone_product import BaseDeltaOneProduct
from quantark.asset.equity.product.option.american_option import AmericanOption
from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption

FLOAT_ROLLABLE_PRODUCTS = (EuropeanVanillaOption, AmericanOption, CashOrNothingDigitalOption)
MATURITY_FLOOR = 1e-8          # the trackers' floor (AutocallableLifecycleTracker / BarrierLifecycleTracker)


def _float_only(product: Any) -> bool:
    return (getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity_date", None) is None
            and getattr(product, "maturity", None) is not None)


def is_float_rollable(product: Any) -> bool:
    """Whitelisted class with a float maturity and no dates."""
    return isinstance(product, FLOAT_ROLLABLE_PRODUCTS) and _float_only(product)


def has_unrolled_float_maturity(product: Any) -> bool:
    """Float maturity, no dates, and no roll rule: repriced with a constant maturity.

    Delta-one products are excluded: the futures hedge is a constant-maturity
    proxy by design. The KO-reset snowball is NOT excluded here — registration
    warns about it with a more specific message and marks it warned, but
    registration only runs when the backtest handles lifecycle events, so the
    generic warning has to cover the flag-off case (Kimi review 2026-09-03).
    """
    if is_float_rollable(product) or isinstance(product, BaseDeltaOneProduct):
        return False
    return _float_only(product)


class FloatMaturityRoller:
    """Per-position (base_date, m0); first sight wins."""

    def __init__(self) -> None:
        self._base: Dict[str, Tuple[pd.Timestamp, float]] = {}

    def register(self, position_id: str, product: Any, date) -> None:
        if position_id not in self._base:
            self._base[position_id] = (pd.Timestamp(date).normalize(), float(product.maturity))

    def rolled(self, position_id: str, product: Any, date) -> Any:
        base_date, m0 = self._base[position_id]
        elapsed_days = max(0, (pd.Timestamp(date).normalize() - base_date).days)
        copy = deepcopy(product)
        copy.maturity = max(MATURITY_FLOOR, m0 - elapsed_days / 365.0)
        return copy

    def retain(self, position_ids: Iterable[str]) -> None:
        keep = set(position_ids)
        for pid in [p for p in self._base if p not in keep]:
            del self._base[pid]

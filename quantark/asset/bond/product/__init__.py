"""Bond product definitions."""

from .base_bond_product import BaseBondProduct
from .couponbond.fixed_bond import FixedBond, create_simple_fixed_bond
from .couponbond.frn import FloatingRateBond, FloatingCashFlow, create_simple_frn
from .forward.base_bond_forward import BaseBondForward
from .forward.bond_forward import BondForward
from .futures.bond_futures import BondFutures, DeliverableBond
from .option.euro_short_term_bond_option import (
    EuroShortTermBondOption,
    create_bond_option,
)
from .convertible.convertible_bond import (
    ConvertibleBond,
    CallScheduleEntry,
    PutScheduleEntry,
    DiscreteDividend,
)

__all__ = [
    "BaseBondProduct",
    "FixedBond",
    "create_simple_fixed_bond",
    "FloatingRateBond",
    "FloatingCashFlow",
    "create_simple_frn",
    "BaseBondForward",
    "BondForward",
    "BondFutures",
    "DeliverableBond",
    "EuroShortTermBondOption",
    "create_bond_option",
    "ConvertibleBond",
    "CallScheduleEntry",
    "PutScheduleEntry",
    "DiscreteDividend",
]

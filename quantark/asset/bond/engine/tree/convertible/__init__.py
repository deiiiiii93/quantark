"""
Tree-based pricing engines for convertible bonds.

This module provides:
- ConvertibleBondTreeParams: Configuration for tree-based pricing
- ConvertibleBondBinomialEngine: Goldman Sachs credit-adjusted binomial model
- ConvertibleBondTrinomialEngine: Hull-White trinomial model with default
"""
from quantark.asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTreeParams
from quantark.asset.bond.engine.tree.convertible.binomial_engine import (
    ConvertibleBondBinomialEngine,
)
from quantark.asset.bond.engine.tree.convertible.trinomial_engine import (
    ConvertibleBondTrinomialEngine,
)

__all__ = [
    "ConvertibleBondTreeParams",
    "ConvertibleBondBinomialEngine",
    "ConvertibleBondTrinomialEngine",
]

"""
Tree-based pricing engines for convertible bonds.

This module provides:
- ConvertibleBondTreeParams: Configuration for tree-based pricing
- ConvertibleBondBinomialEngine: Goldman Sachs credit-adjusted binomial model
- ConvertibleBondTrinomialEngine: Hull-White trinomial model with default
- ConvertibleBondTFTreeEngine: Tsiveriotis-Fernandes binomial model (COCB split)
"""
from quantark.asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTreeParams
from quantark.asset.bond.engine.tree.convertible.binomial_engine import (
    ConvertibleBondBinomialEngine,
)
from quantark.asset.bond.engine.tree.convertible.trinomial_engine import (
    ConvertibleBondTrinomialEngine,
)
from quantark.asset.bond.engine.tree.convertible.tf_tree_engine import (
    ConvertibleBondTFTreeEngine,
    ConvertibleBondTFTreeResult,
)

__all__ = [
    "ConvertibleBondTreeParams",
    "ConvertibleBondBinomialEngine",
    "ConvertibleBondTrinomialEngine",
    "ConvertibleBondTFTreeEngine",
    "ConvertibleBondTFTreeResult",
]

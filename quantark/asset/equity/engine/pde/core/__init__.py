"""The 1D PDE core (spec 2 §5): prepare -> march -> readout over column groups.

- ``groups``: column groups and their policies.
- ``backends``: the sparse-LU and banded arithmetic and the factorization cache.
- ``samples``: what ``prepare`` sampled from the environment.
- ``coefficients``: the stencil a march uses.
- ``components``: the operator, stepper and readout seams and their built-ins.
- ``identity``: the fingerprints every key is built from.
"""
from .backends import FactorizationCache
from .components import CentralDifference, CrankNicolson, StandardReadout
from .groups import Backend, ColumnGroup, GroupPolicy

__all__ = ["Backend", "ColumnGroup", "GroupPolicy", "FactorizationCache",
           "CentralDifference", "CrankNicolson", "StandardReadout"]

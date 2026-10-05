"""The independent skewed-surface oracle (spec 3C decision 14): the 1D local-vol barrier kernel, a second
discretization of a local-vol barrier on its own uniform grid. No equity engine calls it after spec 3C Task 18.

It stays in the library (``quantark.volmodels.localvol.pde_kernel``) rather than moving here: the Heston and SLV
barrier kernels' deterministic sigma -> 0 limits (``_deterministic_barrier_price``, ``_deterministic_slv_barrier``)
call it (Task 18 ruling). This module is the test tree's one name for the oracle.
"""
from quantark.volmodels.localvol.pde_kernel import (  # noqa: F401
    _align_discrete_barrier_midcell,
    _solve_ko_lv,
    price_barrier_lv_pde,
)

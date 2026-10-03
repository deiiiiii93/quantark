"""The core kernels against the legacy step formulas (spec 2 §9.3).

The legacy arithmetic is written out here; the comparison is at golden_compare
tolerance (a byte-pinned array would be a cross-architecture golden). The
exact gate is the same-machine byte book.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from golden_compare import assert_close
from quantark.asset.equity.engine.pde.core.backends import (
    FactorizationCache, banded_advance, operator_matrix, sparse_lu_advance)
from quantark.asset.equity.engine.pde.core.coefficients import StepSet
from quantark.asset.equity.engine.pde.core.groups import (
    VALUE, Backend, ColumnGroup, GroupState)

RNG = np.random.default_rng(7)
N = 12


def _lcu():
    l = 0.8 + 0.1 * RNG.random(N); c = -2.0 - 0.1 * RNG.random(N); u = 0.9 + 0.1 * RNG.random(N)
    return l, c, u


def _set(l, c, u):
    return StepSet(index=0, l=l, c=c, u=u, operator=lambda: operator_matrix(l, c, u, N),
                   factorizations=FactorizationCache())


def _legacy_banded(l, c, u, dt, theta, v_next, lower, upper):
    lower_d = -theta * dt * l[2:-1]; main = 1.0 - theta * dt * c[1:-1]; upper_d = -theta * dt * u[1:-2]
    ab = np.zeros((3, len(main))); ab[0, 1:] = upper_d; ab[1, :] = main; ab[2, :-1] = lower_d
    lower1 = (1.0 - theta) * dt * l[2:-1]; main1 = 1.0 + (1.0 - theta) * dt * c[1:-1]; upper1 = (1.0 - theta) * dt * u[1:-2]
    cols = []
    for k in range(v_next.shape[1]):
        vn = v_next[1:-1, k]
        rhs = main1 * vn; rhs[1:] += lower1 * vn[:-1]; rhs[:-1] += upper1 * vn[1:]
        rhs[0] += dt * ((1.0 - theta) * l[1] * v_next[0, k] + theta * l[1] * lower[k])
        rhs[-1] += dt * ((1.0 - theta) * u[-2] * v_next[-1, k] + theta * u[-2] * upper[k])
        cols.append(solve_banded((1, 1), ab, rhs))
    return np.column_stack(cols)


def test_two_column_banded_group_matches_the_legacy_step():
    l, c, u = _lcu(); v = RNG.random((N, 2)); lo = RNG.random(2); up = RNG.random(2)
    g = ColumnGroup("value", ("alive", "ki"), Backend.BANDED, VALUE)
    out = banded_advance(GroupState(g, v), _set(l, c, u), np.float64(0.01), 0.5, (lo, up))
    assert_close(out.tolist(), _legacy_banded(l, c, u, 0.01, 0.5, v, lo, up).tolist())


def test_rolling_value_group_matches_the_legacy_banded_step():
    # the step arithmetic does not depend on the policy: the rolling statistics group (spec 3B §5.1)
    l, c, u = _lcu(); v = RNG.random((N, 6))
    g = ColumnGroup("stats", ("stats_v0", "stats_v1"), Backend.BANDED, VALUE, rolling=True)
    out = banded_advance(GroupState(g, v), _set(l, c, u), 0.01, 1.0, (v[0], v[-1]))
    assert_close(out.tolist(), _legacy_banded(l, c, u, 0.01, 1.0, v, v[0], v[-1]).tolist())


def test_single_column_lu_matches_the_legacy_step():
    l, c, u = _lcu(); v = RNG.random((N, 1)); lo = RNG.random(1); up = RNG.random(1)
    g = ColumnGroup("value", ("value",), Backend.SPARSE_LU, VALUE)
    out = sparse_lu_advance(GroupState(g, v), _set(l, c, u), 0.01, 0.5, (lo, up))
    A = operator_matrix(l, c, u, N); I = sp.eye(N - 2, format="csc")
    M1 = I + 0.5 * 0.01 * A; lu = spla.splu(I - 0.5 * 0.01 * A)
    rhs = M1 @ v[1:-1, 0]
    rhs[0] += 0.01 * (0.5 * l[1] * v[0, 0] + 0.5 * l[1] * lo[0])
    rhs[-1] += 0.01 * (0.5 * u[-2] * v[-1, 0] + 0.5 * u[-2] * up[0])
    assert_close(out[:, 0].tolist(), lu.solve(rhs).tolist())

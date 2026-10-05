"""Nodal coefficient sets never collapse to the scalar stencil (spec 3C decision 11). The end-to-end row - a
local-vol European with time_steps=1 not equal to the flat-vol price - arrives with Task 17."""
import numpy as np

from quantark.asset.equity.engine.pde.core.adapters.base import FamilyAdapter
from quantark.asset.equity.engine.pde.core.samples import SampleKind


class _Sets:
    n_unique = 1
    lcu_sets = (np.zeros((3, 4)),)


def test_collapse_is_keyed_on_the_sample_kind(monkeypatch):
    import quantark.asset.equity.engine.pde.core.adapters.base as base
    monkeypatch.setattr(base, "validate_coefficients", lambda *a, **k: None)

    class Solver:
        class params:
            from quantark.asset.equity.engine.pde.core.components import CentralDifference
            operator = CentralDifference()

        def _calculate_coefficients(self, *a):
            return "flat"

        def _flat_exact_step_coefficients(self, sc, *a):
            return sc

    class Layout:
        class spatial:
            dx = np.ones(3)
            x = np.zeros(4)

        class time:
            dt = np.ones(1)

    class Session:
        coefficients = _Sets()

    for kind, collapsed in ((SampleKind.SCALAR, True), (SampleKind.NODAL, False)):
        adapter = FamilyAdapter(Solver())
        adapter.sample_kind = kind
        adapter._scalars = lambda product, env, tau: (0.0, 0.0, 0.2)
        out = adapter.coefficients(None, None, Layout, 1.0, None, Session)
        assert (out.sets is None) is collapsed, kind

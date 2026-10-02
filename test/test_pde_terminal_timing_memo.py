"""The terminal settlement timing is resolved once per march, not once per step (stage-1 timing fix).

The barrier, european and american boundary and event hooks read the terminal payment time and
delay DF on every step; ``BasePDESolver._terminal_timing`` keeps it in the march's market scope.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "pde_bitwise"))
import capture_book as cb  # noqa: E402

import quantark.asset.equity.engine.pde.american_pde_solver as american_mod  # noqa: E402
import quantark.asset.equity.engine.pde.base_pde_solver as base_mod  # noqa: E402
import quantark.asset.equity.engine.pde.european_pde_solver as european_mod  # noqa: E402
from quantark.asset.equity.engine import pde  # noqa: E402
from quantark.asset.equity.engine import settlement_support  # noqa: E402
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402

CASES = {
    "barrier": (pde.BarrierPDESolver, lambda: cb.barrier(ObservationType.CONTINUOUS)),
    "european": (pde.EuropeanPDESolver, cb.european),
    "american": (pde.AmericanPDESolver, cb.american_tenor),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_warm_price_resolves_terminal_timing_a_handful_of_times(name, monkeypatch):
    cls, factory = CASES[name]
    solver, product, env = cls(params=PDEParams()), factory(), cb.env_flat()
    solver.price(product, env)
    calls = []
    real = settlement_support.resolve_terminal_timing

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    for mod in (base_mod, european_mod, american_mod, settlement_support):
        if hasattr(mod, "resolve_terminal_timing"):
            monkeypatch.setattr(mod, "resolve_terminal_timing", counting)
    solver.price(product, env)
    assert 0 < len(calls) <= 10, len(calls)        # ~1000 steps: per-step resolution would be >= 1000

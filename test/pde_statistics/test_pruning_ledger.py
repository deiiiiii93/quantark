"""Pruning a stream drops its report only (spec 3B §5.1 "streams", Review Focus 2): the ledger stays."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.cashleg.event_distribution import EventType
from pde_statistics.cases import CASES


@pytest.mark.parametrize("streams", [frozenset({EventType.KO}), frozenset(), frozenset({EventType.KI})],
                         ids=["ko", "none", "ki"])
def test_snowball_pruning_is_reporting_only(value_mode, streams):
    value_mode("snowball")
    solver, product, e = SnowballPDESolver(PDEParams()), CASES["sb_cont"][0](), CASES["sb_cont"][1]()
    full, lean = solver.calculate_event_stats(product, e), solver.calculate_event_stats(product, e, streams=streams)
    for f in ("determination_times", "payment_times", "expected_discounted_cashflows", "ko_probability",
              "survival_probability", "expected_discounted_ko_cashflow"):
        assert np.array_equal(getattr(full, f), getattr(lean, f)), f
    assert full.pv == lean.pv and full.reconciliation_error == lean.reconciliation_error
    assert full.expected_discounted_maturity_cashflow == lean.expected_discounted_maturity_cashflow
    if EventType.KI in streams:
        assert lean.ki_ever_probability == full.ki_ever_probability

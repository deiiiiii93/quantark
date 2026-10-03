"""Pruning a stream drops its report only (spec 3B §5.1 "streams", Review Focus 2): the ledger stays."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.cashleg.event_distribution import EventType
from pde_statistics.cases import CASES

LEDGER = ("determination_times", "payment_times", "expected_discounted_cashflows", "ko_probability",
          "survival_probability", "expected_discounted_ko_cashflow")


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


@pytest.mark.parametrize("case", ["ph_cont", "ph_memory"])
@pytest.mark.parametrize("streams", [frozenset({EventType.KO}), frozenset(), frozenset({EventType.KI})],
                         ids=["ko", "none", "ki"])
def test_phoenix_pruning_is_reporting_only(value_mode, case, streams):
    value_mode("phoenix")
    solver, product, e = PhoenixPDESolver(PDEParams()), CASES[case][0](), CASES[case][1]()
    full, lean = solver.calculate_event_stats(product, e), solver.calculate_event_stats(product, e, streams=streams)
    for f in LEDGER:
        assert np.array_equal(getattr(full, f), getattr(lean, f)), f
    assert full.pv == lean.pv and full.reconciliation_error == lean.reconciliation_error
    assert full.expected_discounted_maturity_cashflow == lean.expected_discounted_maturity_cashflow
    if EventType.COUPON not in streams:    # the coupon report is pruned, its ledger rows are not (decision 3)
        assert lean.coupon_probability.size == 0 and lean.expected_discounted_coupon_cashflow.size == 0

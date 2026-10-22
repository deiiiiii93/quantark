"""Estimate-blind Neyman allocation and precision stopping (spec WS-S)."""

import dataclasses

import numpy as np
import pytest
from scipy.stats import t as student_t

from quantark.util.exceptions import ValidationError
from quantark.validation.adaptive_allocation import (
    CellPrecision,
    neyman_allocation,
    precision_stop,
    projected_aggregate_halfwidth,
)


def _cells(sds=(1.0, 1.0, 1.0), costs=(1.0, 1.0, 1.0), n=32):
    return [
        CellPrecision(name=f"c{i}", n_batches=n, batch_sd=s, seconds_per_batch=c)
        for i, (s, c) in enumerate(zip(sds, costs))
    ]


def test_blindness_no_estimate_field():
    # S-G1: the type itself must not carry an estimate.
    assert "estimate" not in {f.name for f in dataclasses.fields(CellPrecision)}
    with pytest.raises(TypeError):
        CellPrecision(
            name="x", n_batches=4, batch_sd=1.0, seconds_per_batch=1.0, estimate=0.5
        )


def test_neyman_matches_analytic_optimum():
    # S-G2: n_j proportional to sd_j / sqrt(cost_j); equal costs, sds 1:2:4.
    cells = _cells(sds=(1.0, 2.0, 4.0), costs=(1.0, 1.0, 1.0))
    alloc = neyman_allocation(cells, budget_seconds=7000.0, min_batches=16)
    assert alloc["c1"] / alloc["c0"] == pytest.approx(2.0, rel=0.05)
    assert alloc["c2"] / alloc["c0"] == pytest.approx(4.0, rel=0.05)
    spent = sum(alloc[c.name] * c.seconds_per_batch for c in cells)
    assert spent <= 7000.0


def test_neyman_cost_weighting():
    # Quadrupling a cell's cost halves its batch share (1/sqrt(4)).
    cheap = neyman_allocation(
        _cells(sds=(1.0, 1.0), costs=(1.0, 1.0)), budget_seconds=2000.0
    )
    costly = neyman_allocation(
        _cells(sds=(1.0, 1.0), costs=(1.0, 4.0)), budget_seconds=2000.0
    )
    assert cheap["c1"] / cheap["c0"] == pytest.approx(1.0, rel=0.05)
    assert costly["c1"] / costly["c0"] == pytest.approx(0.5, rel=0.1)


def test_min_batches_floor():
    cells = _cells(sds=(0.001, 5.0), costs=(1.0, 1.0))
    alloc = neyman_allocation(cells, budget_seconds=5000.0, min_batches=16)
    assert alloc["c0"] >= 16


def test_projected_halfwidth_matches_hand_computation():
    cells = _cells(sds=(0.7, 1.1), costs=(1.0, 1.0), n=64)
    se = np.sqrt((0.7**2 / 64 + 1.1**2 / 64) / 4.0)
    df = 63 + 63
    expected = float(student_t.ppf(0.975, df)) * se
    assert projected_aggregate_halfwidth(cells) == pytest.approx(expected, rel=1e-12)


def test_precision_stop_target_reached():
    cells = _cells(sds=(0.01, 0.01), costs=(1.0, 1.0), n=64)
    decision = precision_stop(
        cells, target_halfwidth=0.02, elapsed_seconds=10.0, budget_seconds=100.0
    )
    assert decision.stop is True
    assert decision.trigger == "target-reached"
    assert decision.projected_halfwidth <= 0.02


def test_precision_stop_budget_cap():
    cells = _cells(sds=(5.0, 5.0), costs=(1.0, 1.0), n=8)
    decision = precision_stop(
        cells, target_halfwidth=0.02, elapsed_seconds=101.0, budget_seconds=100.0
    )
    assert decision.stop is True
    assert decision.trigger == "budget-cap"


def test_precision_stop_keep_going():
    cells = _cells(sds=(5.0, 5.0), costs=(1.0, 1.0), n=8)
    decision = precision_stop(
        cells, target_halfwidth=0.02, elapsed_seconds=10.0, budget_seconds=100.0
    )
    assert decision.stop is False
    assert decision.trigger is None


def test_rejects_degenerate_inputs():
    with pytest.raises(ValidationError):
        CellPrecision(name="x", n_batches=1, batch_sd=1.0, seconds_per_batch=1.0)
    with pytest.raises(ValidationError):
        CellPrecision(name="x", n_batches=4, batch_sd=1.0, seconds_per_batch=0.0)
    with pytest.raises(ValidationError):
        neyman_allocation(_cells(), budget_seconds=0.0)
    with pytest.raises(ValidationError):
        projected_aggregate_halfwidth([])

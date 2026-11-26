"""QUAD V1 statistics on the forward route (0.16.0): every outcome column off one adjoint forward pass.

The default statistics read the knock-out units and the terminal maturity / knocked-in settlement columns off the
march's adjoint forward pass instead of marching one backward column per outcome (the 0.13.0 cost: one row per
knock-out record). The stacked route stays for a march the adjoint cannot transpose, and as the reference here:

* AC1: forward and stacked agree within the adjoint identity (knock-out probability per record 1e-12, terminal
  probabilities 1e-11, cash 1e-9 on a 100 notional); the knock-in path fields and the coupon fields are bitwise.
* AC3: a refused transpose re-runs the stacked march and gives its numbers bitwise.
* price() is untouched (the statistics' ``pv`` is ``price()`` on both routes).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.core.statistics.adjoint import AdjointUnavailable
from quantark.asset.equity.engine.quad import v1_statistics

from test_asset.equity.engine_quad.test_quad_v1_event_statistics import ADJOINT_CASES, adjoint_case
from test_asset.equity.engine_quad.test_quad_v1_exact_mass import CASES as EXACT_MASS
from test_asset.equity.engine_quad.test_quad_v1_exact_mass import _env as exact_env

PROBABILITY_TOL = 1e-12
TERMINAL_TOL = 1e-11
CASH_TOL = 1e-9
#: fields whose numbers come from the outcome columns (compared to tolerance); everything else is bitwise
OUTCOME_PATHS = ("observations", "maturity", "ki_settlement", "ledger", "total_discounted_cash", "expected_lifetime",
                 "probability_residual", "reconciliation_error", "knock_in.expected_loss",
                 # derived from the knock-out probabilities: P(KI and no KO) = P(KI settlement), coupon survival
                 "knock_in.probability_no_ko", "coupons.survival", "total_cash_to_pay")


def _stats(engine, product, market, route):
    engine._v1_statistics_route = route
    return engine.calculate_event_stats(product, market)


def _near(a, b, path, tol):
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape, path
        if a.dtype.kind in "fc":
            np.testing.assert_allclose(a, b, rtol=0.0, atol=tol, err_msg=path)
        else:
            assert np.array_equal(a, b), path
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol, f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, path


def _tol(path):
    if path.endswith(("discounted_cash", "cash_to_pay", "expected_loss", "total_discounted_cash", "total_cash_to_pay",
                      "reconciliation_error")):
        return CASH_TOL
    if path.startswith(("maturity", "ki_settlement", "probability_residual", "expected_lifetime")):
        return TERMINAL_TOL
    return PROBABILITY_TOL


def compare(a, b, path="stats"):
    """``a`` (forward) vs ``b`` (stacked): outcome fields within the adjoint identity, the rest bitwise."""
    assert type(a) is type(b), path
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            compare(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)) and not isinstance(a, str):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]")
    elif isinstance(a, (np.ndarray, float)):
        outcome = any(path.startswith(f"stats.{p}") for p in OUTCOME_PATHS)
        if outcome:
            _near(a, b, path, _tol(path[len("stats."):]))
        elif isinstance(a, float):
            assert (math.isnan(a) and math.isnan(b)) or a == b, f"{path}: {a!r} != {b!r}"
        else:
            assert a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("name", ADJOINT_CASES)
def test_forward_route_matches_the_stacked_route(name):
    engine, product, market = adjoint_case(name)
    forward = _stats(engine, product, market, "forward")
    stacked = _stats(type(engine)(engine.params), product, market, "stacked")
    assert forward.pv == stacked.pv == type(engine)(engine.params).price(product, market)
    compare(forward, stacked)


@pytest.mark.parametrize("name", ["daily_ko_1y_vol05", "quarter_hour_ki_8001"])
def test_forward_route_matches_on_corrected_local_steps(name):
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.param import QuadParams

    build, vol, params, *_ = EXACT_MASS[name]
    product, market = build(), exact_env(vol)
    forward = _stats(SnowballQuadEngine(params or QuadParams()), product, market, "forward")
    stacked = _stats(SnowballQuadEngine(params or QuadParams()), product, market, "stacked")
    compare(forward, stacked)


@pytest.mark.parametrize("name", ["sb_cki", "ph_mem_expiry_cki", "kr_dki"])
def test_a_refused_transpose_reruns_the_stacked_march(name, monkeypatch):
    engine, product, market = adjoint_case(name)

    def refuse(log):
        raise AdjointUnavailable("undefined: test refusal")

    monkeypatch.setattr(v1_statistics, "forward_pass", refuse)
    forward = _stats(engine, product, market, "forward")
    stacked = _stats(type(engine)(engine.params), product, market, "stacked")
    assert forward.unavailable["knock_in.probability"] == "undefined: test refusal"
    compare_bitwise(forward, stacked)


def compare_bitwise(a, b, path="stats"):
    assert type(a) is type(b), path
    if dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            compare_bitwise(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), path
        for k in a:
            compare_bitwise(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)) and not isinstance(a, str):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare_bitwise(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), path
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == b, f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_the_route_switch_refuses_an_unknown_route():
    from quantark.util.exceptions import ValidationError

    engine, product, market = adjoint_case("sb_cki")
    engine._v1_statistics_route = "sideways"
    with pytest.raises(ValidationError, match="route"):
        engine.calculate_event_stats(product, market)

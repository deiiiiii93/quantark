"""Tier-1 tests for the ONE spatial builder (spec §4.4)."""

import logging

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid import (
    GridConfig,
    GridRequest,
    MarketSnapshot,
    resolve_config,
)
from quantark.asset.equity.engine.pde.grid.space import build_space
from quantark.util.exceptions import ValidationError

MKT = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)


def CFG(**kw):
    return resolve_config("standard", GridConfig(**kw) if kw else None)


def req(**kw):
    base = dict(
        tau=1.0,
        bound_anchors=(100.0,),
        critical_prices=(80.0, 100.0, 103.0),
        hard_lower=None,
        hard_upper=None,
        event_times=(),
    )
    base.update(kw)
    return GridRequest(**base)


def test_auto_bounds_formula():
    lay = build_space(req(), MKT, CFG())
    h = 4.0 * 0.2 * 1.0 + abs(0.03 - 0.01 - 0.5 * 0.04) * 1.0
    lo, hi = lay.bounds
    assert lo <= 100.0 * np.exp(-h) * (1 + 1e-12)
    assert hi >= 100.0 * np.exp(h) * (1 - 1e-12)


def test_bound_anchor_envelope_covers_strike():
    lay = build_space(req(bound_anchors=(100.0, 150.0)), MKT, CFG())
    h = 4.0 * 0.2 + abs(0.03 - 0.01 - 0.5 * 0.04)
    assert lay.bounds[1] >= 150.0 * np.exp(h) * (1 - 1e-12)


def test_critical_price_margin():
    # a far critical price still sits strictly interior with 5-cell margin
    cfg = CFG()
    lay = build_space(req(critical_prices=(100.0, 260.0)), MKT, cfg)
    margin = 5.0 * np.log1p(cfg.eps_crit)
    assert np.log(lay.bounds[1]) >= np.log(260.0) + margin * (1 - 1e-9)


def test_near_expiry_floor():
    lay = build_space(req(tau=1e-4, critical_prices=(100.0,)), MKT, CFG())
    assert lay.bounds[0] <= 100.0 / 1.10 * (1 + 1e-12)
    assert lay.bounds[1] >= 100.0 * 1.10 * (1 - 1e-12)


def test_hard_bound_is_domain_edge():
    lay = build_space(req(hard_upper=103.0), MKT, CFG())
    assert lay.bounds[1] == 103.0 and lay.s[-1] == pytest.approx(103.0, abs=0.0)
    # other side stays auto
    assert lay.bounds[0] < 80.0


def test_achieved_eps_meets_target():
    cfg = CFG()
    lay = build_space(req(), MKT, cfg)
    assert lay.achieved_eps <= cfg.eps_crit * (1 + 1e-6)


def test_uniform_when_no_interior_criticals():
    lay = build_space(req(critical_prices=(100.0,), bound_anchors=(100.0,)), MKT, CFG())
    # a single critical at spot still concentrates; empty → uniform:
    lay2 = build_space(
        req(critical_prices=(), bound_anchors=(100.0,)), MKT, CFG()
    )
    dx2 = np.diff(lay2.x)
    assert np.allclose(dx2, dx2[0])
    assert lay.x.shape == lay2.x.shape


def test_uniform_clamp_when_already_fine():
    # tight hard bounds + many points: uniform spacing already beats eps_crit
    lay = build_space(
        req(hard_lower=95.0, hard_upper=105.0, critical_prices=(98.0, 102.0)),
        MKT,
        CFG(points=2000),
    )
    dx = np.diff(lay.x)
    assert dx.max() / dx.min() < 1.05  # near-uniform


def test_unreachable_target_fails_closed():
    with pytest.raises(ValidationError, match="GridConfig\\(points"):
        build_space(
            req(critical_prices=(80.0, 100.0, 125.0)),
            MKT,
            CFG(points=31, eps_crit=1e-4),
        )


def test_unreachable_critical_does_not_change_grid():
    base = req(critical_prices=(80.0, 100.0, 103.0))
    marked = req(critical_prices=(80.0, 100.0, 103.0, 10_000.0))
    base_layout = build_space(base, MKT, CFG())
    marked_layout = build_space(marked, MKT, CFG())

    assert marked_layout.bounds == base_layout.bounds
    assert np.array_equal(marked_layout.x, base_layout.x)
    assert marked_layout.active_critical_prices == base_layout.active_critical_prices


@pytest.mark.parametrize("points", [201, 401, 801, 1001, 2001])
def test_dense_critical_point_sweep_is_finite_monotone(points):
    criticals = tuple(np.linspace(75.0, 105.0, 34))
    lay = build_space(
        req(critical_prices=criticals),
        MKT,
        CFG(points=points, max_points=2500),
    )
    dx = np.diff(lay.x)
    assert np.all(np.isfinite(lay.x))
    assert np.all(dx > 0.0)
    assert dx.max() / dx.min() <= 100.0 * (1.0 + 1e-8)


def test_critical_outside_hard_bounds_excluded(caplog):
    with caplog.at_level(logging.INFO):
        lay = build_space(
            req(hard_lower=90.0, hard_upper=110.0, critical_prices=(80.0, 100.0)),
            MKT,
            CFG(),
        )
    assert lay.bounds == (90.0, 110.0)  # 80 did not drag the domain


def test_monotone_and_consistent():
    lay = build_space(req(), MKT, CFG())
    assert np.all(np.diff(lay.x) > 0)
    assert np.allclose(lay.s, np.exp(lay.x))
    assert np.allclose(lay.dx, np.diff(lay.x))
    assert len(lay.s) == CFG().points


def test_immutability_and_identity():
    lay = build_space(req(), MKT, CFG())
    with pytest.raises(ValueError):
        lay.x[0] = 0.0
    other = build_space(req(), MKT, CFG())
    assert lay != other and lay == lay  # eq=False

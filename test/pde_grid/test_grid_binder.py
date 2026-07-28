"""Tier-1 tests for GridBinder / Layout / validate_external_layout (spec §4.2, §4.6)."""

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid import (
    GridConfig,
    GridRequest,
    MarketSnapshot,
)
from quantark.asset.equity.engine.pde.grid.binder import (
    GridBinder,
    validate_external_layout,
)
from quantark.util.exceptions import NumericalError, PricingError, ValidationError

MKT = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)


def req(**kw):
    base = dict(
        tau=1.0,
        bound_anchors=(100.0,),
        critical_prices=(100.0, 103.0),
        hard_lower=None,
        hard_upper=None,
        event_times=(0.25, 0.5, 0.75),
    )
    base.update(kw)
    return GridRequest(**base)


def binder(**kw):
    args = dict(cache_enabled=True, cache_max_entries=128)
    args.update(kw)
    return GridBinder("standard", None, **args)


def test_bind_returns_layout_with_provenance():
    b = binder()
    lay = b.bind(req(), MKT)
    assert lay.request == req()
    assert lay.spatial.s.shape == (400,)
    assert lay.time.step_of[0.5] > 0
    assert lay.config_key == b.config.key


def test_cache_identity_reuse():
    b = binder()
    assert b.bind(req(), MKT) is b.bind(req(), MKT)


def test_cache_disabled_distinct_objects():
    b = binder(cache_enabled=False)
    a, c = b.bind(req(), MKT), b.bind(req(), MKT)
    assert a is not c and a != c  # eq=False identity semantics


def test_lru_eviction():
    b = binder(cache_max_entries=2)
    l1 = b.bind(req(tau=1.0), MKT)
    b.bind(req(tau=2.0), MKT)
    b.bind(req(tau=3.0), MKT)  # evicts tau=1.0
    assert b.bind(req(tau=1.0), MKT) is not l1


def test_bind_shared_empty_rejected():
    with pytest.raises(ValidationError):
        binder().bind_shared([], MKT)


def test_bind_shared_single_equivalent_to_bind():
    b = binder()
    shared = b.bind_shared([req()], MKT)[0]
    solo = b.bind(req(), MKT)
    assert np.array_equal(shared.spatial.s, solo.spatial.s)
    assert np.array_equal(shared.time.t, solo.time.t)


def test_bind_shared_spatial_identity_and_time_dedup():
    b = binder()
    r1, r2, r3 = req(), req(event_times=(0.1, 0.9)), req()  # r3 == r1
    layouts = b.bind_shared([r1, r2, r3], MKT)
    assert len({id(l.spatial) for l in layouts}) == 1
    assert layouts[0].time is layouts[2].time  # value-identical share
    assert layouts[0].time is not layouts[1].time


def test_bind_shared_union_bounds_cover_all():
    b = binder()
    r_far = req(critical_prices=(100.0, 160.0), tau=2.0)
    layouts = b.bind_shared([req(), r_far], MKT)
    lo, hi = layouts[0].spatial.bounds
    assert hi > 160.0  # union criticals + max tau widened the shared domain


def test_bind_shared_rejects_hard_bounds():
    with pytest.raises(ValidationError):
        binder().bind_shared([req(), req(hard_upper=103.0)], MKT)


def test_rebind_time_keeps_spatial_identity():
    b = binder()
    base = b.bind(req(), MKT)
    rolled = b.rebind_time(base, req(tau=0.9, event_times=(0.15, 0.4, 0.65)))
    assert rolled.spatial is base.spatial
    assert rolled.time is not base.time
    assert rolled.request.tau == 0.9


def test_infeasible_hard_bounds_raise_pricing_error():
    with pytest.raises(PricingError):
        binder().bind(req(hard_lower=150.0, hard_upper=200.0), MKT)


def test_same_side_expert_hard_conflict():
    b_lo = GridBinder("standard", GridConfig(bounds=(60.0, None)))
    with pytest.raises(ValidationError):
        b_lo.bind(req(hard_lower=80.0), MKT)
    b_hi = GridBinder("standard", GridConfig(bounds=(None, 140.0)))
    with pytest.raises(ValidationError):
        b_hi.bind(req(hard_upper=103.0), MKT)
    # opposite sides compose fine
    ok = GridBinder("standard", GridConfig(bounds=(60.0, None)))
    lay = ok.bind(req(hard_upper=103.0), MKT)
    assert lay.spatial.bounds == (60.0, 103.0)


def test_external_validation_accepts_match_and_drifted_criticals():
    b = binder()
    lay = b.bind(req(), MKT)
    fresh = req(critical_prices=(101.0, 103.0))  # spot drifted under a bump
    validate_external_layout(lay, fresh, MarketSnapshot(101.0, 0.2, 0.03, 0.01))


def test_external_validation_accepts_unchanged_unreachable_marker():
    b = binder()
    marked = req(
        critical_prices=(100.0, 103.0, 10_000.0),
        event_times=(0.1, 0.2),
    )
    lay = b.bind(marked, MKT)
    assert 10_000.0 not in lay.spatial.active_critical_prices
    assert lay.time.step_at(0.1) > 0
    assert lay.time.step_at(0.2) > lay.time.step_at(0.1)
    validate_external_layout(lay, marked, MKT)


def test_external_validation_rejects_alignment_mismatches():
    b = binder()
    lay = b.bind(req(), MKT)
    for bad in (
        req(tau=2.0),
        req(event_times=(0.25, 0.5)),
        req(hard_lower=80.0),
        req(hard_upper=120.0),
    ):
        with pytest.raises(ValidationError):
            validate_external_layout(lay, bad, MKT)


def test_external_validation_coverage():
    b = binder()
    lay = b.bind(req(), MKT)
    far = MarketSnapshot(500.0, 0.2, 0.03, 0.01)  # spot outside frozen domain
    with pytest.raises(NumericalError):
        validate_external_layout(lay, req(), far)
    inside = req(critical_prices=(100.0, 5000.0))  # critical outside domain
    with pytest.raises(NumericalError):
        validate_external_layout(lay, inside, MKT)


def test_import_surface_complete():
    import quantark.asset.equity.engine.pde.grid as g

    for name in (
        "GridRequest", "MarketSnapshot", "GridConfig", "resolve_config",
        "GridBinder", "Layout", "validate_external_layout",
        "SpatialLayout", "TimeLayout", "build_space", "build_time",
        "EventSchedule", "breach_weights", "project_between", "project_piecewise",
    ):
        assert hasattr(g, name), name


def test_external_validation_hard_edge_exemption():
    b = binder()
    lay = b.bind(req(hard_upper=103.0, critical_prices=(100.0, 103.0)), MKT)
    # 103.0 IS the edge — is_close exemption keeps this valid
    validate_external_layout(
        lay, req(hard_upper=103.0, critical_prices=(100.0, 103.0)), MKT
    )

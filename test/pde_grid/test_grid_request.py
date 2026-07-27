"""Tier-1 tests for GridRequest / MarketSnapshot (spec §4.2, §4.9)."""

import pytest

from quantark.asset.equity.engine.pde.grid import GridRequest, MarketSnapshot
from quantark.util.exceptions import ValidationError


def make_request(**kw):
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


def test_valid_request_hashable_and_equal_by_value():
    assert make_request() == make_request()
    assert hash(make_request()) == hash(make_request())


def test_tau_must_be_positive():
    with pytest.raises(ValidationError):
        make_request(tau=0.0)


def test_event_times_must_be_interior():
    with pytest.raises(ValidationError):
        make_request(event_times=(0.0, 0.5))  # t=0 belongs to valuation_readout
    with pytest.raises(ValidationError):
        make_request(event_times=(0.5, 1.0))  # t=tau belongs to terminal stage


def test_event_times_deduplicated_and_sorted():
    r = make_request(event_times=(0.5, 0.25, 0.5 + 1e-14))
    assert r.event_times == (0.25, 0.5)


def test_hard_bounds_ordering():
    with pytest.raises(ValidationError):
        make_request(hard_lower=120.0, hard_upper=80.0)
    r = make_request(hard_lower=80.0, hard_upper=None)  # single-sided is fine
    assert r.hard_lower == 80.0 and r.hard_upper is None


def test_prices_positive():
    with pytest.raises(ValidationError):
        make_request(critical_prices=(100.0, -3.0))
    with pytest.raises(ValidationError):
        make_request(bound_anchors=(0.0,))


def test_bound_anchors_required():
    with pytest.raises(ValidationError):
        make_request(bound_anchors=())


def test_market_snapshot_value_semantics():
    a = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)
    b = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)
    assert a == b and hash(a) == hash(b)


def test_market_snapshot_validation():
    with pytest.raises(ValidationError):
        MarketSnapshot(spot=0.0, sigma_ref=0.2, r_ref=0.0, q_ref=0.0)
    with pytest.raises(ValidationError):
        MarketSnapshot(spot=100.0, sigma_ref=-0.1, r_ref=0.0, q_ref=0.0)

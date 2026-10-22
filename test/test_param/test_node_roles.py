"""NodeRole metadata tests (spec WP3.2)."""
import pytest

from quantark.param.node_roles import NodeRole, resolve_node_roles
from quantark.util.exceptions import ValidationError


def test_inference_fallback():
    info = resolve_node_roles([0.25, 0.5, 1.0], None, None)
    assert info.roles == (NodeRole.CALIBRATED,) * 3
    assert info.last_observable_tenor == 1.0
    assert info.roles_inferred is True


def test_explicit_roles_respected():
    roles = [NodeRole.CALIBRATED, NodeRole.CALIBRATED, NodeRole.EXTRAPOLATED]
    info = resolve_node_roles([0.25, 0.5, 2.0], roles, 0.5)
    assert info.roles[2] is NodeRole.EXTRAPOLATED
    assert info.last_observable_tenor == 0.5
    assert info.roles_inferred is False


def test_last_observable_defaults_to_last_calibrated():
    roles = [NodeRole.CALIBRATED, NodeRole.EXTRAPOLATED]
    info = resolve_node_roles([0.5, 2.0], roles, None)
    assert info.last_observable_tenor == 0.5


def test_length_mismatch_raises():
    with pytest.raises(ValidationError):
        resolve_node_roles([0.25, 0.5], [NodeRole.CALIBRATED], None)


def test_curves_accept_metadata():
    from quantark.param.div.forward_carry_curve import ForwardCarryCurve
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.param.vol.vol_surface import TermStructureVolSurface

    c = ForwardCarryCurve(
        [(0.25, -0.02), (2.0, -0.2)],
        node_roles=[NodeRole.CALIBRATED, NodeRole.EXTRAPOLATED],
        last_observable_tenor=0.25,
    )
    assert c.last_observable_tenor == 0.25

    r = LinearRateCurve(
        [(0.5, 0.03), (1.0, 0.035)],
        node_roles=[NodeRole.CALIBRATED, NodeRole.CALIBRATED],
        last_observable_tenor=1.0,
    )
    assert r.tenors == [0.5, 1.0]
    assert r.node_roles == [NodeRole.CALIBRATED, NodeRole.CALIBRATED]

    v = TermStructureVolSurface(
        times=[0.5, 1.0], vols=[0.2, 0.21],
        node_roles=[NodeRole.CALIBRATED, NodeRole.CALIBRATED],
        last_observable_tenor=1.0,
    )
    assert v.last_observable_tenor == 1.0

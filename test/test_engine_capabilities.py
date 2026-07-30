import pytest

from quantark.asset.equity.engine.capabilities import (
    VolDynamicsType,
    get_engine_capability,
    validate_engine_capability,
)
from quantark.execution.errors import CapabilityError
from quantark.util.enum.engine_enums import EngineType


def test_bsm_quad_is_supported_and_term_structured():
    cap = get_engine_capability(VolDynamicsType.BSM, EngineType.QUADRATURE)

    assert cap.supported is True
    assert cap.supports_rate_term_structure is True
    assert cap.supports_carry_term_structure is True
    assert cap.supports_market_vol_term_structure is True


@pytest.mark.parametrize(
    "dynamics",
    [
        VolDynamicsType.LOCAL_VOL,
        VolDynamicsType.HESTON,
        VolDynamicsType.SLV,
    ],
)
def test_vol_model_quad_routes_are_explicitly_unsupported(dynamics):
    cap = get_engine_capability(dynamics, EngineType.QUADRATURE)

    assert cap.supported is False
    with pytest.raises(CapabilityError, match="QUAD is not supported"):
        validate_engine_capability(dynamics, EngineType.QUADRATURE)


def test_heston_pde_documents_model_vol_semantics():
    cap = get_engine_capability(VolDynamicsType.HESTON, EngineType.PDE)

    assert cap.supported is True
    assert cap.supports_rate_term_structure is True
    assert cap.supports_carry_term_structure is True
    assert cap.supports_market_vol_term_structure is False
    assert "HestonParams" in cap.notes

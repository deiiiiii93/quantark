"""Capability registry for equity model-engine combinations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quantark.execution.errors import CapabilityError
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError


class VolDynamicsType(Enum):
    BSM = "bsm"
    LOCAL_VOL = "local_vol"
    HESTON = "heston"
    SLV = "slv"


class SettlementSupport(Enum):
    """Cashflow payment-timing forms implemented by an engine."""

    NONE = "none"
    TERMINAL_ONLY = "terminal_only"
    EVENT_AND_TERMINAL = "event_and_terminal"
    AMERICAN_EXERCISE = "american_exercise"


@dataclass(frozen=True)
class EngineCapability:
    engine_type: EngineType
    dynamics_type: VolDynamicsType
    supported: bool
    supports_rate_term_structure: bool
    supports_carry_term_structure: bool
    supports_market_vol_term_structure: bool
    supports_path_dependent_payoff: bool
    notes: str
    settlement_support: SettlementSupport = SettlementSupport.NONE


def _cap(
    dynamics_type: VolDynamicsType,
    engine_type: EngineType,
    supported: bool,
    supports_market_vol_term_structure: bool,
    notes: str,
    settlement_support: SettlementSupport = SettlementSupport.NONE,
) -> EngineCapability:
    return EngineCapability(
        engine_type=engine_type,
        dynamics_type=dynamics_type,
        supported=supported,
        supports_rate_term_structure=supported,
        supports_carry_term_structure=supported,
        supports_market_vol_term_structure=supports_market_vol_term_structure,
        supports_path_dependent_payoff=supported,
        notes=notes,
        settlement_support=settlement_support,
    )


_CAPABILITIES = {
    (VolDynamicsType.BSM, EngineType.MONTE_CARLO): _cap(
        VolDynamicsType.BSM,
        EngineType.MONTE_CARLO,
        True,
        True,
        "BSM MC consumes per-step r, carry, and Black step vols.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.BSM, EngineType.PDE): _cap(
        VolDynamicsType.BSM,
        EngineType.PDE,
        True,
        True,
        "BSM PDE consumes per-step r, carry, and Black step vols.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.BSM, EngineType.QUADRATURE): _cap(
        VolDynamicsType.BSM,
        EngineType.QUADRATURE,
        True,
        True,
        "BSM QUAD consumes per-observation r, carry, and Black step vols.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.LOCAL_VOL, EngineType.MONTE_CARLO): _cap(
        VolDynamicsType.LOCAL_VOL,
        EngineType.MONTE_CARLO,
        True,
        True,
        "Local Vol MC consumes per-step r/carry and a LocalVolSurface.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.LOCAL_VOL, EngineType.PDE): _cap(
        VolDynamicsType.LOCAL_VOL,
        EngineType.PDE,
        True,
        True,
        "Local Vol PDE consumes per-step r/carry and a LocalVolSurface.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.LOCAL_VOL, EngineType.QUADRATURE): _cap(
        VolDynamicsType.LOCAL_VOL,
        EngineType.QUADRATURE,
        False,
        False,
        "Local Vol + QUAD is not supported. Current QUAD engines implement lognormal BSM transition recursion only.",
    ),
    (VolDynamicsType.HESTON, EngineType.MONTE_CARLO): _cap(
        VolDynamicsType.HESTON,
        EngineType.MONTE_CARLO,
        True,
        False,
        "Heston MC consumes per-step r/carry; volatility dynamics are HestonParams, not pricing_env.vol_surface.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.HESTON, EngineType.PDE): _cap(
        VolDynamicsType.HESTON,
        EngineType.PDE,
        True,
        False,
        "Heston PDE consumes per-step r/carry for path-dependent ADI; volatility dynamics are HestonParams.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.HESTON, EngineType.QUADRATURE): _cap(
        VolDynamicsType.HESTON,
        EngineType.QUADRATURE,
        False,
        False,
        "Heston + QUAD is not supported. Current QUAD engines implement lognormal BSM transition recursion only.",
    ),
    (VolDynamicsType.SLV, EngineType.MONTE_CARLO): _cap(
        VolDynamicsType.SLV,
        EngineType.MONTE_CARLO,
        True,
        False,
        "SLV MC consumes per-step r/carry plus HestonParams, LocalVolSurface, and LeverageSurface artifacts.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.SLV, EngineType.PDE): _cap(
        VolDynamicsType.SLV,
        EngineType.PDE,
        True,
        False,
        "SLV PDE consumes per-step r/carry plus HestonParams, LocalVolSurface, and LeverageSurface artifacts.",
        SettlementSupport.TERMINAL_ONLY,
    ),
    (VolDynamicsType.SLV, EngineType.QUADRATURE): _cap(
        VolDynamicsType.SLV,
        EngineType.QUADRATURE,
        False,
        False,
        "SLV + QUAD is not supported. Current QUAD engines implement one-dimensional lognormal BSM transition recursion only.",
    ),
}


def _normalize_dynamics(dynamics_type) -> VolDynamicsType:
    if isinstance(dynamics_type, VolDynamicsType):
        return dynamics_type
    if isinstance(dynamics_type, str):
        raw = dynamics_type.lower()
        for member in VolDynamicsType:
            if raw in {member.value, member.name.lower()}:
                return member
    raise ValidationError(f"unknown volatility dynamics type: {dynamics_type!r}")


def _normalize_engine(engine_type) -> EngineType:
    if isinstance(engine_type, tuple) and engine_type:
        engine_type = engine_type[0]
    if isinstance(engine_type, EngineType):
        return engine_type
    if isinstance(engine_type, str):
        raw = engine_type.replace(" ", "_").upper()
        try:
            return EngineType[raw]
        except KeyError as exc:
            raise ValidationError(f"unknown engine type: {engine_type!r}") from exc
    raise ValidationError(f"unknown engine type: {engine_type!r}")


def get_engine_capability(dynamics_type, engine_type) -> EngineCapability:
    dynamics = _normalize_dynamics(dynamics_type)
    engine = _normalize_engine(engine_type)
    try:
        return _CAPABILITIES[(dynamics, engine)]
    except KeyError as exc:
        raise ValidationError(
            f"no capability registered for {dynamics.value} + {engine.name}"
        ) from exc


def validate_engine_capability(dynamics_type, engine_type) -> EngineCapability:
    cap = get_engine_capability(dynamics_type, engine_type)
    if not cap.supported:
        raise CapabilityError(cap.notes)
    return cap

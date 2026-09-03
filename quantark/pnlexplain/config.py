"""PnLExplainConfig and the stencil / term tables (spec §5.5, §7.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.riskmeasures.greeks import registry
from quantark.pnlexplain.base import MARKET_FACTORS, ExplainMethod, Factor
from quantark.util.enum.engine_enums import GreeksCalculationMode
from quantark.util.enum.greek_conventions import GreekConvention
from quantark.util.exceptions import ValidationError

STENCILS = {
    "first_order": ("delta", "vega", "theta", "rho", "dividend_rho"),
    "standard": ("delta", "vega", "theta", "rho", "dividend_rho", "gamma", "volga", "vanna"),
    "extended": (
        "delta", "vega", "theta", "rho", "dividend_rho", "gamma", "volga", "vanna",
        "speed", "zomma", "charm", "color", "vega_theta", "dividend_volga", "delta_q",
    ),
}
THETA_SUBROWS: Tuple[str, ...] = ("r_theta", "q_theta", "convexity_theta", "gamma_theta")
TERM_FACTOR = {
    "delta": Factor.SPOT, "gamma": Factor.SPOT, "speed": Factor.SPOT,
    "vega": Factor.VOL, "volga": Factor.VOL, "vanna": Factor.VOL, "zomma": Factor.VOL,
    "theta": Factor.TIME, "charm": Factor.TIME, "color": Factor.TIME, "vega_theta": Factor.TIME,
    "r_theta": Factor.TIME, "q_theta": Factor.TIME, "convexity_theta": Factor.TIME,
    "gamma_theta": Factor.TIME, "theta_contract": Factor.TIME, "ledger_carry": Factor.TIME,
    "rho": Factor.RATE,
    "dividend_rho": Factor.DIVIDEND, "dividend_volga": Factor.DIVIDEND, "delta_q": Factor.DIVIDEND,
}
_INTERACTIONS = ("sequential", "shapley")
_TIME_TERMS = ("exact_gap", "per_step")
_THETA_MODES = ("estimate", "exact")
_GREEK_METHODS = ("auto", "analytical", "numerical")
_CLOCKS = (None, "1d", "1td")


def _canonical(name: object) -> str:
    """One stencil entry -> canonical registry name; clock qualifiers rejected."""
    requests = registry.normalize_greeks([name])
    (req,) = tuple(requests)
    if req.clock is not None:
        raise ValidationError(
            f"stencil entries carry no clock qualifier ({name!r}); use PnLExplainConfig.clock"
        )
    return req.canonical


@dataclass(frozen=True)
class PnLExplainConfig:
    methods: Tuple[ExplainMethod, ...] = (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)
    waterfall_order: Tuple[Factor, ...] = MARKET_FACTORS
    interaction: str = "sequential"
    spot_convention: GreekConvention = GreekConvention.STICKY_STRIKE
    stencil: Union[str, Sequence[str]] = "standard"
    time_term: str = "exact_gap"
    clock: Optional[str] = None
    theta_decomposition_mode: str = "estimate"
    bucketed: bool = False
    greeks_method: str = "auto"
    greeks_mode: GreeksCalculationMode = GreeksCalculationMode.BUMP
    params: Optional[EngineParams] = None
    _terms: Tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)
    _subrows: Tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        methods = tuple(self.methods)
        if not methods or len(set(methods)) != len(methods) or any(
            m not in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR) for m in methods
        ):
            raise ValidationError(
                "methods must be a non-empty, duplicate-free subset of {WATERFALL, TAYLOR}"
            )
        object.__setattr__(self, "methods", methods)
        order = tuple(self.waterfall_order)
        if not all(isinstance(f, Factor) for f in order) or len(order) != len(MARKET_FACTORS) \
                or set(order) != set(MARKET_FACTORS):
            raise ValidationError(
                "waterfall_order must be a permutation of the seven market Factor members"
            )
        object.__setattr__(self, "waterfall_order", order)
        if not isinstance(self.bucketed, bool):
            raise ValidationError(f"bucketed must be a bool, got {self.bucketed!r}")
        if self.interaction not in _INTERACTIONS:
            raise ValidationError(f"interaction must be one of {_INTERACTIONS}")
        if self.time_term not in _TIME_TERMS:
            raise ValidationError(f"time_term must be one of {_TIME_TERMS}")
        if self.theta_decomposition_mode not in _THETA_MODES:
            raise ValidationError(f"theta_decomposition_mode must be one of {_THETA_MODES}")
        if self.greeks_method not in _GREEK_METHODS:
            raise ValidationError(f"greeks_method must be one of {_GREEK_METHODS}")
        if self.clock not in _CLOCKS:
            raise ValidationError("clock must be None, '1d' or '1td'")
        if self.clock is not None and self.time_term != "per_step":
            raise ValidationError("clock is only meaningful with time_term='per_step'")
        if not isinstance(self.spot_convention, GreekConvention):
            raise ValidationError("spot_convention must be a GreekConvention")
        if self.spot_convention not in (GreekConvention.STICKY_STRIKE, GreekConvention.STICKY_MONEYNESS):
            raise ValidationError("spot_convention must be STICKY_STRIKE or STICKY_MONEYNESS")
        if not isinstance(self.greeks_mode, GreeksCalculationMode):
            raise ValidationError("greeks_mode must be a GreeksCalculationMode")
        terms, subrows = self._resolve_stencil()
        object.__setattr__(self, "_terms", terms)
        object.__setattr__(self, "_subrows", subrows)

    def _resolve_stencil(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        if isinstance(self.stencil, str):
            if self.stencil not in STENCILS:
                raise ValidationError(f"unknown stencil {self.stencil!r}; known {tuple(STENCILS)}")
            terms = STENCILS[self.stencil]
            subrows = ("r_theta", "q_theta", "convexity_theta") + (
                ("gamma_theta",) if self.stencil == "extended" else ()
            )
            return terms, subrows
        names = [_canonical(n) for n in self.stencil]
        if len(set(names)) != len(names):
            raise ValidationError(f"duplicate stencil entries after alias resolution: {names}")
        unknown = [n for n in names if n not in TERM_FACTOR or n in ("theta_contract", "ledger_carry")]
        if unknown:
            raise ValidationError(f"stencil entries are not Taylor terms: {unknown}")
        subrows = tuple(n for n in names if n in THETA_SUBROWS)
        terms = tuple(n for n in names if n not in THETA_SUBROWS)
        if subrows and "theta" not in terms:
            raise ValidationError("theta sub-rows require 'theta' in the stencil")
        return terms, subrows


def resolve_stencil(config: PnLExplainConfig) -> Tuple[str, ...]:
    return config._terms


def resolved_subrows(config: PnLExplainConfig) -> Tuple[str, ...]:
    return config._subrows

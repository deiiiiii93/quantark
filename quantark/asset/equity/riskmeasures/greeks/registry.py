"""Single source of truth for scalar greek names.

Every requestable greek is a ``GreekDef`` here; request validation, alias
resolution, the ``greeks=None`` default set, the analytical auto-routing
set, and linear-product values all derive from this one table, so a name
can no longer be requestable-but-uncomputed (the charm/color silent-miss
class of bug).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Optional, Sequence, Set, Tuple

from quantark.util.enum import CommonGreek, EquityGreek
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class GreekDef:
    """One requestable scalar greek.

    numerical: bump-based implementation ``(calc, ctx) -> float`` used by the
        registry-driven part of dispatch. Derived names (theta components) and
        names whose dispatch is hand-ordered in the facade may leave it None;
        the completeness test in test_greeks_registry.py is the guard that
        every requestable name actually produces a value.
    """

    name: str
    aliases: Tuple[str, ...] = ()
    analytical_auto: bool = False
    default: bool = False
    linear_value: float = 0.0
    supports_clock: bool = False
    requires: Tuple[str, ...] = ()
    numerical: Optional[Callable] = field(default=None, compare=False)


@dataclass(frozen=True)
class GreekRequest:
    """A normalized request entry: output key + canonical name + clock."""

    key: str
    canonical: str
    clock: Optional[str] = None  # None | "1d" | "1td"


_DEFS = (
    GreekDef("price", default=True, analytical_auto=True),
    GreekDef("delta", default=True, analytical_auto=True, linear_value=1.0),
    GreekDef("gamma", default=True, analytical_auto=True),
    GreekDef("vega", default=True, analytical_auto=True),
    GreekDef("theta", default=True, analytical_auto=True, supports_clock=True),
    GreekDef("rho", default=True, analytical_auto=True),
    GreekDef(
        "dividend_rho",
        aliases=("rhoq", "div_rho", "dividendrho"),
        default=True,
    ),
    GreekDef("vanna"),
    GreekDef("volga"),
    GreekDef(
        "delta_q",
        aliases=("deltaq", "deltadq", "d_delta_d_q", "d_delta_dq"),
    ),
    # New names carry analytical_auto=True: they never had an incumbent
    # route, so auto-routing them to closed forms for vanillas breaks
    # nothing (vanna/volga/delta_q keep their incumbent numerical routing).
    GreekDef("charm", supports_clock=True, analytical_auto=True),
    GreekDef("color", supports_clock=True, analytical_auto=True),
    GreekDef("speed", analytical_auto=True),
    GreekDef("zomma", analytical_auto=True),
    GreekDef("dividend_volga", analytical_auto=True),
    GreekDef(
        "vega_theta",
        aliases=("veta",),
        supports_clock=True,
        analytical_auto=True,
    ),
    GreekDef(
        "gamma_theta",
        supports_clock=True,
        requires=("gamma",),
        analytical_auto=True,
    ),
    GreekDef(
        "convexity_theta",
        default=True,
        supports_clock=True,
        requires=("theta", "rho", "dividend_rho"),
    ),
    GreekDef(
        "r_theta",
        default=True,
        supports_clock=True,
        requires=("theta", "rho", "dividend_rho"),
    ),
    GreekDef(
        "q_theta",
        default=True,
        supports_clock=True,
        requires=("theta", "rho", "dividend_rho"),
    ),
)

REGISTRY: Dict[str, GreekDef] = {d.name: d for d in _DEFS}

ALIASES: Dict[str, str] = {
    alias: d.name for d in _DEFS for alias in d.aliases
}

DEFAULT_SET: FrozenSet[str] = frozenset(d.name for d in _DEFS if d.default)

ANALYTICAL_AUTO_SET: FrozenSet[str] = frozenset(
    d.name for d in _DEFS if d.analytical_auto
)


def _resolve_name(greek: object) -> str:
    """One request entry -> canonical registry name (aliases resolved)."""
    if isinstance(greek, (CommonGreek, EquityGreek)):
        name = greek.value
    elif isinstance(greek, str):
        name = greek.strip().lower()
    else:
        raise ValidationError(
            f"Unsupported greek identifier type: {type(greek).__name__}"
        )
    return ALIASES.get(name, name)


def _split_clock(name: str) -> Tuple[str, Optional[str]]:
    """Strip a trailing clock qualifier: theta_1td -> (theta, "1td")."""
    if name.endswith("_1td"):
        return name[:-4], "1td"
    if name.endswith("_1d"):
        return name[:-3], "1d"
    return name, None


def normalize_greeks(
    greeks: Optional[Sequence[object]],
) -> Optional[Set[GreekRequest]]:
    """Normalize a request list against the registry.

    Returns None for a None request (meaning: the default set), an empty set
    for an explicitly empty request, and otherwise one GreekRequest per
    distinct requested name. Clock qualifiers (``theta_1d`` / ``theta_1td``)
    are accepted only on greeks with ``supports_clock``; the output key uses
    the canonical spelling plus the qualifier (``veta_1td`` ->
    ``vega_theta_1td``).
    """
    if greeks is None:
        return None
    if len(greeks) == 0:
        return set()
    normalized: Set[GreekRequest] = set()
    for greek in greeks:
        name = _resolve_name(greek)
        if name in REGISTRY:
            normalized.add(GreekRequest(key=name, canonical=name))
            continue
        base, clock = _split_clock(name)
        base = ALIASES.get(base, base)
        if clock is not None and base in REGISTRY:
            if not REGISTRY[base].supports_clock:
                raise ValidationError(
                    f"Greek {base!r} does not support the {clock!r} clock "
                    f"qualifier (requested: {name!r})"
                )
            normalized.add(
                GreekRequest(key=f"{base}_{clock}", canonical=base, clock=clock)
            )
            continue
        raise ValidationError(f"Unknown greek name: {name}")
    return normalized

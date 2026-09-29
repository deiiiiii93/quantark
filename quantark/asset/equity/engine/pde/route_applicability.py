"""Per-route applicability of the shared PDEParams surface (spec §4.1).

One params object, four route contracts. Shared-grid solvers honour every
field (no entry here). Every other route lists the fields it cannot honour:

- INAPPLICABLE: the route's kernel has no such control. The preset default
  is accepted with no effect; an explicit override raises.
- REJECTED: an explicit value always raises (``allowed`` lists explicit
  values the route does honour, e.g. DCN's NODAL).

A field not listed for a route is honoured or vacuous there. "Explicit" is a
non-None ``grid.*`` overlay field, or a scalar away from its dataclass
default. The check runs once, before any grid is built. Review note: this
table is compared verbatim against spec §4.1.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Tuple

from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.exceptions import ValidationError

INAPPLICABLE = "inapplicable"
REJECTED = "rejected"
_LIFT = "sub-project 3 moves this route onto the shared core and lifts the limit"


@dataclass(frozen=True)
class Rule:
    """One field a route cannot honour, and why."""

    field: str  # "grid.<name>" or a PDEParams field name
    status: str  # INAPPLICABLE | REJECTED
    reason: str
    allowed: Tuple[Any, ...] = ()


_LV_DOMAIN = (
    Rule("grid.eps_crit", INAPPLICABLE, "the kernel's S-grid is uniform (no concentration)"),
    Rule(
        "grid.num_std",
        INAPPLICABLE,
        "the kernel sets its own domain, [0, 4 x max(spot, strike)]",
    ),
    Rule(
        "grid.bounds",
        REJECTED,
        "the kernel pins 0 below and grows the top by strike/barrier alignment, "
        "so there is no hard edge to honour",
    ),
)
_NO_DAMPING = "the local-vol barrier kernel has no damping schedule"
_DCN_CADENCE = (
    "DCN's time resolution is its observation schedule (one step per "
    "observation interval)"
)

ROUTE_RULES = {
    "local-vol vanilla": _LV_DOMAIN
    + (
        Rule(
            "grid.terminal_damping_steps",
            REJECTED,
            "the kernel damps with one two-half-step startup (1) or none (0); a "
            "count of 2 or more cannot be applied",
            allowed=(0, 1),
        ),
    ),
    "local-vol barrier": _LV_DOMAIN
    + (
        Rule("grid.terminal_damping_steps", INAPPLICABLE, _NO_DAMPING),
        Rule("grid.event_damping_steps", INAPPLICABLE, _NO_DAMPING),
        Rule("event_theta", INAPPLICABLE, _NO_DAMPING),
        Rule(
            "event_projection",
            REJECTED,
            "the route's event treatment is a mid-cell-aligned nodal mask; leave "
            "event_projection=None",
        ),
    ),
    "DCN": (
        Rule("grid.steps_per_day", REJECTED, _DCN_CADENCE),
        Rule("grid.time_steps", REJECTED, _DCN_CADENCE),
        Rule(
            "event_projection",
            REJECTED,
            "DCN's events are nodal masks",
            allowed=(EventProjectionMode.NODAL,),
        ),
    ),
}


def _explicit(params, field: str):
    """The explicit value of ``field`` on ``params``, or None when not set."""
    if field.startswith("grid."):
        grid = getattr(params, "grid", None)
        return None if grid is None else getattr(grid, field[len("grid."):])
    value = getattr(params, field)
    default = next(f.default for f in dataclasses.fields(params) if f.name == field)
    return None if value == default else value


def check_route_applicability(params, route: str) -> None:
    """Raise ``ValidationError`` for the first field ``route`` cannot honour."""
    for rule in ROUTE_RULES[route]:
        value = _explicit(params, rule.field)
        if value is None or value in rule.allowed:
            continue
        raise ValidationError(
            f"PDEParams {rule.field}={value!r} is {rule.status} on the {route} "
            f"route: {rule.reason}. {_LIFT}."
        )

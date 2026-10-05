"""Per-route applicability of the shared PDEParams surface (spec §4.1; spec 3C §5.7).

One params object. Every 1D solver marches on the shared core and honours
every field (no entry here). One route keeps a contract: the 2D ADI solvers
(``route = "2D"``), which list the fields they cannot honour:

- INAPPLICABLE: the route has no such control. The preset default is
  accepted with no effect; an explicit override raises.
- REJECTED: an explicit value always raises (``allowed`` lists explicit
  values the route does honour).

A field not listed is honoured or vacuous there. "Explicit" is a non-None
``grid.*`` overlay field, a ``stepper.*`` setting away from the built-in
``CrankNicolson()``'s, or a scalar away from its dataclass default. The check
runs once, before any grid is built.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Tuple

from quantark.util.exceptions import ValidationError

INAPPLICABLE = "inapplicable"
REJECTED = "rejected"
_LIFT = "These are the 2D route's limits; no spec is committed to moving this route onto the shared core"


@dataclass(frozen=True)
class Rule:
    """One field a route cannot honour, and why."""

    field: str  # "grid.<name>" or a PDEParams field name
    status: str  # INAPPLICABLE | REJECTED
    reason: str
    allowed: Tuple[Any, ...] = ()


ROUTE_RULES = {
    "2D": (
        Rule(
            "stepper.damping",
            REJECTED,
            "the 2D ADI solvers damp in whole steps; damping='half_step' cannot be applied",
            allowed=("full_step",),
        ),
    ),
}


def _explicit(params, field: str):
    """The explicit value of ``field`` on ``params``, or None when not set."""
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson

    if field.startswith("stepper."):
        name = field[len("stepper."):]
        stepper = params.stepper
        if type(stepper) is not CrankNicolson:
            return None      # a custom stepper is used as given
        value = getattr(stepper, name)
        return None if value == getattr(CrankNicolson(), name) else value
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

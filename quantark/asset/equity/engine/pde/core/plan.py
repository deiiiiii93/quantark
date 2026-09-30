"""``explain``: the resolved plan of one preparation, without the march (spec 2 §4.7).

``build_plan`` reads every section from a ``PreparedProblem`` -- nothing is
recomputed -- so the plan describes the problem ``price`` marches. Each section
is a read-only mapping that also allows attribute access (``plan.time.actual_steps``);
node lists and arrays appear only in ``to_dict()``.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .identity import plain
from .prepare import CompositeProblem, Operation, OutputRequest

__all__ = ["PDEPlan", "build_plan"]

_SECTIONS = ("route", "settings", "space", "time", "events", "scheme", "outputs", "identity", "terminal")
_TERMINAL_OPS = (
    (Operation.PRICE, OutputRequest(Operation.PRICE)),
    (Operation.GREEKS, OutputRequest(Operation.GREEKS, greeks=True)),
    (Operation.PRICE_WITH_EVENTS, OutputRequest(Operation.PRICE_WITH_EVENTS, event_stats=True)),
    (Operation.EVENT_STATS, OutputRequest(Operation.EVENT_STATS, event_stats=True)),
    (Operation.LIFE_SURFACE, OutputRequest(Operation.LIFE_SURFACE, life_surface=True)),
)


class Section(Mapping):
    """A read-only mapping with attribute access to its keys."""

    def __init__(self, data=None):
        object.__setattr__(self, "_data", dict(data or {}))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("a plan section is read-only")

    def __repr__(self) -> str:
        return f"Section({self._data!r})"


@dataclass(frozen=True, eq=False)
class PDEPlan:
    """What one preparation resolved: route, settings, grid, events, scheme, outputs, identity."""

    route: Section
    settings: Section
    space: Section
    time: Section
    events: Section
    scheme: Section
    outputs: Section
    identity: Section
    terminal: Section
    layout_bound: bool
    parts: tuple = ()
    _detail: Mapping = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """Every section as a plain dict, with the node lists and arrays ``__str__`` omits."""
        out = {}
        for name in _SECTIONS:
            section = dict(getattr(self, name))
            section.update(self._detail.get(name, {}))
            out[name] = section
        out["layout_bound"] = self.layout_bound
        out["parts"] = [p.to_dict() for p in self.parts]
        return out

    def __str__(self) -> str:
        lines = []
        for name in _SECTIONS:
            section = getattr(self, name)
            lines.append(f"[{name}]")
            width = max((len(str(k)) for k in section), default=0)
            for key, value in section.items():
                lines.append(f"  {str(key).ljust(width)}  {value}")
        for i, part in enumerate(self.parts):
            lines.append(f"[part {i}]")
            lines.extend("  " + line for line in str(part).splitlines())
        return "\n".join(lines)


# --- sections ---------------------------------------------------------------------

def _terminal(solver, product, env, lifecycle_state) -> Section:
    """Per operation: the valuation-date outcome's reason, "marches", or what raises."""
    adapter = solver.adapter
    out = {}
    for op, request in _TERMINAL_OPS:
        try:
            adapter.validate(product, env, request, lifecycle_state)
            if adapter.composite(product, env, request, lifecycle_state) is not None:
                out[op.value] = "composes"
                continue
            outcome = adapter.terminal_outcome(product, env, request, lifecycle_state)
            out[op.value] = "marches" if outcome is None else outcome.reason
        except Exception as exc:          # the plan reports what the operation would raise
            out[op.value] = f"raises {type(exc).__name__}"
    return Section(out)


def _settings(solver) -> Section:
    """Per PDEParams field and per GridConfig field: requested, effective, source, applicability."""
    params = solver.params
    overlay = getattr(params, "grid", None)
    effective = solver.grid_binder.config
    rules = {}
    route = getattr(solver.adapter, "route", None)
    if route is not None:
        from quantark.asset.equity.engine.pde.route_applicability import ROUTE_RULES
        rules = {r.field: r.status for r in ROUTE_RULES[route]}
    from .components import CentralDifference, CrankNicolson, StandardReadout

    builtins = {"stepper": CrankNicolson(), "operator": CentralDifference(), "readout": StandardReadout()}
    out = {}
    for f in dataclasses.fields(params):
        if f.name == "grid":
            continue
        value = getattr(params, f.name)
        if f.name in builtins:
            # a component slot: preset is the built-in; a custom class is judged by the
            # route's component rule, the built-in stepper's settings one by one below
            builtin = builtins[f.name]
            custom = type(value) is not type(builtin)
            key = "component.stepper_class" if f.name == "stepper" else f"component.{f.name}"
            out[f.name] = {"requested": type(value).__name__, "effective": type(value).__name__,
                           "source": "preset" if value == builtin else "overlay",
                           "applicability": rules.get(key, "honoured") if custom else "honoured"}
            if f.name == "stepper" and not custom:
                resolved = solver.adapter.stepper(params)
                for s in dataclasses.fields(value):
                    requested = getattr(value, s.name)
                    out[f"stepper.{s.name}"] = {
                        "requested": plain(requested), "effective": plain(getattr(resolved, s.name)),
                        "source": "preset" if requested == getattr(builtin, s.name) else "overlay",
                        "applicability": rules.get(f"stepper.{s.name}", "honoured")}
            continue
        default = f.default if f.default is not dataclasses.MISSING else None
        out[f.name] = {"requested": plain(value), "effective": plain(value),
                       "source": "preset" if value == default else "overlay",
                       "applicability": rules.get(f.name, "honoured")}
    for f in dataclasses.fields(effective):
        requested = None if overlay is None else getattr(overlay, f.name)
        out[f"grid.{f.name}"] = {"requested": plain(requested), "effective": plain(getattr(effective, f.name)),
                                 "source": "preset" if requested is None else "overlay",
                                 "applicability": rules.get(f"grid.{f.name}", "honoured")}
    return Section(out)


def _space(prepared) -> tuple:
    layout, config = prepared.layout, prepared.solver.grid_binder.config
    spatial, request = layout.spatial, layout.request
    bounds = getattr(config, "bounds", None) or (None, None)

    def rule(hard, configured):
        if hard is not None:
            return "hard edge"
        if configured is not None:
            return "bounds"
        return "envelope"

    section = {
        "points": int(len(spatial.x)),
        "bounds": tuple(float(b) for b in spatial.bounds),
        "rule": (rule(request.hard_lower, bounds[0]), rule(request.hard_upper, bounds[1])),
        "achieved_eps": float(spatial.achieved_eps),
        "active_critical_prices": tuple(float(p) for p in spatial.active_critical_prices),
        "ignored_critical_prices": tuple(float(p) for p in spatial.ignored_critical_prices),
    }
    return Section(section), {"x": np.asarray(spatial.x), "s": np.asarray(spatial.s)}


def _time(prepared) -> tuple:
    layout, config = prepared.layout, prepared.solver.grid_binder.config
    time, request = layout.time, layout.request
    if config.time_steps is not None:
        basis = {"basis": "time_steps", "value": int(config.time_steps)}
    else:
        basis = {"basis": "steps_per_day", "value": float(config.steps_per_day),
                 "day_count": plain(config.day_count)}
    section = dict(basis,
                   requested_steps=int(time.requested_steps), actual_steps=int(time.actual_steps),
                   fill_scaled=bool(time.fill_scaled), cap_exceeded=bool(time.cap_exceeded),
                   mandatory_nodes=len(request.event_times) + len(request.extra_times))
    detail = {"t": np.asarray(time.t), "dt": np.asarray(time.dt), "event_times": tuple(request.event_times),
              "extra_times": tuple(request.extra_times), "request": request}
    return Section(section), detail


def _events(prepared) -> Section:
    events = prepared.events
    if events is None:
        return Section({"schedule": "applied inline by the family's march", "rows": ()})
    rows = events.evidence() if hasattr(events, "evidence") else plain(events)
    return Section({"schedule": type(events).__name__, "rows": rows})


def _scheme(prepared) -> tuple:
    stepper, schedule, coeffs = prepared.stepper, prepared.theta_schedule, prepared.coefficients
    event_damp, term_damp = stepper._damping_sets(prepared.layout)
    values, counts = np.unique(np.asarray(schedule, dtype=float), return_counts=True)
    zero = sum(1 for j in range(len(schedule)) if coeffs.is_zero_diffusion(coeffs.set_at(j)))
    section = {"stepper": type(stepper).__name__, "resolved": dict(stepper.evidence()),
               "terminal_damped_steps": len(term_damp), "event_damped_steps": len(event_damp),
               "theta_counts": {float(v): int(c) for v, c in zip(values, counts)},
               "zero_diffusion_steps": int(zero)}
    return Section(section), {"theta_by_step": np.asarray(schedule, dtype=float)}


def _outputs(prepared) -> Section:
    groups = []
    for group in prepared.groups:
        stats = prepared.statistics is not None and group is prepared.statistics.group
        columns = (2 * prepared.statistics.n_cols if stats
                   else 1 if group.one_dimensional_rhs else len(group.surfaces))
        groups.append({"name": group.name, "surfaces": tuple(group.surfaces), "columns": columns,
                       "backend": group.backend.value, "retention": "rolling" if group.rolling else "full"})
    return Section({"groups": tuple(groups), "readout": prepared.adapter.readout_mode(prepared)})


def _identity(prepared) -> Section:
    from quantark.execution.cache.fingerprint import try_fingerprint

    solver = prepared.solver
    cls = type(solver)
    grid_artefact = try_fingerprint(("pde-grid", f"{cls.__module__}.{cls.__qualname__}", prepared.grid_request,
                                     prepared.market, solver.grid_binder.config.key))
    ckey = prepared.identity.get("coefficient_key")
    problem = prepared.identity.get("problem")
    return Section({"layout": prepared.identity.get("layout"), "grid_artefact": grid_artefact,
                    "coefficient_key": None if ckey is None else try_fingerprint(ckey),
                    "factorization": prepared.identity.get("factorization"), "problem": problem,
                    "fingerprintable": problem is not None})


def _route(prepared) -> Section:
    adapter = prepared.adapter
    return Section({"solver": type(prepared.solver).__name__, "adapter": type(adapter).__name__,
                    "samples": adapter.sample_kind.value, "solve_state": plain(prepared.solve_state)})


# --- the plan ---------------------------------------------------------------------

def build_plan(problem, solver, product, env, lifecycle_state=None) -> PDEPlan:
    """The plan ``solver`` resolved for ``product`` in ``env`` (a composite: one plan per part)."""
    terminal = _terminal(solver, product, env, lifecycle_state)
    if isinstance(problem, CompositeProblem):
        parts = tuple(build_plan(p, p.solver, p.product, p.env, p.lifecycle_state) for p in problem.parts)
        return PDEPlan(
            route=Section({"solver": type(solver).__name__, "adapter": type(solver.adapter).__name__,
                           "composition": problem.description}),
            settings=_settings(solver), space=Section(), time=Section(), events=Section(),
            scheme=Section(), outputs=Section(), identity=Section(), terminal=terminal,
            layout_bound=any(p.layout_bound for p in parts), parts=parts)
    if problem.terminal is not None:
        return PDEPlan(
            route=Section({"solver": type(solver).__name__, "adapter": type(problem.adapter).__name__,
                           "terminal": problem.terminal.reason}),
            settings=_settings(solver), space=Section(), time=Section(), events=Section(), scheme=Section(),
            outputs=Section(), identity=Section(), terminal=terminal, layout_bound=False)
    space, space_detail = _space(problem)
    time, time_detail = _time(problem)
    scheme, scheme_detail = _scheme(problem)
    return PDEPlan(route=_route(problem), settings=_settings(solver), space=space, time=time,
                   events=_events(problem), scheme=scheme, outputs=_outputs(problem),
                   identity=_identity(problem), terminal=terminal, layout_bound=True,
                   _detail={"space": space_detail, "time": time_detail, "scheme": scheme_detail})

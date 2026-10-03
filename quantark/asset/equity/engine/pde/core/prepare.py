"""The one preparation path (spec 2 §5.1, §5.2): ``prepare -> run``.

``prepare`` resolves everything a march consumes -- the outcome that needs no
grid, the layout, the coefficients, the stepper's schedule, the column groups
and a fresh factorization cache -- through the solver's ``FamilyAdapter``.
``run`` marches a prepared problem and reads it, returns a terminal outcome
as is, and composes a composite problem from its parts.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from collections import abc
from typing import Any, Callable, Mapping, Optional

import numpy as np

from quantark.asset.equity.engine.pde.grid import resolve_bound_layout

from .backends import FactorizationCache
from .components import validate_schedule
from .identity import factorization_key, market_fingerprints, problem_identity
from .march import march          # run resolves it here: a spy on core.prepare.march sees every entry
from .readout import read
from .statistics import create_statistics

__all__ = ["Operation", "OutputRequest", "TerminalOutcome", "SessionArtefacts", "PreparedProblem",
           "CompositeProblem", "prepare", "run", "market_scope", "request_geometry", "bind_layout",
           "session_step_sets"]


class Operation(Enum):
    PRICE = "price"
    GREEKS = "greeks"
    PRICE_WITH_EVENTS = "price_with_events"
    EVENT_STATS = "event_stats"
    LIFE_SURFACE = "life_surface"
    SESSION_OUTPUTS = "session_outputs"


@dataclass(frozen=True)
class OutputRequest:
    operation: Operation
    price: bool = True
    greeks: bool = False
    event_stats: bool = False
    streams: Optional[frozenset] = None
    grid: bool = False
    life_surface: bool = False
    extra_times: tuple = ()
    external_pv: Optional[float] = None
    #: False: no valuation-date short-circuit (a spot curve reads the solved surface itself)
    short_circuit: bool = True


@dataclass(frozen=True)
class TerminalOutcome:
    """An outcome known without a grid (expired, knocked out or touched at valuation)."""
    reason: str
    outputs: Any


@dataclass(frozen=True, eq=False)
class SessionArtefacts:
    """What a prepared execution session built for its engine clone (spec 2 §5.10).

    ``layout`` is the Layout ``bind_layout`` bound for the session's product;
    ``coefficients`` the pre-flat-exact step sets (``session_step_sets``); the
    packs the warmed factorizations (``march.warm_factorizations``), consulted
    before the march's own cache.
    """
    layout: Any = None
    coefficients: Any = None
    lu_pack: Optional[Mapping] = None
    banded_pack: Optional[Mapping] = None


@dataclass(frozen=True, eq=False)
class PreparedProblem:
    """Everything one march consumes. With ``terminal`` set the geometry fields are ``None``."""
    solver: Any
    adapter: Any
    product: Any
    env: Any
    request: OutputRequest
    lifecycle_state: Any
    tau: Optional[float]
    terminal: Optional[TerminalOutcome]
    solve_state: Mapping[str, Any]
    market: Any = None
    grid_request: Any = None
    layout: Any = None
    samples: Optional[Callable[[], Any]] = None
    coefficients: Any = None
    stepper: Any = None
    theta_schedule: Optional[np.ndarray] = None
    groups: tuple = ()
    factorizations: Optional[FactorizationCache] = None
    readout: Any = None
    events: Any = ()                  # what the adapter's prepare_events built (the snowball event table)
    identity: Optional[Mapping[str, Optional[str]]] = None
    market_fps: Optional[Mapping[str, Optional[str]]] = None
    statistics: Any = None            # the statistics group's columns, events and readout (§5.4)

    @classmethod
    def terminal_only(cls, solver, adapter, product, env, request, lifecycle_state, tau, terminal):
        return cls(solver=solver, adapter=adapter, product=product, env=env, request=request,
                   lifecycle_state=lifecycle_state, tau=tau, terminal=terminal, solve_state={},
                   identity={})


@dataclass(frozen=True, eq=False)
class CompositeProblem:
    """A result composed from independently prepared problems (decision 21)."""
    parts: tuple
    compose: Callable[[tuple], Any]
    request: OutputRequest
    description: str


class LazyIdentity(abc.Mapping):
    """``PreparedProblem.identity``: plain values, or zero-argument thunks evaluated
    once on first access (the identities a march never reads)."""

    def __init__(self, values):
        self._values = dict(values)

    def update(self, **thunks):
        self._values.update(thunks)

    def __getitem__(self, key):
        value = self._values[key]
        if callable(value):
            value = self._values[key] = value()
        return value

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _lazy(fn):
    """A zero-argument callable that evaluates ``fn`` once and caches the result."""
    cell = []

    def get():
        if not cell:
            cell.append(fn())
        return cell[0]

    return get


@contextmanager
def market_scope(solver, env, fps):
    """The environment's fingerprints for the memo lookups inside (decision 14).

    The solver's discount and coefficient memos read them through
    ``BasePDESolver._market_fps``; an outer scope is restored on exit.
    """
    outer = getattr(solver, "_market_fp", None)
    solver._market_fp = (env, fps, {})   # the third slot: memos resolved inside this scope
    try:
        yield
    finally:
        solver._market_fp = outer


def _geometry(adapter, solver, product, env, request, tau):
    """Steps 2-4 inside the market scope: (solve state, market snapshot, grid request)."""
    solve_state = adapter.solve_state(product, env, request)              # step 2
    adapter.market_checks(product, env, tau)                              # _solve's guard, before the grid
    market = solver.market_snapshot(product, env)                         # step 3
    grid_request = adapter.grid_request(product, market, tau, request)    # step 4
    return solve_state, market, grid_request


def _session_layout(solver, stored, grid_request):
    """The session's layout for ``grid_request`` (step 5).

    The stored layout itself when it was bound for this request. Otherwise (a
    knock-in's knock-out leg) the request's time layout around the stored spatial
    grid, verified node for node: a stale artefact raises instead of silently
    changing the discretization.
    """
    if stored.request == grid_request:
        return stored
    if (stored.request.hard_lower, stored.request.hard_upper) != (grid_request.hard_lower,
                                                                 grid_request.hard_upper):
        from quantark.util.exceptions import PricingError

        raise PricingError(
            "the session's layout was bound for hard bounds "
            f"({stored.request.hard_lower}, {stored.request.hard_upper}) but this solve requires "
            f"({grid_request.hard_lower}, {grid_request.hard_upper}); re-prepare the session"
        )
    from quantark.asset.equity.engine.pde.grid import Layout
    from quantark.asset.equity.engine.pde.grid.space import SpatialLayout
    from quantark.asset.equity.engine.pde.grid.time import build_time
    from quantark.util.exceptions import PricingError

    config = solver.grid_binder.config
    time_layout = build_time(grid_request, config)
    if not np.array_equal(time_layout.t, stored.time.t):
        raise PricingError(
            "injected session grids diverge from the declarative layer "
            "time geometry; re-prepare the session"
        )
    s_vec = stored.spatial.s
    spatial = SpatialLayout(
        s=s_vec,
        x=stored.spatial.x,
        dx=stored.spatial.dx,
        bounds=(float(s_vec[0]), float(s_vec[-1])),
        achieved_eps=float("nan"),
        active_critical_prices=tuple(
            float(p)
            for p in grid_request.critical_prices
            if float(s_vec[0]) < float(p) < float(s_vec[-1])
        ),
        ignored_critical_prices=(),
    )
    return Layout(spatial=spatial, time=time_layout, request=grid_request, config_key=config.key)


def request_geometry(solver, product, env, request):
    """``prepare`` steps 0-4 for the product itself: (grid request, market snapshot),
    or ``None`` when the request has a terminal outcome.

    No composite: a knock-in binds its own request, whose layout its knock-out
    leg then reuses (frozen) or is assembled around (session), as before.
    """
    adapter = solver.adapter
    adapter.validate(product, env, request, None)
    if adapter.terminal_outcome(product, env, request, None) is not None:
        return None
    tau = float(product.get_maturity(env))
    adapter.check_route(solver.params)
    with market_scope(solver, env, market_fingerprints(env)):
        _, market, grid_request = _geometry(adapter, solver, product, env, request, tau)
    return grid_request, market


def bind_layout(solver, product, env, request, *, frozen=None):
    """``prepare`` steps 0-5: the Layout a march of ``request`` binds, or ``None``
    when the request is terminal (spec 2 §5.10). Bump contexts freeze it; sessions
    store it."""
    geometry = request_geometry(solver, product, env, request)
    if geometry is None:
        return None
    grid_request, market = geometry
    return resolve_bound_layout(solver.grid_binder, frozen, grid_request, market)


def session_step_sets(solver, product, env, layout):
    """The pre-flat-exact step sets ``prepare`` builds on ``layout`` (the session's
    step-coefficient artefact): the adapter's ``step_sets`` in the solve's scopes."""
    adapter = solver.adapter
    tau = float(product.get_maturity(env))
    with market_scope(solver, env, market_fingerprints(env)), adapter.scope(product, env):
        return adapter.step_sets(product, env, layout, tau)


def prepare(solver, product, env, request, *, lifecycle_state=None, frozen=None, session=None):
    """The one preparation path (spec 2 §5.2). explain() is this function without the march."""
    adapter = solver.adapter
    adapter.validate(product, env, request, lifecycle_state)              # the facades' raising checks
    composite = adapter.composite(product, env, request, lifecycle_state)
    if composite is not None:                                             # decision 21
        return composite
    terminal = adapter.terminal_outcome(product, env, request, lifecycle_state)   # step 0
    if terminal is not None:
        return PreparedProblem.terminal_only(solver, adapter, product, env, request, lifecycle_state,
                                             None, terminal)
    tau = float(product.get_maturity(env))
    adapter.check_route(solver.params)                                    # step 1
    fps = market_fingerprints(env)                                        # decision 14: once per prepare
    with market_scope(solver, env, fps), adapter.scope(product, env):
        solve_state, market, grid_request = _geometry(adapter, solver, product, env, request, tau)
        if session is not None and session.layout is not None:           # step 5
            layout = _session_layout(solver, session.layout, grid_request)
        else:
            layout = resolve_bound_layout(solver.grid_binder, frozen, grid_request, market)
        adapter.observation_maps(product, env, layout, tau)               # step 6
        # step 7: the event stages -- for the value surfaces and for a statistics group alike (decision 14):
        # the statistics-only path (an external pv, the 2D surrogate) marches no value group but needs them
        marches_statistics = request.event_stats and adapter.supports_statistics_group
        events = (adapter.prepare_events(product, env, layout, tau, request)
                  if (request.price or marches_statistics) else None)
        samples = _lazy(lambda: adapter.samples(product, env, layout, tau))   # step 8
        coefficients = adapter.coefficients(product, env, layout, tau, samples, session)
        stepper = adapter.stepper(solver.params, product)                 # step 9
        schedule = stepper.schedule(layout)
        validate_schedule(schedule, layout, stepper)
        groups = adapter.groups(product, layout, request) if request.price else ()   # step 10
        factorizations = adapter.factorizations(session)                  # step 11
        identity = LazyIdentity({"layout": layout.fingerprint,            # step 12
                                 "coefficient_key": adapter.coefficient_key(product, env, layout, fps)})
        problem = PreparedProblem(solver=solver, adapter=adapter, product=product, env=env, request=request,
                                  lifecycle_state=lifecycle_state, tau=tau, terminal=None,
                                  solve_state=solve_state, market=market, grid_request=grid_request,
                                  layout=layout, samples=samples, coefficients=coefficients, stepper=stepper,
                                  theta_schedule=schedule, groups=groups, factorizations=factorizations,
                                  readout=solver.params.readout, events=events, identity=identity, market_fps=fps)
        if request.event_stats and adapter.supports_statistics_group:     # the statistics group (§5.4)
            statistics = create_statistics(problem, request.streams)
            if statistics is not None:
                problem = replace(problem, statistics=statistics, groups=groups + (statistics.group,))
        # step 12: what the march's factorizations and the march itself depend on. Read
        # by explain, sessions and tests, never by the march: computed on first access.
        final = problem
        identity.update(factorization=lambda: factorization_key(final),
                        problem=lambda: problem_identity(final))
    return problem


def run(problem):
    """March and read a prepared problem; a terminal outcome as is; a composite from its parts."""
    if isinstance(problem, CompositeProblem):
        return problem.compose(tuple(run(p) for p in problem.parts))
    if problem.terminal is not None:
        return problem.terminal.outputs
    with market_scope(problem.solver, problem.env, problem.market_fps):
        return read(problem, march(problem))

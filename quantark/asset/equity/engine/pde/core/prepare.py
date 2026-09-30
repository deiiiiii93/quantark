"""The one preparation path (spec 2 §5.1, §5.2): ``prepare -> run``.

``prepare`` resolves everything a march consumes -- the outcome that needs no
grid, the layout, the coefficients, the stepper's schedule, the column groups
and a fresh factorization cache -- through the solver's ``FamilyAdapter``.
``run`` marches a prepared problem and reads it, returns a terminal outcome
as is, and composes a composite problem from its parts.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

import numpy as np

from quantark.asset.equity.engine.pde.grid import resolve_bound_layout

from .backends import FactorizationCache
from .components import StandardReadout, validate_schedule
from .identity import market_fingerprints

__all__ = ["Operation", "OutputRequest", "TerminalOutcome", "SessionArtefacts", "PreparedProblem",
           "CompositeProblem", "prepare", "run", "market_scope"]


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


@dataclass(frozen=True)
class TerminalOutcome:
    """An outcome known without a grid (expired, knocked out or touched at valuation)."""
    reason: str
    outputs: Any


@dataclass(frozen=True, eq=False)
class SessionArtefacts:
    """What a prepared execution session injected into its engine clone.

    A bridge over the solver's four ``_session_*`` attributes, removed in Task 10.
    ``grids`` is the injected 5-tuple: the layout is assembled around it once the
    grid request is known (``BasePDESolver._layout_from_session_grids``).
    """
    layout: Any = None
    coefficients: Any = None
    lu_pack: Optional[Mapping] = None
    banded_pack: Optional[Mapping] = None
    grids: Optional[tuple] = None

    @classmethod
    def from_solver(cls, solver, product=None, env=None) -> Optional["SessionArtefacts"]:
        grids = getattr(solver, "_session_grids", None)
        coefficients = getattr(solver, "_session_step_coefficients", None)
        lu_pack = getattr(solver, "_session_matrix_pack", None)
        banded_pack = getattr(solver, "_session_banded_pack", None)
        if grids is None and coefficients is None and lu_pack is None and banded_pack is None:
            return None
        return cls(coefficients=coefficients, lu_pack=lu_pack, banded_pack=banded_pack, grids=grids)


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
    solver._market_fp = (env, fps)
    try:
        yield
    finally:
        solver._market_fp = outer


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
        solve_state = adapter.solve_state(product, env, request)          # step 2
        adapter.market_checks(product, env, tau)                          # _solve's guard, before the grid
        market = solver.market_snapshot(product, env)                     # step 3
        grid_request = adapter.grid_request(product, market, tau, request)    # step 4
        if session is not None and session.layout is not None:           # step 5
            layout = session.layout
        elif session is not None and session.grids is not None:
            layout = solver._layout_from_session_grids(grid_request)
        else:
            layout = resolve_bound_layout(solver.grid_binder, frozen, grid_request, market)
        solver._active_layout = layout    # read by helpers, the intraday route (until Task 11) and tests
        adapter.observation_maps(product, env, layout, tau)               # step 6
        events = adapter.prepare_events(product, env, layout, tau, request)   # step 7
        samples = _lazy(lambda: adapter.samples(product, env, layout, tau))   # step 8
        coefficients = adapter.coefficients(product, env, layout, tau, samples, session)
        stepper = adapter.stepper(solver.params)                          # step 9
        schedule = stepper.schedule(layout)
        validate_schedule(schedule, layout, stepper)
        groups = adapter.groups(product, layout, request)                 # step 10
        factorizations = adapter.factorizations(session)                  # step 11
        identity = {"layout": layout.fingerprint,                         # step 12 (problem id: Task 11)
                    "coefficient_key": adapter.coefficient_key(product, env, layout, fps)}
    return PreparedProblem(solver=solver, adapter=adapter, product=product, env=env, request=request,
                           lifecycle_state=lifecycle_state, tau=tau, terminal=None, solve_state=solve_state,
                           market=market, grid_request=grid_request, layout=layout, samples=samples,
                           coefficients=coefficients, stepper=stepper, theta_schedule=schedule,
                           groups=groups, factorizations=factorizations, readout=StandardReadout(),
                           events=events, identity=identity, market_fps=fps)


def run(problem):
    """March and read a prepared problem; a terminal outcome as is; a composite from its parts."""
    if isinstance(problem, CompositeProblem):
        return problem.compose(tuple(run(p) for p in problem.parts))
    if problem.terminal is not None:
        return problem.terminal.outputs
    from .march import march
    from .readout import read

    with market_scope(problem.solver, problem.env, problem.market_fps):
        return read(problem, march(problem))

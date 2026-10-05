"""The family seam (spec 2 §5.1): what a product family contributes to prepare, march and read.

Every default body reproduces the legacy ``BasePDESolver`` call its comment
names; a family adapter overrides only what its solver overrode.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from ..backends import FactorizationCache
from ..coefficients import CoefficientsInForce
from ..components import CentralDifference, resolve_stepper, validate_coefficients, validate_readout
from ..groups import VALUE, Backend, ColumnGroup
from ..identity import coefficient_key, component_fingerprint
from ..prepare import Operation, TerminalOutcome
from ..samples import SampleKind, sample_bsm
from quantark.util.enum import ObservationType

__all__ = ["FamilyAdapter", "continuous_knock_in"]


def continuous_knock_in(product) -> bool:
    """Whether the contract monitors a knock-in barrier continuously (spec 3A decision 12).

    Contract terms only: never spot, the knocked-in state or a lifecycle flag, which a bump or
    an event can move. A bump therefore cannot change a solve's damping class.
    """
    if not getattr(product, "has_ki_barrier", False):
        return False
    bc = product.barrier_config
    return bool(bc.ki_continuous or bc.ki_observation_type == ObservationType.CONTINUOUS)


class FamilyAdapter:
    """The base family: one value surface, sparse LU, Dirichlet rows from the solver."""

    #: The route a solver declares (``check_route_applicability``); None = every shared-grid field honoured.
    route = None
    #: Whether event statistics march as a column group of this family (§5.4).
    supports_statistics_group = False
    #: What the route samples per step (explain's route section).
    sample_kind = SampleKind.SCALAR
    #: The model the family's own statistics belong to (StatisticsProvenance.statistics_model)
    statistics_model = "flat_vol_1d"

    def statistics_family(self, prepared, streams):
        """The value-mode statistics group (spec 3B §5.1); per family, Tasks 12-15."""
        raise NotImplementedError(f"{type(self).__name__} has no value-mode statistics")

    def readout_mode(self, prepared) -> str:
        """How the price is read (explain's outputs section)."""
        return "final column"

    def __init__(self, solver):
        self.solver = solver

    # --- prepare ------------------------------------------------------------------

    def validate(self, product, env, request, lifecycle_state) -> None:
        """The facades' raising checks (none in the base)."""

    def composite(self, product, env, request, lifecycle_state):
        return None

    def bump_parts(self, product, env):
        """``{part: (solver, product)}`` a bump context freezes one layout for each (spec 3C decision 10);
        ``None``, a single layout for the product itself."""
        return None

    def scope(self, product, env):
        """A context held around preparation (the local-vol routes build their surface in it)."""
        return nullcontext()

    def factorizations(self, session) -> FactorizationCache:
        # retention: full, per prepared problem; a prepared session's packs are consulted first
        return FactorizationCache(
            enabled=self.solver._is_cache_enabled(),
            lu_seed=None if session is None else session.lu_pack,
            banded_seed=None if session is None else session.banded_pack)

    def solve_state(self, product, env, request):
        # _solve's state preamble (BasePDESolver._prepare_solve_state); the base has none
        self.solver._prepare_solve_state(product, env)
        return {}

    def terminal_outcome(self, product, env, request, lifecycle_state):
        # BasePDESolver._price_with_solution / calculate_greeks: tau <= 0 short-circuits
        tau = product.get_maturity(env)
        if tau > 0:
            return None
        spot = env.spot
        solver = self.solver
        if request.operation is Operation.GREEKS:
            return TerminalOutcome("expired", {
                "price": solver._calculate_intrinsic(product, spot),
                "delta": solver._intrinsic_delta(product, spot),
                "gamma": 0.0,
            })
        price = solver._calculate_intrinsic(product, spot)
        if request.operation is Operation.SESSION_OUTPUTS:
            return TerminalOutcome("expired", self._session_outputs(price, None))
        return TerminalOutcome("expired", price)

    def check_route(self, params) -> None:
        if self.route is not None:
            from quantark.asset.equity.engine.pde.route_applicability import check_route_applicability
            check_route_applicability(params, self.route)
        # the operator must assemble what this route samples (§5.8)
        if self.sample_kind not in params.operator.supports:
            from quantark.util.exceptions import ValidationError
            raise ValidationError(
                f"PDEParams.operator {type(params.operator).__name__} does not support the "
                f"{self.sample_kind.value} samples of {type(self.solver).__name__}"
            )

    def market_checks(self, product, env, tau) -> None:
        # _solve: this solver diffuses with a single strike-selected constant vol;
        # a smile surface is collapsed (skew ignored). Make that explicit.
        from quantark.param.vol.collapse_guard import guard_constant_vol
        solver = self.solver
        guard_constant_vol(env.vol_surface, type(solver).__name__,
                           strict=getattr(solver.params, "strict_smile", False))

    def grid_request(self, product, market, tau, request):
        return self.solver.grid_request(product, market, tau)

    def observation_maps(self, product, env, layout, tau) -> None:
        self.solver._populate_observation_maps(product, env, layout, tau)

    def prepare_events(self, product, env, layout, tau, request) -> tuple:
        return ()

    def _strike(self, product, env):
        return getattr(product, "strike", env.spot)

    def _scalars(self, product, env, tau):
        # _solve: r, q, sigma at maturity (the scalar stencil / flat-exact inputs)
        strike = self._strike(product, env)
        return env.get_rate(tau), env.get_div_yield(tau), env.get_vol(strike, tau)

    def samples(self, product, env, layout, tau):
        r, q, sigma = self._scalars(product, env, tau)
        return sample_bsm(env, layout.time.t, ref_strike=self._strike(product, env),
                          r_tau=r, q_tau=q, sigma_tau=sigma)

    def step_sets(self, product, env, layout, tau):
        """The per-step sets before the flat-exact replacement (the session's
        step-coefficient artefact): the solver's memoized builder for the
        built-in operator, else the operator's own assembly (no memo)."""
        operator = self.solver.params.operator
        if type(operator) is not CentralDifference:
            return operator.assemble(self.samples(product, env, layout, tau), layout)
        dx, num_x = layout.spatial.dx, len(layout.spatial.x)
        return self.solver._build_step_coefficients(env, self._strike(product, env), layout.time.t,
                                                    dx, num_x)

    def coefficients(self, product, env, layout, tau, samples, session):
        # _solve: the session's (pre-flat-exact) sets, else step_sets;
        # the flat-exact replacement; the scalar stencil when one set remains
        solver = self.solver
        dx, num_x = layout.spatial.dx, len(layout.spatial.x)
        r, q, sigma = self._scalars(product, env, tau)
        if session is not None and session.coefficients is not None:
            sc = session.coefficients
        else:
            sc = self.step_sets(product, env, layout, tau)
        operator = solver.params.operator
        if type(operator) is not CentralDifference:
            # a custom operator's sets are marched as assembled: never collapsed to the
            # built-in scalar stencil, never replaced by the built-in flat-exact sets
            validate_coefficients(sc, len(layout.time.dt), num_x, operator)
            return CoefficientsInForce(flat=sc.lcu_sets[0], sets=sc, num_x=num_x)
        # The solver's own stencil and flat-exact methods: a subclass override stays in force.
        flat = solver._calculate_coefficients(r, q, sigma, dx, num_x)
        sc = solver._flat_exact_step_coefficients(sc, r, q, sigma, dx, num_x)
        validate_coefficients(sc, len(layout.time.dt), num_x, CentralDifference())
        return CoefficientsInForce(flat=flat, sets=None if sc.n_unique == 1 else sc, num_x=num_x)

    def damping_key(self, product) -> str:
        """The ``DAMPING_DEFAULTS`` row this product's solves resolve against (decision 12)."""
        return "core"

    def stepper(self, params, product):
        from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
        return resolve_stepper(params, DAMPING_DEFAULTS[self.damping_key(product)])

    def coefficient_key(self, product, env, layout, fps):
        return coefficient_key(layout.fingerprint, fps, self._strike(product, env),
                               component_fingerprint(self.solver.params.operator))

    def groups(self, product, layout, request) -> tuple:
        return (ColumnGroup("value", ("value",), Backend.SPARSE_LU, VALUE),)

    # --- march --------------------------------------------------------------------

    def allocate(self, prepared) -> dict:
        layout = prepared.layout
        return {"value": np.zeros((len(layout.spatial.x), len(layout.time.t)))}

    def terminal(self, prepared, states, captures) -> None:
        sp = prepared.layout.spatial
        self.solver.set_terminal_condition(states["value"], sp.x, sp.s, prepared.product, prepared.env)

    def boundaries(self, prepared, group, states, j) -> None:
        sp, t = prepared.layout.spatial, prepared.layout.time.t
        self.solver.set_boundary_conditions(states["value"], sp.x, sp.s, j, prepared.tau - t[j],
                                            prepared.product, prepared.env)

    def after_solve(self, prepared, group, states, j) -> None:
        sp, t = prepared.layout.spatial, prepared.layout.time.t
        self.solver._apply_step_modifications(states["value"], sp.x, sp.s, j, prepared.tau - t[j],
                                              prepared.product, prepared.env)

    def step_events(self, prepared, states, captures, j) -> None:
        pass

    # --- read ---------------------------------------------------------------------

    def read(self, prepared, result):
        column = result.states["value"][:, 0]
        spot = prepared.env.spot
        readout, layout = prepared.readout, prepared.layout
        price = readout.price(column, spot, layout)
        validate_readout(price, readout)
        op = prepared.request.operation
        if op is Operation.GREEKS:
            delta, gamma = readout.delta_gamma(column, spot, layout)
            validate_readout((delta, gamma), readout)
            return {"price": price, "delta": delta, "gamma": gamma}
        if op is Operation.SESSION_OUTPUTS:
            solution = None
            if prepared.request.grid:
                from quantark.asset.equity.engine.pde.base_pde_solver import PDESolutionResult
                solution = PDESolutionResult(solution_vec=column, x_vec=layout.spatial.x,
                                             s_vec=layout.spatial.s, spot_log=np.log(spot))
            return self._session_outputs(price, solution)
        return price

    @staticmethod
    def _session_outputs(npv, solution):
        # BasePDESolver._session_outputs: one value solve serving PV (+ grid projection);
        # event fields are autocallable-only and None here
        from quantark.asset.equity.engine.pde.base_pde_solver import PDESessionOutputs
        return PDESessionOutputs(npv=float(npv), solution=solution, event_stats=None,
                                 event_distribution=None)

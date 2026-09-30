"""The family seam (spec 2 §5.1): what a product family contributes to prepare, march and read.

Every default body reproduces the legacy ``BasePDESolver`` call its comment
names; a family adapter overrides only what its solver overrode.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from ..backends import FactorizationCache
from ..coefficients import CoefficientsInForce
from ..components import CentralDifference, scheme_stepper, validate_coefficients, validate_readout
from ..groups import VALUE, Backend, ColumnGroup
from ..identity import coefficient_key, component_fingerprint
from ..prepare import Operation, TerminalOutcome
from ..samples import sample_bsm

__all__ = ["FamilyAdapter"]


class FamilyAdapter:
    """The base family: one value surface, sparse LU, Dirichlet rows from the solver."""

    #: The route a solver declares (``check_route_applicability``); None = every shared-grid field honoured.
    route = None

    def __init__(self, solver):
        self.solver = solver

    # --- prepare ------------------------------------------------------------------

    def validate(self, product, env, request, lifecycle_state) -> None:
        """The facades' raising checks (none in the base)."""

    def composite(self, product, env, request, lifecycle_state):
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

    def coefficients(self, product, env, layout, tau, samples, session):
        # _solve: the session's (pre-flat-exact) sets, else the memoized builder;
        # the flat-exact replacement; the scalar stencil when one set remains
        solver = self.solver
        strike = self._strike(product, env)
        dx, num_x = layout.spatial.dx, len(layout.spatial.x)
        r, q, sigma = self._scalars(product, env, tau)
        if session is not None and session.coefficients is not None:
            sc = session.coefficients
        else:
            sc = solver._build_step_coefficients(env, strike, layout.time.t, dx, num_x)
        # The solver's own stencil and flat-exact methods: a subclass override stays in force.
        flat = solver._calculate_coefficients(r, q, sigma, dx, num_x)
        sc = solver._flat_exact_step_coefficients(sc, r, q, sigma, dx, num_x)
        validate_coefficients(sc, len(layout.time.dt), num_x, CentralDifference())
        return CoefficientsInForce(flat=flat, sets=None if sc.n_unique == 1 else sc, num_x=num_x)

    def stepper(self, params):
        return scheme_stepper(params, self.solver.grid_binder.config)

    def coefficient_key(self, product, env, layout, fps):
        return coefficient_key(layout.fingerprint, fps, self._strike(product, env),
                               component_fingerprint(CentralDifference()))

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

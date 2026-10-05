"""The DCN on the core (spec 3C §5.4): two surfaces - alive (never knocked in) and ki (knocked in) - advanced as
one banded group; the contract's events at each daily date as one projection per surface; Dirichlet rows from the
contract's certain outcomes at the edges."""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.exceptions import PricingError

from ..groups import VALUE, Backend, ColumnGroup
from ..prepare import Operation
from .base import FamilyAdapter

__all__ = ["DCNAdapter", "DCNEvents"]


class DCNEvents:
    """One solve's contract on its layout: cash per observation, events per date, edge rows per time."""

    def __init__(self, solver, product, env, layout):
        self.solver, self.product, self.env, self.layout = solver, product, env, layout
        self.ctx = ctx = build_dcn_grid_context(product)
        self.df = make_df_fn(env)
        time = layout.time
        self.node_of = {float(t): time.step_at(float(t)) for t in ctx.times[1:-1]}
        self.event_nodes = frozenset(self.node_of.values())
        self.obs_at_node = {self.node_of[float(ctx.times[c])]: j
                            for j, c in enumerate(ctx.obs_cols) if 0 < c < ctx.times.size - 1}
        s = self.s = layout.spatial.s
        self.ki_mask, self.ko_mask, self.cp_mask = s <= product.ki_barrier, s >= product.ko_barrier, s >= product.coupon_barrier
        self.T = float(time.t[-1])
        self.nodal = solver._base_family_projection() is EventProjectionMode.NODAL

    def amounts(self, j, t_obs):
        """(coupon, knock-out) cash at observation ``j`` valued at its date, as today's ``event_amounts``."""
        p, ctx, df = self.product, self.ctx, self.df
        cpn = ko = None
        if ctx.obs_is_coupon[j]:
            cpn = p.participation * p.coupon_rate * ctx.coupon_accruals[j] * p.notional * (df(ctx.coupon_pay_times[j]) / df(t_obs))
        if ctx.obs_is_ko[j]:
            ko = p.participation * p.ko_coupon_rate * ctx.ko_accruals[j] * p.notional * (df(ctx.ko_pay_times[j]) / df(t_obs))
        return cpn, ko

    def apply(self, v0, v1, j, t_obs):
        """The contract at one date (backward order coupon, knock-out, knock-in); ``j`` None: knock-in only."""
        from quantark.asset.equity.engine.pde.dcn_pde_solver import apply_dcn_events
        from quantark.asset.equity.engine.pde.grid.events import project_between, project_piecewise
        p, sp = self.product, self.layout.spatial
        cpn, ko = (None, None) if j is None else self.amounts(j, t_obs)
        if self.nodal:                                   # decision 23: the legacy masks
            return apply_dcn_events(v0, v1, self.cp_mask, self.ko_mask, self.ki_mask, cpn, ko)
        if ko is not None:
            v1 = project_between(sp, p.ko_barrier, True, np.full_like(v1, ko), v1)
        thresholds = sorted({p.ki_barrier} | ({p.coupon_barrier} if cpn is not None else set())
                            | ({p.ko_barrier} if ko is not None else set()))
        branches = []
        for r in range(len(thresholds) + 1):
            lo = 0.0 if r == 0 else thresholds[r - 1]
            if r == 0:
                branches.append(v1)                       # below the knock-in barrier
            elif ko is not None and lo >= p.ko_barrier:
                branches.append(np.full_like(v0, ko))     # knock-out has priority over the coupon
            elif cpn is not None and lo >= p.coupon_barrier:
                branches.append(v0 + cpn)
            else:
                branches.append(v0)
        return project_piecewise(sp, thresholds, branches), v1

    def rows(self, t):
        """((alive, ki) at the lower edge, (alive, ki) at the upper edge) just after the events of time ``t``."""
        p, ctx, df, solver, env = self.product, self.ctx, self.df, self.solver, self.env
        s_lo = float(self.s[0])
        # lower edge: knock-in at the next date is certain and no knock-out: the loss put deep in the money, on the
        # edge carried forward to the horizon (spec 3A §5.10's forward factors)
        fwd = s_lo * solver._carry_df_between_times(env, t, self.T) / solver._df_between_times(env, t, self.T)
        loss = -(p.notional / p.initial_price) * p.participation * max(p.k_loss - fwd, 0.0) * df(ctx.loss_pay_time) / df(t)
        # upper edge: knock-out at the next eligible observation is certain; alive also collects the coupons before it
        ki_up = alive_up = 0.0
        for j, c in enumerate(ctx.obs_cols):
            t_obs = float(ctx.times[c])
            if t_obs <= t:
                continue
            cpn, ko = self.amounts(j, t_obs)
            if ko is not None:
                ki_up += ko * df(t_obs) / df(t)
                alive_up += ko * df(t_obs) / df(t)
                break
            if cpn is not None:
                alive_up += cpn * df(t_obs) / df(t)
        return (loss, loss), (alive_up, ki_up)


class DCNAdapter(FamilyAdapter):
    """``DCNPDEEngine`` (spec 3C §5.4)."""

    def validate(self, product, env, request, lifecycle_state) -> None:
        from quantark.asset.equity.engine.settlement_support import validate_settlement_capability
        from quantark.asset.equity.product.option.dcn_option import DCNOption
        if not isinstance(product, DCNOption):
            raise PricingError(f"{type(self.solver).__name__} only supports DCNOption")
        validate_settlement_capability(self.solver, product, lifecycle_state)

    def terminal_outcome(self, product, env, request, lifecycle_state):
        return None          # the contract's grid context sets the horizon; today's engine had no expiry branch

    def market_checks(self, product, env, tau) -> None:
        """Today's DCN engine read the term vol at the initial price and never guarded the collapse."""

    def _strike(self, product, env):
        return product.initial_price

    def prepare_events(self, product, env, layout, tau, request):
        return DCNEvents(self.solver, product, env, layout)

    def groups(self, product, layout, request) -> tuple:
        return (ColumnGroup("value", ("alive", "ki"), Backend.BANDED, VALUE),)

    def allocate(self, prepared) -> dict:
        shape = (len(prepared.layout.spatial.x), len(prepared.layout.time.t))
        return {"alive": np.zeros(shape), "ki": np.zeros(shape)}

    def terminal(self, prepared, states, captures) -> None:
        ev, p = prepared.events, prepared.product
        v1 = (-(p.notional / p.initial_price) * p.participation * np.maximum(p.k_loss - ev.s, 0.0)
              * ev.df(ev.ctx.loss_pay_time) / ev.df(ev.T))
        v0, v1 = ev.apply(np.zeros_like(v1), v1, len(ev.ctx.obs_cols) - 1, ev.T)
        states["alive"][:, -1], states["ki"][:, -1] = v0, v1

    def boundaries(self, prepared, group, states, j) -> None:
        (a_lo, k_lo), (a_up, k_up) = prepared.events.rows(float(prepared.layout.time.t[j]))
        states["alive"][0, j], states["alive"][-1, j] = a_lo, a_up
        states["ki"][0, j], states["ki"][-1, j] = k_lo, k_up

    def after_solve(self, prepared, group, states, j) -> None:
        pass

    after_solve.noop = True

    def step_events(self, prepared, states, captures, j) -> None:
        ev = prepared.events
        if j == 0 or j not in ev.event_nodes:
            return
        t_j = float(prepared.layout.time.t[j])
        v0, v1 = ev.apply(states["alive"][:, j].copy(), states["ki"][:, j].copy(), ev.obs_at_node.get(j), t_j)
        states["alive"][:, j], states["ki"][:, j] = v0, v1

    def read(self, prepared, result):
        from quantark.asset.equity.engine.pde.base_pde_solver import BranchSolution
        from ..components import validate_readout
        product, env, layout, readout = prepared.product, prepared.env, prepared.layout, prepared.readout
        spot, sign = env.spot, float(product.direction_sign)
        flagged = bool(product.knocked_in_at_valuation)
        alive, ki = result.states["alive"][:, 0], result.states["ki"][:, 0]
        column = ki if (flagged or spot <= product.ki_barrier) else alive
        price = sign * readout.price(column, spot, layout)
        validate_readout(price, readout)
        op = prepared.request.operation
        if op is Operation.GREEKS:
            delta, gamma = readout.delta_gamma(column, spot, layout)
            return {"price": price, "delta": sign * delta, "gamma": sign * gamma}
        if op is Operation.SESSION_OUTPUTS:
            solution = None
            if prepared.request.grid:
                solution = BranchSolution(alive=sign * alive, ki=sign * ki, x_vec=layout.spatial.x,
                                          s_vec=layout.spatial.s, ki_barrier=float(product.ki_barrier),
                                          knocked_in=flagged)
            return self._session_outputs(price, solution)
        return price

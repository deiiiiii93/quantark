"""DCN PDE event statistics from the exact forward adjoint of the price's own march (unified event statistics spec
T15; DESIGN §4.5 option A, D12, D23, AC7, AC8, AC13, AC14, AC17, AC18).

The DCN price marches two surfaces backward -- ``alive`` (never knocked in) and ``ki`` (knocked in) -- through
(a) one step per time node with Dirichlet rows at both spot edges and (b) the contract's events at every daily date
(backward order coupon, knock-out, knock-in projection), from the terminal loss leg. Every operation is affine in the
surfaces, so the price is ``Σ <dual, source>`` over the march's sources: the terminal values, the coupon and
knock-out cash of each event, and the edge rows. This module runs the exact transpose of that march forward in time
from the readout's own weights at the spot (``DCNPDEEngine`` / ``LocalVolDCNPDEEngine``: the core's banded
Crank-Nicolson step, ``adjoint.banded_step_transpose``; ``HestonDCNPDESolver``: the core's own Douglas / Craig-Sneyd
step with its implicit reaction, ``CoreStepTranspose``), and integrates the dual against each source, labelled by the
outcome it pays (D23: the statistics describe the price's own march):

* an event (``_EventStage``): the dual knocked out (alive: no knock-in before it, the knocked-in surface: after one;
  a same-instant knock-in then knock-out counts as the knock-out, KO wins the tie), the dual paid the fixed coupon,
  and the dual moved from ``alive`` to ``ki`` (the first knock-in at that daily date, before any knock-out);
* a lower edge row (the loss leg, ``DCNEvents.edge_terms`` / the ADI hooks): a knock-in certain at the next daily date,
  settled knocked in, the row's loss as its cash;
* an upper edge row: on the 1D core the coupons up to the next knock-out observation and that knock-out (none left:
  the alive edge matures never knocked in, the knocked-in edge settles knocked in, worth 0); the ADI hooks write 0
  there, which the statistics read as no event (alive: matured, knocked-in: settled knocked in, both worth 0);
* the terminal surfaces after the final date's events: matured never knocked in (alive, cash 0) and settled knocked in
  (ki, the loss leg).

Each discounted mass at a source of time ``t`` divided by ``DF(t)`` is that outcome's probability (the normalisation
of every grid engine's unit columns); each cash is the probability times the leg amount and its payment DF -- the same
products the march sums, so ``reconciliation_error`` is round-off and ``probability_residual`` the march's own
discounting of a unit claim (measured, shrinks under refinement). ``pv`` is ``price()``'s value (D23).

One pass yields every outcome, the knock-in path statistics included: there is no cheaper route to the knock-out,
coupon and survivor outcomes, so ``streams`` is accepted and nothing is pruned (as the DCN MC engines; T11).

``backward_claims`` marches unit claims backward with the engines' own step and event primitives (the adjoint
identity's other side, spec AC14).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from quantark.asset.equity.engine.dcn_statistics import (
    DCNRows,
    dcn_statistics,
    decided_row_statistics,
    terminated_dcn_statistics,
)
from quantark.asset.equity.engine.event_stats import EventStatistics, ReceivablesTable, StatisticsProvenance
from quantark.util.exceptions import PricingError
from quantark.util.numerical import Tolerance, is_close, same_instant

from .core.statistics.adjoint import AdjointUnavailable, banded_step_transpose
from .vol_model_adjoint import StepTranspose, _transposed_tridiagonal

__all__ = [
    "ALIVE",
    "KI",
    "CoreStepTranspose",
    "DCNForwardResult",
    "backward_claims_1d",
    "backward_claims_adi",
    "dcn_1d_statistics",
    "dcn_adi_statistics",
]

#: the two surfaces, in the duals' stacking order
ALIVE, KI = 0, 1


# --- the event stage, transposed ----------------------------------------------------------------------------------

class _EventStage:
    """One date's DCN event ``(v0, v1) <- f(v0, v1; cpn, ko)`` as its transposed parts, read off the engine's own
    primitive ``apply(v0, v1, cpn, ko)`` (affine in all four, acting on axis 0) evaluated on the identity:
    ``v0' = A00 v0 + A01 v1 + w_cpn cpn + w_ko0 ko``, ``v1' = A11 v1 + w_ko1 ko``."""

    def __init__(self, apply: Callable, n: int, coupon_on: bool, ko_on: bool, vector_shape: tuple):
        eye, zero = np.eye(n), np.zeros((n, n))
        cpn = 0.0 if coupon_on else None
        ko = 0.0 if ko_on else None
        a0, a1 = apply(eye.copy(), zero.copy(), cpn, ko)
        b0, b1 = apply(zero.copy(), eye.copy(), cpn, ko)
        if np.any(np.asarray(a1) != 0.0):
            raise AdjointUnavailable("undefined: the DCN event writes the knocked-in surface from the alive one")
        self.a00t = sp.csr_matrix(np.asarray(a0, dtype=float).T)
        self.a01t = sp.csr_matrix(np.asarray(b0, dtype=float).T)
        self.a11t = sp.csr_matrix(np.asarray(b1, dtype=float).T)
        # the affine weights on one zero column of the shape the engine's masks broadcast against
        z = np.zeros(vector_shape)
        flat = lambda a: np.asarray(a, dtype=float).reshape(n).copy()   # noqa: E731
        self.w_cpn = flat(apply(z.copy(), z.copy(), 1.0, ko)[0]) if coupon_on else None
        if ko_on:
            k0, k1 = apply(z.copy(), z.copy(), cpn, 1.0)
            self.w_ko0, self.w_ko1 = flat(k0), flat(k1)
        else:
            self.w_ko0 = self.w_ko1 = None

    def transpose(self, p: np.ndarray):
        """``(p_in, masses)`` from the duals ``p`` (``(2, n, m)``) on the event's output; ``masses``: the discounted
        dual knocked out from alive / from ki, paid the coupon, and moved to ki (first knock-in)."""
        p0, p1 = p[ALIVE], p[KI]
        moved = self.a01t @ p0
        out = np.empty_like(p)
        out[ALIVE] = self.a00t @ p0
        out[KI] = moved + self.a11t @ p1
        masses = {
            "ko_alive": 0.0 if self.w_ko0 is None else float(np.sum(self.w_ko0[:, None] * p0)),
            "ko_ki": 0.0 if self.w_ko1 is None else float(np.sum(self.w_ko1[:, None] * p1)),
            "coupon": 0.0 if self.w_cpn is None else float(np.sum(self.w_cpn[:, None] * p0)),
            "knock_in": float(np.sum(moved)),
        }
        return out, masses


class _Stages:
    """The event stages of one march, one per active-branch pattern (the cash enters through the affine weights)."""

    def __init__(self, apply: Callable, n: int, vector_shape: tuple):
        self.apply, self.n, self.vector_shape = apply, n, vector_shape
        self._cache: Dict[tuple, _EventStage] = {}

    def __call__(self, coupon_on: bool, ko_on: bool) -> _EventStage:
        key = (bool(coupon_on), bool(ko_on))
        stage = self._cache.get(key)
        if stage is None:
            stage = self._cache[key] = _EventStage(self.apply, self.n, *key, self.vector_shape)
        return stage


# --- the tally of every outcome ------------------------------------------------------------------------------------

class DCNForwardResult:
    """The outcomes one forward pass integrated, in probability units (discounted mass / DF at the source's time),
    plus the knocked-in settlement's discounted loss cash (buyer view) and, with ``reconstruct``, the pass's value of
    the price (``Σ <dual, source>``, the adjoint identity's side)."""

    def __init__(self, rows: DCNRows):
        self.rows = rows
        n_obs = rows.times.size
        self.ko = np.zeros(n_obs)
        self.clean_ko = np.zeros(n_obs)
        self.coupon = np.zeros(n_obs)
        self.maturity = 0.0
        self.ki_settlement = 0.0
        self.ki_settlement_cash = 0.0
        self.ki_rows = np.zeros(rows.ki_times.size)
        self.reconstructed = 0.0

    def ki_row_at(self, t: float) -> int:
        """The knock-in row of the daily date ``t``."""
        k = int(np.searchsorted(self.rows.ki_times, t - Tolerance.ZERO, side="left"))
        if k >= self.rows.ki_times.size or not same_instant(self.rows.ki_times[k], t):
            raise PricingError(f"the forward pass found a knock-in at t = {t!r}, not a daily KI date")
        return k

    def next_ki_row(self, t: float) -> Optional[int]:
        """The first daily KI date strictly after ``t`` (a certain knock-in from an edge row at ``t``)."""
        k = int(np.searchsorted(self.rows.ki_times, t + Tolerance.ZERO, side="right"))
        return k if k < self.rows.ki_times.size else None

    def event(self, t: float, ki_time: float, j_obs: Optional[int], masses: dict, df_t: float) -> None:
        ko = (masses["ko_alive"] + masses["ko_ki"]) / df_t
        if j_obs is not None:
            self.ko[j_obs] += ko
            self.clean_ko[j_obs] += masses["ko_alive"] / df_t
            self.coupon[j_obs] += masses["coupon"] / df_t
        elif masses["ko_alive"] != 0.0 or masses["ko_ki"] != 0.0 or masses["coupon"] != 0.0:
            raise PricingError(f"the forward pass found monthly cash at a daily date t = {t!r}")
        if masses["knock_in"] != 0.0:
            self.ki_rows[self.ki_row_at(ki_time)] += masses["knock_in"] / df_t

    def lower_edge(self, t: float, alive: float, ki: float, value: float, df_t: float) -> None:
        """The loss row: settled knocked in (both surfaces); the alive edge knocks in at the next daily date."""
        self.ki_settlement += (alive + ki) / df_t
        self.ki_settlement_cash += (alive + ki) * value
        if alive != 0.0:
            k = self.next_ki_row(t)
            if k is None:
                raise PricingError(f"the forward pass found alive loss-row mass after the last KI date (t = {t!r})")
            self.ki_rows[k] += alive / df_t

    def upper_edge(self, alive: float, ki: float, coupons: Sequence[int], ko_row: Optional[int],
                   ki_ko_row: Optional[int], df_t: float) -> None:
        """The far-spot row: the alive edge collects ``coupons`` then knocks out at ``ko_row`` (``None``: matures
        never knocked in); the knocked-in edge knocks out at ``ki_ko_row`` (``None``: settles knocked in, worth 0)."""
        a, k = alive / df_t, ki / df_t
        for j in coupons:
            self.coupon[j] += a
        if ko_row is None:
            self.maturity += a
        else:
            self.ko[ko_row] += a
            self.clean_ko[ko_row] += a
        if ki_ko_row is None:
            self.ki_settlement += k
        else:
            self.ko[ki_ko_row] += k

    def terminal(self, p: np.ndarray, v1_terminal: np.ndarray, df_T: float) -> None:
        self.maturity += float(np.sum(p[ALIVE])) / df_T
        self.ki_settlement += float(np.sum(p[KI])) / df_T
        self.ki_settlement_cash += float(np.sum(p[KI] * v1_terminal))


def _assemble(rows: DCNRows, result: DCNForwardResult, *, pv: float, coupon_today: bool, receivables,
              provenance: StatisticsProvenance) -> EventStatistics:
    coupon = result.coupon.copy()
    if coupon_today:
        # W10b: the valuation-date row's coupon, decided at the spot (the march never applies t = 0)
        coupon[0] += 1.0
    return dcn_statistics(rows, pv=pv, ko_probability=result.ko, coupon_probability=coupon,
                          maturity_probability=result.maturity, ki_settlement_probability=result.ki_settlement,
                          ki_settlement_cash=result.ki_settlement_cash, ki_row_probability=result.ki_rows,
                          clean_ko_probability=result.clean_ko, knocked_out_now=False, receivables=receivables,
                          provenance=provenance)


def _receivables(product, state) -> ReceivablesTable:
    return ReceivablesTable.empty() if state is None else state.receivables_table(product.schedule.valuation_date)


# --- the PDE 1D core routes (DCNPDEEngine, LocalVolDCNPDEEngine) ----------------------------------------------------

def _readout_weights_1d(problem, column: np.ndarray) -> np.ndarray:
    """The readout's weight on every node, probed node by node and checked on the read column (a readout that is not
    linear has no dual and is refused)."""
    readout, spot, layout = problem.readout, problem.env.spot, problem.layout
    n = column.shape[0]
    unit = np.zeros(n)
    weights = np.empty(n)
    for i in range(n):
        unit[i] = 1.0
        weights[i] = float(readout.price(unit, spot, layout))
        unit[i] = 0.0
    value = float(readout.price(np.ascontiguousarray(column), spot, layout))
    if not is_close(float(weights @ column), value, rel_tol=Tolerance.RELATIVE,
                    abs_tol=Tolerance.ZERO * max(1.0, float(np.max(np.abs(column))))):
        raise AdjointUnavailable(f"undefined: the readout {type(readout).__name__} is not linear in the column")
    return weights


def _forward_1d(problem, *, read_ki: bool, weights: np.ndarray, rows: DCNRows) -> DCNForwardResult:
    from .core.statistics.adjoint import _builtin_crank_nicolson
    from .core.groups import Backend

    if not _builtin_crank_nicolson(problem.stepper):
        # TODO(unified event statistics): a custom stepper component's step has no transpose the pass can build
        raise AdjointUnavailable(f"undefined: the DCN forward pass transposes the built-in Crank-Nicolson step; the "
                                 f"custom stepper {type(problem.stepper).__name__} has none")
    group = problem.groups[0]
    if group.backend is not Backend.BANDED:
        raise AdjointUnavailable("undefined: the DCN forward pass transposes the banded value group only")
    ev, layout = problem.events, problem.layout
    t = np.asarray(layout.time.t, dtype=float)
    N = t.size
    n = len(layout.spatial.x)
    ctx = ev.ctx
    time_of_node = {node: float(tc) for tc, node in ev.node_of.items()}
    stages = _Stages(ev.apply_amounts, n, (n,))   # the 1D masks are (n,)
    thetas = [float(x) for x in problem.theta_schedule]
    zd = bool(group.policy.zero_diffusion_theta_one)
    bands: dict = {}
    result = DCNForwardResult(rows)
    p = np.zeros((2, n, 1))
    p[KI if read_ki else ALIVE, :, 0] = weights
    last = len(ctx.obs_cols) - 1
    for j in range(N - 1):
        out, on_lower, on_upper = banded_step_transpose(problem, j, p[:, :, 0], thetas, zd, bands)
        tj = float(t[j])
        df_t = float(ev.df(tj))
        lower_value, (coupons, ko_row), ki_ko_row = ev.edge_terms(tj)
        result.lower_edge(tj, float(on_lower[ALIVE]), float(on_lower[KI]), lower_value, df_t)
        result.upper_edge(float(on_upper[ALIVE]), float(on_upper[KI]), coupons, ko_row, ki_ko_row, df_t)
        p = out[:, :, None]
        node = j + 1
        if node == N - 1:
            j_obs, ki_time = last, float(ctx.times[-1])
        elif node in ev.event_nodes:
            j_obs, ki_time = ev.obs_at_node.get(node), time_of_node[node]
        else:
            continue
        t_node = float(t[node])
        cpn, ko = (None, None) if j_obs is None else ev.amounts(j_obs, t_node)
        p, masses = stages(cpn is not None, ko is not None).transpose(p)
        result.event(t_node, ki_time, j_obs, masses, float(ev.df(t_node)))
    product = problem.product
    T = float(t[-1])
    s = np.asarray(ev.s, dtype=float)
    v1_terminal = (-(product.notional / product.initial_price) * product.participation
                   * np.maximum(product.k_loss - s, 0.0) * ev.df(ctx.loss_pay_time) / ev.df(T))
    result.terminal(p[:, :, 0], v1_terminal, float(ev.df(T)))
    return result


def _provenance_1d(solver) -> StatisticsProvenance:
    model = solver.adapter.statistics_model
    return StatisticsProvenance(model, model)


def dcn_1d_statistics(solver, product, pricing_env, *, lifecycle_state=None):
    """``(pv, EventStatistics or None)`` of ``DCNPDEEngine`` / ``LocalVolDCNPDEEngine``: ``pv`` from the price's own
    ``prepare -> march -> read`` (``price()`` bitwise), the statistics from the forward pass over that march.

    ``None`` statistics: a terminated lifecycle state with nothing unpaid (spec AC17).

    Raises:
        AdjointUnavailable: the march has no exact transpose here (a custom stepper, a non-banded group, a nonlinear
            readout).
    """
    from quantark.asset.equity.engine.dcn_valuation import observe_valuation_row, resolve_dcn_lifecycle_state

    from .core.march import march
    from .core.prepare import Operation, OutputRequest, market_scope, prepare
    from .core.readout import read

    problem = prepare(solver, product, pricing_env, OutputRequest(Operation.PRICE), lifecycle_state=lifecycle_state,
                      frozen=solver._frozen_base_layout, session=solver._session_artefacts)
    state = resolve_dcn_lifecycle_state(product, pricing_env, lifecycle_state)
    provenance = _provenance_1d(solver)
    if problem.terminal is not None:
        pv = problem.terminal.outputs
        if state is not None and state.terminal:
            return pv, terminated_dcn_statistics(product, state, pv=pv, provenance=provenance)
        from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
        ctx = build_dcn_grid_context(product)
        row = observe_valuation_row(product, pricing_env, ctx)
        return pv, decided_row_statistics(product, pricing_env, ctx, row, pv=pv,
                                          receivables=_receivables(product, state), provenance=provenance)
    with market_scope(problem.solver, problem.env, problem.market_fps):
        marched = march(problem)
        pv = read(problem, marched)
        ev = problem.events
        spot = float(pricing_env.spot)
        read_ki = bool(product.knocked_in_at_valuation) or (ev.ctx.valuation_monitored
                                                            and spot <= product.ki_barrier)
        column = marched.states["ki" if read_ki else "alive"][:, 0]
        weights = _readout_weights_1d(problem, column)
        rows = DCNRows(product, pricing_env, ev.ctx)
        result = _forward_1d(problem, read_ki=read_ki, weights=weights, rows=rows)
    row = observe_valuation_row(product, pricing_env, ev.ctx, ev.df)
    stats = _assemble(rows, result, pv=pv, coupon_today=bool(row is not None and row.coupon_paid),
                      receivables=_receivables(product, state), provenance=provenance)
    return pv, stats


# --- the Heston DCN ADI route (HestonDCNPDESolver's own loop) ------------------------------------------------------

class CoreStepTranspose(StepTranspose):
    """The transpose of ``HestonSLVADICore._douglas_step_core`` / ``_cs_step_core`` (the steps ``HestonDCNPDESolver``
    takes): ``StepTranspose``'s batched-path transpose with the core's own reaction convention -- ``-rU`` inside
    ``A1`` (interior and the degenerate ``v = 0`` column) and ``+θ dt r`` on the implicit spot sweep's diagonal, no
    exponential discount -- and the boundary hook's rows (``_bc`` and the sweeps' right-hand sides)."""

    def __init__(self, core, dt: float, theta: float, t_mid: float, douglas: bool):
        super().__init__(core, dt, theta, t_mid, douglas)
        r, _ = core._rq_at(t_mid)
        self.discount = 1.0
        cm, c0, cp = self.a1
        self.a1 = (cm, c0 - r, cp)
        if self.degenerate:
            cm0, c00, cp0 = self.a1_v0
            self.a1_v0 = (cm0, c00 - r, cp0)
        self.tri_s = _transposed_tridiagonal(*core._tri_S(dt, theta, t_mid))


def _refuse_core(core) -> None:
    if core.use_sparse or core._semi_lagrangian_v:
        # TODO(unified event statistics): the core's per-claim sparse / semi-Lagrangian sweeps need their own transposes
        raise AdjointUnavailable("undefined: the DCN ADI forward pass transposes the batched tridiagonal step; the "
                                 "sparse / semi-Lagrangian sweep has none")


def _readout_weights_2d(core, log_spot: float, v0: float, surface: np.ndarray) -> np.ndarray:
    """``core.interpolate``'s weights at ``(log_spot, v0)``: probed on the four nodes of its cell, checked on the
    read surface."""
    X, V = core.X_grid, core.V_grid
    x_val = max(min(log_spot, X[-1]), X[0])
    v_val = max(min(v0, V[-1]), V[0])
    i0 = max(min(int(np.searchsorted(X, x_val, side="right") - 1), len(X) - 2), 0)
    j0 = max(min(int(np.searchsorted(V, v_val, side="right") - 1), len(V) - 2), 0)
    weights = np.zeros((len(X), len(V)))
    unit = np.zeros_like(weights)
    for i in (i0, i0 + 1):
        for j in (j0, j0 + 1):
            unit[i, j] = 1.0
            weights[i, j] = core.interpolate(unit, log_spot, v0)
            unit[i, j] = 0.0
    value = core.interpolate(surface, log_spot, v0)
    if not is_close(float(np.sum(weights * surface)), value, rel_tol=Tolerance.RELATIVE,
                    abs_tol=Tolerance.ZERO * max(1.0, float(np.max(np.abs(surface))))):
        raise AdjointUnavailable("undefined: the ADI readout is not linear in the surface")
    return weights


def _forward_adi(log: dict, *, rows: DCNRows, weights: np.ndarray, read_ki: bool) -> DCNForwardResult:
    core, product = log["core"], log["product"]
    _refuse_core(core)
    df = log["df"]
    ctx = rows.ctx
    n_x, n_v = core.N_S, core.N_V
    stages = _Stages(log["apply"], n_x, (n_x, 1))   # the 2D masks are (n_x, 1)
    result = DCNForwardResult(rows)
    p = np.zeros((2, n_x, n_v))
    p[KI if read_ki else ALIVE] = weights
    steps: Dict[tuple, CoreStepTranspose] = {}
    for entry in reversed(log["entries"]):
        kind = entry[0]
        if kind == "step":
            _, dt, tau, t_mid, theta, douglas, t_new = entry
            key = (dt, theta, t_mid, douglas)
            op = steps.get(key)
            if op is None:
                op = steps[key] = CoreStepTranspose(core, dt, theta, t_mid, douglas)
            out, dlow, dhigh = op.apply(p.transpose(1, 2, 0))
            p = np.ascontiguousarray(out.transpose(2, 0, 1))
            low = log["low"](tau)
            df_t = float(df(t_new))
            # the hooks' far high-spot rows are 0 on both surfaces: no event (alive matures, ki settles knocked in)
            result.lower_edge(t_new, float(dlow[ALIVE]), float(dlow[KI]), low, df_t)
            result.upper_edge(float(dhigh[ALIVE]), float(dhigh[KI]), (), None, None, df_t)
        elif kind == "event":
            _, t_node, ki_time, j_obs, cpn, ko = entry
            p, masses = stages(cpn is not None, ko is not None).transpose(p)
            result.event(t_node, ki_time, j_obs, masses, float(df(t_node)))
        else:   # pragma: no cover - a log bug
            raise PricingError(f"unknown DCN ADI log entry {kind!r}")
    terminal = log["terminal"]
    _, T, j_obs, cpn, ko = terminal
    p, masses = stages(cpn is not None, ko is not None).transpose(p)
    result.event(T, float(ctx.times[-1]), j_obs, masses, float(df(T)))
    result.terminal(p, log["v1_terminal"], float(df(T)))
    return result


def dcn_adi_statistics(solver, product, pricing_env, *, lifecycle_state=None):
    """``(pv, EventStatistics or None)`` of ``HestonDCNPDESolver``: ``pv`` from the price's own run (``price()``
    bitwise; the march records its steps and events), the statistics from the forward pass over that march.

    Raises:
        AdjointUnavailable: the core's sweep has no transpose here (sparse / semi-Lagrangian).
    """
    from quantark.asset.equity.engine.dcn_valuation import observe_valuation_row, resolve_dcn_lifecycle_state
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
    from quantark.asset.equity.product.option.dcn_option import DCNOption

    if not isinstance(product, DCNOption):
        raise PricingError("HestonDCNPDESolver only supports DCNOption")
    state = resolve_dcn_lifecycle_state(product, pricing_env, lifecycle_state)
    provenance = StatisticsProvenance(solver._price_model, solver._price_model)
    if state is not None and state.terminal:
        pv = solver.price(product, pricing_env, lifecycle_state=lifecycle_state)
        return pv, terminated_dcn_statistics(product, state, pv=pv, provenance=provenance)
    log: dict = {}
    contingent = solver._price_contingent(product, pricing_env, None, log=log)
    pv = contingent.pv if state is None else contingent.pv + state.pending_pv
    ctx = build_dcn_grid_context(product)
    receivables = _receivables(product, state)
    row = observe_valuation_row(product, pricing_env, ctx)
    if row is not None and row.ends_contract:
        return pv, decided_row_statistics(product, pricing_env, ctx, row, pv=pv, receivables=receivables,
                                          provenance=provenance)
    rows = DCNRows(product, pricing_env, ctx)
    core = log["core"]
    read_ki = log["read_ki"]
    weights = _readout_weights_2d(core, float(np.log(float(pricing_env.spot))), float(solver.model_params.v0),
                                  log["read_surface"])
    result = _forward_adi(log, rows=rows, weights=weights, read_ki=read_ki)
    stats = _assemble(rows, result, pv=pv, coupon_today=bool(row is not None and row.coupon_paid),
                      receivables=receivables, provenance=provenance)
    return pv, stats


# --- the backward side of the adjoint identity (spec AC14) ---------------------------------------------------------

class DCNClaim:
    """A claim marched backward on a DCN march's own grid: the price's sources replaced by the claim's.

    ``event(j_obs, t)``: the ``(coupon, knock-out)`` cash at a monthly row (an active branch with zero cash is ``0.0``,
    an inactive one ``None``, exactly as the row's own); ``lower(t)`` / ``upper(t)``: the ``(alive, ki)`` edge rows;
    ``terminal``: ``(alive, ki)`` before the final date's events. ``ki_reset(t, t_ki)`` (a knock-in claim): the value a
    first knock-in at the daily date ``t_ki`` (march time ``t``) pays, written over the knocked-in surface just before
    that date's events -- the knock-out at the same date then takes it back (KO wins the tie), and the alive surface's
    knock-in projection reads it.
    """

    def __init__(self, event: Callable, lower: Callable, upper: Callable, terminal: tuple,
                 ki_reset: Optional[Callable] = None):
        self.event, self.lower, self.upper, self.terminal = event, lower, upper, terminal
        self.ki_reset = ki_reset


def unit_claims(rows: DCNRows, df: Callable, *, s_terminal: np.ndarray, lower_value: Callable, upper_terms: Callable,
                price_amounts: Callable, price_rows: Callable, shape: tuple) -> Dict[str, DCNClaim]:
    """The termination outcomes as unit claims (probability = the claim's readout, spec AC14) and the price itself:
    ``ko_total`` (every knock-out row), ``ko_<j>`` / ``coupon_<j>`` per monthly row, ``maturity``, ``ki_settlement``,
    ``ki_settlement_cash`` (the loss leg, buyer view), ``ki_total`` / ``ki_time`` (P(first knock-in before any
    knock-out) and E[its time x 1{...}]) and ``price`` (the product's own cash, unsigned).

    ``upper_terms(t)``: ``(coupons, ko_row, ki_ko_row)`` of the upper edge at ``t``; ``lower_value(t)``: the loss row.
    """
    ctx, product = rows.ctx, rows.product
    n_obs = rows.times.size
    T = float(ctx.times[-1])
    zeros = np.zeros(shape)
    loss_T = (-(product.notional / product.initial_price) * product.participation
              * np.maximum(product.k_loss - s_terminal, 0.0) * df(ctx.loss_pay_time) / df(T))
    ones_T = np.full(shape, 1.0 / float(df(T)))

    def branches(j, t, ko_rows=(), coupon_rows=()):
        cpn = None if not ctx.obs_is_coupon[j] else (1.0 / float(df(t)) if j in coupon_rows else 0.0)
        ko = None if not ctx.obs_is_ko[j] else (1.0 / float(df(t)) if j in ko_rows else 0.0)
        return cpn, ko

    def upper_unit(alive_hit: Callable, ki_hit: Callable):
        def upper(t):
            coupons, ko_row, ki_ko_row = upper_terms(t)
            unit = 1.0 / float(df(t))
            return (unit if alive_hit(coupons, ko_row) else 0.0, unit if ki_hit(ki_ko_row) else 0.0)
        return upper

    none = lambda t: (0.0, 0.0)   # noqa: E731
    claims: Dict[str, DCNClaim] = {
        "price": DCNClaim(price_amounts, lambda t: (price_rows(t)[0][0], price_rows(t)[0][1]),
                          lambda t: price_rows(t)[1], (zeros, loss_T)),
        "ko_total": DCNClaim(lambda j, t: branches(j, t, ko_rows=range(n_obs)), none,
                             upper_unit(lambda c, k: k is not None, lambda k: k is not None), (zeros, zeros)),
        "maturity": DCNClaim(lambda j, t: branches(j, t), none,
                             upper_unit(lambda c, k: k is None, lambda k: False), (ones_T, zeros)),
        "ki_settlement": DCNClaim(lambda j, t: branches(j, t), lambda t: (1.0 / float(df(t)),) * 2,
                                  upper_unit(lambda c, k: False, lambda k: k is None), (zeros, ones_T)),
        "ki_settlement_cash": DCNClaim(lambda j, t: branches(j, t), lambda t: (lower_value(t),) * 2, none,
                                       (zeros, loss_T)),
    }
    ki_times = rows.ki_times

    def next_ki(t):
        k = int(np.searchsorted(ki_times, t + Tolerance.ZERO, side="right"))
        return None if k >= ki_times.size else float(ki_times[k])

    def ki_claim(pays: Callable):
        """A first knock-in paying ``pays(t_ki)`` (probability units) at its date: the reset at each daily date, and
        the alive loss row (a knock-in certain at the next daily date)."""
        def lower(t):
            nxt = next_ki(t)
            return (0.0 if nxt is None else pays(nxt) / float(df(t)), 0.0)
        return DCNClaim(lambda j, t: branches(j, t), lower, none, (zeros, zeros),
                        ki_reset=lambda t, t_ki: pays(t_ki) / float(df(t)))

    claims["ki_total"] = ki_claim(lambda t_ki: 1.0)
    claims["ki_time"] = ki_claim(lambda t_ki: t_ki)
    for row in range(n_obs):
        claims[f"ko_{row}"] = DCNClaim(
            lambda j, t, row=row: branches(j, t, ko_rows=(row,)), none,
            upper_unit(lambda c, k, row=row: k == row, lambda k, row=row: k == row), (zeros, zeros))
        claims[f"coupon_{row}"] = DCNClaim(
            lambda j, t, row=row: branches(j, t, coupon_rows=(row,)), none,
            upper_unit(lambda c, k, row=row: row in c, lambda k: False), (zeros, zeros))
    return claims


def backward_claims_1d(solver, product, pricing_env, *, lifecycle_state=None) -> Dict[str, float]:
    """Every ``unit_claims`` claim marched backward on the 1D core with the march's own primitives -- the banded
    Crank-Nicolson advance (``banded_advance``, same set, ``dt``, ``θ``), the family's event projection
    (``DCNEvents.apply_amounts``) -- and read at the spot as the price is: ``{claim: readout}`` (unsigned)."""
    from .core.backends import banded_advance
    from .core.groups import GroupState
    from .core.march import march
    from .core.prepare import Operation, OutputRequest, market_scope, prepare

    def run():
        problem = prepare(solver, product, pricing_env, OutputRequest(Operation.PRICE),
                          lifecycle_state=lifecycle_state, frozen=solver._frozen_base_layout,
                          session=solver._session_artefacts)
        if problem.terminal is not None:
            raise PricingError("a terminal outcome has no march")
        ev, layout = problem.events, problem.layout
        t = np.asarray(layout.time.t, dtype=float)
        N = t.size
        n = len(layout.spatial.x)
        rows = DCNRows(product, pricing_env, ev.ctx)

        def upper_terms(tt):
            _, (coupons, ko_row), ki_ko_row = ev.edge_terms(tt)
            return coupons, ko_row, ki_ko_row

        claims = unit_claims(rows, ev.df, s_terminal=np.asarray(ev.s, dtype=float),
                             lower_value=lambda tt: ev.edge_terms(tt)[0], upper_terms=upper_terms,
                             price_amounts=lambda j, tt: ev.amounts(j, tt), price_rows=ev.rows, shape=(n,))
        names = list(claims)
        group = problem.groups[0]
        claim_group = type(group)("claims", tuple(names), group.backend, group.policy)
        coeffs, thetas = problem.coefficients, [float(x) for x in problem.theta_schedule]
        last = len(ev.ctx.obs_cols) - 1
        v0 = np.zeros((n, len(names)))
        v1 = np.zeros((n, len(names)))

        def events(v0, v1, j_obs, tt, t_ki):
            for c, name in enumerate(names):
                claim = claims[name]
                if claim.ki_reset is not None:
                    v1[:, c] = claim.ki_reset(tt, t_ki)
                cpn, ko = (None, None) if j_obs is None else claim.event(j_obs, tt)
                v0[:, c], v1[:, c] = ev.apply_amounts(v0[:, c].copy(), v1[:, c].copy(), cpn, ko)
            return v0, v1

        for c, name in enumerate(names):
            v0[:, c], v1[:, c] = claims[name].terminal
        time_of_node = {node: float(tc) for tc, node in ev.node_of.items()}
        v0, v1 = events(v0, v1, last, float(t[-1]), float(ev.ctx.times[-1]))
        with market_scope(problem.solver, problem.env, problem.market_fps):
            columns = {}
            for j in range(N - 2, -1, -1):
                tj = float(t[j])
                lower = np.array([claims[name].lower(tj) for name in names])     # (K, 2): (alive, ki)
                upper = np.array([claims[name].upper(tj) for name in names])
                k = coeffs.set_at(j)
                s = coeffs.step_set(k, problem.factorizations)
                theta = thetas[j]
                if group.policy.zero_diffusion_theta_one and coeffs.is_zero_diffusion(k):
                    theta = 1.0
                new = []
                for surf, v in ((ALIVE, v0), (KI, v1)):
                    out = banded_advance(GroupState(claim_group, v), s, layout.time.dt[j], theta,
                                         (lower[:, surf], upper[:, surf]))
                    col = np.empty_like(v)
                    col[0], col[-1], col[1:-1] = lower[:, surf], upper[:, surf], out
                    new.append(col)
                v0, v1 = new
                if j > 0 and j in ev.event_nodes:
                    v0, v1 = events(v0, v1, ev.obs_at_node.get(j), tj, time_of_node[j])
            spot = float(pricing_env.spot)
            read_ki = bool(product.knocked_in_at_valuation) or (ev.ctx.valuation_monitored
                                                                and spot <= product.ki_barrier)
            surface = v1 if read_ki else v0
            for c, name in enumerate(names):
                columns[name] = float(problem.readout.price(np.ascontiguousarray(surface[:, c]), spot, layout))
        return columns

    if hasattr(solver, "_with_surface"):
        return solver._with_surface(pricing_env, run)
    return run()


def backward_claims_adi(solver, product, pricing_env, names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """``unit_claims`` marched backward by replaying ``HestonDCNPDESolver``'s own march (its recorded steps through
    the core's own ``_douglas_step`` / ``_cs_step`` with the claim's rows in the boundary hook, its events through
    ``_apply_events``) and read as the price is: ``{claim: readout}`` (unsigned). ``names``: a subset of the claims."""
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context

    log: dict = {}
    solver._price_contingent(product, pricing_env, None, log=log)
    core = log["core"]
    ctx = build_dcn_grid_context(product)
    rows = DCNRows(product, pricing_env, ctx)
    df = log["df"]
    n_x, n_v = core.N_S, core.N_V
    T = float(ctx.times[-1])
    s_mesh = np.repeat(np.asarray(core.S_grid, dtype=float)[:, None], n_v, axis=1)

    def price_rows(t):
        low = log["low"](T - t)
        return (low, low), (0.0, 0.0)

    obs_at = {}
    claims = unit_claims(rows, df, s_terminal=s_mesh, lower_value=lambda t: log["low"](T - t),
                         upper_terms=lambda t: ((), None, None),
                         price_amounts=lambda j, t: obs_at[(j, t)], price_rows=price_rows, shape=(n_x, n_v))
    for entry in log["entries"]:
        if entry[0] == "event" and entry[3] is not None:
            obs_at[(entry[3], entry[1])] = (entry[4], entry[5])
    _, _, j_last, cpn_last, ko_last = log["terminal"]
    obs_at[(j_last, T)] = (cpn_last, ko_last)
    apply = log["apply"]
    out: Dict[str, float] = {}
    for name in (names if names is not None else list(claims)):
        claim = claims[name]
        v0, v1 = (np.array(a, dtype=float) for a in claim.terminal)
        if claim.ki_reset is not None:
            v1 = np.full_like(v1, claim.ki_reset(T, float(ctx.times[-1])))
        cpn, ko = claim.event(j_last, T)
        v0, v1 = apply(v0, v1, cpn, ko)

        def hook_of(surface):
            def hook(U, tau):
                t = min(max(T - float(tau), 0.0), T)
                U[0, :] = claim.lower(t)[surface]
                U[-1, :] = claim.upper(t)[surface]
                return U
            return hook

        hooks = (hook_of(KI), hook_of(ALIVE))
        for entry in log["entries"]:
            if entry[0] == "step":
                _, dt, tau, t_mid, theta, douglas, _ = entry
                surfaces = []
                for U, hook in zip((v1, v0), hooks):
                    core._boundary_hook = hook
                    step = core._douglas_step if douglas else core._cs_step
                    surfaces.append(step(U, dt, tau, theta, t_mid))
                core._boundary_hook = None
                core._S_tri_cache.clear()
                v1, v0 = surfaces
            else:
                _, t_node, t_ki, j_obs, _, _ = entry
                if claim.ki_reset is not None:
                    v1 = np.full_like(v1, claim.ki_reset(t_node, t_ki))
                c, k = (None, None) if j_obs is None else claim.event(j_obs, t_node)
                if j_obs is None:
                    v0 = apply(v0, v1, None, None)[0]
                else:
                    v0, v1 = apply(v0, v1, c, k)
        surface = v1 if log["read_ki"] else v0
        out[name] = core.interpolate(surface, float(np.log(float(pricing_env.spot))), float(solver.model_params.v0))
    return out

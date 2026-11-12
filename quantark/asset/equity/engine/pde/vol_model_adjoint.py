"""The forward adjoint pass of the native ADI statistics march (unified event statistics spec T14; DESIGN §4.5 option A).

``vol_model_statistics.native_adi`` marches the PDE 1D family's statistics columns backward on the Heston / SLV ADI
mesh: per step one batched ADI column step (``adi_columns.step_columns``: Douglas, or Craig-Sneyd with its
mixed-derivative corrector; per-event damped Douglas steps and the terminal half steps included) with the family's
Dirichlet spot rows, then the family's event stages at the step's node, variance column by variance column
(FIRST_PASSAGE with each column's own coefficients).

This module runs the exact transpose of that linear march forward in time, from the readout's own weights at
``(spot, v0)``:

* a step: the transpose of every operation of ``step_columns``' batched path in reverse order -- the explicit
  predictor ``U + dt (A1 + A2 + A0) U``, the implicit spot and variance sweeps (transposed tridiagonal solves, the
  same matrices), the boundary copies of ``bc`` and the corrector -- plus the duals of the step's two spot rows
  (``low``, ``high``), summed over every place the rows enter (they are affine constants of the march);
* an event stage: the PDE 1D pass's stage plans (``adjoint_stages`` on the family's statistics class, the family's
  own primitives evaluated on the identity), applied to every variance column; FIRST_PASSAGE's rank-one correction
  per column from the solver's own per-column geometry (``adjoint.first_passage`` on the columnar coefficients);
* the masses: the knock-in copy / first-passage / knock-in-side edge / clean knock-out masses per march node, read
  exactly as the PDE 1D pass reads them (``adjoint.knock_in_from_forward_pass`` builds the table).

The terminal damping's half step adds one node (``T - dt/2``) to the pass's time axis; the knock-in table is built on
that axis (its rows stay on the product's KI schedule, or on trading days).

The scalar / sparse column sweeps and the semi-Lagrangian variance transport (``column_backend='scalar'``,
``v_drift_scheme='semi_lagrangian'``) run the core's own per-claim steps, which this pass does not transpose: they
are refused with a reason (``AdjointUnavailable``), never approximated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from quantark.util.exceptions import PricingError
from quantark.util.numerical import Tolerance, is_close
from quantark.util.numerical.tridiag import solve_tridiag_batch

from .core.statistics.adjoint import (
    ALIVE,
    KI,
    AdjointOps,
    AdjointUnavailable,
    ColumnarFirstPassage,
    ForwardResult,
    KnockIn,
    Seed,
    SurfaceMap,
)

__all__ = ["ADIStep", "ADIEvents", "adi_forward_pass", "readout_weights_2d", "StepTranspose"]

#: ``step_columns``' threshold for a mixed-derivative term (``abs(core.rho) <= 1e-12``: no ``A0``)
_RHO_OFF = 1e-12


@dataclass(frozen=True, eq=False)
class ADIStep:
    """One backward ADI column step as the march took it: ``j`` its march index, ``node`` the time of the column it
    produced (``core.T - tau``), the step's ``dt`` / ``theta`` / ``t_mid`` / ``douglas``, the statistics surfaces'
    spot rows (``low`` / ``high``, ``2 n_cols`` each: not-knocked-in then knocked-in) and whether the price's own
    not-knocked-in knock-in-side row knocks in on this step (``rows.knocks_in``, D23)."""

    j: int
    node: float
    dt: float
    tau: float
    theta: float
    t_mid: float
    douglas: bool
    low: np.ndarray
    high: np.ndarray
    knocks_in: bool
    #: the terminal damping's first half step: it produces the half node ``T - dt/2`` (not a march node)
    half: bool


@dataclass(frozen=True, eq=False)
class ADIEvents:
    """The event stages the march applied at index ``j`` (``half``: the terminal damping's half node, continuous
    knock-in only)."""

    j: int
    half: bool


# --- the ADI column step, transposed ----------------------------------------------------------------------------

def _transposed_tridiagonal(a, b, c):
    """``(sub, diag, sup)`` of ``T^T`` for ``T`` in ``solve_tridiag_batch``'s full-length convention (row ``i``:
    ``a[i] x[i-1] + b[i] x[i] + c[i] x[i+1]``): ``sub'[i] = c[i-1]``, ``sup'[i] = a[i+1]``."""
    a, b, c = (np.asarray(x, dtype=float) for x in (a, b, c))
    sub = np.zeros_like(a)
    sup = np.zeros_like(c)
    sub[..., 1:] = c[..., :-1]
    sup[..., :-1] = a[..., 1:]
    return sub, b, sup


class StepTranspose:
    """The transpose of ``adi_columns.step_columns``' batched path for one step's ``(dt, theta, t_mid, douglas)``.

    ``apply(P)``: ``P`` the duals on the step's output columns ``(n_s, n_v, n_c)``; returns the duals on its input
    columns and on its rows ``low`` / ``high`` (``(n_c,)`` each)."""

    def __init__(self, core, dt: float, theta: float, t_mid: float, douglas: bool):
        self.core = core
        self.dt, self.theta, self.douglas = float(dt), float(theta), bool(douglas)
        r, q = core._rq_at(t_mid)
        self.discount = np.exp(-r * dt)
        n_s, n_v = core.N_S, core.N_V
        self.n_s, self.n_v = n_s, n_v
        self.degenerate = bool(core._degenerate_v0)
        V = core.V_grid
        lev = np.asarray(core._L(t_mid), dtype=float)
        diffusion = .5 * lev[:, None] ** 2 * V[None, 1:-1]
        convection = (r - q) - diffusion
        if core._uniform:
            dx = core.dx
            xx = (np.full(n_s - 2, 1.0 / dx ** 2), np.full(n_s - 2, -2.0 / dx ** 2), np.full(n_s - 2, 1.0 / dx ** 2))
            x1 = (np.full(n_s - 2, -1.0 / (2 * dx)), np.zeros(n_s - 2), np.full(n_s - 2, 1.0 / (2 * dx)))
            dV = core.dV
            v1 = (np.full(n_v - 2, -1.0 / (2 * dV)), np.zeros(n_v - 2), np.full(n_v - 2, 1.0 / (2 * dV)))
        else:
            xx = tuple(np.asarray(w, dtype=float) for w in core._xx)
            x1 = tuple(np.asarray(w, dtype=float) for w in core._x1)
            v1 = tuple(np.asarray(w, dtype=float) for w in core._v1) if abs(core.rho) > _RHO_OFF else None
        # A1: the spot generator (no reaction) on the interior; the degenerate v = 0 column's drift term
        self.a1 = tuple(diffusion * w2[:, None] + convection * w1[:, None] for w2, w1 in zip(xx, x1))
        self.a1_v0 = tuple((r - q) * w1 for w1 in x1) if self.degenerate else None
        # A2: the shared variance generator on the interior; the degenerate row
        sub, diag, sup, *_ = core._build_v_generator_coefficients()
        self.a2 = (np.asarray(sub, dtype=float), np.asarray(diag, dtype=float), np.asarray(sup, dtype=float))
        self.a2_v0 = (core.kappa * core.theta / (V[1] - V[0])) if self.degenerate else None
        # A0: the mixed derivative
        self.rho_on = abs(core.rho) > _RHO_OFF
        if self.rho_on:
            self.a0 = core.rho * core.sig_eff * lev[:, None] * V[None, 1:-1]
            self.x1, self.v1 = x1, v1
        # the implicit sweeps, as step_columns builds them, transposed
        a_s, b_s, c_s = core._tri_S(dt, theta, t_mid)
        b_s = b_s.copy()
        b_s[:, 1:-1] -= theta * dt * r
        self.tri_s = _transposed_tridiagonal(a_s, b_s, c_s)                     # (n_v, n_s) each
        self.tri_v = _transposed_tridiagonal(*core._tri_V(dt, theta))           # (n_v,) each

    # --- the operators' transposes ---------------------------------------------------------------------------

    def _a1_t(self, P):
        out = np.zeros_like(P)
        cm, c0, cp = self.a1
        Q = P[1:-1, 1:-1]
        out[:-2, 1:-1] += cm[:, :, None] * Q
        out[1:-1, 1:-1] += c0[:, :, None] * Q
        out[2:, 1:-1] += cp[:, :, None] * Q
        if self.degenerate:
            cm0, c00, cp0 = self.a1_v0
            Q0 = P[1:-1, 0]
            out[:-2, 0] += cm0[:, None] * Q0
            out[1:-1, 0] += c00[:, None] * Q0
            out[2:, 0] += cp0[:, None] * Q0
        return out

    def _a2_t(self, P):
        out = np.zeros_like(P)
        sub, diag, sup = self.a2
        Q = P[1:-1, 1:-1]
        out[1:-1, :-2] += sub[None, :, None] * Q
        out[1:-1, 1:-1] += diag[None, :, None] * Q
        out[1:-1, 2:] += sup[None, :, None] * Q
        if self.degenerate:
            g = self.a2_v0 * P[1:-1, 0]
            out[1:-1, 1] += g
            out[1:-1, 0] -= g
        return out

    def _a0_t(self, P):
        out = np.zeros_like(P)
        if not self.rho_on:
            return out
        xm, x0, xp = self.x1
        vm, v0, vp = self.v1
        Q = self.a0[:, :, None] * P[1:-1, 1:-1]
        X = np.zeros((P.shape[0] - 2, P.shape[1], P.shape[2]))
        X[:, :-2] += vm[None, :, None] * Q
        X[:, 1:-1] += v0[None, :, None] * Q
        X[:, 2:] += vp[None, :, None] * Q
        out[:-2] += xm[:, None, None] * X
        out[1:-1] += x0[:, None, None] * X
        out[2:] += xp[:, None, None] * X
        return out

    # --- the step ----------------------------------------------------------------------------------------------

    def apply(self, P: np.ndarray):
        n_s, n_v, n_c = P.shape
        dt, theta = self.dt, self.theta
        rows = {"lo": np.zeros(n_c), "hi": np.zeros(n_c)}         # duals of the scaled rows low / discount, high / ...

        def bc_t(Q):
            Q = Q.copy()
            rows["lo"] += Q[0].sum(axis=0)
            rows["hi"] += Q[-1].sum(axis=0)
            Q[0] = 0.0
            Q[-1] = 0.0
            Q[:, -2] += Q[:, -1]
            Q[:, -1] = 0.0
            if not self.degenerate:
                Q[:, 1] += Q[:, 0]
                Q[:, 0] = 0.0
            return Q

        def solve_s_t(Q):
            Q = bc_t(Q)
            sub, diag, sup = (np.broadcast_to(m, (n_c, n_v, n_s)).reshape(-1, n_s) for m in self.tri_s)
            R = solve_tridiag_batch(sub, diag, sup, Q.transpose(2, 1, 0).reshape(-1, n_s))
            R = R.reshape(n_c, n_v, n_s).transpose(2, 1, 0).copy()
            rows["lo"] += R[0].sum(axis=0)
            rows["hi"] += R[-1].sum(axis=0)
            R[0] = 0.0
            R[-1] = 0.0
            return R, -theta * dt * R

        def solve_v_t(Q):
            Q = bc_t(Q)
            sub, diag, sup = (np.broadcast_to(m, (n_c, n_s, n_v)).reshape(-1, n_v) for m in self.tri_v)
            R = solve_tridiag_batch(sub, diag, sup, Q.transpose(2, 0, 1).reshape(-1, n_v))
            R = R.reshape(n_c, n_s, n_v).transpose(1, 2, 0).copy()
            if not self.degenerate:
                R[:, 0] = 0.0
            R[:, -1] = 0.0
            return R, -theta * dt * R

        dlow = P[0].sum(axis=0)
        dhigh = P[-1].sum(axis=0)
        P = P.copy()
        P[0] = 0.0
        P[-1] = 0.0
        P *= self.discount
        p_y0 = np.zeros_like(P)
        p_a1 = np.zeros_like(P)
        p_a2 = np.zeros_like(P)
        p_a0 = np.zeros_like(P)
        p_y2 = None
        if self.douglas:
            p_y2 = P
        else:
            p_z1, g = solve_v_t(P)
            p_a2 += g
            p_c, g = solve_s_t(p_z1)
            p_a1 += g
            if self.rho_on:
                p_c = bc_t(p_c)
                p_y0 += p_c
                p_y2 = .5 * dt * self._a0_t(p_c)
                p_a0 -= .5 * dt * p_c
            else:
                p_y0 += p_c
        if p_y2 is not None:
            p_y1, g = solve_v_t(p_y2)
            p_a2 += g
            p_y0b, g = solve_s_t(p_y1)
            p_a1 += g
            p_y0 += p_y0b
        out = (p_y0 + self._a1_t(dt * p_y0 + p_a1) + self._a2_t(dt * p_y0 + p_a2)
               + self._a0_t(dt * p_y0 + p_a0))
        return out, dlow + rows["lo"] / self.discount, dhigh + rows["hi"] / self.discount


# --- the readout ------------------------------------------------------------------------------------------------

def readout_weights_2d(core, v0: float, prepared, read_block: np.ndarray) -> np.ndarray:
    """The statistics readout's weight on every ``(spot, variance)`` node: ``native_adi`` reduces the variance axis
    by linear interpolation at ``v0`` and reads the spot column by the linear readout; the weights are the variance
    weights times the spot readout's (probed node by node), checked on the read block's own columns."""
    V = core.V_grid
    n_v = V.size
    i = max(0, min(int(np.searchsorted(V, v0)) - 1, n_v - 2))
    w = (v0 - V[i]) / (V[i + 1] - V[i])
    wv = np.zeros(n_v)
    wv[i], wv[i + 1] = 1.0 - w, w
    readout, spot, layout = prepared.readout, prepared.env.spot, prepared.layout
    n_s = read_block.shape[0]
    unit = np.zeros(n_s)
    wx = np.empty(n_s)
    for k in range(n_s):
        unit[k] = 1.0
        wx[k] = float(readout.price(unit, spot, layout))
        unit[k] = 0.0
    weights = wx[:, None] * wv[None, :]
    for c in range(read_block.shape[2]):
        block = read_block[:, :, c]
        value = float(readout.price((1 - w) * block[:, i] + w * block[:, i + 1], spot, layout))
        dual = float(np.sum(weights * block))
        if not is_close(dual, value, rel_tol=Tolerance.RELATIVE,
                        abs_tol=Tolerance.ZERO * max(1.0, float(np.max(np.abs(block))))):
            raise AdjointUnavailable(f"undefined: the ADI readout is not linear in the column (weights give "
                                     f"{dual!r}, the readout {value!r})")
    return weights


# --- the event stages, per variance column -------------------------------------------------------------------

def _apply(op, X):
    if op.diag is not None:
        return op.diag[:, None] * X
    return np.asarray(op.mt @ X, dtype=float)


def _knock_in_t(stage: KnockIn, P: np.ndarray):
    """``KnockIn.transpose`` on every variance column: ``(P_in, transfer_copy, transfer_fp)``."""
    pa = P[ALIVE]
    fp_transfer = None
    if stage.fp is not None:
        if not isinstance(stage.fp, ColumnarFirstPassage):
            raise PricingError("the ADI forward pass expects per-column first-passage coefficients")
        G, K = stage.fp.g, stage.fp.k
        moved = np.einsum("ik,ik->k", G, pa)
        cols = np.arange(pa.shape[1])
        pa = pa.copy()
        pa[K, cols] -= moved
        fp_transfer = np.zeros_like(pa)
        fp_transfer[K, cols] = moved
    take = _apply(stage.take, pa)
    out = np.empty_like(P)
    out[ALIVE] = _apply(stage.copy, pa)
    out[KI] = P[KI] + take
    if fp_transfer is not None:
        out[KI] += fp_transfer
    return out, take, fp_transfer


@dataclass(eq=False)
class _Masses:
    ki_copy: np.ndarray
    ki_fp: np.ndarray
    clean_ko: np.ndarray
    ki_edge: np.ndarray


def _events_t(stages, P: np.ndarray, at: int, masses: _Masses) -> np.ndarray:
    """The PDE 1D pass's ``_Pass.events`` on every variance column (the same tracer bookkeeping: a knock-out at the
    knock-in's instant takes the knocked-in mass back)."""
    P = P.copy()
    tracers: List[list] = []
    for stage in reversed(stages):
        if isinstance(stage, SurfaceMap):
            if stage.breach is not None:
                if stage.surface == ALIVE:
                    masses.clean_ko[at] += float(np.sum(stage.breach @ P[ALIVE]))
                for tracer in tracers:
                    masses.clean_ko[at] += float(np.sum(stage.breach @ tracer[1][stage.surface]))
            P[stage.surface] = _apply(stage.op, P[stage.surface])
            for tracer in tracers:
                tracer[1][stage.surface] = _apply(stage.op, tracer[1][stage.surface])
        elif isinstance(stage, Seed):
            P[stage.surface] = 0.0
            for tracer in tracers:
                tracer[1][stage.surface] = 0.0
        elif isinstance(stage, KnockIn):
            for tracer in tracers:
                tracer[1] = _knock_in_t(stage, tracer[1])[0]
            P, take, fp_transfer = _knock_in_t(stage, P)
            moved = np.zeros_like(P)
            moved[KI] = take
            tracers.append(["copy", moved])
            if fp_transfer is not None:
                moved = np.zeros_like(P)
                moved[KI] = fp_transfer
                tracers.append(["fp", moved])
        else:   # pragma: no cover - a stage plan bug
            raise PricingError(f"unknown adjoint stage {stage!r}")
    for kind, moved in tracers:
        mass = float(np.sum(moved[KI]))
        if kind == "copy":
            masses.ki_copy[at] += mass
        else:
            masses.ki_fp[at] += mass
    return P


# --- the pass ---------------------------------------------------------------------------------------------------

def adi_forward_pass(*, owner, family, prepared, core, statistics, log, fp_2d, captures_have_t0_pre: bool,
                     read_block: np.ndarray, reconstruct: bool = False,
                     affine_events: Optional[Callable] = None) -> ForwardResult:
    """The forward pass over a native ADI statistics march (``log``: its steps and event stages in march order).

    ``affine_events(entry)``: with ``reconstruct``, the march's own event stages at ``entry`` applied to zero
    statistics blocks ``(n_s, n_v, n_cols)`` (the reconstruction's event sources).

    Raises:
        AdjointUnavailable: a column sweep the pass does not transpose (scalar / sparse / semi-Lagrangian).
    """
    if getattr(owner, "column_backend", "auto") == "scalar" or core.use_sparse or core._semi_lagrangian_v:
        # TODO(unified event statistics): the core's per-claim sweeps (scalar backend, sparse factors, the
        # semi-Lagrangian variance transport) need their own transposes
        raise AdjointUnavailable("undefined: the ADI forward adjoint pass transposes the batched column step; the "
                                 "scalar / sparse / semi-Lagrangian sweep has none")
    t = np.asarray(prepared.layout.time.t, dtype=float)
    N = len(t) - 1
    halves = [e for e in log if isinstance(e, ADIStep) and e.half]
    damped = bool(halves)
    if damped:
        half_node = float(halves[0].node)
        times = np.concatenate([t[:N], [half_node], t[N:]])
    else:
        times = t.copy()

    def node_of(j: int) -> int:
        """The pass axis index of the march's node ``j`` (the terminal shifts past the half node)."""
        return j + 1 if (damped and j == N) else j

    n_nodes = times.size
    masses = _Masses(np.zeros(n_nodes), np.zeros(n_nodes), np.zeros(n_nodes), np.zeros(n_nodes))
    ops = AdjointOps(core.N_S)
    reverse = bool(statistics.product.is_reverse)
    weights = readout_weights_2d(core, float(owner.model_params.v0), prepared, read_block)
    P = np.zeros((2, core.N_S, core.N_V))
    P[KI if statistics.knocked_in_at_valuation else ALIVE] = weights
    recon = np.zeros(statistics.n_cols) if reconstruct else None
    steps = {}
    previous_fp = family._ki_fp

    def stages_at(entry):
        if entry.half:
            family._ki_fp = None
            return [statistics._adjoint_ki_columns(entry.j, ops)] if prepared.solve_state["ki_continuous"] else []
        family._ki_fp = fp_2d
        return statistics.adjoint_stages(entry.j, ops)

    def event_entry(entry, P):
        at = N if entry.half else node_of(entry.j)
        if reconstruct:
            zero = affine_events(entry)
            recon[...] += (np.einsum("ik,ikc->c", P[ALIVE], zero["stats_v0"])
                           + np.einsum("ik,ikc->c", P[KI], zero["stats_v1"]))
        return _events_t(stages_at(entry), P, at, masses)

    try:
        first = True
        for entry in reversed(log):
            if isinstance(entry, ADIEvents):
                if first and entry.j == 0 and not entry.half:
                    first = False
                    if captures_have_t0_pre:
                        continue                # the readout reads the column before today's events
                P = event_entry(entry, P)
                continue
            first = False
            key = (entry.dt, entry.theta, entry.t_mid, entry.douglas)
            op = steps.get(key)
            if op is None:
                op = steps[key] = StepTranspose(core, entry.dt, entry.theta, entry.t_mid, entry.douglas)
            out, dlow, dhigh = op.apply(P.transpose(1, 2, 0))
            P = np.ascontiguousarray(out.transpose(2, 0, 1))
            at = N if entry.half else entry.j           # the pass axis index of the column the step produced
            if entry.knocks_in:
                # the price's not-knocked-in knock-in-side row knocks in on this step (D23): its absorbed dual is
                # first knock-in mass (the row's value 1 at the step's node), normalised to the node index's DF
                edge = float((dhigh if reverse else dlow)[ALIVE])
                masses.ki_edge[at] += edge
            if reconstruct:
                w = statistics.n_cols
                recon += (dlow[ALIVE] * entry.low[:w] + dlow[KI] * entry.low[w:]
                          + dhigh[ALIVE] * entry.high[:w] + dhigh[KI] * entry.high[w:])
        # the terminal conditions and terminal events
        terminal = ADIEvents(j=N, half=False)
        if reconstruct:
            zero = affine_events(terminal)
            recon += (np.einsum("ik,ikc->c", P[ALIVE], zero["stats_v0"])
                      + np.einsum("ik,ikc->c", P[KI], zero["stats_v1"]))
        family._ki_fp = fp_2d
        P = _events_t(statistics.adjoint_stages(N, ops), P, node_of(N), masses)
    finally:
        family._ki_fp = previous_fp
    layout, tau = prepared.layout, float(statistics.tau)

    def column_of(x: float) -> int:
        if is_close(x, 0.0):
            return 0
        if is_close(x, tau):
            return node_of(int(layout.time.actual_steps))
        return int(layout.time.step_at(x))

    return ForwardResult(times=times, ki_copy=masses.ki_copy, ki_fp=masses.ki_fp, clean_ko=masses.clean_ko,
                         ki_edge=masses.ki_edge, reconstructed=recon, column_of=column_of)

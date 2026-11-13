"""The forward adjoint pass of the PDE 1D statistics march (unified event statistics spec T13; DESIGN §4.5, option A).

The statistics group marches its columns backward on the value's own grid (``march.py``). Per step ``j`` (from
``t[j+1]`` to ``t[j]``) each surface takes one Crank-Nicolson solve with the family's Dirichlet rows, then the
family's event stages at ``t[j]`` in its own order (``step_events``; ``terminal`` at maturity). Every operation is
affine in a column, and every probability column -- the knock-out units, the knocked-in settlement unit ``KIS``, the
maturity unit ``U0`` (KO-reset: ``U_pre`` / ``U_post``) and the maturity cash ``M0`` / ``M1`` -- takes the same
linear part: a knock-out sets its target (a constant), a knock-in copies the knocked-in surface into the
not-knocked-in one, a coupon projects a column against itself.

This module runs the exact transpose of that linear part forward in time, from the readout's own weights at the
spot (the readout is linear interpolation; its weights are read by probing it node by node):

* a step: ``x = M2^{-1} (M1 v + inject(rows))`` per surface; its transpose ``q = M2^{-T} p``, ``p' = M1^T q`` on the
  interior and the ``(1 - θ)`` injection weights on the boundary nodes, same set, ``dt``, ``θ`` (Rannacher /
  zero-diffusion steps included) and the very factorization the march used;
* an event stage: the transpose of the family's own primitive (``_project_event_values``, ``project_between``,
  ``_ko_update``, ``_coupon_update``, ``project_piecewise_event`` with ``_joint_branches``, the nodal masks) --
  each primitive's matrix is the primitive evaluated on the identity, so the pass transposes exactly the arithmetic
  the march applies; FIRST_PASSAGE's rank-one correction ``alive += g (ki[k*] - alive[k*])`` from the solver's own
  ``_geometry``;
* the stage plans mirror each family's ``step_events`` / ``terminal`` (``adjoint_stages`` on the statistics class).

At a knock-in stage the mass moving from the not-knocked-in to the knocked-in surface is followed through the rest of
that instant's stages (a knock-out at the same instant takes it back: the knock-out wins a same-instant discrete
tie, as the backward march decides it) and summed: the discounted value of "1 paid at ``t[j]`` on a first knock-in
at ``t[j]`` before any knock-out". Divided by ``DF(t[j])`` it is the first knock-in probability -- the same
normalisation as every unit column's (T12). A knock-out on the not-knocked-in surface likewise gives the clean
knock-out mass (no knock-in before it), which the KI table's survival column needs.

The adjoint identity (spec AC14) is checked by ``reconstruct=True``: with the march's own affine sources (its rows,
its events applied to a zero block, its terminal block) the pass reconstructs the backward readout of every column.

A route whose operator the pass cannot transpose exactly is refused with a reason (``AdjointUnavailable``), never
approximated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import get_lapack_funcs

from quantark.asset.equity.engine.knock_in_days import (
    STEP_OVERLAP,
    knock_in_day_axis,
    not_a_grid_time,
    step_overlap,
)
from quantark.util.exceptions import NumericalError, PricingError
from quantark.util.numerical import Tolerance, is_close, is_zero

from ..backends import factor
from ..components import CrankNicolson
from ..groups import Backend
from .event_statistics import NO_KNOCK_IN

__all__ = [
    "ALIVE",
    "KI",
    "AdjointOps",
    "AdjointUnavailable",
    "ColumnarFirstPassage",
    "ForwardResult",
    "KnockIn",
    "Seed",
    "SurfaceMap",
    "forward_pass",
    "knock_in_from_forward_pass",
]

#: the statistics block's surfaces, in the march's stacking order (``stats_v0``, ``stats_v1``)
ALIVE, KI = 0, 1
#: expected knock-in time when the forward pass finds no first knock-in mass beyond round-off
ZERO_TO_ROUND_OFF = "undefined: P(knock in) is zero to round-off"

_GTSV, = get_lapack_funcs(("gtsv",), (np.empty(1),))


class AdjointUnavailable(PricingError):
    """The forward pass cannot transpose this route's march exactly; ``reason`` says why (it is reported, never
    replaced by an approximation)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --- transposed operators -------------------------------------------------------------------------------------

class _Transposed:
    """``M^T`` of a linear map on one surface's nodes: a diagonal, or a sparse matrix."""

    __slots__ = ("diag", "mt")

    def __init__(self, diag: Optional[np.ndarray] = None, mt=None):
        self.diag, self.mt = diag, mt

    def __call__(self, v: np.ndarray) -> np.ndarray:
        if self.diag is not None:
            return self.diag * v
        return np.asarray(self.mt @ v, dtype=float)

    @classmethod
    def of_columns(cls, dense: np.ndarray) -> "_Transposed":
        """The transpose of the map whose response to node ``i`` is ``dense[:, i]``."""
        dense = np.asarray(dense, dtype=float)
        off = dense - np.diag(np.diag(dense))
        if not off.any():
            return cls(diag=np.diag(dense).copy())
        return cls(mt=sp.csr_matrix(dense.T))


@dataclass(frozen=True, eq=False)
class SurfaceMap:
    """``v[surface] <- M v[surface] + target`` (a knock-out's survive part, a coupon's projection of a column against
    itself, a joint projection's linear part). ``breach``: the stage's response to a unit target on a zero block
    (where, and how much, a knock-out fires), ``None`` for a stage without a target."""

    surface: int
    op: _Transposed
    breach: Optional[np.ndarray] = None


@dataclass(frozen=True, eq=False)
class Seed:
    """``v[surface] <- constant`` (KO-reset: the not-knocked-in block seeded at the first schedule's maturity)."""

    surface: int


@dataclass(frozen=True, eq=False)
class ColumnarFirstPassage:
    """The FIRST_PASSAGE correction of a 2D (spot, variance) march: per variance column ``m`` the rank-one
    ``alive[:, m] += g[:, m] (ki[k[m], m] - alive[k[m], m])`` with that column's own coefficients (``g[:, m]`` zero
    where the column adds nothing)."""

    g: np.ndarray                   # (n_spot, n_variance)
    k: np.ndarray                   # (n_variance,) int


@dataclass(frozen=True, eq=False)
class KnockIn:
    """``alive <- A alive + B ki``, then (continuous KI, FIRST_PASSAGE) ``alive += g (ki[k] - alive[k])``."""

    copy: _Transposed               # A^T
    take: _Transposed               # B^T
    fp: Optional[object] = None     # (g, k) on a 1D march; ColumnarFirstPassage on a 2D one

    def transpose(self, p: np.ndarray):
        """``(p_in, transfer_copy, transfer_fp)``: the duals on the stage's inputs and the duals moved onto the
        knocked-in surface by the copy and by the first-passage correction."""
        pa = p[ALIVE]
        fp_transfer = None
        if isinstance(self.fp, ColumnarFirstPassage):
            # the per-variance-column coefficients belong to the 2D solvers' own pass (``pde/vol_model_adjoint.py``)
            raise AdjointUnavailable("undefined: the forward adjoint pass of a 1D march has no per-column "
                                     "first-passage coefficients")
        if self.fp is not None:
            g, k = self.fp
            moved = float(g @ pa)
            pa = pa.copy()
            pa[k] -= moved
            fp_transfer = np.zeros_like(pa)
            fp_transfer[k] = moved
        take = self.take(pa)
        out = np.empty_like(p)
        out[ALIVE] = self.copy(pa)
        out[KI] = p[KI] + take
        if fp_transfer is not None:
            out[KI] += fp_transfer
        return out, take, fp_transfer


class AdjointOps:
    """Builds the stages' transposed operators from the family's own primitives, evaluated on the identity, once
    per distinct argument set (``key``)."""

    def __init__(self, n: int):
        self.n = n
        self._cache: Dict[tuple, object] = {}
        self._eye: Optional[np.ndarray] = None

    @property
    def eye(self) -> np.ndarray:
        if self._eye is None:
            self._eye = np.eye(self.n)
            self._eye.setflags(write=False)
        return self._eye

    def _cached(self, key, build):
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = build()
        return hit

    def surface_map(self, surface: int, key: tuple, response: Callable) -> SurfaceMap:
        """A knock-out stage ``response(v_survive, v_breach)`` (linear in both): its survive part on ``surface``."""
        n = self.n

        def build():
            linear = _Transposed.of_columns(response(self.eye.copy(), 0.0))
            breach = np.asarray(response(np.zeros(n), np.ones(n)), dtype=float).copy()
            return linear, breach

        linear, breach = self._cached(("ko",) + key, build)
        return SurfaceMap(surface, linear, breach)

    def nodal_ko(self, surface: int, mask: np.ndarray) -> SurfaceMap:
        """A nodal knock-out: the breached nodes take the target."""
        mask = np.asarray(mask, dtype=bool)
        return SurfaceMap(surface, _Transposed(diag=np.where(mask, 0.0, 1.0)), mask.astype(float))

    def smoothing(self, surface: int, key: tuple, response: Callable) -> SurfaceMap:
        """A stage without a target, ``v <- response(v)`` (a coupon projecting a column against itself)."""
        linear = self._cached(("smooth",) + key, lambda: _Transposed.of_columns(response(self.eye.copy())))
        return SurfaceMap(surface, linear, None)

    def joint(self, surface: int, key: tuple, linear: Callable, breach: Optional[Callable]) -> SurfaceMap:
        """A one-pass piecewise projection: ``linear()`` its matrix on the identity, ``breach()`` its knock-out
        weights (``None``: no knock-out on this surface)."""
        op = self._cached(("joint_l",) + key, lambda: _Transposed.of_columns(linear()))
        weights = None if breach is None else self._cached(("joint_b",) + key,
                                                           lambda: np.asarray(breach(), dtype=float).copy())
        return SurfaceMap(surface, op, weights)

    def knock_in(self, key: tuple, response: Callable, fp=None) -> KnockIn:
        """A projected discrete knock-in ``alive <- response(alive, ki)``."""
        def build():
            return (_Transposed.of_columns(response(self.eye.copy(), 0.0)),
                    _Transposed.of_columns(response(0.0, self.eye.copy())))

        copy, take = self._cached(("ki",) + key, build)
        return KnockIn(copy, take, fp)

    @staticmethod
    def nodal_knock_in(mask: np.ndarray, fp=None) -> KnockIn:
        """A nodal knock-in: the breached nodes take the knocked-in surface (then the first-passage correction)."""
        mask = np.asarray(mask, dtype=bool)
        return KnockIn(_Transposed(diag=np.where(mask, 0.0, 1.0)), _Transposed(diag=mask.astype(float)), fp)


def first_passage(solver, t_idx: int, s_vec: np.ndarray, barrier: float):
    """``(g, k*)`` of the FIRST_PASSAGE correction the march adds at ``t_idx`` (``alive += g (ki[k*] - alive[k*])``),
    from the solver's own per-solve state and geometry; ``None`` where it adds nothing. Per-variance-column
    coefficients (the 2D solvers', ``_ContinuousKIFirstPassage._columnar``) give a ``ColumnarFirstPassage`` from the
    same geometry, column by column (``_columnar_correction``'s arithmetic)."""
    fp = solver._ki_fp
    if fp is None:
        return None
    if fp._columnar:
        dt = float(fp._dt[t_idx])
        mu, sig2 = np.asarray(fp._mu[t_idx], dtype=float), np.asarray(fp._sig2[t_idx], dtype=float)
        g = np.zeros((len(s_vec), mu.size))
        k = np.zeros(mu.size, dtype=int)
        for col in range(mu.size):
            G, k_star, dist = fp._geometry(dt, float(mu[col]), float(sig2[col]), s_vec, float(barrier))
            if G is not None:
                g[:, col] = np.asarray(G, dtype=float) / dist
                k[col] = int(k_star)
        return ColumnarFirstPassage(g, k)
    G, k_star, dist = fp._geometry(float(fp._dt[t_idx]), float(fp._mu[t_idx]), float(fp._sig2[t_idx]),
                                   s_vec, float(barrier))
    if G is None:
        return None
    return np.asarray(G, dtype=float) / dist, int(k_star)


# --- the pass ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ForwardResult:
    """What one forward pass read, per time index ``j`` of the march's grid ``times``.

    ``ki_copy[j]`` / ``ki_fp[j]``: the discounted first knock-in mass of the stages at ``t[j]`` -- moved by the
    knock-in copy (the nodes in the knock-in region at ``t[j]``) and by the first-passage correction (crossings
    inside the step after ``t[j]``); ``clean_ko[j]``: the discounted mass knocked out at ``t[j]`` with no knock-in
    before it. ``ki_edge[j]``: the not-knocked-in surface's dual absorbed by its knock-in-side Dirichlet row over the
    step from ``t[j]``, on the steps where the price's own row there knocks in (D23, ``rows.knocks_in``; the
    statistics rows settle exactly those knocked in, ``_ki_rows``: ``KIS`` = the step's DF), so
    ``ki_edge[j] / DF(t[j])`` is first knock-in mass whose knock-in is certain from ``t[j]`` (continuous: the edge
    copies the knocked-in row; discrete: the next KI observation); 0 where the row has no event. ``reconstructed``: with
    ``reconstruct=True``, the pass's value of every column of the read surface
    (the adjoint identity: equal to the backward readout).
    """

    times: np.ndarray
    ki_copy: np.ndarray
    ki_fp: np.ndarray
    clean_ko: np.ndarray
    ki_edge: np.ndarray
    reconstructed: Optional[np.ndarray] = None
    #: the index on ``times`` of a KI observation time; ``None``: the march layout's own (the PDE 1D pass, whose
    #: ``times`` are the layout's; the ADI pass adds the terminal damping's half node)
    column_of: Optional[Callable[[float], int]] = None


def _builtin_crank_nicolson(stepper) -> bool:
    from .. import march as march_module

    return type(stepper) is CrankNicolson and CrankNicolson.step is march_module._BUILTIN_STEP


def readout_weights(statistics, states, captures) -> np.ndarray:
    """The readout's weight on every node of the block it reads (``_spot_block``), probed node by node and checked
    on the block's own columns: a readout that is not linear has no dual and is refused."""
    prepared = statistics.prepared
    readout, spot, layout = prepared.readout, prepared.env.spot, prepared.layout
    grid = statistics._spot_block(states, captures)
    n = grid.shape[0]
    unit = np.zeros(n)
    weights = np.empty(n)
    for i in range(n):
        unit[i] = 1.0
        weights[i] = float(readout.price(unit, spot, layout))
        unit[i] = 0.0
    for c in range(grid.shape[1]):
        column = np.ascontiguousarray(grid[:, c])
        value = float(readout.price(column, spot, layout))
        dual = float(weights @ column)
        if not is_close(dual, value, rel_tol=Tolerance.RELATIVE,
                        abs_tol=Tolerance.ZERO * max(1.0, float(np.max(np.abs(column))))):
            raise AdjointUnavailable(f"undefined: the readout {type(readout).__name__} is not linear in the "
                                     f"column (weights give {dual!r}, the readout {value!r})")
    return weights


def banded_step_transpose(prepared, j: int, p: np.ndarray, thetas, zero_diffusion_theta_one: bool, bands: dict):
    """The transpose of the march's banded Crank-Nicolson step ``j`` (``banded_advance``: ``V_pre(j) = [rows;
    M2^{-1}(M1 V(j+1) + inject); rows]``) on duals ``p`` (``(n_surfaces, n_nodes)``) on its output: ``(duals on
    V(j+1), duals on the step's lower rows, duals on its upper rows)`` -- same set, ``dt``, ``θ`` (Rannacher /
    zero-diffusion steps included) and the very factorization the march used. ``bands`` caches ``M2^T``'s bands per
    factorization."""
    coeffs = prepared.coefficients
    k = coeffs.set_at(j)
    s = coeffs.step_set(k, prepared.factorizations)
    theta = thetas[j]
    if zero_diffusion_theta_one and coeffs.is_zero_diffusion(k):
        theta = 1.0
    dt = prepared.layout.time.dt[j]
    banded, lower1, main1, upper1 = factor(Backend.BANDED, s, dt, theta)
    cached = bands.get(id(banded))
    if cached is None or cached[0] is not banded:
        # M2^T: its sub-diagonal is M2's super-diagonal and the other way round
        cached = bands[id(banded)] = (banded, np.ascontiguousarray(banded[0, 1:]),
                                      np.ascontiguousarray(banded[1, :]),
                                      np.ascontiguousarray(banded[2, :-1]))
    _, dl_t, d_t, du_t = cached
    rhs = np.array(p[:, 1:-1].T, order="F")
    _, _, _, q, info = _GTSV(dl_t, d_t, du_t, rhs, False, False, False, True)
    if info != 0:
        raise NumericalError(f"the transposed step {j} is singular (gtsv info {info})")
    out = np.zeros_like(p)
    r = main1[:, None] * q
    r[:-1] += lower1[:, None] * q[1:]
    r[1:] += upper1[:, None] * q[:-1]
    out[:, 1:-1] = r.T
    l1, u2 = s.l[1], s.u[-2]
    out[:, 0] = dt * (1.0 - theta) * l1 * q[0]
    out[:, -1] = dt * (1.0 - theta) * u2 * q[-1]
    on_lower = p[:, 0] + dt * theta * l1 * q[0]
    on_upper = p[:, -1] + dt * theta * u2 * q[-1]
    return out, on_lower, on_upper


class _Pass:
    """One forward pass over a marched statistics block."""

    def __init__(self, statistics, reconstruct: bool):
        prepared = statistics.prepared
        self.statistics = statistics
        self.prepared = prepared
        self.times = np.asarray(prepared.layout.time.t, dtype=float)
        self.dt = prepared.layout.time.dt
        self.N = len(self.times)
        self.n = len(prepared.layout.spatial.x)
        self.ops = AdjointOps(self.n)
        self.ki_copy = np.zeros(self.N)
        self.ki_fp = np.zeros(self.N)
        self.clean_ko = np.zeros(self.N)
        self.ki_edge = np.zeros(self.N)
        self.ki_edge_row = -1 if bool(statistics.product.is_reverse) else 0
        self.reconstruct = reconstruct
        self.recon = np.zeros(statistics.n_cols) if reconstruct else None
        self._bands: Dict[int, tuple] = {}
        group = statistics.group
        self.zero_diffusion_theta_one = bool(group.policy.zero_diffusion_theta_one)
        self.thetas = [float(x) for x in prepared.theta_schedule]

    # --- a Crank-Nicolson step, transposed ------------------------------------------------------------------

    def step(self, j: int, p: np.ndarray) -> np.ndarray:
        """Duals on ``V(j+1)`` from the duals ``p`` on the step's output ``V_pre(j) = [rows; M2^{-1}(M1 V(j+1) +
        inject); rows]`` (``banded_advance``)."""
        out, on_lower, on_upper = banded_step_transpose(self.prepared, j, p, self.thetas,
                                                        self.zero_diffusion_theta_one, self._bands)
        if self.statistics.alive_ki_edge_knocks_in(j):
            # the price's not-knocked-in row there knocks in (D23, ``rows.knocks_in``): its absorbed dual is first
            # knock-in mass; a row with no event matures not knocked in and is no knock-in
            self.ki_edge[j] = float((on_upper if self.ki_edge_row == -1 else on_lower)[ALIVE])
        if self.reconstruct:
            w = self.statistics.n_cols
            lower, upper = self.statistics.rows(j, None)
            self.recon += (on_lower[ALIVE] * lower[:w] + on_lower[KI] * lower[w:]
                           + on_upper[ALIVE] * upper[:w] + on_upper[KI] * upper[w:])
        return out

    # --- an instant's event stages, transposed ---------------------------------------------------------------

    def events(self, j: int, p: np.ndarray) -> np.ndarray:
        """Duals on ``V_pre(j)`` (terminal: the terminal conditions) from the duals ``p`` on ``V(j)``."""
        statistics = self.statistics
        stages = statistics.adjoint_stages(j, self.ops)
        if self.reconstruct:
            self._affine(j, p)
        p = p.copy()
        tracers: List[list] = []                    # [kind, duals moved onto the knocked-in surface at t[j]]
        for stage in reversed(stages):              # code order is backward time: transposed in reverse
            if isinstance(stage, SurfaceMap):
                if stage.breach is not None:
                    if stage.surface == ALIVE:
                        self.clean_ko[j] += float(p[ALIVE] @ stage.breach)
                    for tracer in tracers:          # knocked in and out at the same instant: the knock-out wins
                        self.clean_ko[j] += float(tracer[1][stage.surface] @ stage.breach)
                p[stage.surface] = stage.op(p[stage.surface])
                for tracer in tracers:
                    tracer[1][stage.surface] = stage.op(tracer[1][stage.surface])
            elif isinstance(stage, Seed):
                p[stage.surface] = 0.0
                for tracer in tracers:
                    tracer[1][stage.surface] = 0.0
            elif isinstance(stage, KnockIn):
                for tracer in tracers:
                    tracer[1] = stage.transpose(tracer[1])[0]
                p, take, fp_transfer = stage.transpose(p)
                moved = np.zeros_like(p)
                moved[KI] = take
                tracers.append(["copy", moved])
                if fp_transfer is not None:
                    moved = np.zeros_like(p)
                    moved[KI] = fp_transfer
                    tracers.append(["fp", moved])
            else:   # pragma: no cover - a stage plan bug
                raise PricingError(f"unknown adjoint stage {stage!r}")
        for kind, moved in tracers:
            mass = float(np.sum(moved[KI]))
            if kind == "copy":
                self.ki_copy[j] += mass
            else:
                self.ki_fp[j] += mass
        return p

    def _affine(self, j: int, p: np.ndarray) -> None:
        """The reconstruction's event sources: the stages applied to a zero block (terminal: the terminal block)."""
        statistics = self.statistics
        n, w = self.n, statistics.n_cols
        block = {"stats_v0": np.zeros((n, w)), "stats_v1": np.zeros((n, w))}
        if j == self.N - 1:
            statistics.terminal(block)
        else:
            statistics.step_events(block, {}, j)
        self.recon += p[ALIVE] @ block["stats_v0"] + p[KI] @ block["stats_v1"]

    def run(self, states, captures) -> ForwardResult:
        statistics = self.statistics
        weights = readout_weights(statistics, states, captures)
        p = np.zeros((2, self.n))
        p[KI if statistics.knocked_in_at_valuation else ALIVE] = weights
        if "stats_t0_pre" not in captures:          # the readout reads the column after today's events
            p = self.events(0, p)
        for j in range(self.N - 1):
            p = self.step(j, p)
            p = self.events(j + 1, p)
        return ForwardResult(times=self.times, ki_copy=self.ki_copy, ki_fp=self.ki_fp, clean_ko=self.clean_ko,
                             ki_edge=self.ki_edge,
                             reconstructed=self.recon)


def forward_pass(statistics, states, captures, *, reconstruct: bool = False) -> ForwardResult:
    """The forward adjoint pass of a marched statistics block (``states``, ``captures``: the march's).

    Raises:
        AdjointUnavailable: the route's march has no exact transpose here (a custom stepper, a non-banded
            statistics group, a readout that is not linear, per-column first-passage coefficients).
    """
    prepared = statistics.prepared
    if not _builtin_crank_nicolson(prepared.stepper):
        # TODO(unified event statistics): a custom stepper component's step has no transpose the pass can build
        raise AdjointUnavailable(f"undefined: the forward adjoint pass transposes the built-in Crank-Nicolson "
                                 f"step; the custom stepper {type(prepared.stepper).__name__} has none")
    if statistics.group.backend is not Backend.BANDED:
        raise AdjointUnavailable("undefined: the forward adjoint pass transposes the banded statistics group only")
    return _Pass(statistics, reconstruct).run(states, captures)


# --- the knock-in statistics from a pass ------------------------------------------------------------------------

def _df(env, t: float) -> float:
    return float(env.get_discount_factor(float(t)))


def _column_of(layout, t: float, tau: float) -> int:
    """The march's time index of an observation (``_populate_observation_maps``' rule)."""
    if is_close(t, 0.0):
        return 0
    if is_close(t, tau):
        return int(layout.time.actual_steps)
    return int(layout.time.step_at(t))


def _survival(times: np.ndarray, probability: np.ndarray, ko_times: np.ndarray, ko_probability: np.ndarray):
    """P(not yet knocked in and not knocked out after each row): a knock-out at a row's own time is out by then (it
    wins the tie against a discrete knock-in)."""
    order = np.argsort(ko_times, kind="stable")
    cumulative_ko = np.concatenate([[0.0], np.cumsum(ko_probability[order])])
    ko_by_row = cumulative_ko[np.searchsorted(ko_times[order], times + Tolerance.ZERO, side="right")]
    return 1.0 - np.cumsum(probability) - ko_by_row


def knock_in_from_forward_pass(statistics, states, captures):
    """``(probability, expected_time, KnockInTable, expected_time_reason)`` of the knock-in path statistics from the
    forward pass (DESIGN §4.5): rows on the product's KI schedule (discrete), or one per trading day of the window
    (continuous, ``step_overlap`` where a day boundary is not a grid time); ``expected_time_reason`` is set when the
    expected time is undefined.

    A statistics block marched by another engine family's loop carries its own pass (``statistics.forward_runner``:
    the native ADI march's, ``pde/vol_model_adjoint.py``); the table is built from its result the same way.

    Raises:
        AdjointUnavailable: as ``forward_pass``.
    """
    from quantark.asset.equity.engine.event_stats import KnockInTable

    runner = getattr(statistics, "forward_runner", None)
    result = runner() if runner is not None else forward_pass(statistics, states, captures)
    prepared, product, env = statistics.prepared, statistics.product, statistics.env
    layout = prepared.layout
    t = result.times
    df = np.array([_df(env, x) for x in t], dtype=float)
    copy_p, fp_p, edge_p = result.ki_copy / df, result.ki_fp / df, result.ki_edge / df
    ko_times, ko_probability = t, result.clean_ko / df
    horizon = float(statistics.knock_in_horizon())
    if statistics.solver._ki_continuous:
        days, dates = knock_in_day_axis(product, env, horizon)
        # copy at t[j]: crossed in (t[j-1], t[j]]; FP and the knock-in edge over the step from t[j]: (t[j], t[j+1]]
        interval = copy_p[1:] + fp_p[:-1] + edge_p[:-1]
        probability = step_overlap(t[1:], interval, days)
        if probability.size:
            probability[0] += copy_p[0]             # a knock-in read at the valuation instant
        total = float(copy_p[0] + np.sum(interval))
        midpoint_mass = float(np.sum(interval * 0.5 * (t[:-1] + t[1:])))
        starts = np.concatenate([[0.0], days[:-1]])
        interpolated = not_a_grid_time(days, t) | not_a_grid_time(starts, t)
        times = days
    else:
        tau = float(statistics.tau)
        records = [r for r in product.resolve_ki_observations(env)
                   if r.observation_time is not None
                   and -Tolerance.ZERO <= float(r.observation_time) <= horizon + Tolerance.ZERO]
        times = np.array([float(r.observation_time) for r in records], dtype=float)
        column_of = result.column_of or (lambda x: _column_of(layout, x, tau))
        columns = [column_of(float(x)) for x in times]
        mass = copy_p + fp_p
        off_schedule = np.ones(len(t), dtype=bool)
        off_schedule[columns] = False
        if np.any(mass[off_schedule] != 0.0):
            raise PricingError("the forward pass found first knock-in mass off the product's KI schedule at "
                               f"t = {t[off_schedule & (mass != 0.0)].tolist()}")
        probability = np.array([mass[c] for c in columns], dtype=float)
        # the knock-in edge over the step from t[j]: the price's row there knocks in only while a KI observation
        # remains at or after t[j] (``_knock_in_side_row``), so the knock-in is certain at the next one; mass the
        # price's rows knock in with no observation of the horizon left is refused (an invariant, never rebinned)
        row_of_step = np.searchsorted(times, t[:-1] - Tolerance.ZERO, side="left")
        live = row_of_step < times.size
        if np.any(edge_p[:-1][~live] != 0.0):
            raise PricingError("the forward pass found knock-in-side edge mass after the product's last KI "
                               f"observation in the horizon (t = {t[:-1][~live & (edge_p[:-1] != 0.0)].tolist()})")
        np.add.at(probability, row_of_step[live], edge_p[:-1][live])
        total = float(np.sum(probability))
        midpoint_mass = float(np.sum(times * probability))
        dates = tuple(r.observation_date for r in records)
        dates = dates if dates and all(d is not None for d in dates) else None
        interpolated = np.zeros(times.size, dtype=bool)
    if np.any(probability < -Tolerance.PROBABILITY):
        # the march's own discrete transition is not monotone there (the ADI's mixed-derivative splitting on a coarse
        # mesh, at short times far from the spot): its first knock-in mass on a row is negative beyond the contract's
        # probability slack. Reported, never clipped (refining the mesh removes it)
        k = int(np.argmin(probability))
        raise AdjointUnavailable(f"undefined: the march's first knock-in mass is negative on the row at t = "
                                 f"{float(times[k])!r} ({float(probability[k])!r}): its discrete transition is not "
                                 f"monotone on this grid; refine the grid")
    survival = _survival(times, probability, ko_times, ko_probability)
    table = KnockInTable(times=times, payment_times=times, dates=dates, payment_dates=dates,
                         probability=probability, survival=survival, interpolated=interpolated,
                         interpolation_scheme=STEP_OVERLAP if bool(np.any(interpolated)) else None)
    probability_total = float(np.sum(probability))
    if total == 0.0:
        return 0.0, None, table, NO_KNOCK_IN
    if is_zero(total):
        return probability_total, None, table, ZERO_TO_ROUND_OFF
    return probability_total, midpoint_mass / total, table, None

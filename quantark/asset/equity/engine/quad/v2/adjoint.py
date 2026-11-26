"""The forward adjoint pass of the QUAD V2 backward recursion (unified event statistics spec T16; DESIGN §4.5).

QUAD V2 values a contract backward (``context.prepare_compiled``): at each event the contract's actions map the
continuation states to piecewise functions of log spot (``Event.apply``: per incoming state and spot region, a
cash/asset payoff plus a weighted sum of continuation states); between events every state function is integrated
against the interval's Gaussian increment onto the mesh nodes (``Operator.integrate``: analytic moments for the
affine parts, the cell kernels for the grid sources -- the main block kernel with edge extension and the partial-cell
kernels at region ends), or, for a not-knocked-in state under continuous knock-in, by the reflection readout of the
pair (``continuous_grid_readout``); the price is the Gaussian readout of the first event's function at the spot
(``gaussian.readout`` / ``continuous_readout``, or the deterministic evaluation of an event at the valuation instant).

Every one of these maps is linear in the continuation values, so the price is ``Σ <dual, source>``. This module runs
the exact transpose forward in time, from the readout at the spot: a dual vector on each note state's mesh nodes
after each event, and per (event, incoming note state, spot region) the discounted Gaussian mass ``m0`` and asset
moment ``m1`` that the backward recursion integrates the region's affine cash against. Every backward unit column is
``Σ m0 x cash + m1 x asset`` over the actions, so the pass reproduces them to round-off (spec AC14; the termination
probabilities are such columns: each action pays ``1 / DF(t)`` in its outcome's probability channel).

From the same masses:

* the first knock-in mass at each event: an action of a not-knocked-in state that continues into a knocked-in state,
  or ends the note knocked in (a knock-out action never counts: the knock-out wins a same-instant tie, as the
  backward actions decide it);
* under continuous knock-in, the mass the reflection kernel moves to the knocked-in state over each interval (the
  constant claim's value through the same reflection arithmetic), split into trading days by the interval's own
  Brownian first-passage law (the law the reflection kernel prices with), normalised to the interval's mass;
* the clean knock-out mass (knock-outs of not-knocked-in states) for the knock-in table's survival.

Only the note states (``alive:m`` / ``ki:m``) are followed: the EXPIRY coupons' termination states carry coupon value,
not probability (their channel-selected continuations, ``channel_terms``, are not path transitions).

Refused with a reason (``AdjointUnavailable``), never approximated: a zero-variance interval after the valuation (a
deterministic shift of the grid functions), a continuous route off the barrier-aligned reflection grid (``pointwise``
backend, unaligned mesh, shifted sources), and reflection nodes whose multiplier leaves the ordinary range (the
pointwise survival integral).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.pde.core.statistics.adjoint import AdjointUnavailable
from quantark.util.exceptions import NumericalError

from .basis import gauss, interpolation_weights
from .gaussian import _INV_SQRT_2PI, affine_moments

__all__ = ["MarchLog", "ForwardPass", "forward_pass"]

#: ``continuous_grid_readout``'s ordinary reflection range: ``|log weight| < 500``
_ORDINARY_LOG_WEIGHT = 500
#: ``continuous_grid_readout``'s own barrier-alignment test of the mesh (its literal, mirrored, not a tolerance choice)
_ALIGNED = 1e-9


@dataclass(frozen=True, eq=False)
class MarchLog:
    """What the forward pass needs from one backward recursion (``prepare_compiled`` keeps it on a statistics context).

    Attributes:
        contract: the compiled contract (events, states, initial state, continuous barrier).
        mesh: the recursion's mesh (``None`` when no interval has positive variance).
        op: the recursion's operator (its kernel cache).
        means, variances, discounts: the interval moments, one per event (``[0]`` from the valuation).
        params: the engine parameters.
    """

    contract: object
    mesh: object
    op: object
    means: np.ndarray
    variances: np.ndarray
    discounts: np.ndarray
    params: object


    @property
    def pointwise_recursion(self) -> bool:
        """Whether the recursion read a grid function pointwise anywhere -- a zero-variance interval after the
        valuation (the shifted sources' readout in ``Operator.integrate``) or a continuous knock-in off the
        barrier-aligned reflection grid (``continuous_readout`` on every node): batched readouts whose reductions
        run over every channel together, so a statistics context's price channel need not be the price context's
        bitwise there (spec D23)."""
        if self.mesh is None:
            return False
        if np.any(np.asarray(self.variances[1:]) == 0.0):
            return True
        barrier = self.contract.continuous_barrier
        if barrier is None:
            return False
        coordinate = (2 * barrier - 2 * self.mesh.lo) / self.mesh.h
        if self.params.continuous_grid_backend == "pointwise" or abs(coordinate - round(coordinate)) > _ALIGNED:
            return True
        x = self.mesh.nodes.ravel()
        safe = x < barrier if self.contract.reverse else x > barrier
        for mean, variance in zip(self.means[1:], self.variances[1:]):
            if np.any(np.abs(-2 * mean / variance * (x[safe] - barrier)) >= _ORDINARY_LOG_WEIGHT):
                return True
        return False


@dataclass(frozen=True, eq=False)
class ForwardPass:
    """The forward pass's masses.

    Attributes:
        times: event times (``len(events)``).
        actions: per event, ``(state, action, m0, m1)`` for every note state's region (and the deterministic readout's
            boundary action) that received mass; ``m0`` is the discounted Gaussian mass, ``m1`` the asset moment.
        first_knock_in: per event, the discounted first knock-in mass decided AT the event (an action).
        interval_knock_in: per event ``i``, the discounted mass moved to a knocked-in state by continuous monitoring
            over ``(t[i-1], t[i]]`` (``t[-1] = 0``), discounted to ``t[i]``.
        interval_law: per event ``i``, ``(starts, weights, mean, variance, barrier, reverse)`` of that mass: the
            not-knocked-in log spots at ``t[i-1]`` and their weights, so ``Σ w P_x(hit by fraction f)`` is the mass
            crossed by a fraction ``f`` of the interval's variance; ``None`` without continuous mass there.
        clean_knock_out: per event, the discounted knock-out mass of not-knocked-in states.
    """

    times: np.ndarray
    actions: List[list]
    first_knock_in: np.ndarray
    interval_knock_in: np.ndarray
    interval_law: List[Optional[tuple]]
    clean_knock_out: np.ndarray
    #: per event with a same-instant knock-in split (``Event.ki_split``): the discounted first knock-in mass of each
    #: record's band, contract order (the narrower levels' bands, then the widest's)
    knock_in_parts: Dict[int, np.ndarray] = field(default_factory=dict)

    def reconstruct(self, n_channels: int) -> np.ndarray:
        """``Σ m0 x cash + m1 x asset`` per channel: every backward column whose continuations are note terms."""
        total = np.zeros(n_channels)
        for rows in self.actions:
            for _, action, m0, m1 in rows:
                total += m0 * np.asarray(action.cash, dtype=float) + m1 * np.asarray(action.asset, dtype=float)
        return total


# --- the dual of one state function ---------------------------------------------------------------------------------


@dataclass
class _FunctionDual:
    """The dual of one state's function at an event: per spot region the mass ``m0``, the asset moment ``m1`` and
    the dual on the continuation grids its sources read (``d``); ``boundary`` an exact-break action's weights."""

    m0: np.ndarray
    m1: np.ndarray
    d: Dict[int, np.ndarray] = field(default_factory=dict)
    boundary: Dict[int, Tuple[float, float, Optional[np.ndarray]]] = field(default_factory=dict)

    @classmethod
    def zeros(cls, n_regions: int) -> "_FunctionDual":
        return cls(np.zeros(n_regions), np.zeros(n_regions))

    def add(self, other: "_FunctionDual", sign: float = 1.0, regions=None) -> None:
        keep = range(len(self.m0)) if regions is None else regions
        for r in keep:
            self.m0[r] += sign * other.m0[r]
            self.m1[r] += sign * other.m1[r]
            if r in other.d:
                self.d[r] = self.d.get(r, 0.0) + sign * other.d[r]


def _regions(event) -> List[Tuple[float, float]]:
    edges = (-np.inf,) + tuple(event.breaks) + (np.inf,)
    return list(zip(edges[:-1], edges[1:]))


def _note_targets(action, n_note: int) -> List[Tuple[int, float]]:
    return [(int(k), float(w)) for k, w in action.terms if int(k) < n_note]


class _Pass:
    def __init__(self, log: MarchLog, n_note: int, n_memory: int):
        self.log = log
        self.contract = log.contract
        self.mesh, self.op, self.params = log.mesh, log.op, log.params
        self.n_note, self.n_memory = n_note, n_memory
        self.pairs = {a: h for a, h in self.contract.ki_pairs if a < n_note}
        self.ki_parts: Dict[int, np.ndarray] = {}
        self._moments: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        if self.mesh is not None:
            self.nodes = self.mesh.nodes
            if self.contract.continuous_barrier is not None:
                self._check_reflection_grid()

    # --- the region masses and the integrate transpose --------------------------------------------------------------

    def _moment(self, mean, variance, lo, hi):
        key = (float(mean), float(variance), float(lo), float(hi))
        cached = self._moments.get(key)
        if cached is None:
            cached = (affine_moments(self.nodes, mean, variance, lo, hi, 0)[0],
                      affine_moments(self.nodes, mean, variance, lo, hi, 1)[0])
            self._moments[key] = cached
        return cached

    def _correlate(self, g, mean, variance):
        """Transpose of the block kernel with edge extension (``apply_direct`` / ``apply_fft``)."""
        mesh = self.mesh
        kernel, _ = self.op.kernel(mean, variance)
        band = (len(kernel) - 1) // 2
        cells = mesh.cells
        padded = np.zeros((cells + 2 * band, mesh.order))
        for j in range(2 * band + 1):
            padded[j:j + cells] += g @ kernel[j].T
        c = padded[band:band + cells].copy()
        c[0] += padded[:band].sum(axis=0)
        c[cells - 1] += padded[band + cells:].sum(axis=0)
        return c

    def integrate_transpose(self, g, regions, mean, variance, discount, wanted) -> _FunctionDual:
        """The dual of ``Operator.integrate(F, mean, variance, discount)`` on ``F``'s regions, given the dual ``g`` on
        the integrated nodes; source duals only for the regions in ``wanted``."""
        mesh = self.mesh
        dual = _FunctionDual.zeros(len(regions))
        for r, (lo, hi) in enumerate(regions):
            c0, c1 = self._moment(mean, variance, lo, hi)
            dual.m0[r] = discount * float(np.sum(g * c0))
            dual.m1[r] = discount * float(np.sum(g * c1))
        if not wanted:
            return dual
        conv = None
        for r in wanted:
            lo, hi = regions[r]
            ua = max(0.0, (lo - mesh.lo) / mesh.h)
            ub = min(float(mesh.cells), (hi - mesh.lo) / mesh.h)
            d = np.zeros((mesh.cells, mesh.order))
            if ua < ub:
                start, stop = int(np.ceil(ua)), int(np.floor(ub))
                if stop > start:
                    if conv is None:
                        conv = self._correlate(g, mean, variance)
                    d[start:stop] = conv[start:stop]
                partial = []
                if int(np.floor(ua)) == int(np.floor(ub)):
                    partial.append((int(np.floor(ua)), ua % 1, ub % 1))
                else:
                    if ua < start:
                        partial.append((int(np.floor(ua)), ua % 1, 1.0))
                    if stop < ub:
                        partial.append((stop, 0.0, ub - stop))
                for cell, a, b in partial:
                    if b <= a or cell >= mesh.cells:
                        continue
                    kernel, _ = self.op.kernel(mean, variance, a, b)
                    band = (len(kernel) - 1) // 2
                    i0, i1 = max(0, cell - band), min(mesh.cells, cell + band + 1)
                    idx = cell - np.arange(i0, i1) + band
                    d[cell] += np.einsum("nr,nqr->q", g[i0:i1], kernel[idx])
            dual.d[r] = discount * d
        return dual

    # --- point readouts (``gaussian.readout`` with one query, ``_source_readout``) -----------------------------------

    def point_transpose(self, points, coefficients, regions, mean, variance, wanted) -> _FunctionDual:
        """The dual of ``Σ_p c_p readout(F, points[p], mean, variance, 1)`` on ``F``'s regions: analytic moments for
        the affine parts, the transpose of ``_source_readout`` (one query each) for the grid sources."""
        points = np.asarray(points, dtype=float).reshape(-1)
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        dual = _FunctionDual.zeros(len(regions))
        for r, (lo, hi) in enumerate(regions):
            dual.m0[r] = float(np.sum(coefficients * affine_moments(points, mean, variance, lo, hi, 0)[0]))
            dual.m1[r] = float(np.sum(coefficients * affine_moments(points, mean, variance, lo, hi, 1)[0]))
            if r in wanted:
                dual.d[r] = self._source_readout_transpose(points, coefficients, mean, variance, lo, hi)
        return dual

    def _source_readout_transpose(self, points, coefficients, mean, variance, lo, hi):
        """Transpose of ``gaussian._source_readout`` (shift 0, the recursion's mesh), value row, summed over points."""
        mesh, params = self.mesh, self.params
        gx, gw = gauss(params.readout_order)
        sd = np.sqrt(variance)
        tail = params.tail_sd
        d = np.zeros((mesh.cells, mesh.order))
        theta = mesh.theta
        for x, coefficient in zip(points.tolist(), coefficients.tolist()):
            if coefficient == 0.0:
                continue
            center = x + mean
            left, right = max(lo, center - tail * sd), min(hi, center + tail * sd)
            if left >= right:
                continue
            k = int(np.clip(np.floor((center - mesh.lo) / mesh.h), 0, mesh.cells - 1))
            spots = np.exp(mesh.lo + (k + theta) * mesh.h)
            width = spots[-1] - spots[0]
            base0 = affine_moments(np.array([x]), mean, variance, lo, hi, 0)[0, 0]
            base1 = affine_moments(np.array([x]), mean, variance, lo, hi, 1)[0, 0]
            i0 = max(0, int(np.floor((left - mesh.lo) / mesh.h)) + 1)
            i1 = min(mesh.cells, int(np.ceil((right - mesh.lo) / mesh.h)))
            internal = mesh.lo + np.arange(i0, i1) * mesh.h
            internal = internal[(internal > left) & (internal < right)]
            zcuts = np.arange(-tail, tail + 1, 2.0) * sd + center
            cuts = np.unique(np.r_[left, internal, zcuts[(zcuts > left) & (zcuts < right)], right])
            a, b = (cuts[:-1] - center) / sd, (cuts[1:] - center) / sd
            half, mid = (b - a) / 2, (b + a) / 2
            z = mid[:, None] + half[:, None] * gx
            y = center + sd * z
            u = (y - mesh.lo) / mesh.h
            cells = np.clip(np.floor(u).astype(int), 0, mesh.cells - 1)
            t = np.clip(u - cells, 0, 1)
            weights = interpolation_weights(theta, t)                      # (order, panels, points)
            dw = half[:, None] * gw * _INV_SQRT_2PI * np.exp(-z * z / 2)   # the value row's score is 1
            scaled = weights * dw[None]
            for q in range(mesh.order):
                np.add.at(d[:, q], cells.ravel(), coefficient * scaled[q].ravel())
            d_cash = base0 - float(np.sum(weights.sum(axis=0) * dw))
            d_slope = base1 - float(np.sum(np.exp(y) * dw))
            d[k, 0] += coefficient * (d_cash * (1.0 + spots[0] / width) - d_slope / width)
            d[k, -1] += coefficient * (-d_cash * spots[0] / width + d_slope / width)
        return d

    # --- deterministic evaluation (a zero-variance readout) ---------------------------------------------------------

    def evaluate_dual(self, event, state, y, discount, *, pieces_only=False) -> _FunctionDual:
        """The dual of ``F(y) x discount`` for the event's function of ``state`` (``Function.evaluate``: the region
        holding ``y``, an exact break's boundary action unless ``pieces_only``)."""
        regions = _regions(event)
        dual = _FunctionDual.zeros(len(regions))
        breaks = tuple(event.breaks)
        weights = None
        if self.mesh is not None:
            u = (y - self.mesh.lo) / self.mesh.h
            k = int(np.clip(np.floor(u), 0, self.mesh.cells - 1))
            t = float(np.clip(u - k, 0, 1))
            w = interpolation_weights(self.mesh.theta, np.array(t))
            weights = np.zeros((self.mesh.cells, self.mesh.order))
            weights[k] = discount * np.asarray(w, dtype=float).reshape(-1)
        if not pieces_only and y in breaks and event.boundary_rows:
            j = breaks.index(y)
            dual.boundary[j] = (discount, discount * float(np.exp(y)), weights)
            return dual
        r = int(np.searchsorted(np.asarray(breaks, dtype=float), y, side="right"))
        dual.m0[r] = discount
        dual.m1[r] = discount * float(np.exp(y))
        if weights is not None:
            dual.d[r] = weights
        return dual

    # --- the continuous knock-in reflection -------------------------------------------------------------------------

    def _check_reflection_grid(self) -> None:
        mesh, params = self.mesh, self.params
        barrier = self.contract.continuous_barrier
        coordinate = (2 * barrier - 2 * mesh.lo) / mesh.h
        if params.continuous_grid_backend == "pointwise" or abs(coordinate - round(coordinate)) > _ALIGNED:
            # TODO(unified event statistics):
            # transpose the pointwise continuous readout on every node (continuous_readout)
            raise AdjointUnavailable(
                "undefined: the forward pass transposes the barrier-aligned reflection grid only; this route reads "
                "the continuous knock-in pointwise (continuous_grid_backend='pointwise' or an unaligned mesh)"
            )
        self.coordinate = int(round(coordinate))

    def reflection_transpose(self, g, event, mean, variance, discount, alive, hit, duals, interval) -> None:
        """Transpose of ``continuous_grid_readout`` for the not-knocked-in state ``alive`` paired with ``hit``: the
        hit function takes the plain integral (``rolled_hit``) less the reflected difference on the safe regions,
        the alive function the reflected difference. Records the interval's crossing law."""
        mesh, contract = self.mesh, self.contract
        barrier, reverse = contract.continuous_barrier, contract.reverse
        regions = _regions(event)
        safe_regions = [r for r, (lo, hi) in enumerate(regions)
                        if not ((reverse and lo >= barrier) or (not reverse and hi <= barrier))]
        hit_wanted = self._wanted(event, hit)
        alive_wanted = self._wanted(event, alive)
        base = self.integrate_transpose(g, regions, mean, variance, discount, hit_wanted)
        x = mesh.nodes.ravel()
        flat = g.ravel()
        safe = x < barrier if reverse else x > barrier
        indices = np.flatnonzero(safe)
        hit_side = base
        if len(indices):
            log_weight = -2 * mean / variance * (x[indices] - barrier)
            ordinary = np.abs(log_weight) < _ORDINARY_LOG_WEIGHT
            if np.any(flat[indices[~ordinary]] != 0.0):
                # TODO(unified event statistics):
                # transpose the pointwise survival integral of the non-ordinary reflection nodes
                raise AdjointUnavailable(
                    "undefined: a reflection node's multiplier leaves the ordinary range (|log weight| >= 500); the "
                    "forward pass does not transpose the pointwise survival integral there"
                )
            idx = indices[ordinary]
            w = np.exp(log_weight[ordinary])
            q = np.zeros(mesh.cells * mesh.order)
            q[idx] += discount * flat[idx]
            cells, nodes = idx // mesh.order, idx % mesh.order
            mirror_cells = self.coordinate - cells - 1
            mirror_nodes = mesh.order - nodes - 1
            inside = (mirror_cells >= 0) & (mirror_cells < mesh.cells)
            np.add.at(q, mirror_cells[inside] * mesh.order + mirror_nodes[inside],
                      -discount * flat[idx[inside]] * w[inside])
            wanted = sorted(set(hit_wanted) | set(alive_wanted))
            diff = self.integrate_transpose(q.reshape(mesh.cells, mesh.order), regions, mean, variance, 1.0,
                                            [r for r in wanted if r in safe_regions])
            outside = ~inside
            if np.any(outside):
                points = 2 * barrier - x[idx[outside]]
                coefficients = -discount * flat[idx[outside]] * w[outside]
                self._outside_images_take_the_pointwise_readout(points, mean, variance)
                image = self.point_transpose(points, coefficients, regions, mean, variance,
                                             [r for r in wanted if r in safe_regions])
                diff.add(image)
            alive_side = _FunctionDual.zeros(len(regions))
            alive_side.add(diff, regions=safe_regions)
            hit_side = _FunctionDual.zeros(len(regions))
            hit_side.add(base)
            hit_side.add(diff, sign=-1.0, regions=safe_regions)
            self._accumulate(duals, alive, alive_side)
        self._accumulate(duals, hit, hit_side)
        crossed = float(np.sum(hit_side.m0))
        if crossed != 0.0:
            interval["mass"] += crossed
            interval["laws"].append((x.copy(), discount * flat.copy()))

    def _outside_images_take_the_pointwise_readout(self, points, mean, variance) -> None:
        """``readout(diff, points)`` of mirrors beyond the mesh runs ``_source_readout`` per point (never the batch
        kernel) -- the arithmetic this pass transposes."""
        mesh, params = self.mesh, self.params
        sd = np.sqrt(variance)
        if mesh.h > sd * 0.75:
            return
        band = int(np.ceil(params.tail_sd * sd / mesh.h)) + 2
        cell0 = np.floor((points + mean - mesh.lo) / mesh.h).astype(int)
        if np.any((cell0 - band >= 0) & (cell0 + band < mesh.cells)):
            raise NumericalError("a mirror outside the QUAD V2 mesh reads inside its kernel band")

    # --- the pass ---------------------------------------------------------------------------------------------------

    def _wanted(self, event, state) -> List[int]:
        return [r for r, action in enumerate(event.rows[state]) if _note_targets(action, self.n_note)]

    @staticmethod
    def _accumulate(duals, state, dual) -> None:
        if state in duals:
            duals[state].add(dual)
        else:
            duals[state] = dual

    def readout(self, x0) -> Dict[int, _FunctionDual]:
        """The duals of the first event's functions from ``PreparedQuad.evaluate`` at the spot ``x0``."""
        log, contract = self.log, self.contract
        event = contract.events[0]
        regions = _regions(event)
        mean, variance, discount = float(log.means[0]), float(log.variances[0]), float(log.discounts[0])
        init = contract.initial_state
        hit = self.pairs.get(init) if contract.continuous_barrier is not None else None
        duals: Dict[int, _FunctionDual] = {}
        if variance == 0:
            if hit is None:
                duals[init] = self.evaluate_dual(event, init, x0 + mean, discount)
                return duals
            # ``continuous_readout`` at zero variance: the shifted pair joined at the deterministic boundary
            barrier, reverse = contract.continuous_barrier, contract.reverse
            boundary = min(barrier, barrier - mean) if reverse else max(barrier, barrier - mean)
            through = x0 >= boundary if reverse else x0 <= boundary
            state = hit if through else init
            duals[state] = self.evaluate_dual(event, state, x0 + mean, discount, pieces_only=True)
            return duals
        if hit is None:
            duals[init] = self.point_transpose([x0], [discount], regions, mean, variance, self._wanted(event, init))
            return duals
        barrier, reverse = contract.continuous_barrier, contract.reverse
        hit_wanted, alive_wanted = self._wanted(event, hit), self._wanted(event, init)
        base = self.point_transpose([x0], [discount], regions, mean, variance, hit_wanted)
        safe = x0 < barrier if reverse else x0 > barrier
        if not safe:
            duals[hit] = base
            self.valuation_law = None
            return duals
        c = -2 * mean / variance
        log_weight = c * (x0 - barrier)
        if abs(log_weight) >= _ORDINARY_LOG_WEIGHT:
            # TODO(unified event statistics): transpose ``_stable_survival`` at the spot
            raise AdjointUnavailable(
                "undefined: the spot's reflection multiplier leaves the ordinary range (|log weight| >= 500); the "
                "forward pass does not transpose the stable survival integral"
            )
        safe_regions = [r for r, (lo, hi) in enumerate(regions)
                        if not ((reverse and lo >= barrier) or (not reverse and hi <= barrier))]
        wanted = [r for r in sorted(set(hit_wanted) | set(alive_wanted)) if r in safe_regions]
        diff = self.point_transpose([x0, 2 * barrier - x0], [discount, -discount * np.exp(log_weight)], regions,
                                    mean, variance, wanted)
        alive_side = _FunctionDual.zeros(len(regions))
        alive_side.add(diff, regions=safe_regions)
        hit_side = _FunctionDual.zeros(len(regions))
        hit_side.add(base)
        hit_side.add(diff, sign=-1.0, regions=safe_regions)
        duals[init] = alive_side
        duals[hit] = hit_side
        self.valuation_law = (float(np.sum(hit_side.m0)), (np.array([x0]), np.array([discount])))
        return duals

    def run(self, x0) -> ForwardPass:
        log, contract = self.log, self.contract
        events = contract.events
        n = len(events)
        times = np.array([float(e.time) for e in events])
        self.valuation_law = None
        actions: List[list] = [[] for _ in range(n)]
        first_ki = np.zeros(n)
        interval_ki = np.zeros(n)
        laws: List[Optional[tuple]] = [None] * n
        clean_ko = np.zeros(n)
        duals = self.readout(x0)
        if self.valuation_law is not None:
            interval_ki[0] = self.valuation_law[0]
            laws[0] = (self.valuation_law[1], float(log.means[0]), float(log.variances[0]))
        for i in range(n):
            event = events[i]
            grids: Dict[int, np.ndarray] = {}
            band_of = self._ki_bands(event)
            for state, dual in sorted(duals.items()):
                if state >= self.n_note:
                    continue
                rows = event.rows[state]
                for r, action in enumerate(rows):
                    self._record(actions[i], state, action, dual.m0[r], dual.m1[r], first_ki, clean_ko, i,
                                 band=None if band_of is None else band_of[0][r])
                    if r in dual.d:
                        for k, w in _note_targets(action, self.n_note):
                            grids[k] = grids.get(k, 0.0) + w * dual.d[r]
                for j, (m0, m1, weights) in dual.boundary.items():
                    action = event.boundary_rows[state][j]
                    self._record(actions[i], state, action, m0, m1, first_ki, clean_ko, i,
                                 band=None if band_of is None else band_of[1][j])
                    if weights is not None:
                        for k, w in _note_targets(action, self.n_note):
                            grids[k] = grids.get(k, 0.0) + w * weights
            if i == n - 1 or not grids:
                break
            nxt = events[i + 1]
            mean, variance, discount = float(log.means[i + 1]), float(log.variances[i + 1]), float(log.discounts[i + 1])
            if variance == 0:
                # TODO(unified event statistics):
                # transpose the deterministic shift (``Function.shifted`` + ``deterministic_continuous``)
                raise AdjointUnavailable(
                    "undefined: a zero-variance interval after the valuation shifts the grid functions "
                    "deterministically; the forward pass does not transpose that shift"
                )
            regions = _regions(nxt)
            duals = {}
            interval = {"mass": 0.0, "laws": []}
            for state, g in sorted(grids.items()):
                hit = self.pairs.get(state) if contract.continuous_barrier is not None else None
                if hit is None:
                    self._accumulate(duals, state, self.integrate_transpose(
                        g, regions, mean, variance, discount, self._wanted(nxt, state)))
                else:
                    self.reflection_transpose(g, nxt, mean, variance, discount, state, hit, duals, interval)
            interval_ki[i + 1] = interval["mass"]
            if interval["laws"]:
                x = interval["laws"][0][0]
                weights = np.sum([w for _, w in interval["laws"]], axis=0)
                laws[i + 1] = ((x, weights), mean, variance)
        return ForwardPass(times=times, actions=actions, first_knock_in=first_ki, interval_knock_in=interval_ki,
                           interval_law=laws, clean_knock_out=clean_ko, knock_in_parts=self.ki_parts)

    def _ki_bands(self, event):
        """``(per region, per boundary)`` the band of an event's knock-in records each spot region / break point lies
        in (0.16.0 zero-gap rule: the first record in contract order whose level it breaches; the widest's band last);
        ``None`` without a split. The event's breaks include every split level, so a region lies in one band."""
        split = tuple(getattr(event, "ki_split", ()) or ())
        if not split:
            return None
        reverse = bool(self.contract.reverse)

        def band(x):
            for k, level in enumerate(split):
                if (x >= level) if reverse else (x <= level):
                    return k
            return len(split)

        edges = (-np.inf,) + tuple(event.breaks) + (np.inf,)
        regions = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            x = hi - 1.0 if not np.isfinite(lo) else lo + 1.0 if not np.isfinite(hi) else 0.5 * (lo + hi)
            regions.append(band(x))
        return regions, [band(b) for b in event.breaks]

    def _record(self, rows, state, action, m0, m1, first_ki, clean_ko, i, band=None) -> None:
        if m0 == 0.0 and m1 == 0.0:
            return
        rows.append((state, action, m0, m1))
        if state >= self.n_memory:
            return
        if self._knocks_out(action):
            clean_ko[i] += m0
        elif self._knocks_in(action):
            first_ki[i] += m0
            if band is not None:
                parts = self.ki_parts.get(i)
                if parts is None:
                    parts = self.ki_parts[i] = np.zeros(len(self.contract.events[i].ki_split) + 1)
                parts[band] += m0

    def _knocks_out(self, action) -> bool:
        return any(action.cash[c] != 0.0 for c in self.ko_channels)

    def _knocks_in(self, action) -> bool:
        if any(self.n_memory <= k < self.n_note for k, _ in _note_targets(action, self.n_note)):
            return True
        return self.ki_terminal is not None and action.cash[self.ki_terminal] != 0.0


def forward_pass(log: MarchLog, spot: float) -> ForwardPass:
    """The forward adjoint pass of ``log``'s recursion read at ``spot``.

    Raises:
        AdjointUnavailable: a route whose maps the pass does not transpose (module docstring); the reason says why.
    """
    contract = log.contract
    names = contract.state_names
    n_note = sum(1 for name in names if name.startswith(("alive:", "ki:")))
    if names[:n_note] != tuple(names[:n_note]) or any(not n.startswith(("alive:", "ki:")) for n in names[:n_note]):
        raise NumericalError("QUAD V2 note states must lead the state list")
    n_memory = n_note // 2
    runner = _Pass(log, n_note, n_memory)
    channels = contract.channels
    runner.ko_channels = [i for i, name in enumerate(channels) if name.startswith("ko_probability:")]
    runner.ki_terminal = channels.index("terminal_ki_probability") if "terminal_ki_probability" in channels else None
    return runner.run(float(np.log(spot)))

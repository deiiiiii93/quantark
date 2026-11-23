"""The QUAD V1 exact first step (sub-resolution run S3b, spec "F").

The first step runs from the valuation instant, where the claim's state is a POINT (the spot), to the first event
time ``t_1``. The lattice applies it as one more diffusion step onto the post-event surfaces at ``t_1`` and reads the
result at the spot. Within a cell or two of a barrier observed at ``t_1`` that loses the information the price needs:
the post-event surface jumps inside a cell and the lattice cannot see where. A step narrower than a cell cannot be
sampled at all (S1: lattice candidates up to 3.8e-2 wrong); a step a few cells wide is sampled but its sum over the
jump is biased (S1: today's V1 up to 1.1e-3 at a one-day lead, 1.1e-2 just above the guard threshold).

F evaluates the first step directly at the spot::

    V(0) = disc * E[ event(pre)(X) ],   X ~ N(m, s^2),
    m = 2 alpha tau,  s^2 = 2 tau,  disc = exp(-(beta - alpha^2) tau)

on the PRE-event surfaces at ``t_1`` (smooth: diffused from ``t_2``), interpolated by a local 4-point cubic, with the
engine's own event map applied POINTWISE (each quadrature node on its own side of every barrier, the contract's
event exactly). The expectation is a Gauss-Legendre sum on pieces cut at the barriers, at the lattice nodes (the
interpolant is a cubic between them) and at standardized points ``m + s z`` -- never one rule across the barrier:
S1 found plain Gauss-Legendre across the step 0.48% low when ``s << h``. Mass beyond ``14 s`` (< 1e-44) is dropped.

Under continuous knock-in the not-knocked-in state rolls with the Brownian-bridge touch probability from the spot to
each node (``_forward_seed_touch_probability``'s formula), so the first interval's touch is exact too.

Everything here is linear in the pre-event surfaces, so it transposes: :meth:`GaussianFirstStep.lift_transpose`
gives the adjoint statistics pass its row (``v1_adjoint``) and the forward-density seed.

Several centres can share one evaluation (the t=0 value surface near a barrier: F at each lattice node of a band).
The event map is pointwise, so the nodes of all centres are evaluated together.
"""
from __future__ import annotations

import math
import threading
import weakref
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GaussianFirstStep",
    "pointwise_breach",
    "pointwise_event_values",
    "pointwise_piecewise_event",
]

#: Gaussian support kept per centre, in standard deviations (the tail beyond carries < 1e-44 of the mass)
_SUPPORT = 14.0
#: standardized breakpoints: a Gauss-Legendre rule is exact to round-off on pieces up to about 2 s wide
_STANDARD_POINTS = (-10.0, -7.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
#: Gauss-Legendre orders: wide pieces (> s/2) take the long rule, cell pieces of a wide Gaussian the short one
_GL_LONG = np.polynomial.legendre.leggauss(24)
_GL_SHORT = np.polynomial.legendre.leggauss(10)
_GL_CELL = np.polynomial.legendre.leggauss(6)
_GL_FINE = np.polynomial.legendre.leggauss(4)
_SQ2PI = math.sqrt(2.0 * math.pi)


def pointwise_breach(x: np.ndarray, barrier_log: float, breach_up: bool) -> np.ndarray:
    """The breached indicator at each point (the limit of ``breach_fractions`` for a cell of width zero)."""
    x = np.asarray(x, dtype=float)
    return (x >= barrier_log).astype(float) if breach_up else (x <= barrier_log).astype(float)


def pointwise_event_values(x: np.ndarray, barrier_log: float, v_survive, v_breach, breach_up: bool) -> np.ndarray:
    """``project_event_values`` evaluated pointwise: the breach branch where the point breaches, else survive
    (spot on axis zero, as ``project_event_values``)."""
    x = np.asarray(x, dtype=float)
    hit = (x >= barrier_log) if breach_up else (x <= barrier_log)
    v_s, v_b = np.broadcast_arrays(np.asarray(v_survive, dtype=float), np.asarray(v_breach, dtype=float))
    if v_s.ndim == 0:
        v_s, v_b = np.full(x.shape, float(v_s)), np.full(x.shape, float(v_b))
    mask = hit.reshape(hit.shape + (1,) * (v_s.ndim - 1))
    return np.array(np.where(mask, v_b, v_s), dtype=float)


def pointwise_piecewise_event(x: np.ndarray, breaks: Sequence[float], branches) -> np.ndarray:
    """``project_piecewise_event`` evaluated pointwise: region ``j`` (``b_j <= x < b_(j+1)``) takes ``branches[j]``
    (spot on axis zero)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    breaks = [float(b) for b in breaks]
    arrs = [np.asarray(v, dtype=float) for v in branches]
    if all(a.ndim == 0 for a in arrs):
        arrs = [np.full(n, float(a)) for a in arrs]
    else:
        arrs = [np.array(a, dtype=float) for a in np.broadcast_arrays(*arrs)]
    region = np.searchsorted(breaks, x, side="right")
    return np.array(np.stack(arrs)[region, np.arange(n)], dtype=float)


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _SQ2PI


class _MatrixPool:
    """Storage for the quadrature's dense centres x nodes matrices (``W``, ``P``, ``W * P``), leased per quadrature
    and taken back when the quadrature is collected (sub-resolution run S3c).

    Why: F's own arithmetic is about 1% of a monthly snowball's price, but allocating and freeing centres x nodes
    arrays on every solve (0.2 .. 8 MB: the allocator's medium range on macOS) left numpy's FFT temporaries on a heap
    layout where every transform of the following solves ran about 13% slower -- +5.4 s on the 42 s book benchmark
    (60 monthly snowballs x 3 spots), the same transforms on the same inputs. Plain numpy reproduces it: one 1 MB
    allocate/free between batches of transforms slows them 6%; 0.12 MB blocks, 10 MB blocks, or 2 MB blocks kept
    alive and reused do not. So the long-lived matrices come from here, and the transient ones are built on blocks of
    centres small enough for the small range (``_row_blocks``); either half alone left the book at 1.11x.

    Matrices above ``_MAX_BYTES`` (sub-cell steps with a band: each centre has its own nodes) are the allocator's
    large range, unaffected, and allocated fresh as before rather than retained.

    The values are unchanged: a leased matrix is a C-contiguous view of the same shape, filled by the same elementwise
    operations as the fresh array it replaces, and read by the same products.
    """

    #: free buffers kept for reuse (a solve leases at most three at once; the last solve's quadrature holds its own)
    _KEEP = 4
    #: largest matrix leased (bytes); above it the allocator's large range (mapped per allocation) is used directly
    _MAX_BYTES = 8 << 20

    def __init__(self) -> None:
        self._free: list = []
        self._lock = threading.RLock()       # re-entrant: a collection inside ``lease`` may give a buffer back

    def lease(self, owner, shape: Tuple[int, int], fill: Optional[float] = None) -> np.ndarray:
        """A ``shape`` matrix (filled with ``fill``, else uninitialised) whose storage returns to the pool when
        ``owner`` is collected."""
        size = int(shape[0]) * int(shape[1])
        if size * 8 > self._MAX_BYTES:
            if fill is None:
                return np.empty(shape)
            return np.zeros(shape) if fill == 0.0 else np.full(shape, float(fill))
        matrix = self._take(owner, size).reshape(shape)
        if fill is not None:
            matrix.fill(float(fill))
        return matrix

    def _take(self, owner, size: int) -> np.ndarray:
        with self._lock:
            best = None
            for i, buf in enumerate(self._free):
                # at most twice the request: a small matrix never pins a large recycled buffer (CP2)
                if size <= buf.size <= 2 * max(size, 1) and (best is None or buf.size < self._free[best].size):
                    best = i
            buf = self._free.pop(best) if best is not None else np.empty(max(size, 1), dtype=float)
        weakref.finalize(owner, self._give_back, buf).atexit = False
        return buf[:size]

    def _give_back(self, buf: np.ndarray) -> None:
        with self._lock:
            self._free.append(buf)
            if len(self._free) > self._KEEP:
                smallest = min(range(len(self._free)), key=lambda i: self._free[i].size)
                del self._free[smallest]


_POOL = _MatrixPool()

#: entries of a centres x nodes temporary: 96 kB of float64, inside the allocator's small range (``_MatrixPool``)
_BLOCK_ENTRIES = 12_000


def _row_blocks(n_rows: int, n_cols: int):
    """Slices of at most ``_BLOCK_ENTRIES // n_cols`` rows (at least one) covering ``range(n_rows)``."""
    step = max(1, _BLOCK_ENTRIES // max(int(n_cols), 1))
    return [slice(lo, min(lo + step, n_rows)) for lo in range(0, int(n_rows), step)]


class GaussianFirstStep:
    """Quadrature of ``disc * E[f(c + X)]``, ``X ~ N(mean, std^2)``, for each centre ``c`` (log-spot offsets from the
    spot; the price is the centre 0), with ``f`` the engine's pointwise event map on interpolated pre-event surfaces.

    Two layouts, the same rule: pieces cut at the breakpoints and the lattice nodes, Gauss-Legendre on each.

    * a kernel narrower than a cell (``std < h``): each centre has its own nodes, cut also at its standardized points
      (the pieces between two lattice nodes are far wider than the kernel);
    * a kernel a cell or wider: every centre shares one set of nodes on the lattice cells (each piece at most a cell,
      so at most ``std`` wide), and only the Gaussian weights differ per centre (``W``, centres x nodes).

    Attributes:
        y: quadrature nodes (log offsets from the spot).
        W: ``disc * GL weight * Gaussian density``, centres x nodes (zero off a centre's support); each row sums to
            ``disc`` to round-off.
        P: the continuous knock-in touch probability from each centre to each node (``None`` without one).
    """

    def __init__(self, grid: np.ndarray, h: float, *, mean: float, std: float, disc: float,
                 centres: Sequence[float], breakpoints: Sequence[float] = (),
                 touch: Optional[Tuple[float, bool]] = None, kinks: Sequence[float] = ()) -> None:
        self.grid = np.asarray(grid, dtype=float)
        self._kinks = tuple(float(k) for k in kinks)
        self.h = float(h)
        self.mean = float(mean)
        self.std = float(std)
        self.disc = float(disc)
        self.centres = np.asarray(centres, dtype=float)
        self._breaks = np.asarray(sorted(float(b) for b in breakpoints), dtype=float)
        self._WP = None                   # ``W * P`` (``expect_bridged``), built on first use
        if self.std < self.h:
            self._own_nodes(touch)
        else:
            self._shared_nodes(touch)
        self._build_interpolation()

    def _pieces(self, cuts):
        cuts = np.unique(np.asarray(cuts, dtype=float))
        a, b = cuts[:-1], cuts[1:]
        keep = b > a
        return a[keep], b[keep]

    def _gauss_legendre(self, a, b):
        """Nodes and weights of the pieces ``[a, b]``: 24 points on pieces wider than ``std / 2``, 10 down to
        ``std / 4``, 6 down to ``std / 6``, 4 below (a cubic times a Gaussian over a sixth of its width: exact to
        round-off)."""
        nodes, weights = [], []
        width = b - a
        tiers = ((_GL_LONG, width > 0.5 * self.std), (_GL_SHORT, (width <= 0.5 * self.std) & (width > 0.25 * self.std)),
                 (_GL_CELL, (width <= 0.25 * self.std) & (width > self.std / 6.0)),
                 (_GL_FINE, width <= self.std / 6.0))
        for (gx, gw), select in tiers:
            if not np.any(select):
                continue
            pa, pb = a[select], b[select]
            half = 0.5 * (pb - pa)
            nodes.append((half[:, None] * gx[None, :] + (0.5 * (pa + pb))[:, None]).ravel())
            weights.append((half[:, None] * gw[None, :]).ravel())
        if not nodes:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(nodes), np.concatenate(weights)

    def _density(self, y, centre):
        z = (y - (centre + self.mean)) / self.std
        return np.where(np.abs(z) <= _SUPPORT, _phi(z) / self.std, 0.0)

    def _own_nodes(self, touch) -> None:
        nodes, blocks = [], []
        for centre in self.centres:
            mu = centre + self.mean
            lo, hi = mu - _SUPPORT * self.std, mu + _SUPPORT * self.std
            cuts = [lo, hi]
            cuts += [float(b) for b in self._breaks if lo < b < hi]
            cuts += [float(g) for g in self.grid[(self.grid > lo) & (self.grid < hi)]]
            cuts += [mu + self.std * z for z in _STANDARD_POINTS]
            y, w = self._gauss_legendre(*self._pieces(cuts))
            nodes.append(y)
            blocks.append(w)
        self.y = np.concatenate(nodes) if nodes else np.zeros(0)
        n_c = self.centres.size
        self.W = _POOL.lease(self, (n_c, self.y.size), fill=0.0)
        self.P = None if touch is None else _POOL.lease(self, (n_c, self.y.size), fill=1.0)
        start = 0
        for k, (y, w) in enumerate(zip(nodes, blocks)):
            stop = start + y.size
            self.W[k, start:stop] = self.disc * w * self._density(y, self.centres[k])
            if touch is not None:
                self.P[k, start:stop] = self._touch(y, self.centres[k], touch)
            start = stop

    def _shared_nodes(self, touch) -> None:
        reach = _SUPPORT * self.std
        mus = self.centres + self.mean
        intervals = sorted(zip(mus - reach, mus + reach))
        merged = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        a_all, b_all = [], []
        for lo, hi in merged:
            cuts = [lo, hi]
            cuts += [float(b) for b in self._breaks if lo < b < hi]
            cuts += [float(g) for g in self.grid[(self.grid > lo) & (self.grid < hi)]]
            a, b = self._pieces(cuts)
            a_all.append(a)
            b_all.append(b)
        a, b = np.concatenate(a_all), np.concatenate(b_all)
        # whole lattice cells share one rule and one set of offsets: the Gaussian of a centre on a lattice node is
        # then a function of (cell - centre node, Gauss point) only, tabulated once (``_lattice_rows``)
        n = self.grid.size
        cell = np.clip(np.searchsorted(self.grid, a, side="right") - 1, 0, n - 2)
        whole = (a == self.grid[cell]) & (b == self.grid[cell + 1])
        width = self.h
        rule = (_GL_LONG if width > 0.5 * self.std else _GL_SHORT if width > 0.25 * self.std
                else _GL_CELL if width > self.std / 6.0 else _GL_FINE)
        gx, gw = rule
        aw, bw = a[whole], b[whole]
        y_whole = (0.5 * (aw + bw))[:, None] + (0.5 * (bw - aw))[:, None] * gx[None, :]
        w_whole = (0.5 * (bw - aw))[:, None] * gw[None, :]
        y_part, w_part = self._gauss_legendre(a[~whole], b[~whole])
        self.y = np.concatenate([y_whole.ravel(), y_part])
        w = np.concatenate([w_whole.ravel(), w_part])
        n_whole = y_whole.size
        node_cell = np.repeat(cell[whole], gx.size)
        node_point = np.tile(np.arange(gx.size), int(whole.sum()))
        self.W = _POOL.lease(self, (self.centres.size, self.y.size), fill=0.0)
        on_node = np.flatnonzero(np.isin(self.centres, self.grid))
        off_node = np.setdiff1d(np.arange(self.centres.size), on_node)
        # centres x nodes work runs on blocks of centres (``_row_blocks``: the allocator's small range, see
        # ``_MatrixPool``); every entry is the same elementwise expression as on the whole matrix
        if on_node.size:
            index = np.searchsorted(self.grid, self.centres[on_node])
            reach_cells = int(math.ceil((_SUPPORT * self.std + abs(self.mean)) / self.h)) + 2
            offsets = np.arange(-reach_cells, reach_cells + 1)
            xi = 0.5 * self.h * (1.0 + gx)
            z = (offsets[:, None] * self.h + xi[None, :] - self.mean) / self.std
            table = np.where(np.abs(z) <= _SUPPORT, _phi(z) / self.std, 0.0)
            for block in _row_blocks(on_node.size, self.y.size):
                rows_at = on_node[block]
                k = node_cell[None, :] - index[block, None]
                inside = np.abs(k) <= reach_cells
                rows = np.where(inside, table[np.clip(k + reach_cells, 0, offsets.size - 1), node_point[None, :]],
                                0.0)
                self.W[rows_at, :n_whole] = self.disc * w[None, :n_whole] * rows
                self.W[rows_at, n_whole:] = self.disc * w[None, n_whole:] * self._density(
                    self.y[None, n_whole:], self.centres[rows_at, None])
        for block in _row_blocks(off_node.size, self.y.size):
            rows_at = off_node[block]
            self.W[rows_at] = self.disc * w[None, :] * self._density(self.y[None, :], self.centres[rows_at, None])
        self.P = None
        if touch is not None:
            self.P = _POOL.lease(self, (self.centres.size, self.y.size))
            for block in _row_blocks(self.centres.size, self.y.size):
                self.P[block] = self._touch(self.y[None, :], self.centres[block, None], touch)

    def _touch(self, y: np.ndarray, centre, touch: Tuple[float, bool]) -> np.ndarray:
        """Brownian-bridge probability that the path from ``centre`` to ``y`` over the step touches the barrier
        (``SnowballQuadEngine._forward_seed_touch_probability``; 1 on the knocked side)."""
        log_barrier, is_reverse = touch
        denom = self.std * self.std
        if is_reverse:
            d0, d1 = log_barrier - y, log_barrier - centre
        else:
            d0, d1 = y - log_barrier, centre - log_barrier
        safe = (d0 > 0.0) & (d1 > 0.0)
        exponent = np.clip(np.where(safe, -2.0 * d0 * d1 / denom, 0.0), -745.0, 0.0)
        return np.where(safe, np.exp(exponent), 1.0)

    def _build_interpolation(self) -> None:
        """Local 4-point Lagrange cubic from the lattice to the nodes (edge constants beyond the lattice).

        The stencil is the cell's two nodes and one more each side, except next to a kink of the pre-event surfaces
        (``kinks``: a continuously monitored knock-in level, where the not-knocked-in surface joins the knocked-in
        one with a slope jump): there it takes four nodes on the point's own side, never across the kink."""
        n = self.grid.size
        t = (self.y - self.grid[0]) / self.h
        cell = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
        start = np.clip(cell - 1, 0, n - 4)
        tol = 1e-9 * self.h
        for kink in self._kinks:
            node = np.searchsorted(self.grid, kink - tol)            # first node at or above the kink
            last = np.searchsorted(self.grid, kink + tol, side="right") - 1   # last node at or below it
            above = self.y > kink
            crosses_up = above & (self.grid[start] < kink - tol)
            crosses_down = ~above & (self.grid[np.minimum(start + 3, n - 1)] > kink + tol)
            start = np.where(crosses_up, np.clip(node, 0, n - 4), start)
            start = np.where(crosses_down, np.clip(last - 3, 0, n - 4), start)
        index = start[:, None] + np.arange(4)[None, :]
        u = (self.y[:, None] - self.grid[index]) / self.h            # from each node: no cancellation
        coef = np.ones((self.y.size, 4))
        for j in range(4):
            for m in range(4):
                if m != j:
                    coef[:, j] *= u[:, m] / (j - m)
        below, above = self.y <= self.grid[0], self.y >= self.grid[-1]
        coef[below | above] = 0.0
        coef[below, 0] = 1.0
        index[below, 0] = 0
        coef[above, 0] = 1.0
        index[above, 0] = n - 1
        self._index = index
        self._coef = coef

    # --- linear maps -------------------------------------------------------------------------------------------

    def lift(self, values) -> np.ndarray:
        """Lattice surfaces (``..., n``) at the quadrature nodes (``..., n_y``)."""
        values = np.asarray(values, dtype=float)
        return np.einsum("...yk,yk->...y", values[..., self._index], self._coef)

    def lift_transpose(self, dual) -> np.ndarray:
        """The transpose of :meth:`lift`: node duals (``..., n_y``) onto the lattice (``..., n``)."""
        dual = np.asarray(dual, dtype=float)
        lead = dual.shape[:-1]
        flat = dual.reshape(-1, dual.shape[-1])
        out = np.zeros((flat.shape[0], self.grid.size), dtype=float)
        for k in range(4):
            idx = self._index[:, k]
            for r in range(flat.shape[0]):
                out[r] += np.bincount(idx, weights=flat[r] * self._coef[:, k], minlength=self.grid.size)
        return out.reshape(lead + (self.grid.size,))

    def expect(self, values_y) -> np.ndarray:
        """Per centre ``sum_y W * values`` (``..., n_y`` -> ``..., n_centres``)."""
        return np.asarray(values_y, dtype=float) @ self.W.T

    def expect_bridged(self, v_out, v_in) -> np.ndarray:
        """The not-knocked-in state's expectation over a first interval with continuous knock-in monitoring:
        ``sum_y W [(1 - P) v_out + P v_in]`` (``expect(v_out)`` without one)."""
        if self.P is None:
            return self.expect(v_out)
        v_out = np.asarray(v_out, dtype=float)
        if self._WP is None:
            # ``W * P``, leased and filled by blocks of centres (``_MatrixPool``)
            self._WP = _POOL.lease(self, self.W.shape)
            for block in _row_blocks(self.W.shape[0], self.W.shape[1]):
                np.multiply(self.W[block], self.P[block], out=self._WP[block])
        return v_out @ self.W.T + (np.asarray(v_in, dtype=float) - v_out) @ self._WP.T

    def readout_dual(self) -> np.ndarray:
        """The spot's (centre 0) quadrature weights: the dual F puts on the nodes (``v1_adjoint``). A copy: ``W``'s
        storage is leased (``_MatrixPool``) and goes back to the pool with this quadrature."""
        return self.W[0].copy()

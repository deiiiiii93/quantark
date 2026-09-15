"""Independent Gaussian-transition quadrature reference for discretely
monitored autocallables, with delta from the analytic density derivative.

Nothing here imports quantark.  The caller supplies the contract as payoff
callables and a resolved observation schedule, so the reference shares no
code, no grid and no event machinery with the engine under test, and no
convention either: whatever the product says its payoffs are is what this
prices.

Method.  With x = log S the log return over one monitoring interval is
Gaussian with mean m = (r - q - sigma^2/2) dt and variance v = sigma^2 dt:

    C(x)        = D * Integral F(y) p(y | x) dy
    dC/dx (x)   = D * Integral F(y) * (y - x - m)/v * p(y | x) dy

so the derivative differentiates the transition density, never the event
data.  The event map is applied when F is assembled, and every integral is
split at each active barrier and payoff kink, so no quadrature panel ever
straddles a jump.

What is stored between steps is the CONTINUATION pair (A, B), which is
smooth because it is an integral against a Gaussian.  The post-event
function F, which carries the jumps, is rebuilt at each step from A, B and
the event rules and is never interpolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

Payoff = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Market:
    r: float
    q: float
    sigma: float


@dataclass(frozen=True)
class Deal:
    """One discretely monitored deal, in times measured FROM the valuation date.

    ``payoff_alive`` is the maturity payoff of a path that never knocked out
    and never knocked in; ``payoff_ki`` that of a path that knocked in.  The
    maturity observations themselves are applied here, not by the caller.
    """

    maturity: float
    payoff_alive: Payoff
    payoff_ki: Payoff
    ki_barrier: Optional[float] = None
    ki_times: Tuple[float, ...] = ()
    ko_times: Tuple[float, ...] = ()
    ko_barriers: Tuple[float, ...] = ()
    ko_payoffs: Tuple[float, ...] = ()
    ko_survives_ki: bool = True
    #: Spot levels where a payoff has a kink or jump, so panels split there.
    terminal_breaks: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.ko_times) != len(self.ko_barriers) or len(self.ko_times) != len(self.ko_payoffs):
            raise ValueError("ko_times, ko_barriers and ko_payoffs must have equal length")
        for name, times in (("ki_times", self.ki_times), ("ko_times", self.ko_times)):
            arr = np.asarray(times, dtype=float)
            if arr.size and (np.any(arr <= 0.0) or np.any(arr > self.maturity * (1.0 + 1e-12))):
                raise ValueError(f"{name} must lie in (0, maturity]")
            if arr.size and np.any(np.diff(arr) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")


def _event_times(deal: Deal) -> np.ndarray:
    """Every time the state can change, maturity included, ascending."""
    times = set(float(t) for t in deal.ki_times)
    times.update(float(t) for t in deal.ko_times)
    times.add(float(deal.maturity))
    return np.array(sorted(times), dtype=float)


def _ko_at(deal: Deal, t: float) -> Optional[Tuple[float, float]]:
    """(barrier, cash) of the KO observation at ``t``, or None."""
    for u, barrier, cash in zip(deal.ko_times, deal.ko_barriers, deal.ko_payoffs):
        if abs(float(u) - t) <= 1e-12 * max(1.0, abs(t)):
            return float(barrier), float(cash)
    return None


def _is_ki_time(deal: Deal, t: float) -> bool:
    return any(abs(float(u) - t) <= 1e-12 * max(1.0, abs(t)) for u in deal.ki_times)


def _panel_integral(x_out: np.ndarray, f, breaks: Sequence[float], m: float, v: float,
                    *, n_gl: int, n_sd: float, derivative: bool) -> np.ndarray:
    """Integrate ``f`` against the transition density, splitting at ``breaks``.

    ``breaks`` are log-spot levels where ``f`` is discontinuous or kinked.
    Each panel between consecutive breaks is integrated on its own, so a
    jump is always a panel edge and never inside one.
    """
    sd = np.sqrt(v)
    lo = x_out + m - n_sd * sd
    hi = x_out + m + n_sd * sd
    edges = [lo] + [np.clip(np.full_like(x_out, b), lo, hi) for b in sorted(breaks)] + [hi]
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)

    total = np.zeros_like(x_out)
    for a, b in zip(edges[:-1], edges[1:]):
        half, mid = 0.5 * (b - a), 0.5 * (a + b)
        if not np.any(half > 0.0):
            continue
        y = mid[:, None] + half[:, None] * gl_x[None, :]
        d = y - x_out[:, None] - m
        dens = np.exp(-0.5 * d * d / v) / np.sqrt(2.0 * np.pi * v)
        kernel = dens * (d / v) if derivative else dens
        total += half * np.sum(gl_w[None, :] * f(y) * kernel, axis=1)
    return total


@dataclass
class _State:
    """The continuation pair, as callables of log spot."""

    alive: Callable[[np.ndarray], np.ndarray]
    ki: Callable[[np.ndarray], np.ndarray]


def _post_event(deal: Deal, state: _State, t: float, *, terminal: bool) -> Tuple[Callable, Callable, list]:
    """The pair of functions the previous interval integrates, plus its breaks.

    At ``t`` the observations resolve: a knock-out pays cash above its
    barrier, a knock-in moves the alive state onto the knocked-in branch at
    or below the barrier.  Both are applied pointwise here, which is the
    whole point: the jump lives in this function and is never interpolated.
    """
    breaks: list = []
    ko = _ko_at(deal, t)
    ki_here = _is_ki_time(deal, t) and deal.ki_barrier is not None
    x_ki = np.log(deal.ki_barrier) if ki_here else None
    if ko is not None:
        breaks.append(float(np.log(ko[0])))
    if ki_here:
        breaks.append(float(x_ki))
    if terminal:
        breaks.extend(float(np.log(b)) for b in deal.terminal_breaks)

    base_alive, base_ki = state.alive, state.ki

    def alive(y: np.ndarray) -> np.ndarray:
        out = base_alive(y)
        if ki_here:
            out = np.where(y <= x_ki, base_ki(y), out)
        if ko is not None:
            out = np.where(y >= np.log(ko[0]), ko[1], out)
        return out

    def knocked_in(y: np.ndarray) -> np.ndarray:
        out = base_ki(y)
        if ko is not None and deal.ko_survives_ki:
            out = np.where(y >= np.log(ko[0]), ko[1], out)
        return out

    return alive, knocked_in, sorted(set(breaks))


def value_and_delta(deal: Deal, market: Market, spot: float, *, n_grid: int = 2400,
                    n_gl: int = 240, n_sd: float = 9.0, span: float = 1.4):
    """Value and delta of an ALIVE path at the valuation date.

    The valuation date's own observation is assumed already resolved: the
    path is alive, which is exactly the state the engine is asked to mark.
    """
    x0 = float(np.log(spot))
    times = _event_times(deal)
    grid = np.linspace(x0 - span, x0 + span, n_grid)

    state = _State(alive=lambda y: deal.payoff_alive(np.exp(y)),
                   ki=lambda y: deal.payoff_ki(np.exp(y)))

    prev = 0.0 if times.size == 0 else float(times[-1])
    for k in range(times.size - 1, -1, -1):
        t = float(times[k])
        alive_f, ki_f, breaks = _post_event(deal, state, t, terminal=(k == times.size - 1))
        dt = t - (float(times[k - 1]) if k > 0 else 0.0)
        m = (market.r - market.q - 0.5 * market.sigma ** 2) * dt
        v = market.sigma ** 2 * dt
        discount = np.exp(-market.r * dt)
        if k == 0:
            query = np.array([x0])
            value = discount * _panel_integral(query, alive_f, breaks, m, v,
                                               n_gl=n_gl, n_sd=n_sd, derivative=False)[0]
            dvdx = discount * _panel_integral(query, alive_f, breaks, m, v,
                                              n_gl=n_gl, n_sd=n_sd, derivative=True)[0]
            return float(value), float(dvdx / spot)
        a = discount * _panel_integral(grid, alive_f, breaks, m, v,
                                       n_gl=n_gl, n_sd=n_sd, derivative=False)
        b = discount * _panel_integral(grid, ki_f, breaks, m, v,
                                       n_gl=n_gl, n_sd=n_sd, derivative=False)
        state = _State(alive=CubicSpline(grid, a, extrapolate=True),
                       ki=CubicSpline(grid, b, extrapolate=True))
        prev = t
    raise RuntimeError("a deal with no event times cannot be valued")


def no_event_control(deal: Deal, market: Market, spot: float, *, n_gl: int = 4000,
                     n_sd: float = 12.0) -> float:
    """One-shot European value of ``payoff_alive``, ignoring every observation.

    Run the same deal with its barriers removed and the many-step induction
    must reproduce this single integral.  It checks the stepping, the
    splines and the discounting together, and it needs no closed form.
    """
    x0 = float(np.log(spot))
    tau = float(deal.maturity)
    m = (market.r - market.q - 0.5 * market.sigma ** 2) * tau
    v = market.sigma ** 2 * tau
    breaks = [float(np.log(b)) for b in deal.terminal_breaks]
    total = _panel_integral(np.array([x0]), lambda y: deal.payoff_alive(np.exp(y)),
                            breaks, m, v, n_gl=n_gl, n_sd=n_sd, derivative=False)[0]
    return float(np.exp(-market.r * tau) * total)


# ---------------------------------------------------------------------------
# Operator form.
#
# The panel form above places its quadrature nodes relative to each output
# point, so nothing is reusable and a year of daily monitoring is out of
# reach.  Put the nodes on a FIXED mesh instead and the one-step transition
# becomes translation invariant, because a Gaussian transition density
# depends only on the separation y - x.  The whole step is then a banded
# matrix product against a kernel built once per distinct interval.
#
# The state is the continuation pair (A, B) sampled at the mesh's Gauss
# nodes.  Both are smooth everywhere, so no jump is ever interpolated: the
# event map only selects WHICH of them applies on each side of a barrier,
# and the barrier is an integration limit, handled by a sub-cell kernel
# whose weights are exact for the cell's own interpolating polynomial.
# That is the reference's version of "save the branches".
# ---------------------------------------------------------------------------

from scipy.special import ndtr  # noqa: E402  (kept beside the code that uses it)

_CELL_TOL = 1e-9


@dataclass(frozen=True)
class _Mesh:
    """A uniform mesh of cells, each carrying a Gauss-Legendre node set."""

    x_lo: float
    h: float
    n_cells: int
    theta: np.ndarray = field(repr=False)     # Gauss abscissae mapped to (0, 1)
    weights: np.ndarray = field(repr=False)

    @property
    def n_q(self) -> int:
        return int(self.theta.size)

    @property
    def x_hi(self) -> float:
        return self.x_lo + self.n_cells * self.h

    @property
    def nodes(self) -> np.ndarray:
        """(n_cells, n_q) log spots of every quadrature point."""
        return self.x_lo + (np.arange(self.n_cells)[:, None] + self.theta[None, :]) * self.h


def _build_mesh(lo: float, hi: float, target_h: float, n_q: int,
                align: Optional[float] = None) -> _Mesh:
    """A mesh covering [lo, hi]; ``align`` is placed exactly on a cell edge.

    Aligning matters only where the represented function is KINKED inside a
    cell, which for these products means the terminal payoff's strike: a
    cell polynomial drawn through a kink is worthless.  Barriers need no
    alignment, because the functions either side of one are smooth.
    """
    gx, gw = np.polynomial.legendre.leggauss(n_q)
    theta = 0.5 * (gx + 1.0)
    weights = 0.5 * gw
    if align is None or not (lo < align < hi):
        n_cells = max(int(np.ceil((hi - lo) / target_h)), 8)
        return _Mesh(lo, (hi - lo) / n_cells, n_cells, theta, weights)
    n_below = max(int(np.ceil((align - lo) / target_h)), 4)
    h = (align - lo) / n_below
    n_above = max(int(np.ceil((hi - align) / h)), 4)
    return _Mesh(align - n_below * h, h, n_below + n_above, theta, weights)


def _lagrange(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n_q, n_t) Lagrange basis of the cell's nodes, evaluated at ``t``."""
    n = theta.size
    basis = np.ones((n, t.size))
    for q in range(n):
        for p in range(n):
            if p != q:
                basis[q] *= (t - theta[p]) / (theta[q] - theta[p])
    return basis


class _Operator:
    """Transition kernels on a fixed mesh, built once per distinct interval.

    ``kernel(dt, lo, hi)[d, q, r]`` integrates the cell's q-th basis
    function over the sub-range [lo, hi] of a cell sitting ``d`` cells above
    the output point, against the transition density, and reports the result
    at the output cell's r-th node.  Only the separation enters, so one
    array serves every cell in the mesh.
    """

    def __init__(self, mesh: _Mesh, market: Market, *, n_sd: float = 9.0,
                 n_sub: Optional[int] = None) -> None:
        self.mesh = mesh
        self.market = market
        self.n_sd = float(n_sd)
        self._n_sub = int(n_sub) if n_sub else 2 * mesh.n_q + 8
        self._kernels: dict = {}
        self._bands: dict = {}

    def moments(self, dt: float) -> Tuple[float, float]:
        m = (self.market.r - self.market.q - 0.5 * self.market.sigma ** 2) * dt
        v = self.market.sigma ** 2 * dt
        return float(m), float(v)

    def band(self, dt: float) -> int:
        key = round(float(dt), 15)
        if key not in self._bands:
            m, v = self.moments(dt)
            reach = abs(m) + self.n_sd * np.sqrt(v)
            self._bands[key] = max(int(np.ceil(reach / self.mesh.h)) + 1, 1)
        return self._bands[key]

    def kernel(self, dt: float, lo: float = 0.0, hi: float = 1.0,
               *, derivative: bool = False) -> np.ndarray:
        key = (round(float(dt), 15), round(float(lo), 12), round(float(hi), 12), derivative)
        cached = self._kernels.get(key)
        if cached is not None:
            return cached
        mesh, (m, v), band = self.mesh, self.moments(dt), self.band(dt)
        gx, gw = np.polynomial.legendre.leggauss(self._n_sub)
        t = 0.5 * (hi - lo) * gx + 0.5 * (hi + lo)
        wt = 0.5 * (hi - lo) * gw
        basis = _lagrange(mesh.theta, t)                                  # (q, n)
        d = np.arange(-band, band + 1)
        sep = (d[:, None, None] + t[None, :, None] - mesh.theta[None, None, :]) * mesh.h
        z = sep - m
        dens = np.exp(-0.5 * z * z / v) / np.sqrt(2.0 * np.pi * v)
        if derivative:
            dens = dens * (z / v)
        out = mesh.h * np.einsum("n,qn,dnr->dqr", wt, basis, dens, optimize=True)
        self._kernels[key] = out
        return out


def _apply(mesh: _Mesh, kernel: np.ndarray, state: np.ndarray, band: int) -> np.ndarray:
    """Banded transition product over every cell of ``state`` at once."""
    padded = np.pad(state, ((band, band), (0, 0)), mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * band + 1, axis=0)
    flat = np.ascontiguousarray(window.transpose(0, 2, 1)).reshape(mesh.n_cells, -1)
    return flat @ kernel.reshape(-1, mesh.n_q)


def _apply_cell(mesh: _Mesh, kernel: np.ndarray, values: np.ndarray, k: int,
                band: int, out: np.ndarray) -> None:
    """Add the contribution of one cell, integrated over a sub-range."""
    i0, i1 = max(0, k - band), min(mesh.n_cells, k + band + 1)
    if i1 <= i0:
        return
    idx = k - np.arange(i0, i1) + band
    out[i0:i1] += np.einsum("q,nqr->nr", values, kernel[idx], optimize=True)


def _const_segment(mesh: _Mesh, m: float, v: float, lo: Optional[float],
                   hi: Optional[float], cash: float) -> np.ndarray:
    """A constant payoff over [lo, hi], integrated in closed form."""
    y = mesh.nodes
    sd = np.sqrt(v)
    upper = 1.0 if hi is None else ndtr((hi - y - m) / sd)
    lower = 0.0 if lo is None else ndtr((lo - y - m) / sd)
    return cash * (upper - lower)


def _segment_cells(mesh: _Mesh, lo: Optional[float], hi: Optional[float]):
    """Split [lo, hi] into whole cells plus at most two sub-cell pieces."""
    n = mesh.n_cells
    ua = 0.0 if lo is None else (lo - mesh.x_lo) / mesh.h
    ub = float(n) if hi is None else (hi - mesh.x_lo) / mesh.h
    ua = min(max(ua, 0.0), float(n))
    ub = min(max(ub, 0.0), float(n))
    if ub <= ua + _CELL_TOL:
        return None, []
    ka = int(np.floor(ua))
    fa = ua - ka
    if fa > 1.0 - _CELL_TOL:
        ka, fa = ka + 1, 0.0
    elif fa < _CELL_TOL:
        fa = 0.0
    kb = int(np.floor(ub))
    fb = ub - kb
    if fb < _CELL_TOL:
        fb = 0.0
    elif fb > 1.0 - _CELL_TOL:
        kb, fb = kb + 1, 0.0
    if ka == kb:
        return None, ([(ka, fa, fb)] if fb > fa and ka < n else [])
    partial = []
    if fa > 0.0 and ka < n:
        partial.append((ka, fa, 1.0))
    if fb > 0.0 and kb < n:
        partial.append((kb, 0.0, fb))
    first_full = ka if fa == 0.0 else ka + 1
    last_full = kb - 1
    whole = (first_full, min(last_full, n - 1)) if last_full >= first_full else None
    return whole, partial


def _operator_segments(deal: Deal, t: float):
    """Where each branch supplies the post-event function, low spot to high.

    A segment names its source: ``"A"`` for the surviving branch, ``"B"``
    for the knocked-in one, or a float for a cash region.
    """
    ko = _ko_at(deal, t)
    ki_here = _is_ki_time(deal, t) and deal.ki_barrier is not None
    x_ki = float(np.log(deal.ki_barrier)) if ki_here else None
    x_ko = float(np.log(ko[0])) if ko is not None else None
    if x_ki is not None and x_ko is not None and x_ki >= x_ko:
        raise ValueError("this reference assumes the knock-in sits below the knock-out")

    alive = []
    if ki_here:
        alive.append((None, x_ki, "B"))
    alive.append((x_ki, x_ko, "A"))
    if ko is not None:
        alive.append((x_ko, None, float(ko[1])))

    if ko is not None and deal.ko_survives_ki:
        knocked_in = [(None, x_ko, "B"), (x_ko, None, float(ko[1]))]
    else:
        knocked_in = [(None, None, "B")]
    return alive, knocked_in


def _integrate(op: _Operator, dt: float, state: dict, segments) -> np.ndarray:
    """One backward step: integrate the post-event function over one interval."""
    mesh = op.mesh
    m, v = op.moments(dt)
    band = op.band(dt)
    out = np.zeros((mesh.n_cells, mesh.n_q))
    merged = np.zeros_like(out)
    for lo, hi, src in segments:
        if not isinstance(src, str):
            out += _const_segment(mesh, m, v, lo, hi, float(src))
            continue
        values = state[src]
        whole, partial = _segment_cells(mesh, lo, hi)
        if whole is not None:
            merged[whole[0]:whole[1] + 1] = values[whole[0]:whole[1] + 1]
        for k, a, b in partial:
            _apply_cell(mesh, op.kernel(dt, a, b), values[k], k, band, out)
    out += _apply(mesh, op.kernel(dt), merged, band)
    return out


def _cell_interpolator(mesh: _Mesh, values: np.ndarray):
    """The cell's own Gauss-node polynomial, in barycentric form.

    Used only for the final leg, where the state is smooth: the jumps have
    all been consumed by earlier steps.
    """
    theta = mesh.theta
    bary = np.array([1.0 / np.prod([theta[q] - theta[p] for p in range(theta.size) if p != q])
                     for q in range(theta.size)])

    def evaluate(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        u = (y - mesh.x_lo) / mesh.h
        if np.any(u < -_CELL_TOL) or np.any(u > mesh.n_cells + _CELL_TOL):
            raise ValueError("the final leg reached beyond the mesh; widen span_sd")
        k = np.clip(np.floor(u).astype(int), 0, mesh.n_cells - 1)
        t = u - k
        diff = t[..., None] - theta
        exact = np.abs(diff) < 1e-14
        weight = bary / np.where(exact, 1.0, diff)
        cells = values[k]
        result = np.sum(weight * cells, axis=-1) / np.sum(weight, axis=-1)
        if exact.any():
            hit = np.take_along_axis(cells, np.argmax(exact, axis=-1)[..., None], axis=-1)[..., 0]
            result = np.where(exact.any(axis=-1), hit, result)
        return result

    return evaluate


def value_and_delta_operator(deal: Deal, market: Market, spots, *,
                             cells_per_sd: float = 2.0, n_q: int = 8,
                             span_sd: float = 10.0, n_gl: int = 240,
                             n_sd: float = 9.0):
    """Value and delta of an ALIVE path, for a whole ladder of spots.

    The backward induction from maturity to the first future observation is
    the expensive part and does not depend on the query spot, so it runs
    once for the whole ladder; only the final leg is per spot.
    """
    spots = np.asarray(spots, dtype=float).reshape(-1)
    if spots.size == 0:
        raise ValueError("value_and_delta_operator needs at least one spot")
    times = _event_times(deal)
    if times.size == 0:
        raise ValueError("a deal with no event times cannot be valued")

    steps = np.diff(np.concatenate(([0.0], times)))
    target_h = market.sigma * np.sqrt(steps.min()) / float(cells_per_sd)
    # The final leg integrates n_sd deviations out from the query spot, so
    # the mesh has to reach at least that far or the interpolant is asked
    # for values it does not hold.
    reach = max(float(span_sd), n_sd + 0.5) * market.sigma * np.sqrt(float(deal.maturity))
    x_spots = np.log(spots)
    lo = float(x_spots.min()) - reach
    hi = float(x_spots.max()) + reach
    if len(deal.terminal_breaks) > 1:
        raise NotImplementedError(
            "more than one terminal break needs sub-cell payoff sampling, "
            "which this reference does not yet build")
    align = float(np.log(deal.terminal_breaks[0])) if deal.terminal_breaks else None
    mesh = _build_mesh(lo, hi, target_h, n_q, align=align)
    op = _Operator(mesh, market, n_sd=n_sd)

    grid_spot = np.exp(mesh.nodes)
    state = {"A": np.asarray(deal.payoff_alive(grid_spot), dtype=float),
             "B": np.asarray(deal.payoff_ki(grid_spot), dtype=float)}

    for k in range(times.size - 1, 0, -1):
        t = float(times[k])
        dt = t - float(times[k - 1])
        segs_alive, segs_ki = _operator_segments(deal, t)
        discount = np.exp(-market.r * dt)
        state = {"A": discount * _integrate(op, dt, state, segs_alive),
                 "B": discount * _integrate(op, dt, state, segs_ki)}

    t1 = float(times[0])
    pair = _State(alive=_cell_interpolator(mesh, state["A"]),
                  ki=_cell_interpolator(mesh, state["B"]))
    alive_f, _ki_f, breaks = _post_event(deal, pair, t1, terminal=(times.size == 1))
    m = (market.r - market.q - 0.5 * market.sigma ** 2) * t1
    v = market.sigma ** 2 * t1
    discount = np.exp(-market.r * t1)
    value = discount * _panel_integral(x_spots, alive_f, breaks, m, v,
                                       n_gl=n_gl, n_sd=n_sd, derivative=False)
    dvdx = discount * _panel_integral(x_spots, alive_f, breaks, m, v,
                                      n_gl=n_gl, n_sd=n_sd, derivative=True)
    return value, dvdx / spots

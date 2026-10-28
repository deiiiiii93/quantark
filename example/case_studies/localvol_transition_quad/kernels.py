"""Interval transition operators on the finite-volume log grid.

A kernel maps point values at the cell centres at ``t1`` to discounted point
values at ``t0``. Row i is the law of ln S_{t1} given ln S_{t0} = c_i, as a
probability per cell. Rounding each landing point to its cell centre adds the
cell's quantization variance (and, where widths change, a mean bias), so
Gaussian rows are moment matched on the centres. The end cells are
semi-infinite in probability: mass is conserved and the out-of-domain region
inherits the edge value.
"""
from __future__ import annotations

import math
from enum import Enum

import numpy as np
from scipy.linalg.lapack import zgtsv
from scipy.special import ive, ndtr

from quantark.util.numerical import Tolerance

from .grid import LogGrid, UnresolvedGrid
from .market import Market

MOMENT_PASSES = 2


class KernelKind(Enum):
    BSM_EXACT = "bsm_exact"                  # constant-vol control (tests only)
    CTMC = "ctmc"                            # Markov-chain generator, exact exponential (kernel A)
    GAUSS_RICHARDSON = "gauss_richardson"    # moment-matched Euler Gaussian + Richardson (kernel B)


MAX_PIECE = 1.0 / 252.0      # longest single time piece: sigma moves within a slab (resolution, not semantics)


def pieces_for(length: float, minimum: int, max_piece: float) -> int:
    """Equal pieces for a pillar-cut interval: at least ``minimum``, none longer than ``max_piece``."""
    return max(minimum, math.ceil(length / max_piece - Tolerance.ZERO))


def split_at_pillars(t0: float, t1: float, pillars: np.ndarray) -> list[tuple[float, float]]:
    """Sub-intervals of [t0, t1] cut at surface time pillars, so sigma**2 is smooth in t inside each."""
    points = [t0] + [float(p) for p in pillars if t0 < p < t1] + [t1]
    return list(zip(points[:-1], points[1:]))


BOUNDARY_CELLS = 16


def interior_rows(grid: LogGrid, sd: float, n_sd: float = 8.0) -> np.ndarray:
    """Rows at least ``n_sd`` interval standard deviations AND ``BOUNDARY_CELLS`` cells from both edges.

    The cell floor matters for short pillar-cut pieces: a Markov chain moves cell
    by cell, so the reflecting end rows reach a fixed number of neighbours even
    when the interval's standard deviation is a fraction of a cell (spec D16).
    """
    c = grid.centres
    rows = (c - grid.edges[0] >= n_sd * sd) & (grid.edges[-1] - c >= n_sd * sd)
    rows[:BOUNDARY_CELLS] = False
    rows[-BOUNDARY_CELLS:] = False
    return rows


def _cell_probabilities(grid: LogGrid, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    cdf = ndtr((grid.edges[None, 1:-1] - mean[:, None]) / sd[:, None])
    out = np.empty((mean.size, grid.size))
    out[:, 0] = cdf[:, 0]
    out[:, 1:-1] = np.diff(cdf, axis=1)
    out[:, -1] = 1.0 - cdf[:, -1]
    return out


def _initial_net_variance(grid: LogGrid, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    cell = np.clip(np.searchsorted(grid.edges, mean) - 1, 0, grid.size - 1)
    return var - grid.widths[cell] ** 2 / 12.0


def gaussian_rows(grid: LogGrid, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Dense cell probabilities of a Gaussian per source row, moment matched on the CENTRES.

    The Gaussian's mean and variance are corrected so that the realized law on
    the centres has the target first two moments (``MOMENT_PASSES`` passes).
    """
    c = grid.centres
    mu = np.array(mean, dtype=float)
    s2 = _initial_net_variance(grid, mean, var)
    for k in range(MOMENT_PASSES + 1):
        if np.any(s2 <= 0):
            raise UnresolvedGrid("interval variance below the cell quantization variance; refine the grid")
        p = _cell_probabilities(grid, mu, np.sqrt(s2))
        if k < MOMENT_PASSES:
            m1 = p @ c
            mu = mu + (mean - m1)
            s2 = s2 + (var - (p @ (c * c) - m1 * m1))
    return p


def banded_gaussian(grid: LogGrid, mean: np.ndarray, var: np.ndarray, n_sd: float = 8.0):
    """Moment-matched Gaussian rows restricted to a +/- n_sd window: (indices, probabilities).

    Mass beyond the window (< Phi(-n_sd)) is dropped and appears in the
    validated mass residual. Windows are shifted, not clipped, at the domain ends.
    """
    n, c, e = grid.size, grid.centres, grid.edges
    sd_max = float(np.sqrt(np.max(var)))
    lo = np.searchsorted(e, mean - n_sd * sd_max) - 1
    hi = np.searchsorted(e, mean + n_sd * sd_max)
    width = int(min(n, np.max(hi - lo) + 1))
    start = np.clip(lo, 0, n - width)
    idx = start[:, None] + np.arange(width)[None, :]
    upper = np.where(idx < n - 1, e[np.minimum(idx + 1, n)], np.inf)
    lower = np.where(idx > 0, e[idx], -np.inf)
    cw = c[idx]
    mu = np.array(mean, dtype=float)
    s2 = _initial_net_variance(grid, mean, var)
    for k in range(MOMENT_PASSES + 1):
        if np.any(s2 <= 0):
            raise UnresolvedGrid("interval variance below the cell quantization variance; refine the grid")
        sd = np.sqrt(s2)[:, None]
        p = ndtr((upper - mu[:, None]) / sd) - ndtr((lower - mu[:, None]) / sd)
        if k < MOMENT_PASSES:
            m1 = np.sum(p * cw, axis=1)
            mu = mu + (mean - m1)
            s2 = s2 + (var - (np.sum(p * cw * cw, axis=1) - m1 * m1))
    return idx, p


class BSMExactKernel:
    """Exact Black-Scholes transition for a constant local-vol surface (validation control)."""

    kind = KernelKind.BSM_EXACT

    def __init__(self):
        self.diagnostics: list[dict] = []
        self._cache: dict = {}

    @staticmethod
    def _sigma(market: Market) -> float:
        grid = market.surface.lv_grid
        if np.ptp(grid) != 0.0:
            raise ValueError("BSM_EXACT requires a constant local-vol surface")
        return float(grid.flat[0])

    def matrix(self, grid: LogGrid, market: Market, t0: float, t1: float) -> np.ndarray:
        h = t1 - t0
        key = (grid.edges.tobytes(), h, market.r, market.q, self._sigma(market))
        cached = self._cache.get(key)
        if cached is None:
            sigma = key[-1]
            mean = grid.centres + (market.r - market.q - 0.5 * sigma ** 2) * h
            cached = gaussian_rows(grid, mean, np.full(grid.size, sigma ** 2 * h))
            self._cache = {key: cached}
        return cached

    def propagate(self, values: np.ndarray, grid: LogGrid, market: Market, t0: float, t1: float) -> np.ndarray:
        p = self.matrix(grid, market, t0, t1)
        return math.exp(-market.r * (t1 - t0)) * (values @ p.T)


class _ValidatingKernel:
    """Validation-mode bookkeeping (spec D16). Timed mode records and checks nothing per operator."""

    def __init__(self, validate: bool = False, tolerance: float | None = None):
        self.validate = validate
        self.tolerance = tolerance          # per-operator; the engine sets it from the PV budget
        self.diagnostics: list[dict] = []
        self.operators = 0

    def _check(self, record: dict) -> None:
        self.diagnostics.append(record)
        if self.tolerance is None:
            return
        for key in ("mass", "forward", "semigroup"):
            if record.get(key, 0.0) > self.tolerance:
                raise UnresolvedGrid(f"{type(self).__name__} {key} residual {record[key]:.3g} exceeds "
                                     f"{self.tolerance:.3g} on [{record['t0']:.6f}, {record['t1']:.6f}]")

    @staticmethod
    def _identities(grid: LogGrid, market: Market, apply, dt: float, sd: float) -> dict:
        s = np.exp(grid.centres)
        moved = apply(np.vstack((np.ones_like(s), s)))
        rows = interior_rows(grid, max(sd, Tolerance.ZERO))
        mass = float(np.max(np.abs(moved[0, rows] - 1.0)))
        forward = float(np.max(np.abs(moved[1, rows] / (s[rows] * math.exp((market.r - market.q) * dt)) - 1.0)))
        return dict(mass=mass, forward=forward, rows=int(rows.sum()))


class CTMCKernel(_ValidatingKernel):
    """Kernel A: tridiagonal Markov-chain generator on the cell centres, applied by exact exponential.

    Each interior row has an up and a down rate chosen so that the stock forward
    is a martingale, ``u(e^{h+}-1) + d(e^{-h-}-1) = r - q``, and the log-variance
    rate is the local variance, ``u h+**2 + d h-**2 = sigma**2``; mass and forward
    hold by construction, the end rows reflect, and a negative rate is refused,
    never clipped. ``exp(dt Q) v`` is the Chebyshev-Bessel series on the
    Gershgorin interval, truncated a priori at ``series_tolerance``. Intervals are
    cut at surface pillars and each piece uses Q(t_mid) (Magnus order 2).
    """

    kind = KernelKind.CTMC

    def __init__(self, validate: bool = False, tolerance: float | None = None, series_tolerance: float = 1e-13,
                 time_pieces: int = 1, backend: str = "contour", contour_nodes: int = 24,
                 max_piece: float = MAX_PIECE):
        super().__init__(validate, tolerance)
        if not isinstance(time_pieces, int) or time_pieces < 1:
            raise ValueError("positive integer time_pieces required")
        if not math.isfinite(max_piece) or max_piece <= 0:
            raise ValueError("positive finite max_piece required")
        self.max_piece = max_piece
        if backend not in ("chebyshev", "contour"):
            raise ValueError("backend must be 'chebyshev' or 'contour'")
        if not isinstance(contour_nodes, int) or contour_nodes < 8 or contour_nodes % 2:
            raise ValueError("contour_nodes must be an even integer >= 8")
        self.series_tolerance = series_tolerance
        self.time_pieces = time_pieces       # Magnus-2 pieces per pillar-cut interval (time-error ladder)
        self.backend = backend
        self.matvecs = 0
        self.solves = 0
        # Weideman (2006) optimized Talbot contour z(t) = N(-0.6122 + 0.5017 t cot(0.6407 t) + 0.2645 i t);
        # trapezoid error ~ 3.89**-N for operators with spectrum on (-inf, 0]. Conjugate symmetry keeps
        # the upper half of the nodes (N/2 complex tridiagonal solves per application).
        n = contour_nodes
        theta = -np.pi + (np.arange(1, n + 1) - 0.5) * 2 * np.pi / n
        theta = theta[theta > 0]
        alpha = 0.6407
        cot = 1.0 / np.tan(alpha * theta)
        self._nodes = n * (-0.6122 + 0.5017 * theta * cot + 0.2645j * theta)
        dz = n * (0.5017 * (cot - alpha * theta / np.sin(alpha * theta) ** 2) + 0.2645j)
        self._weights = 2.0 * (2 * np.pi / n) / (2j * np.pi) * np.exp(self._nodes) * dz

    def rates(self, grid: LogGrid, market: Market, t: float) -> tuple[np.ndarray, np.ndarray]:
        c = grid.centres
        sigma2 = np.asarray(market.surface.local_vol(np.exp(c), t), dtype=float) ** 2
        hp, hm = np.diff(c)[1:], np.diff(c)[:-1]
        ep, em = np.expm1(hp), np.expm1(-hm)
        det = ep * hm ** 2 - em * hp ** 2
        carry = market.r - market.q
        s2 = sigma2[1:-1]
        up, down = np.zeros(c.size), np.zeros(c.size)
        up[1:-1] = (carry * hm ** 2 - s2 * em) / det
        down[1:-1] = (s2 * ep - carry * hp ** 2) / det
        up[0] = sigma2[0] / (c[1] - c[0]) ** 2
        down[-1] = sigma2[-1] / (c[-1] - c[-2]) ** 2
        if np.any(up < 0) or np.any(down < 0):
            raise UnresolvedGrid("negative Markov-chain rate (drift-dominated cell); refine the grid")
        return up, down

    def series_terms(self, z: float) -> np.ndarray:
        """Bessel coefficients ive(k, z) up to the a-priori truncation point."""
        k = np.arange(0, int(z + 20.0 + 6.0 * math.sqrt(z + 1.0)))
        coeff = ive(k, z)
        tail = 2.0 * np.cumsum(coeff[::-1])[::-1]
        below = np.nonzero(tail < self.series_tolerance)[0]
        if below.size == 0:
            raise UnresolvedGrid("Chebyshev-Bessel series did not reach its tolerance")
        return coeff[:max(int(below[0]), 2)]

    def expm_apply(self, values: np.ndarray, up: np.ndarray, down: np.ndarray, dt: float) -> np.ndarray:
        """exp(dt Q) applied to the rows of ``values`` (channels x cells)."""
        if self.backend == "contour":
            return self._contour_apply(values, up, down, dt)
        return self._chebyshev_apply(values, up, down, dt)

    def _contour_apply(self, values, up, down, dt):
        """Talbot-contour quadrature of exp(A) v = (1/2 pi i) int e^z (zI - A)^{-1} v dz, A = dt Q."""
        lower, upper = -dt * down[1:], -dt * up[:-1]
        base = dt * (up + down)
        rhs = np.ascontiguousarray(values.T, dtype=complex)
        out = np.zeros(values.shape[::-1])
        for z, w in zip(self._nodes, self._weights):
            _, _, _, x, info = zgtsv(lower.astype(complex), base + z, upper.astype(complex), rhs)
            if info != 0:
                raise UnresolvedGrid(f"singular contour resolvent (zgtsv info={info})")
            out += (w * x).real
        self.solves += self._nodes.size
        return np.ascontiguousarray(out.T)

    def _chebyshev_apply(self, values, up, down, dt):
        rate = up + down
        nu = dt * 2.0 * float(np.max(rate))
        if nu == 0.0:
            return values.copy()
        a = 2.0 * dt / nu                      # A = (2/nu) dt Q + I has spectrum in [-1, 1]
        upv, downv, diagv = a * up[:-1], a * down[1:], 1.0 - a * rate

        def matvec(v):
            out = diagv * v
            out[:, :-1] += upv * v[:, 1:]
            out[:, 1:] += downv * v[:, :-1]
            return out

        coeff = self.series_terms(0.5 * nu)
        prev, cur = values, matvec(values)
        out = coeff[0] * prev + 2.0 * coeff[1] * cur
        for k in range(2, coeff.size):
            prev, cur = cur, 2.0 * matvec(cur) - prev
            out += 2.0 * coeff[k] * cur
        self.matvecs += coeff.size - 1
        return out

    def propagate(self, values: np.ndarray, grid: LogGrid, market: Market, t0: float, t1: float) -> np.ndarray:
        out = values
        for lo, hi in reversed(split_at_pillars(t0, t1, market.surface.time_grid)):   # latest piece first
            pieces = pieces_for(hi - lo, self.time_pieces, self.max_piece)
            step = (hi - lo) / pieces
            for k in range(pieces - 1, -1, -1):               # backward: the latest piece acts first
                a, b = lo + k * step, lo + (k + 1) * step
                up, down = self.rates(grid, market, 0.5 * (a + b))
                self.operators += 1
                if self.validate:
                    self._validate(grid, market, up, down, a, b, out)
                out = self.expm_apply(out, up, down, b - a)
        return math.exp(-market.r * (t1 - t0)) * out

    def _validate(self, grid, market, up, down, a, b, values):
        dt = b - a
        sd = math.sqrt(float(np.max(market.surface.lv_grid)) ** 2 * dt)
        record = dict(t0=a, t1=b, **self._identities(grid, market, lambda v: self.expm_apply(v, up, down, dt), dt, sd))
        rows = interior_rows(grid, sd)
        once = self.expm_apply(values, up, down, 2 * dt)
        twice = self.expm_apply(self.expm_apply(values, up, down, dt), up, down, dt)
        scale = max(1.0, float(np.max(np.abs(values))))
        record["semigroup"] = float(np.max(np.abs(once - twice)[:, rows])) / scale
        self._check(record)


class GaussRichardsonKernel(_ValidatingKernel):
    """Kernel B: moment-matched Euler-in-log Gaussian substeps, Richardson-extrapolated per interval.

    One substep from centre c_i over dt is N(c_i + (r - q - s_i**2/2) dt, s_i**2 dt)
    with s_i = sigma_loc(e^{c_i}, t_mid) (Euler in space, midpoint in time),
    banded and moment matched on the centres. Each pillar-cut piece is advanced
    with ``substeps`` and ``2*substeps`` substeps and combined as 2*fine - coarse.
    """

    kind = KernelKind.GAUSS_RICHARDSON

    def __init__(self, substeps: int = 1, richardson: bool = True, validate: bool = False,
                 tolerance: float | None = None, n_sd: float = 8.0, max_piece: float = MAX_PIECE):
        super().__init__(validate, tolerance)
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("positive integer substeps required")
        if not math.isfinite(max_piece) or max_piece <= 0:
            raise ValueError("positive finite max_piece required")
        self.substeps, self.richardson, self.n_sd, self.max_piece = substeps, richardson, n_sd, max_piece

    def step(self, grid: LogGrid, market: Market, t: float, dt: float):
        c = grid.centres
        sigma2 = np.asarray(market.surface.local_vol(np.exp(c), t + 0.5 * dt), dtype=float) ** 2
        mean = c + (market.r - market.q - 0.5 * sigma2) * dt
        self.operators += 1
        return banded_gaussian(grid, mean, sigma2 * dt, self.n_sd)

    @staticmethod
    def apply(values: np.ndarray, op) -> np.ndarray:
        idx, p = op
        return np.einsum("cij,ij->ci", values[:, idx], p)

    def _advance(self, values, grid, market, a, b, m):
        dt = (b - a) / m
        out = values
        for k in range(m - 1, -1, -1):          # backward in time: the last substep acts first
            op = self.step(grid, market, a + k * dt, dt)
            if self.validate:
                sd = math.sqrt(float(np.max(market.surface.lv_grid)) ** 2 * dt)
                self._check(dict(t0=a + k * dt, t1=a + (k + 1) * dt,
                                 **self._identities(grid, market, lambda v: self.apply(v, op), dt, sd)))
            out = self.apply(out, op)
        return out

    def propagate(self, values: np.ndarray, grid: LogGrid, market: Market, t0: float, t1: float) -> np.ndarray:
        out = values
        for a, b in reversed(split_at_pillars(t0, t1, market.surface.time_grid)):     # latest piece first
            m = pieces_for(b - a, self.substeps, self.max_piece)
            fine = self._advance(out, grid, market, a, b, 2 * m if self.richardson else m)
            out = 2.0 * fine - self._advance(out, grid, market, a, b, m) if self.richardson else fine
        return math.exp(-market.r * (t1 - t0)) * out

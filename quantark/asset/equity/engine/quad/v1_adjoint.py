"""The forward adjoint pass of the QUAD V1 recursion (unified event statistics spec T17, AC13, AC14).

The V1 statistics recursion (``SnowballQuadEngine._compute_event_stats``, ``KOResetSnowballQuadEngine``) marches
indicator and cash columns backward through the price's own operators: per step, the observation's event map on the
two note states (not knocked in ``out``, knocked in ``in``) and then the diffusion back to the previous step (the
Brownian-bridge correction on the ``out`` state under continuous knock-in). This module runs the exact transpose of
that march forward from the readout at the spot, carrying a dual (a discounted density) on each state's nodes:

* readout: the linear interpolation at the spot (``QuadratureMath.interpolate``), or the transition readout
  (``_readout_at_spot``) as one row vector -- Simpson weights, the spectral filter and the tail term transposed;
* diffusion (``_diffuse_fft``): the FFT convolution transposed as the convolution with the reversed kernel (the
  spectral filter is real and symmetric, so it transposes to itself), the Simpson weights and the tail correction
  transposed; the bridge correction (``_diffuse_with_bridge``) transposed kernel by kernel;
* the event map: read off the engine's own event code (``_apply_observation`` / ``_apply_reset_observation``) by
  coloured unit probes, which recover a local (banded) linear map exactly; every event is checked against a random
  probe and refused (:class:`AdjointUnavailable`) if it is not local in the probe band. Its affine part (the
  knock-out sources) is the event applied to zero surfaces.

Masses (all discounted to the valuation; a probability divides by ``DF(t)``):

* the knock-out source of a record: ``dual . source`` at its event (AC14 against the backward knock-out column);
  the clean knock-out (of not-knocked-in paths) is the ``out`` part;
* first knock-in at an event: the dual the event moves from the ``out`` state onto the ``in`` continuation -- a path
  knocked out at the same instant is in the knock-out source, never in the continuation, so the knock-out wins a
  tie exactly as the backward event decides it; under continuous knock-in also the bridge mass of each interval;
* terminal: ``dual . terminal`` on each state (the maturity and knocked-in settlement columns, unit and cash).

Extension hook (Phoenix coupon columns, spec D7): ``MarchLog.extension`` may carry extra dual rows through the same
march -- diffused (stacked with the main dual) and bridged exactly as the main dual, mapped at each event by the
extension itself (``EventStep.extension`` / ``FirstStepLog.extension``: its per-step data) -- so the extra masses ride
the ONE forward march. ``extension.start()`` returns a run with ``readout(beta_out, beta_in) -> (x_out, x_in)`` (rows x
nodes per state), ``event(data, banded, n, time, alpha_out, alpha_in, x_out, x_in) -> (x_out, x_in)`` (``banded``: the
step's linear event map, ``alpha_*``: the main dual before it), ``finish(beta_out, beta_in, x_out, x_in)`` at the end
(on the exact first step's nodes when it reads the terminal there) and ``result()`` (``ForwardResult.extension``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erfc

from quantark.asset.equity.engine.pde.core.statistics.adjoint import AdjointUnavailable
from quantark.asset.equity.engine.quad.local_step import local_bridge_hit, local_step_transpose, local_step_weights
from quantark.util.exceptions import PricingError

__all__ = [
    "AdjointUnavailable",
    "EventStep",
    "FirstStepLog",
    "ForwardResult",
    "MarchLog",
    "forward_pass",
    "probe_event",
    "transpose_pair",
]

#: probe half band of the event maps: a dual-cell projection couples a node to its neighbours (one cell); the joint
#: Phoenix projection to at most two
_HALF_BAND = 4
_COLOURS = 2 * _HALF_BAND + 1
#: the random-probe check of an event map's band (relative to the probe's magnitude)
_BAND_CHECK_RTOL = 1e-11


@dataclass(eq=False)
class EventStep:
    """One step of the backward march: the event map at ``time`` and the diffusion back to the previous step.

    ``event(v_out, v_in, source)`` applies the observation to stacked surfaces (rows x nodes); ``source`` is
    ``None`` for the linear part, else the key of one knock-out source (``sources``), applied with every other source
    zero. ``diffusion``: ``(tau, alpha, beta, prefactor, omega)`` of the step into this slice; ``bridge``:
    ``(log_barrier, vol, dt, is_reverse)`` when the ``out`` state rolls back with the knock-in bridge;
    ``out_diffused``: whether the ``out`` state rolls back at all (KO-reset: not after its first schedule ends);
    ``seed_out``: the ``out`` state is seeded with its terminal value at this step, before its event (KO-reset).
    ``source_units``: per source key, the unit the event puts on the breach (the backward knock-out column's
    ``DF(settle)/DF(obs)``); the clean knock-out mass is the source's mass over its unit -- a probability mass
    discounted to the step (``DF(obs) P``), never ``DF(settle) P`` (CP6).
    """

    time: float
    event: Optional[Callable]
    sources: Tuple = ()
    diffusion: Tuple = ()
    bridge: Optional[Tuple] = None
    out_diffused: bool = True
    seed_out: bool = False
    source_units: Dict = field(default_factory=dict)
    #: the march extension's data for this step (``None``: the extension maps its rows by the step's event map)
    extension: Optional[object] = None
    #: a knock-in at the widest level of an env instant's records (0.16.0 zero-gap rule): the same event map at each
    #: narrower effective level, contract order (all but the widest); their first knock-in masses are cumulative
    ki_parts: Tuple = ()


@dataclass(eq=False)
class FirstStepLog:
    """The exact first step of the march (``first_step``, S3b), when the engine priced it: the first event and the
    step from the valuation instant are one linear functional of the PRE-event surfaces at the first event,
    ``readout . event_y(lift(pre))`` on the quadrature nodes (bridged on the ``out`` state under continuous
    knock-in). ``event`` is the engine's event map on the nodes (pointwise); ``terminal_out`` / ``terminal_in``:
    a state's maturity seed on the nodes when the first event is where it is seeded (its masses read there);
    ``values_out`` / ``values_in``: the columns at the spot (per row) for each readout state."""

    quadrature: object
    event: Callable
    terminal_out: Optional[Dict[str, np.ndarray]] = None
    terminal_in: Optional[Dict[str, np.ndarray]] = None
    values_out: Optional[np.ndarray] = None
    values_in: Optional[np.ndarray] = None
    #: the march extension's data for the first event, on the nodes
    extension: Optional[object] = None
    #: ``EventStep.ki_parts`` of the first event, on the nodes
    ki_parts: Tuple = ()


@dataclass(eq=False)
class MarchLog:
    """The operators of one statistics march, logged in backward order by the engine (``steps[0]`` is the last
    step); ``engine`` and ``math_utils`` give the diffusion and bridge kernels."""

    engine: object
    math_utils: object
    readout_in: bool
    transition: Optional[Tuple] = None          # (prefactor, alpha, beta, tau) of the transition readout
    steps: List[EventStep] = field(default_factory=list)
    #: backward column readouts at the spot, by name (filled by the engine; AC14 compares against them)
    columns: Dict[str, float] = field(default_factory=dict)
    terminal_out: Dict[str, np.ndarray] = field(default_factory=dict)
    terminal_in: Dict[str, np.ndarray] = field(default_factory=dict)
    #: the exact first step, replacing the readout, the first diffusion and the first event (``None``: the lattice's)
    first_step: Optional[FirstStepLog] = None
    #: extra dual rows riding the pass (module docstring; ``None``: none)
    extension: Optional[object] = None


@dataclass(frozen=True)
class ForwardResult:
    """Per step in forward time (``times``): the first knock-in mass at the event (``first_knock_in``), the bridge
    mass of the interval ending there (``interval_knock_in``), the clean knock-out mass (``clean_knock_out``: the
    not-knocked-in probability mass knocked out there, discounted to the step, ``DF(t) P``); per source key its mass
    on the backward column's unit (``DF(t) P DF(settle)/DF(t)``); the ``out`` dual at each interval start (``interval_law``, the first-passage split's
    weights); the terminal masses per name."""

    times: np.ndarray
    first_knock_in: np.ndarray
    interval_knock_in: np.ndarray
    clean_knock_out: np.ndarray
    sources: Dict[object, float]
    interval_start_dual: List[Optional[np.ndarray]]
    terminal: Dict[str, float]
    #: the march extension's result (``None``: no extension)
    extension: Optional[object] = None
    #: per step holding a same-instant knock-in split (``EventStep.ki_parts``): the cumulative first knock-in mass at
    #: each narrower effective level, contract order
    knock_in_parts: Dict[int, np.ndarray] = field(default_factory=dict)


def _part_masses(parts, n: int, time: float, alpha_out: np.ndarray) -> np.ndarray:
    """The first knock-in mass of each ``ki_parts`` event map (the in -> out block on the ``out`` dual)."""
    return np.array([float(np.sum(_BandedEvent(part, n, time).transpose("in", "out", alpha_out))) for part in parts],
                    dtype=float)


# --- the transposed operators ---------------------------------------------------------------------------------------


def _simpson_coefficients(math_utils) -> np.ndarray:
    n = math_utils.grid.size
    return math_utils.simpson_weights(np.ones(n), 0, n - 1, (n - 1) % 2)[:n]


def _tail_vectors(math_utils, alpha: float, beta: float, tau: float):
    grid = math_utils.grid
    sqrt_tau = math.sqrt(tau)
    u_left = (grid - grid[0] + 2.0 * tau * alpha) / (2.0 * sqrt_tau)
    u_right = (grid - grid[-1] + 2.0 * tau * alpha) / (2.0 * sqrt_tau)
    scale = 0.5 * math.exp(tau * (alpha * alpha - beta))
    return scale * erfc(u_left), scale * erfc(-u_right)


def diffuse_transpose(values: np.ndarray, math_utils, omega: np.ndarray, prefactor: float, alpha: float,
                      beta: float, tau: float, *, engine=None) -> np.ndarray:
    """``D^T`` of ``SnowballQuadEngine._diffuse_fft`` on stacked duals (rows x nodes).

    A step the engine applies by its local stencil (``engine._local_step``) transposes as the stencil.

    ``D v = prefactor * conv(omega, c v) + v_0 TL + v_last TR`` (``c`` the Simpson coefficients); the convolution's
    output ``i`` pairs input ``j`` through the kernel at ``n - 1 + i - j`` (spectrally filtered, circular), so its
    transpose is the same convolution with the kernel reversed (the filter is real and even)."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if engine is not None and engine._local_step(tau, math_utils.h):
        return local_step_transpose(values, local_step_weights(float(math_utils.h), alpha, beta, tau))
    n = math_utils.grid.size
    padded = np.zeros((values.shape[0], 2 * n - 1), dtype=float)
    padded[:, :n] = values
    conv = math_utils.convolution_fft_many(np.ascontiguousarray(omega[::-1]), padded)
    out = _simpson_coefficients(math_utils)[None, :] * (prefactor * conv)
    left, right = _tail_vectors(math_utils, alpha, beta, tau)
    out[:, 0] += values @ left
    out[:, -1] += values @ right
    return out


def bridge_transpose(dual_out: np.ndarray, engine, math_utils, prefactor: float, alpha: float, tau: float,
                     log_barrier: float, vol: float, dt: float, is_reverse: bool, *, beta: float = 0.0) -> np.ndarray:
    """``B^T`` of the bridge correction of ``_diffuse_with_bridge`` (``correction[target] += P k work[source]``);
    on a local step, the transpose of the local stencil's touch-weighted correction (``beta`` gives its discount)."""
    grid = math_utils.grid
    if engine._local_step(tau, math_utils.h):
        weights = local_step_weights(float(math_utils.h), alpha, beta, tau)
        return local_step_transpose(dual_out, weights, local_bridge_hit(grid, log_barrier, vol * vol * dt, is_reverse))
    band = engine._bridge_band(tau, math_utils.h, grid.size)
    kernels = engine._bridge_kernels(math_utils, band, tau, alpha, vol * vol * dt, log_barrier, is_reverse)
    out = np.zeros_like(dual_out)
    for source, target, kernel in kernels:
        out[source] += prefactor * kernel * dual_out[target]
    return out


def interpolation_weights(math_utils) -> np.ndarray:
    """The row vector of ``np.interp(0, grid, v)`` (``QuadratureMath.interpolate`` at the spot)."""
    grid = math_utils.grid
    n = grid.size
    k = int(np.searchsorted(grid, 0.0, side="right")) - 1
    k = min(max(k, 0), n - 2)
    w = (0.0 - grid[k]) / (grid[k + 1] - grid[k])
    row = np.zeros(n, dtype=float)
    row[k] += 1.0 - w
    row[k + 1] += w
    return row


def transition_readout_weights(math_utils, prefactor: float, alpha: float, beta: float, tau: float) -> np.ndarray:
    """The row vector of ``SnowballQuadEngine._readout_at_spot``: ``R v = P s (F c v) . omega(x_m) + tail(v)(0)``,
    ``F`` the spectral filter on the padded lattice (real, even: ``F^T y = ifft(filter fft(y))``)."""
    grid = math_utils.grid
    n = grid.size
    h = float(math_utils.h)
    scale = h if math_utils.integration_rule == "trapezoid" else h / 3.0
    base_len = 2 * n - 1
    fft_len, real = math_utils.fft_length(base_len)
    fft_filter = math_utils._get_fft_filter(fft_len)
    length = base_len if fft_filter is None else fft_len
    offsets = grid[0] + np.arange(length, dtype=float) * h
    omega = np.exp(-(offsets * offsets) / (4.0 * tau) + alpha * offsets)
    if fft_filter is not None:
        omega = math_utils.apply_fft_filter(omega, fft_len, real)
    row = _simpson_coefficients(math_utils) * (prefactor * scale * omega[:n])
    sqrt_tau = math.sqrt(tau)
    u_left = (-grid[0] + 2.0 * tau * alpha) / (2.0 * sqrt_tau)
    u_right = (-grid[-1] + 2.0 * tau * alpha) / (2.0 * sqrt_tau)
    tail_scale = 0.5 * math.exp(tau * (alpha * alpha - beta))
    row[0] += tail_scale * erfc(u_left)
    row[-1] += tail_scale * erfc(-u_right)
    return row


class _BandedEvent:
    """The linear part of one event map, read off the engine's event code by coloured unit probes.

    Probe ``c`` puts a one on every node ``j = c (mod 2b + 1)`` of one state; for a map coupling each output node to
    inputs within ``b`` nodes, output node ``i`` of probe ``c`` is exactly the entry ``A[i, j]`` of the one such
    ``j`` within the band. The four state blocks (``out``/``in`` to ``out``/``in``) are kept as probe responses."""

    def __init__(self, event: Callable, n: int, time: float):
        self.n = n
        index = np.arange(n)
        colours = np.zeros((_COLOURS, n), dtype=float)
        for c in range(_COLOURS):
            colours[c, c::_COLOURS] = 1.0
        rng = np.random.default_rng(12345)
        check_out, check_in = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, n)
        zero = np.zeros((_COLOURS, n))
        v_out = np.vstack([colours, zero, check_out[None, :], np.zeros((1, n))])
        v_in = np.vstack([zero, colours, np.zeros((1, n)), check_in[None, :]])
        w_out, w_in = event(v_out, v_in, None)
        w_out, w_in = np.asarray(w_out, dtype=float), np.asarray(w_in, dtype=float)
        p = _COLOURS
        self.blocks = {
            ("out", "out"): w_out[:p], ("out", "in"): w_in[:p],
            ("in", "out"): w_out[p:2 * p], ("in", "in"): w_in[p:2 * p],
        }
        self._offsets = np.array([((c - index + _HALF_BAND) % _COLOURS) - _HALF_BAND for c in range(p)])
        # the band check: the recovered map applied to random surfaces against the event code itself
        for state, check, row in (("out", check_out, 2 * p), ("in", check_in, 2 * p + 1)):
            for target, response in (("out", w_out[row]), ("in", w_in[row])):
                recovered = self._apply(self.blocks[(state, target)], check)
                scale = max(1.0, float(np.max(np.abs(response))))
                if not np.allclose(recovered, response, rtol=_BAND_CHECK_RTOL, atol=_BAND_CHECK_RTOL * scale):
                    raise AdjointUnavailable(
                        f"undefined: the QUAD V1 event map at t = {time!r} couples nodes beyond the adjoint's probe "
                        f"band of {_HALF_BAND} nodes; no exact transpose is read off it"
                    )

    def _columns(self, c: int):
        j = np.arange(self.n) + self._offsets[c]
        valid = (j >= 0) & (j < self.n)
        return j, valid

    def _apply(self, responses: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n, dtype=float)
        for c in range(_COLOURS):
            j, valid = self._columns(c)
            out[valid] += responses[c][valid] * v[j[valid]]
        return out

    def transpose(self, state: str, target: str, dual: np.ndarray) -> np.ndarray:
        """``A^T dual`` of the block from ``state`` (input) to ``target`` (output)."""
        responses = self.blocks[(state, target)]
        out = np.zeros(self.n, dtype=float)
        for c in range(_COLOURS):
            j, valid = self._columns(c)
            out += np.bincount(j[valid], weights=responses[c][valid] * dual[valid], minlength=self.n)
        return out


def probe_event(event: Callable, n: int, time: float) -> _BandedEvent:
    """The linear map ``event(v_out, v_in, None)`` on ``n`` nodes, read off by the coloured probes and band-checked.

    Raises:
        AdjointUnavailable: the map is not local in the probe band (reason given).
    """
    return _BandedEvent(event, n, time)


def transpose_pair(banded: _BandedEvent, dual_out: np.ndarray, dual_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``A^T`` of an event map on the two note states: the duals on its inputs from the duals on its outputs."""
    return (banded.transpose("out", "out", dual_out) + banded.transpose("out", "in", dual_in),
            banded.transpose("in", "out", dual_out) + banded.transpose("in", "in", dual_in))


def _bridge_rows(rows: np.ndarray, engine, math_utils, prefactor, alpha, tau, bridge, beta) -> np.ndarray:
    """``B^T`` on each extension row of the ``out`` state (``bridge_transpose`` takes one dual)."""
    log_barrier, vol, dt, reverse = bridge
    moved = np.zeros_like(rows)
    for r in range(rows.shape[0]):
        moved[r] = bridge_transpose(rows[r], engine, math_utils, prefactor, alpha, tau, log_barrier, vol, dt, reverse,
                                    beta=beta)
    return moved


def _source_vectors(event: Callable, n: int, key) -> Tuple[np.ndarray, np.ndarray]:
    zero = np.zeros((1, n), dtype=float)
    w_out, w_in = event(zero.copy(), zero.copy(), key)
    return np.asarray(w_out, dtype=float)[0], np.asarray(w_in, dtype=float)[0]


def _first_step_forward(first: FirstStepLog, step: EventStep, log: MarchLog, n: int, k: int, first_ki, interval_ki,
                        clean_ko, sources, starts, terminal, ext=None, ki_parts=None):
    """The transpose of the exact first step: the readout's quadrature weights on the nodes, the touch of a
    continuous knock-in over the first interval, the event on the nodes (probed like any event map: pointwise, so
    diagonal), then the nodes' duals back onto the lattice (``lift_transpose``) -- or, for a state seeded at the
    first event, its terminal masses read on the nodes. The extension's rows (``ext``) ride every operator.
    Returns ``(beta_out, beta_in, done_out, done_in, x_out, x_in)``."""
    fs = first.quadrature
    n_y = fs.y.size
    d = fs.readout_dual()
    zero_y = np.zeros(n_y, dtype=float)
    beta_out, beta_in = (zero_y, d) if log.readout_in else (d, zero_y)
    alpha_out = beta_out if step.out_diffused else zero_y
    alpha_in = beta_in
    x_out = x_in = None
    if ext is not None:
        x_out, x_in = ext.readout(beta_out, beta_in)
        if not step.out_diffused:
            x_out = np.zeros_like(x_out)
    if step.bridge is not None and step.out_diffused:
        # the interval starts at the spot: a point mass there is its first-passage law
        starts[k] = (np.array([0.0]), np.array([float(np.sum(beta_out))]))
        touch = fs.P[0] if fs.P is not None else 0.0
        moved = beta_out * touch
        alpha_out = alpha_out - moved
        alpha_in = alpha_in + moved
        interval_ki[k] = float(np.sum(moved))
        if ext is not None:
            x_moved = x_out * touch
            x_out, x_in = x_out - x_moved, x_in + x_moved
    if step.event is None:
        y_out, y_in = alpha_out, alpha_in
    else:
        banded = _BandedEvent(first.event, n_y, step.time)
        for key in step.sources:
            s_out, s_in = _source_vectors(first.event, n_y, key)
            out_mass = float(alpha_out @ s_out)
            sources[key] = out_mass + float(alpha_in @ s_in)
            clean_ko[k] += out_mass / step.source_units[key]
        moved = banded.transpose("in", "out", alpha_out)
        first_ki[k] = float(np.sum(moved))
        if first.ki_parts:
            ki_parts[k] = _part_masses(first.ki_parts, n_y, step.time, alpha_out)
        y_out = banded.transpose("out", "out", alpha_out) + banded.transpose("out", "in", alpha_in)
        y_in = moved + banded.transpose("in", "in", alpha_in)
        if ext is not None:
            x_out, x_in = ext.event(first.extension, banded, n_y, step.time, alpha_out, alpha_in, x_out, x_in)
    done_out = done_in = False
    zero = np.zeros(n, dtype=float)
    if ext is not None:
        if (first.terminal_out is None) != (first.terminal_in is None):
            raise PricingError("the QUAD V1 march extension reads its terminal on both note states at once")
        if first.terminal_out is not None:
            ext.finish(y_out, y_in, x_out, x_in)
            x_out, x_in = np.zeros((x_out.shape[0], n)), np.zeros((x_in.shape[0], n))
        else:
            x_out, x_in = fs.lift_transpose(x_out), fs.lift_transpose(x_in)
    if first.terminal_out is not None:
        for name, value in first.terminal_out.items():
            terminal[name] = float(y_out @ value)
        beta_out, done_out = zero, True
    else:
        beta_out = fs.lift_transpose(y_out)
    if first.terminal_in is not None:
        for name, value in first.terminal_in.items():
            terminal[name] = float(y_in @ value)
        beta_in, done_in = zero, True
    else:
        beta_in = fs.lift_transpose(y_in)
    return beta_out, beta_in, done_out, done_in, x_out, x_in


# --- the pass -------------------------------------------------------------------------------------------------------


def forward_pass(log: MarchLog) -> ForwardResult:
    """Run the transpose of the logged march forward from the readout (module docstring).

    Raises:
        AdjointUnavailable: an event map is not local in the probe band (reason given).
    """
    mu = log.math_utils
    engine = log.engine
    n = mu.grid.size
    steps = list(reversed(log.steps))                       # forward time
    K = len(steps)
    zero = np.zeros(n, dtype=float)
    first_ki = np.zeros(K, dtype=float)
    interval_ki = np.zeros(K, dtype=float)
    clean_ko = np.zeros(K, dtype=float)
    sources: Dict[object, float] = {}
    starts: List[Optional[np.ndarray]] = [None] * K
    terminal: Dict[str, float] = {}
    ki_parts: Dict[int, np.ndarray] = {}

    first = log.first_step
    ext = None if log.extension is None else log.extension.start()
    x_out = x_in = None
    done_out = done_in = False
    if log.transition is not None:
        prefactor, alpha, beta, tau = log.transition
        row = transition_readout_weights(mu, prefactor, alpha, beta, tau)
    else:
        row = interpolation_weights(mu)
    beta_out, beta_in = (zero, row) if log.readout_in else (row, zero)
    if ext is not None and first is None:
        x_out, x_in = ext.readout(beta_out, beta_in)
    for k, step in enumerate(steps):
        if k == 0 and first is not None:
            beta_out, beta_in, done_out, done_in, x_out, x_in = _first_step_forward(
                first, step, log, n, k, first_ki, interval_ki, clean_ko, sources, starts, terminal, ext=ext,
                ki_parts=ki_parts)
            continue
        if k == 0 and log.transition is not None:
            alpha_out, alpha_in = beta_out, beta_in         # the readout row is already the dual of the slice
        else:
            tau, alpha, beta, prefactor, omega = step.diffusion
            stacked = diffuse_transpose(np.vstack([beta_out, beta_in]), mu, omega, prefactor, alpha, beta, tau,
                                        engine=engine)
            if ext is not None:
                # the extension's rows in their own call: the main dual stays the extension-free pass bit for bit
                # (a BLAS matrix-vector product's summation order can depend on the row count, e.g. on x86)
                r = x_out.shape[0]
                x_stacked = diffuse_transpose(np.vstack([x_out, x_in]), mu, omega, prefactor, alpha, beta, tau,
                                              engine=engine)
                x_beta_out = x_out
                x_out = x_stacked[:r] if step.out_diffused else np.zeros_like(x_out)
                x_in = x_stacked[r:2 * r]
            alpha_out = stacked[0] if step.out_diffused else zero
            alpha_in = stacked[1]
            if step.bridge is not None and step.out_diffused:
                log_barrier, vol, dt, reverse = step.bridge
                starts[k] = beta_out.copy()
                moved = bridge_transpose(beta_out, engine, mu, prefactor, alpha, tau, log_barrier, vol, dt, reverse,
                                         beta=beta)
                alpha_out = alpha_out - moved
                alpha_in = alpha_in + moved
                interval_ki[k] = float(np.sum(moved))
                if ext is not None:
                    x_moved = _bridge_rows(x_beta_out, engine, mu, prefactor, alpha, tau, step.bridge, beta)
                    x_out, x_in = x_out - x_moved, x_in + x_moved
        if step.event is None:
            beta_out, beta_in = alpha_out, alpha_in
        else:
            banded = _BandedEvent(step.event, n, step.time)
            for key in step.sources:
                s_out, s_in = _source_vectors(step.event, n, key)
                out_mass = float(alpha_out @ s_out)
                sources[key] = out_mass + float(alpha_in @ s_in)
                # the source carries the record's unit DF(settle)/DF(obs): the clean mass is its probability mass
                clean_ko[k] += out_mass / step.source_units[key]
            # blocks are (input state, output state): the knocked-in continuation feeds the not-knocked-in value
            # (in -> out), so its transpose moves the out dual onto the in state
            moved = banded.transpose("in", "out", alpha_out)
            first_ki[k] = float(np.sum(moved))
            if step.ki_parts:
                ki_parts[k] = _part_masses(step.ki_parts, n, step.time, alpha_out)
            beta_out = banded.transpose("out", "out", alpha_out) + banded.transpose("out", "in", alpha_in)
            beta_in = moved + banded.transpose("in", "in", alpha_in)
            if ext is not None:
                x_out, x_in = ext.event(step.extension, banded, n, step.time, alpha_out, alpha_in, x_out, x_in)
        if step.seed_out:
            if ext is not None:
                raise PricingError("the QUAD V1 march extension has no seeded note state")
            for name, value in log.terminal_out.items():
                terminal[name] = float(beta_out @ value)
            beta_out = zero
    if ext is not None and not (done_out and done_in):
        ext.finish(beta_out, beta_in, x_out, x_in)
    if not done_out and not any(step.seed_out for step in steps):
        for name, value in log.terminal_out.items():
            terminal[name] = float(beta_out @ value)
    if not done_in:
        for name, value in log.terminal_in.items():
            terminal[name] = float(beta_in @ value)
    times = np.array([float(step.time) for step in steps], dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise PricingError("the QUAD V1 march log is not in strictly increasing time")
    return ForwardResult(times=times, first_knock_in=first_ki, interval_knock_in=interval_ki,
                         clean_knock_out=clean_ko, sources=sources, interval_start_dual=starts, terminal=terminal,
                         extension=None if ext is None else ext.result(), knock_in_parts=ki_parts)

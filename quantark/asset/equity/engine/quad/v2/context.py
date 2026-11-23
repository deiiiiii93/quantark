"""Prepared, snapshot-based valuation contexts and vector spot queries."""
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import Tolerance, is_close
from .basis import ResourceLimitError, local_mesh, make_mesh
from .contract import affine, grid_function
from .gaussian import readout
from .operator import Operator


@dataclass(frozen=True)
class PreparedQuad:
    """Immutable market/contract snapshot; evaluation does not mutate caches.

    Point Greeks differentiate the same Gaussian integral used for PV.
    Desk finite-bump Greeks remain a separately named engine operation.
    """

    _function: object = field(repr=False)
    _mean: float
    _variance: float
    _discount: float
    _params: object = field(repr=False)
    spot_range: tuple[float, float]
    channels: tuple[str, ...]
    diagnostics: object
    pending_pv: float = 0.0
    _continuous: object = field(default=None, repr=False)
    #: the backward recursion's log (``adjoint.MarchLog``), kept on a statistics context for the forward pass
    _march: object = field(default=None, repr=False)

    def evaluate(self, spots):
        spots = np.asarray(spots, dtype=float).reshape(-1)
        if np.any(~np.isfinite(spots)) or np.any(spots <= 0):
            raise ValidationError("query spots must be positive and finite")
        if np.any(spots < self.spot_range[0]) or np.any(spots > self.spot_range[1]):
            raise ValidationError(
                f"query outside prepared spot range {self.spot_range}; prepare a wider context"
            )
        if self._continuous is None:
            r = readout(
                self._function,
                np.log(spots),
                self._mean,
                self._variance,
                self._discount,
                self._params,
            )
        else:
            from .continuous import continuous_readout

            hit, barrier, reverse = self._continuous
            r = continuous_readout(
                self._function,
                hit,
                np.log(spots),
                self._mean,
                self._variance,
                self._discount,
                barrier,
                reverse,
                self._params,
            )
        result = {
            "price": r[0, 0] + self.pending_pv,
            "delta": r[1, 0] / spots,
            "gamma": (r[2, 0] - r[1, 0]) / spots**2,
        }
        if np.any(~np.isfinite(r[0])) or np.any(~np.isfinite(result["price"])):
            raise NumericalError("QUAD V2 produced a nonfinite price or cashflow")
        # NaN is intentional at a contractual jump or kink; infinite risk is
        # an arithmetic failure, never a qualified result.
        if np.any(np.isinf(r[1:])) or any(
            np.any(np.isinf(result[name])) for name in ("delta", "gamma")
        ):
            raise NumericalError("QUAD V2 produced an infinite spot Greek")
        if len(self.channels) > 1:
            result["components"] = {
                name: r[0, i].copy() for i, name in enumerate(self.channels) if i
            }
            result["components"]["pending"] = np.full(len(spots), self.pending_pv)
        return result


def _first_channel(function):
    """``function`` restricted to its first (price) channel: the pieces and grids a one-channel recursion holds."""
    from .contract import Function, Grid, Piece, Source

    grids = {}

    def piece(p):
        sources = []
        for s in p.sources:
            if id(s.grid) not in grids:
                grids[id(s.grid)] = Grid(s.grid.mesh, s.grid.values[:1])
            sources.append(Source(grids[id(s.grid)], s.weight, s.shift))
        return Piece(p.lo, p.hi, p.cash[:1], p.asset[:1], tuple(sources))

    return Function(tuple(piece(p) for p in function.pieces),
                    tuple((b, piece(p)) for b, p in function.boundaries))


def price_channel(context):
    """A statistics context restricted to its price channel (spec D23): its readout runs the arithmetic of the price
    context's one-channel readout on the same values, so it reads ``price()`` bitwise."""
    continuous = context._continuous
    if continuous is not None:
        continuous = (_first_channel(continuous[0]),) + tuple(continuous[1:])
    return replace(context, _function=_first_channel(context._function), channels=context.channels[:1],
                   _continuous=continuous, _march=None)


def interval_moments(env, strike, times):
    positive = np.asarray([t for t in times if t > 0])
    means = np.zeros(len(times))
    variances = np.zeros(len(times))
    discounts = np.ones(len(times))
    if len(positive):
        grid = np.r_[0.0, positive]
        dfs = np.r_[1.0, [env.get_discount_factor(float(t)) for t in positive]]
        carry = np.r_[0.0, [env.get_div_yield(float(t)) * t for t in positive]]
        surface = env.vol_surface
        if getattr(surface, "exposes_exact_total_variance", False):
            total = np.asarray(
                surface.total_variance(strike, grid, env.spot), dtype=float
            )
        else:
            vols = np.array([env.get_vol(strike, float(t)) for t in positive])
            if np.any(vols < 0) or np.any(~np.isfinite(vols)):
                raise ValidationError("QUAD V2 requires finite nonnegative volatility")
            total = np.r_[0.0, vols**2 * positive]
        if (
            total.shape != grid.shape
            or np.any(~np.isfinite(total))
            or np.any(total < 0)
        ):
            raise ValidationError("invalid cumulative variance")
        var = np.diff(total)
        if np.any(var < 0) or np.any(~np.isfinite(var)):
            raise ValidationError(
                "QUAD V2 requires finite nonnegative forward variance"
            )
        mask = np.asarray(times) > 0
        if np.any(dfs <= 0) or np.any(~np.isfinite(dfs)) or np.any(~np.isfinite(carry)):
            raise ValidationError("invalid deterministic discount/carry inputs")
        step_dfs = dfs[1:] / dfs[:-1]
        means[mask] = -np.log(step_dfs) - np.diff(carry) - var / 2
        variances[mask] = var
        discounts[mask] = step_dfs
    return means, variances, discounts


def _shortest_interval(events, variances) -> str:
    """The event times bounding the interval of least positive variance, for a resource-limit refusal (message only:
    the contract clock already books one float per instant, so two times this close are two instants)."""
    variances = np.asarray(variances, dtype=float)
    # the mesh is sized from the intervals after the first (``make_mesh`` reads ``variances[1:]``)
    positive = np.flatnonzero(variances > 0)
    positive = positive[positive >= 1]
    if positive.size == 0:
        return ""
    i = int(positive[np.argmin(variances[positive])])
    end = float(events[i].time)
    start = float(events[i - 1].time) if i > 0 else 0.0
    return (
        f" (the shortest interval runs from event time {start!r} to {end!r}, {end - start:.3g} years; "
        "QuadV2Params.event_time_tolerance declares near-coincident event times one instant)"
    )


def _jump_breaks(event):
    """The breaks of ``event`` where some incoming state's map jumps in value: the two regions' actions continue into
    different states, or their affine cash differs at the break. A kink (same continuations, equal value at the break)
    is not a jump: its projection error is O(h^2) smaller and needs no cell edge."""
    jumps = set()
    for j, b in enumerate(event.breaks):
        for row in event.rows:
            left, right = row[j], row[j + 1]
            if sorted(left.terms) != sorted(right.terms) or sorted(left.channel_terms) != sorted(right.channel_terms):
                jumps.add(float(b))
                break
            cash_l, cash_r = np.asarray(left.cash, dtype=float), np.asarray(right.cash, dtype=float)
            asset_l, asset_r = np.asarray(left.asset, dtype=float), np.asarray(right.asset, dtype=float)
            value_l, value_r = cash_l + asset_l * np.exp(b), cash_r + asset_r * np.exp(b)
            scale = np.maximum(1.0, np.maximum(np.abs(value_l), np.abs(value_r)))
            if np.any(np.abs(value_l - value_r) > Tolerance.RELATIVE * scale):
                jumps.add(float(b))
                break
    return jumps


def _same_level(a, b):
    return is_close(a, b, rel_tol=0.0, abs_tol=Tolerance.ZERO)


#: two levels within this many standard deviations of the local intervals between them cut the same crossing layer
_LAYER_WIDTHS = 3.0


def _same_layer(a, b, variance):
    """Two log levels cut one crossing layer when they lie within ``_LAYER_WIDTHS`` standard deviations of the
    locally priced variance between their observations (CP2: 103 / 103.001 at vol 60%, 5 min, left the second
    observation's probability 1.6e-4 off; exactly equal levels are the limit)."""
    return abs(a - b) < max(Tolerance.ZERO, _LAYER_WIDTHS * float(np.sqrt(max(variance, 0.0))))


def _local_breaks(contract, variances, threshold, jumps):
    """The jump breaks a mesh must carry on cell edges when every interval of variance below ``threshold`` (after the
    first) is priced locally; refuses, with the reason, what the local treatment cannot price exactly. ``jumps``: a
    cache of each event's jump levels, shared across thresholds.

    A local step leaves its end event's jumps smeared over a width s far below a cell: the mesh carries them sharp on
    cell edges (``local_mesh``), which differs from the true function by a layer of mass ~ J s / sqrt(2 pi) on each
    side. A later operator that integrates across the break cancels the two sides (O(s^2)); one that CUTS at the same
    level -- a later observation of the same barrier, reached through local steps only -- keeps one side and loses the
    crossing mass between the two observations (S2: second observation's probability 2.68e-4 against 3.97e-4 at 5 min;
    QUAD V1 refuses the same pairs, CP1). The continuous knock-in barrier is cut by the reflection kernel, which
    vanishes at the barrier, so its layer costs O(s^2) (measured: 1.2e-6 rel at vol 60%, 5 min) and it only needs its
    cell edge.
    """
    events = contract.events
    barrier = contract.continuous_barrier
    local = [i for i in range(1, len(events)) if 0.0 < variances[i] < threshold]
    breaks = set() if barrier is None else {float(barrier)}

    def jump_levels(k):
        if k not in jumps:
            jumps[k] = {b for b in _jump_breaks(events[k]) if barrier is None or not _same_level(b, barrier)}
        return jumps[k]

    deterministic = [i for i in range(1, len(events)) if variances[i] == 0.0]
    if deterministic:
        i = deterministic[0]
        raise ResourceLimitError(
            f"the interval to event time {float(events[i].time)!r} has zero variance: its deterministic shift moves "
            "the locally priced intervals' jump breaks off the mesh's cell edges"
        )
    for i in local:
        own = jump_levels(i)
        breaks |= own
        k = i - 1
        between = float(variances[i])                 # the locally priced variance between observations k and i
        while own:
            shared = sorted(b for b in own if any(_same_layer(b, c, between) for c in jump_levels(k)))
            if shared:
                raise ResourceLimitError(
                    f"the locally priced interval ending at event time {float(events[i].time)!r} and the observation "
                    f"at {float(events[k].time)!r} observe the same barrier (log spot {shared[0]:.6g}) with only "
                    "locally priced intervals between them: the mass crossing it in between lies in a layer narrower "
                    "than a cell, which the mesh does not carry"
                )
            if k == 0 or not 0.0 < variances[k] < threshold:
                break
            between += float(variances[k])
            k -= 1
    return breaks


def _local_mesh(contract, query_logs, means, variances, params):
    """The mesh when ``make_mesh`` refuses (bitwise priority: only there): the smallest interval variances are excluded
    from the sizing in turn (equal ones together) until the mesh fits ``max_nodes``; ``(mesh, threshold)``, every
    interval after the first of variance below ``threshold`` being priced locally.

    Raises:
        ResourceLimitError: the reason the local treatment cannot price the excluded intervals, or no mesh fits.
    """
    variances = np.asarray(variances, dtype=float)
    candidates = np.unique(variances[1:][variances[1:] > 0.0])
    jumps = {}
    for threshold in candidates[1:]:
        # alignment adds at most one cell (one break) or narrows the cells (two), so a threshold whose unaligned mesh
        # exceeds max_nodes is skipped before its breaks are read
        if local_mesh(query_logs, means, variances, params, float(threshold), ()) is None:
            continue
        breaks = _local_breaks(contract, variances, float(threshold), jumps)
        mesh = local_mesh(query_logs, means, variances, params, float(threshold), breaks)
        if mesh is not None:
            return mesh, float(threshold)
    raise ResourceLimitError("no interval's variance sizes a mesh within max_nodes")


def prepare_compiled(contract, env, strike, params, spots, *, pending_pv=0.0, local_intervals=True):
    """Solve ``contract`` backward onto a mesh over ``spots``.

    ``local_intervals``: where the mesh the shortest interval needs exceeds ``max_nodes``, price the shortest
    intervals locally (``_local_mesh``) instead of refusing; ``False`` keeps the refusal (a ``spot_batch`` session's
    wide prepare, whose fallback to the scalar contexts is bitwise ``price()``).
    """
    spots = np.asarray(spots, dtype=float).reshape(-1)
    if not len(spots) or np.any(~np.isfinite(spots)) or np.any(spots <= 0):
        raise ValidationError("prepare needs positive finite spots")
    params = deepcopy(params)
    events = contract.events
    means, variances, discounts = interval_moments(
        env, strike, [e.time for e in events]
    )
    local_variance = 0.0
    try:
        mesh = make_mesh(
            np.log(spots), means, variances, params, align=contract.continuous_barrier
        )
    except ResourceLimitError as refusal:
        named = f"{refusal}{_shortest_interval(events, variances)}"
        if not local_intervals:
            raise ResourceLimitError(named) from refusal
        try:
            mesh, local_variance = _local_mesh(contract, np.log(spots), means, variances, params)
        except ResourceLimitError as why:
            raise ResourceLimitError(
                f"{named}; the intervals below the mesh's resolution cannot be priced locally: {why}"
            ) from refusal
    n_states, n_channels = len(contract.state_names), len(contract.channels)
    workspace_columns = n_states * n_channels
    if (
        mesh
        and workspace_columns * mesh.cells * mesh.order * 8 * 4
        > params.max_work_bytes
    ):
        raise NumericalError("QUAD V2 state arrays exceed workspace budget")
    op = Operator(mesh, params, local_variance) if mesh else None
    states = tuple(affine(np.zeros(n_channels)) for _ in range(n_states))
    continuous_pair = None
    for i in range(len(events) - 1, -1, -1):
        functions = events[i].apply(states)
        if i == 0:
            if contract.continuous_barrier is not None:
                pair = dict(contract.ki_pairs).get(contract.initial_state)
                if pair is not None:
                    continuous_pair = (
                        functions[pair],
                        contract.continuous_barrier,
                        contract.reverse,
                    )
            break
        if variances[i] == 0:
            states = tuple(f.shifted(means[i], discounts[i]) for f in functions)
            if contract.continuous_barrier is not None:
                from .continuous import deterministic_continuous

                states = deterministic_continuous(
                    states,
                    contract.ki_pairs,
                    contract.continuous_barrier,
                    contract.reverse,
                    means[i],
                )
        else:
            # The survival kernel below replaces the alive transition in full.
            # Avoid computing a plain transition that would be discarded.
            alive_indices = {alive for alive, _ in contract.ki_pairs} if contract.continuous_barrier is not None else set()
            rolled = [
                None if state in alive_indices else op.integrate(f, means[i], variances[i], discounts[i])
                for state, f in enumerate(functions)
            ]
            if contract.continuous_barrier is not None:
                from .continuous import continuous_grid_readout

                for alive, hit in contract.ki_pairs:
                    rolled[alive] = continuous_grid_readout(
                        functions[alive], functions[hit], means[i], variances[i],
                        discounts[i], contract.continuous_barrier, contract.reverse,
                        op, rolled_hit=rolled[hit],
                    )
            states = tuple(grid_function(mesh, values) for values in rolled)
    march = None
    if contract.metadata.get("statistics"):
        from .adjoint import MarchLog

        march = MarchLog(contract, mesh, op, means, variances, discounts, params)
    diagnostics = MappingProxyType(
        {
            "nodes": 0 if mesh is None else mesh.cells * mesh.order,
            "cells": 0 if mesh is None else mesh.cells,
            "log_domain": None if mesh is None else (mesh.lo, mesh.hi),
            "cell_width": None if mesh is None else mesh.h,
            "events": len(events),
            "states": n_states,
            "kernel_cache_bytes": 0 if op is None else op.cache.bytes,
            "backends": () if op is None else tuple(sorted(op.backends_used)),
            # sub-resolution run S4: present only where the local treatment engaged (bitwise priority)
            **({} if not local_variance else {
                "local_variance": local_variance,
                "local_intervals": int(np.sum((variances[1:] > 0) & (variances[1:] < local_variance))),
            }),
            "model": "deterministic Gaussian log increments",
            "qualification": "requires independent accuracy/refinement evidence",
            **contract.metadata,
        }
    )
    return PreparedQuad(
        functions[contract.initial_state],
        float(means[0]),
        float(variances[0]),
        float(discounts[0]),
        params,
        (float(min(spots)), float(max(spots))),
        contract.channels,
        diagnostics,
        float(pending_pv),
        continuous_pair,
        march,
    )

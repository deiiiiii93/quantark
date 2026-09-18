"""Piecewise state maps, independent of product classes and integration."""

from dataclasses import dataclass, field
import numpy as np

from .basis import Mesh


def frozen_array(value):
    result = np.array(value, dtype=float, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False)
class Grid:
    mesh: Mesh
    values: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "values", frozen_array(self.values))


@dataclass(frozen=True)
class Source:
    grid: Grid
    weight: float = 1.0
    shift: float = 0.0


@dataclass(frozen=True)
class Piece:
    lo: float
    hi: float
    cash: np.ndarray
    asset: np.ndarray
    sources: tuple[Source, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "cash", frozen_array(self.cash))
        object.__setattr__(self, "asset", frozen_array(self.asset))

    def value(self, y, derivative=0):
        y = np.asarray(y)
        shape = (len(self.cash),) + (1,) * y.ndim
        result = self.asset.reshape(shape) * np.exp(y)
        if derivative == 0:
            result = result + self.cash.reshape(shape)
        for src in self.sources:
            result = result + src.weight * src.grid.mesh.interpolate(
                src.grid.values, y + src.shift, derivative
            )
        return result


@dataclass(frozen=True)
class Function:
    pieces: tuple[Piece, ...]
    # Contract-specific equality ownership at observation thresholds.
    boundaries: tuple[tuple[float, Piece], ...] = ()

    @property
    def channels(self):
        return len(self.pieces[0].cash)

    @property
    def breaks(self):
        return tuple(
            sorted({b for p in self.pieces for b in (p.lo, p.hi) if np.isfinite(b)})
        )

    def evaluate(self, y, derivative=0):
        y = np.asarray(y, dtype=float)
        result = np.zeros((self.channels,) + y.shape)
        for piece in self.pieces:
            mask = (y >= piece.lo) & (y < piece.hi)
            if np.any(mask):
                result[:, mask] = piece.value(y[mask], derivative)
        for b, piece in self.boundaries:
            mask = y == b
            if np.any(mask):
                result[:, mask] = piece.value(y[mask], derivative)
        return result

    def shifted(self, mean, discount):
        def transform(p):
            return Piece(
                p.lo - mean,
                p.hi - mean,
                p.cash * discount,
                p.asset * (discount * np.exp(mean)),
                tuple(
                    Source(s.grid, s.weight * discount, s.shift + mean)
                    for s in p.sources
                ),
            )

        return Function(
            tuple(transform(p) for p in self.pieces),
            tuple((b - mean, transform(p)) for b, p in self.boundaries),
        )


def affine(cash, asset=None):
    cash = np.atleast_1d(cash)
    return Function(
        (Piece(-np.inf, np.inf, cash, np.zeros_like(cash) if asset is None else asset),)
    )


def grid_function(mesh, values):
    zeros = np.zeros(values.shape[0])
    return Function(
        (Piece(-np.inf, np.inf, zeros, zeros, (Source(Grid(mesh, values)),)),)
    )


@dataclass(frozen=True)
class Action:
    """Linear combination of continuation states plus affine cashflows."""

    terms: tuple[tuple[int, float], ...] = ()
    cash: tuple[float, ...] = (0.0,)
    asset: tuple[float, ...] = (0.0,)
    # Continuation terms restricted to selected channels. Deferred coupons use
    # these to share termination values while retaining observation ownership.
    channel_terms: tuple[tuple[int, float, tuple[int, ...]], ...] = ()


@dataclass(frozen=True)
class Event:
    time: float
    breaks: tuple[float, ...]
    # One action per spot region per incoming state.
    rows: tuple[tuple[Action, ...], ...]
    boundary_rows: tuple[tuple[Action, ...], ...] = ()

    def apply(self, states):
        edges = (-np.inf,) + self.breaks + (np.inf,)
        states = list(states)
        masked_states, masked_grids = {}, {}

        def select_channels(action):
            if not action.channel_terms:
                return action
            terms = list(action.terms)
            for i, weight, channels in action.channel_terms:
                key = (i, channels)
                if key not in masked_states:
                    function = states[i]
                    mask = np.zeros(function.channels)
                    mask[list(channels)] = 1.0

                    def masked_piece(piece):
                        sources = []
                        for source in piece.sources:
                            grid_key = (source.grid, channels)
                            if grid_key not in masked_grids:
                                masked_grids[grid_key] = Grid(
                                    source.grid.mesh,
                                    source.grid.values * mask[:, None, None],
                                )
                            sources.append(
                                Source(
                                    masked_grids[grid_key], source.weight, source.shift
                                )
                            )
                        return Piece(
                            piece.lo,
                            piece.hi,
                            piece.cash * mask,
                            piece.asset * mask,
                            tuple(sources),
                        )

                    masked_states[key] = len(states)
                    states.append(
                        Function(
                            tuple(masked_piece(p) for p in function.pieces),
                            tuple((b, masked_piece(p)) for b, p in function.boundaries),
                        )
                    )
                terms.append((masked_states[key], weight))
            return Action(tuple(terms), action.cash, action.asset)

        rows = tuple(tuple(select_channels(a) for a in row) for row in self.rows)
        boundary_rows = tuple(
            tuple(select_channels(a) for a in row) for row in self.boundary_rows
        )

        def combine(action, lo, hi):
            cuts = sorted(
                {lo, hi}
                | {b for i, _ in action.terms for b in states[i].breaks if lo < b < hi}
            )
            result = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                c, s = np.array(action.cash), np.array(action.asset)
                sources = []
                for i, weight in action.terms:
                    p = next(p for p in states[i].pieces if p.lo <= a and p.hi >= b)
                    c = c + weight * p.cash
                    s = s + weight * p.asset
                    sources.extend(
                        Source(q.grid, weight * q.weight, q.shift) for q in p.sources
                    )
                result.append(Piece(a, b, c, s, tuple(sources)))
            return result

        result = []
        for i, row in enumerate(rows):
            pieces = tuple(
                p
                for a, b, action in zip(edges[:-1], edges[1:], row)
                for p in combine(action, a, b)
            )
            boundaries = []
            # Carry equality ownership through deterministic intervals even
            # when the current event has no threshold at the shifted break.
            inherited = {
                b
                for action in row
                for k, _ in action.terms
                for b, _ in states[k].boundaries
            }
            for b in sorted(inherited - set(self.breaks)):
                j = int(np.searchsorted(self.breaks, b, side="right"))
                action = row[j]
                c, s, sources = np.array(action.cash), np.array(action.asset), []
                for k, weight in action.terms:
                    p = dict(states[k].boundaries).get(b)
                    if p is None:
                        p = next(p for p in states[k].pieces if p.lo <= b < p.hi)
                    c = c + weight * p.cash
                    s = s + weight * p.asset
                    sources.extend(
                        Source(q.grid, weight * q.weight, q.shift) for q in p.sources
                    )
                boundaries.append((b, Piece(b, b, c, s, tuple(sources))))
            if boundary_rows:
                for j, b in enumerate(self.breaks):
                    action = boundary_rows[i][j]
                    c, s, sources = np.array(action.cash), np.array(action.asset), []
                    for k, weight in action.terms:
                        p = dict(states[k].boundaries).get(b)
                        if p is None:
                            p = next(p for p in states[k].pieces if p.lo <= b < p.hi)
                        c = c + weight * p.cash
                        s = s + weight * p.asset
                        sources.extend(
                            Source(q.grid, weight * q.weight, q.shift)
                            for q in p.sources
                        )
                    boundaries.append((b, Piece(b, b, c, s, tuple(sources))))
            result.append(Function(pieces, tuple(boundaries)))
        return tuple(result)


@dataclass(frozen=True)
class CompiledContract:
    events: tuple[Event, ...]
    state_names: tuple[str, ...]
    initial_state: int
    channels: tuple[str, ...]
    reference_spot: float
    continuous_barrier: float | None = None
    reverse: bool = False
    ki_pairs: tuple[tuple[int, int], ...] = ()
    metadata: dict = field(default_factory=dict, compare=False)

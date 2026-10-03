"""The two-surface statistics in the value mode (spec 3B §5.1, §5.2).

Columns per surface: KO_0..KO_{n-1}, KIS, KIE, M (``ColumnLayout(n_ko, ki=want_ki, leg=True)``).
Rows: the value's own rows, allocated (cash) and the economically right ones (probabilities).
θ: the value groups' (``VALUE``). Events: the adapter's stages through a ``Breach``.

Every cash operation the value applies is linear and is applied here column for column with the
cash split by the column that pays it, so ``value = Σ_k payoff_k·KO_k + M`` node for node; the
readout reports that sum's residual against the pv as ``reconciliation_error``.
"""
from __future__ import annotations

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

from ..groups import VALUE, Backend, ColumnGroup
from .columns import ColumnLayout
from .group import RollingStatistics
from .readout import assemble_autocallable

__all__ = ["SnowballStatistics", "SnowballBreach"]


def _want(streams):
    """(want_ki, want_coupon_report) from a stream selection (spec 3B §5.1 "streams")."""
    if streams is None:
        return True, True
    from quantark.cashleg.event_distribution import EventType
    return bool(streams & {EventType.KI, EventType.MATURITY_WITH_KI}), EventType.COUPON in streams


class SnowballBreach:
    """Knock-out targets for a statistics block (spec 3B §5.1 "Events")."""

    def __init__(self, stats):
        self.stats = stats

    def unit(self, record):
        """The knock-out unit: one paid at the record's settlement, at its observation."""
        s = self.stats
        return s.solver._cashflow_value_at_time(
            pricing_env=s.env, cashflow=1.0, current_time=float(record.observation_time),
            settlement_time=record.settlement_time)

    def ko(self, record, block, surface):
        s, L = self.stats, self.stats.layout
        target = np.zeros_like(block)
        target[:, s.column_of_record(record, surface)] = self.unit(record)
        if L.kie is not None:
            target[:, L.kie] = block[:, L.kie]          # knock-in-ever is exempt from the knock-out
        return target

    def seed(self, block):
        raise ValidationError("a two-surface Snowball has no staged surface")


class SnowballStatistics(RollingStatistics):

    @classmethod
    def create(cls, prepared, streams):
        stats = cls(prepared, streams)
        return None if stats.layout.n_ko == 0 else stats

    def __init__(self, prepared, streams):
        self.prepared = prepared
        self.solver, self.product, self.env = prepared.solver, prepared.product, prepared.env
        self.tau = prepared.tau
        sp = prepared.layout.spatial
        self.x, self.s, self.t = sp.x, sp.s, prepared.layout.time.t
        if (len(self.x) - 2) <= 2:
            raise ValidationError("Event stats PDE currently requires banded solver path.")
        self.want_ki, self.want_coupon = _want(streams)
        self.ko_records = self.solver._filter_observations_by_tau(
            self.product.resolve_ko_observations(self.env), self.tau)
        self._col_by_time = {}
        self.layout = self._layout()
        for k, rec in enumerate(self.ko_records):
            self._col_by_time.setdefault(float(rec.observation_time), int(self.layout.ko[k]))
        self.n_cols = self.layout.n_cols
        self.group = ColumnGroup("stats", ("stats_v0", "stats_v1"), Backend.BANDED, VALUE, rolling=True)
        self.breach = self._breach()
        self.knocked_in_at_valuation = bool(prepared.solve_state["knocked_in"])
        self.already_knocked_in = bool(getattr(self.product, "_otc_lifecycle_knocked_in", False))

    def _layout(self):
        return ColumnLayout(len(self.ko_records), ki=self.want_ki, leg=True)

    def _breach(self):
        return SnowballBreach(self)

    def column_of_record(self, record_or_time, surface="alive"):
        t = float(getattr(record_or_time, "observation_time", record_or_time))
        col = self._col_by_time.get(t)
        if col is not None:
            return col
        for k, rec in enumerate(self.ko_records):
            if is_close(float(rec.observation_time), t):
                return int(self.layout.ko[k])
        raise ValidationError(f"no statistics column for the knock-out record at t={t}")

    # --- the march hooks ------------------------------------------------------------

    def allocate(self, states):
        n = len(self.x)
        states["stats_v0"] = np.zeros((n, self.n_cols))
        states["stats_v1"] = np.zeros((n, self.n_cols))
        if not self.prepared.request.price:
            # the v0 continuous-KI row reads the value's V1 grid, which a statistics-only march does
            # not allocate: drop a previous solve's grid so its shape cannot be indexed (the
            # statistics copy the knocked-in row themselves, ``EdgeRow.copy_from_ki``)
            self.solver._grid_v0 = self.solver._grid_v1 = None

    def terminal(self, states):
        solver, product, env, L = self.solver, self.product, self.env, self.layout
        v0, v1 = states["stats_v0"], states["stats_v1"]
        num_t = len(self.t)
        tmp = np.zeros((len(self.x), 1))
        solver._set_terminal_condition_v0(tmp, self.x, self.s, product, env)
        v0[:, L.maturity[0]] = tmp[:, -1]
        solver._set_terminal_condition_v1(tmp, self.x, self.s, product, env)
        v1[:, L.maturity[0]] = tmp[:, -1]
        if L.kis is not None:
            v1[:, L.kis] = 1.0
            v1[:, L.kie] = 1.0
        # the value adapter's terminal events, in its order (SnowballAdapter.terminal)
        is_terminal_ki = product.has_ki_barrier and (
            solver._ki_continuous or (num_t - 1) in solver._ki_observation_indices)
        terminal_ko = solver._has_terminal_ko and solver._ko_terminal_record is not None
        ko_last = terminal_ko and is_terminal_ki and solver._ko_jump_goes_last(product)
        rec = solver._ko_terminal_record
        cols, names = self._ko_targets(v0, v1)
        if terminal_ko and not ko_last:
            solver._terminal_ko_columns(cols, self.s, rec, product, env, breach=self.breach, surfaces=names)
        if is_terminal_ki:
            solver._ki_columns(v0, v1, self.s, num_t - 1, num_t, product)
        if ko_last:
            solver._terminal_ko_columns(cols, self.s, rec, product, env, breach=self.breach, surfaces=names)

    def _ko_targets(self, v0, v1):
        if self.solver._ko_survives_ki(self.product):
            return [v0, v1], ("alive", "ki")
        return [v0], ("alive",)

    def rows(self, j, v_next):
        solver, product, env = self.solver, self.product, self.env
        tau_rem = self.tau - self.t[j]
        current = solver._total_tau - tau_rem
        lo1, up1 = solver._boundary_rows_v1(j, tau_rem, product, env, self.s)
        lo0, up0 = solver._boundary_rows_v0(j, tau_rem, product, env, self.s)
        r_lo1, r_up1 = self._cash_row(lo1, "ki"), self._cash_row(up1, "ki")
        r_up0 = self._cash_row(up0, "alive")
        if lo0.copy_from_ki:
            # continuous KI: the value's alive lower row is the knocked-in row (copied, allocation and
            # all) -- or 0.0 on a family whose value never exposes that row (``_alive_lower_copies_ki``)
            r_lo0 = r_lo1.copy() if self._alive_lower_copies_ki else np.zeros(self.n_cols)
        else:
            r_lo0 = self._cash_row(lo0, "alive")
        if self.layout.kis is not None:
            self._ki_rows(current, r_lo0, r_up0, r_lo1, r_up1)
        self._probability_rows(j, current, (r_lo0, r_up0, r_lo1, r_up1), (lo0, up0, lo1, up1))
        return np.concatenate([r_lo0, r_lo1]), np.concatenate([r_up0, r_up1])

    #: SnowballAdapter.allocate exposes the knocked-in grid to the alive lower row (``solver._grid_v1``)
    _alive_lower_copies_ki = True

    def _ki_rows(self, current, r_lo0, r_up0, r_lo1, r_up1):
        """The knock-in columns' rows, in determination units ``DF(t, T)``, by edge (spec 3B §5.1).

        The knock-out edge is the upper one (the lower on a reverse product). There the knocked-in
        surface settles knocked in unless the knock-out survives the knock-in, and the
        not-knocked-in surface never knocks in; at the knock-in edge both settle knocked in (on the
        not-knocked-in surface: copied under continuous KI, a certain discrete knock-in otherwise).
        """
        solver, product, L = self.solver, self.product, self.layout
        df_det = solver._df_between_times(self.env, current, solver._total_tau)
        ko1, ki1 = (r_lo1, r_up1) if product.is_reverse else (r_up1, r_lo1)
        ko0, ki0 = (r_lo0, r_up0) if product.is_reverse else (r_up0, r_lo0)
        ki1[L.kis] = df_det
        ki1[L.kie] = df_det
        ko1[L.kis] = 0.0 if solver._ko_survives_ki(product) else df_det
        ko1[L.kie] = df_det
        reachable = df_det if product.has_ki_barrier else 0.0
        ki0[L.kis] = reachable
        ki0[L.kie] = reachable
        ko0[L.kis] = 0.0
        ko0[L.kie] = 0.0

    def _probability_rows(self, j, current, rows, edges):
        """Family hook (Phoenix: the trigger rows; KO-reset: the maturity units). ``rows`` are the
        statistics rows ``(lo0, up0, lo1, up1)``, filled in place; ``edges`` the value's EdgeRows."""

    def _cash_row(self, edge, surface):
        row = np.zeros(self.n_cols)
        for term in edge.terms:
            if term.column == "KO" and term.record_time is None:
                continue                                 # no knock-out ahead: the term's unit is 0.0
            row[self._column_of_term(term, surface)] = term.value
        return row

    def _column_of_term(self, term, surface):
        if term.column == "M":
            return int(self.layout.maturity[0])
        if term.column == "KO":
            return self.column_of_record(term.record_time, surface)
        raise ValidationError(f"row term {term.column!r} has no column in {type(self).__name__}")

    def store(self, states, out, lower, upper):
        col = 0
        for name in self.group.surfaces:
            width = states[name].shape[1]
            new = states[name].copy()
            new[1:-1, :] = out[:, col:col + width]
            new[0, :] = lower[col:col + width]
            new[-1, :] = upper[col:col + width]
            states[name] = new
            col += width

    def step_events(self, states, captures, j):
        solver, product = self.solver, self.product
        v0, v1 = states["stats_v0"], states["stats_v1"]
        if j == 0 and solver._t0_has_events(product):
            captures["stats_t0_pre"] = (v0.copy(), v1.copy())
        schedule = self.prepared.events
        if schedule is not None and j > 0:
            block = schedule.apply(j, {"alive": v0, "ki": v1}, breach=self.breach)
            block = schedule.continuous(j, block)
            states["stats_v0"], states["stats_v1"] = block["alive"], block["ki"]
        else:
            solver._two_surface_events(
                v0, v1, self.s, j, len(self.t), solver._total_tau - (self.tau - self.t[j]),
                product, self.env, breach=self.breach)

    # --- the readout ------------------------------------------------------------

    def _spot_block(self, states, captures):
        pre = captures.get("stats_t0_pre")
        sel = 1 if self.knocked_in_at_valuation else 0
        return pre[sel] if pre is not None else (states["stats_v1"] if sel else states["stats_v0"])

    def read(self, states, captures, pv):
        L, env, prepared = self.layout, self.env, self.prepared
        grid = self._spot_block(states, captures)

        def at(col):
            # the value's own readout, so the identity holds through it (a linear map of the column)
            return float(prepared.readout.price(grid[:, col], env.spot, prepared.layout))

        ed_unit = np.array([at(c) for c in L.ko])
        ki = self._ki_fields(at)
        return assemble_autocallable(self.solver, self.product, env, pv=pv, ko_records=self.ko_records,
                                     ed_unit=ed_unit, ki=ki, extra_fields={},
                                     maturity=at(L.maturity[0]))

    def _ki_fields(self, at):
        """The legacy readout's knock-in branch (the lifecycle flag's override stays until stage 2 (d))."""
        if self.already_knocked_in:
            return (1.0, 1.0, np.array([0.0]), np.array([1.0]), np.array([0.0]))
        if self.layout.kis is None:
            return (0.0, 0.0, np.array([]), np.array([]), np.array([]))
        df_T = self.env.get_discount_factor(float(self.tau))
        return (float(at(self.layout.kis) / df_T) if df_T > 0.0 else 0.0,
                float(at(self.layout.kie) / df_T) if df_T > 0.0 else 0.0,
                np.array([]), np.array([]), np.array([]))

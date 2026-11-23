"""The two-surface statistics in the value mode (spec 3B §5.1, §5.2; unified event statistics T12).

Columns per surface: KO_0..KO_{n-1}, KIS, M0, M1, U0 (``ColumnLayout(n_ko, leg=True)``): the knock-out units,
the knocked-in settlement unit, the maturity cash of the not-knocked-in and of the knocked-in outcome and the
not-knocked-in maturity unit -- one column per termination outcome (DESIGN §4.1, §4.7), none of them pruned.
Rows: the value's own rows, allocated (cash) and the economically right ones (probabilities).
θ: the value groups' (``VALUE``). Events: the adapter's stages through a ``Breach``.

Every cash operation the value applies is linear and is applied here column for column with the
cash split by the column that pays it, so ``value = Σ_k payoff_k·KO_k + M0 + M1`` node for node; the
readout reports that sum's residual against the pv as ``reconciliation_error``.

``read_statistics`` returns ``EventStatistics`` (the PDE 1D facades and the native ADI march, unified event
statistics T12-T14).
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import same_instant

from quantark.asset.equity.engine.pde.grid import project_between

from ..groups import VALUE, Backend, ColumnGroup
from . import adjoint
from . import event_statistics as es
from .columns import ColumnLayout
from .group import RollingStatistics
from .rows import knocks_in, knocks_out

__all__ = ["SnowballStatistics", "SnowballBreach"]


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
        s = self.stats
        target = np.zeros_like(block)
        target[:, s.column_of_record(record, surface)] = self.unit(record)
        return target

    def seed(self, block):
        raise ValidationError("a two-surface Snowball has no staged surface")


class SnowballStatistics(RollingStatistics):

    #: ``EventStatistics.product_family``
    family = "snowball"
    #: the forward adjoint pass of a march run by another engine family's loop (the native ADI march sets its own,
    #: ``pde/vol_model_adjoint.py``); ``None``: the PDE 1D pass (``adjoint.forward_pass``)
    forward_runner = None

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
        self.want_ki_path = es.want_knock_in_path(streams)
        self.want_coupon_probability = es.want_coupon_probability(streams)
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
        #: step -> whether the price's not-knocked-in knock-in-side row knocks in (``rows``; the forward pass)
        self._edge_knocks_in = {}

    def _layout(self):
        # the termination outcomes are never pruned (DESIGN §4.9b): KIS, M0, M1 and U0 on every request
        return ColumnLayout(len(self.ko_records), ki=True, leg=True)

    def _breach(self):
        return SnowballBreach(self)

    def column_of_record(self, record_or_time, surface="alive"):
        t = float(getattr(record_or_time, "observation_time", record_or_time))
        col = self._col_by_time.get(t)
        if col is not None:
            return col
        for k, rec in enumerate(self.ko_records):
            if same_instant(rec.observation_time, t):
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
        v0[:, L.maturity_unit[0]] = 1.0
        solver._set_terminal_condition_v1(tmp, self.x, self.s, product, env)
        v1[:, L.maturity[1]] = tmp[:, -1]
        v1[:, L.kis] = 1.0
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
        lo1, up1 = self._value_rows("v1", j, tau_rem)
        lo0, up0 = self._value_rows("v0", j, tau_rem)
        # the hstacked row pair, filled through per-surface views (surface 0 first, as the march stacks)
        w = self.n_cols
        lower, upper = np.zeros(2 * w), np.zeros(2 * w)
        r_lo0, r_lo1, r_up0, r_up1 = lower[:w], lower[w:], upper[:w], upper[w:]
        reverse = bool(product.is_reverse)
        # the price's own label of its not-knocked-in knock-in-side row (D23): the statistics settle that edge
        # knocked in exactly where the price's row knocks in, and the forward pass reads the same per-step flag
        knocked_in_edge = knocks_in(up0 if reverse else lo0)
        self._edge_knocks_in[j] = knocked_in_edge
        self._fill_cash(r_lo1, lo1, "ki")
        self._fill_cash(r_up1, up1, "ki")
        for r0, r1, edge, ki_edge in ((r_up0, r_up1, up0, reverse), (r_lo0, r_lo1, lo0, not reverse)):
            if edge.copy_from_ki:
                # continuous KI: the value's alive knock-in-side row (lower; upper on a reverse product) is the
                # knocked-in row (copied, allocation and all) -- or 0.0 on a family whose value never exposes that
                # row (``_alive_ki_edge_copies_ki``)
                if self._alive_ki_edge_copies_ki:
                    r0[:] = r1
            else:
                self._fill_cash(r0, edge, "alive", ki_edge and knocked_in_edge)
        if self.layout.kis is not None:
            self._ki_rows(current, r_lo0, r_up0, r_lo1, r_up1, (lo0, up0, lo1, up1), knocked_in_edge)
        self._probability_rows(j, current, (r_lo0, r_up0, r_lo1, r_up1), (lo0, up0, lo1, up1))
        return lower, upper

    def alive_ki_edge_knocks_in(self, j) -> bool:
        """Whether the not-knocked-in surface's knock-in-side row at step ``j`` knocks in: the price's own label of
        that row (``rows.knocks_in``), recorded when the march set the row (else read off the price's row function)."""
        flag = self._edge_knocks_in.get(j)
        if flag is None:
            lo0, up0 = self._value_rows("v0", j, self.tau - self.t[j])
            flag = self._edge_knocks_in[j] = knocks_in(up0 if self.product.is_reverse else lo0)
        return flag

    #: SnowballAdapter.allocate exposes the knocked-in grid to the alive knock-in-side row (``solver._grid_v1``)
    _alive_ki_edge_copies_ki = True

    def _value_rows(self, which, j, tau_rem):
        """The surface's EdgeRows at step ``j``: the very rows the value group set this step when it
        marches alongside (the march advances the value groups before the statistics group, so the
        stash is this step's), else the adapter's row function (decision 14)."""
        solver = self.solver
        if self.prepared.request.price:
            stash = getattr(solver, f"_rows_{which}_stash", None)
            if (stash is not None and stash[0] == j and stash[1] == tau_rem and stash[2] is self.product
                    and stash[3] is self.env):
                return stash[4], stash[5]
        return getattr(solver, f"_boundary_rows_{which}")(j, tau_rem, self.product, self.env, self.s)

    def _ki_rows(self, current, r_lo0, r_up0, r_lo1, r_up1, edges, knocked_in_edge):
        """The maturity outcomes' unit rows, in determination units ``DF(t, T)``, by edge (spec 3B §5.1), each
        labelled by the price's own row there (D23; ``edges`` = the price's ``(lo0, up0, lo1, up1)``).

        The knock-out edge is the upper one (the lower on a reverse product). There the knocked-in surface's row
        pays its knocked-in maturity unless it carries a knock-out (the price's knocked-in rows never do: it settles
        knocked in), and the not-knocked-in surface knocks out on a record ahead, else matures not knocked in. At
        the knock-in edge the knocked-in surface settles knocked in; the not-knocked-in surface settles knocked in
        exactly where the price's row knocks in (``knocked_in_edge``: copied under continuous KI beyond the level, a
        certain discrete knock-in while an observation remains) and otherwise matures not knocked in (``U0``).
        """
        solver, product, L = self.solver, self.product, self.layout
        df_det = solver._df_between_times(self.env, current, solver._total_tau)
        lo0, up0, lo1, up1 = edges
        ko1, ki1 = (r_lo1, r_up1) if product.is_reverse else (r_up1, r_lo1)
        ko0, ki0 = (r_lo0, r_up0) if product.is_reverse else (r_up0, r_lo0)
        ko_edge0, ko_edge1 = (lo0, lo1) if product.is_reverse else (up0, up1)
        ki1[L.kis] = df_det
        ko1[L.kis] = 0.0 if knocks_out(ko_edge1) else df_det
        ki0[L.kis] = df_det if knocked_in_edge else 0.0
        ko0[L.kis] = 0.0
        u0 = int(L.maturity_unit[0])
        ki1[u0] = 0.0
        ko1[u0] = 0.0
        ki0[u0] = 0.0 if knocked_in_edge else df_det
        ko0[u0] = 0.0 if knocks_out(ko_edge0) else df_det

    def _probability_rows(self, j, current, rows, edges):
        """Family hook (Phoenix: the trigger rows; KO-reset: the maturity units). ``rows`` are the
        statistics rows ``(lo0, up0, lo1, up1)``, filled in place; ``edges`` the value's EdgeRows."""

    def _fill_cash(self, row, edge, surface, ki_edge=False):
        """Write an EdgeRow's cash terms into ``row`` (zeros), each in the column that pays it."""
        for term in edge.terms:
            if term.column == "KO" and term.record_time is None:
                continue                                 # no knock-out ahead: the term's unit is 0.0
            row[self._column_of_term(term, surface, ki_edge)] = term.value

    def _maturity_column(self, surface, ki_edge):
        """The maturity cash column of an ``M`` term: the knocked-in outcome's on the knocked-in surface and on the
        not-knocked-in surface's knock-in edge where the price's row knocks in (``ki_edge``: the caller's
        ``rows.knocks_in`` of that row; the unit rows settle it knocked in, ``_ki_rows``), else the not-knocked-in
        outcome's."""
        knocked_in = surface == "ki" or ki_edge
        return int(self.layout.maturity[1 if knocked_in else 0])

    def _column_of_term(self, term, surface, ki_edge=False):
        if term.column == "M":
            return self._maturity_column(surface, ki_edge)
        if term.column == "KO":
            return self.column_of_record(term.record_time, surface)
        raise ValidationError(f"row term {term.column!r} has no column in {type(self).__name__}")

    def store(self, states, out, lower, upper):
        col = 0
        for name in self.group.surfaces:
            width = states[name].shape[1]
            new = np.empty_like(states[name])                # every row is written below
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

    def _reader(self, states, captures):
        """``at(col)``: a column of the selected surface read at the spot by the value's own readout, so the
        identity holds through it (a linear map of the column)."""
        prepared, env = self.prepared, self.env
        grid = self._spot_block(states, captures)

        def at(col):
            return float(prepared.readout.price(grid[:, col], env.spot, prepared.layout))

        return at

    # --- the EventStatistics readout (unified event statistics T12) -------------------------------

    def read_statistics(self, states, captures, pv):
        """``EventStatistics`` of the march: every outcome from its own column (DESIGN §4.7)."""
        L, env, product = self.layout, self.env, self.product
        at = self._reader(states, captures)
        observations = es.observation_table(env, self.ko_records, np.array([at(c) for c in L.ko]))
        terminal = resolve_terminal_timing(product, env)
        maturity, ki_settlement = self._maturity_rows(at, terminal)
        unavailable = {}
        knock_in, ki_table = self._knock_in(ki_settlement, unavailable, states, captures)
        lifetime = float(np.sum(observations.probability * observations.times)) + (
            maturity.probability + ki_settlement.probability) * float(terminal.determination_time)
        return es.assemble(family=self.family, pv=pv, observations=observations, coupons=None, maturity=maturity,
                           ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table,
                           expected_lifetime=lifetime, provenance=self.solver._native_provenance(),
                           unavailable=unavailable)

    def _maturity_rows(self, at, terminal):
        """The maturity (never KO, never KI) and knock-in settlement (KI, no KO) rows at the terminal timing:
        probability from the outcome's unit (``DF(determination)``), cash from its maturity column."""
        L, env = self.layout, self.env
        det, pay = float(terminal.determination_time), float(terminal.payment_time)
        df_det = float(env.get_discount_factor(det))
        timing = dict(time=det, payment_time=pay, date=terminal.determination_date,
                      payment_date=terminal.payment_date)
        maturity = es.outcome_row(env, probability=at(L.maturity_unit[0]) / df_det, discounted=at(L.maturity[0]),
                                  **timing)
        ki_settlement = es.outcome_row(env, probability=at(L.kis) / df_det, discounted=at(L.maturity[1]), **timing)
        return maturity, ki_settlement

    def _knock_in(self, ki_settlement, unavailable, states, captures):
        """``(KnockInStatistics, ki_table)``: the valuation-date rules, else the forward adjoint pass (T13), run only
        when the request asks for the knock-in stream (D17)."""
        rule = es.knock_in_rule(self.product, knocked_out_now=False, knocked_in_now=self.knocked_in_at_valuation)
        loss = es.expected_loss(self.product, self.env, probability=ki_settlement.probability,
                                discounted_cash=ki_settlement.discounted_cash, payment_time=ki_settlement.payment_time)
        return es.knock_in_fields(rule, want=self.want_ki_path, probability_no_ko=ki_settlement.probability,
                                  expected_loss=loss, unavailable=unavailable,
                                  forward=lambda: adjoint.knock_in_from_forward_pass(self, states, captures),
                                  valuation_dates=es.at_valuation_dates(self.product, self.env))

    # --- the forward adjoint pass's stage plans (unified event statistics T13) ----------------------------------
    #
    # Each plan lists, in the march's own order, the stages ``step_events`` (``terminal`` at maturity) applies at
    # time index ``j`` to a probability column, built from the very primitive the march calls with the same
    # arguments (``adjoint.py``).

    def knock_in_horizon(self) -> float:
        """The end of the knock-in monitoring window: the note's maturity."""
        return float(self.tau)

    def adjoint_stages(self, j, ops):
        if j == len(self.t) - 1:
            return self._adjoint_terminal(ops)
        schedule = self.prepared.events
        if schedule is not None and j > 0:
            return self._adjoint_schedule(j, schedule, ops)
        return self._adjoint_two_surface(j, ops)

    def _ko_surfaces(self):
        return [adjoint.ALIVE, adjoint.KI] if self.solver._ko_survives_ki(self.product) else [adjoint.ALIVE]

    def _adjoint_ko_columns(self, j, record, surfaces, ops):
        """``solver._ko_columns`` at ``j`` on ``surfaces``."""
        solver, s, rev = self.solver, self.s, bool(self.product.is_reverse)
        barrier = record.barrier
        if solver._event_uses_projection(j):
            fn = lambda vs, vb: solver._project_event_values(s, barrier, rev, True, vs, vb)  # noqa: E731
            return [ops.surface_map(sf, ("ko_columns", barrier, rev), fn) for sf in surfaces]
        mask = solver._event_nodal_mask(s, barrier, rev, True, at_valuation=(j == 0))
        return [ops.nodal_ko(sf, mask) for sf in surfaces]

    def _adjoint_terminal_ko(self, record, surfaces, ops):
        """``solver._terminal_ko_columns`` on ``surfaces``."""
        solver, s, rev = self.solver, self.s, bool(self.product.is_reverse)
        barrier = record.barrier
        if solver._use_cell_average_events():
            fn = lambda vs, vb: solver._project_event_values(s, barrier, rev, True, vs, vb)  # noqa: E731
            return [ops.surface_map(sf, ("ko_columns", barrier, rev), fn) for sf in surfaces]
        mask = solver._get_barrier_mask(s, barrier, rev, is_up_barrier=True)
        return [ops.nodal_ko(sf, mask) for sf in surfaces]

    def _adjoint_ki_columns(self, j, ops):
        """``solver._ki_columns`` at ``j``: the projected discrete knock-in, or the nodal one, then FIRST_PASSAGE on
        an interior step."""
        solver, s, rev = self.solver, self.s, bool(self.product.is_reverse)
        barrier = solver._resolve_ki_barrier_at_tidx(j)
        discrete = not solver._ki_continuous
        fp = None
        if solver._ki_fp is not None and 0 < j < len(self.t) - 1:
            fp = adjoint.first_passage(solver, j, s, barrier)
        if solver._event_uses_projection(j) and discrete:
            fn = lambda alive, ki: solver._project_event_values(s, barrier, rev, False, alive, ki)  # noqa: E731
            return ops.knock_in(("ki_columns", barrier, rev), fn, fp)
        mask = solver._event_nodal_mask(s, barrier, rev, False, at_valuation=(j == 0 and discrete))
        return ops.nodal_knock_in(mask, fp)

    def _adjoint_ki_applies(self, j) -> bool:
        solver = self.solver
        return bool(self.product.has_ki_barrier and (solver._ki_continuous or j in solver._ki_observation_indices))

    def _adjoint_two_surface(self, j, ops):
        """``solver._two_surface_events`` at ``j``."""
        solver = self.solver
        record = solver._ko_observation_indices.get(j)
        ki = self._adjoint_ki_applies(j)
        ko_last = record is not None and ki and solver._ko_jump_goes_last(self.product)
        stages = []
        if record is not None and not ko_last:
            stages += self._adjoint_ko_columns(j, record, self._ko_surfaces(), ops)
        if ki:
            stages.append(self._adjoint_ki_columns(j, ops))
        if ko_last:
            stages += self._adjoint_ko_columns(j, record, self._ko_surfaces(), ops)
        return stages

    def _adjoint_schedule(self, j, schedule, ops):
        """The event table's interior stages at ``j`` (its evidence rows are its staging order) and its continuous
        stage (``SnowballAdapter._event_schedule``)."""
        solver, rev = self.solver, bool(self.product.is_reverse)
        spatial = self.prepared.layout.spatial
        stages = []
        rows = schedule.evidence()
        for row in rows:
            if row[0] == "ko" and row[1] == j:
                _, _, _, barrier, _, breach_up, hits_ki, treatment = row
                surfaces = [adjoint.ALIVE, adjoint.KI] if hits_ki else [adjoint.ALIVE]
                if treatment == "nodal":
                    mask = solver._event_nodal_mask(spatial.s, barrier, rev, True)
                    stages += [ops.nodal_ko(sf, mask) for sf in surfaces]
                else:
                    def fn(vs, vb, _b=barrier, _up=breach_up):
                        return project_between(spatial, _b, _up, vb, vs)
                    stages += [ops.surface_map(sf, ("schedule_ko", barrier, breach_up), fn) for sf in surfaces]
            elif row[0] == "ki" and row[1] == j:
                _, _, _, barrier, breach_up, treatment = row
                if treatment == "nodal":
                    stages.append(ops.nodal_knock_in(solver._event_nodal_mask(spatial.s, barrier, rev, False)))
                else:
                    def fn(alive, ki, _b=barrier, _up=breach_up):
                        return project_between(spatial, _b, _up, ki, alive)
                    stages.append(ops.knock_in(("schedule_ki", barrier, breach_up), fn))
        for row in rows:
            if row[0] == "continuous_ki":
                _, barrier, reverse = row
                s = spatial.s
                mask = (s >= barrier) if reverse else (s <= barrier)
                stages.append(ops.nodal_knock_in(mask, adjoint.first_passage(solver, j, s, barrier)))
        return stages

    def _adjoint_terminal(self, ops):
        """``terminal``'s events: the terminal knock-out and knock-in in the value adapter's order."""
        solver, product = self.solver, self.product
        num_t = len(self.t)
        is_terminal_ki = self._adjoint_ki_applies(num_t - 1)
        terminal_ko = solver._has_terminal_ko and solver._ko_terminal_record is not None
        ko_last = terminal_ko and is_terminal_ki and solver._ko_jump_goes_last(product)
        surfaces = [adjoint.ALIVE, adjoint.KI] if solver._ko_survives_ki(product) else [adjoint.ALIVE]
        stages = []
        if terminal_ko and not ko_last:
            stages += self._adjoint_terminal_ko(solver._ko_terminal_record, surfaces, ops)
        if is_terminal_ki:
            stages.append(self._adjoint_ki_columns(num_t - 1, ops))
        if ko_last:
            stages += self._adjoint_terminal_ko(solver._ko_terminal_record, surfaces, ops)
        return stages

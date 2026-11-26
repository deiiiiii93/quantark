"""The Phoenix statistics in the value mode (spec 3B §5.3, §5.1; unified event statistics T12): INSTANT and
EXPIRY coupons, with memory.

Columns per surface (``ColumnLayout(n_ko, trigger=True, leg=True, coupon_cash=True, n_states=2, expiry=...,
undiscounted=...)`` with memory, ``n_states=1`` without -- the value's memory fan, spec 2026-10-15 C4):

* the memory-free columns ``F``: KO_k, the triggers P_k, KIS, M0, M1, U0 -- a knock-out, a knock-in and the
  maturity do not depend on the arrears owed;
* per memory state ``m`` the slice ``S(m)`` of coupon cash ``C_k^(m)``: the coupon paid at observation
  ``k`` on a path that is in state ``m`` now (EXPIRY: also the payment buckets ``D^(j,m)`` and, with memory,
  the coupon earned at ``k`` valued at ``k``, ``Ĉ_k^(m)``, whose readout over ``DF(t_k)`` is the coupon's
  cash to pay -- it is paid at the note's random end, DESIGN §4.6). State 0 owes ``A0`` (the arrears carried
  from before the pricing date, at the first observation only), state 1 one unit more
  (``PhoenixPDESolver._state_arrears``).

The value's state-``m`` surface is ``F`` + ``S(m)`` read with the knock-out payoffs: every value stage is
linear, and each stage here applies it to ``F`` and to every slice with the cash split by the column
that pays it -- the coupon fan-in (pay: state 0 plus the coupon earned, miss: the coupon owed,
``S(0) + earned·(S(1) - S(0))``), the knock-in copy, the knock-out (the release rule: the arrears whatever
the spot, the coupon on its barrier) and their joint projection on a shared date. A coupon missed at the
valuation instant is read the same way from the pre-event block (spec 2026-10-15 C3b); ``S(1)`` is the
unit carrier and no readout reads it raw.

The coupon paid with a knock-out (current coupon and released arrears) is coupon cash, never knock-out cash
(DESIGN §4.3, AC11): the knock-out unit pays the record's redemption only.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.event_stats import CashLedger, CouponTable
from quantark.asset.equity.engine.pde.grid.events import project_piecewise_event
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.util.enum import CouponPayType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import same_instant

from . import adjoint
from . import event_statistics as es
from .columns import ColumnLayout
from .snowball import SnowballBreach, SnowballStatistics

__all__ = ["PhoenixStatistics", "PhoenixBreach"]


class PhoenixBreach(SnowballBreach):
    """Snowball's knock-out target with the Phoenix value's own discount (``_apply_ko_jump_vector``'s
    ``df``: 1.0, or ``DF(t, settlement)`` when the record settles later) at the stage's time."""

    def __init__(self, stats):
        super().__init__(stats)
        self.current_time = None        # the stage's time, set by PhoenixStatistics before each knock-out
        self.termination = False        # True while _termination_ko_columns transforms the T_j/U slice

    def unit(self, record):
        s = self.stats
        t = float(record.observation_time) if self.current_time is None else self.current_time
        if record.settlement_time is not None and record.settlement_time > t:
            return s.solver._df_between_times(s.env, t, record.settlement_time)
        return 1.0

    def ko(self, record, block, surface):
        if not self.termination:
            return super().ko(record, block, surface)
        # the termination slice (T_0..T_{n-1}, U): a knock-out ends the note, one unit paid at its settlement
        target = np.zeros_like(block)
        target[:, self.stats.column_of_record(record) - int(self.stats.layout.ko[0])] = self.unit(record)
        return target


class PhoenixStatistics(SnowballStatistics):

    family = "phoenix"

    #: PhoenixAdapter.boundaries exposes the knocked-in grid to the alive knock-in-side row under continuous KI
    #: (``solver._grid_v1``), as SnowballAdapter does: the statistics copy the knocked-in row too
    _alive_ki_edge_copies_ki = True

    def __init__(self, prepared, streams):
        super().__init__(prepared, streams)
        solver, L = self.solver, self.layout
        # coupon observation index (the solver's, over every resolved record) <-> statistics column k
        self._k_of_obs, self._obs_of_k = {}, {}
        rank = {}         # an env instant's first column -> its records mapped so far (0.16.0 D14: by contract index)
        for obs_idx, rec in enumerate(solver._get_cached_ko_records(self.env, self.product)):
            for k, own in enumerate(self.ko_records):
                if same_instant(own.observation_time, rec.observation_time):
                    if self._shared_cols:
                        # the instant's j-th record is its j-th row (both lists keep contract order inside it)
                        j = rank.get(k, 0)
                        rank[k] = j + 1
                        if k + j >= len(self.ko_records) or not same_instant(
                                self.ko_records[k + j].observation_time, rec.observation_time):
                            break
                        k = k + j
                    self._k_of_obs[obs_idx], self._obs_of_k[k] = k, obs_idx
                    break
        self._S = [np.r_[L.cash[:, m], L.bucket[:, m], L.cash_u[:, m]].astype(int) for m in range(self.n_states)]
        # the undiscounted coupon carrier's offset in a slice (after C_k and the buckets)
        self._u_off = L.n_ko + L.bucket.shape[0]
        for m, cols in enumerate(self._S):     # C_k^(m) sits at position k of its slice, in every state
            assert all(int(cols[k]) == int(L.cash[k, m]) for k in range(L.n_ko))
            assert all(int(cols[self._u_off + k]) == int(L.cash_u[k, m]) for k in range(L.cash_u.shape[0]))
        # EXPIRY's termination units T_0..T_{n-1}, U: one contiguous slice, outside F and every S(m) -- they
        # take only the knock-in copy and the termination knock-out, as W does (plan F4)
        self._TU = slice(int(L.term_unit[0]), int(L.unit) + 1) if L.expiry else None
        tu = set(range(self._TU.start, self._TU.stop)) if L.expiry else set()
        in_slices = set(np.concatenate(self._S).tolist()) | tu
        self._F = np.array([c for c in range(self.n_cols) if c not in in_slices], dtype=int)
        self._pos_F = {int(c): i for i, c in enumerate(self._F)}
        self._NT = np.array([c for c in range(self.n_cols) if c not in tu], dtype=int)   # all but T/U

    def _layout(self):
        product = self.product
        self.n_states = 2 if product.has_memory_coupon else 1     # the value's memory fan (spec 2026-10-15 C4)
        expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
        return ColumnLayout(len(self.ko_records), trigger=True, ki=True, leg=True, coupon_cash=True,
                            n_states=self.n_states, expiry=expiry,
                            undiscounted=expiry and product.has_memory_coupon)

    def _breach(self):
        return PhoenixBreach(self)

    def _k(self, obs_idx):
        k = self._k_of_obs.get(obs_idx)
        if k is None:
            raise ValidationError(f"no statistics column for the coupon observation {obs_idx}")
        return k

    def _trigger_unit(self, j, obs_idx):
        """A triggered coupon's indicator value: one paid at its payment time (F8; legacy's df_delay)."""
        solver = self.solver
        return solver._cashflow_value_at_time(
            pricing_env=self.env, cashflow=1.0, current_time=float(self.t[j]),
            settlement_time=solver._coupon_payment_time(self.product, self.env, obs_idx))

    def _earned(self, obs_idx, m):
        solver = self.solver
        memory = self.product.has_memory_coupon
        return float(solver._coupon_amounts[obs_idx]) + (solver._state_arrears(obs_idx, m) if memory else 0.0)

    # --- the march hooks ------------------------------------------------------------

    def terminal(self, states):
        # PhoenixAdapter.terminal's order: the conditions, the terminal coupon, the knock-in, the knock-out
        solver, product, env, L = self.solver, self.product, self.env, self.layout
        v0, v1 = states["stats_v0"], states["stats_v1"]
        num_t = len(self.t)
        tmp0, tmp1 = np.zeros((len(self.x), 1)), np.zeros((len(self.x), 1))
        solver._set_terminal_condition_vector([tmp0], [tmp1], self.x, self.s, product, env)
        v0[:, L.maturity[0]] = tmp0[:, -1]
        v0[:, L.maturity_unit[0]] = 1.0
        v1[:, L.maturity[1]] = tmp1[:, -1]
        v1[:, L.kis] = 1.0
        terminal = num_t - 1
        group = solver._phx_groups.get(terminal) if solver._phx_groups else None
        if group is not None:
            # several records on the maturity instant (0.16.0 D14): the knock-in, then the zero-gap stage
            if product.has_ki_barrier and (solver._ki_continuous or terminal in solver._ki_observation_indices):
                solver._ki_columns(v0, v1, self.s, terminal, num_t, product)
            self._group_stage((v0, v1), terminal, self.tau, group, terminal=True)
        else:
            obs_idx = solver._coupon_observation_indices.get(terminal)
            if obs_idx is not None:
                self._coupon_stage((v0, v1), terminal, self.tau, obs_idx)
            if product.has_ki_barrier and (solver._ki_continuous or terminal in solver._ki_observation_indices):
                solver._ki_columns(v0, v1, self.s, terminal, num_t, product)
            if solver._has_terminal_ko and solver._ko_terminal_record is not None:
                self._ko_stage((v0, v1), terminal, self.tau, solver._ko_terminal_record, terminal=True)
        if L.expiry:
            # the value creates W after its terminal events: T_j = 0, U = the maturity-to-payment discount
            for X in (v0, v1):
                X[:, self._TU] = 0.0
                X[:, L.unit] = solver._term_df_maturity[-1]
        if not self.prepared.request.price:
            # PhoenixAdapter.terminal builds the FIRST_PASSAGE state after its terminal events; a
            # statistics-only march runs no value terminal, so it builds the same state here (decision 14)
            solver._prepare_continuous_ki_correction(product, env, self.t)

    def step_events(self, states, captures, j):
        # _apply_step_modifications_vector_surface's order: coupon, knock-in, knock-out
        solver, product = self.solver, self.product
        v0, v1 = states["stats_v0"], states["stats_v1"]
        if j == 0 and solver._t0_has_events(product):
            captures["stats_t0_pre"] = (v0.copy(), v1.copy())
        current = solver._total_tau - (self.tau - self.t[j])
        group = solver._phx_groups.get(j) if solver._phx_groups else None
        if group is not None:
            # several records on this env instant (0.16.0 D14): the knock-in, then the zero-gap stage
            if product.has_ki_barrier and (solver._ki_continuous or j in solver._ki_observation_indices):
                solver._ki_columns(v0, v1, self.s, j, len(self.t), product)
            self._group_stage((v0, v1), j, current, group)
            return
        obs_idx = solver._coupon_observation_indices.get(j)
        if obs_idx is not None and not solver._joint_coupon_ko_projection_active(j):
            self._coupon_stage((v0, v1), j, current, obs_idx)
        if product.has_ki_barrier and (solver._ki_continuous or j in solver._ki_observation_indices):
            solver._ki_columns(v0, v1, self.s, j, len(self.t), product)    # every column, every state
        rec = solver._ko_observation_indices.get(j)
        if rec is not None:
            self._ko_stage((v0, v1), j, current, rec)

    def _coupon_stage(self, blocks, j, current_time, obs_idx):
        """``_apply_coupon_jump_vector`` on statistics blocks, in place."""
        solver, product, env, s = self.solver, self.product, self.env, self.s
        if obs_idx < 0 or obs_idx >= solver._coupon_barriers.shape[0]:
            return
        k = self._k(obs_idx)
        barrier, rev = float(solver._coupon_barriers[obs_idx]), product.is_reverse
        disc = solver._coupon_discount_scalar(j, current_time, product, env, obs_idx)
        pay_mask = None
        if not solver._event_uses_projection(j):
            pay_mask = solver._event_nodal_mask(s, barrier, rev, True, at_valuation=(j == 0))
        S, F = self._S, self._F
        p_pos = self._pos_F[int(self.layout.trigger_cols[k])]
        df_pay = self._trigger_unit(j, obs_idx)
        at_terminal = j == len(self.t) - 1
        for X in blocks:
            X0 = X.copy()
            for m in range(self.n_states):
                earned = self._earned(obs_idx, m)
                pay = X0[:, S[0]]
                self._credit(pay, X0, k, earned, disc, at_terminal)
                X[:, S[m]] = solver._coupon_update(s, barrier, rev, j, self._missed(X0, earned), pay, pay_mask)
            free_pay = X0[:, F]
            free_pay[:, p_pos] = df_pay                              # today's trigger value
            X[:, F] = solver._coupon_update(s, barrier, rev, j, X0[:, F], free_pay, pay_mask)

    def _missed(self, X0, earned):
        """A coupon of ``earned`` missed from the block ``X0`` (before the event), shaped like ``S(0)``: owed
        with memory, ``S(0) + earned·(S(1) - S(0))`` (spec 2026-10-15 C2); lost without, ``S(0)``."""
        s0 = X0[:, self._S[0]]
        if not self.product.has_memory_coupon:
            return s0
        return s0 + earned * (X0[:, self._S[1]] - s0)

    def _credit(self, pay, X0, k, earned, disc, at_terminal):
        """Add a coupon earned now to a pay branch shaped like ``S(0)``, at the state-0 positions.

        INSTANT pays it now: ``C_k += earned·disc``. EXPIRY pays it at termination (spec 3B §5.3,
        decision 5): ``C_k += earned·W`` per earning date, and by payment time ``D^(j) += earned·T_j``,
        ``D^(T) += earned·U`` with ``W = Σ T_j + U`` -- except at maturity, where the value has no
        ``W`` yet and discounts to the terminal payment (``disc``), all of it into ``D^(T)``. The
        undiscounted carrier (memory EXPIRY) takes the coupon at its earning date: ``Ĉ_k += earned``.
        """
        n = self.layout.n_ko
        if not self.layout.expiry:
            pay[:, k] += earned * disc
        elif at_terminal:
            pay[:, k] += earned * disc
            pay[:, 2 * n] += earned * disc
        else:
            T = X0[:, self._TU.start:self._TU.stop - 1]
            U = X0[:, self._TU.stop - 1]
            pay[:, k] += earned * (T.sum(axis=1) + U)
            pay[:, n:2 * n] += earned * T
            pay[:, 2 * n] += earned * U
        if self.layout.cash_u.size:
            pay[:, self._u_off + k] += earned

    def _termination_ko(self, blocks, j, current_time, rec):
        """``_apply_termination_ko`` on the T_j/U slices (views), after the knock-out stage."""
        solver = self.solver
        cols = [blocks[0][:, self._TU]]
        if solver._ko_survives_ki(self.product):
            cols.append(blocks[1][:, self._TU])
        self.breach.termination = True
        try:
            solver._termination_ko_columns(cols, self.s, j, current_time, self.product, self.env, rec,
                                           breach=self.breach)
        finally:
            self.breach.termination = False

    def _ko_stage(self, blocks, j, current_time, rec, terminal=False):
        """``_apply_ko_jump_vector`` on statistics blocks, in place (the release rule), then under
        EXPIRY ``_apply_termination_ko`` on T_j/U -- which the value skips at maturity (no ``W`` yet)."""
        solver, product, L, s = self.solver, self.product, self.layout, self.s
        rev, memory = product.is_reverse, product.has_memory_coupon
        obs_idx = solver._coupon_observation_indices.get(j)
        self.breach.current_time = current_time
        df = self.breach.unit(rec)
        projection = solver._event_uses_projection(j)
        if projection and solver._joint_coupon_ko_projection_active(j):
            self._joint_stage(blocks, j, current_time, rec, obs_idx, df)
            if L.expiry and not terminal:
                self._termination_ko(blocks, j, current_time, rec)
            return
        pay_mask = ko_mask = None
        if not projection:
            pay_mask, ko_mask = solver._coupon_masks(s, j, obs_idx, rec, product)
            if not ko_mask.any():
                return                        # the termination knock-out writes the same empty mask
        k = cb = df_pay = None                # a record without a coupon observation: KO_k only
        if obs_idx is not None:
            k, cb = self._k(obs_idx), float(solver._coupon_barriers[obs_idx])
            df_pay = self._trigger_unit(j, obs_idx)

        def paid(cash, without=0.0):          # the cash where the coupon pays with the knock-out, else `without`
            if projection:
                return solver._ko_coupon_breach(s, cb, rev, without, cash)
            return np.where(pay_mask, cash, without)

        for X, name in zip(blocks, ("alive", "ki")):
            if name == "ki" and not solver._ko_survives_ki(product):
                continue                      # disable_ko_after_ki: the knocked-in surface runs on
            target = self.breach.ko(rec, X, name)
            if obs_idx is not None:
                target[:, L.trigger_cols[k]] = paid(df_pay)     # a coupon triggered on the knock-out date counts
                for m in range(self.n_states):
                    arrears = solver._state_arrears(obs_idx, m) if memory else 0.0
                    coupon_at_ko = float(solver._coupon_amounts[obs_idx]) + arrears
                    if coupon_at_ko > 0.0:
                        # the release rule (decision 4): the arrears whatever the spot, the coupon on its barrier
                        target[:, L.cash[k, m]] = paid(coupon_at_ko * df, arrears * df if arrears > 0.0 else 0.0)
                        if L.expiry:          # paid with the knock-out, at its settlement: bucket k
                            target[:, L.bucket[k, m]] = target[:, L.cash[k, m]]
                        if L.cash_u.size:     # the coupon paid at the knock-out, valued at the knock-out
                            target[:, L.cash_u[k, m]] = paid(coupon_at_ko, arrears if arrears > 0.0 else 0.0)
            NT = self._NT
            X[:, NT] = solver._ko_update(s, rec.barrier, rev, j, X[:, NT], target[:, NT], ko_mask)
        if L.expiry and not terminal:
            self._termination_ko(blocks, j, current_time, rec)

    def _joint_stage(self, blocks, j, current_time, rec, obs_idx, df):
        """``_apply_joint_coupon_ko_projection`` on statistics blocks, in place: one piecewise cell
        average per slice, with the value's four region branches split by column."""
        solver, product, L, s = self.solver, self.product, self.layout, self.s
        memory = product.has_memory_coupon
        k = self._k(obs_idx)
        cb, kob = float(solver._coupon_barriers[obs_idx]), float(rec.barrier)
        disc = solver._coupon_discount_scalar(j, current_time, product, self.env, obs_idx)
        x_vec = np.log(np.asarray(s, dtype=float))
        trig_up = not bool(product.is_reverse)
        breaks = sorted((np.log(cb), np.log(kob)))
        lows, his = [-np.inf] + breaks, breaks + [np.inf]
        b_c_x, b_ko_x = np.log(cb), np.log(kob)
        S, F = self._S, self._F
        p_pos = self._pos_F[int(L.trigger_cols[k])]
        df_pay = self._trigger_unit(j, obs_idx)
        amt = float(solver._coupon_amounts[obs_idx])
        u_off, undiscounted = self._u_off, bool(L.cash_u.size)
        for X, name in zip(blocks, ("alive", "ki")):
            ko_here = True if name == "alive" else solver._ko_survives_ki(product)
            X0 = X.copy()
            f0 = X0[:, F]
            f_ko = self.breach.ko(rec, X0, name)[:, F]
            f_pay = f0.copy()
            f_pay[:, p_pos] = df_pay

            def f_ko_value(m_pay, _f=f_ko):
                v = _f.copy()
                v[:, p_pos] = df_pay if m_pay else 0.0
                return v

            X[:, F] = project_piecewise_event(x_vec, breaks, solver._joint_branches(
                lows, his, b_c_x, b_ko_x, trig_up, ko_here, f_ko_value, f_pay, f0))
            for m in range(self.n_states):
                arrears = solver._state_arrears(obs_idx, m) if memory else 0.0
                coupon_at_ko = amt + arrears

                def s_ko_value(m_pay, _c=coupon_at_ko, _a=arrears, _w=len(S[m])):
                    v = np.zeros((len(s), _w))
                    v[:, k] = (_c if m_pay else _a) * df       # the release rule (decision 4)
                    if L.expiry:              # bucket k: paid with the knock-out, at its settlement
                        v[:, L.n_ko + k] = v[:, k]
                    if undiscounted:          # the coupon paid at the knock-out, valued at the knock-out
                        v[:, u_off + k] = _c if m_pay else _a
                    return v

                earned = self._earned(obs_idx, m)
                pay = X0[:, S[0]]
                self._credit(pay, X0, k, earned, disc, j == len(self.t) - 1)
                X[:, S[m]] = project_piecewise_event(x_vec, breaks, solver._joint_branches(
                    lows, his, b_c_x, b_ko_x, trig_up, ko_here, s_ko_value, pay, self._missed(X0, earned)))

    def _group_stage(self, blocks, j, current_time, group, terminal=False):
        """``PhoenixPDESolver._apply_phoenix_group`` on statistics blocks, in place (0.16.0 D14): the zero-gap chain
        per pattern, split by the column that pays each cash -- the deciding record's knock-out unit and trigger, the
        trigger of every record that paid before it, each coupon on its own record's cash column (INSTANT at its
        record's discount; EXPIRY at the termination units, or with a knock-out of the instant at its settlement,
        bucket and all), the termination units knocked out by the record that fires (not at maturity: no W yet)."""
        solver, product, L, s = self.solver, self.product, self.layout, self.s
        combine, patterns = solver._group_patterns(s, j, group, product, solver._event_uses_projection(j))
        memory, expiry = bool(product.has_memory_coupon), bool(L.expiry)
        at_terminal = j == len(self.t) - 1
        self.breach.current_time = current_time
        first = group[0][0]
        S, F, n, u_off = self._S, self._F, L.n_ko, self._u_off
        undiscounted = bool(L.cash_u.size)
        ks = [self._k(obs_idx) for obs_idx, _, _ in group]
        p_pos = [self._pos_F[int(L.trigger_cols[k])] for k in ks]
        df_pay = [self._trigger_unit(j, obs_idx) for obs_idx, _, _ in group]
        disc = [solver._coupon_discount_scalar(j, current_time, product, self.env, obs_idx) for obs_idx, _, _ in group]
        df_ko = [self.breach.unit(rec) for _, rec, _ in group]
        for X, name in zip(blocks, ("alive", "ki")):
            ko_here = name == "alive" or solver._ko_survives_ki(product)
            X0 = X.copy()
            f0 = X0[:, F]
            f_branches = []
            for pays, kos in patterns:
                coupons, knock_out, _ = solver._zero_gap_outcome(group, pays, kos, ko_here, 0.0, memory)
                if knock_out is None:
                    v = f0.copy()
                else:
                    pos = knock_out[0]
                    v = self.breach.ko(group[pos][1], X0, name)[:, F]
                    v[:, p_pos[pos]] = df_pay[pos] if pays[pos] else 0.0
                for cpos, _ in coupons:
                    v[:, p_pos[cpos]] = df_pay[cpos]
                f_branches.append(v)
            X[:, F] = combine(f_branches)
            s0 = X0[:, S[0]]
            s1 = X0[:, S[1]] if memory else None
            for m in range(self.n_states):
                arrears = solver._state_arrears(first, m) if memory else 0.0
                branches = []
                for pays, kos in patterns:
                    coupons, knock_out, after = solver._zero_gap_outcome(group, pays, kos, ko_here, arrears, memory)
                    if knock_out is None:
                        v = s0 + after * (s1 - s0) if memory else s0.copy()
                        for cpos, earned in coupons:
                            self._credit(v, X0, ks[cpos], earned, disc[cpos], at_terminal)
                    else:
                        pos, paid = knock_out
                        kk = ks[pos]
                        v = np.zeros((len(s), len(S[m])))
                        v[:, kk] = paid * df_ko[pos]          # the release rule (decision 4)
                        if expiry:                            # paid with the knock-out, at its settlement: bucket kk
                            v[:, n + kk] = v[:, kk]
                        if undiscounted:
                            v[:, u_off + kk] = paid
                        for cpos, earned in coupons:
                            kc = ks[cpos]
                            if expiry:                        # deferred to the note's end: this knock-out
                                v[:, kc] += earned * df_ko[pos]
                                v[:, n + kk] += earned * df_ko[pos]
                            else:
                                v[:, kc] += earned * disc[cpos]
                            if undiscounted:
                                v[:, u_off + kc] += earned
                    branches.append(v)
                X[:, S[m]] = combine(branches)
            if expiry and not terminal and ko_here:
                tu0 = X0[:, self._TU]
                tu_branches = []
                self.breach.termination = True
                try:
                    for pays, kos in patterns:
                        knock_out = solver._zero_gap_outcome(group, pays, kos, True, 0.0, False)[1]
                        tu_branches.append(tu0.copy() if knock_out is None
                                           else self.breach.ko(group[knock_out[0]][1], tu0, name))
                finally:
                    self.breach.termination = False
                X[:, self._TU] = combine(tu_branches)

    # --- the forward adjoint pass's stage plans (unified event statistics T13) ----------------------------------
    #
    # The probability columns sit in ``F`` (not the trigger): the coupon stage projects them against themselves,
    # the knock-in copies them, the knock-out (``_ko_update``) or the joint projection sets their target.

    def adjoint_stages(self, j, ops):
        """``step_events`` at ``j`` (``terminal`` at maturity): coupon, knock-in, knock-out."""
        solver = self.solver
        num_t = len(self.t)
        group = solver._phx_groups.get(j) if solver._phx_groups else None
        if group is not None:
            # the march's order at a same-instant group (0.16.0 D14): the knock-in, then the zero-gap stage
            stages = [self._adjoint_ki_columns(j, ops)] if self._adjoint_ki_applies(j) else []
            return stages + self._adjoint_group(j, group, ops)
        obs_idx = solver._coupon_observation_indices.get(j)
        stages = []
        if j == num_t - 1:
            if obs_idx is not None:
                stages += self._adjoint_coupon(j, obs_idx, ops)
            if self._adjoint_ki_applies(j):
                stages.append(self._adjoint_ki_columns(j, ops))
            if solver._has_terminal_ko and solver._ko_terminal_record is not None:
                stages += self._adjoint_ko_stage(j, solver._ko_terminal_record, ops)
            return stages
        if obs_idx is not None and not solver._joint_coupon_ko_projection_active(j):
            stages += self._adjoint_coupon(j, obs_idx, ops)
        if self._adjoint_ki_applies(j):
            stages.append(self._adjoint_ki_columns(j, ops))
        record = solver._ko_observation_indices.get(j)
        if record is not None:
            stages += self._adjoint_ko_stage(j, record, ops)
        return stages

    def _adjoint_coupon(self, j, obs_idx, ops):
        """``_coupon_stage`` on a probability column: its pay and miss branches are the column itself, so a nodal
        fan-in leaves it unchanged and the projection averages it across the coupon barrier's cell."""
        solver, s, rev = self.solver, self.s, bool(self.product.is_reverse)
        if obs_idx < 0 or obs_idx >= solver._coupon_barriers.shape[0]:
            return []
        if not solver._event_uses_projection(j):
            return []
        barrier = float(solver._coupon_barriers[obs_idx])
        fn = lambda v: solver._coupon_update(s, barrier, rev, j, v, v, None)  # noqa: E731
        return [ops.smoothing(sf, ("coupon", barrier, rev), fn) for sf in (adjoint.ALIVE, adjoint.KI)]

    def _adjoint_ko_stage(self, j, record, ops):
        """``_ko_stage`` on a probability column: ``_ko_update``, or the joint coupon + knock-out projection."""
        solver, product, s = self.solver, self.product, self.s
        rev = bool(product.is_reverse)
        obs_idx = solver._coupon_observation_indices.get(j)
        projection = solver._event_uses_projection(j)
        if projection and solver._joint_coupon_ko_projection_active(j):
            return self._adjoint_joint(j, record, obs_idx, ops)
        ko_mask = None
        if not projection:
            _, ko_mask = solver._coupon_masks(s, j, obs_idx, record, product)
            if not ko_mask.any():
                return []
        surfaces = [adjoint.ALIVE] + ([adjoint.KI] if solver._ko_survives_ki(product) else [])
        barrier = record.barrier
        fn = lambda vs, vb: solver._ko_update(s, barrier, rev, j, vs, vb, ko_mask)  # noqa: E731
        key = ("phoenix_ko", barrier, rev, projection, None if projection else (j == 0))
        return [ops.surface_map(sf, key, fn) for sf in surfaces]

    def _adjoint_joint(self, j, record, obs_idx, ops):
        """``_joint_stage`` on a probability column: the knock-out region takes its target, the pay and miss regions
        the column itself -- one piecewise cell average (``_joint_branches``)."""
        solver, product, s, n = self.solver, self.product, self.s, len(self.s)
        cb, kob = float(solver._coupon_barriers[obs_idx]), float(record.barrier)
        x_vec = np.log(np.asarray(s, dtype=float))
        trig_up = not bool(product.is_reverse)
        breaks = sorted((np.log(cb), np.log(kob)))
        lows, his = [-np.inf] + breaks, breaks + [np.inf]
        b_c_x, b_ko_x = np.log(cb), np.log(kob)
        stages = []
        for surface, ko_here in ((adjoint.ALIVE, True), (adjoint.KI, solver._ko_survives_ki(product))):
            def linear(_ko=ko_here):
                eye = np.eye(n)
                return project_piecewise_event(x_vec, breaks, solver._joint_branches(
                    lows, his, b_c_x, b_ko_x, trig_up, _ko, lambda m_pay: np.zeros((n, n)), eye, eye))

            def breach():
                return project_piecewise_event(x_vec, breaks, solver._joint_branches(
                    lows, his, b_c_x, b_ko_x, trig_up, True, lambda m_pay: np.ones(n), np.zeros(n), np.zeros(n)))

            stages.append(ops.joint(surface, ("joint", cb, kob, trig_up, ko_here), linear,
                                    breach if ko_here else None))
        return stages

    def _adjoint_group(self, j, group, ops):
        """``_group_stage`` on a probability column (0.16.0 D14): every coupon branch is the column itself, the
        regions (nodes) where a record of the instant knocks out take its target -- one piecewise cell average, or
        the nodal knock-out on the union of the live masks."""
        solver, product, s, n = self.solver, self.product, self.s, len(self.s)
        projection = solver._event_uses_projection(j)
        combine, patterns = solver._group_patterns(s, j, group, product, projection)
        stages = []
        for surface, ko_here in ((adjoint.ALIVE, True), (adjoint.KI, solver._ko_survives_ki(product))):
            fires = tuple(solver._zero_gap_outcome(group, pays, kos, ko_here, 0.0, False)[1] is not None
                          for pays, kos in patterns)
            if not projection:
                if any(fires):
                    stages.append(ops.nodal_ko(surface, combine([np.ones(n) if f else np.zeros(n) for f in fires])
                                               > 0.5))
                continue

            def linear(_f=fires):
                eye = np.eye(n)
                return combine([np.zeros((n, n)) if f else eye for f in _f])

            def breach(_f=fires):
                return combine([np.ones(n) if f else np.zeros(n) for f in _f])

            stages.append(ops.joint(surface, ("phoenix_group", j, fires), linear, breach if any(fires) else None))
        return stages

    def _column_of_term(self, term, surface, ki_edge=False):
        if term.column == "U":
            return int(self.layout.unit)
        if term.column == "T":
            return int(self.layout.term_unit[self.column_of_record(term.record_time) - int(self.layout.ko[0])])
        return super()._column_of_term(term, surface, ki_edge)

    def _probability_rows(self, j, current, rows, edges):
        """The not-knocked-in knock-out edge pays the next observation's coupon with its knock-out (F8);
        under EXPIRY the termination units take W's rows, allocated (``_termination_rows``)."""
        if self.layout.expiry:
            w_lower, w_upper = self.solver._termination_rows(j)
            for row, edge in ((rows[0], w_lower), (rows[2], w_lower), (rows[1], w_upper), (rows[3], w_upper)):
                for term in edge.terms:
                    row[self._column_of_term(term, None)] = term.value
        _, up0 = edges[0], edges[1]
        r_up0 = rows[1]
        for term in up0.terms:
            if term.column == "KO" and term.record_time is not None:
                k = self.column_of_record(term.record_time)
                obs_idx = self._obs_of_k.get(k)
                if obs_idx is not None:
                    r_up0[self.layout.trigger_cols[k]] = self.solver._cashflow_value_at_time(
                        pricing_env=self.env, cashflow=1.0, current_time=current,
                        settlement_time=self.solver._coupon_payment_time(self.product, self.env, obs_idx))

    # --- the readout ------------------------------------------------------------

    def _t0_coupon(self):
        """``(pays, obs_idx)`` of a coupon decided at valuation on the cell-average route (the value's
        ``_capture_t0_coupon_readout``), else ``None``."""
        solver, product = self.solver, self.product
        obs0 = solver._coupon_observation_indices.get(0)
        if obs0 is None or not solver._use_cell_average_events():
            return None
        if obs0 < 0 or obs0 >= solver._coupon_barriers.shape[0]:
            return None
        pays = bool(solver._event_nodal_mask(np.asarray([float(self.env.spot)], dtype=float),
                                             float(solver._coupon_barriers[obs0]), product.is_reverse, True,
                                             at_valuation=True)[0])
        return pays, obs0

    def _spot_columns(self, states, captures):
        """The columns read at the spot: ``(at, ed_unit, cash, trigger, buckets, cash_u)``, the coupon decided
        at valuation applied as the value applies it."""
        solver, product, env, L, prepared = self.solver, self.product, self.env, self.layout, self.prepared
        grid = self._spot_block(states, captures)

        def at(col):
            return float(prepared.readout.price(grid[:, col], env.spot, prepared.layout))

        group = solver._phx_groups.get(0) if solver._phx_groups else None
        if group is not None and solver._use_cell_average_events():
            return self._spot_columns_group(grid, at, group)
        t0 = self._t0_coupon()
        owed = None
        if t0 is not None and not t0[0] and product.has_memory_coupon:
            # a memory coupon missed at valuation owes it (F16): the value reads S(0) + earned·u from its
            # pre-event columns (C3), so the cash composes the same state from the pre-event block (C3b) --
            # S(1) itself is the unit carrier, one unit owed, never read raw
            owed = self._earned(t0[1], 0)

        def at_state(table, i):
            """The readout of row ``i`` of a per-state column table (``L.cash``, ``L.bucket``, ``L.cash_u``):
            state 0's, or the owed state's at a valuation miss."""
            if owed is None:
                return at(table[i, 0])
            s0, s1 = grid[:, table[i, 0]], grid[:, table[i, 1]]
            return float(prepared.readout.price(s0 + owed * (s1 - s0), env.spot, prepared.layout))

        n_ko = L.n_ko
        ed_unit = np.array([at(c) for c in L.ko])
        cash = np.array([at_state(L.cash, k) for k in range(n_ko)], dtype=float)
        trigger = np.array([at(c) for c in L.trigger_cols], dtype=float)
        buckets = (np.array([at_state(L.bucket, j) for j in range(n_ko + 1)], dtype=float)
                   if L.expiry else None)
        cash_u = (np.array([at_state(L.cash_u, k) for k in range(n_ko)], dtype=float)
                  if L.cash_u.size else None)
        if t0 is not None:
            pays, obs0 = t0
            k0 = self._k(obs0)
            if pays:
                earned = self._earned(obs0, 0)
                if L.expiry:
                    # the value interpolates W at the spot (_capture_t0_coupon_readout): W = Σ T_j + U
                    units = np.array([at(c) for c in range(self._TU.start, self._TU.stop)], dtype=float)
                    cash[k0] = cash[k0] + earned * float(units.sum())
                    buckets = buckets + earned * units
                else:
                    current0 = solver._total_tau - (self.tau - self.t[0])
                    disc0 = solver._coupon_discount_scalar(0, current0, product, env, obs0)
                    cash[k0] = cash[k0] + earned * disc0
                if cash_u is not None:
                    cash_u[k0] = cash_u[k0] + earned
                trigger[k0] = self._trigger_unit(0, obs0)
            else:
                cash[k0] = 0.0
                trigger[k0] = 0.0
                if cash_u is not None:
                    cash_u[k0] = 0.0
        return at, ed_unit, cash, trigger, buckets, cash_u

    def _spot_columns_group(self, grid, at, group):
        """``_spot_columns`` when several records sit on the valuation instant (0.16.0 D14): the chain decided at the
        spot as the value decides it (``_capture_t0_group_readout``) -- each paid record's coupon on its own row,
        the survivors' columns read in the state the instant leaves (``S(0) + A·(S(1) - S(0))``)."""
        solver, product, env, L, prepared = self.solver, self.product, self.env, self.layout, self.prepared
        memory = bool(product.has_memory_coupon)
        ko_here = (not self.knocked_in_at_valuation) or solver._ko_survives_ki(product)
        pays, kos = solver._spot_indicators(group, product, env.spot)
        arrears = solver._state_arrears(group[0][0], 0) if memory else 0.0
        coupons, knock_out, after = solver._zero_gap_outcome(group, pays, kos, ko_here, arrears, memory)
        if knock_out is not None:
            raise PricingError("a knock-out decided at the valuation instant reached the march")
        owed = after if (memory and after != 0.0) else None

        def at_state(table, i):
            if owed is None:
                return at(table[i, 0])
            s0, s1 = grid[:, table[i, 0]], grid[:, table[i, 1]]
            return float(prepared.readout.price(s0 + owed * (s1 - s0), env.spot, prepared.layout))

        n_ko = L.n_ko
        ed_unit = np.array([at(c) for c in L.ko])
        cash = np.array([at_state(L.cash, k) for k in range(n_ko)], dtype=float)
        trigger = np.array([at(c) for c in L.trigger_cols], dtype=float)
        buckets = (np.array([at_state(L.bucket, i) for i in range(n_ko + 1)], dtype=float) if L.expiry else None)
        cash_u = (np.array([at_state(L.cash_u, k) for k in range(n_ko)], dtype=float) if L.cash_u.size else None)
        paid = dict(coupons)
        current0 = solver._total_tau - (self.tau - self.t[0])
        units = (np.array([at(c) for c in range(self._TU.start, self._TU.stop)], dtype=float) if L.expiry else None)
        for pos, (obs_idx, _rec, _live) in enumerate(group):
            k = self._k(obs_idx)
            earned = paid.get(pos)
            if earned is None:
                cash[k] = 0.0
                trigger[k] = 0.0
                if cash_u is not None:
                    cash_u[k] = 0.0
                continue
            if L.expiry:
                cash[k] = cash[k] + earned * float(units.sum())
                buckets = buckets + earned * units
            else:
                cash[k] = cash[k] + earned * solver._coupon_discount_scalar(0, current0, product, env, obs_idx)
            if cash_u is not None:
                cash_u[k] = cash_u[k] + earned
            trigger[k] = self._trigger_unit(0, obs_idx)
        return at, ed_unit, cash, trigger, buckets, cash_u

    def _coupon_probability(self, trigger):
        solver, product, env = self.solver, self.product, self.env
        n_ko = self.layout.n_ko
        coupon_probability = np.zeros(n_ko, dtype=float)
        for k in range(n_ko):
            obs_idx = self._obs_of_k.get(k)
            if obs_idx is None:
                continue
            df0 = env.get_discount_factor(solver._coupon_payment_time(product, env, obs_idx))
            if df0 > 0.0:
                coupon_probability[k] = float(trigger[k] / df0)
        return coupon_probability

    def read_statistics(self, states, captures, pv):
        """``EventStatistics`` of the march: the knock-out redemption on the observation table, the coupon paid at
        each row (memory arrears and the coupon paid with a knock-out included) on the coupon table."""
        solver, product, env, L = self.solver, self.product, self.env, self.layout
        at, ed_unit, cash, trigger, buckets, cash_u = self._spot_columns(states, captures)
        n = L.n_ko
        is_coupon = np.array([self._obs_of_k.get(k) is not None for k in range(n)], dtype=bool)
        observations = es.observation_table(env, self.ko_records, ed_unit, is_coupon_obs=is_coupon)
        terminal = resolve_terminal_timing(product, env)
        unavailable = {}
        coupons, path_dependent_cash = self._coupon_table(observations, terminal, cash, trigger, buckets, cash_u,
                                                          unavailable)
        maturity, ki_settlement = self._maturity_rows(at, terminal)
        knock_in, ki_table = self._knock_in(ki_settlement, unavailable, states, captures)
        lifetime = float(np.sum(observations.probability * observations.times)) + (
            maturity.probability + ki_settlement.probability) * float(terminal.determination_time)
        return es.assemble(family=self.family, pv=pv, observations=observations, coupons=coupons,
                           maturity=maturity, ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table,
                           expected_lifetime=lifetime, provenance=solver._native_provenance(),
                           unavailable=unavailable, path_dependent_cash=path_dependent_cash)

    def _coupon_table(self, observations, terminal, cash, trigger, buckets, cash_u, unavailable):
        """The coupon table and, under EXPIRY, the ledger's payment-event buckets (DESIGN §4.3, §4.6).

        INSTANT: each coupon settles with its record (a coupon paid with a knock-out settles with it), so its
        cash to pay is ``discounted / DF(settlement)``. EXPIRY: the coupon is rolled up and paid when the note
        ends, so the row's payment is path-dependent (its payment time a placeholder, the terminal payment);
        its cash to pay is the coupon earned at the row -- ``amount x P(coupon)`` without memory, the
        undiscounted carrier ``Ĉ_k`` with memory -- and the ledger holds the cash at each ending event, whose
        payment time is deterministic per bucket."""
        solver, product, env, L = self.solver, self.product, self.env, self.layout
        n = L.n_ko
        probability = self._coupon_probability(trigger)
        is_coupon = observations.is_coupon_obs
        survival = np.concatenate([[1.0], observations.survival[:-1]]) if n else np.zeros(0)
        coupon_probability = probability if self.want_coupon_probability else None
        if coupon_probability is None:
            unavailable["coupons.probability"] = es.NOT_REQUESTED
        if not L.expiry:
            payment = np.array([solver._coupon_payment_time(product, env, self._obs_of_k[k]) if is_coupon[k]
                                else float(observations.payment_times[k]) for k in range(n)], dtype=float)
            payment_dates = observations.payment_dates
            coupons = CouponTable(probability=coupon_probability, survival=survival, payment_times=payment,
                                  payment_dates=payment_dates, discounted_cash=cash,
                                  cash_to_pay=es.to_pay(env, cash, payment),
                                  payment_is_path_dependent=np.zeros(n, dtype=bool))
            return coupons, None
        terminal_pay = float(terminal.payment_time)
        if cash_u is not None:
            df_row = np.array([float(env.get_discount_factor(float(t))) for t in observations.times], dtype=float)
            cash_to_pay = cash_u / df_row
        else:
            amounts = np.array([float(solver._coupon_amounts[self._obs_of_k[k]]) if is_coupon[k] else 0.0
                                for k in range(n)], dtype=float)
            cash_to_pay = amounts * probability
        coupons = CouponTable(
            probability=coupon_probability, survival=survival, payment_times=np.full(n, terminal_pay),
            payment_dates=None if terminal.payment_date is None else (terminal.payment_date,) * n,
            discounted_cash=cash, cash_to_pay=cash_to_pay, payment_is_path_dependent=np.full(n, True),
        )
        determination = np.concatenate([observations.times, [float(terminal.determination_time)]])
        payment = np.concatenate([observations.payment_times, [terminal_pay]])
        determination_dates = (None if observations.dates is None or terminal.determination_date is None
                               else observations.dates + (terminal.determination_date,))
        payment_dates = (None if observations.payment_dates is None or terminal.payment_date is None
                         else observations.payment_dates + (terminal.payment_date,))
        ledger = CashLedger(determination_times=determination, payment_times=payment,
                            determination_dates=determination_dates, payment_dates=payment_dates,
                            discounted_cash=buckets, cash_to_pay=es.to_pay(env, buckets, payment))
        return coupons, ledger

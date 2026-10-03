"""The Phoenix statistics in the value mode (spec 3B §5.3, §5.1): INSTANT coupons, with memory.

Columns per surface (``ColumnLayout(n_ko, trigger=True, ki=want_ki, leg=True, coupon_cash=True,
n_states=K + 1)``, ``K`` the coupon count with memory, else 0):

* the memory-free columns ``F``: KO_k, the triggers P_k, KIS, KIE, M -- a knock-out, a knock-in and the
  maturity do not depend on how many coupons were missed;
* per memory state ``m`` the slice ``S(m)`` of coupon cash ``C_k^(m)``: the coupon paid at observation
  ``k`` on a path that is in state ``m`` now.

The value's state-``m`` surface is ``F`` + ``S(m)`` read with the knock-out payoffs: every value stage is
linear, and each stage here applies it to ``F`` and to every slice with the cash split by the column
that pays it -- the coupon fan-in (pay: state 0 plus the coupon, miss: state ``m + 1``), the knock-in
copy, the knock-out (the 0.8.0 forfeit rule: arrears released only above the coupon barrier) and their
joint projection on a shared date. A state the value leaves out of a stage (``m`` past the fan) is
unreachable from then on, and no readout reads it.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.pde.grid.events import project_piecewise_event
from quantark.util.enum import CouponPayType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close

from .columns import ColumnLayout
from .readout import assemble_autocallable
from .snowball import SnowballBreach, SnowballStatistics

__all__ = ["PhoenixStatistics", "PhoenixBreach"]


class PhoenixBreach(SnowballBreach):
    """Snowball's knock-out target with the Phoenix value's own discount (``_apply_ko_jump_vector``'s
    ``df``: 1.0, or ``DF(t, settlement)`` when the record settles later) at the stage's time."""

    def __init__(self, stats):
        super().__init__(stats)
        self.current_time = None        # the stage's time, set by PhoenixStatistics before each knock-out

    def unit(self, record):
        s = self.stats
        t = float(record.observation_time) if self.current_time is None else self.current_time
        if record.settlement_time is not None and record.settlement_time > t:
            return s.solver._df_between_times(s.env, t, record.settlement_time)
        return 1.0


class PhoenixStatistics(SnowballStatistics):

    #: PhoenixAdapter never sets ``solver._grid_v1``: the value's alive lower row under continuous KI is 0.0
    _alive_lower_copies_ki = False

    @classmethod
    def create(cls, prepared, streams):
        if prepared.product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY:
            raise PricingError("value-mode Phoenix statistics cover INSTANT coupons; EXPIRY is spec 3B §5.3's "
                               "termination units (plan Task 14)")
        return super().create(prepared, streams)

    def __init__(self, prepared, streams):
        super().__init__(prepared, streams)
        solver, L = self.solver, self.layout
        # coupon observation index (the solver's, over every resolved record) <-> statistics column k
        self._k_of_obs, self._obs_of_k = {}, {}
        for obs_idx, rec in enumerate(solver._get_cached_ko_records(self.env, self.product)):
            for k, own in enumerate(self.ko_records):
                if is_close(float(own.observation_time), float(rec.observation_time)):
                    self._k_of_obs[obs_idx], self._obs_of_k[k] = k, obs_idx
                    break
        self._S = [np.r_[L.cash[:, m], L.bucket[:, m]].astype(int) for m in range(self.n_states)]
        for m, cols in enumerate(self._S):     # C_k^(m) sits at position k of its slice, in every state
            assert all(int(cols[k]) == int(L.cash[k, m]) for k in range(L.n_ko))
        in_slices = set(np.concatenate(self._S).tolist())
        self._F = np.array([c for c in range(self.n_cols) if c not in in_slices], dtype=int)
        self._pos_F = {int(c): i for i, c in enumerate(self._F)}

    def _layout(self):
        product = self.product
        self.n_states = (len(self.solver._coupon_barriers) if product.has_memory_coupon else 0) + 1
        return ColumnLayout(len(self.ko_records), trigger=True, ki=self.want_ki, leg=True, coupon_cash=True,
                            n_states=self.n_states)

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
        return float(solver._coupon_amounts[obs_idx]) + (solver._accumulated_coupon_amount(obs_idx, m)
                                                          if memory else 0.0)

    # --- the march hooks ------------------------------------------------------------

    def terminal(self, states):
        # PhoenixAdapter.terminal's order: the conditions, the terminal coupon, the knock-in, the knock-out
        solver, product, env, L = self.solver, self.product, self.env, self.layout
        v0, v1 = states["stats_v0"], states["stats_v1"]
        num_t = len(self.t)
        tmp0, tmp1 = np.zeros((len(self.x), 1)), np.zeros((len(self.x), 1))
        solver._set_terminal_condition_vector([tmp0], [tmp1], self.x, self.s, product, env)
        v0[:, L.maturity[0]] = tmp0[:, -1]
        v1[:, L.maturity[0]] = tmp1[:, -1]
        if L.kis is not None:
            v1[:, L.kis] = 1.0
            v1[:, L.kie] = 1.0
        terminal = num_t - 1
        obs_idx = solver._coupon_observation_indices.get(terminal)
        if obs_idx is not None:
            self._coupon_stage((v0, v1), terminal, self.tau, obs_idx)
        if product.has_ki_barrier and (solver._ki_continuous or terminal in solver._ki_observation_indices):
            solver._ki_columns(v0, v1, self.s, terminal, num_t, product)
        if solver._has_terminal_ko and solver._ko_terminal_record is not None:
            self._ko_stage((v0, v1), terminal, self.tau, solver._ko_terminal_record)
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
        memory = product.has_memory_coupon
        disc = solver._coupon_discount_scalar(j, current_time, product, env, obs_idx)
        pay_mask = None
        if not solver._event_uses_projection(j):
            pay_mask = solver._event_nodal_mask(s, barrier, rev, True, at_valuation=(j == 0))
        max_m = obs_idx if memory else 0
        S, F = self._S, self._F
        p_pos = self._pos_F[int(self.layout.trigger_cols[k])]
        df_pay = self._trigger_unit(j, obs_idx)
        for X in blocks:
            X0 = X.copy()
            for m in range(max_m + 1):
                nxt = m + 1 if memory else 0
                pay = X0[:, S[0]]
                pay[:, k] += self._earned(obs_idx, m) * disc        # C_k^(m): the coupon paid now
                X[:, S[m]] = solver._coupon_update(s, barrier, rev, j, X0[:, S[nxt]], pay, pay_mask)
            free_pay = X0[:, F]
            free_pay[:, p_pos] = df_pay                              # today's trigger value
            X[:, F] = solver._coupon_update(s, barrier, rev, j, X0[:, F], free_pay, pay_mask)

    def _ko_stage(self, blocks, j, current_time, rec):
        """``_apply_ko_jump_vector`` on statistics blocks, in place (the 0.8.0 forfeit rule)."""
        solver, product, L, s = self.solver, self.product, self.layout, self.s
        rev, memory = product.is_reverse, product.has_memory_coupon
        obs_idx = solver._coupon_observation_indices.get(j)
        max_m = obs_idx if (obs_idx is not None and memory) else 0
        self.breach.current_time = current_time
        df = self.breach.unit(rec)
        projection = solver._event_uses_projection(j)
        if projection and solver._joint_coupon_ko_projection_active(j):
            self._joint_stage(blocks, j, current_time, rec, obs_idx, df, max_m)
            return
        pay_mask = ko_mask = None
        if not projection:
            pay_mask, ko_mask = solver._coupon_masks(s, j, obs_idx, rec, product)
            if not ko_mask.any():
                return
        k = cb = df_pay = None                # a record without a coupon observation: KO_k and KIE only
        if obs_idx is not None:
            k, cb = self._k(obs_idx), float(solver._coupon_barriers[obs_idx])
            df_pay = self._trigger_unit(j, obs_idx)

        def paid(cash):                       # the cash where the coupon pays with the knock-out, else 0
            if projection:
                return solver._ko_coupon_breach(s, cb, rev, 0.0, cash)
            return np.where(pay_mask, cash, 0.0)

        for X, name in zip(blocks, ("alive", "ki")):
            if name == "ki" and not solver._ko_survives_ki(product):
                continue                      # disable_ko_after_ki: the knocked-in surface runs on
            target = self.breach.ko(rec, X, name)
            if obs_idx is not None:
                target[:, L.trigger_cols[k]] = paid(df_pay)     # a coupon triggered on the knock-out date counts
                for m in range(self.n_states):
                    coupon_at_ko = (float(solver._coupon_amounts[obs_idx])
                                    + (solver._accumulated_coupon_amount(obs_idx, min(m, max_m)) if memory else 0.0))
                    if coupon_at_ko > 0.0:
                        target[:, L.cash[k, m]] = paid(coupon_at_ko * df)
            X[...] = solver._ko_update(s, rec.barrier, rev, j, X, target, ko_mask)

    def _joint_stage(self, blocks, j, current_time, rec, obs_idx, df, max_m):
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
                lows, his, b_c_x, b_ko_x, trig_up, ko_here, True, f_ko_value, f_pay, f0, f0))
            for m in range(self.n_states):
                in_fan = m <= max_m
                coupon_at_ko = amt + (solver._accumulated_coupon_amount(obs_idx, min(m, max_m)) if memory else 0.0)

                def s_ko_value(m_pay, _c=coupon_at_ko, _w=len(S[m])):
                    v = np.zeros((len(s), _w))
                    v[:, k] = (_c if m_pay else 0.0) * df
                    return v

                pay = miss = None
                if in_fan:
                    pay = X0[:, S[0]]
                    pay[:, k] += self._earned(obs_idx, m) * disc
                    miss = X0[:, S[m + 1 if memory else 0]]
                X[:, S[m]] = project_piecewise_event(x_vec, breaks, solver._joint_branches(
                    lows, his, b_c_x, b_ko_x, trig_up, ko_here, in_fan, s_ko_value, pay, miss, X0[:, S[m]]))

    def _probability_rows(self, j, current, rows, edges):
        """The not-knocked-in knock-out edge pays the next observation's coupon with its knock-out (F8)."""
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

    def read(self, states, captures, pv):
        solver, product, env, L, prepared = self.solver, self.product, self.env, self.layout, self.prepared
        grid = self._spot_block(states, captures)

        def at(col):
            return float(prepared.readout.price(grid[:, col], env.spot, prepared.layout))

        t0 = self._t0_coupon()
        # a memory coupon missed at valuation reads the one-missed state, as the value does (F16)
        s_state = 1 if (t0 is not None and not t0[0] and product.has_memory_coupon) else 0
        n_ko = L.n_ko
        ed_unit = np.array([at(c) for c in L.ko])
        cash = np.array([at(L.cash[k, s_state]) for k in range(n_ko)], dtype=float)
        trigger = np.array([at(c) for c in L.trigger_cols], dtype=float)
        if t0 is not None:
            pays, obs0 = t0
            k0 = self._k(obs0)
            if pays:
                current0 = solver._total_tau - (self.tau - self.t[0])
                disc0 = solver._coupon_discount_scalar(0, current0, product, env, obs0)
                cash[k0] = cash[k0] + self._earned(obs0, 0) * disc0
                trigger[k0] = self._trigger_unit(0, obs0)
            else:
                cash[k0] = 0.0
                trigger[k0] = 0.0
        coupon_probability = np.zeros(n_ko, dtype=float)
        for k in range(n_ko):
            obs_idx = self._obs_of_k.get(k)
            if obs_idx is None:
                continue
            df0 = env.get_discount_factor(solver._coupon_payment_time(product, env, obs_idx))
            if df0 > 0.0:
                coupon_probability[k] = float(trigger[k] / df0)
        recs = self.ko_records
        coupon_rows = (
            [float(r.observation_time) for r in recs],
            [float(r.settlement_time if r.settlement_time is not None else r.observation_time) for r in recs],
            cash,
            [r.observation_date for r in recs],
            [r.settlement_date for r in recs],
        )
        if self.want_coupon:
            extra = {"coupon_probability": coupon_probability, "expected_discounted_coupon_cashflow": cash}
        else:       # the report is pruned; the ledger keeps the coupon rows (decision 3)
            extra = {"coupon_probability": np.array([]), "expected_discounted_coupon_cashflow": np.array([])}
        return assemble_autocallable(solver, product, env, pv=pv, ko_records=recs, ed_unit=ed_unit,
                                     ki=self._ki_fields(at), extra_fields=extra,
                                     maturity=at(L.maturity[0]), coupon_rows=coupon_rows)

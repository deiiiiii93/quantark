"""The KO-reset statistics in the value mode (spec 3B §5.4; decision 6).

Columns per surface (``ColumnLayout(n_pre, n_post, ki=want_ki, leg=True, two_maturities=True)``): the
pre-schedule knock-outs ``KO_pre`` (the not-knocked-in surface fires them), the post-schedule ones
``KO_post`` (the knocked-in surface), KIS, KIE, the two maturity legs ``M_pre``, ``M_post`` and the two
maturity units ``U_pre``, ``U_post`` (one paid at that maturity's determination). The not-knocked-in
surface is staged exactly as the value's: zero above the pre-schedule maturity, seeded there.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.event_stats import payment_aware_cashflow_fields
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, is_close

from .columns import ColumnLayout
from .snowball import SnowballBreach, SnowballStatistics

__all__ = ["KOResetStatistics", "KOResetBreach"]


class KOResetBreach(SnowballBreach):
    """Snowball's knock-out target -- the record's column found by surface (``"alive"``: the pre
    schedule, ``"ki"``: the post schedule) -- and the staged seed of the not-knocked-in block."""

    def seed(self, block):
        # the value's _v0_seed_values: _set_terminal_condition_v0 at the pre-schedule maturity,
        # recomputed so the statistics-only path needs no value terminal (decision 14)
        s, L = self.stats, self.stats.layout
        tmp = np.zeros((block.shape[0], 1))
        s.solver._set_terminal_condition_v0(tmp, s.x, s.s, s.product, s.env)
        out = np.zeros_like(block)
        out[:, L.maturity[0]] = tmp[:, -1]
        out[:, L.maturity_unit[0]] = 1.0
        return out


class KOResetStatistics(SnowballStatistics):

    @classmethod
    def create(cls, prepared, streams):
        stats = cls(prepared, streams)
        return None if stats.layout.n_ko + stats.layout.n_post == 0 else stats

    def _layout(self):
        # the value's own records (their payoffs and settlements), each schedule cut to [0, tau]
        solver, env, product = self.solver, self.env, self.product
        self.ko_records = solver._filter_observations_by_tau(solver._get_cached_pre_ko_records(env, product), self.tau)
        self.post_records = solver._filter_observations_by_tau(
            solver._get_cached_post_ko_records(env, product), self.tau)
        return ColumnLayout(len(self.ko_records), len(self.post_records), ki=self.want_ki, leg=True,
                            two_maturities=True)

    def _breach(self):
        return KOResetBreach(self)

    def column_of_record(self, record_or_time, surface="alive"):
        t = float(getattr(record_or_time, "observation_time", record_or_time))
        records, cols = ((self.ko_records, self.layout.ko) if surface == "alive"
                         else (self.post_records, self.layout.post_ko))
        for k, rec in enumerate(records):
            if is_close(float(rec.observation_time), t):
                return int(cols[k])
        raise ValidationError(f"no {surface} statistics column for the knock-out record at t={t}")

    def _column_of_term(self, term, surface):
        if term.column == "M":
            return int(self.layout.maturity[0 if surface == "alive" else 1])
        return super()._column_of_term(term, surface)

    # --- the march hooks ------------------------------------------------------------

    def terminal(self, states):
        # KOResetAdapter.terminal: V1's terminal, V0 staged or seeded, then the terminal events
        solver, product, env, L = self.solver, self.product, self.env, self.layout
        v0, v1 = states["stats_v0"], states["stats_v1"]
        num_t = len(self.t)
        tmp = np.zeros((len(self.x), 1))
        solver._set_terminal_condition_v1(tmp, self.x, self.s, product, env)
        v1[:, L.maturity[1]] = tmp[:, -1]
        v1[:, L.maturity_unit[1]] = 1.0
        if L.kis is not None:
            v1[:, L.kis] = 1.0
            v1[:, L.kie] = 1.0
        pre_maturity = float(product.get_pre_maturity_time(env))
        staged = bool(solver._pre_ko_observation_indices and pre_maturity < self.tau
                      and not is_close(pre_maturity, self.tau, abs_tol=Tolerance.PRECISION))
        self.seed_idx = max(solver._pre_ko_observation_indices) if staged else None
        if not self.prepared.request.price:
            # the value terminal sets the seed index the stages read; a statistics-only march has none
            solver._v0_seed_idx = self.seed_idx
        if not staged:
            solver._set_terminal_condition_v0(tmp, self.x, self.s, product, env)
            v0[:, L.maturity[0]] = tmp[:, -1]
            v0[:, L.maturity_unit[0]] = 1.0

        pre_terminal_ko = solver._has_pre_terminal_ko and solver._pre_ko_terminal_record is not None
        pre_ko_last = pre_terminal_ko and solver._pre_ko_goes_last()
        if pre_terminal_ko and not pre_ko_last:
            solver._terminal_ko_columns([v0], self.s, solver._pre_ko_terminal_record, product, env,
                                        breach=self.breach, surfaces=("alive",))
        if (solver._has_post_terminal_ko and solver._post_ko_terminal_record is not None
                and not product.barrier_config.disable_ko_after_ki):
            solver._terminal_ko_columns([v1], self.s, solver._post_ko_terminal_record, product, env,
                                        breach=self.breach, surfaces=("ki",))
        if product.has_ki_barrier and not staged and (
                solver._ki_continuous or (num_t - 1) in solver._ki_observation_indices):
            solver._ki_columns(v0, v1, self.s, num_t - 1, num_t, product)
        if pre_ko_last:
            solver._terminal_ko_columns([v0], self.s, solver._pre_ko_terminal_record, product, env,
                                        breach=self.breach, surfaces=("alive",))

    def step_events(self, states, captures, j):
        # no event table: every step takes the inline stages, the seed included; no t=0 capture
        # (the KO-reset reads the post-event column 0, KOResetAdapter._t0_readout)
        solver = self.solver
        solver._two_surface_events(states["stats_v0"], states["stats_v1"], self.s, j, len(self.t),
                                   solver._total_tau - (self.tau - self.t[j]), self.product, self.env,
                                   breach=self.breach)

    def _probability_rows(self, j, current, rows, edges):
        """The maturity units' rows, in determination units: the knocked-in surface matures on the post
        schedule unless its knock-out edge knocks it out; the not-knocked-in surface knocks in at its
        knock-in edge (a post-schedule maturity then) and knocks out at the other."""
        solver, product, L = self.solver, self.product, self.layout
        r_lo0, r_up0, r_lo1, r_up1 = rows
        det_post = solver._df_between_times(self.env, current, solver._total_tau)
        ko1, ki1 = (r_lo1, r_up1) if product.is_reverse else (r_up1, r_lo1)
        ko0, ki0 = (r_lo0, r_up0) if product.is_reverse else (r_up0, r_lo0)
        u_pre, u_post = int(L.maturity_unit[0]), int(L.maturity_unit[1])
        ki1[u_post] = det_post
        ko1[u_post] = 0.0 if solver._ko_survives_ki(product) else det_post
        ki0[u_pre], ki0[u_post] = 0.0, (det_post if product.has_ki_barrier else 0.0)
        ko0[u_pre], ko0[u_post] = 0.0, 0.0
        for r in (r_lo1, r_up1):
            r[u_pre] = 0.0

    # --- the readout ------------------------------------------------------------

    def read(self, states, captures, pv):
        solver, product, env, L, prepared = self.solver, self.product, self.env, self.layout, self.prepared
        grid = states["stats_v1"] if self.knocked_in_at_valuation else states["stats_v0"]

        def at(col):
            return float(prepared.readout.price(grid[:, col], env.spot, prepared.layout))

        def schedule(records, cols):
            times = np.array([float(r.observation_time) for r in records], dtype=float)
            pay = np.array([float(r.settlement_time if r.settlement_time is not None else r.observation_time)
                            for r in records], dtype=float)
            unit = np.array([at(c) for c in cols], dtype=float)
            payoff = np.array([float(r.payoff) if r.payoff is not None else 0.0 for r in records], dtype=float)
            df0 = np.array([env.get_discount_factor(t) for t in pay], dtype=float)
            prob = np.where(df0 > 0.0, unit / np.where(df0 > 0.0, df0, 1.0), 0.0)
            return times, pay, prob, unit * payoff

        pre_t, pre_pay, pre_prob, pre_cash = schedule(self.ko_records, L.ko)
        post_t, post_pay, post_prob, post_cash = schedule(self.post_records, L.post_ko)
        m_pre, m_post = at(L.maturity[0]), at(L.maturity[1])
        terminal = resolve_terminal_timing(product, env)
        delay = float(terminal.payment_time) - float(terminal.determination_time)
        pre_maturity, post_maturity = float(product.get_pre_maturity_time(env)), float(self.tau)
        df_pre, df_post = env.get_discount_factor(pre_maturity), env.get_discount_factor(post_maturity)
        u_pre_prob = at(L.maturity_unit[0]) / df_pre if df_pre > 0.0 else 0.0
        u_post_prob = at(L.maturity_unit[1]) / df_post if df_post > 0.0 else 0.0

        # the pre maturity's dates: its last record's; its payment date only when no delay separates them
        last_pre = self.ko_records[-1] if self.ko_records else None
        pre_det_date = (last_pre.observation_date
                        if last_pre is not None and is_close(float(last_pre.observation_time), pre_maturity)
                        else None)
        pre_pay_date = pre_det_date if is_close(delay, 0.0) else None
        ledger = payment_aware_cashflow_fields(
            env,
            determination_times=[*pre_t, *post_t, pre_maturity, post_maturity],
            payment_times=[*pre_pay, *post_pay, pre_maturity + delay, post_maturity + delay],
            expected_discounted_cashflows=[*pre_cash, *post_cash, m_pre, m_post],
            determination_dates=[*(r.observation_date for r in self.ko_records),
                                 *(r.observation_date for r in self.post_records),
                                 pre_det_date, terminal.determination_date],
            payment_dates=[*(r.settlement_date for r in self.ko_records),
                           *(r.settlement_date for r in self.post_records),
                           pre_pay_date, terminal.payment_date],
        )
        ki_probability, ki_ever, ki_times, ki_event, ki_survival = self._ki_fields(at)
        maturity = m_pre + m_post
        return solver._make_event_stats(
            pv=float(pv),
            ko_times=pre_t,
            ko_probability=pre_prob,
            survival_probability=np.maximum(0.0, 1.0 - np.cumsum(pre_prob)),
            expected_discounted_ko_cashflow=pre_cash,
            ki_probability=ki_probability,
            expected_discounted_maturity_cashflow=maturity,
            reconciliation_error=float(pv - float(np.sum(pre_cash)) - float(np.sum(post_cash)) - m_pre - m_post),
            ki_times=ki_times,
            ki_event_probability=ki_event,
            ki_survival_probability=ki_survival,
            ki_ever_probability=ki_ever,
            ki_survive_knocked_in_probability=ki_probability,
            pre_ko_times=pre_t,
            pre_ko_probability=pre_prob,
            post_ko_times=post_t,
            post_ko_probability=post_prob,
            pre_ko_probability_total=float(np.sum(pre_prob)),
            post_ko_probability_total=float(np.sum(post_prob)),
            expected_discounted_post_ko_cashflow=float(np.sum(post_cash)),
            pre_ko_payment_times=pre_pay,
            post_ko_payment_times=post_pay,
            expected_discounted_pre_maturity_cashflow=m_pre,
            expected_discounted_post_maturity_cashflow=m_post,
            pre_maturity_probability=float(u_pre_prob),
            post_maturity_probability=float(u_post_prob),
            pre_maturity_time=pre_maturity,
            post_maturity_time=post_maturity,
            pre_maturity_payment_time=pre_maturity + delay,
            post_maturity_payment_time=post_maturity + delay,
            **ledger,
        )

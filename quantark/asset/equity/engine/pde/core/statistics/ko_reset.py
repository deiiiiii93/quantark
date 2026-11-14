"""The KO-reset statistics in the value mode (spec 3B §5.4; decision 6; unified event statistics T12).

Columns per surface (``ColumnLayout(n_pre, n_post, ki=False, leg=True, two_maturities=True)``): the
pre-schedule knock-outs ``KO_pre`` (the not-knocked-in surface fires them), the post-schedule ones
``KO_post`` (the knocked-in surface), the two maturity legs ``M_pre``, ``M_post`` and the two maturity units
``U_pre``, ``U_post`` (one paid at that maturity's determination). The not-knocked-in surface is staged exactly
as the value's: zero above the pre-schedule maturity, seeded there.

The termination outcomes (DESIGN §4.1): a knock-out on either schedule, the maturity of the first schedule never
knocked in (``M_pre``/``U_pre``, the maturity row) and the maturity of the second schedule knocked in and never
knocked out (``M_post``/``U_post``, the knock-in settlement row). ABSOLUTE schedules only: the PDE refuses
REBASED (DESIGN §4.9a).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from quantark.asset.equity.engine.settlement_support import first_schedule_settlement, resolve_terminal_timing
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, is_close

from . import adjoint
from . import event_statistics as es
from .columns import ColumnLayout
from .rows import knocks_out
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

    family = "ko_reset"

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
        # the knocked-in settlement unit is U_post (no KIS); the outcomes are never pruned (DESIGN §4.9b)
        return ColumnLayout(len(self.ko_records), len(self.post_records), ki=False, leg=True,
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

    # --- the forward adjoint pass's stage plans (unified event statistics T13) ----------------------------------

    def knock_in_horizon(self) -> float:
        """The knock-in monitors the not-knocked-in surface, which lives up to the first schedule's maturity."""
        return float(self.product.get_pre_maturity_time(self.env))

    def adjoint_stages(self, j, ops):
        if j == len(self.t) - 1:
            return self._adjoint_terminal(ops)
        return self._adjoint_two_surface(j, ops)

    def _adjoint_two_surface(self, j, ops):
        """``KOResetSnowballPDESolver._two_surface_events`` at ``j``: the staged seed, the first schedule on the
        not-knocked-in surface, the second on the knocked-in one, the knock-in (while the first lives)."""
        solver, product = self.solver, self.product
        stages = []
        seed = solver._v0_seed_idx
        if seed is not None and j == seed:
            stages.append(adjoint.Seed(adjoint.ALIVE))
        pre = solver._pre_ko_observation_indices.get(j)
        pre_last = pre is not None and solver._pre_ko_goes_last()
        if pre is not None and not pre_last:
            stages += self._adjoint_ko_columns(j, pre, [adjoint.ALIVE], ops)
        post = solver._post_ko_observation_indices.get(j)
        if post is not None and not product.barrier_config.disable_ko_after_ki:
            stages += self._adjoint_ko_columns(j, post, [adjoint.KI], ops)
        if (seed is None or j <= seed) and self._adjoint_ki_applies(j):
            stages.append(self._adjoint_ki_columns(j, ops))
        if pre_last:
            stages += self._adjoint_ko_columns(j, pre, [adjoint.ALIVE], ops)
        return stages

    def _adjoint_terminal(self, ops):
        """``terminal``'s events (the not-knocked-in surface takes none when it is staged)."""
        solver, product = self.solver, self.product
        num_t = len(self.t)
        staged = self.seed_idx is not None
        pre_terminal = solver._has_pre_terminal_ko and solver._pre_ko_terminal_record is not None
        pre_last = pre_terminal and solver._pre_ko_goes_last()
        stages = []
        if pre_terminal and not pre_last:
            stages += self._adjoint_terminal_ko(solver._pre_ko_terminal_record, [adjoint.ALIVE], ops)
        if (solver._has_post_terminal_ko and solver._post_ko_terminal_record is not None
                and not product.barrier_config.disable_ko_after_ki):
            stages += self._adjoint_terminal_ko(solver._post_ko_terminal_record, [adjoint.KI], ops)
        if not staged and self._adjoint_ki_applies(num_t - 1):
            stages.append(self._adjoint_ki_columns(num_t - 1, ops))
        if pre_last:
            stages += self._adjoint_terminal_ko(solver._pre_ko_terminal_record, [adjoint.ALIVE], ops)
        return stages

    def _probability_rows(self, j, current, rows, edges):
        """The maturity units' rows, in determination units, each labelled by the price's own row there (D23): the
        knocked-in surface matures on the post schedule unless its knock-out-side row knocks out (the price's
        knocked-in rows never do); the not-knocked-in surface knocks in at its knock-in edge where the price's row
        knocks in (a post-schedule maturity then), else matures on the first schedule (``U_pre``, while that
        surface lives), and knocks out at the other."""
        solver, product, L = self.solver, self.product, self.layout
        r_lo0, r_up0, r_lo1, r_up1 = rows
        lo0, up0, lo1, up1 = edges
        det_post = solver._df_between_times(self.env, current, solver._total_tau)
        pre_maturity = self._pre_maturity_time()
        alive = current <= pre_maturity or is_close(current, pre_maturity, abs_tol=Tolerance.PRECISION)
        det_pre = solver._df_between_times(self.env, current, pre_maturity) if alive else 0.0
        ko1, ki1 = (r_lo1, r_up1) if product.is_reverse else (r_up1, r_lo1)
        ko0, ki0 = (r_lo0, r_up0) if product.is_reverse else (r_up0, r_lo0)
        ko_edge1 = lo1 if product.is_reverse else up1
        knocked_in_edge = self._edge_knocks_in[j]
        u_pre, u_post = int(L.maturity_unit[0]), int(L.maturity_unit[1])
        ki1[u_post] = det_post
        ko1[u_post] = 0.0 if knocks_out(ko_edge1) else det_post
        ki0[u_pre], ki0[u_post] = (0.0, det_post) if knocked_in_edge else (det_pre, 0.0)
        ko0[u_pre], ko0[u_post] = 0.0, 0.0
        for r in (r_lo1, r_up1):
            r[u_pre] = 0.0

    def _pre_maturity_time(self) -> float:
        """The first schedule's maturity (``get_pre_maturity_time``), resolved once per statistics group -- one per
        solve -- rather than on every step's rows (T25: it resolves the whole first schedule)."""
        pre = self.__dict__.get("_pre_maturity")
        if pre is None:
            pre = self._pre_maturity = float(self.product.get_pre_maturity_time(self.env))
        return pre

    # --- the readout ------------------------------------------------------------

    def _maturity_timing(self, terminal):
        """``(pre, post)`` timing dicts of the two maturities: the first schedule's end (paid on the explicit settlement
        date when booked, else after the terminal delay -- ``first_schedule_settlement``; its dates its last record's
        when that record sits at it) and the product's terminal."""
        product, env = self.product, self.env
        delay = float(terminal.payment_time) - float(terminal.determination_time)
        pre_maturity = float(product.get_pre_maturity_time(env))
        last_pre = self.ko_records[-1] if self.ko_records else None
        pre_det_date = (last_pre.observation_date
                        if last_pre is not None and is_close(float(last_pre.observation_time), pre_maturity)
                        else None)
        # paid on the explicit settlement date when booked (spec D33), else the terminal delay after its maturity
        settlement = first_schedule_settlement(product, env, terminal)
        pre_pay_date = (settlement.payment_date if settlement.settlement_date_governs
                        else pre_det_date if is_close(delay, 0.0) else None)
        pre = dict(time=pre_maturity, payment_time=settlement.payment_time, date=pre_det_date,
                   payment_date=pre_pay_date)
        post = dict(time=float(self.tau), payment_time=float(self.tau) + delay, date=terminal.determination_date,
                    payment_date=terminal.payment_date)
        return pre, post

    def read_statistics(self, states, captures, pv):
        """``EventStatistics`` (ABSOLUTE): both schedules on one observation table tagged ``pre``/``post`` (the pre
        record first at a tie), survival with the F5 drop of the first schedule's maturity mass at its own
        determination; the maturity row is the first schedule's end never knocked in, the knock-in settlement
        row the second schedule's end knocked in and never knocked out (DESIGN §4.2, §4.4, §4.5)."""
        L, env, product = self.layout, self.env, self.product
        at = self._reader(states, captures)
        records = list(self.ko_records) + list(self.post_records)
        n_pre, n_post = len(self.ko_records), len(self.post_records)
        unit = np.array([at(c) for c in L.ko] + [at(c) for c in L.post_ko], dtype=float)
        times = np.array([float(r.observation_time) for r in records], dtype=float)
        tags = np.r_[np.zeros(n_pre), np.ones(n_post)]
        order = np.lexsort((tags, times))
        schedule = ["pre"] * n_pre + ["post"] * n_post
        observations = es.observation_table(env, records, unit, schedule=schedule, order=order)
        terminal = resolve_terminal_timing(product, env)
        pre_timing, post_timing = self._maturity_timing(terminal)
        maturity = es.outcome_row(env, probability=at(L.maturity_unit[0]) / float(env.get_discount_factor(
            pre_timing["time"])), discounted=at(L.maturity[0]), **pre_timing)
        ki_settlement = es.outcome_row(env, probability=at(L.maturity_unit[1]) / float(env.get_discount_factor(
            post_timing["time"])), discounted=at(L.maturity[1]), **post_timing)
        obs_times = observations.times
        if obs_times.size and maturity.time < float(obs_times[-1]) - Tolerance.ZERO:
            # F5: the first schedule's maturity mass leaves the survival at its own determination
            observations = replace(observations, survival=observations.survival - np.where(
                obs_times >= maturity.time, maturity.probability, 0.0))
        unavailable = {}
        knock_in, ki_table = self._knock_in(ki_settlement, unavailable, states, captures)
        lifetime = (float(np.sum(observations.probability * obs_times)) + maturity.probability * maturity.time
                    + ki_settlement.probability * ki_settlement.time)
        return es.assemble(family=self.family, pv=pv, observations=observations, coupons=None, maturity=maturity,
                           ki_settlement=ki_settlement, knock_in=knock_in, ki_table=ki_table,
                           expected_lifetime=lifetime, provenance=self.solver._native_provenance(),
                           unavailable=unavailable)

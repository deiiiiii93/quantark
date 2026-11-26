"""The Phoenix coupon columns of the QUAD V1 statistics off the adjoint forward pass (spec 2026-11-26 D7).

The stacked route marches the coupon columns backward (``phoenix_quad_engine._CouponCashRows``): per observation an
indicator row, a cash row ``U_k``, a release row ``N_k`` (memory) and, under EXPIRY pay, the payment buckets and the
earned-coupon rows -- O(n_obs) rows through every step. Each of them is a linear functional of the march's operators,
so it reads off the ONE forward pass (``v1_adjoint.forward_pass``) instead, with at most two extra dual rows riding it
(``MarchLog.extension``). With ``p`` the main dual (the discounted density of the note's states) before observation
``j`` and the observation's branch-resolved event map -- ``A_pay`` (the continuation after a paid coupon), ``A_miss``
(after a missed coupon, no knock-out), the knock-out values with (``K_pay``) and without (``K_miss``) the coupon:

* indicator ``I_j = p . (A_pay 1 + K_pay 1)`` -- the stacked row's own source (paid branch 1, knocked out with the
  coupon 1);
* the coupon paid at ``j`` on paths carrying no arrears: ``c_j DF_j I_j`` (INSTANT; ``DF_j`` the settlement delay) or
  ``c_j [p . A_pay W_j + DF_j p . K_pay 1]`` (EXPIRY: ``W_j`` the termination row's continuation at ``j``, one unit
  paid when the note ends -- the only coupon row still marched backward, O(1));
* memory: the arrears-weighted dual ``m`` -- paths still holding arrears, weighted by them -- starts as
  ``carried_arrears * p`` and maps by the miss branch alone, ``m <- A_miss^T (m + c_j p)``; at ``j`` the arrears are
  released by a paid coupon (``A_pay``: ``DF_j`` or ``W_j``) or by any knock-out (``DF_j (K_pay + K_miss) 1``). Row
  ``j``'s cash is ``U_j + carried N_j`` of the stacked route in one mass;
* EXPIRY buckets: a deferred-coupon dual ``d`` (the coupons paid, and arrears released, before the note's end) maps
  by the full event map and gains ``A_pay^T (c_j p + m)`` at each observation; a knock-out at ``b`` releases it into
  bucket ``b`` (``DF_b d . (K_pay + K_miss) 1``), the end of the note into the maturity bucket (the terminal unit);
  bucket ``b`` also pays the coupon decided with its knock-out and the arrears that knock-out releases;
* EXPIRY earned coupons (memory): row ``j``'s cash with unit 1 for the discount.

The decided coupon at the valuation instant is added at the readout, as on the stacked route.

Records on one env instant (0.16.0 D14 zero-gap rule, ``CouponGroupStep``): the backward event is one piecewise
projection whose region ``r`` takes the records' chain on that region's indicators (``PhoenixQuadEngine.
_apply_group_observation``). Its transpose: each dual is projected onto region ``r`` (``W_r^T``, read off by the
coloured probes), and there the chain is pointwise -- record k, still alive, passes its coupon (paid / missed) and
knock-out sources exactly as a single record's, the arrears-weighted and deferred duals map by its miss / full / paid
branch as selections -- so every per-record mass above is the sum over the regions of the same formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .v1_adjoint import AdjointUnavailable, probe_event, transpose_pair

__all__ = ["CouponGroupStep", "CouponStep", "ForwardCouponRows", "NodalGroupStep"]


@dataclass(frozen=True)
class CouponStep:
    """One coupon observation of the march: its index, the branch-resolved event map
    ``branches(pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss) -> (out, in)`` (stacked rows x nodes)
    and, under EXPIRY, the termination row's continuation at the observation (``w_out``, ``w_in``)."""

    index: int
    branches: Callable
    w_out: Optional[np.ndarray] = None
    w_in: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CouponGroupStep:
    """The coupon observations of one env instant (0.16.0 D14): per record (contract order) its index; per region of
    the group's piecewise event its projection ``project(v)`` (region ``r`` alone on the stacked nodes, spot last),
    the records' coupon and knock-out indicators there (an inert knock-out never hits) and whether a discrete
    knock-in hits there; ``ki_mask``: the continuous knock-in's nodes (the event then copies the knocked-in columns
    onto the not-knocked-in ones there); ``disable``: knock-outs stop after a knock-in; under EXPIRY the termination
    row after the instant (``w_out``, ``w_in``)."""

    indices: Tuple[int, ...]
    regions: Tuple[Tuple[Callable, Tuple[bool, ...], Tuple[bool, ...], bool], ...]
    disable: bool
    ki_mask: Optional[np.ndarray] = None
    w_out: Optional[np.ndarray] = None
    w_in: Optional[np.ndarray] = None


class NodalGroupStep:
    """A same-instant group on nodal events: the records blend with fractional weights, so the region decomposition
    ``CouponGroupStep`` transposes does not exist; the pass refuses it (``AdjointUnavailable``) and the statistics take
    the stacked route."""


@dataclass(frozen=True)
class ForwardCouponRows:
    """The Phoenix coupon layout of the forward statistics route.

    Backward, the march carries only the termination row ``W`` (EXPIRY: row 0, one unit paid when the note ends) before
    the four terminal rows; forward, it is the pass's extension (module docstring)."""

    n_ko: int
    memory: bool
    expiry: bool
    amounts: np.ndarray
    delay_dfs: np.ndarray
    terminal_delay_df: float
    carried_arrears: float
    decided_index: Optional[int]
    decided_coupon: float
    #: per record ``(index, cash, triggered)``: coupons decided at the valuation instant by several records of one env
    #: instant (0.16.0 D14), each on its own row at its own settlement (``decided_index`` is then ``None``)
    decided_legs: tuple = ()

    termination_row = 0

    # --- the backward rows ------------------------------------------------------------------------------------------

    @property
    def n_extra_rows(self) -> int:
        """``W`` under EXPIRY, else none."""
        return 1 if self.expiry else 0

    def initialize(self, v_out, v_in) -> None:
        """``W`` at maturity: one unit paid at the terminal payment, valued at maturity."""
        if self.expiry:
            v_out[self.termination_row] = self.terminal_delay_df
            v_in[self.termination_row] = self.terminal_delay_df

    def write_observation(self, j, pay_out, pay_in, miss_out, miss_in, breach_pay, breach_miss) -> None:
        """A knock-out at ``j`` ends the note: ``W`` is its settlement delay there, coupon or not."""
        if self.expiry:
            breach_pay[self.termination_row] = breach_miss[self.termination_row] = float(self.delay_dfs[j])

    def step(self, index: int, branches: Callable, pre: Tuple[np.ndarray, np.ndarray]) -> CouponStep:
        """The pass's data for observation ``index``; ``pre``: the columns before its event (``W`` read off them)."""
        if not self.expiry:
            return CouponStep(index=int(index), branches=branches)
        w = self.termination_row
        return CouponStep(index=int(index), branches=branches, w_out=np.array(pre[0][w], dtype=float),
                          w_in=np.array(pre[1][w], dtype=float))

    # --- the forward extension --------------------------------------------------------------------------------------

    def start(self) -> "_CouponRun":
        return _CouponRun(self)

    # --- the readout ------------------------------------------------------------------------------------------------

    def coupon_fields(self, masses: "_CouponMasses", values_at_spot, ko_records, df_fn) -> dict:
        """``coupon_table``'s keywords from the pass's masses, as ``_coupon_columns`` reads the stacked rows."""
        n = self.n_ko
        coupon_probability = np.zeros(n, dtype=float)
        for i, rec in enumerate(ko_records):
            df_obs = float(df_fn(float(rec.observation_time)))
            if df_obs > 0.0:
                coupon_probability[i] = float(masses.indicator[i] / df_obs)
        cash = np.array(masses.cash, dtype=float)
        if self.decided_index is not None:
            # the coupon decided at the valuation instant is paid with certainty
            coupon_probability[self.decided_index] = 1.0
            discount = (float(values_at_spot[self.termination_row]) if self.expiry
                        else float(self.delay_dfs[self.decided_index]))
            cash[self.decided_index] += self.decided_coupon * discount
        for index, amount, triggered in self.decided_legs:
            if triggered:
                coupon_probability[index] = 1.0
            discount = (float(values_at_spot[self.termination_row]) if self.expiry
                        else float(self.delay_dfs[index]))
            cash[index] += float(amount) * discount
        earned = buckets = None
        if self.expiry:
            buckets = np.array(masses.buckets, dtype=float)
            if self.decided_index is not None:
                buckets += self.decided_coupon * masses.units
            for _, amount, _ in self.decided_legs:
                buckets += float(amount) * masses.units
            if self.memory:
                earned = np.array(masses.earned, dtype=float)
                if self.decided_index is not None:
                    earned[self.decided_index] += self.decided_coupon
                for index, amount, _ in self.decided_legs:
                    earned[index] += float(amount)
                earned = earned / np.array([float(df_fn(float(r.observation_time))) for r in ko_records], dtype=float)
            else:
                # without memory a row pays exactly its own coupon when its barrier is met
                earned = np.asarray(self.amounts, dtype=float) * coupon_probability
        return dict(probability=coupon_probability, cash=cash, expiry=bool(self.expiry), earned=earned,
                    buckets=buckets)


@dataclass(frozen=True)
class _CouponMasses:
    """The pass's coupon masses (discounted to the valuation; before the decided coupon): per observation the
    indicator, the cash and (EXPIRY memory) the earned coupon valued at the row; per EXPIRY bucket (knock-out records,
    then the maturity) the unit paid when the note ends there and the coupon cash paid there."""

    indicator: np.ndarray
    cash: np.ndarray
    earned: np.ndarray
    units: np.ndarray
    buckets: np.ndarray


def _dot(dual_out, dual_in, out, inn) -> float:
    return float(dual_out @ out) + float(dual_in @ inn)


class _CouponRun:
    """One forward pass's coupon accumulator (``v1_adjoint`` extension protocol): rows ``m`` (memory) then ``d``
    (EXPIRY) of the extra dual."""

    def __init__(self, rows: ForwardCouponRows):
        self.rows = rows
        n = rows.n_ko
        self.m = 0 if rows.memory else None
        self.d = (1 if rows.memory else 0) if rows.expiry else None
        self.n_rows = int(rows.memory) + int(rows.expiry)
        self.indicator = np.zeros(n, dtype=float)
        self.cash = np.zeros(n, dtype=float)
        self.earned = np.zeros(n, dtype=float)
        self.units = np.zeros(n + 1, dtype=float)
        self.buckets = np.zeros(n + 1, dtype=float)

    def readout(self, beta_out, beta_in):
        x_out = np.zeros((self.n_rows, beta_out.size), dtype=float)
        x_in = np.zeros((self.n_rows, beta_in.size), dtype=float)
        if self.m is not None:
            # the arrears carried into the first future observation weight every path
            x_out[self.m] = self.rows.carried_arrears * beta_out
            x_in[self.m] = self.rows.carried_arrears * beta_in
        return x_out, x_in

    def event(self, data: Optional[CouponStep], banded, n, time, p_out, p_in, x_out, x_in):
        if isinstance(data, CouponGroupStep):
            return self._group_event(data, n, time, p_out, p_in, x_out, x_in)
        if isinstance(data, NodalGroupStep):
            raise AdjointUnavailable(
                f"undefined: the coupon observations sharing the instant t = {time!r} blend by nodal weights; the "
                "forward coupon extension has no transpose of them"
            )
        new_out, new_in = np.zeros_like(x_out), np.zeros_like(x_in)
        if data is None:
            # no coupon here: every extra row maps by the event map itself
            for r in range(self.n_rows):
                new_out[r], new_in[r] = transpose_pair(banded, x_out[r], x_in[r])
            return new_out, new_in
        rows = self.rows
        j = data.index
        c = float(rows.amounts[j])
        df = float(rows.delay_dfs[j])
        src_out, src_in = self._sources(data, n)

        def masses(dual_out, dual_in):
            """The dual's mass on each source row: (indicator, knocked out with coupon, without, paid W)."""
            return [_dot(dual_out, dual_in, src_out[row], src_in[row]) for row in range(src_out.shape[0])]

        p_ind, p_ko_pay, p_ko_miss, *p_w = masses(p_out, p_in)
        self.indicator[j] = p_ind
        # the coupon paid at j on paths carrying no arrears: settled with the record, or (EXPIRY) at the note's end
        cash = c * (p_w[0] + df * p_ko_pay) if rows.expiry else c * df * p_ind
        if self.m is not None:
            m_out, m_in = x_out[self.m], x_in[self.m]
            m_ind, m_ko_pay, m_ko_miss, *m_w = masses(m_out, m_in)
            # the arrears released at j: by a paid coupon (indicator's paid branch), or by any knock-out
            cash += m_w[0] + df * (m_ko_pay + m_ko_miss) if rows.expiry else df * (m_ind + m_ko_miss)
            if rows.expiry:
                self.earned[j] = c * p_ind + m_ind + m_ko_miss
            miss = probe_event(_branch(data.branches, "miss"), n, time)
            new_out[self.m], new_in[self.m] = transpose_pair(miss, m_out + c * p_out, m_in + c * p_in)
        self.cash[j] = cash
        if self.d is not None:
            d_out, d_in = x_out[self.d], x_in[self.d]
            _, d_ko_pay, d_ko_miss, _ = masses(d_out, d_in)
            self.units[j] = df * (p_ko_pay + p_ko_miss)
            # bucket j: the deferred coupons a knock-out here pays, the coupon decided with it, the arrears it releases
            bucket = df * (d_ko_pay + d_ko_miss) + c * df * p_ko_pay
            deferred_out, deferred_in = c * p_out, c * p_in
            if self.m is not None:
                bucket += df * (m_ko_pay + m_ko_miss)
                deferred_out, deferred_in = deferred_out + m_out, deferred_in + m_in
            self.buckets[j] = bucket
            pay = probe_event(_branch(data.branches, "pay"), n, time)
            full_out, full_in = transpose_pair(banded, d_out, d_in)
            paid_out, paid_in = transpose_pair(pay, deferred_out, deferred_in)
            new_out[self.d], new_in[self.d] = full_out + paid_out, full_in + paid_in
        return new_out, new_in

    def _sources(self, data: CouponStep, n):
        """The affine parts of observation ``data`` in one call: rows (0) paid branch 1 and knocked out with the coupon
        1 -- the indicator's source; (1) knocked out with the coupon 1; (2) knocked out without it 1; (3, EXPIRY) the
        paid branch on the termination row's continuation."""
        k = 4 if self.rows.expiry else 3
        pay_out, pay_in = np.zeros((k, n)), np.zeros((k, n))
        breach_pay, breach_miss = np.zeros((k, n)), np.zeros((k, n))
        pay_out[0] = pay_in[0] = 1.0
        breach_pay[0] = breach_pay[1] = 1.0
        breach_miss[2] = 1.0
        if self.rows.expiry:
            pay_out[3], pay_in[3] = data.w_out, data.w_in
        out, inn = data.branches(pay_out, pay_in, np.zeros((k, n)), np.zeros((k, n)), breach_pay, breach_miss)
        return np.asarray(out, dtype=float), np.asarray(inn, dtype=float)

    def _group_event(self, data: CouponGroupStep, n, time, p_out, p_in, x_out, x_in):
        """A same-instant group (module docstring): the masses of every record, region by region and note state by
        note state, and the extra duals mapped onto the columns after the instant."""
        rows = self.rows
        new_out, new_in = np.zeros_like(x_out), np.zeros_like(x_in)
        duals = {"p": [np.asarray(p_out, dtype=float), np.asarray(p_in, dtype=float)]}
        if self.m is not None:
            duals["m"] = [np.asarray(x_out[self.m], dtype=float), np.asarray(x_in[self.m], dtype=float)]
        if self.d is not None:
            duals["d"] = [np.asarray(x_out[self.d], dtype=float), np.asarray(x_in[self.d], dtype=float)]
        if data.ki_mask is not None:
            # the continuous knock-in copies the knocked-in columns onto the not-knocked-in ones on its nodes (after
            # the group, backward): its transpose moves the not-knocked-in dual there onto the knocked-in state
            mask = np.asarray(data.ki_mask, dtype=float)
            duals = {key: [out * (1.0 - mask), inn + out * mask] for key, (out, inn) in duals.items()}
        w_after = {"out": data.w_out, "in": data.w_in}
        for project, coupon_hits, ko_hits, knocked_in in data.regions:
            banded = probe_event(_region_event(project), n, time)
            region = {key: [banded.transpose("out", "out", out), banded.transpose("out", "out", inn)]
                      for key, (out, inn) in duals.items()}
            # the not-knocked-in output: knock-outs active, landing on the knocked-in columns where a knock-in hits;
            # the knocked-in output: knock-outs active unless disabled after a knock-in
            for state, ko_active, land in ((0, True, "in" if knocked_in else "out"), (1, not data.disable, "in")):
                m, d = self._group_chain(data, coupon_hits, ko_hits, ko_active, w_after[land],
                                         {key: value[state] for key, value in region.items()})
                target_out = land == "out"
                if self.m is not None:
                    (new_out if target_out else new_in)[self.m] += m
                if self.d is not None:
                    (new_out if target_out else new_in)[self.d] += d
        return new_out, new_in

    def _group_chain(self, data, coupon_hits, ko_hits, ko_active, w_land, duals):
        """One region and note state of a group: the records in contract order on pointwise duals (``p`` the paths
        alive before the record, ``m`` arrears-weighted, ``d`` deferred), each record's masses as a single record's
        (``event``); returns the ``m`` and ``d`` duals reaching the columns after the instant."""
        rows = self.rows
        p = duals["p"]
        m = duals.get("m")
        d = duals.get("d")
        k_last = len(data.indices)
        # the termination row's continuation after each record (EXPIRY): its settlement where a later record of the
        # group knocks out, else the row after the instant
        w_cont = [None] * k_last
        if rows.expiry:
            w = np.asarray(w_land, dtype=float)
            for pos in range(k_last - 1, -1, -1):
                w_cont[pos] = w
                if ko_active and ko_hits[pos]:
                    w = np.full_like(w, float(rows.delay_dfs[data.indices[pos]]))
        for pos, j in enumerate(data.indices):
            c = float(rows.amounts[j])
            df = float(rows.delay_dfs[j])
            paid = bool(coupon_hits[pos])
            ko = bool(ko_active and ko_hits[pos])
            p_mass = float(np.sum(p))
            p_ind = p_mass if paid else 0.0
            p_ko_pay = p_mass if (ko and paid) else 0.0
            p_ko_miss = p_mass if (ko and not paid) else 0.0
            p_w = float(p @ w_cont[pos]) if (rows.expiry and paid and not ko) else 0.0
            self.indicator[j] += p_ind
            cash = c * (p_w + df * p_ko_pay) if rows.expiry else c * df * p_ind
            new_m = None
            if m is not None:
                m_mass = float(np.sum(m))
                m_ind = m_mass if paid else 0.0
                m_ko_pay = m_mass if (ko and paid) else 0.0
                m_ko_miss = m_mass if (ko and not paid) else 0.0
                m_w = float(m @ w_cont[pos]) if (rows.expiry and paid and not ko) else 0.0
                cash += m_w + df * (m_ko_pay + m_ko_miss) if rows.expiry else df * (m_ind + m_ko_miss)
                if rows.expiry:
                    self.earned[j] += c * p_ind + m_ind + m_ko_miss
                # the miss branch carries the arrears (and this record's missed coupon) on
                new_m = m + c * p if (not paid and not ko) else np.zeros_like(m)
            self.cash[j] += cash
            new_d = None
            if d is not None:
                d_mass = float(np.sum(d))
                d_ko = d_mass if ko else 0.0
                self.units[j] += df * (p_ko_pay + p_ko_miss)
                bucket = df * d_ko + c * df * p_ko_pay
                deferred = c * p
                if m is not None:
                    bucket += df * (m_ko_pay + m_ko_miss)
                    deferred = deferred + m
                self.buckets[j] += bucket
                new_d = np.zeros_like(d) if ko else (d + deferred if paid else d)
            if ko:
                p = np.zeros_like(p)
            m, d = new_m, new_d
        return m, d

    def finish(self, p_out, p_in, x_out, x_in) -> None:
        """The end of the note: the maturity bucket's unit (one paid at the terminal payment, valued at maturity) and
        the deferred coupons paid there."""
        if self.d is None:
            return
        tdf = float(self.rows.terminal_delay_df)
        self.units[-1] = tdf * (float(np.sum(p_out)) + float(np.sum(p_in)))
        self.buckets[-1] = tdf * (float(np.sum(x_out[self.d])) + float(np.sum(x_in[self.d])))

    def result(self) -> _CouponMasses:
        return _CouponMasses(indicator=self.indicator, cash=self.cash, earned=self.earned, units=self.units,
                             buckets=self.buckets)


def _region_event(project: Callable) -> Callable:
    """A group region's projection as an event map ``event(v_out, v_in, source)`` (both states alike), for the
    coloured-probe recovery of ``W_r``."""

    def event(v_out, v_in, _source):
        return project(np.asarray(v_out, dtype=float)), project(np.asarray(v_in, dtype=float))

    return event


def _branch(branches: Callable, which: str) -> Callable:
    """One branch of the observation as an event map ``event(v_out, v_in, source)`` (every other input zero), for the
    coloured-probe recovery."""

    def event(v_out, v_in, _source):
        zero = np.zeros_like(np.asarray(v_out, dtype=float))
        if which == "pay":
            return branches(v_out, v_in, zero, zero, zero, zero)
        return branches(zero, zero, v_out, v_in, zero, zero)

    return event

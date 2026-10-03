"""Event statistics as a column group of the value march (spec 2 §5.4, Level 1).

Moved verbatim from ``core/statistics.py`` (spec 3B §10 step 1): the column indices come from
:class:`ColumnLayout`, the Phoenix coupon hooks are the module functions below (moved from
``PhoenixPDESolver`` with ``self.`` -> ``solver.``, dispatched on the product), and the
readout's dataclass-and-ledger tail is :func:`readout.assemble_autocallable`.

The legacy statistics sweep (``SnowballPDESolver._compute_event_stats``) marched
stacked indicator columns -- ``[KO_0..KO_{n-1}, <extra coupon cols>, KI, KI_ever]``
for both regimes -- through the value operator after the value solve. Here the
same columns are one rolling banded group of the value march: the boundary rows
are carried from the previous step, the step's events follow the value group's,
and the zero-slope rows close each step. Every block below is the legacy body it
cites, moved verbatim with ``self.`` rewritten to ``solver.``.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

from ..groups import STATISTICS_LEGACY, Backend, ColumnGroup
from .columns import ColumnLayout
from .group import RollingStatistics
from .readout import assemble_autocallable

__all__ = ["StatisticsLegacy"]


class StatisticsLegacy(RollingStatistics):
    """The statistics sweep's columns, events and readout for one prepared problem."""

    group = ColumnGroup("stats", ("stats_v0", "stats_v1"), Backend.BANDED, STATISTICS_LEGACY, rolling=True)

    @classmethod
    def create(cls, prepared, streams) -> Optional["StatisticsLegacy"]:
        """The statistics for ``prepared``, or ``None`` when the product has no KO observation."""
        stats = cls(prepared, streams)
        return None if stats.n_ko == 0 else stats

    def __init__(self, prepared, streams):
        solver, product, env = prepared.solver, prepared.product, prepared.env
        layout = prepared.layout
        self.solver, self.product, self.env, self.layout = solver, product, env, layout
        self.spot = env.spot
        self.tau = tau = prepared.tau
        self.x_vec, self.s_vec, self.t_vec = layout.spatial.x, layout.spatial.s, layout.time.t
        num_x, num_t = len(self.x_vec), len(self.t_vec)

        # Determine knocked-in state at valuation (the solve state computed the same call)
        self.already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        self.knocked_in_at_valuation = bool(prepared.solve_state["knocked_in"])

        ko_records = solver._filter_observations_by_tau(
            product.resolve_ko_observations(env), tau
        )
        self.ko_records = ko_records
        self.n_ko = n_ko = len(ko_records)
        if not ko_records:
            return

        # Map time index -> ko record index: the layout's own map (§8), not a
        # nearest-node search.
        ko_index_by_tidx: Dict[int, int] = {}
        for k, rec in enumerate(ko_records):
            obs_time = float(rec.observation_time)
            if is_close(obs_time, 0.0):
                t_idx = 0
            elif is_close(obs_time, tau):
                t_idx = num_t - 1
            else:
                try:
                    t_idx = int(layout.time.step_at(obs_time))
                except KeyError:
                    t_idx = None
            if t_idx is None or not is_close(float(self.t_vec[t_idx]), obs_time):
                raise ValidationError(
                    "Time grid must align with KO observation times for event stats."
                )
            ko_index_by_tidx[t_idx] = k
        self.ko_index_by_tidx = ko_index_by_tidx

        # Surface columns: [KO_0..KO_{n-1}, <extra coupon cols>, KI_indicator,
        # KI_ever_indicator]. The KI_indicator carries the "settles knocked-in"
        # semantics (absorbed to 0 on any KO). The KI_ever_indicator tracks
        # P(the underlying breaches the KI barrier at any point in [0, T]),
        # independent of KO/autocall -- a pure first-passage statistic, EXEMPT
        # from the KO absorption (matching the QUAD and MC ki_ever definition).
        # Stream selection [§11.1]: prune auxiliary indicator columns the caller
        # does not need. KO columns are always present; the KI regime jump always
        # runs (it drives the KO columns), only the KI *indicator* columns and
        # the Phoenix coupon columns are optional.
        if streams is None:
            want_ki = True
            want_coupon = True
        else:
            from quantark.cashleg.event_distribution import EventType

            want_ki = bool(
                streams & {EventType.KI, EventType.MATURITY_WITH_KI}
            )
            want_coupon = EventType.COUPON in streams
        self.want_ki, self.want_coupon = want_ki, want_coupon

        n_extra = _n_extra_event_cols(solver, product, n_ko) if want_coupon else 0
        layout_cols = ColumnLayout(
            n_ko, trigger=want_coupon and isinstance(product, PhoenixOption), ki=want_ki)
        assert layout_cols.n_cols == n_ko + n_extra + (2 if want_ki else 0)
        if want_ki:
            ki_col = layout_cols.kis
            ki_ever_col = layout_cols.kie
        else:
            ki_col = -1
            ki_ever_col = -1
        n_cols = layout_cols.n_cols
        self.columns = layout_cols
        self.ki_col, self.ki_ever_col, self.n_cols = ki_col, ki_ever_col, n_cols

        use_banded = (num_x - 2) > 2
        if not use_banded:
            raise ValidationError("Event stats PDE currently requires banded solver path.")

        # Valuation-date events are deterministic at the known spot: capture
        # the smooth 0+ indicator columns before the j==0 events land, and
        # overlay the exact t=0 outcomes on the spot readout below.
        self.t0_capture = solver._use_cell_average_events() and (
            0 in ko_index_by_tidx
            or (
                product.has_ki_barrier
                and not solver._ki_continuous
                and 0 in solver._ki_observation_indices
            )
        )

    # --- the march hooks ------------------------------------------------------------

    def allocate(self, states) -> None:
        # Terminal conditions at maturity (t = T):
        # - KO indicators are zero at maturity (KO only at discrete observations via jumps)
        # - Both KI indicators are 1 on the KI surface and 0 on the no-KI surface
        num_x = len(self.x_vec)
        v0_next = np.zeros((num_x, self.n_cols), dtype=float)
        v1_next = np.zeros((num_x, self.n_cols), dtype=float)
        if self.want_ki:
            v1_next[:, self.ki_col] = 1.0
            v1_next[:, self.ki_ever_col] = 1.0
        states["stats_v0"], states["stats_v1"] = v0_next, v1_next

    def terminal(self, states) -> None:
        solver, product, pricing_env = self.solver, self.product, self.env
        s_vec, t_vec = self.s_vec, self.t_vec
        v0_next, v1_next = states["stats_v0"], states["stats_v1"]
        want_ki, want_coupon = self.want_ki, self.want_coupon
        ki_ever_col, n_ko = self.ki_ever_col, self.n_ko
        ko_records, ko_index_by_tidx = self.ko_records, self.ko_index_by_tidx
        num_t = len(t_vec)
        # Apply terminal KO/KI events at maturity if observation schedules include t=T.
        terminal_tidx = num_t - 1
        terminal_ko_idx = ko_index_by_tidx.get(terminal_tidx)
        if terminal_ko_idx is not None:
            rec = ko_records[terminal_ko_idx]
            barrier = float(rec.barrier) if rec.barrier is not None else 0.0
            df_delay = solver._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=1.0,
                current_time=float(t_vec[terminal_tidx]),
                settlement_time=rec.settlement_time,
            )
            if solver._use_cell_average_events():
                # KO absorbs every surface to 0 (KO_i to df_delay); KI-ever is
                # exempt (pure first-passage statistic) so its breach target
                # is its own current value.
                for v in (v0_next, v1_next):
                    target = np.zeros_like(v)
                    target[:, terminal_ko_idx] = df_delay
                    if want_ki:
                        target[:, ki_ever_col] = v[:, ki_ever_col]
                    v[:] = solver._project_event_values(
                        s_vec, barrier, product.is_reverse, True, v, target
                    )
            else:
                mask_ko = solver._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

                # KI-ever is exempt from KO absorption (pure first-passage statistic).
                if want_ki:
                    ever0 = v0_next[mask_ko, ki_ever_col].copy()
                    ever1 = v1_next[mask_ko, ki_ever_col].copy()
                v0_next[mask_ko, :] = 0.0
                v1_next[mask_ko, :] = 0.0
                v0_next[mask_ko, terminal_ko_idx] = df_delay
                v1_next[mask_ko, terminal_ko_idx] = df_delay
                if want_ki:
                    v0_next[mask_ko, ki_ever_col] = ever0
                    v1_next[mask_ko, ki_ever_col] = ever1
            if want_coupon:
                _set_extra_event_indicators(
                    solver, v0_next, v1_next, s_vec, n_ko, terminal_ko_idx, rec,
                    product, pricing_env, t_vec, terminal_tidx,
                )

        is_terminal_ki = product.has_ki_barrier and (
            solver._ki_continuous
            or terminal_tidx in solver._ki_observation_indices
        )
        if is_terminal_ki:
            ki_barrier = solver._resolve_ki_barrier_at_tidx(terminal_tidx)
            if solver._use_cell_average_events() and not solver._ki_continuous:
                v0_next[:] = solver._project_event_values(
                    s_vec, ki_barrier, product.is_reverse, False, v0_next, v1_next
                )
            else:
                mask_ki = solver._get_barrier_mask(s_vec, ki_barrier, product.is_reverse, is_up_barrier=False)
                v0_next[mask_ki, :] = v1_next[mask_ki, :]

        # FIRST_PASSAGE continuous-KI state for the stats sweep's KI stage:
        # the indicator columns follow the same regime dynamic programming as
        # the value surfaces, so they take the same crossing correction.
        solver._prepare_continuous_ki_correction(product, pricing_env, t_vec)

    def step_events(self, states, captures, j) -> None:
        solver, product, pricing_env = self.solver, self.product, self.env
        s_vec, t_vec = self.s_vec, self.t_vec
        v0_cur, v1_cur = states["stats_v0"], states["stats_v1"]
        want_ki, want_coupon = self.want_ki, self.want_coupon
        ki_ever_col, n_ko = self.ki_ever_col, self.n_ko
        ko_records, ko_index_by_tidx = self.ko_records, self.ko_index_by_tidx
        if j == 0 and self.t0_capture:
            captures["stats_t0_pre"] = (v0_cur.copy(), v1_cur.copy())
        # Apply KO jump (if observation time).
        ko_idx = ko_index_by_tidx.get(j)
        if ko_idx is not None:
            rec = ko_records[ko_idx]
            barrier = float(rec.barrier) if rec.barrier is not None else 0.0
            df_delay = solver._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=1.0,
                current_time=float(t_vec[j]),
                settlement_time=rec.settlement_time,
            )
            ko_surfaces = (
                (v0_cur, v1_cur) if solver._ko_survives_ki(product) else (v0_cur,)
            )
            if solver._event_uses_projection(j):
                # Zero all event surfaces in the KO region (KO_i indicator
                # to df_delay); KI-ever is exempt (first-passage statistic).
                for v in ko_surfaces:
                    target = np.zeros_like(v)
                    target[:, ko_idx] = df_delay
                    if want_ki:
                        target[:, ki_ever_col] = v[:, ki_ever_col]
                    v[:] = solver._project_event_values(
                        s_vec, barrier, product.is_reverse, True, v, target
                    )
            else:
                mask_ko = solver._event_nodal_mask(
                    s_vec, barrier, product.is_reverse, True,
                    at_valuation=(j == 0),
                )

                # Zero all event surfaces in KO region, then set the KO_i indicator.
                # KI-ever is exempt (pure first-passage statistic, no KO absorption).
                for v in ko_surfaces:
                    if want_ki:
                        ever = v[mask_ko, ki_ever_col].copy()
                    v[mask_ko, :] = 0.0
                    v[mask_ko, ko_idx] = df_delay
                    if want_ki:
                        v[mask_ko, ki_ever_col] = ever
            if want_coupon:
                _set_extra_event_indicators(
                    solver, v0_cur, v1_cur, s_vec, n_ko, ko_idx, rec,
                    product, pricing_env, t_vec, j,
                )

        # Apply KI jump (continuous every step, or discrete at obs indices).
        if product.has_ki_barrier:
            should_apply_ki = (
                solver._ki_continuous
                or j in solver._ki_observation_indices
            )
            if should_apply_ki:
                ki_barrier = solver._resolve_ki_barrier_at_tidx(j)
                ki_discrete = not solver._ki_continuous
                if solver._event_uses_projection(j) and ki_discrete:
                    v0_cur[:] = solver._project_event_values(
                        s_vec, ki_barrier, product.is_reverse, False,
                        v0_cur, v1_cur,
                    )
                else:
                    mask_ki = solver._event_nodal_mask(
                        s_vec, ki_barrier, product.is_reverse, False,
                        at_valuation=(j == 0 and ki_discrete),
                    )
                    v0_cur[mask_ki, :] = v1_cur[mask_ki, :]
                    fp = solver._ki_fp
                    if fp is not None and 0 < j < len(t_vec) - 1:
                        d = v1_cur - v0_cur
                        v0_cur += fp.step_correction(j, s_vec, ki_barrier, d)

        # Enforce simple Neumann-like boundary (zero slope) for stability.
        v0_cur[0, :] = v0_cur[1, :]
        v0_cur[-1, :] = v0_cur[-2, :]
        v1_cur[0, :] = v1_cur[1, :]
        v1_cur[-1, :] = v1_cur[-2, :]

    def read(self, states, captures, pv):
        solver, product, pricing_env = self.solver, self.product, self.env
        spot, tau = self.spot, self.tau
        x_vec, t_vec = self.x_vec, self.t_vec
        v0_next, v1_next = states["stats_v0"], states["stats_v1"]
        want_ki, want_coupon = self.want_ki, self.want_coupon
        ki_col, ki_ever_col, n_ko = self.ki_col, self.ki_ever_col, self.n_ko
        ko_records, ko_index_by_tidx = self.ko_records, self.ko_index_by_tidx
        knocked_in_at_valuation = self.knocked_in_at_valuation
        already_knocked_in = self.already_knocked_in
        t0_pre_grids = captures.get("stats_t0_pre")
        # Select initial regime based on knocked-in at valuation. When the
        # valuation date carries events, read the smooth 0+ branch columns
        # (mirroring the loop's boundary enforcement on the captured copies)
        # and overlay the exact t=0 outcomes at the known spot.
        if t0_pre_grids is not None:
            for g in t0_pre_grids:
                g[0, :] = g[1, :]
                g[-1, :] = g[-2, :]
            initial_grid = (
                t0_pre_grids[1] if knocked_in_at_valuation else t0_pre_grids[0]
            )
        else:
            initial_grid = v1_next if knocked_in_at_valuation else v0_next
        spot_log = float(np.log(spot))

        # Exact valuation-date outcomes at the known spot. Untriggered t=0
        # streams are exactly zero on the pre-event columns already; a
        # triggered t=0 KO resolves every stream deterministically.
        t0_overrides: Dict[int, float] = {}
        rec0_pos = ko_index_by_tidx.get(0)
        if t0_pre_grids is not None and rec0_pos is not None:
            rec0 = ko_records[rec0_pos]
            b0 = float(rec0.barrier) if rec0.barrier is not None else 0.0
            t0_ko_triggered = bool(
                solver._event_nodal_mask(
                    np.asarray([spot], dtype=float),
                    b0,
                    product.is_reverse,
                    True,
                    at_valuation=True,
                )[0]
            )
            df_delay0 = solver._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=1.0,
                current_time=float(t_vec[0]),
                settlement_time=rec0.settlement_time,
            )
            if t0_ko_triggered:
                for i in range(n_ko):
                    t0_overrides[i] = 0.0
                t0_overrides[rec0_pos] = df_delay0
                if want_ki:
                    t0_overrides[ki_col] = 0.0
                    t0_overrides[ki_ever_col] = 0.0
            else:
                t0_overrides[rec0_pos] = 0.0
            if want_coupon:
                t0_overrides.update(
                    _t0_extra_indicator_overrides(
                        solver,
                        product,
                        pricing_env,
                        spot,
                        n_ko,
                        rec0_pos,
                        rec0,
                        t_vec,
                        t0_ko_triggered,
                        df_delay0,
                    )
                )

        def _read_col(col: int) -> float:
            if col in t0_overrides:
                return t0_overrides[col]
            return float(np.interp(spot_log, x_vec, initial_grid[:, col]))

        ed_unit = np.array([_read_col(i) for i in range(n_ko)], dtype=float)

        ki_times = np.array([], dtype=float)
        ki_event_probability = np.array([], dtype=float)
        ki_survival_probability = np.array([], dtype=float)
        if already_knocked_in:
            ki_probability = 1.0
            ki_ever_probability = 1.0
            ki_times = np.array([0.0], dtype=float)
            ki_event_probability = np.array([1.0], dtype=float)
            ki_survival_probability = np.array([0.0], dtype=float)
        elif want_ki:
            df_T = pricing_env.get_discount_factor(float(tau))
            ed_ki = _read_col(ki_col)
            ki_probability = float(ed_ki / df_T) if df_T > 0.0 else 0.0
            ed_ki_ever = _read_col(ki_ever_col)
            ki_ever_probability = float(ed_ki_ever / df_T) if df_T > 0.0 else 0.0
        else:
            # KI columns were pruned (no leg reads KI): report 0, KO/pv are intact.
            ki_probability = 0.0
            ki_ever_probability = 0.0

        if want_coupon:
            extra_fields = _extract_extra_event_stats(
                solver,
                initial_grid,
                x_vec,
                spot_log,
                n_ko,
                ko_records,
                pricing_env,
                product,
                col_overrides=t0_overrides,
            )
        else:
            extra_fields = {}
        return assemble_autocallable(
            solver, product, pricing_env, pv=pv, ko_records=ko_records, ed_unit=ed_unit,
            ki=(ki_probability, ki_ever_probability, ki_times, ki_event_probability, ki_survival_probability),
            extra_fields=extra_fields,
        )


# --- the coupon hooks (moved from PhoenixPDESolver; a Snowball has none) -------------

def _n_extra_event_cols(solver, product, n_ko):
    return n_ko if isinstance(product, PhoenixOption) else 0


def _set_extra_event_indicators(
    solver, v0, v1, s_vec, n_ko, ko_idx, rec, product, pricing_env, t_vec, t_idx
) -> None:
    if not isinstance(product, PhoenixOption):
        return None
    # Set the coupon-trigger column on the coupon-pay mask AFTER the KO jump,
    # so a coupon at a simultaneous KO is still counted (matches the Phoenix MC
    # reference: coupon_hit gated on "alive entering obs i", incl. first_ko_idx==i).
    if ko_idx is None or ko_idx >= solver._coupon_barriers.shape[0]:
        return
    coupon_barrier = float(solver._coupon_barriers[ko_idx])
    coupon_payment_time = solver._coupon_payment_time(
        product,
        pricing_env,
        ko_idx,
    )
    df_delay = solver._cashflow_value_at_time(
        pricing_env=pricing_env,
        cashflow=1.0,
        current_time=float(t_vec[t_idx]),
        settlement_time=coupon_payment_time,
    )
    coup_col = n_ko + ko_idx
    if solver._event_uses_projection(t_idx):
        for v in (v0, v1):
            v[:, coup_col] = solver._project_event_values(
                s_vec, coupon_barrier, product.is_reverse, True,
                v[:, coup_col], df_delay,
            )
        return
    pay_mask = solver._event_nodal_mask(
        s_vec, coupon_barrier, product.is_reverse, True,
        at_valuation=(t_idx == 0),
    )
    v0[pay_mask, coup_col] = df_delay
    v1[pay_mask, coup_col] = df_delay


def _t0_extra_indicator_overrides(
    solver,
    product,
    pricing_env,
    spot,
    n_ko,
    rec0_pos,
    rec0,
    t_vec,
    ko_triggered,
    df_delay0,
) -> dict:
    """Deterministic valuation-date coupon indicator at the known spot.

    Mirrors _set_extra_event_indicators (a coupon at a simultaneous KO is
    still counted); a triggered t=0 KO kills every later coupon stream."""
    if not isinstance(product, PhoenixOption):
        return {}
    overrides: dict = {}
    if rec0_pos is None or rec0_pos >= solver._coupon_barriers.shape[0]:
        return overrides
    coupon_barrier = float(solver._coupon_barriers[rec0_pos])
    pay = bool(
        solver._event_nodal_mask(
            np.asarray([float(spot)], dtype=float),
            coupon_barrier,
            product.is_reverse,
            True,
            at_valuation=True,
        )[0]
    )
    coupon_payment_time = solver._coupon_payment_time(
        product,
        pricing_env,
        rec0_pos,
    )
    coupon_delay0 = solver._cashflow_value_at_time(
        pricing_env=pricing_env,
        cashflow=1.0,
        current_time=float(t_vec[0]),
        settlement_time=coupon_payment_time,
    )
    overrides[n_ko + rec0_pos] = coupon_delay0 if pay else 0.0
    if ko_triggered:
        for i in range(n_ko):
            if i != rec0_pos:
                overrides[n_ko + i] = 0.0
    return overrides


def _extract_extra_event_stats(
    solver,
    initial_grid,
    x_vec,
    spot_log,
    n_ko,
    ko_records,
    pricing_env,
    product,
    col_overrides=None,
) -> dict:
    if not isinstance(product, PhoenixOption):
        return {}
    overrides = col_overrides or {}

    def _read(col: int) -> float:
        if col in overrides:
            return float(overrides[col])
        return float(np.interp(spot_log, x_vec, initial_grid[:, col]))

    ed_coup = np.array([_read(n_ko + i) for i in range(n_ko)], dtype=float)
    coupon_probability = np.zeros(n_ko, dtype=float)
    for i, rec in enumerate(ko_records):
        obs_time = float(rec.observation_time)
        settle = solver._coupon_payment_time(
            product,
            pricing_env,
            i,
        )
        df0 = pricing_env.get_discount_factor(settle)
        if df0 > 0.0:
            coupon_probability[i] = float(ed_coup[i] / df0)
    result = {"coupon_probability": coupon_probability}
    ecc = _coupon_cashflow_from_probability(
        solver, coupon_probability, n_ko, ko_records, pricing_env, product
    )
    if ecc is not None:
        result["expected_discounted_coupon_cashflow"] = ecc
    return result


def _coupon_cashflow_from_probability(
    solver, coupon_probability, n_ko, ko_records, pricing_env, product
):
    """Expected discounted coupon cashflow for non-memory coupons, else None.

    With a deterministic per-period coupon amount and a deterministic
    settlement time, E[DF(0->settle) * amount * 1{coupon}] factors exactly as
    DF(0->settle) * amount * P(coupon). For memory coupons the paid amount is
    path-dependent and cannot be recovered from the trigger indicator, so we
    omit the field (probability stays correct) rather than report a wrong value.
    """
    if product.has_memory_coupon:
        return None
    ecc = np.zeros(n_ko, dtype=float)
    for i, rec in enumerate(ko_records):
        settle = solver._coupon_payment_time(
            product,
            pricing_env,
            i,
        )
        amt = (
            float(solver._coupon_amounts[i])
            if i < solver._coupon_amounts.shape[0]
            else 0.0
        )
        ecc[i] = float(
            pricing_env.get_discount_factor(settle) * amt * coupon_probability[i]
        )
    return ecc

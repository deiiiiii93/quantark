"""DCN Monte Carlo engine (spec WP1.3): curve-aware GBM on the SSE daily grid.

Time grid = the contract's actual trading-day grid on the env's axis, so daily KI is
exact discrete monitoring by construction — NO barrier continuity correction.
Each cash flow discounts at its own payment time, from
``DCNOption.payment_dates()`` (the loss leg at the independent settlement_date,
or every flow on the settlement convention's dates when one is set, W15). PV and all legs are signed by direction_sign,
and pv is DEFINED as the sum of the signed legs (exact invariant).

Simulation is batched (``num_batches``) so large path counts (2^20+) run at
bounded memory; per-batch Sobol streams are shifted by batch_id exactly as in
the snowball/phoenix engines.

Event statistics (unified event statistics spec, T11): ``price_detailed``,
``calculate_event_stats`` and ``price_with_events`` return an
``EventStatistics`` built from the price's own batches
(``_dcn_event_statistics``); ``price()`` records nothing.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.dcn_valuation import (
    DCNStateValuation,
    bind_dcn_fold,
    folded_row_cash,
    refuse_terminal_fold,
    resolve_dcn_lifecycle_state,
    settle_final_observation,
)
from quantark.asset.equity.engine.mc._dcn_event_statistics import (
    dcn_event_statistics,
    terminated_dcn_statistics,
)
from quantark.asset.equity.engine.mc.dcn_payoff import (
    DCNPathCashflows,
    compute_dcn_cashflows,
)
from quantark.asset.equity.engine.mc.qmc_draws import qmc_normals
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.montecarlo.std_error import antithetic_pair_means, mean_and_std_error
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError

_BATCH_SEED_STRIDE = 1000  # matches the snowball/phoenix batch-seed convention


def _default_num_workers() -> int:
    """Batch-level thread workers from QUANTARK_DCN_MC_WORKERS (default 1).

    Threads are effective because the per-batch work (Sobol generation,
    ndtri, path arithmetic, payoff) is GIL-releasing array code; results are
    reduced in batch-id order so any worker count is bit-identical to serial.
    """
    try:
        workers = int(os.environ.get("QUANTARK_DCN_MC_WORKERS", "1"))
    except ValueError:
        workers = 1
    return max(workers, 1)


@dataclass(frozen=True)
class DCNMCResult:
    pv: float
    std_error: float
    num_paths: int
    seed: int
    elapsed_seconds: float
    direction_sign: float
    pv_fixed_coupons: float
    pv_fixed_coupons_by_period: Tuple[float, ...]
    pv_ko_coupons: float
    pv_ko_coupons_by_period: Tuple[float, ...]
    pv_loss_leg: float
    ki_probability: float
    ko_probability: float
    ko_timing_distribution: Tuple[float, ...]
    coupon_probability: Tuple[float, ...]
    expected_life_years: float
    prob_survive_no_ki: float
    prob_survive_ki: float
    #: the run's ``EventStatistics`` (``price_detailed``; ``None`` from a plain ``price`` and for a terminated,
    #: fully settled lifecycle state); ``event_stats.pv == pv`` (the receivables are its receivables table)
    event_stats: Optional[EventStatistics] = field(repr=False, default=None)
    # W14: PV of the lifecycle state's pending receivables (signed), included
    # in ``pv``; the legs and s.e. are the contingent claim's.
    pv_pending_receivables: float = 0.0

    def to_dict(self) -> dict:
        d = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in self.__dict__.items()
            if k != "event_stats"
        }
        d["event_stats"] = (
            _json_value(self.event_stats) if self.event_stats is not None else None
        )
        return d


def _json_value(value):
    """A JSON-safe copy of a statistics value: dataclasses and mappings as dicts, arrays and tuples as lists,
    dates as ISO strings."""
    import dataclasses
    from collections.abc import Mapping
    from datetime import datetime

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dcn_batch_record(cf, obs_times: np.ndarray, n_obs: int, n_cols: int) -> dict:
    """One batch's event-statistics sums (BUYER-perspective, unsigned, discounted), read-only on the
    batch's path cashflows: the batch-mean standard errors need them batch by batch.

    ``ko_clean_count``: knock-outs with no knock-in before them (the knock-in table's survival);
    ``ki_first_count``: first knock-ins per grid column (the daily KI dates; column 0 = carried or at the
    valuation close), from ``cf.ki_col`` (the kernel's ``first_knock_in`` run)."""
    if cf.ki_col is None:
        raise ValidationError("the DCN statistics record needs the kernel's first knock-in columns")
    ko_mask = cf.ko_obs_row >= 0
    rows = np.maximum(cf.ko_obs_row, 0)
    survive = ~ko_mask
    clean = ko_mask & ~cf.knocked_in
    return {
        "kind": "native",
        "n": int(cf.ko_obs_row.size),
        "pv_sum": float(cf.total_pv.sum()),
        "fixed_disc_cash": float(cf.fixed_coupon_pv.sum()),
        "fixed_disc_cash_by_period": cf.fixed_coupon_pv_by_period.sum(axis=0),
        "ko_count": np.bincount(rows[ko_mask], minlength=n_obs).astype(float)[:n_obs],
        "ko_disc_cash": np.bincount(rows[ko_mask], weights=cf.ko_pv[ko_mask], minlength=n_obs)[:n_obs],
        "coupon_count": cf.coupon_paid.sum(axis=0).astype(float),
        "loss_disc_cash": float(cf.loss_pv.sum()),
        "ki_count": float(cf.knocked_in.sum()),
        "survive_no_ki_count": float((survive & ~cf.knocked_in).sum()),
        "survive_ki_count": float((survive & cf.knocked_in).sum()),
        "life_sum": float(np.where(ko_mask, obs_times[rows], obs_times[-1]).sum()),
        "ko_clean_count": np.bincount(rows[clean], minlength=n_obs).astype(float)[:n_obs],
        "ki_first_count": np.bincount(cf.ki_col[cf.ki_col >= 0], minlength=n_cols).astype(float)[:n_cols],
    }


@dataclass
class _LegAccumulator:
    """Running sums over batches (BUYER-perspective, unsigned)."""

    n_obs: int
    #: grid columns (``ctx.times.size``): the knock-in table's per-column counts
    n_cols: int = 0
    #: keep one statistics record per added batch (``price_detailed``); a plain ``price`` records nothing
    record: bool = False
    n: int = 0
    totals: list = field(default_factory=list)  # per-path totals, for stderr
    fixed_sum: float = 0.0
    ko_sum: float = 0.0
    loss_sum: float = 0.0
    fixed_by_period_sum: Optional[np.ndarray] = None
    ko_by_period_sum: Optional[np.ndarray] = None
    coupon_paid_sum: Optional[np.ndarray] = None
    ko_timing_count: Optional[np.ndarray] = None
    ki_count: int = 0
    survive_no_ki_count: int = 0
    survive_ki_count: int = 0
    life_sum: float = 0.0
    #: one record of per-batch sums per added batch, in batch order (the statistics' batch means)
    batch_records: list = field(default_factory=list)

    def __post_init__(self):
        self.fixed_by_period_sum = np.zeros(self.n_obs)
        self.ko_by_period_sum = np.zeros(self.n_obs)
        self.coupon_paid_sum = np.zeros(self.n_obs)
        self.ko_timing_count = np.zeros(self.n_obs)

    def add(self, cf, obs_times: np.ndarray) -> None:
        if self.record:
            self.batch_records.append(_dcn_batch_record(cf, obs_times, self.n_obs, self.n_cols))
        n_b = cf.ko_obs_row.size
        self.n += n_b
        self.totals.append(cf.total_pv)
        self.fixed_sum += float(cf.fixed_coupon_pv.sum())
        self.ko_sum += float(cf.ko_pv.sum())
        self.loss_sum += float(cf.loss_pv.sum())
        self.fixed_by_period_sum += cf.fixed_coupon_pv_by_period.sum(axis=0)
        ko_mask = cf.ko_obs_row >= 0
        counts = np.bincount(
            cf.ko_obs_row[ko_mask], minlength=self.n_obs
        ).astype(float)
        self.ko_timing_count += counts
        ko_vals = np.zeros(self.n_obs)
        np.add.at(ko_vals, cf.ko_obs_row[ko_mask], cf.ko_pv[ko_mask])
        self.ko_by_period_sum += ko_vals
        self.coupon_paid_sum += cf.coupon_paid.sum(axis=0)
        self.ki_count += int(cf.knocked_in.sum())
        survive = ~ko_mask
        self.survive_no_ki_count += int((survive & ~cf.knocked_in).sum())
        self.survive_ki_count += int((survive & cf.knocked_in).sum())
        life = np.where(
            ko_mask, obs_times[np.maximum(cf.ko_obs_row, 0)], obs_times[-1]
        )
        self.life_sum += float(life.sum())


def _finalize_dcn_result(
    acc,
    product,
    seed,
    stderr,
    t0,
    *,
    pricing_env,
    ctx,
    receivables: Optional[DCNStateValuation] = None,
    engine_name: Optional[str] = None,
    deterministic: bool = False,
) -> "DCNMCResult":
    """Assemble a DCNMCResult from a fully-populated accumulator.

    Shared verbatim by the direct path and the execution-framework batch
    reducer so the two can never drift arithmetically. ``receivables`` (a
    lifecycle state's pending cash, W14) adds to ``pv`` only. An accumulator
    that recorded its batches (``acc.record``) also gets the run's
    ``EventStatistics`` (provenance ``engine_name``; ``deterministic``: the
    settled final row, nothing simulated).
    """
    sign = product.direction_sign
    n = acc.n
    # legs first; pv is DEFINED as their sum (exact invariant)
    pv_fixed = float(sign * acc.fixed_sum / n)
    pv_ko = float(sign * acc.ko_sum / n)
    pv_loss = float(sign * acc.loss_sum / n)
    pv = pv_fixed + pv_ko + pv_loss

    pending = 0.0
    if receivables is not None:
        pending = receivables.pending_pv
        pv = pv + pending
    event_stats = None
    if acc.record:
        event_stats = dcn_event_statistics(
            product, pricing_env, ctx, pv=pv, records=acc.batch_records, engine_name=engine_name,
            state=receivables, deterministic=deterministic,
        )
    return DCNMCResult(
        pv=pv,
        std_error=stderr,
        num_paths=n,
        seed=seed,
        elapsed_seconds=time.perf_counter() - t0,
        direction_sign=sign,
        pv_fixed_coupons=pv_fixed,
        pv_fixed_coupons_by_period=tuple(
            float(sign * x) for x in acc.fixed_by_period_sum / n
        ),
        pv_ko_coupons=pv_ko,
        pv_ko_coupons_by_period=tuple(
            float(sign * x) for x in acc.ko_by_period_sum / n
        ),
        pv_loss_leg=pv_loss,
        ki_probability=float(acc.ki_count / n),
        ko_probability=float(acc.ko_timing_count.sum() / n),
        ko_timing_distribution=tuple(
            float(x) for x in acc.ko_timing_count / n
        ),
        coupon_probability=tuple(
            float(x) for x in acc.coupon_paid_sum / n
        ),
        expected_life_years=float(acc.life_sum / n),
        prob_survive_no_ki=float(acc.survive_no_ki_count / n),
        prob_survive_ki=float(acc.survive_ki_count / n),
        event_stats=event_stats,
        pv_pending_receivables=pending,
    )


def receivables_only_result(product, seed, t0, receivables: DCNStateValuation, *,
                            engine_name: Optional[str] = None, statistics: bool = False) -> "DCNMCResult":
    """A knocked-out or matured lifecycle state (W14): the price is the
    ledger's pending receivable PV; nothing is simulated (``num_paths`` 0,
    legs, s.e. and statistics 0). ``statistics``: the event statistics are
    the receivables alone, or ``None`` when nothing is left unpaid (spec AC17)."""
    n_obs = len(product.schedule.monthly)
    zeros = tuple(0.0 for _ in range(n_obs))
    return DCNMCResult(
        pv=receivables.pending_pv, std_error=0.0, num_paths=0, seed=seed,
        elapsed_seconds=time.perf_counter() - t0, direction_sign=product.direction_sign,
        pv_fixed_coupons=0.0, pv_fixed_coupons_by_period=zeros, pv_ko_coupons=0.0,
        pv_ko_coupons_by_period=zeros, pv_loss_leg=0.0, ki_probability=0.0, ko_probability=0.0,
        ko_timing_distribution=zeros, coupon_probability=zeros, expected_life_years=0.0,
        prob_survive_no_ki=0.0, prob_survive_ki=0.0,
        event_stats=(terminated_dcn_statistics(product, receivables, pv=receivables.pending_pv,
                                               engine_name=engine_name) if statistics else None),
        pv_pending_receivables=receivables.pending_pv,
    )


def settled_dcn_result(product, pricing_env, ctx, seed, t0, receivables=None, fold=None, *,
                       engine_name: Optional[str] = None, statistics: bool = False) -> "DCNMCResult":
    """The DCN on its final observation date (W10, ruling Q6): the settled
    final row from ``dcn_valuation`` as a one-path result, so pv, legs and
    event statistics are the row's 0/1 decisions; ``std_error`` is 0.0 and
    ``expected_life_years`` 0.0. ``fold`` (a bound ``DCNFoldCash``,
    ``price_folded`` only) adds the decided event's folded cash to its leg:
    the knock-out leg on a knock-out, the maturity leg on a survivor (D13).
    ``statistics`` (never with ``fold``): the row's event statistics."""
    row = settle_final_observation(product, pricing_env, ctx)
    ko_pv, loss_pv = row.ko_pv, row.loss_pv
    if fold is not None:
        cash = folded_row_cash(row, fold, ctx, make_df_fn(pricing_env))
        if row.knocked_out:
            ko_pv = ko_pv + cash
        else:
            loss_pv = loss_pv + cash
    cf = DCNPathCashflows(
        fixed_coupon_pv=np.array([row.coupon_pv]),
        fixed_coupon_pv_by_period=np.array([[row.coupon_pv]]),
        ko_pv=np.array([ko_pv]),
        loss_pv=np.array([loss_pv]),
        ko_obs_row=np.array([0 if row.knocked_out else -1], dtype=int),
        knocked_in=np.array([row.knocked_in]),
        coupon_paid=np.array([[row.coupon_paid]]),
        ki_col=np.array([0 if row.knocked_in else -1]),
    )
    acc = _LegAccumulator(n_obs=1, n_cols=int(ctx.times.size), record=statistics and fold is None)
    acc.add(cf, ctx.times[ctx.obs_cols])
    return _finalize_dcn_result(
        acc, product, seed, 0.0, t0, pricing_env=pricing_env, ctx=ctx,
        receivables=receivables, engine_name=engine_name, deterministic=True,
    )


def pathwise_stderr(batch_totals, sign, n, *, antithetic: bool) -> float:
    """Pathwise standard error of the pooled per-path totals of all batches.

    ``batch_totals`` is one array per batch.  Plain runs use the iid formula on the pooled paths.  An antithetic
    run pairs WITHIN each batch (``[z; -z][:batch]`` layout, spec I3/A9), pools the pair means of all batches
    into one unit set, and reports ``std(units, ddof=1) / sqrt(#units)``.  Single-batch Sobol keeps the iid
    formula, which is only a conservative proxy for deterministic QMC (use ``num_batches >= 2``).
    """
    if antithetic:
        units = np.concatenate(
            [antithetic_pair_means(sign * t, len(t)) for t in batch_totals]
        )
        return mean_and_std_error(units)[1]
    totals = sign * np.concatenate(batch_totals)
    return float(totals.std(ddof=1) / np.sqrt(n))


class DCNMCEngine(BaseEngine):
    """GBM MC engine for DCNOption; flat markets are just flat curves.

    Sobol QMC by default (fixed seed => bit-reproducible in the same
    environment). Subclasses override ``_simulate`` only (path generation);
    payoff, discounting, leg decomposition and event stats are shared.
    """

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True  # DCNLifecycleState (W14)
    supports_cash_leg_folding = True
    #: the overlay of a running ``price_folded``; None on every other call
    _fold = None
    # Session-path hook (execution-framework adapter clones only): a callable
    # ``(n_dims, n_paths, batch_id) -> ndarray | None`` consulted before the
    # engine's own generators; None on every direct-path engine.
    _draw_provider = None
    #: the per-batch statistics records of the last simulated price (``price_detailed``), batch order;
    #: empty when the price simulated nothing (a settled final row or receivables only)
    _last_stats_records: tuple = ()

    def __init__(
        self,
        num_paths: int = 131072,
        seed: int = 42,
        use_sobol: bool = True,
        use_antithetic: bool = False,
        num_batches: int = 1,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        if use_sobol and use_antithetic:
            raise ValidationError("Sobol QMC and antithetic are mutually exclusive")
        if num_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {num_paths}")
        if num_batches <= 0 or num_paths % num_batches != 0:
            raise ValidationError(
                "num_batches must be positive and divide num_paths, got "
                f"{num_batches} for {num_paths} paths"
            )
        if num_workers is None:
            num_workers = _default_num_workers()
        if (
            isinstance(num_workers, bool)
            or not isinstance(num_workers, int)
            or num_workers < 1
        ):
            raise ValidationError(
                f"num_workers must be a positive integer, got {num_workers!r}"
            )
        self.num_paths = int(num_paths)
        self.seed = int(seed)
        self.use_sobol = bool(use_sobol)
        self.use_antithetic = bool(use_antithetic)
        self.num_batches = int(num_batches)
        self.num_workers = int(num_workers)
        self._last_result: Optional[DCNMCResult] = None

    # -- cash-leg folding (ship spec 2026-10-21 D4, D9, D11c, D13) --
    def price_folded(self, product, pricing_env, overlay, *, lifecycle_state=None) -> float:
        """Per-unit PV with ``overlay``'s (buyer-view) cash legs folded into each path's legs.

        One ``price`` on the same paths: knock-out cash joins the knock-out
        coupon of every path knocking out at that observation, maturity cash
        the loss leg of every surviving path on its branch (knocked in or
        not), both before their payment discount and before
        ``direction_sign``; the final observation date's settled row pays
        its decided event's cash. ``overlay.constant_pv`` is not included.
        An empty overlay is ``price()`` bitwise; a knocked-out or matured
        lifecycle state refuses a non-empty one (CapabilityError).
        """
        if overlay.is_empty:
            return self.price(product, pricing_env, lifecycle_state=lifecycle_state)
        self._fold = overlay
        try:
            return self.price(product, pricing_env, lifecycle_state=lifecycle_state)
        finally:
            self._fold = None

    def fold_drilldown_refusal(self, product, pricing_env) -> Optional[str]:
        """None on every DCN MC layout (D11c): the path set never depends on
        the payoff. ``num_batches`` is fixed (no adaptive stopping), each
        batch's draws are keyed by its batch id alone, and worker threads are
        reduced in batch-id order, so a folded and a plain solve run the
        same paths and their per-leg differences are exact."""
        return None

    # -- BaseEngine-compatible entry point --
    def price(self, product, pricing_env, *, lifecycle_state=None) -> float:
        return self._price_run(
            product, pricing_env, lifecycle_state=lifecycle_state, statistics=False
        ).pv

    def calculate_event_stats(
        self, product, pricing_env, *, streams=None, lifecycle_state=None
    ) -> Optional[EventStatistics]:
        """The DCN's ``EventStatistics`` from one ``price()`` run on its own paths (``pv`` is that price, D23).

        ``None`` for a knocked-out or matured lifecycle state with nothing left unpaid (spec AC17). ``streams`` is
        accepted and ignored: every number comes from the same paths, nothing is pruned (as the autocallable MC
        engines).
        """
        return self._price_run(
            product, pricing_env, lifecycle_state=lifecycle_state, statistics=True
        ).event_stats

    def price_with_events(
        self, product, pricing_env, emit_distribution=True, streams=None, *, lifecycle_state=None
    ):
        """NPV and the DCN's event distribution from one simulation (ship spec 2026-10-21 D10; ``npv`` is
        ``price()``).

        The distribution is ``EventDistribution.from_statistics`` of the run's statistics; a knocked-out or
        matured lifecycle state has no event left (``EventDistribution.ended``: an event-dependent leg refuses it,
        spec AC17). ``emit_distribution=False`` gets the maturity-only distribution without recording statistics.
        """
        from quantark.cashleg.event_distribution import EventDistribution, PricingResult, knock_in_was_carried

        result = self._price_run(
            product, pricing_env, lifecycle_state=lifecycle_state, statistics=bool(emit_distribution)
        )
        if not emit_distribution:
            dist = EventDistribution.trivial(max(float(product.get_maturity(pricing_env)), 0.0))
        elif result.event_stats is not None:
            dist = EventDistribution.from_statistics(
                result.event_stats, knock_in_carried=knock_in_was_carried(product, lifecycle_state))
        else:
            # a terminated, fully settled note: no statistics (AC17) and no event
            dist = EventDistribution.ended()
        return PricingResult(npv=result.pv, event_distribution=dist)

    def price_detailed(
        self, product, pricing_env, *, lifecycle_state=None
    ) -> DCNMCResult:
        """Price ``product``; a ``DCNLifecycleState`` adds its pending
        receivables, and a knocked-out / matured one prices them alone (W14).
        The result carries the run's ``EventStatistics`` (``event_stats.pv ==
        pv``)."""
        return self._price_run(
            product, pricing_env, lifecycle_state=lifecycle_state, statistics=True
        )

    def _price_run(
        self, product, pricing_env, *, lifecycle_state=None, statistics: bool
    ) -> DCNMCResult:
        """One priced run; ``statistics`` also records the batches and builds
        the ``EventStatistics`` (never while folding). The price arithmetic is
        the same either way."""
        if not isinstance(product, DCNOption):
            raise PricingError(
                f"{type(self).__name__} only supports DCNOption, "
                f"got {type(product).__name__}"
            )
        statistics = bool(statistics) and self._fold is None
        name = type(self).__name__
        t0 = time.perf_counter()
        self._last_stats_records = ()
        receivables = resolve_dcn_lifecycle_state(
            product, pricing_env, lifecycle_state
        )
        if receivables is not None and receivables.terminal:
            if self._fold is not None:
                refuse_terminal_fold()
            result = receivables_only_result(
                product, self.seed, t0, receivables, engine_name=name, statistics=statistics
            )
            self._last_result = result
            return result
        ctx = build_dcn_grid_context(product, pricing_env)
        # folded cash legs (price_folded only), bound to this grid's rows
        fold = None if self._fold is None else bind_dcn_fold(self._fold, ctx)
        if ctx.times.size == 1:
            # the final observation date (W10): the settled final row, no
            # simulation and no model preparation
            result = settled_dcn_result(
                product, pricing_env, ctx, self.seed, t0, receivables, fold=fold,
                engine_name=name, statistics=statistics,
            )
            self._last_result = result
            return result
        dt_array = np.diff(ctx.times)
        if dt_array.size == 0:
            raise ValidationError("DCN grid has no steps (valuation at maturity?)")
        term = build_mc_term_inputs(
            pricing_env,
            ref_strike=product.initial_price,
            maturity=float(ctx.times[-1]),
            time_steps=dt_array.size,
            dt_array=dt_array,
        )
        df = make_df_fn(pricing_env)
        spot0 = float(pricing_env.spot)
        n_obs = ctx.obs_cols.size
        obs_times = ctx.times[ctx.obs_cols]

        acc = _LegAccumulator(n_obs=n_obs, n_cols=int(ctx.times.size), record=statistics)
        batch_size = self.num_paths // self.num_batches
        self._prepare_simulation(product, pricing_env)

        def run_batch(b: int):
            batch_id = None if self.num_batches == 1 else b
            paths = self._simulate(
                spot0, term, dt_array, pricing_env,
                n_paths=batch_size, batch_id=batch_id,
            )
            return compute_dcn_cashflows(paths, product, ctx, df, fold=fold, first_knock_in=statistics)

        workers = min(self.num_workers, self.num_batches)
        if workers > 1:
            # The per-batch work is GIL-releasing array code, so threads
            # scale; reducing in batch-id order keeps every worker count
            # bit-identical to the serial loop.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                cashflows = list(pool.map(run_batch, range(self.num_batches)))
        else:
            cashflows = [run_batch(b) for b in range(self.num_batches)]
        for cf in cashflows:
            acc.add(cf, obs_times)
        # the statistics' batch records of this price (one per batch, batch order; none from a plain price)
        self._last_stats_records = tuple(acc.batch_records)

        sign = product.direction_sign
        n = acc.n
        if self.use_sobol and self.num_batches >= 2:
            # RQMC: paths inside one Sobol batch are NOT independent, so the
            # IID pathwise formula is invalid; estimate the error from the
            # variance of the independent per-scramble batch means (same
            # convention as the repo's RQMC driver).
            batch_means = np.array(
                [float(sign * t.mean()) for t in acc.totals]
            )
            stderr = float(
                batch_means.std(ddof=1) / np.sqrt(batch_means.size)
            )
        else:
            # pseudorandom IID paths (exact), or single-batch Sobol where
            # the pathwise formula is only a conservative upper-bound proxy
            # — use num_batches >= 2 for a valid QMC error estimate.
            stderr = pathwise_stderr(
                acc.totals, sign, n, antithetic=self._antithetic_pairs_in_draws()
            )
        result = _finalize_dcn_result(
            acc,
            product,
            self.seed,
            stderr,
            t0,
            pricing_env=pricing_env,
            ctx=ctx,
            receivables=receivables,
            engine_name=name,
        )
        self._last_result = result
        return result

    def _antithetic_pairs_in_draws(self) -> bool:
        """True when each batch's draws are the per-batch ``[z; -z][:n]`` layout (see ``_draws``)."""
        return (
            bool(self.use_antithetic)
            and not self.use_sobol
            and self._draw_provider is None
        )

    # -- per-price-call setup (overridable; base = nothing to prepare) --
    def _prepare_simulation(self, product, pricing_env) -> None:
        """Called once per ``price_detailed`` before the batch loop.

        Subclasses hoist environment-invariant per-call work here (e.g. the
        LV engine's Dupire inversion) instead of repeating it per batch; the
        default engine has none.
        """

    # -- path generation (overridable; base = term-aware GBM) --
    def _draws(
        self, n_dims: int, n_paths: int, batch_id: Optional[int]
    ) -> np.ndarray:
        if self._draw_provider is not None:
            block = self._draw_provider(n_dims, n_paths, batch_id)
            if block is not None:
                return block
        if self.use_sobol:
            return qmc_normals(self.seed, n_paths, n_dims, batch_id=batch_id)
        seed = self.seed + (
            0 if batch_id is None else int(batch_id) * _BATCH_SEED_STRIDE
        )
        rng = np.random.default_rng(seed)
        if self.use_antithetic:
            half = (n_paths + 1) // 2
            z = rng.standard_normal((half, n_dims))
            return np.vstack([z, -z])[:n_paths]
        return rng.standard_normal((n_paths, n_dims))

    def _simulate(
        self, spot0, term, dt_array, pricing_env,
        n_paths: int, batch_id: Optional[int],
    ) -> np.ndarray:
        n_steps = dt_array.size
        z_all = self._draws(n_steps, n_paths, batch_id)
        nodes = np.empty((z_all.shape[0], n_steps + 1))
        nodes[:, 0] = spot0
        log_s = np.full(z_all.shape[0], np.log(spot0))
        sqrt_dt = np.sqrt(dt_array)
        for i in range(n_steps):
            drift = float(term.rrf[i] - term.div[i])
            vol = float(term.vol[i])
            log_s = (
                log_s
                + (drift - 0.5 * vol * vol) * dt_array[i]
                + vol * sqrt_dt[i] * z_all[:, i]
            )
            nodes[:, i + 1] = np.exp(log_s)
        return nodes

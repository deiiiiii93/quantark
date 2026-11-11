"""
Path-based analysis utilities for autocallable products (Snowball-first).

This module provides a Monte Carlo based analyzer that produces:
- risk-neutral event statistics (:class:`EventStatistics`) from its own simulated paths
- historical replay (shock-based) and parametric scenario PnL distributions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from quantark.asset.equity.engine.event_stats import (
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    ReceivablesTable,
    StatisticsProvenance,
    build_cash_ledger,
)
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine, _ki_level_breached
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError

#: ``provenance.statistics_model`` of the analyzer's statistics
ANALYZER_STATISTICS_MODEL = "path_analyzer_mc"
#: ``unavailable`` reason of the continuous-KI fields the analyzer cannot compute
CONTINUOUS_KI_REASON = (
    "path analyzer: a continuous knock-in is known only to its monitoring step, not its first-crossing time"
)


@dataclass(frozen=True)
class ShockPnLDistribution:
    """PnL distribution from factor shocks applied to today's state."""

    pnl: np.ndarray

    def summary(self) -> Dict[str, float]:
        if self.pnl.size == 0:
            return {"count": 0.0}
        quantiles = np.quantile(self.pnl, [0.01, 0.05, 0.5, 0.95, 0.99])
        return {
            "count": float(self.pnl.size),
            "mean": float(np.mean(self.pnl)),
            "std": float(np.std(self.pnl, ddof=1)) if self.pnl.size > 1 else 0.0,
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p95": float(quantiles[3]),
            "p99": float(quantiles[4]),
        }


class AutocallablePathAnalyzer:
    """Snowball-first path analyzer for event stats and cashflow attribution."""

    def __init__(
        self,
        mc_params: Optional[MCParams] = None,
        *,
        q_bump: float = 1e-4,
    ) -> None:
        if mc_params is None:
            mc_params = MCParams()
        if q_bump <= 0:
            raise ValidationError(f"q_bump must be positive, got {q_bump}")
        self.mc_params = mc_params
        self.q_bump = q_bump

    def analyze_snowball_risk_neutral(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        *,
        seed_offset: int = 1337,
    ) -> EventStatistics:
        """
        Risk-neutral event statistics of a Snowball from the analyzer's own simulated paths.

        Every path ends in exactly one outcome: a knock-out at its first valid KO observation, the maturity
        row (never knocked out, never knocked in) or the knock-in settlement row (knocked in, never knocked
        out). A knock-in counts only before a knock-out (DESIGN §4.5); on one grid node the knock-out wins
        unless ``disable_ko_after_ki`` invalidated that knock-out.

        Notes:
        - Uses the SnowballMCEngine's time grid builder and barrier checks.
        - ``pv`` is the MC engine's ``price()`` (spec D23); the analyzer's own cash sum differs from it by
          ``reconciliation_error`` (separately simulated paths).
        - Continuous KI: P(knock in) is exact on the analyzer's grid (a crossing inside a step precedes the
          step's end), but the first-crossing time is known only to its step, so the per-date KI table and
          the expected KI time are unavailable (:data:`CONTINUOUS_KI_REASON`).
        - One path set is one batch: ``standard_errors`` is ``None`` (``"undefined: single batch"``).
        - No dates are reported (times on the pricing axis only).
        """
        engine = SnowballMCEngine(params=self.mc_params)
        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)

        all_times, dt_array, ko_indices, ki_indices = engine._build_time_grid(
            product, pricing_env, maturity
        )
        generator = engine._create_path_generator(spot, rate, div, vol, maturity, dt_array)
        paths, _ = generator.generate_paths(return_aux=False)

        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_times = np.array(ko_profile["observation_times"], dtype=float)
        ko_payoffs = np.array(ko_profile["payoffs"], dtype=float)
        ko_settlement_times = np.array(ko_profile["settlement_times"], dtype=float)
        ko_barriers = np.array(ko_profile["barriers"], dtype=float)

        ko_triggered, first_ko_idx = engine._check_ko_barriers(
            paths, ko_indices, ko_barriers, product.is_reverse
        )

        ki_continuous = False
        ki_triggered = np.zeros(len(paths), dtype=bool)
        first_ki_idx = np.full(len(paths), -1, dtype=int)
        if product.has_ki_barrier:
            ki_continuous = (
                product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
                or product.barrier_config.ki_continuous
            )
            ki_profile = product.get_ki_observation_profile(pricing_env)
            ki_barriers = np.array(ki_profile["barriers"], dtype=float)
            if ki_continuous:
                if ki_barriers.shape not in ((), (1,)):
                    raise ValidationError(
                        "Continuous KI monitoring requires a scalar ki_barrier."
                    )
                ki_barrier = float(ki_barriers.reshape(-1)[0])
                ki_triggered, first_ki_idx = engine._check_ki_barriers_continuous_with_bridge(
                    paths=paths,
                    all_times=all_times,
                    ki_barrier=ki_barrier,
                    sigma=float(vol),
                    is_reverse=product.is_reverse,
                    rng_seed=int(self.mc_params.seed) + int(seed_offset),
                )
            else:
                ki_triggered, first_ki_idx = engine._check_ki_barriers(
                    paths, ki_indices, ki_barriers, product.is_reverse
                )

        if product.barrier_config.disable_ko_after_ki and product.has_ki_barrier:
            ko_trigger_times = np.where(first_ko_idx >= 0, ko_times[first_ko_idx], np.inf)
            if len(ki_indices) > 0:
                ki_obs_times = all_times[ki_indices]
                ki_trigger_times = np.where(
                    first_ki_idx >= 0, ki_obs_times[first_ki_idx], np.inf
                )
            else:
                ki_trigger_times = np.full(len(paths), np.inf, dtype=float)
            ko_valid = ko_triggered & (ko_trigger_times < ki_trigger_times)
        else:
            ko_valid = ko_triggered

        is_ko = ko_valid
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        # Grid node of each path's first knock-out / knock-in (-1: decided at valuation; `never`: past the grid).
        n_paths = len(paths)
        never = len(all_times)
        ko_node = np.where(is_ko, ko_indices[np.maximum(first_ko_idx, 0)], never)
        if ki_continuous:
            # a crossing inside step k (or a breach at its end) lies in (t_{k-1}, t_k]: node k; at valuation: -1
            ki_node = np.where(ki_triggered, first_ki_idx, never)
            ki_node[ki_triggered & _ki_level_breached(paths[:, 0], ki_barrier, product.is_reverse)] = -1
            ki_first = ki_triggered & (ki_node <= ko_node)
        elif len(ki_indices) > 0:
            ki_node = np.where(ki_triggered, ki_indices[np.maximum(first_ki_idx, 0)], never)
            ki_first = ki_triggered & (ki_node < ko_node)        # a knock-out on the same node wins
        else:
            ki_node = np.full(n_paths, never, dtype=int)
            ki_first = np.zeros(n_paths, dtype=bool)

        # --- knock-out table: one row per KO observation, settled at its settlement time --------------------
        rows = np.arange(ko_times.size)
        ko_probability = np.array([float(np.mean(is_ko & (first_ko_idx == i))) for i in rows], dtype=float)
        ko_survival = np.array([float(np.mean(~(is_ko & (first_ko_idx <= i)))) for i in rows], dtype=float)
        ko_dfs = np.array([pricing_env.get_discount_factor(float(t)) for t in ko_settlement_times], dtype=float)
        ko_cash_to_pay = ko_probability * ko_payoffs
        observations = ObservationTable(
            times=ko_times, payment_times=ko_settlement_times,
            is_ko_obs=np.ones(rows.size, dtype=bool), is_coupon_obs=np.zeros(rows.size, dtype=bool),
            probability=ko_probability, survival=ko_survival,
            discounted_cash=ko_cash_to_pay * ko_dfs, cash_to_pay=ko_cash_to_pay,
            payment_is_path_dependent=np.zeros(rows.size, dtype=bool),
        )

        # --- terminal outcomes, paid at maturity: never knocked in (V0) / knocked in (V1) -----------------
        maturity_spots = paths[:, -1]
        maturity_df = pricing_env.get_discount_factor(float(maturity))
        maturity_payoff = np.zeros(n_paths, dtype=float)
        if is_v0.any():
            maturity_payoff[is_v0] = np.array(
                [product.get_maturity_payoff_v0(float(s), pricing_env) for s in maturity_spots[is_v0]],
                dtype=float,
            )
        if is_v1.any():
            maturity_payoff[is_v1] = np.array(
                [product.get_maturity_payoff_v1(float(s), pricing_env) for s in maturity_spots[is_v1]],
                dtype=float,
            )

        def outcome(mask: np.ndarray) -> OutcomeRow:
            cash = float(np.mean(np.where(mask, maturity_payoff, 0.0)))
            return OutcomeRow(time=float(maturity), payment_time=float(maturity), probability=float(np.mean(mask)),
                              discounted_cash=cash * maturity_df, cash_to_pay=cash)

        maturity_row = outcome(is_v0)
        ki_settlement = outcome(is_v1)

        # --- knock-in path statistics -----------------------------------------------------------------------
        unavailable: Dict[str, str] = {"standard_errors": "undefined: single batch"}
        # the knocked-in redemption is principal + downside (SnowballOption.get_maturity_payoff_v1): loss = -downside
        principal = (
            product.initial_price * product.contract_multiplier if product.payoff_config.include_principal else 0.0
        )
        expected_loss = float(np.mean(np.where(is_v1, principal - maturity_payoff, 0.0))) * maturity_df
        p_ki = float(np.mean(ki_first))
        ki_table: Optional[KnockInTable]
        expected_time: Optional[float] = None
        if not product.has_ki_barrier:
            ki_table = KnockInTable.empty()
        elif ki_continuous:
            ki_table = None
            unavailable["ki_table"] = CONTINUOUS_KI_REASON
            if p_ki > 0.0:
                unavailable["knock_in.expected_time"] = CONTINUOUS_KI_REASON
        else:
            nodes = np.unique(ki_indices)                       # the product's KI schedule on the grid
            node_times = np.where(nodes < 0, 0.0, all_times[np.maximum(nodes, 0)])
            first_here = ki_first[:, None] & (ki_node[:, None] == nodes[None, :])
            ended_by = (ki_first[:, None] & (ki_node[:, None] <= nodes[None, :])) | (
                is_ko[:, None] & (ko_node[:, None] <= nodes[None, :])
            )
            ki_probability = first_here.mean(axis=0)
            ki_table = KnockInTable(times=node_times, payment_times=node_times, probability=ki_probability,
                                    survival=(~ended_by).mean(axis=0), interpolated=np.zeros(nodes.size, dtype=bool))
            if p_ki > 0.0:
                expected_time = float(np.dot(node_times, ki_probability)) / p_ki
        if p_ki == 0.0:
            unavailable["knock_in.expected_time"] = "undefined: P(knock in) = 0"
        knock_in = KnockInStatistics(probability=p_ki, probability_no_ko=ki_settlement.probability,
                                     expected_time=expected_time, expected_loss=expected_loss)

        receivables = ReceivablesTable.empty()
        terminal_mass = maturity_row.probability + ki_settlement.probability
        return EventStatistics(
            product_family="snowball",
            pv=float(engine.price(product, pricing_env)),
            observations=observations,
            coupons=None,
            maturity=maturity_row,
            ki_settlement=ki_settlement,
            knock_in=knock_in,
            ki_table=ki_table,
            receivables=receivables,
            ledger=build_cash_ledger(observations=observations, coupons=None, maturity=maturity_row,
                                     ki_settlement=ki_settlement, receivables=receivables),
            expected_lifetime=float(np.dot(ko_probability, ko_times)) + terminal_mass * float(maturity),
            provenance=StatisticsProvenance(ANALYZER_STATISTICS_MODEL, type(engine).__name__),
            unavailable=unavailable,
        )

    def historical_shock_pnl(
        self,
        base_price: float,
        price_fn,
        *,
        spot_series: Sequence[float],
        q_series: Sequence[float],
        horizon_steps: int = 1,
    ) -> ShockPnLDistribution:
        """
        Historical simulation by applying historical shocks to today's state.

        Inputs:
        - spot_series: historical spot levels S_t
        - q_series: historical flat dividend yields q_t
        """
        spot = np.asarray(spot_series, dtype=float)
        q = np.asarray(q_series, dtype=float)
        if spot.shape != q.shape:
            raise ValidationError(
                f"spot_series and q_series must have same length, got {spot.shape} vs {q.shape}"
            )
        if spot.size <= horizon_steps:
            return ShockPnLDistribution(pnl=np.array([], dtype=float))

        spot_mult = spot[horizon_steps:] / spot[:-horizon_steps]
        q_shift = q[horizon_steps:] - q[:-horizon_steps]

        pnl = np.zeros(spot_mult.size, dtype=float)
        for i in range(spot_mult.size):
            shocked_price = float(price_fn(float(spot_mult[i]), float(q_shift[i])))
            pnl[i] = shocked_price - base_price
        return ShockPnLDistribution(pnl=pnl)

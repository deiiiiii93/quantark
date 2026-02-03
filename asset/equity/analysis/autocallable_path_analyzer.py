"""
Path-based analysis utilities for autocallable products (Snowball-first).

This module provides a Monte Carlo based analyzer that produces:
- risk-neutral event probabilities (KO/KI) per observation
- expected discounted cashflows and PV reconciliation
- historical replay (shock-based) and parametric scenario PnL distributions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.param import MCParams
from asset.equity.product.option.snowball_option import SnowballOption
from priceenv import PricingEnvironment
from util.enum import ObservationType
from util.exceptions import ValidationError


@dataclass(frozen=True)
class RiskNeutralSnowballEventStats:
    """Risk-neutral event stats and cashflow attribution for a SnowballOption."""

    pv_mc: float
    std_error: float
    num_paths: int

    ko_times: np.ndarray
    ko_prob: np.ndarray
    survive_prob: np.ndarray
    expected_discounted_ko_cf: np.ndarray

    ki_probability: float
    expected_discounted_maturity_cf: float

    reconciliation_error: float


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
    ) -> RiskNeutralSnowballEventStats:
        """
        Compute risk-neutral event probabilities and expected discounted cashflows.

        Notes:
        - Uses the SnowballMCEngine's time grid builder and barrier checks.
        - PV is computed as the expected discounted cashflows implied by simulated paths.
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

        ko_prob = np.zeros(len(ko_times), dtype=float)
        expected_discounted_ko_cf = np.zeros(len(ko_times), dtype=float)
        for i in range(len(ko_times)):
            hit_i = is_ko & (first_ko_idx == i)
            ko_prob[i] = float(np.mean(hit_i))
            if hit_i.any():
                df = pricing_env.get_discount_factor(float(ko_settlement_times[i]))
                expected_discounted_ko_cf[i] = float(np.mean(hit_i) * ko_payoffs[i] * df)

        survive_prob = np.ones(len(ko_times), dtype=float)
        cumulative_ko = 0.0
        for i in range(len(ko_times)):
            cumulative_ko += ko_prob[i]
            survive_prob[i] = max(0.0, 1.0 - cumulative_ko)

        maturity_spots = paths[:, -1]
        maturity_df = pricing_env.get_discount_factor(float(maturity))

        # Compute ensured maturity expected CF as mean over all paths for stability.
        maturity_payoff_all = np.zeros(len(paths), dtype=float)
        if is_v0.any():
            maturity_payoff_all[is_v0] = np.array(
                [
                    product.get_maturity_payoff_v0(float(s), pricing_env=pricing_env)
                    for s in maturity_spots[is_v0]
                ],
                dtype=float,
            )
        if is_v1.any():
            maturity_payoff_all[is_v1] = np.array(
                [product.get_maturity_payoff_v1(float(s), pricing_env) for s in maturity_spots[is_v1]],
                dtype=float,
            )
        expected_discounted_maturity_cf = float(np.mean(maturity_payoff_all * maturity_df))

        pv_mc = float(np.sum(expected_discounted_ko_cf) + expected_discounted_maturity_cf)

        # Standard error from discounted total payoff.
        total_discounted = np.zeros(len(paths), dtype=float)
        if is_ko.any():
            payoff = ko_payoffs[first_ko_idx[is_ko]]
            settle = ko_settlement_times[first_ko_idx[is_ko]]
            df = np.array([pricing_env.get_discount_factor(float(t)) for t in settle], dtype=float)
            total_discounted[is_ko] = payoff * df
        total_discounted[~is_ko] = maturity_payoff_all[~is_ko] * maturity_df

        std_error = float(np.std(total_discounted, ddof=1) / np.sqrt(len(paths)))

        engine_price = engine.price(product, pricing_env)
        reconciliation_error = float(pv_mc - engine_price)

        return RiskNeutralSnowballEventStats(
            pv_mc=pv_mc,
            std_error=std_error,
            num_paths=int(len(paths)),
            ko_times=ko_times,
            ko_prob=ko_prob,
            survive_prob=survive_prob,
            expected_discounted_ko_cf=expected_discounted_ko_cf,
            ki_probability=float(np.mean(ki_triggered)),
            expected_discounted_maturity_cf=expected_discounted_maturity_cf,
            reconciliation_error=reconciliation_error,
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

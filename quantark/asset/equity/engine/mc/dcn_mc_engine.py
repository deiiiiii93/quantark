"""DCN Monte Carlo engine (spec WP1.3): curve-aware GBM on the SSE daily grid.

Time grid = the contract's actual trading-day grid (ACT/365F), so daily KI is
exact discrete monitoring by construction — NO barrier continuity correction.
Each cash flow discounts at its own payment time (incl. the loss leg at the
independent settlement_date). PV and all legs are signed by direction_sign,
and pv is DEFINED as the sum of the signed legs (exact invariant).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.event_stats import DCNEventStats
from quantark.asset.equity.engine.mc.dcn_payoff import compute_dcn_cashflows
from quantark.asset.equity.engine.mc.qmc_draws import qmc_normals
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.exceptions import PricingError, ValidationError


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
    event_stats: Optional[DCNEventStats] = field(repr=False, default=None)

    def to_dict(self) -> dict:
        d = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in self.__dict__.items()
            if k != "event_stats"
        }
        d["event_stats"] = (
            self.event_stats.to_dict() if self.event_stats is not None else None
        )
        return d


class DCNMCEngine:
    """GBM MC engine for DCNOption; flat markets are just flat curves.

    Sobol QMC by default (fixed seed => bit-reproducible in the same
    environment). Subclasses override ``_simulate`` only (path generation);
    payoff, discounting, leg decomposition and event stats are shared.
    """

    def __init__(
        self,
        num_paths: int = 131072,
        seed: int = 42,
        use_sobol: bool = True,
        use_antithetic: bool = False,
    ):
        if use_sobol and use_antithetic:
            raise ValidationError("Sobol QMC and antithetic are mutually exclusive")
        if num_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {num_paths}")
        self.num_paths = int(num_paths)
        self.seed = int(seed)
        self.use_sobol = bool(use_sobol)
        self.use_antithetic = bool(use_antithetic)
        self._last_result: Optional[DCNMCResult] = None

    # -- BaseEngine-compatible entry point --
    def price(self, product, pricing_env) -> float:
        return self.price_detailed(product, pricing_env).pv

    def price_detailed(self, product, pricing_env) -> DCNMCResult:
        if not isinstance(product, DCNOption):
            raise PricingError(
                f"{type(self).__name__} only supports DCNOption, "
                f"got {type(product).__name__}"
            )
        t0 = time.perf_counter()
        ctx = build_dcn_grid_context(product)
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

        paths = self._simulate(spot0, term, dt_array, pricing_env)
        cf = compute_dcn_cashflows(paths, product, ctx, df)

        sign = product.direction_sign
        totals = sign * cf.total_pv
        n = totals.size
        stderr = float(totals.std(ddof=1) / np.sqrt(n))
        # legs first; pv is DEFINED as their sum (exact invariant)
        pv_fixed = float(sign * cf.fixed_coupon_pv.mean())
        pv_ko = float(sign * cf.ko_pv.mean())
        pv_loss = float(sign * cf.loss_pv.mean())
        pv = pv_fixed + pv_ko + pv_loss

        n_obs = ctx.obs_cols.size
        ko_mask = cf.ko_obs_row >= 0
        ko_timing = np.array([(cf.ko_obs_row == j).mean() for j in range(n_obs)])
        # life ends at the KO observation, else the final observation
        obs_times = ctx.times[ctx.obs_cols]
        life = np.where(
            ko_mask, obs_times[np.maximum(cf.ko_obs_row, 0)], obs_times[-1]
        )
        survive = ~ko_mask
        stats = DCNEventStats(
            ki_probability=float(cf.knocked_in.mean()),
            ko_probability=float(ko_mask.mean()),
            ko_timing_distribution=tuple(float(x) for x in ko_timing),
            coupon_probability=tuple(
                float(x) for x in cf.coupon_paid.mean(axis=0)
            ),
            expected_life_years=float(life.mean()),
            prob_survive_no_ki=float((survive & ~cf.knocked_in).mean()),
            prob_survive_ki=float((survive & cf.knocked_in).mean()),
            expected_discounted_loss_leg=pv_loss,
        )
        result = DCNMCResult(
            pv=pv,
            std_error=stderr,
            num_paths=n,
            seed=self.seed,
            elapsed_seconds=time.perf_counter() - t0,
            direction_sign=sign,
            pv_fixed_coupons=pv_fixed,
            pv_fixed_coupons_by_period=tuple(
                float(sign * x) for x in cf.fixed_coupon_pv_by_period.mean(axis=0)
            ),
            pv_ko_coupons=pv_ko,
            pv_ko_coupons_by_period=tuple(
                float(sign * cf.ko_pv[cf.ko_obs_row == j].sum() / n)
                for j in range(n_obs)
            ),
            pv_loss_leg=pv_loss,
            ki_probability=stats.ki_probability,
            ko_probability=stats.ko_probability,
            ko_timing_distribution=stats.ko_timing_distribution,
            coupon_probability=stats.coupon_probability,
            expected_life_years=stats.expected_life_years,
            prob_survive_no_ki=stats.prob_survive_no_ki,
            prob_survive_ki=stats.prob_survive_ki,
            event_stats=stats,
        )
        self._last_result = result
        return result

    # -- path generation (overridable; base = term-aware GBM) --
    def _draws(self, n_dims: int) -> np.ndarray:
        if self.use_sobol:
            return qmc_normals(self.seed, self.num_paths, n_dims, batch_id=None)
        rng = np.random.default_rng(self.seed)
        if self.use_antithetic:
            half = (self.num_paths + 1) // 2
            z = rng.standard_normal((half, n_dims))
            return np.vstack([z, -z])[: self.num_paths]
        return rng.standard_normal((self.num_paths, n_dims))

    def _simulate(self, spot0, term, dt_array, pricing_env) -> np.ndarray:
        n_steps = dt_array.size
        z_all = self._draws(n_steps)
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

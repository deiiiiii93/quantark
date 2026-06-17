"""
Monte Carlo engine for FX range accruals under Garman-Kohlhagen dynamics.

This is the simulation counterpart to
:class:`~quantark.asset.fx.engine.analytical.fx_range_accrual_engine.FxRangeAccrualAnalyticalEngine`.
It serves two purposes:

1. **Cross-validation** — under a flat (Black-Scholes) surface the analytical
   range-digital decomposition and this simulation must agree to within Monte
   Carlo error, for both per-fixing analytical methods (which themselves
   coincide under flat vol). This is the correctness gate for the whole FX
   range-accrual family.
2. **Foundation for path-dependent variants** — knock-out, target-redemption,
   memory and callable accruals couple the fixings and so break the linearity
   of expectation the analytical engine relies on. They need a path model; this
   engine is that base.

Model scope (deliberately bounded — no silent approximation): the dynamics are
**constant-volatility Garman-Kohlhagen**,

    S_{t_i} = F(0, t_i) * exp(-0.5 * sigma^2 * t_i + sigma * W_{t_i}),

anchored on the parity forward ``F(0, t_i)`` so the simulated rate matches the
curve forward exactly at every fixing for an arbitrary rate term structure. The
single ``sigma`` is sampled from the surface at the at-the-money spot for the
product maturity. Under a flat surface this is exact; under a Vanna-Volga smile
it is a *flat-vol* model — smile-consistent path pricing belongs to the FX
local-vol / SLV / Heston MC engines, not here.

Range accruals are *discretely monitored*: the spot only matters at the fixing
times, so there is no time-discretization error to control — the GBM is sampled
exactly on the (possibly non-uniform) fixing grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.process.garman_kohlhagen_process import GarmanKohlhagenProcess
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_range_accrual_option import (
    FxRangeAccrualOption,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import CouponPayType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero


@dataclass
class FxRangeAccrualMCResult:
    """Full breakdown of an FX range accrual Monte Carlo run."""

    price: float
    std_error: float
    num_paths: int
    in_range_ratio_mean: float
    in_range_ratio_std: float
    past_in_range_weights: float
    future_expected_in_range_weights: float
    total_weights: float
    num_past_observations: int
    num_future_observations: int
    sigma: float


class FxRangeAccrualMCEngine(BaseFxEngine):
    """
    Constant-volatility Garman-Kohlhagen Monte Carlo for FX range accruals.

    Prices the plain domestic-coupon :class:`FxRangeAccrualOption` only (the
    quanto / foreign subclasses price under a different measure and are left to
    a future MC extension). Supports weighted and per-fixing barriers, settled
    past fixings, reverse-range mode and EXPIRY / INSTANT coupon timing, matching
    the analytical engine's conventions exactly so the two are directly
    comparable.

    A fixed seed gives common random numbers across bump-and-reprice, so the
    inherited finite-difference Greeks are low-noise.
    """

    engine_type = EngineType.MONTE_CARLO

    def __init__(
        self,
        num_paths: int = 200_000,
        seed: int = 42,
        use_antithetic: bool = True,
        use_qmc: bool = False,
        params: Optional[FxEngineParams] = None,
    ):
        super().__init__(params)
        if num_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {num_paths}")
        self.num_paths = int(num_paths)
        self.seed = int(seed)
        self.use_antithetic = bool(use_antithetic)
        self.use_qmc = bool(use_qmc)
        self._last_result: Optional[FxRangeAccrualMCResult] = None

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """Price an FX range accrual by simulation; breakdown via
        :meth:`get_last_result`."""
        option = self._check_product(product)
        config = option.range_config

        spot = fx_env.effective_spot()
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")

        T = option.get_maturity(fx_env)
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")

        T_pay = option.get_delivery(fx_env)
        year_fraction = option.get_year_fraction(fx_env)
        scale = option.notional * config.accrual_rate * year_fraction
        df_pay = fx_env.get_domestic_df(T_pay)

        # Near-expiry: all fixings are settled; the coupon is deterministic.
        if is_zero(T):
            past_in_range, _ = option.get_past_accrual(fx_env)
            total = option.get_total_weights()
            ratio = 0.0 if is_zero(total) else past_in_range / total
            price = scale * ratio * df_pay
            self._last_result = FxRangeAccrualMCResult(
                price=price,
                std_error=0.0,
                num_paths=0,
                in_range_ratio_mean=ratio,
                in_range_ratio_std=0.0,
                past_in_range_weights=past_in_range,
                future_expected_in_range_weights=0.0,
                total_weights=total,
                num_past_observations=0,
                num_future_observations=0,
                sigma=0.0,
            )
            return price

        past_obs, future_obs, total_weights = option.resolve_observations(fx_env)
        if is_zero(total_weights):
            raise ValidationError("Total observation weight must be positive")

        instant = config.accrual_pay_type == CouponPayType.INSTANT
        past_in_range = sum(w for w, ok in past_obs if ok)
        # Under EXPIRY a settled fixing still folds into the final coupon
        # (discounted from the pay date); under INSTANT it was already paid.
        past_term = 0.0 if instant else past_in_range * df_pay

        # No future fixings left: deterministic, no simulation needed.
        if len(future_obs) == 0:
            ratio = past_in_range / total_weights
            price = scale * past_term / total_weights
            self._last_result = FxRangeAccrualMCResult(
                price=price,
                std_error=0.0,
                num_paths=0,
                in_range_ratio_mean=ratio,
                in_range_ratio_std=0.0,
                past_in_range_weights=past_in_range,
                future_expected_in_range_weights=0.0,
                total_weights=total_weights,
                num_past_observations=len(past_obs),
                num_future_observations=0,
                sigma=0.0,
            )
            return price

        sigma = self._reference_vol(fx_env, spot, T)
        spots = self._simulate_fixings(fx_env, future_obs, sigma)

        in_range = self._in_range_mask(option, spots, future_obs)  # (P, M) bool
        fut_w = np.array([w for w, _, _ in future_obs])
        if instant:
            disc = np.array(
                [fx_env.get_domestic_df(t) for _, t, _ in future_obs]
            )
        else:
            disc = np.full(len(future_obs), df_pay)

        # Per-path discounted in-range weight (future leg) and undiscounted
        # in-range fraction (for reporting / variance diagnostics).
        path_future_disc = (in_range * (fut_w * disc)[None, :]).sum(axis=1)
        path_future_weight = (in_range * fut_w[None, :]).sum(axis=1)

        pv_per_path = scale * (past_term + path_future_disc) / total_weights
        ratio_per_path = (past_in_range + path_future_weight) / total_weights

        price = float(pv_per_path.mean())
        std_error = float(pv_per_path.std(ddof=1)) / math.sqrt(pv_per_path.size)
        if price < 0:
            raise PricingError(f"Negative price computed: {price}")

        self._last_result = FxRangeAccrualMCResult(
            price=price,
            std_error=std_error,
            num_paths=int(pv_per_path.size),
            in_range_ratio_mean=float(ratio_per_path.mean()),
            in_range_ratio_std=float(ratio_per_path.std(ddof=1)),
            past_in_range_weights=past_in_range,
            future_expected_in_range_weights=float(path_future_weight.mean()),
            total_weights=total_weights,
            num_past_observations=len(past_obs),
            num_future_observations=len(future_obs),
            sigma=sigma,
        )
        return price

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _reference_vol(
        self, fx_env: FxPricingEnvironment, spot: float, T: float
    ) -> float:
        """The single GK volatility: the surface at the at-the-money spot.

        Exact for a flat surface; a flat-vol representative under a smile.
        """
        sigma = fx_env.get_vol(spot, T)
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        return float(sigma)

    def _simulate_fixings(
        self,
        fx_env: FxPricingEnvironment,
        future_obs: List[Tuple[float, float, int]],
        sigma: float,
    ) -> np.ndarray:
        """
        Simulate the FX rate at each future fixing time.

        Exact discrete GBM on the (sorted, possibly non-uniform) fixing grid,
        anchored on the parity forward so ``E[S_{t_i}] = F(0, t_i)`` for any
        rate term structure:

            S_{t_i} = F(0, t_i) * exp(-0.5 sigma^2 t_i + sigma W_{t_i}),
            W_{t_i} = sum_{k<=i} sqrt(t_k - t_{k-1}) Z_k.

        Returns an array of shape ``(num_paths, num_future_obs)``.
        """
        times = np.array([t for _, t, _ in future_obs], dtype=float)
        prev = np.concatenate(([0.0], times[:-1]))
        dt = times - prev
        if np.any(dt < 0.0):
            raise ValidationError("future fixing times must be non-decreasing")

        # Reuse the FX process's variance-reduction-aware normal generator
        # (antithetic / scrambled Sobol) so the whole FX MC stack draws
        # consistently. Shape (num_paths, num_steps).
        normals = GarmanKohlhagenProcess._generate_normals(
            time_steps=times.size,
            num_paths=self.num_paths,
            seed=self.seed,
            antithetic=self.use_antithetic,
            use_qmc=self.use_qmc,
        )
        brownian = np.cumsum(normals * np.sqrt(dt)[None, :], axis=1)

        forwards = np.array([fx_env.get_forward(t) for t in times])
        drift = -0.5 * sigma * sigma * times
        log_spots = np.log(forwards)[None, :] + drift[None, :] + sigma * brownian
        return np.exp(log_spots)

    def _in_range_mask(
        self,
        option: FxRangeAccrualOption,
        spots: np.ndarray,
        future_obs: List[Tuple[float, float, int]],
    ) -> np.ndarray:
        """Boolean ``(num_paths, num_future_obs)`` mask of accruing fixings.

        Uses the product's resolved ``[lower, upper]`` per fixing (config-level,
        matching the analytical engine) and honours reverse-range mode.
        """
        lowers = np.empty(len(future_obs))
        uppers = np.empty(len(future_obs))
        for i, (_, _, obs_idx) in enumerate(future_obs):
            lowers[i], uppers[i] = option.barriers_for(obs_idx)
        inside = (spots >= lowers[None, :]) & (spots <= uppers[None, :])
        return ~inside if option.range_config.is_reverse else inside

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def calculate_greeks(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> Dict[str, float]:
        """FX Greeks by central bump-and-reprice (base-class conventions).

        The fixed seed gives common random numbers across bumps, so the
        finite-difference noise largely cancels.
        """
        self._check_product(product)
        return super().calculate_greeks(product, fx_env)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxRangeAccrualOption:
        if not isinstance(product, FxRangeAccrualOption):
            raise PricingError(
                f"FxRangeAccrualMCEngine only supports FxRangeAccrualOption, "
                f"got {type(product).__name__}"
            )
        # Quanto / foreign coupons price under a different measure or currency.
        if type(product) is not FxRangeAccrualOption:
            raise PricingError(
                f"FxRangeAccrualMCEngine prices plain domestic-coupon range "
                f"accruals; {type(product).__name__} needs its own MC engine"
            )
        if product.range_config is None:
            raise ValidationError("range_config is required")
        return product

    def get_last_result(self) -> Optional[FxRangeAccrualMCResult]:
        """Full breakdown from the most recent :meth:`price` call."""
        return self._last_result

    def get_last_std_error(self) -> Optional[float]:
        """Standard error of the most recent pricing run."""
        return None if self._last_result is None else self._last_result.std_error

    def __repr__(self):
        return (
            f"FxRangeAccrualMCEngine(num_paths={self.num_paths}, "
            f"use_qmc={self.use_qmc})"
        )

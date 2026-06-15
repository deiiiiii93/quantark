"""Shared orchestration for trade-level structured volatility-model risk."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np

from quantark.param import GridVolSurface
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.volmodels.curves import forward_carry_on_grid, forward_rates_on_grid
from quantark.volmodels.heston import HestonParams, MarketOption, calibrate_heston
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol
from quantark.volmodels.risk import (
    HestonCalibrationSpec,
    MarketVegaRequest,
    ModelRiskRequest,
    SlvCalibrationSpec,
    SlvLeverageRiskMode,
    VolRiskPoint,
    VolRiskResult,
    bump_grid_vol_surface,
    bump_heston_parameter,
    bump_leverage_surface,
    bump_local_vol_surface,
)
from quantark.volmodels.slv import BinMethod, LeverageSurface, calibrate_leverage_surface


_HESTON_NAMES = ("v0", "kappa", "theta", "sigma", "rho")


class BaseVolModelRiskCalculator:
    """Asset-aware implementation behind the equity and FX public calculators."""

    def __init__(
        self,
        *,
        is_fx: bool,
        heston_calibration_spec: Optional[HestonCalibrationSpec] = None,
        slv_calibration_spec: Optional[SlvCalibrationSpec] = None,
    ) -> None:
        self._is_fx = is_fx
        self.heston_calibration_spec = heston_calibration_spec or HestonCalibrationSpec()
        self.slv_calibration_spec = slv_calibration_spec or SlvCalibrationSpec()

    def calculate_model_risk(
        self, product, env, engine, request: Optional[ModelRiskRequest] = None
    ) -> VolRiskResult:
        request = request or ModelRiskRequest()
        self._validate_asset(engine, env)
        if self._is_local(engine):
            return self._local_model_risk(product, env, engine, request)
        if self._is_heston(engine):
            return self._heston_model_risk(product, env, engine, request)
        if self._is_slv(engine):
            return self._slv_model_risk(product, env, engine, request)
        raise ValidationError(f"unsupported vol-model engine: {type(engine).__name__}")

    def calculate_market_vega(
        self, product, env, engine, request: Optional[MarketVegaRequest] = None
    ) -> VolRiskResult:
        request = request or MarketVegaRequest()
        self._validate_asset(engine, env)
        self._grid_surface(env)
        if self._is_local(engine):
            return self._local_market_vega(product, env, engine, request)
        if self._is_heston(engine):
            return self._heston_market_vega(product, env, engine, request)
        if self._is_slv(engine):
            return self._slv_market_vega(product, env, engine, request)
        raise ValidationError(f"unsupported vol-model engine: {type(engine).__name__}")

    def _validate_asset(self, engine, env) -> None:
        if self._is_fx:
            if not isinstance(env, FxPricingEnvironment):
                raise ValidationError("FX vol-model risk requires FxPricingEnvironment")
        else:
            if not isinstance(env, PricingEnvironment):
                raise ValidationError("equity vol-model risk requires PricingEnvironment")
        module = type(engine).__module__
        expected = "quantark.asset.fx." if self._is_fx else "quantark.asset.equity."
        if not module.startswith(expected) or not (
            self._is_local(engine) or self._is_heston(engine) or self._is_slv(engine)
        ):
            raise ValidationError(
                f"{'FX' if self._is_fx else 'equity'} calculator does not support "
                f"{type(engine).__name__}"
            )
        seed = getattr(engine, "seed", getattr(getattr(engine, "params", None), "seed", 42))
        if seed is None:
            raise ValidationError("structured risk requires a fixed MC seed")

    def _is_local(self, engine) -> bool:
        # Structural dispatch avoids importing both eager asset packages here, which
        # would create a riskmeasures import cycle. Asset-module validation above keeps
        # this intentionally narrow to QuantArk's supported equity/FX engine families.
        return callable(getattr(engine, "_price_with_surface", None)) and not hasattr(
            engine, "model_params"
        )

    def _is_heston(self, engine) -> bool:
        return isinstance(getattr(engine, "model_params", None), HestonParams) and not hasattr(
            engine, "eta"
        )

    def _is_slv(self, engine) -> bool:
        return isinstance(getattr(engine, "model_params", None), HestonParams) and hasattr(
            engine, "eta"
        ) and (
            hasattr(engine, "leverage_surface")
            or hasattr(engine, "_prebuilt_leverage")
        )

    @staticmethod
    def _grid_surface(env) -> GridVolSurface:
        if not isinstance(env.vol_surface, GridVolSurface):
            raise PricingError("market vega requires a GridVolSurface in the pricing environment")
        return env.vol_surface

    def _build_local_vol(self, env, surface: GridVolSurface) -> LocalVolSurface:
        if self._is_fx:
            return build_dupire_local_vol(
                surface,
                spot=float(env.effective_spot()),
                rate_curve=env.domestic_curve,
                div_yield=env.get_foreign_rate,
            )
        return build_dupire_local_vol(
            surface, spot=float(env.spot), rate_curve=env.rate_curve, div_yield=env.get_div_yield
        )

    def _resolve_local_vol(self, env, engine) -> LocalVolSurface:
        prebuilt = getattr(engine, "_prebuilt", None)
        if prebuilt is not None:
            return prebuilt
        return self._build_local_vol(env, self._grid_surface(env))

    @staticmethod
    def _finite_price(value) -> float:
        price = float(value)
        if not np.isfinite(price):
            raise NumericalError(f"scenario produced non-finite price: {price}")
        return price

    @staticmethod
    def _price_local(engine, product, env, surface: LocalVolSurface) -> float:
        return BaseVolModelRiskCalculator._finite_price(
            engine._price_with_surface(product, env, surface)
        )

    @staticmethod
    def _price_heston(engine, product, env, params: HestonParams) -> float:
        bumped_engine = deepcopy(engine)
        bumped_engine.model_params = params
        return BaseVolModelRiskCalculator._finite_price(bumped_engine.price(product, env))

    @staticmethod
    def _central_point(
        name: str,
        bump_size: float,
        base_price: float,
        price_up: Callable[[], float],
        price_down: Callable[[], float],
        allow_one_sided: bool = False,
    ) -> VolRiskPoint:
        up_price = None
        down_price = None
        up_error = None
        down_error = None
        try:
            up_price = BaseVolModelRiskCalculator._finite_price(price_up())
        except Exception as exc:  # each scenario is intentionally isolated
            up_error = exc
        try:
            down_price = BaseVolModelRiskCalculator._finite_price(price_down())
        except Exception as exc:  # each scenario is intentionally isolated
            down_error = exc
        if up_error is None and down_error is None:
            try:
                return VolRiskPoint.success(name, bump_size, base_price, up_price, down_price)
            except Exception as exc:
                return VolRiskPoint.failed(
                    name, bump_size, base_price, f"finite difference: {exc}", up_price, down_price
                )
        if allow_one_sided and up_error is None:
            try:
                return VolRiskPoint.success(
                    name, bump_size, base_price, up_price, None, difference_mode="one_sided_up"
                )
            except Exception as exc:
                up_error = exc
        if allow_one_sided and down_error is None:
            try:
                return VolRiskPoint.success(
                    name, bump_size, base_price, None, down_price, difference_mode="one_sided_down"
                )
            except Exception as exc:
                down_error = exc
        errors = []
        if up_error is not None:
            errors.append(f"up: {up_error}")
        if down_error is not None:
            errors.append(f"down: {down_error}")
        return VolRiskPoint.failed(
            name, bump_size, base_price, "; ".join(errors), up_price, down_price
        )

    def _local_model_risk(self, product, env, engine, request: ModelRiskRequest) -> VolRiskResult:
        local_vol = self._resolve_local_vol(env, engine)
        base_price = self._price_local(engine, product, env, local_vol)
        points = []
        for bump in request.surface_bumps:
            points.append(
                self._central_point(
                    f"local_vol.{bump.label}",
                    bump.bump_size,
                    base_price,
                    lambda b=bump: self._price_local(
                        engine, product, env, bump_local_vol_surface(local_vol, b, 1)
                    ),
                    lambda b=bump: self._price_local(
                        engine, product, env, bump_local_vol_surface(local_vol, b, -1)
                    ),
                    request.allow_one_sided,
                )
            )
        return VolRiskResult(base_price, tuple(points), {"model": "dupire_local_vol"})

    def _heston_model_risk(
        self, product, env, engine, request: ModelRiskRequest
    ) -> VolRiskResult:
        params = engine.model_params
        base_price = self._price_heston(engine, product, env, params)
        points = []
        for name in request.parameter_names:
            if name not in _HESTON_NAMES:
                continue
            bump_size = self._heston_bump_size(params, name, request)
            points.append(
                self._central_point(
                    f"heston.{name}",
                    bump_size,
                    base_price,
                    lambda n=name: self._price_heston(
                        engine, product, env, self._bump_heston(params, n, 1, request)
                    ),
                    lambda n=name: self._price_heston(
                        engine, product, env, self._bump_heston(params, n, -1, request)
                    ),
                    request.allow_one_sided,
                )
            )
        return VolRiskResult(base_price, tuple(points), {"model": "heston"})

    @staticmethod
    def _heston_bump_size(params: HestonParams, name: str, request: ModelRiskRequest) -> float:
        _, bump_size = bump_heston_parameter(
            params,
            name,
            1,
            request.relative_parameter_bump,
            request.variance_bump_floor,
            request.positive_parameter_bump_floor,
            request.rho_bump,
        )
        return bump_size

    @staticmethod
    def _bump_heston(
        params: HestonParams, name: str, direction: int, request: ModelRiskRequest
    ) -> HestonParams:
        bumped, _ = bump_heston_parameter(
            params,
            name,
            direction,
            request.relative_parameter_bump,
            request.variance_bump_floor,
            request.positive_parameter_bump_floor,
            request.rho_bump,
        )
        return bumped

    def _slv_model_risk(self, product, env, engine, request: ModelRiskRequest) -> VolRiskResult:
        if request.slv_leverage_mode is None:
            raise ValidationError("SLV model risk requires an explicit slv_leverage_mode")
        base_leverage = self._existing_leverage(engine)
        local_vol = getattr(engine, "_prebuilt", None)
        if request.slv_leverage_mode == SlvLeverageRiskMode.RECALIBRATE or base_leverage is None:
            local_vol = self._resolve_local_vol(env, engine)
        if request.slv_leverage_mode == SlvLeverageRiskMode.RECALIBRATE or base_leverage is None:
            base_leverage = self._calibrate_leverage(
                product, env, engine.model_params, engine.eta, local_vol
            )
        base_price = self._price_slv(
            engine, product, env, engine.model_params, engine.eta, local_vol, base_leverage
        )
        points = []
        for name in request.parameter_names:
            if name in _HESTON_NAMES:
                bump_size = self._heston_bump_size(engine.model_params, name, request)
                points.append(
                    self._central_point(
                        f"slv.{name}",
                        bump_size,
                        base_price,
                        lambda n=name: self._price_slv_parameter(
                            engine, product, env, local_vol, base_leverage, request, n, 1
                        ),
                        lambda n=name: self._price_slv_parameter(
                            engine, product, env, local_vol, base_leverage, request, n, -1
                        ),
                        request.allow_one_sided,
                    )
                )
            elif name == "eta":
                bump_size = max(
                    abs(float(engine.eta)) * request.relative_parameter_bump,
                    request.positive_parameter_bump_floor,
                )
                points.append(
                    self._central_point(
                        "slv.eta",
                        bump_size,
                        base_price,
                        lambda: self._price_slv_eta(
                            engine, product, env, local_vol, base_leverage, request, bump_size
                        ),
                        lambda: self._price_slv_eta(
                            engine, product, env, local_vol, base_leverage, request, -bump_size
                        ),
                        request.allow_one_sided,
                    )
                )
        for bump in request.surface_bumps:
            points.append(
                self._central_point(
                    f"leverage.{bump.label}",
                    bump.bump_size,
                    base_price,
                    lambda b=bump: self._price_slv(
                        engine,
                        product,
                        env,
                        engine.model_params,
                        engine.eta,
                        local_vol,
                        bump_leverage_surface(base_leverage, b, 1),
                    ),
                    lambda b=bump: self._price_slv(
                        engine,
                        product,
                        env,
                        engine.model_params,
                        engine.eta,
                        local_vol,
                        bump_leverage_surface(base_leverage, b, -1),
                    ),
                    request.allow_one_sided,
                )
            )
        return VolRiskResult(
            base_price,
            tuple(points),
            {
                "model": "heston_slv",
                "slv_leverage_mode": request.slv_leverage_mode.value,
            },
        )

    def _price_slv_parameter(
        self, engine, product, env, local_vol, base_leverage, request, name, direction
    ) -> float:
        params = self._bump_heston(engine.model_params, name, direction, request)
        leverage = base_leverage
        if request.slv_leverage_mode == SlvLeverageRiskMode.RECALIBRATE:
            leverage = self._calibrate_leverage(product, env, params, engine.eta, local_vol)
        return self._price_slv(engine, product, env, params, engine.eta, local_vol, leverage)

    def _price_slv_eta(
        self, engine, product, env, local_vol, base_leverage, request, eta_change
    ) -> float:
        eta = float(engine.eta) + eta_change
        if eta < 0:
            raise ValidationError("eta must be non-negative")
        leverage = base_leverage
        if request.slv_leverage_mode == SlvLeverageRiskMode.RECALIBRATE:
            leverage = self._calibrate_leverage(product, env, engine.model_params, eta, local_vol)
        return self._price_slv(engine, product, env, engine.model_params, eta, local_vol, leverage)

    @staticmethod
    def _price_slv(engine, product, env, params, eta, local_vol, leverage) -> float:
        bumped_engine = deepcopy(engine)
        bumped_engine.model_params = params
        bumped_engine.eta = eta
        if callable(getattr(bumped_engine, "_price_with_artifacts", None)):
            return BaseVolModelRiskCalculator._finite_price(
                bumped_engine._price_with_artifacts(product, env, local_vol, leverage)
            )
        bumped_engine.leverage_surface = leverage
        return BaseVolModelRiskCalculator._finite_price(bumped_engine.price(product, env))

    @staticmethod
    def _existing_leverage(engine) -> Optional[LeverageSurface]:
        if callable(getattr(engine, "_price_with_artifacts", None)):
            return getattr(engine, "_prebuilt_leverage", None)
        return engine.leverage_surface

    def _calibrate_leverage(
        self, product, env, params: HestonParams, eta: float, local_vol: LocalVolSurface
    ) -> LeverageSurface:
        spec = self.slv_calibration_spec
        maturity = float(product.get_maturity(env))
        if maturity <= 0:
            raise ValidationError("maturity must be positive")
        t_grid = np.linspace(0.0, maturity, spec.time_steps + 1)
        if self._is_fx:
            r_fwd = forward_rates_on_grid(env.domestic_curve, t_grid)
            carry_fwd = forward_rates_on_grid(env.foreign_curve, t_grid)
            spot = float(env.effective_spot())
        else:
            r_fwd = forward_rates_on_grid(env.rate_curve, t_grid)
            carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
            spot = float(env.spot)
        return calibrate_leverage_surface(
            spot,
            params,
            local_vol,
            np.diff(t_grid),
            r_fwd,
            carry_fwd,
            eta=eta,
            num_paths=spec.num_paths,
            num_bins=spec.num_bins,
            bin_method=BinMethod(spec.bin_method),
            seed=spec.seed,
            n_strike_nodes=spec.n_strike_nodes,
            strike_span_stds=spec.strike_span_stds,
        )

    def _local_market_vega(
        self, product, env, engine, request: MarketVegaRequest
    ) -> VolRiskResult:
        quote_surface = self._grid_surface(env)
        base_local = self._build_local_vol(env, quote_surface)
        base_price = self._price_local(engine, product, env, base_local)
        points = []
        for bump in request.surface_bumps:
            points.append(
                self._central_point(
                    f"market_iv.{bump.label}",
                    bump.bump_size,
                    base_price,
                    lambda b=bump: self._price_local(
                        engine,
                        product,
                        env,
                        self._build_local_vol(env, bump_grid_vol_surface(quote_surface, b, 1)),
                    ),
                    lambda b=bump: self._price_local(
                        engine,
                        product,
                        env,
                        self._build_local_vol(env, bump_grid_vol_surface(quote_surface, b, -1)),
                    ),
                )
            )
        return VolRiskResult(base_price, tuple(points), {"model": "dupire_local_vol"})

    def _heston_market_vega(
        self, product, env, engine, request: MarketVegaRequest
    ) -> VolRiskResult:
        surface = self._grid_surface(env)
        initial = self.heston_calibration_spec.initial_params or engine.model_params
        base_params = self._calibrate_heston(env, surface, initial)
        base_price = self._price_heston(engine, product, env, base_params)
        points = []
        for bump in request.surface_bumps:
            points.append(
                self._central_point(
                    f"market_iv.{bump.label}",
                    bump.bump_size,
                    base_price,
                    lambda b=bump: self._price_heston(
                        engine,
                        product,
                        env,
                        self._calibrate_heston(
                            env, bump_grid_vol_surface(surface, b, 1), base_params
                        ),
                    ),
                    lambda b=bump: self._price_heston(
                        engine,
                        product,
                        env,
                        self._calibrate_heston(
                            env, bump_grid_vol_surface(surface, b, -1), base_params
                        ),
                    ),
                )
            )
        return VolRiskResult(
            base_price,
            tuple(points),
            {
                "model": "heston",
                "base_heston_params": asdict(base_params),
            },
        )

    def _calibrate_heston(
        self, env, surface: GridVolSurface, initial: HestonParams
    ) -> HestonParams:
        spec = self.heston_calibration_spec
        options = [
            MarketOption(K=float(strike), T=float(maturity), iv=float(surface.iv_grid[i, j]))
            for i, maturity in enumerate(surface.maturities)
            for j, strike in enumerate(surface.strikes)
        ]
        if self._is_fx:
            spot = float(env.effective_spot())
            rate = env.get_domestic_rate
            carry = env.get_foreign_rate
        else:
            spot = float(env.spot)
            rate = env.get_rate
            carry = env.get_div_yield
        result = calibrate_heston(
            spot,
            options,
            rate,
            carry,
            initial,
            bounds=spec.bounds,
            target="iv",
            regularize_feller=spec.regularize_feller,
            method=spec.method,
            max_nfev=spec.max_nfev,
            xtol=spec.xtol,
            ftol=spec.ftol,
            gtol=spec.gtol,
        )
        if not result.success:
            raise NumericalError(f"Heston calibration failed: {result.message}")
        return result.params

    def _slv_market_vega(
        self, product, env, engine, request: MarketVegaRequest
    ) -> VolRiskResult:
        surface = self._grid_surface(env)
        base_local = self._build_local_vol(env, surface)
        base_leverage = self._calibrate_leverage(
            product, env, engine.model_params, engine.eta, base_local
        )
        base_price = self._price_slv(
            engine, product, env, engine.model_params, engine.eta, base_local, base_leverage
        )

        def price_bumped(bump, direction):
            local_vol = self._build_local_vol(
                env, bump_grid_vol_surface(surface, bump, direction)
            )
            leverage = self._calibrate_leverage(
                product, env, engine.model_params, engine.eta, local_vol
            )
            return self._price_slv(
                engine, product, env, engine.model_params, engine.eta, local_vol, leverage
            )

        points = [
            self._central_point(
                f"market_iv.{bump.label}",
                bump.bump_size,
                base_price,
                lambda b=bump: price_bumped(b, 1),
                lambda b=bump: price_bumped(b, -1),
            )
            for bump in request.surface_bumps
        ]
        return VolRiskResult(
            base_price,
            tuple(points),
            {"model": "heston_slv", "market_vega_convention": "recalibrated_leverage"},
        )

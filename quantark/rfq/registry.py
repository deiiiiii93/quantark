"""
Registries and adapters for RFQ normalization.
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from inspect import signature
from typing import Any, Callable, Dict, Iterable, Optional

from quantark.asset.equity.engine import PDEEngine
from quantark.asset.equity.engine.analytical import (
    AmericanOptionAnalyticalEngine,
    AsianOptionAnalyticalEngine,
    BarrierAnalyticalEngine,
    BlackScholesEngine,
    DeltaOneEngine,
    DigitalOptionAnalyticalEngine,
    DoubleBarrierOptionAnalyticalEngine,
    OneTouchAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import (
    AmericanOptionMCEngine,
    AsianOptionMCEngine,
    BarrierOptionMCEngine,
    DigitalOptionMCEngine,
    PhoenixMCEngine,
    RangeAccrualMCEngine,
    SnowballMCEngine,
    EuropeanMCEngine,
)
from quantark.asset.equity.engine.quad import (
    BarrierQuadEngine,
    EuropeanQuadEngine,
    KOResetSnowballQuadEngine,
    OneTouchQuadEngine,
    PhoenixQuadEngine,
    SnowballQuadEngine,
)
from quantark.asset.equity.param import EngineParams, MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.deltaone import Futures, SpotInstrument
from quantark.asset.equity.product.option import (
    AccrualConfig,
    AmericanOption,
    AsianObservationRecord,
    AsianOption,
    BarrierConfig,
    BarrierOption,
    CashOrNothingDigitalOption,
    CouponBarrierConfig,
    DoubleBarrierOption,
    DoubleOneTouchOption,
    EuropeanVanillaOption,
    KnockOutResetSnowballOption,
    ObservationSchedule,
    OneTouchOption,
    PayoffConfig,
    PhoenixOption,
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
    SnowballOption,
)
from quantark.asset.equity.product.option.snowball_config import AirbagConfig
from quantark.param import ContinuousDividendYield, FlatVolSurface, NoDividend
from quantark.priceenv import PricingEnvironment
from quantark.rfq.models import RFQEngineSpec, RFQUnknownSpec
from quantark.util.exceptions import ValidationError


def _camel_to_snake(name: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0 and not name[index - 1].isupper():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


def _is_frozen_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and getattr(value, "__dataclass_params__", None).frozen


def get_dotted_attr(root: Any, path: str) -> Any:
    """Resolve a dotted attribute path."""
    current = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def set_dotted_attr(root: Any, path: str, value: Any) -> Any:
    """Assign to a dotted attribute path, replacing frozen dataclasses as needed."""
    parts = path.split(".")

    def _assign(current: Any, remaining: list[str], candidate: Any) -> Any:
        head = remaining[0]
        if len(remaining) == 1:
            if _is_frozen_dataclass_instance(current):
                return replace(current, **{head: candidate})
            setattr(current, head, candidate)
            return current

        child = getattr(current, head)
        updated_child = _assign(child, remaining[1:], candidate)
        if _is_frozen_dataclass_instance(current):
            return replace(current, **{head: updated_child})
        setattr(current, head, updated_child)
        return current

    return _assign(root, parts, value)


def _build_from_dict(builder: Callable[..., Any], value: Any) -> Any:
    if isinstance(value, dict):
        return builder(**value)
    return value


def _build_sequence(builder: Callable[..., Any], values: Any) -> Any:
    if not isinstance(values, list):
        return values
    normalized = []
    for item in values:
        normalized.append(_build_from_dict(builder, item))
    return normalized


class ProductBuilderRegistry:
    """Registry of term-sheet product builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(
        self,
        product_class: type,
        *,
        nested_builders: Optional[Dict[str, Callable[..., Any]]] = None,
        list_builders: Optional[Dict[str, Callable[..., Any]]] = None,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        nested_builders = nested_builders or {}
        list_builders = list_builders or {}
        aliases = tuple(aliases or ())
        params = set(signature(product_class.__init__).parameters) - {"self"}

        def _builder(kwargs: Dict[str, Any]) -> Any:
            unknown = set(kwargs) - params
            if unknown:
                unknown_list = ", ".join(sorted(unknown))
                raise ValidationError(
                    f"Unsupported kwargs for {product_class.__name__}: {unknown_list}"
                )

            normalized = dict(kwargs)
            for key, builder in nested_builders.items():
                if key in normalized:
                    normalized[key] = _build_from_dict(builder, normalized[key])
            for key, builder in list_builders.items():
                if key in normalized:
                    normalized[key] = _build_sequence(builder, normalized[key])
            return product_class(**normalized)

        for name in {product_class.__name__, _camel_to_snake(product_class.__name__), *aliases}:
            self._builders[name.lower()] = _builder

    def build(self, product_type: str, kwargs: Dict[str, Any]) -> Any:
        builder = self._builders.get(product_type.lower())
        if builder is None:
            raise ValidationError(f"Unsupported RFQ product_type: {product_type}")
        return builder(kwargs)


class EngineBuilderRegistry:
    """Registry of term-sheet engine builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[[RFQEngineSpec], Any]] = {}
        self._params_classes = {
            "engine_params": EngineParams,
            "mc_params": MCParams,
            "quad_params": QuadParams,
            "pde_params": PDEParams,
        }

    def register(
        self,
        engine_class: type,
        *,
        default_params_type: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        aliases = tuple(aliases or ())
        init_params = signature(engine_class.__init__).parameters
        supports_params = "params" in init_params
        supports_method = "method" in init_params

        def _builder(spec: RFQEngineSpec) -> Any:
            kwargs = dict(spec.engine_kwargs)

            if spec.params_type is not None:
                params_key = spec.params_type.lower()
            else:
                params_key = default_params_type

            if spec.params_kwargs:
                if not supports_params:
                    raise ValidationError(
                        f"{engine_class.__name__} does not accept params"
                    )
                if params_key is None:
                    raise ValidationError(
                        f"params_type is required for {engine_class.__name__}"
                    )
                params_class = self._params_classes.get(params_key)
                if params_class is None:
                    raise ValidationError(
                        f"Unsupported params_type for engine spec: {spec.params_type}"
                    )
                kwargs["params"] = params_class(**spec.params_kwargs)

            if spec.method is not None:
                if not supports_method:
                    raise ValidationError(
                        f"{engine_class.__name__} does not accept method"
                    )
                kwargs["method"] = spec.method

            return engine_class(**kwargs)

        for name in {engine_class.__name__, _camel_to_snake(engine_class.__name__), *aliases}:
            self._builders[name.lower()] = _builder

    def build(self, spec: RFQEngineSpec) -> Any:
        builder = self._builders.get(spec.engine_name.lower())
        if builder is None:
            raise ValidationError(
                f"Unsupported RFQ engine_name: {spec.engine_name}"
            )
        return builder(spec)


class QuoteableFieldAdapter:
    """Resolve and mutate supported quoteable fields."""

    def __init__(
        self,
        field_path: str,
        *,
        supported_types: Optional[tuple[type, ...]] = None,
        getter: Optional[Callable[[Any, PricingEnvironment], float]] = None,
        setter: Optional[Callable[[Any, PricingEnvironment, float], None]] = None,
    ) -> None:
        self.field_path = field_path
        self.supported_types = supported_types
        self._getter = getter
        self._setter = setter

    def supports(self, product: Any, pricing_env: PricingEnvironment) -> bool:
        del pricing_env
        if self.supported_types is None:
            return True
        return isinstance(product, self.supported_types)

    def get_value(self, product: Any, pricing_env: PricingEnvironment) -> float:
        if self._getter is None:
            return float(get_dotted_attr(product, self.field_path))
        return float(self._getter(product, pricing_env))

    def set_value(
        self, product: Any, pricing_env: PricingEnvironment, candidate: float
    ) -> None:
        if self._setter is None:
            set_dotted_attr(product, self.field_path, float(candidate))
            return
        self._setter(product, pricing_env, float(candidate))


PRODUCT_BUILDERS = ProductBuilderRegistry()
ENGINE_BUILDERS = EngineBuilderRegistry()
UNKNOWN_ADAPTERS: Dict[str, QuoteableFieldAdapter] = {}


def register_unknown_adapter(adapter: QuoteableFieldAdapter, *aliases: str) -> None:
    UNKNOWN_ADAPTERS[adapter.field_path.lower()] = adapter
    for alias in aliases:
        UNKNOWN_ADAPTERS[alias.lower()] = adapter


def resolve_unknown_adapter(
    unknown: RFQUnknownSpec,
    product: Any,
    pricing_env: PricingEnvironment,
) -> QuoteableFieldAdapter:
    adapter = UNKNOWN_ADAPTERS.get(unknown.field_path.lower())
    if adapter is None:
        raise ValidationError(f"Unsupported RFQ field_path: {unknown.field_path}")
    if not adapter.supports(product, pricing_env):
        raise ValidationError(
            f"field_path {unknown.field_path} is not supported for "
            f"{type(product).__name__}"
        )
    if adapter.field_path == "pricing_env.div_yield.div_yield":
        div_yield = pricing_env.div_yield
        if div_yield not in (None,) and not isinstance(
            div_yield, (ContinuousDividendYield, NoDividend)
        ):
            raise ValidationError(
                "RFQ q solving only supports flat ContinuousDividendYield or NoDividend"
            )
    if adapter.field_path == "pricing_env.vol_surface.volatility":
        vol_surface = pricing_env.vol_surface
        if vol_surface is not None and not isinstance(vol_surface, FlatVolSurface):
            raise ValidationError(
                "RFQ volatility solving only supports FlatVolSurface"
            )
    return adapter


def _sync_schedule_scalar(
    schedule: Optional[ObservationSchedule],
    *,
    attr_name: str,
    value: Any,
) -> None:
    if schedule is None:
        return
    for record in schedule.records:
        setattr(record, attr_name, value)


def _set_barrier_config_field(
    product: Any,
    config_attr: str,
    field_name: str,
    candidate: float,
) -> None:
    config = getattr(product, config_attr)
    updated = replace(config, **{field_name: candidate})
    schedule = getattr(updated, "ko_observation_schedule", None)
    if field_name == "ko_rate":
        _sync_schedule_scalar(schedule, attr_name="return_rate", value=candidate)
    elif field_name in {"ko_barrier", "ki_barrier"}:
        target_schedule = (
            getattr(updated, "ko_observation_schedule", None)
            if field_name == "ko_barrier"
            else getattr(updated, "ki_observation_schedule", None)
        )
        _sync_schedule_scalar(target_schedule, attr_name="barrier", value=candidate)
    setattr(product, config_attr, updated)


def _register_default_products() -> None:
    PRODUCT_BUILDERS.register(EuropeanVanillaOption)
    PRODUCT_BUILDERS.register(AmericanOption)
    PRODUCT_BUILDERS.register(
        AsianOption,
        list_builders={"observation_records": AsianObservationRecord},
    )
    PRODUCT_BUILDERS.register(CashOrNothingDigitalOption)
    PRODUCT_BUILDERS.register(BarrierOption)
    PRODUCT_BUILDERS.register(DoubleBarrierOption)
    PRODUCT_BUILDERS.register(OneTouchOption)
    PRODUCT_BUILDERS.register(DoubleOneTouchOption)
    PRODUCT_BUILDERS.register(
        SnowballOption,
        nested_builders={
            "barrier_config": BarrierConfig,
            "payoff_config": PayoffConfig,
            "accrual_config": AccrualConfig,
            "airbag_config": AirbagConfig,
        },
    )
    PRODUCT_BUILDERS.register(
        KnockOutResetSnowballOption,
        nested_builders={
            "barrier_config": BarrierConfig,
            "post_barrier_config": BarrierConfig,
            "payoff_config": PayoffConfig,
            "accrual_config": AccrualConfig,
            "airbag_config": AirbagConfig,
        },
        aliases=("ko_reset_snowball_option",),
    )
    PRODUCT_BUILDERS.register(
        PhoenixOption,
        nested_builders={
            "barrier_config": BarrierConfig,
            "coupon_config": CouponBarrierConfig,
            "payoff_config": PayoffConfig,
            "accrual_config": AccrualConfig,
            "airbag_config": AirbagConfig,
        },
    )
    PRODUCT_BUILDERS.register(
        RangeAccrualOption,
        nested_builders={"range_config": RangeAccrualConfig},
        list_builders={"observation_records": RangeAccrualObservationRecord},
    )
    PRODUCT_BUILDERS.register(SpotInstrument)
    PRODUCT_BUILDERS.register(Futures)


def _register_default_engines() -> None:
    PRODUCT = "engine_params"
    MC = "mc_params"
    QUAD = "quad_params"
    PDE = "pde_params"

    ENGINE_BUILDERS.register(BlackScholesEngine, default_params_type=PRODUCT)
    ENGINE_BUILDERS.register(DeltaOneEngine, default_params_type=PRODUCT)
    ENGINE_BUILDERS.register(
        AmericanOptionAnalyticalEngine,
        default_params_type=PRODUCT,
    )
    ENGINE_BUILDERS.register(
        DigitalOptionAnalyticalEngine,
        default_params_type=PRODUCT,
    )
    ENGINE_BUILDERS.register(BarrierAnalyticalEngine, default_params_type=PRODUCT)
    ENGINE_BUILDERS.register(
        DoubleBarrierOptionAnalyticalEngine,
        default_params_type=PRODUCT,
    )
    ENGINE_BUILDERS.register(OneTouchAnalyticalEngine, default_params_type=PRODUCT)
    ENGINE_BUILDERS.register(AsianOptionAnalyticalEngine, default_params_type=PRODUCT)
    ENGINE_BUILDERS.register(
        RangeAccrualAnalyticalEngine, default_params_type=PRODUCT
    )
    ENGINE_BUILDERS.register(EuropeanMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(AmericanOptionMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(SnowballMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(PhoenixMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(AsianOptionMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(DigitalOptionMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(BarrierOptionMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(RangeAccrualMCEngine, default_params_type=MC)
    ENGINE_BUILDERS.register(EuropeanQuadEngine, default_params_type=QUAD)
    ENGINE_BUILDERS.register(BarrierQuadEngine, default_params_type=QUAD)
    ENGINE_BUILDERS.register(OneTouchQuadEngine, default_params_type=QUAD)
    ENGINE_BUILDERS.register(SnowballQuadEngine, default_params_type=QUAD)
    ENGINE_BUILDERS.register(KOResetSnowballQuadEngine, default_params_type=QUAD)
    ENGINE_BUILDERS.register(PhoenixQuadEngine, default_params_type=QUAD)
    ENGINE_BUILDERS.register(PDEEngine, default_params_type=PDE)


def _register_default_unknowns() -> None:
    vanilla_types = (EuropeanVanillaOption, AmericanOption, AsianOption, BarrierOption, DoubleBarrierOption)
    register_unknown_adapter(
        QuoteableFieldAdapter("strike", supported_types=vanilla_types)
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "payout", supported_types=(CashOrNothingDigitalOption,)
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "barrier", supported_types=(BarrierOption, OneTouchOption)
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "upper_barrier", supported_types=(DoubleBarrierOption, DoubleOneTouchOption)
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "lower_barrier", supported_types=(DoubleBarrierOption, DoubleOneTouchOption)
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "rebate",
            supported_types=(
                BarrierOption,
                DoubleBarrierOption,
                OneTouchOption,
                DoubleOneTouchOption,
            ),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "initial_price",
            supported_types=(
                AsianOption,
                SnowballOption,
                KnockOutResetSnowballOption,
                PhoenixOption,
                RangeAccrualOption,
            ),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "barrier_config.ko_rate",
            supported_types=(SnowballOption, KnockOutResetSnowballOption, PhoenixOption),
            setter=lambda product, pricing_env, candidate: _set_barrier_config_field(
                product, "barrier_config", "ko_rate", candidate
            ),
        ),
        "ko_rate",
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "barrier_config.ko_barrier",
            supported_types=(SnowballOption, KnockOutResetSnowballOption, PhoenixOption),
            setter=lambda product, pricing_env, candidate: _set_barrier_config_field(
                product, "barrier_config", "ko_barrier", candidate
            ),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "barrier_config.ki_barrier",
            supported_types=(SnowballOption, KnockOutResetSnowballOption, PhoenixOption),
            setter=lambda product, pricing_env, candidate: _set_barrier_config_field(
                product, "barrier_config", "ki_barrier", candidate
            ),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "coupon_config.coupon_rate",
            supported_types=(PhoenixOption,),
        ),
        "coupon_rate",
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "coupon_config.coupon_barrier",
            supported_types=(PhoenixOption,),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "post_barrier_config.ko_rate",
            supported_types=(KnockOutResetSnowballOption,),
            setter=lambda product, pricing_env, candidate: _set_barrier_config_field(
                product, "post_barrier_config", "ko_rate", candidate
            ),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "range_config.accrual_rate",
            supported_types=(RangeAccrualOption,),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "range_config.upper_barrier",
            supported_types=(RangeAccrualOption,),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "range_config.lower_barrier",
            supported_types=(RangeAccrualOption,),
        )
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "basis", supported_types=(Futures,)
        )
    )

    register_unknown_adapter(
        QuoteableFieldAdapter(
            "pricing_env.div_yield.div_yield",
            getter=lambda product, pricing_env: (
                0.0
                if pricing_env.div_yield is None
                else (
                    0.0
                    if isinstance(pricing_env.div_yield, NoDividend)
                    else float(pricing_env.div_yield.div_yield)
                )
            ),
            setter=lambda product, pricing_env, candidate: setattr(
                pricing_env,
                "div_yield",
                ContinuousDividendYield(div_yield=float(candidate)),
            ),
        ),
        "q",
        "dividend_yield",
    )
    register_unknown_adapter(
        QuoteableFieldAdapter(
            "pricing_env.vol_surface.volatility",
            getter=lambda product, pricing_env: float(
                pricing_env.vol_surface.volatility
            ),
            setter=lambda product, pricing_env, candidate: setattr(
                pricing_env,
                "vol_surface",
                FlatVolSurface(volatility=float(candidate)),
            ),
        ),
        "vol",
        "sigma",
    )


_register_default_products()
_register_default_engines()
_register_default_unknowns()

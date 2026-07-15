"""Adapter registry with string-keyed MRO resolution (spec section 6.2).

Engines are matched by walking ``type(engine).__mro__`` against registered
``"module.qualname"`` strings, so this module never imports asset, product,
or engine code. Registries freeze for the lifetime of a session; duplicate
or post-freeze registration is a validation error.

Resolution order (spec section 6.2): exact engine class first (MRO position
0), then the nearest registered base class in MRO order. Python's MRO is a
total order, so "nearest" is deterministic; ambiguity is prevented at
registration time by rejecting duplicate paths. Structural capability
detection (resolution step 2) becomes meaningful in Phase 1 when the first
specialized adapters land; Phase 0 has none, so it is a documented no-op.
"""
from quantark.execution.errors import CapabilityError
from quantark.util.exceptions import ValidationError

__all__ = ["AdapterRegistry", "build_default_registry"]


class AdapterRegistry:
    def __init__(self):
        self._factories: dict = {}
        self._frozen = False

    def register(self, engine_class_path: str, adapter_factory) -> None:
        if self._frozen:
            raise ValidationError(
                "AdapterRegistry is frozen; register adapters before "
                "session construction"
            )
        if engine_class_path in self._factories:
            raise ValidationError(
                f"duplicate adapter registration for {engine_class_path}"
            )
        self._factories[engine_class_path] = adapter_factory

    def freeze(self) -> None:
        self._frozen = True

    def registered_paths(self) -> tuple:
        return tuple(self._factories)

    def resolve(self, engine):
        for cls in type(engine).__mro__:
            key = f"{cls.__module__}.{cls.__qualname__}"
            factory = self._factories.get(key)
            if factory is not None:
                return factory()
        raise CapabilityError(
            f"no execution adapter registered for engine type "
            f"{type(engine).__module__}.{type(engine).__qualname__}"
        )


_DEFAULT_REGISTRATIONS = (
    ("quantark.asset.equity.engine.base_engine.BaseEngine", "product_env"),
    ("quantark.asset.fx.engine.base_fx_engine.BaseFxEngine", "product_env"),
    (
        "quantark.asset.credit.engine.base_credit_engine.BaseCreditEngine",
        "product_env",
    ),
    (
        "quantark.asset.bond.engine.pde.convertible.jump_diffusion_engine."
        "ConvertibleBondJumpDiffusionEngine",
        "env_bound",
    ),
    (
        "quantark.asset.bond.engine.pde.convertible.tf_engine."
        "ConvertibleBondTFEngine",
        "env_bound",
    ),
    (
        "quantark.asset.bond.engine.convertible.convertible_bond_engine."
        "ConvertibleBondEngine",
        "env_bound",
    ),
)


def build_default_registry() -> AdapterRegistry:
    """Fresh registry with the serial compatibility adapter registered for
    every known engine-family root. Callers freeze it at session construction."""
    from quantark.execution.legacy_adapter import LegacyPriceAdapter

    registry = AdapterRegistry()
    for path, shape in _DEFAULT_REGISTRATIONS:
        registry.register(
            path,
            (lambda s: (lambda: LegacyPriceAdapter(call_shape=s)))(shape),
        )
    return registry

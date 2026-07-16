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

    def register(self, engine_class_path: str, adapter_factory,
                 exact: bool = False) -> None:
        """``exact=True`` matches ONLY the class itself, never subclasses
        via MRO (code-gate finding 2026-07-16: adapters that reconstruct an
        engine from a fixed constructor signature would silently discard
        subclass state; unknown subclasses must fall through to a base
        registration instead)."""
        if self._frozen:
            raise ValidationError(
                "AdapterRegistry is frozen; register adapters before "
                "session construction"
            )
        if engine_class_path in self._factories:
            raise ValidationError(
                f"duplicate adapter registration for {engine_class_path}"
            )
        self._factories[engine_class_path] = (adapter_factory, exact)

    def freeze(self) -> None:
        self._frozen = True

    def registered_paths(self) -> tuple:
        return tuple(self._factories)

    def resolve(self, engine):
        return self.resolve_class(type(engine))

    def resolve_class(self, engine_class):
        for cls in engine_class.__mro__:
            key = f"{cls.__module__}.{cls.__qualname__}"
            entry = self._factories.get(key)
            if entry is not None:
                factory, exact = entry
                if exact and cls is not engine_class:
                    continue  # exact registrations never match subclasses
                return factory()
        raise CapabilityError(
            f"no execution adapter registered for engine type "
            f"{engine_class.__module__}.{engine_class.__qualname__}"
        )


_DEFAULT_REGISTRATIONS = (
    ("quantark.asset.equity.engine.base_engine.BaseEngine", "product_env"),
    # SABRMCEngine does not inherit the equity BaseEngine (spec section A.4
    # abstraction gap) but uses the standard price(product, env) shape.
    ("quantark.asset.equity.engine.mc.sabr_mc_engine.SABRMCEngine", "product_env"),
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
    # Specialized exact-class adapters (lazy factory import at resolve time —
    # this module keeps zero static asset-code dependencies).
    # exact=True on both (code-gate finding 2026-07-16): these adapters
    # clone the engine around a cached surface with a fixed constructor
    # signature; subclasses (e.g. the frozen quant-mini-project
    # SurfaceAwareLVDCNEngine) keep the plain legacy adapter and their
    # direct-path hooks (spec section 17.1).
    registry.register(
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.LocalVolDCNMCEngine",
        _dcn_mc_adapter, exact=True,
    )
    registry.register(
        "quantark.asset.equity.engine.pde.dcn_vol_pde_solvers.LocalVolDCNPDEEngine",
        _dcn_pde_adapter, exact=True,
    )
    # Phase 2: fixed-batch capability for the plain GBM DCN MC engine.
    # exact=True: the batch adapters reconstruct engines from the exact
    # constructor signature, so unknown subclasses (with extra state) must
    # fall through the MRO to the legacy adapter, never be cloned lossily.
    registry.register(
        "quantark.asset.equity.engine.mc.dcn_mc_engine.DCNMCEngine",
        _dcn_batch_mc_adapter, exact=True,
    )
    # MRO containment: the Heston DCN family inherits DCNMCEngine but its
    # paired normal+uniform draw pipeline is not repository-routed yet
    # (Phase 3), so exact pins keep it on the plain legacy adapter instead
    # of inheriting the batch capability through the DCNMCEngine entry.
    # QEDCNMCEngine resolves via the HestonDCNMCEngine pin (nearest base).
    for heston_dcn in (
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.HestonDCNMCEngine",
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines."
        "CoupledCoarseHestonDCNMCEngine",
    ):
        registry.register(heston_dcn, _legacy_product_env_adapter)
    return registry


def _dcn_mc_adapter():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNLocalVolMCAdapter,
    )

    return DCNLocalVolMCAdapter()


def _dcn_pde_adapter():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNLocalVolPDEAdapter,
    )

    return DCNLocalVolPDEAdapter()


def _dcn_batch_mc_adapter():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNBatchMCAdapter,
    )

    return DCNBatchMCAdapter()


def _legacy_product_env_adapter():
    from quantark.execution.legacy_adapter import LegacyPriceAdapter

    return LegacyPriceAdapter(call_shape="product_env")

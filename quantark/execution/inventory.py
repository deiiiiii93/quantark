"""Checked-in exported-engine capability inventory (spec section 18).

This module is the release-gate source of truth. It contains no static
imports of asset code; ``discover_exported_engine_names`` imports the public
surfaces dynamically when called (from tests/CI only).

Phase 0: every row is ``temporary_legacy`` (session-reachable through the
serial LegacyPriceAdapter) except abstract bases, which are
``not_applicable``. Later phases upgrade rows to ``supported`` as native
adapters land.
"""
import importlib
from dataclasses import dataclass

__all__ = [
    "InventoryRecord",
    "BATCH_STATES",
    "ENGINE_INVENTORY",
    "SUPPORTING_EXPORTS",
    "DISCOVERY_SURFACES",
    "EXPLICIT_FACADES",
    "discover_exported_engine_names",
    "inventory_by_name",
]

_OWNER = "execution-framework"
_MILESTONE = "phase-1+"

# Phase 2 exit gate (spec section 21): every fixed-batch exported MC engine
# either advertises the batch capability or carries a specific rationale.
BATCH_STATES = ("batch_capable", "temporary_legacy", "not_applicable")

_BATCH_ADAPTIVE = (
    "adaptive RQMC compatibility stopping is sequential by contract "
    "(spec 8.4); parallel-wave is a Phase 3 opt-in plan"
)
_BATCH_SINGLE_SOLVE = (
    "single-solve engine without a batch axis; introducing one is a changed "
    "numerical plan - re-scope on Phase 2 benchmark evidence (spec 21)"
)
_BATCH_HESTON_DCN = (
    "Heston draw pipeline (paired normal+uniform streams) not yet routed "
    "through DrawRepository; Phase 3 model families"
)
_BATCH_LSM = (
    "LSM regression couples all paths cross-sectionally; no independent "
    "batch decomposition exists"
)
_BATCH_FX = (
    "FX MC engines are single-solve on the shared montecarlo layer; batch "
    "decomposition re-scoped on Phase 2 benchmark evidence"
)
_BATCH_PDE = (
    "PDE solve has no batch axis; Phase 4 covers PDE preparation and rich "
    "outcomes"
)
_BATCH_NON_MC = "not an MC engine; batch capability does not apply"


@dataclass(frozen=True)
class InventoryRecord:
    name: str
    import_path: str
    engine_type: str      # "mc" | "pde"
    asset_family: str     # "equity" | "fx" | "credit" | "bond"
    model_family: str     # "bsm" | "lv" | "heston" | "slv" | "sabr" | "copula" | "jump_diffusion" | "dispatch"
    product_family: str
    planning: str         # "fixed" | "both" (fixed + adaptive RQMC mode)
    call_shape: str       # "product_env" | "env_bound"
    role: str = "engine"  # "engine" | "facade" | "abstract"
    backends: tuple = ("serial",)
    adoption_state: str = "temporary_legacy"
    owner: str | None = _OWNER
    milestone: str | None = _MILESTONE
    reason: str | None = None
    batch_state: str = "not_applicable"
    batch_rationale: str = _BATCH_NON_MC


def _eq_mc(name, model, product, planning="fixed",
           adoption_state="temporary_legacy",
           batch_state="temporary_legacy", batch_rationale=_BATCH_SINGLE_SOLVE):
    return InventoryRecord(
        name=name,
        import_path=f"quantark.asset.equity.engine.mc.{name}",
        engine_type="mc", asset_family="equity", model_family=model,
        product_family=product, planning=planning, call_shape="product_env",
        adoption_state=adoption_state,
        batch_state=batch_state, batch_rationale=batch_rationale,
    )


def _eq_pde(name, model, product, role="engine", adoption_state=None):
    if adoption_state is None:
        adoption_state = (
            "not_applicable" if role == "abstract" else "temporary_legacy"
        )
    return InventoryRecord(
        name=name,
        import_path=f"quantark.asset.equity.engine.pde.{name}",
        engine_type="pde", asset_family="equity", model_family=model,
        product_family=product, planning="fixed", call_shape="product_env",
        role=role,
        adoption_state=adoption_state,
        owner=None if role == "abstract" else _OWNER,
        milestone=None if role == "abstract" else _MILESTONE,
        reason="abstract base class" if role == "abstract" else None,
        batch_state="not_applicable", batch_rationale=_BATCH_PDE,
    )


def _fx(name, engine_type, model, product):
    sub = "mc" if engine_type == "mc" else "pde"
    if engine_type == "mc":
        batch_state, batch_rationale = "temporary_legacy", _BATCH_FX
    else:
        batch_state, batch_rationale = "not_applicable", _BATCH_PDE
    return InventoryRecord(
        name=name,
        import_path=f"quantark.asset.fx.engine.{sub}.{name}",
        engine_type=engine_type, asset_family="fx", model_family=model,
        product_family=product, planning="fixed", call_shape="product_env",
        batch_state=batch_state, batch_rationale=batch_rationale,
    )


ENGINE_INVENTORY = (
    # --- Equity MC (33 engines) ---
    _eq_mc("EuropeanMCEngine", "bsm", "vanilla"),
    _eq_mc("LocalVolMCEngine", "lv", "vanilla"),
    _eq_mc("HestonMCEngine", "heston", "vanilla"),
    _eq_mc("HestonSLVMCEngine", "slv", "vanilla"),
    _eq_mc("SABRMCEngine", "sabr", "vanilla"),
    _eq_mc("AmericanOptionMCEngine", "bsm", "american",
           batch_state="not_applicable", batch_rationale=_BATCH_LSM),
    _eq_mc("AsianOptionMCEngine", "bsm", "asian"),
    _eq_mc("DigitalOptionMCEngine", "bsm", "digital"),
    _eq_mc("BarrierOptionMCEngine", "bsm", "barrier"),
    _eq_mc("LocalVolBarrierMCEngine", "lv", "barrier"),
    _eq_mc("HestonBarrierMCEngine", "heston", "barrier"),
    _eq_mc("HestonSLVBarrierMCEngine", "slv", "barrier"),
    _eq_mc("SingleSharkfinOptionMCEngine", "bsm", "sharkfin"),
    _eq_mc("DoubleSharkfinOptionMCEngine", "bsm", "sharkfin"),
    _eq_mc("RangeAccrualMCEngine", "bsm", "range_accrual"),
    _eq_mc("AccumulatorMCEngine", "bsm", "accumulator"),
    _eq_mc("SnowballMCEngine", "bsm", "snowball", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("LocalVolSnowballMCEngine", "lv", "snowball", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("HestonSnowballMCEngine", "heston", "snowball", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("QESnowballMCEngine", "heston", "snowball", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("HestonSLVSnowballMCEngine", "slv", "snowball", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("HestonSLVQESnowballMCEngine", "slv", "snowball", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("PhoenixMCEngine", "bsm", "phoenix", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("LocalVolPhoenixMCEngine", "lv", "phoenix", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("HestonPhoenixMCEngine", "heston", "phoenix", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("QEPhoenixMCEngine", "heston", "phoenix", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("HestonSLVPhoenixMCEngine", "slv", "phoenix", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("HestonSLVQEPhoenixMCEngine", "slv", "phoenix", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_ADAPTIVE),
    _eq_mc("DCNMCEngine", "bsm", "dcn", planning="both",
           adoption_state="supported", batch_state="batch_capable",
           batch_rationale=""),
    _eq_mc("LocalVolDCNMCEngine", "lv", "dcn", planning="both",
           adoption_state="supported", batch_state="batch_capable",
           batch_rationale=""),
    _eq_mc("HestonDCNMCEngine", "heston", "dcn", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_HESTON_DCN),
    _eq_mc("QEDCNMCEngine", "heston", "dcn", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_HESTON_DCN),
    _eq_mc("CoupledCoarseHestonDCNMCEngine", "heston", "dcn", planning="both",
           batch_state="temporary_legacy",
           batch_rationale=_BATCH_HESTON_DCN),
    # --- Equity PDE (24 concrete + abstract base) ---
    _eq_pde("BasePDESolver", "dispatch", "any", role="abstract"),
    _eq_pde("EuropeanPDESolver", "bsm", "vanilla"),
    _eq_pde("AmericanPDESolver", "bsm", "american"),
    _eq_pde("BarrierPDESolver", "bsm", "barrier"),
    _eq_pde("DoubleBarrierPDESolver", "bsm", "double_barrier"),
    _eq_pde("OneTouchPDESolver", "bsm", "one_touch"),
    _eq_pde("DoubleOneTouchPDESolver", "bsm", "one_touch"),
    _eq_pde("SnowballPDESolver", "bsm", "snowball"),
    _eq_pde("KOResetSnowballPDESolver", "bsm", "ko_reset_snowball"),
    _eq_pde("PhoenixPDESolver", "bsm", "phoenix"),
    _eq_pde("LocalVolPDESolver", "lv", "vanilla"),
    _eq_pde("HestonPDESolver", "heston", "vanilla"),
    _eq_pde("HestonSLVPDESolver", "slv", "vanilla"),
    _eq_pde("LocalVolBarrierPDESolver", "lv", "barrier"),
    _eq_pde("HestonBarrierPDESolver", "heston", "barrier"),
    _eq_pde("HestonSLVBarrierPDESolver", "slv", "barrier"),
    _eq_pde("LocalVolSnowballPDESolver", "lv", "snowball"),
    _eq_pde("HestonSnowballPDESolver", "heston", "snowball"),
    _eq_pde("HestonSLVSnowballPDESolver", "slv", "snowball"),
    _eq_pde("LocalVolPhoenixPDESolver", "lv", "phoenix"),
    _eq_pde("HestonPhoenixPDESolver", "heston", "phoenix"),
    _eq_pde("HestonSLVPhoenixPDESolver", "slv", "phoenix"),
    _eq_pde("DCNPDEEngine", "bsm", "dcn"),
    _eq_pde("LocalVolDCNPDEEngine", "lv", "dcn", adoption_state="supported"),
    _eq_pde("HestonDCNPDESolver", "heston", "dcn"),
    # --- Equity facade ---
    InventoryRecord(
        name="PDEEngine",
        import_path="quantark.asset.equity.engine.PDEEngine",
        engine_type="pde", asset_family="equity", model_family="dispatch",
        product_family="any", planning="fixed", call_shape="product_env",
        role="facade",
    ),
    # --- FX MC (8) ---
    _fx("FxLocalVolMCEngine", "mc", "lv", "vanilla"),
    _fx("FxHestonMCEngine", "mc", "heston", "vanilla"),
    _fx("FxHestonSLVMCEngine", "mc", "slv", "vanilla"),
    _fx("FxRangeAccrualMCEngine", "mc", "bsm", "range_accrual"),
    _fx("FxBarrierMCEngine", "mc", "bsm", "barrier"),
    _fx("FxSharkfinMCEngine", "mc", "bsm", "sharkfin"),
    _fx("FxTarnForwardMCEngine", "mc", "bsm", "tarn"),
    _fx("FxTargetRedemptionNoteMCEngine", "mc", "bsm", "tarn"),
    # --- FX PDE (3) ---
    _fx("FxLocalVolPDESolver", "pde", "lv", "vanilla"),
    _fx("FxHestonPDESolver", "pde", "heston", "vanilla"),
    _fx("FxHestonSLVPDESolver", "pde", "slv", "vanilla"),
    # --- Credit MC (1) ---
    InventoryRecord(
        name="BasketCDSEngine",
        import_path="quantark.asset.credit.engine.mc.BasketCDSEngine",
        engine_type="mc", asset_family="credit", model_family="copula",
        product_family="basket_cds", planning="fixed",
        call_shape="product_env",
        batch_state="temporary_legacy", batch_rationale=_BATCH_SINGLE_SOLVE,
    ),
    # --- Bond PDE (2 + facade) ---
    InventoryRecord(
        name="ConvertibleBondJumpDiffusionEngine",
        import_path=(
            "quantark.asset.bond.engine.pde.ConvertibleBondJumpDiffusionEngine"
        ),
        engine_type="pde", asset_family="bond",
        model_family="jump_diffusion", product_family="convertible",
        planning="fixed", call_shape="env_bound",
    ),
    InventoryRecord(
        name="ConvertibleBondTFEngine",
        import_path="quantark.asset.bond.engine.pde.ConvertibleBondTFEngine",
        engine_type="pde", asset_family="bond", model_family="bsm",
        product_family="convertible", planning="fixed",
        call_shape="env_bound",
    ),
    InventoryRecord(
        name="ConvertibleBondEngine",
        import_path="quantark.asset.bond.engine.ConvertibleBondEngine",
        engine_type="pde", asset_family="bond", model_family="dispatch",
        product_family="convertible", planning="fixed",
        call_shape="env_bound", role="facade",
    ),
)


DISCOVERY_SURFACES = (
    "quantark.asset.equity.engine.mc",
    "quantark.asset.equity.engine.pde",
    "quantark.asset.fx.engine.mc",
    "quantark.asset.fx.engine.pde",
    "quantark.asset.credit.engine.mc",
    "quantark.asset.bond.engine.pde",
)

# Facades exported from wider (mixed analytical/quad) surfaces are named
# explicitly rather than discovered by __all__ union.
EXPLICIT_FACADES = (
    ("quantark.asset.equity.engine", "PDEEngine"),
    ("quantark.asset.bond.engine", "ConvertibleBondEngine"),
)

SUPPORTING_EXPORTS = {
    "quantark.asset.equity.engine.mc": (
        "AmericanMCResult", "DCNMCResult", "PhoenixMCResult", "AsianMCResult",
        "RangeAccrualMCResult", "coupled_heston_ladder_pair",
    ),
    "quantark.asset.equity.engine.pde": (
        "TimeGrid", "SpatialGrid", "DCNPDEResult",
    ),
    "quantark.asset.fx.engine.mc": (
        "FxRangeAccrualMCResult", "FxBarrierMCResult", "FxSharkfinMCResult",
        "FxTarnMCResult",
    ),
    "quantark.asset.fx.engine.pde": (),
    "quantark.asset.credit.engine.mc": (),
    "quantark.asset.bond.engine.pde": ("ConvertibleBondPDEParams",),
}


def discover_exported_engine_names() -> dict:
    """Union the public ``__all__`` of every discovery surface (dynamic
    import at call time; used by tests/CI, never at library import time)."""
    discovered = {}
    for surface in DISCOVERY_SURFACES:
        module = importlib.import_module(surface)
        discovered[surface] = tuple(module.__all__)
    for module_path, facade_name in EXPLICIT_FACADES:
        module = importlib.import_module(module_path)
        assert facade_name in module.__all__, (module_path, facade_name)
        discovered.setdefault(module_path, ())
        discovered[module_path] = discovered[module_path] + (facade_name,)
    return discovered


def inventory_by_name() -> dict:
    return {record.name: record for record in ENGINE_INVENTORY}

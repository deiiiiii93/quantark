"""Checked-in engine inventory and CI discovery gate (spec section 18)."""
import importlib

from quantark.execution.inventory import (
    DISCOVERY_SURFACES,
    ENGINE_INVENTORY,
    EXPLICIT_FACADES,
    SUPPORTING_EXPORTS,
    discover_exported_engine_names,
    inventory_by_name,
)
from quantark.execution.registry import build_default_registry

assert DISCOVERY_SURFACES and EXPLICIT_FACADES  # imported as part of the gate


def test_every_public_export_is_classified():
    """CI gate: a new public MC/PDE export must be inventoried or classified
    as supporting; otherwise this test fails (spec section 18)."""
    discovered = discover_exported_engine_names()
    inventoried = set(inventory_by_name())
    for surface, names in discovered.items():
        supporting = set(SUPPORTING_EXPORTS.get(surface, ()))
        for name in names:
            assert name in inventoried or name in supporting, (
                f"public export {surface}:{name} is neither inventoried "
                f"nor classified as supporting"
            )


def test_inventory_names_exist_and_import():
    for record in ENGINE_INVENTORY:
        module_path, _, class_name = record.import_path.rpartition(".")
        module = importlib.import_module(module_path)
        assert hasattr(module, class_name), record.import_path


def test_inventory_counts_match_spec_snapshot():
    by_family = {}
    for r in ENGINE_INVENTORY:
        by_family.setdefault((r.asset_family, r.engine_type, r.role), []).append(r)
    assert len(by_family[("equity", "mc", "engine")]) == 33
    assert len(by_family[("equity", "pde", "engine")]) == 24
    assert len(by_family[("equity", "pde", "abstract")]) == 1  # BasePDESolver
    assert len(by_family[("equity", "pde", "facade")]) == 1  # PDEEngine
    assert len(by_family[("fx", "mc", "engine")]) == 8
    assert len(by_family[("fx", "pde", "engine")]) == 3
    assert len(by_family[("credit", "mc", "engine")]) == 1
    assert len(by_family[("bond", "pde", "engine")]) == 2
    assert len(by_family[("bond", "pde", "facade")]) == 1


def test_temporary_legacy_rows_have_owner_and_milestone():
    for record in ENGINE_INVENTORY:
        if record.adoption_state == "temporary_legacy":
            assert record.owner and record.milestone, record.name
        if record.adoption_state == "not_applicable":
            assert record.reason, record.name


def test_every_concrete_engine_is_session_reachable():
    """Phase 0 exit gate: for every concrete inventoried engine class, the
    default registry resolves a serial adapter AND the class exposes a
    ``price`` callable whose arity matches the declared call shape.

    Full direct-versus-session numerical parity for every row is Phase 1's
    exit gate (spec section 21, Phase 1: "direct versus session parity
    across the matrix"); Phase 0 proves reachability plus the representative
    per-family parity matrix in test_session_parity.py.
    """
    import inspect

    registry = build_default_registry()
    registry.freeze()
    for record in ENGINE_INVENTORY:
        if record.role == "abstract":
            continue
        module_path, _, class_name = record.import_path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        fake = cls.__new__(cls)  # resolution is type-based; no construction
        adapter = registry.resolve(fake)
        # Phase 2: DCN MC rows resolve to the batch adapter; Phase 3:
        # Snowball/Phoenix rows resolve to the adaptive adapter; everything
        # else remains on the serial compatibility adapter.
        assert adapter.capabilities().adapter_id in (
            "legacy-price", "dcn-batch-mc", "autocallable-adaptive-mc",
        ), record.name
        assert adapter.call_shape == record.call_shape, record.name
        price = getattr(cls, "price", None)
        assert callable(price), f"{record.name} has no price method"
        params = [
            p for p in inspect.signature(price).parameters.values()
            if p.name != "self" and p.kind is not p.VAR_KEYWORD
        ]
        required = [p for p in params if p.default is p.empty]
        expected_args = 1 if record.call_shape == "env_bound" else 2
        assert len(required) <= expected_args <= len(params), (
            f"{record.name}: price arity {len(params)} does not fit "
            f"declared call shape {record.call_shape}"
        )


# ---------------------------------------------------------------------------
# Phase 2 exit gate: batch-capability audit
# ---------------------------------------------------------------------------

def test_every_row_has_audited_batch_state():
    from quantark.execution.inventory import BATCH_STATES

    for record in ENGINE_INVENTORY:
        assert record.batch_state in BATCH_STATES, record.name
        if record.engine_type == "mc" and record.batch_state != "batch_capable":
            assert record.batch_rationale.strip(), (
                f"{record.name}: non-capable MC rows need a specific rationale"
            )


def test_batch_capable_rows_resolve_to_batch_adapters():
    registry = build_default_registry()
    registry.freeze()
    capable = [
        r for r in ENGINE_INVENTORY if r.batch_state == "batch_capable"
    ]
    assert {r.name for r in capable} == {"DCNMCEngine", "LocalVolDCNMCEngine"}
    for record in capable:
        module_path, _, cls_name = record.import_path.rpartition(".")
        engine_cls = getattr(importlib.import_module(module_path), cls_name)
        adapter = registry.resolve_class(engine_cls)
        assert hasattr(adapter, "plan_batches"), record.name


def test_heston_dcn_family_resolves_to_batch_adapters():
    # Phase 3: the whole Heston DCN family gets exact-registered batch
    # adapters (CoupledCoarse via its pair-aware adapter).
    from quantark.asset.equity.engine.mc import (
        CoupledCoarseHestonDCNMCEngine,
        HestonDCNMCEngine,
        QEDCNMCEngine,
    )

    registry = build_default_registry()
    registry.freeze()
    for cls in (HestonDCNMCEngine, QEDCNMCEngine,
                CoupledCoarseHestonDCNMCEngine):
        adapter = registry.resolve_class(cls)
        assert hasattr(adapter, "plan_batches"), cls.__name__
        assert adapter.capabilities().adapter_id == "dcn-batch-mc"

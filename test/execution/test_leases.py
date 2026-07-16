"""Resource lease accounting (spec section 11, Phase 1 subset)."""
import pytest

from quantark.execution.errors import ResourceBudgetExceeded
from quantark.execution.leases import ResourceLeaseManager
from quantark.execution.policy import ResourceBudget


def test_task_slot_enforces_max_in_flight():
    mgr = ResourceLeaseManager(ResourceBudget(max_in_flight=1))
    with mgr.task_slot():
        with pytest.raises(ResourceBudgetExceeded):
            with mgr.task_slot():
                pass
    with mgr.task_slot():  # released correctly
        pass


def test_byte_leases_enforce_pool_capacity():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=100))
    mgr.lease_bytes(60, "artifact_cache")
    with pytest.raises(ResourceBudgetExceeded):
        mgr.lease_bytes(50, "artifact_cache")
    mgr.release_bytes(60, "artifact_cache")
    mgr.lease_bytes(100, "artifact_cache")
    assert mgr.pool_bytes("artifact_cache") == 100


def test_unlimited_pool_when_budget_none():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=None))
    mgr.lease_bytes(10**12, "artifact_cache")  # no limit configured


def test_lease_manager_rejects_after_close():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=1000))
    mgr.close()
    with pytest.raises(ResourceBudgetExceeded):
        mgr.lease_bytes(1, "artifact_cache")
    with pytest.raises(ResourceBudgetExceeded):
        with mgr.task_slot():
            pass


def test_draw_cache_and_task_scratch_pools_enforced():
    budget = ResourceBudget(
        draw_cache_bytes=100, total_memory_bytes=200, artifact_cache_bytes=50
    )
    mgr = ResourceLeaseManager(budget)
    mgr.lease_bytes(90, "draw_cache")
    with pytest.raises(ResourceBudgetExceeded):
        mgr.lease_bytes(20, "draw_cache")
    mgr.lease_bytes(200, "task_scratch")
    with pytest.raises(ResourceBudgetExceeded):
        mgr.lease_bytes(1, "task_scratch")


def test_task_scratch_unlimited_when_total_memory_unset():
    mgr = ResourceLeaseManager(ResourceBudget())
    mgr.lease_bytes(10**12, "task_scratch")  # no capacity -> no limit

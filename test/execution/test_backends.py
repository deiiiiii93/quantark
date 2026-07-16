"""Ordered, bounded batch execution backends (spec sections 8.2, 11, 12.1-12.2)."""
import threading
import time

import pytest

from quantark.execution.backends import serial, threads
from quantark.execution.contracts import BatchTask
from quantark.execution.errors import ResourceBudgetExceeded
from quantark.execution.leases import ResourceLeaseManager
from quantark.execution.policy import ResourceBudget


class FakePlan:
    def __init__(self, n, est_task_peak_bytes=None, est_outcome_bytes=None):
        self.tasks = tuple(
            BatchTask(plan_id="p", batch_index=i, batch_id=i, n_paths=8)
            for i in range(n)
        )
        self.est_task_peak_bytes = est_task_peak_bytes
        self.est_outcome_bytes = est_outcome_bytes


def test_serial_yields_in_order():
    plan = FakePlan(5)
    out = list(serial.iter_ordered(plan, lambda t: t.batch_index * 10))
    assert out == [(i, i * 10) for i in range(5)]


def test_serial_holds_task_slot_per_batch():
    mgr = ResourceLeaseManager(ResourceBudget(max_in_flight=1))
    plan = FakePlan(3)
    out = list(serial.iter_ordered(
        plan, lambda t: t.batch_index, lease_manager=mgr
    ))
    assert [i for i, _ in out] == [0, 1, 2]
    with mgr.task_slot():  # all slots were released
        pass


def test_threads_yield_in_canonical_order_despite_reversed_completion():
    plan = FakePlan(8)

    def execute(task):
        time.sleep(0.02 * (8 - task.batch_index))  # later batches finish first
        return task.batch_index * 10

    out = list(threads.iter_ordered(plan, execute, workers=8, window=8))
    assert out == [(i, i * 10) for i in range(8)]


def test_threads_bounded_window_and_buffering():
    plan = FakePlan(12)
    seen = {"max_total": 0}
    lock = threading.Lock()

    def observer(in_flight, buffered):
        with lock:
            seen["max_total"] = max(seen["max_total"], in_flight + buffered)

    def execute(task):
        time.sleep(0.01)
        return task.batch_index

    out = list(threads.iter_ordered(
        plan, execute, workers=4, window=4, observer=observer
    ))
    assert [i for i, _ in out] == list(range(12))
    assert seen["max_total"] <= 4  # pending + buffered never exceeds window


def test_threads_stalled_first_batch_cannot_accumulate_unbounded_outcomes():
    # Batch 0 stalls; later batches finish fast. Submission is gated on
    # pending + buffered < window, so completed outcomes cannot pile up.
    plan = FakePlan(12)
    release = threading.Event()
    seen = {"max_buffered": 0, "max_total": 0}
    lock = threading.Lock()

    def observer(in_flight, buffered):
        with lock:
            seen["max_buffered"] = max(seen["max_buffered"], buffered)
            seen["max_total"] = max(seen["max_total"], in_flight + buffered)

    def execute(task):
        if task.batch_index == 0:
            release.wait(timeout=10)
        return task.batch_index

    def unblock():
        time.sleep(0.2)
        release.set()

    threading.Thread(target=unblock).start()
    out = list(threads.iter_ordered(
        plan, execute, workers=4, window=4, observer=observer
    ))
    assert [i for i, _ in out] == list(range(12))
    # retained work (executing + buffered) never exceeds the window, so a
    # stalled batch 0 cannot accumulate unbounded completed outcomes
    assert seen["max_total"] <= 4
    assert seen["max_buffered"] <= 4


def test_threads_respect_max_in_flight_slots():
    # budget.max_in_flight=1 with workers=4: per-batch task slots serialize
    # execution (Codex plan-gate finding: admission control must bind).
    mgr = ResourceLeaseManager(ResourceBudget(max_in_flight=1))
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def execute(task):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.01)
        with lock:
            active["now"] -= 1
        return task.batch_index

    plan = FakePlan(6)
    out = list(threads.iter_ordered(
        plan, execute, workers=4, window=1, lease_manager=mgr
    ))
    assert [i for i, _ in out] == list(range(6))
    assert active["max"] == 1


def test_buffered_outcome_bytes_stay_leased_until_yield():
    mgr = ResourceLeaseManager(
        ResourceBudget(total_memory_bytes=10_000, max_in_flight=4)
    )
    plan = FakePlan(4, est_task_peak_bytes=100, est_outcome_bytes=40)
    observed = []

    def execute(task):
        if task.batch_index == 0:
            time.sleep(0.1)  # others complete first and sit buffered
        return task.batch_index

    for index, _ in threads.iter_ordered(
        plan, execute, workers=4, window=4, lease_manager=mgr
    ):
        observed.append((index, mgr.pool_bytes("task_scratch")))
    # after the final yield every task and outcome lease is back
    assert mgr.pool_bytes("task_scratch") == 0
    assert observed[0][0] == 0


def test_threads_propagates_failure_and_stops():
    plan = FakePlan(6)
    started = []
    lock = threading.Lock()

    def execute(task):
        with lock:
            started.append(task.batch_index)
        if task.batch_index == 1:
            raise ValueError("boom")
        time.sleep(0.05)
        return task.batch_index

    with pytest.raises(ValueError, match="boom"):
        list(threads.iter_ordered(plan, execute, workers=2, window=2))
    assert len(started) < 6  # fail-fast: pending tasks were not all submitted


def test_threads_failure_releases_all_leases():
    mgr = ResourceLeaseManager(
        ResourceBudget(total_memory_bytes=10_000, max_in_flight=3)
    )
    plan = FakePlan(6, est_task_peak_bytes=100, est_outcome_bytes=40)

    def execute(task):
        if task.batch_index == 1:
            raise ValueError("boom")
        time.sleep(0.02)
        return task.batch_index

    with pytest.raises(ValueError, match="boom"):
        list(threads.iter_ordered(
            plan, execute, workers=3, window=3, lease_manager=mgr
        ))
    assert mgr.pool_bytes("task_scratch") == 0
    with mgr.task_slot():  # all slots returned too
        pass


def test_threads_leases_task_scratch_per_in_flight():
    mgr = ResourceLeaseManager(ResourceBudget(total_memory_bytes=250, max_in_flight=2))
    plan = FakePlan(4, est_task_peak_bytes=100)  # window 2 => 200 <= 250 ok
    out = list(threads.iter_ordered(
        plan, lambda t: t.batch_index, workers=2, window=2, lease_manager=mgr
    ))
    assert [i for i, _ in out] == list(range(4))
    assert mgr.pool_bytes("task_scratch") == 0  # all released


def test_single_task_exceeding_budget_fails_before_execution():
    mgr = ResourceLeaseManager(ResourceBudget(total_memory_bytes=50))
    plan = FakePlan(2, est_task_peak_bytes=100)
    executed = []
    with pytest.raises(ResourceBudgetExceeded):
        list(threads.iter_ordered(
            plan, lambda t: executed.append(t), workers=1, window=1,
            lease_manager=mgr,
        ))
    assert executed == []


def test_serial_backend_enforces_byte_admission():
    # Code-gate finding 5: serial must use the same admission primitive.
    mgr = ResourceLeaseManager(ResourceBudget(total_memory_bytes=50))
    plan = FakePlan(2, est_task_peak_bytes=100)
    executed = []
    with pytest.raises(ResourceBudgetExceeded):
        list(serial.iter_ordered(
            plan, lambda t: executed.append(t), lease_manager=mgr
        ))
    assert executed == []
    assert mgr.pool_bytes("task_scratch") == 0


def test_serial_backend_releases_outcome_lease_after_consumption():
    mgr = ResourceLeaseManager(ResourceBudget(total_memory_bytes=1000))
    plan = FakePlan(3, est_task_peak_bytes=100, est_outcome_bytes=40)
    seen = []
    for index, _ in serial.iter_ordered(
        plan, lambda t: t.batch_index, lease_manager=mgr
    ):
        seen.append((index, mgr.pool_bytes("task_scratch")))
    assert [i for i, _ in seen] == [0, 1, 2]
    assert all(b == 40 for _, b in seen)  # outcome charged while consumed
    assert mgr.pool_bytes("task_scratch") == 0

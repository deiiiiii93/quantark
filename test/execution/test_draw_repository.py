"""Session-owned Sobol DrawRepository (spec section 8.3)."""
import numpy as np

from quantark.asset.equity.engine.mc.qmc_draws import qmc_normals
from quantark.execution.cache.draws import DrawRepository
from quantark.execution.leases import ResourceLeaseManager
from quantark.execution.policy import ResourceBudget


def make_repo(draw_bytes=64 * 2**20):
    mgr = ResourceLeaseManager(ResourceBudget(draw_cache_bytes=draw_bytes))
    return DrawRepository(mgr), mgr


def test_values_identical_to_legacy_generator():
    repo, _ = make_repo()
    with repo.normals_handle(seed=42, n_paths=128, dim=16, batch_id=3) as h:
        legacy = qmc_normals(42, 128, 16, batch_id=3)
        np.testing.assert_array_equal(h.value, legacy)


def test_master_is_read_only_and_cached_once():
    repo, mgr = make_repo()
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h1:
        assert not h1.value.flags.writeable
        with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h2:
            assert h2.value is h1.value
    assert repo.stats()["hits"] == 1


def test_writable_copy_is_private():
    repo, _ = make_repo()
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h:
        master = h.value.copy()
    with repo.normals_handle(
        seed=1, n_paths=64, dim=8, batch_id=0, writable=True
    ) as w:
        assert w.value.flags.writeable
        w.value[:] = 0.0
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h2:
        np.testing.assert_array_equal(h2.value, master)


def test_descriptor_completeness_every_field_changes_key():
    base = dict(seed=1, n_paths=64, dim=8, batch_id=0)
    repo, _ = make_repo()
    keys = set()
    for override in ({}, {"seed": 2}, {"n_paths": 128}, {"dim": 16},
                     {"batch_id": 1}, {"batch_id": None}):
        d = repo.descriptor(**{**base, **override})
        keys.add(d.fingerprint)
    assert len(keys) == 6


def test_zero_budget_bypasses_but_stays_correct():
    repo, mgr = make_repo(draw_bytes=0)
    with repo.normals_handle(seed=42, n_paths=64, dim=8, batch_id=0) as h:
        np.testing.assert_array_equal(h.value, qmc_normals(42, 64, 8, batch_id=0))
    assert mgr.pool_bytes("draw_cache") == 0


def test_budget_accounting_exact_bytes():
    repo, mgr = make_repo()
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0):
        assert mgr.pool_bytes("draw_cache") == 64 * 8 * 8

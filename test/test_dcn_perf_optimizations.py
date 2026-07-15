"""Performance-layer invariants: draw cache, batch threads, LV build hoist.

Every optimization here must be BIT-IDENTICAL to the unoptimized path — the
DCN result JSONs claim deterministic economics for a fixed seed, so a cache
hit, a worker count, or a hoisted surface build may change wall-clock only.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
)
from quantark.asset.equity.engine.mc.qmc_draws import qmc_normals, qmc_uniforms
from quantark.montecarlo.qmc_sobol import (
    QMCDrawCache,
    SobolNormalGenerator,
    get_qmc_draw_cache,
)
from quantark.param import GridVolSurface
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn

PATHS = 2 ** 13
BATCHES = 4


@pytest.fixture(autouse=True)
def _fresh_cache():
    get_qmc_draw_cache().clear()
    yield
    get_qmc_draw_cache().clear()


def _grid_env():
    env = flat_env(**FLAT)
    strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    iv = np.full((len(maturities), len(strikes)), FLAT["sigma"])
    env.vol_surface = GridVolSurface(
        strikes=strikes, maturities=maturities, iv_grid=iv
    )
    return env


def _result_fields(result) -> dict:
    d = result.to_dict()
    d.pop("elapsed_seconds")
    d.pop("event_stats")
    return d


# ---------------------------------------------------------------- draw cache

def test_cached_draws_are_bit_identical_and_read_only():
    cache = get_qmc_draw_cache()
    first = qmc_normals(42, 512, 37, batch_id=3)
    assert cache.misses >= 1
    again = qmc_normals(42, 512, 37, batch_id=3)
    assert again is first  # same cached block, not a regeneration
    assert not again.flags.writeable
    reference = SobolNormalGenerator(base_seed=42).normal(512, 37, batch_id=3)
    np.testing.assert_array_equal(first, reference)


def test_writable_draws_are_private_copies():
    block = qmc_uniforms(42, 256, 9, batch_id=0, writable=True)
    assert block.flags.writeable
    block[:] = 0.0
    untouched = qmc_uniforms(42, 256, 9, batch_id=0)
    assert float(untouched.max()) > 0.0


def test_cache_key_separates_kind_seed_batch_and_shape():
    a = qmc_normals(42, 128, 8, batch_id=0)
    for other in (
        qmc_normals(43, 128, 8, batch_id=0),
        qmc_normals(42, 128, 8, batch_id=1),
        qmc_normals(42, 128, 9, batch_id=0),
        qmc_uniforms(42, 128, 8, batch_id=0),
    ):
        assert other is not a
        assert not np.array_equal(other, a)


def test_lru_eviction_respects_byte_budget():
    block = np.arange(1024, dtype=float)  # 8 KiB
    cache = QMCDrawCache(max_bytes=3 * block.nbytes)
    for key in ("a", "b", "c"):
        cache.put(key, block.copy())
    cache.get("a")  # refresh 'a' so 'b' is the LRU entry
    cache.put("d", block.copy())
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("d") is not None
    assert cache.current_bytes <= cache.max_bytes

    tiny = QMCDrawCache(max_bytes=8)
    returned = tiny.put("huge", block.copy())  # larger than the whole budget
    assert tiny.get("huge") is None  # not cached ...
    np.testing.assert_array_equal(returned, block)  # ... but still returned


def test_inplace_normal_transform_matches_copying_reference():
    from scipy import special
    from scipy.stats import qmc as scipy_qmc

    engine = scipy_qmc.Sobol(d=13, scramble=True, seed=99)
    u = engine.random_base2(8)
    reference = special.ndtri(np.clip(u, 1e-12, 1 - 1e-12))

    z = SobolNormalGenerator(base_seed=99).normal(256, 13)
    np.testing.assert_array_equal(z, reference)


# --------------------------------------------------------- batch thread pool

def test_thread_workers_are_bit_identical_to_serial():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    serial = DCNMCEngine(
        num_paths=PATHS, seed=42, num_batches=BATCHES, num_workers=1
    ).price_detailed(p, env)
    threaded = DCNMCEngine(
        num_paths=PATHS, seed=42, num_batches=BATCHES, num_workers=4
    ).price_detailed(p, env)
    assert _result_fields(serial) == _result_fields(threaded)


def test_thread_workers_bit_identical_for_lv_and_heston():
    p = make_dcn(DCN_A)
    env = _grid_env()
    lv_serial = LocalVolDCNMCEngine(
        num_paths=PATHS, seed=42, num_batches=BATCHES, num_workers=1
    ).price_detailed(p, env)
    lv_threaded = LocalVolDCNMCEngine(
        num_paths=PATHS, seed=42, num_batches=BATCHES, num_workers=4
    ).price_detailed(p, env)
    assert _result_fields(lv_serial) == _result_fields(lv_threaded)

    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.6, rho=-0.3)
    flat = flat_env(**FLAT)
    he_serial = HestonDCNMCEngine(
        model_params=params, scheme=HestonMCScheme.QUADEXP_M,
        num_paths=PATHS, seed=42, num_batches=BATCHES, num_workers=1,
    ).price_detailed(p, flat)
    he_threaded = HestonDCNMCEngine(
        model_params=params, scheme=HestonMCScheme.QUADEXP_M,
        num_paths=PATHS, seed=42, num_batches=BATCHES, num_workers=4,
    ).price_detailed(p, flat)
    assert _result_fields(he_serial) == _result_fields(he_threaded)


def test_warm_cache_repricing_is_bit_identical():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    engine = DCNMCEngine(num_paths=PATHS, seed=42, num_batches=BATCHES)
    cold = engine.price_detailed(p, env)   # populates the cache
    warm = engine.price_detailed(p, env)   # served from the cache
    assert get_qmc_draw_cache().hits >= BATCHES
    assert _result_fields(cold) == _result_fields(warm)


def test_num_workers_validation():
    with pytest.raises(ValidationError):
        DCNMCEngine(num_paths=64, num_workers=0)
    with pytest.raises(ValidationError):
        DCNMCEngine(num_paths=64, num_workers=True)


# ------------------------------------------------------------ LV build hoist

class _CountingLVEngine(LocalVolDCNMCEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.builds = 0

    def _build_surface(self, env):
        self.builds += 1
        return super()._build_surface(env)


def test_lv_surface_built_once_per_price_call():
    p = make_dcn(DCN_A)
    env = _grid_env()
    engine = _CountingLVEngine(
        num_paths=PATHS, seed=42, num_batches=BATCHES
    )
    engine.price_detailed(p, env)
    assert engine.builds == 1  # once per call, not once per batch

    engine.price_detailed(p, _grid_env())  # bumped env -> fresh inversion
    assert engine.builds == 2


def test_lv_hoist_is_bit_identical_to_per_batch_rebuild():
    p = make_dcn(DCN_A)
    env = _grid_env()
    hoisted = LocalVolDCNMCEngine(
        num_paths=PATHS, seed=42, num_batches=BATCHES
    ).price_detailed(p, env)

    class _LegacyLVEngine(LocalVolDCNMCEngine):
        def _prepare_simulation(self, product, pricing_env):
            pass  # per-batch rebuild, as before the hoist

    legacy = _LegacyLVEngine(
        num_paths=PATHS, seed=42, num_batches=BATCHES
    ).price_detailed(p, env)
    assert _result_fields(hoisted) == _result_fields(legacy)


def test_direct_simulate_never_uses_a_stale_surface():
    p = make_dcn(DCN_A)
    env_a = _grid_env()
    engine = _CountingLVEngine(num_paths=64, seed=42)
    engine.price_detailed(p, env_a)

    env_b = _grid_env()
    env_b.vol_surface = GridVolSurface(
        strikes=env_b.vol_surface.strikes,
        maturities=env_b.vol_surface.maturities,
        iv_grid=np.asarray(env_b.vol_surface.iv_grid) + 0.10,
    )
    before = engine.builds
    # identity check: a foreign env must trigger a fresh build, never reuse
    # the surface prepared for env_a
    surface = engine._resolve_surface(env_b)
    assert engine.builds == before + 1
    assert surface is not engine._active_surface

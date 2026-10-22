"""generate_paths must route through the shared GBM kernel byte-stably.

These tests recompute what the generator must produce via the NumPy reference
kernel and compare bytes, so they hold whether or not the numba accelerator is
installed -- pinning that the extraction changed no arithmetic.
"""

import numpy as np

from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.process.bsm.qmc_variance_reduction import (
    VarianceReductionConfig,
    apply_variance_reduction_to_normals,
)
from quantark.montecarlo import gbm_kernels


def _gen(stream, is_qmc, vr=None, bridge=False, n_paths=513, n_steps=63):
    k = np.arange(n_steps)
    vol = 0.18 + 0.06 * np.sin(2 * np.pi * k / n_steps) ** 2
    return GBMPathGenerator(
        initial_value=100.0,
        vol=vol,
        rrf=0.03 + 0.02 * k / n_steps,
        div=0.01 + 0.0 * k,
        maturity=1.0,
        time_steps=n_steps,
        num_paths=n_paths,
        model="bsm",
        random_stream=stream,
        use_brownian_bridge=bridge,
        vr_config=vr,
        is_qmc=is_qmc,
    )


def _reference_paths(gen, batch_id=None):
    """Recompute what generate_paths must produce, via the NumPy reference."""
    z = gen._generate_base_normals(batch_id=batch_id)
    z, _, _ = apply_variance_reduction_to_normals(
        n_paths=gen.num_paths,
        dim=gen.time_steps,
        base_normals=z,
        vr_config=gen.vr_config,
        is_qmc=gen.is_qmc,
    )
    dW = gen._build_brownian_increments(z)
    drift_dt = (gen._drift_vec - 0.5 * gen._vol_vec * gen._vol_vec) * gen.dt_vector
    return gbm_kernels.gbm_path_tail_numpy(
        dW, drift_dt, gen._vol_vec, gen.initial_value
    )


def test_pseudo_paths_match_reference_bitwise():
    a, _ = _gen(PseudoRandomNormalGenerator(seed=42), False).generate_paths()
    b = _reference_paths(_gen(PseudoRandomNormalGenerator(seed=42), False))
    assert a.tobytes() == b.tobytes()


def test_sobol_bridge_paths_match_reference_bitwise():
    a, _ = _gen(SobolNormalGenerator(base_seed=7), True, bridge=True).generate_paths(
        batch_id=3
    )
    b = _reference_paths(
        _gen(SobolNormalGenerator(base_seed=7), True, bridge=True), batch_id=3
    )
    assert a.tobytes() == b.tobytes()


def test_antithetic_paths_match_reference_bitwise():
    vr = VarianceReductionConfig(antithetic=True)
    a, _ = _gen(PseudoRandomNormalGenerator(seed=5), False, vr=vr).generate_paths()
    b = _reference_paths(_gen(PseudoRandomNormalGenerator(seed=5), False, vr=vr))
    assert a.tobytes() == b.tobytes()


def test_aux_contract_unchanged():
    gen = _gen(PseudoRandomNormalGenerator(seed=1), False)
    paths, aux = gen.generate_paths(batch_id=2, return_aux=True)
    assert aux["batch_id"] == np.array(2)
    assert paths.shape == (gen.num_paths, gen.time_steps + 1)

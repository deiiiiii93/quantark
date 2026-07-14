"""Coupled Heston timestep-ladder engines (MLMC-style pair coupling)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    CoupledCoarseHestonDCNMCEngine,
    HestonDCNMCEngine,
    coupled_heston_ladder_pair,
)
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn

# Deliberately Feller-violating (2*kappa*theta << sigma^2), matching the
# regime that motivated the coupled ladder in the first place.
FELLER_VIOLATING = HestonParams(
    v0=0.0249, kappa=9.92, theta=0.045, sigma=1.583, rho=-0.025,
)
PATHS = 2 ** 12


def _pair(scheme, coarse_substeps=2, seed=42, **kwargs):
    return coupled_heston_ladder_pair(
        FELLER_VIOLATING,
        coarse_substeps,
        scheme,
        num_paths=PATHS,
        seed=seed,
        use_sobol=True,
        num_batches=1,
        **kwargs,
    )


def test_pair_factory_builds_consistent_pair():
    coarse, fine = _pair(HestonMCScheme.QUADEXP_M)
    assert isinstance(coarse, CoupledCoarseHestonDCNMCEngine)
    assert isinstance(fine, HestonDCNMCEngine)
    assert coarse.substeps_per_interval == 2
    assert fine.substeps_per_interval == 4
    assert coarse.seed == fine.seed
    assert coarse.scheme is fine.scheme
    assert coarse.model_params is fine.model_params


def test_constructor_rejects_inconsistent_pairs():
    fine = HestonDCNMCEngine(
        model_params=FELLER_VIOLATING, substeps_per_interval=4,
        scheme=HestonMCScheme.QUADEXP_M, num_paths=PATHS, seed=42,
        use_sobol=True,
    )
    with pytest.raises(ValidationError):  # wrong ratio
        CoupledCoarseHestonDCNMCEngine(
            fine_engine=fine, model_params=FELLER_VIOLATING,
            substeps_per_interval=3, scheme=HestonMCScheme.QUADEXP_M,
            num_paths=PATHS, seed=42, use_sobol=True,
        )
    with pytest.raises(ValidationError):  # seed mismatch
        CoupledCoarseHestonDCNMCEngine(
            fine_engine=fine, model_params=FELLER_VIOLATING,
            substeps_per_interval=2, scheme=HestonMCScheme.QUADEXP_M,
            num_paths=PATHS, seed=43, use_sobol=True,
        )
    with pytest.raises(ValidationError):  # scheme mismatch
        CoupledCoarseHestonDCNMCEngine(
            fine_engine=fine, model_params=FELLER_VIOLATING,
            substeps_per_interval=2,
            scheme=HestonMCScheme.FULL_TRUNCATION_EULER,
            num_paths=PATHS, seed=42, use_sobol=True,
        )
    with pytest.raises(ValidationError):  # params not shared
        CoupledCoarseHestonDCNMCEngine(
            fine_engine=fine,
            model_params=HestonParams(
                v0=0.0249, kappa=9.92, theta=0.045, sigma=1.583, rho=-0.025,
            ),
            substeps_per_interval=2, scheme=HestonMCScheme.QUADEXP_M,
            num_paths=PATHS, seed=42, use_sobol=True,
        )


@pytest.mark.parametrize(
    "scheme",
    [HestonMCScheme.QUADEXP_M, HestonMCScheme.FULL_TRUNCATION_EULER],
)
def test_coarse_draws_are_pair_aggregated_fine_draws(scheme):
    coarse, fine = _pair(scheme)
    n_steps = 6
    z_var_f, z_ind_f, u_var_f = fine._heston_draws(2 * n_steps, 64, None)
    z_var_c, z_ind_c, u_var_c = coarse._heston_draws(n_steps, 64, None)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(
        z_var_c, (z_var_f[:, 0::2] + z_var_f[:, 1::2]) * inv_sqrt2
    )
    np.testing.assert_allclose(
        z_ind_c, (z_ind_f[:, 0::2] + z_ind_f[:, 1::2]) * inv_sqrt2
    )
    if u_var_f is None:
        assert u_var_c is None
    else:
        np.testing.assert_allclose(u_var_c, u_var_f[:, 0::2])
        # aggregated Gaussians stay standardized; uniforms stay in (0, 1)
        assert abs(float(z_var_c.mean())) < 0.1
        assert abs(float(z_var_c.std()) - 1.0) < 0.1
        assert 0.0 < u_var_c.min() and u_var_c.max() < 1.0


def test_coupled_pair_is_deterministic():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    coarse1, fine1 = _pair(HestonMCScheme.QUADEXP_M)
    coarse2, fine2 = _pair(HestonMCScheme.QUADEXP_M)
    assert coarse1.price(p, env) == coarse2.price(p, env)
    assert fine1.price(p, env) == fine2.price(p, env)


@pytest.mark.parametrize(
    "scheme",
    [HestonMCScheme.QUADEXP_M, HestonMCScheme.FULL_TRUNCATION_EULER],
)
def test_coupled_coarse_marginal_matches_independent_coarse(scheme):
    # Deriving draws from the fine block must not change the coarse
    # estimator's law: compare batch-mean PVs against a standard coarse
    # engine on independent seeds, within joint RQMC noise.
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    seeds = [42 + b for b in range(8)]
    coupled_pvs, independent_pvs = [], []
    for seed in seeds:
        coarse, _ = _pair(scheme, seed=seed)
        coupled_pvs.append(coarse.price(p, env))
        independent_pvs.append(HestonDCNMCEngine(
            model_params=FELLER_VIOLATING, substeps_per_interval=2,
            scheme=scheme, num_paths=PATHS, seed=seed + 900_000,
            use_sobol=True, num_batches=1,
        ).price(p, env))
    coupled_pvs = np.asarray(coupled_pvs)
    independent_pvs = np.asarray(independent_pvs)
    joint_se = np.hypot(
        coupled_pvs.std(ddof=1), independent_pvs.std(ddof=1)
    ) / np.sqrt(len(seeds))
    assert abs(coupled_pvs.mean() - independent_pvs.mean()) < 4.0 * joint_se


@pytest.mark.parametrize(
    "scheme,min_reduction",
    [
        (HestonMCScheme.FULL_TRUNCATION_EULER, 1.2),
        (HestonMCScheme.QUADEXP_M, 1.5),
    ],
)
def test_coupled_ladder_reduces_difference_variance(scheme, min_reduction):
    # The point of the coupling: coarse-minus-fine batch differences must be
    # less noisy than the independent-scramble design. Counterintuitively,
    # QE-M couples BETTER than FTE here despite FTE's exact Brownian
    # aggregation: in this Feller-violating regime FTE's v=0 truncation
    # events flip between resolutions and destroy pathwise closeness, while
    # QE-M's near-exact transition sampling keeps the resolutions aligned.
    # Thresholds reflect measured deterministic ratios (~1.5x FTE, ~2x QE-M
    # in SD) with margin.
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    seeds = [42 + b for b in range(8)]
    coupled_diffs, independent_diffs = [], []
    for seed in seeds:
        coarse, fine = _pair(scheme, seed=seed)
        coupled_diffs.append(coarse.price(p, env) - fine.price(p, env))
        indep_coarse = HestonDCNMCEngine(
            model_params=FELLER_VIOLATING, substeps_per_interval=2,
            scheme=scheme, num_paths=PATHS, seed=seed + 900_000,
            use_sobol=True, num_batches=1,
        ).price(p, env)
        indep_fine = HestonDCNMCEngine(
            model_params=FELLER_VIOLATING, substeps_per_interval=4,
            scheme=scheme, num_paths=PATHS, seed=seed + 950_000,
            use_sobol=True, num_batches=1,
        ).price(p, env)
        independent_diffs.append(indep_coarse - indep_fine)
    coupled_sd = float(np.std(coupled_diffs, ddof=1))
    independent_sd = float(np.std(independent_diffs, ddof=1))
    assert coupled_sd * min_reduction < independent_sd

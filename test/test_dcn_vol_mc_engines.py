"""Surface-aware DCN MC engines (spec WP1.5): flat-surface consistency."""
from types import SimpleNamespace

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
    QEDCNMCEngine,
)
from quantark.param import GridVolSurface
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn

PATHS = 2 ** 14


def _grid_env():
    env = flat_env(**FLAT)
    strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    iv = np.full((len(maturities), len(strikes)), FLAT["sigma"])
    env.vol_surface = GridVolSurface(
        strikes=strikes, maturities=maturities, iv_grid=iv
    )
    return env


def test_local_vol_flat_surface_matches_gbm():
    # On a flat surface, Dupire local vol == flat vol -> LV engine must agree
    # with the GBM engine well within joint MC noise.
    p = make_dcn(DCN_A)
    env = _grid_env()
    gbm = DCNMCEngine(num_paths=PATHS, seed=42).price_detailed(p, env)
    lv = LocalVolDCNMCEngine(num_paths=PATHS, seed=42).price_detailed(p, env)
    assert abs(lv.pv - gbm.pv) < 4.0 * max(lv.std_error, gbm.std_error)


def test_heston_degenerate_params_match_gbm():
    # vol-of-vol -> 0 with v0 = theta = sigma_flat^2 collapses Heston to GBM
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    params = HestonParams(
        v0=FLAT["sigma"] ** 2, kappa=1.0, theta=FLAT["sigma"] ** 2,
        sigma=1e-6, rho=0.0,
    )
    gbm = DCNMCEngine(num_paths=PATHS, seed=42).price_detailed(p, env)
    hz = HestonDCNMCEngine(
        model_params=params, num_paths=PATHS, seed=42
    ).price_detailed(p, env)
    assert abs(hz.pv - gbm.pv) < 4.0 * max(hz.std_error, gbm.std_error)


def test_heston_default_preserves_legacy_fte_seeded_result():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    params = HestonParams(
        v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5
    )
    common = dict(model_params=params, num_paths=2 ** 10, seed=42)

    default = HestonDCNMCEngine(**common).price_detailed(p, env)
    explicit = HestonDCNMCEngine(
        **common, scheme=HestonMCScheme.FULL_TRUNCATION_EULER
    ).price_detailed(p, env)

    assert default.pv == explicit.pv == -68457.58049657497
    assert default.std_error == explicit.std_error == 6916.175942931679


def test_leg_invariant_holds_for_vol_engines():
    p = make_dcn(DCN_A)
    r = LocalVolDCNMCEngine(num_paths=PATHS, seed=7).price_detailed(
        p, _grid_env()
    )
    assert r.pv == r.pv_fixed_coupons + r.pv_ko_coupons + r.pv_loss_leg

    heston = QEDCNMCEngine(
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5),
        num_paths=2 ** 9,
        seed=7,
    ).price_detailed(p, flat_env(**FLAT))
    assert heston.pv == (
        heston.pv_fixed_coupons
        + heston.pv_ko_coupons
        + heston.pv_loss_leg
    )


def test_heston_rejects_invalid_substeps():
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
    with pytest.raises(ValidationError, match="substeps_per_interval"):
        HestonDCNMCEngine(model_params=params, substeps_per_interval=0)


def test_heston_validates_params_and_scheme_strings():
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)

    assert HestonDCNMCEngine(params, scheme="quadexp").scheme is (
        HestonMCScheme.QUADEXP
    )
    assert HestonDCNMCEngine(
        params, scheme="full_truncation_euler"
    ).scheme is HestonMCScheme.FULL_TRUNCATION_EULER
    with pytest.raises(ValidationError, match="does not support EULER"):
        HestonDCNMCEngine(params, scheme=HestonMCScheme.EULER)
    with pytest.raises(ValidationError, match="Invalid DCN Heston MC scheme"):
        HestonDCNMCEngine(params, scheme="not-a-scheme")
    with pytest.raises(ValidationError, match="model_params"):
        HestonDCNMCEngine(object())
    with pytest.raises(ValidationError, match="requires use_sobol=True"):
        HestonDCNMCEngine(
            params, use_sobol=False, fixed_three_stream_sobol=True
        )


def test_qe_dcn_convenience_and_exports():
    from quantark.asset.equity.engine import QEDCNMCEngine as ExportedQEDCNMCEngine

    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
    assert ExportedQEDCNMCEngine is QEDCNMCEngine
    assert QEDCNMCEngine(params).scheme is HestonMCScheme.QUADEXP
    corrected = QEDCNMCEngine(params, martingale_correction=True)
    assert corrected.scheme is HestonMCScheme.QUADEXP_M
    with pytest.raises(ValidationError, match="fixes scheme"):
        QEDCNMCEngine(params, scheme=HestonMCScheme.QUADEXP)


def test_fixed_three_stream_sobol_pairs_fte_and_qe_normals():
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
    common = dict(model_params=params, num_paths=16, seed=17)
    fte = HestonDCNMCEngine(
        **common,
        scheme=HestonMCScheme.FULL_TRUNCATION_EULER,
        fixed_three_stream_sobol=True,
    )
    qe = HestonDCNMCEngine(**common, scheme=HestonMCScheme.QUADEXP)

    fte_draws = fte._heston_draws(5, 16, None)
    qe_draws = qe._heston_draws(5, 16, None)
    for fte_block, qe_block in zip(fte_draws, qe_draws):
        assert np.array_equal(fte_block, qe_block)

    legacy = HestonDCNMCEngine(**common)
    z_var, z_ind, u_var = legacy._heston_draws(5, 16, None)
    expected = legacy._draws(10, 16, None).reshape(-1, 2, 5)
    assert np.array_equal(z_var, expected[:, 0, :])
    assert np.array_equal(z_ind, expected[:, 1, :])
    assert u_var is None


def test_qe_pseudo_antithetic_pairs_normals_and_uniforms():
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
    engine = HestonDCNMCEngine(
        params,
        scheme=HestonMCScheme.QUADEXP,
        num_paths=10,
        seed=29,
        use_sobol=False,
        use_antithetic=True,
    )

    z_var, z_ind, u_var = engine._heston_draws(4, 10, None)
    assert np.array_equal(z_var[5:], -z_var[:5])
    assert np.array_equal(z_ind[5:], -z_ind[:5])
    assert np.array_equal(u_var[5:], 1.0 - u_var[:5])


@pytest.mark.parametrize(
    "scheme",
    [
        HestonMCScheme.FULL_TRUNCATION_EULER,
        HestonMCScheme.EULERLOG,
        HestonMCScheme.QUADEXP,
        HestonMCScheme.QUADEXP_M,
    ],
)
def test_heston_substeps_preserve_contract_grid_output(scheme):
    p = make_dcn(DCN_A)
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
    result = HestonDCNMCEngine(
        model_params=params,
        substeps_per_interval=2,
        scheme=scheme,
        num_paths=2 ** 8,
        seed=42,
    ).price_detailed(p, flat_env(**FLAT))
    assert np.isfinite(result.pv)
    assert result.num_paths == 2 ** 8


def test_dcn_qe_runs_feller_violated_exponential_branch():
    # At the first half-year step psi ~= 1.62 > psi_c=1.5, so this exercises
    # Andersen's mass-at-zero/exponential branch through the DCN path adapter.
    params = HestonParams(
        v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7
    )
    assert not params.feller_satisfied()
    engine = HestonDCNMCEngine(
        params,
        scheme=HestonMCScheme.QUADEXP,
        num_paths=2 ** 12,
        seed=23,
    )
    dt = np.asarray([0.5, 0.5])
    term = SimpleNamespace(
        rrf=np.asarray([0.03, 0.03]),
        div=np.asarray([0.01, 0.01]),
    )

    nodes = engine._simulate(
        100.0, term, dt, pricing_env=None, n_paths=2 ** 12, batch_id=None
    )

    assert nodes.shape == (2 ** 12, 3)
    assert np.all(np.isfinite(nodes))
    assert np.all(nodes > 0.0)
    assert nodes[:, -1].mean() == pytest.approx(100.0 * np.exp(0.02), rel=0.02)

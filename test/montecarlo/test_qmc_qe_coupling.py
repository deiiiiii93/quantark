import numpy as np
from scipy.special import ndtri

from quantark.montecarlo.qmc_brownian_bridge import BrownianBridge
from quantark.montecarlo.qmc_qe_coupling import (
    CoupledQESubstepDrawProvider,
    invert_brownian_bridge,
)


def test_invert_brownian_bridge_round_trip_on_nonuniform_grid():
    rng = np.random.default_rng(731)
    times = np.array([0.03, 0.11, 0.27, 0.65, 1.0])
    z = rng.standard_normal((17, times.size))
    bridge = BrownianBridge.from_time_grid(times)
    increments = bridge.transform(z)
    recovered = invert_brownian_bridge(increments, times)
    np.testing.assert_allclose(recovered, z, rtol=0.0, atol=2e-15)


def test_coupled_qe_target_spot_increments_are_aggregated_fine_increments():
    target_dt = np.array([0.10, 0.15, 0.25])
    fine_dt = np.repeat(target_dt / 2.0, 2)
    common = dict(
        seed=20260803,
        n_paths=64,
        target_dt=target_dt,
        fine_dt=fine_dt,
    )
    target = CoupledQESubstepDrawProvider(role="target", **common)
    fine = CoupledQESubstepDrawProvider(role="fine", **common)
    z_var_target, z_spot_target, u_target = target.draws(
        n_paths=64, dt_array=target_dt, batch_id=3
    )
    z_var_fine, z_spot_fine, u_fine = fine.draws(
        n_paths=64, dt_array=fine_dt, batch_id=3
    )

    np.testing.assert_allclose(
        z_var_target,
        z_var_fine.reshape(64, 3, 2).sum(axis=2) / np.sqrt(2.0),
        rtol=0.0,
        atol=2e-15,
    )
    d_w_target = BrownianBridge.from_time_grid(
        np.cumsum(target_dt)
    ).transform(z_spot_target)
    d_w_fine = BrownianBridge.from_time_grid(
        np.cumsum(fine_dt)
    ).transform(z_spot_fine)
    np.testing.assert_allclose(
        d_w_target,
        d_w_fine.reshape(64, 3, 2).sum(axis=2),
        rtol=0.0,
        atol=3e-15,
    )
    assert np.all((u_target > 0.0) & (u_target < 1.0))
    assert u_fine.shape == (64, 6)
    z_u_fine = ndtri(u_fine)
    z_u_target = ndtri(u_target)
    np.testing.assert_allclose(
        z_u_target,
        (
            z_u_fine * np.sqrt(fine_dt)[None, :]
        ).reshape(64, 3, 2).sum(axis=2)
        / np.sqrt(target_dt)[None, :],
        rtol=0.0,
        atol=6e-15,
    )
    assert target.dimension == fine.dimension == 18


def test_coupled_qe_draw_cache_is_bounded_to_declared_reuses():
    target_dt = np.array([0.25, 0.25])
    fine_dt = np.repeat(target_dt / 2.0, 2)
    provider = CoupledQESubstepDrawProvider(
        seed=20260803,
        n_paths=16,
        target_dt=target_dt,
        fine_dt=fine_dt,
        role="fine",
        reuse_count=3,
    )

    first = provider.draws(n_paths=16, dt_array=fine_dt, batch_id=7)
    second = provider.draws(n_paths=16, dt_array=fine_dt, batch_id=7)
    third = provider.draws(n_paths=16, dt_array=fine_dt, batch_id=7)

    assert all(left is right for left, right in zip(first, second))
    assert all(left is right for left, right in zip(first, third))
    assert not provider._draw_cache
    assert all(not values.flags.writeable for values in first)

    fourth = provider.draws(n_paths=16, dt_array=fine_dt, batch_id=7)
    assert all(left is not right for left, right in zip(first, fourth))
    for left, right in zip(first, fourth):
        assert np.array_equal(left, right)


def test_coupled_qe_path_counts_share_an_exact_nested_sobol_prefix():
    target_dt = np.array([0.10, 0.15, 0.25])
    fine_dt = np.repeat(target_dt / 2.0, 2)
    common = dict(seed=20260803, target_dt=target_dt, fine_dt=fine_dt)

    for role, requested_dt in (("target", target_dt), ("fine", fine_dt)):
        low = CoupledQESubstepDrawProvider(
            n_paths=16, role=role, **common
        ).draws(n_paths=16, dt_array=requested_dt, batch_id=11)
        high = CoupledQESubstepDrawProvider(
            n_paths=64, role=role, **common
        ).draws(n_paths=64, dt_array=requested_dt, batch_id=11)

        for low_stream, high_stream in zip(low, high):
            np.testing.assert_array_equal(low_stream, high_stream[:16])

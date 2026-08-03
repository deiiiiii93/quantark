import numpy as np
import pytest

from quantark.montecarlo import RQMCRunSpec, run_paired_rqmc_greeks


class _QuadraticGenerator:
    def __init__(self, spot, num_paths=8):
        self.spot = float(spot)
        self.num_paths = int(num_paths)

    def generate_paths(self, seed=None, batch_id=None, return_aux=False):
        batch = 0 if batch_id is None else int(batch_id)
        rng = np.random.default_rng(1000 + batch)
        noise = rng.normal(size=self.num_paths)
        paths = np.column_stack([np.full(self.num_paths, self.spot), noise])
        aux = {"batch_id": batch} if return_aux else None
        return paths, aux


def _spec(
    spot,
    *,
    dimension=2,
    paths=8,
    max_batches=8,
    randomization_key=("fake_scramble", 1000),
    with_control=False,
):
    generator = _QuadraticGenerator(spot, paths)

    def pricer(path_array, aux):
        # Identical batch noise cancels only if the three calls are genuinely
        # coupled by batch id.  The deterministic spot response is quadratic.
        return path_array[:, 0] ** 2 + path_array[:, 1]

    def control_pricer(path_array, aux):
        return 3.0 * path_array[:, 0] + path_array[:, 1]

    return RQMCRunSpec(
        pricer_fn=pricer,
        path_generator=generator,
        max_batches=max_batches,
        min_batches=2,
        target_std=1e-6,
        paths_per_batch=paths,
        time_steps=1,
        scheme="quadratic/rqmc",
        finalize=lambda result: result,
        dimension=dimension,
        randomization_key=randomization_key,
        control_pricer_fn=control_pricer if with_control else None,
    )


def test_paired_rqmc_forms_greeks_inside_each_common_scramble():
    spot = 100.0
    result = run_paired_rqmc_greeks(
        _spec(99.0),
        _spec(100.0),
        _spec(101.0),
        spot=spot,
        relative_bump=0.01,
        batches=8,
    )

    assert result.delta == pytest.approx(2.0 * spot, abs=1e-12)
    assert result.gamma == pytest.approx(2.0, abs=1e-12)
    assert result.delta_std_error == pytest.approx(0.0, abs=1e-12)
    assert result.gamma_std_error == pytest.approx(0.0, abs=1e-12)
    assert result.batch_estimates.shape == (8, 5)
    assert result.total_unique_paths == 64
    assert result.total_path_valuations == 192
    assert result.as_dict()["batches_used"] == 8


def test_paired_rqmc_batch_parallelism_is_bitwise_reproducible():
    common = dict(spot=100.0, relative_bump=0.01, batches=8)
    serial = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0), **common
    )
    threaded = run_paired_rqmc_greeks(
        _spec(99.0),
        _spec(100.0),
        _spec(101.0),
        batch_workers=4,
        **common,
    )

    assert np.array_equal(serial.batch_estimates, threaded.batch_estimates)
    assert np.array_equal(serial.covariance, threaded.covariance)
    for name in (
        "price",
        "price_std_error",
        "delta",
        "delta_std_error",
        "gamma",
        "gamma_std_error",
    ):
        assert getattr(serial, name) == getattr(threaded, name)


def test_paired_rqmc_preserves_conditional_control_by_scramble():
    result = run_paired_rqmc_greeks(
        _spec(99.0, with_control=True),
        _spec(100.0, with_control=True),
        _spec(101.0, with_control=True),
        spot=100.0,
        batches=8,
    )

    assert result.control_batch_estimates.shape == (8, 5)
    np.testing.assert_allclose(result.control_batch_estimates[:, 3], 3.0)
    np.testing.assert_allclose(result.control_batch_estimates[:, 4], 0.0, atol=1e-12)
    assert "control_batch_estimates" in result.as_dict()


def test_paired_rqmc_rejects_partial_control_specs():
    with pytest.raises(ValueError, match="all provide the conditional control"):
        run_paired_rqmc_greeks(
            _spec(99.0, with_control=True),
            _spec(100.0),
            _spec(101.0, with_control=True),
            spot=100.0,
            batches=4,
        )


def test_paired_rqmc_rejects_specs_that_cannot_share_one_point_set():
    with pytest.raises(ValueError, match="same dimension"):
        run_paired_rqmc_greeks(
            _spec(99.0, dimension=2),
            _spec(100.0, dimension=3),
            _spec(101.0, dimension=2),
            spot=100.0,
            batches=4,
        )


def test_paired_rqmc_rejects_unproven_or_different_randomization():
    with pytest.raises(ValueError, match="randomization_key"):
        run_paired_rqmc_greeks(
            _spec(99.0, randomization_key=None),
            _spec(100.0),
            _spec(101.0),
            spot=100.0,
            batches=4,
        )

    with pytest.raises(ValueError, match="same randomization_key"):
        run_paired_rqmc_greeks(
            _spec(99.0),
            _spec(100.0),
            _spec(101.0, randomization_key=("fake_scramble", 9999)),
            spot=100.0,
            batches=4,
        )

    with pytest.raises(ValueError, match="at least two batches"):
        run_paired_rqmc_greeks(
            _spec(99.0),
            _spec(100.0),
            _spec(101.0),
            spot=100.0,
            batches=1,
        )

import numpy as np
import pytest

from quantark.montecarlo import (
    RQMCRunSpec,
    concatenate_paired_results,
    run_paired_rqmc_greeks,
)


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


class _HomogeneousGenerator:
    def __init__(self, spot, num_paths=8):
        self.spot = float(spot)
        self.num_paths = int(num_paths)
        self.calls = 0

    def generate_paths(self, seed=None, batch_id=None, return_aux=False):
        self.calls += 1
        batch = 0 if batch_id is None else int(batch_id)
        rng = np.random.default_rng(1000 + batch)
        noise = rng.normal(size=self.num_paths)
        paths = np.column_stack(
            [
                np.full(self.num_paths, self.spot),
                self.spot * (1.0 + 0.001 * noise),
            ]
        )
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


def test_paired_rqmc_generates_homogeneous_spot_paths_once_per_scramble():
    specs = []
    generators = []
    for shifted_spot in (99.0, 100.0, 101.0):
        generator = _HomogeneousGenerator(shifted_spot)
        generators.append(generator)

        def pricer(path_array, aux):
            return path_array[:, 0] ** 2 + path_array[:, 1] / path_array[:, 0]

        specs.append(
            RQMCRunSpec(
                pricer_fn=pricer,
                path_generator=generator,
                max_batches=8,
                min_batches=2,
                target_std=1e-6,
                paths_per_batch=8,
                time_steps=1,
                scheme="homogeneous/rqmc",
                finalize=lambda result: result,
                dimension=2,
                randomization_key=("fake_scramble", 1000),
                homogeneous_spot_scaling=True,
                initial_spot=shifted_spot,
            )
        )

    result = run_paired_rqmc_greeks(
        *specs,
        spot=100.0,
        relative_bump=0.01,
        batches=8,
    )

    assert result.delta == pytest.approx(200.0, abs=1e-12)
    assert result.gamma == pytest.approx(2.0, abs=1e-12)
    assert [generator.calls for generator in generators] == [0, 8, 0]


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


def test_batch_range_chunks_concatenate_into_the_whole_run_bitwise():
    """A cell priced in chunks must bank exactly what one long run banks.

    This is the property that licenses gate-driven stopping: the loop prices a
    chunk, evaluates the certification gate, and continues only if undecided.
    If extending a run perturbed the batches already computed, the accumulated
    mean would not be the mean of a fixed point set and the banked evidence
    would be rewritten by the act of continuing.
    """
    common = dict(spot=100.0, relative_bump=0.01)
    whole = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0), batches=8, **common
    )
    first = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0),
        batches=4, first_batch=0, **common,
    )
    second = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0),
        batches=4, first_batch=4, **common,
    )
    combined = concatenate_paired_results([first, second])

    assert combined.batch_estimates.tobytes() == whole.batch_estimates.tobytes()
    assert combined.covariance.tobytes() == whole.covariance.tobytes()
    assert combined.batches_used == whole.batches_used == 8
    assert combined.total_unique_paths == whole.total_unique_paths
    assert combined.total_path_valuations == whole.total_path_valuations
    for name in (
        "price",
        "price_std_error",
        "delta",
        "delta_std_error",
        "gamma",
        "gamma_std_error",
    ):
        assert getattr(combined, name) == getattr(whole, name), name


def test_batch_range_offsets_select_genuinely_different_scrambles():
    """A later chunk must not silently repeat the first chunk's batches."""
    common = dict(spot=100.0, relative_bump=0.01, batches=4)
    first = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0), first_batch=0, **common
    )
    second = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0), first_batch=4, **common
    )

    assert not np.array_equal(first.batch_estimates, second.batch_estimates)


def test_batch_range_rejects_ranges_the_specs_cannot_cover():
    common = dict(spot=100.0, relative_bump=0.01)
    with pytest.raises(ValueError):
        run_paired_rqmc_greeks(
            _spec(99.0), _spec(100.0), _spec(101.0),
            batches=4, first_batch=-1, **common,
        )
    with pytest.raises(ValueError):
        # max_batches is 8, so [6, 10) runs off the end of the coupled stream.
        run_paired_rqmc_greeks(
            _spec(99.0), _spec(100.0), _spec(101.0),
            batches=4, first_batch=6, **common,
        )


def test_concatenating_incompatible_results_is_refused():
    common = dict(spot=100.0, relative_bump=0.01, batches=4)
    first = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0), first_batch=0, **common
    )
    other_bump = run_paired_rqmc_greeks(
        _spec(99.0), _spec(100.0), _spec(101.0),
        spot=100.0, relative_bump=0.02, batches=4, first_batch=4,
    )

    with pytest.raises(ValueError):
        concatenate_paired_results([first, other_bump])
    with pytest.raises(ValueError):
        concatenate_paired_results([])

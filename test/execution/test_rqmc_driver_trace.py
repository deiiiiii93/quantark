"""Traced RQMC driver: one implementation of the Welford stopping loop.

Bit-identity is proven against an inline verbatim copy of the PRE-REFACTOR
run_rqmc loop (reference implementation pinned in this test module).
"""
import numpy as np
import pytest

from quantark.montecarlo.qmc_rqmc_driver import (
    RQMCCheckpoint,
    RQMCResult,
    run_rqmc,
    run_rqmc_traced,
)


class _FakeGenerator:
    """Deterministic per-batch payoff seeds keyed by batch_id."""

    def __init__(self, num_paths):
        self.num_paths = num_paths

    def generate_paths(self, seed=None, batch_id=None, return_aux=False):
        rng = np.random.default_rng(1000 + int(batch_id or 0))
        paths = rng.standard_normal((self.num_paths, 3))
        aux = {"batch_id": 0 if batch_id is None else int(batch_id)}
        return paths, aux if return_aux else None


def _pricer(paths, aux):
    return 10.0 + paths[:, -1]


def _reference_run_rqmc(pricer_fn, path_generator, max_batches, target_std,
                        min_batches=1):
    """Verbatim pre-refactor loop (bit-identity oracle)."""
    batch_means = []
    n_paths_per_batch = path_generator.num_paths
    mean = 0.0
    m2 = 0.0
    for batch_id in range(max_batches):
        paths, aux = path_generator.generate_paths(
            batch_id=batch_id, return_aux=True
        )
        payoffs = np.asarray(pricer_fn(paths, aux), dtype=float)
        batch_mean = float(payoffs.mean())
        batch_means.append(batch_mean)
        n = batch_id + 1
        delta = batch_mean - mean
        mean += delta / n
        m2 += delta * (batch_mean - mean)
        if n >= min_batches:
            variance = m2 / (n - 1) if n > 1 else 0.0
            std_error = np.sqrt(variance / n)
            if std_error <= target_std or n == max_batches:
                return RQMCResult(
                    price=mean, std_error=std_error,
                    total_paths=n * n_paths_per_batch, batches_used=n,
                    batch_means=np.array(batch_means, dtype=float),
                )
    raise RuntimeError("unreachable")


@pytest.mark.parametrize("target_std,min_batches,max_batches", [
    (1e-6, 4, 12),   # runs to max
    (1.0, 4, 12),    # stops at min_batches
    (0.02, 2, 32),   # stops mid-run
    (1.0, 1, 1),     # single batch, variance 0.0 branch
])
def test_traced_bitwise_matches_reference(target_std, min_batches, max_batches):
    gen = _FakeGenerator(512)
    ref = _reference_run_rqmc(_pricer, gen, max_batches, target_std, min_batches)
    result, trace = run_rqmc_traced(
        _pricer, gen, max_batches, target_std, min_batches
    )
    assert result.price == ref.price
    assert result.std_error == ref.std_error
    assert result.total_paths == ref.total_paths
    assert result.batches_used == ref.batches_used
    assert np.array_equal(result.batch_means, ref.batch_means)
    # wrapper equivalence
    wrapped = run_rqmc(_pricer, gen, max_batches, target_std, min_batches)
    assert wrapped.price == result.price
    assert wrapped.std_error == result.std_error


def test_trace_shape_and_stop_flags():
    min_batches = 2
    gen = _FakeGenerator(512)
    result, trace = run_rqmc_traced(_pricer, gen, 32, 0.02, min_batches)
    assert len(trace) == result.batches_used
    assert all(isinstance(c, RQMCCheckpoint) for c in trace)
    assert [c.batch_index for c in trace] == list(range(len(trace)))
    assert all(not c.stopped for c in trace[:-1])
    assert trace[-1].stopped
    # std_error is None strictly before min_batches
    assert all(c.std_error is None for c in trace[: min_batches - 1])
    assert all(c.std_error is not None for c in trace[min_batches - 1:])
    # running mean at the stop checkpoint IS the result price
    assert trace[-1].running_mean == result.price
    assert trace[-1].std_error == result.std_error


def test_trace_deterministic_across_runs():
    gen = _FakeGenerator(512)
    a = run_rqmc_traced(_pricer, gen, 32, 0.02, 2)
    b = run_rqmc_traced(_pricer, gen, 32, 0.02, 2)
    assert a[0].price == b[0].price
    assert a[0].std_error == b[0].std_error
    assert a[0].batches_used == b[0].batches_used
    assert a[1] == b[1]


def test_validation_errors_preserved():
    gen = _FakeGenerator(8)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 0, 1e-4)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 4, 1e-4, min_batches=0)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 4, 1e-4, min_batches=8)
    with pytest.raises(ValueError):
        run_rqmc_traced(_pricer, gen, 4, -1.0)


def test_payoff_shape_validation_preserved():
    gen = _FakeGenerator(16)

    def bad_pricer(paths, aux):
        return np.zeros((4, 4))

    with pytest.raises(ValueError, match="one payoff per path"):
        run_rqmc_traced(bad_pricer, gen, 4, 1e-4)

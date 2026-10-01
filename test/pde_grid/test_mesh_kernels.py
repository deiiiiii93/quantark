"""The mesh kernels equal the NumPy reference path bit for bit (spec 3A §5.1, §7.2; decision 15).

``QUANTARK_REQUIRE_MESH_KERNEL`` makes the kernel under test mandatory: ``numba`` fails when Numba
does not import, ``python`` fails when it does; unset, whatever is installed is tested. Each CI job
sets it (``.github/workflows/tests.yml``), so neither kernel can be skipped silently.
"""
import os

import numpy as np
import pytest

from quantark.asset.equity.engine import PDEEngine
from quantark.asset.equity.engine.pde.grid import mesh_kernels, space
from quantark.asset.equity.param import PDEParams

REQUIRE = os.environ.get("QUANTARK_REQUIRE_MESH_KERNEL")
KERNELS = sorted(mesh_kernels.KERNELS)


def test_the_required_kernel_is_selected():
    if REQUIRE == "numba":
        assert mesh_kernels.backend() == "numba", "Numba is required but did not import or compile"
    elif REQUIRE == "python":
        assert mesh_kernels.backend() == "python"
    else:
        assert mesh_kernels.backend() in ("numba", "python")
    assert mesh_kernels.integrate is mesh_kernels.KERNELS[mesh_kernels.backend()]


def _frozen_inputs():
    """48 inputs: 1 to 8 criticals (every third has 8), 64 to 1,600 intervals. Never edit the seed."""
    rng = np.random.default_rng(20261001)
    out = []
    for i in range(48):
        m = 8 if i % 3 == 0 else 1 + i % 8
        lo = float(np.log(rng.uniform(5.0, 80.0)))
        hi = float(np.log(rng.uniform(120.0, 900.0)))
        crits = np.sort(rng.uniform(lo + 0.05, hi - 0.05, size=m))
        n = int(rng.choice([64, 128, 400, 800, 1600]))
        out.append((lo, n, float(rng.uniform(0.1, 20.0)), float(10.0 ** rng.uniform(-12, 0)), crits))
    return out


@pytest.mark.parametrize("kernel", KERNELS)
def test_integration_is_bit_identical_to_the_reference(kernel):
    fn = mesh_kernels.KERNELS[kernel]
    for lo, n, a, beta, crits in _frozen_inputs():
        ref = space._legacy_ode_integrate(lo, n, a, beta, crits)
        got = fn(lo, n, a, beta, crits)
        assert got.dtype == ref.dtype and got.shape == ref.shape
        assert np.array_equal(got, ref) and got.tobytes() == ref.tobytes(), (kernel, n, len(crits))


def _production_mesh_inputs():
    """The legacy-mesh calls two real solves make (fast accuracy keeps the reference path cheap)."""
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.modelvalidation.builders.equity_snowball import make_environment, make_snowball
    from quantark.util.enum import BarrierType, ObservationType, OptionType

    calls, orig = [], space._legacy_concentrated_mesh

    def record(*args, **kwargs):
        calls.append(args)
        return orig(*args, **kwargs)

    env = make_environment(dict(spot=100.0, vol=0.22, rate=0.025, div_yield=0.03))
    snowball = make_snowball(dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0,
                                  ko_rate=0.15, rebate_rate=0.15, months=12, maturity=1.0,
                                  contract_multiplier=1.0))
    barrier = BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=130.0,
                            barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.DISCRETE,
                            observation_dates=[k / 12.0 for k in range(1, 13)])
    space._legacy_concentrated_mesh = record
    try:
        PDEEngine(params=PDEParams(accuracy="fast", cache_enabled=False)).price(snowball, env)
        PDEEngine(params=PDEParams(accuracy="fast", cache_enabled=False)).price(barrier, env)
    finally:
        space._legacy_concentrated_mesh = orig
    assert calls, "no legacy-mesh call was made: the capture is not exercising the kernel"
    return calls


@pytest.mark.parametrize("kernel", KERNELS)
def test_production_layouts_are_bit_identical(kernel, monkeypatch):
    monkeypatch.setattr(mesh_kernels, "integrate", mesh_kernels.KERNELS[kernel])
    for args in _production_mesh_inputs():
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            got = space._legacy_concentrated_mesh(*args)
            ref = space._legacy_concentrated_mesh(*args, integrate=space._legacy_ode_integrate)
        assert got.tobytes() == ref.tobytes()


#: (y_min, n, a, beta, crits) on which the reference raises
DEFERRED = {
    "overflow": (0.0, 64, 1.0, 1e-12, np.array([1e200])),      # (y - c) ** 2 overflows
    "divide": (1.0, 64, 1.0, 0.0, np.array([1.0])),            # beta = 0 and y on the critical point
}


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("case", sorted(DEFERRED))
def test_a_deferred_input_raises_what_the_reference_raises(kernel, case, monkeypatch):
    args = DEFERRED[case]
    with pytest.raises(mesh_kernels.Deferred):
        mesh_kernels.KERNELS[kernel](*args)
    monkeypatch.setattr(mesh_kernels, "integrate", mesh_kernels.KERNELS[kernel])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        with pytest.raises(FloatingPointError) as ref:
            space._legacy_ode_integrate(*args)
        with pytest.raises(FloatingPointError) as got:
            space._kernel_integrate(*args)
    assert str(got.value) == str(ref.value)

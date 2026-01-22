import pytest

from asset.equity.param import MCParams
from util.exceptions import ValidationError


class DummyProduct:
    def __init__(self, contract_multiplier: float):
        self.contract_multiplier = contract_multiplier


def test_rqmc_target_std_absolute():
    params = MCParams()
    target = params.resolve_rqmc_target_std()
    assert target == pytest.approx(params.rqmc_target_std)


def test_rqmc_target_std_relative_notional():
    params = MCParams(rqmc_target_std=1e-4, rqmc_target_std_mode="relative_notional")
    product = DummyProduct(contract_multiplier=10000.0)
    target = params.resolve_rqmc_target_std(product=product)
    assert target == pytest.approx(1.0)


def test_rqmc_target_std_scale_override():
    params = MCParams(
        rqmc_target_std=1e-3,
        rqmc_target_std_mode="relative_price",
        rqmc_target_std_scale=500.0,
    )
    target = params.resolve_rqmc_target_std()
    assert target == pytest.approx(0.5)


def test_rqmc_target_std_invalid_mode():
    with pytest.raises(ValidationError):
        MCParams(rqmc_target_std_mode="invalid_mode")


def test_rqmc_paths_mode_total():
    params = MCParams(num_paths=100000, rqmc_paths_mode="total", rqmc_max_batches=4)
    per_batch = params.resolve_rqmc_paths_per_batch()
    assert per_batch == 32768


def test_rqmc_paths_mode_per_batch():
    params = MCParams(num_paths=20000, rqmc_paths_mode="per_batch")
    per_batch = params.resolve_rqmc_paths_per_batch()
    assert per_batch == 20000

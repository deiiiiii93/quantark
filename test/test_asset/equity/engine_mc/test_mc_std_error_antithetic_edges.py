"""T7 CP3: antithetic s.e. edge cases. Fewer than two pair units has no spread: NaN s.e., never a crash or a 0.0."""
import math
import warnings

import numpy as np
import pytest

import mc_se_cases
from quantark.montecarlo import mean_and_std_error


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def test_fewer_than_two_units_is_nan():
    mean, se = mean_and_std_error(np.array([2.0]))
    assert mean == 2.0 and math.isnan(se)
    mean, se = mean_and_std_error(np.array([]))
    assert math.isnan(mean) and math.isnan(se)


def test_one_path_antithetic_euro_prices_with_nan_se():
    from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType
    from quantark.util.enum.engine_enums import MonteCarloMethod

    e = EuropeanMCEngine(params=mc_se_cases._mc(1, True), method=MonteCarloMethod.PSEUDO)
    product = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    price = _quiet(lambda: e.price(product, mc_se_cases._env()))
    assert np.isfinite(price) and math.isnan(e.get_last_std_error())


def test_one_path_antithetic_fx_prices_with_nan_se():
    price, se = _quiet(lambda: mc_se_cases.run_case("fx_barrier_continuous", True, 1))
    assert np.isfinite(price) and math.isnan(se)


@pytest.mark.parametrize("case", ["accumulator", "single_sharkfin_expiry", "single_sharkfin_discrete"])
def test_one_pair_antithetic_reports_nan_not_zero(case):
    _, se = _quiet(lambda: mc_se_cases.run_case(case, True, 2))
    assert math.isnan(se)


def test_intraday_estimator_label_names_the_pair_mean_formula():
    from types import SimpleNamespace

    from quantark.intraday.engines.mc import _estimator
    from quantark.util.enum.engine_enums import MonteCarloMethod

    anti = SimpleNamespace(method=MonteCarloMethod.PSEUDO, params=SimpleNamespace(use_antithetic=True))
    plain = SimpleNamespace(method=MonteCarloMethod.PSEUDO, params=SimpleNamespace(use_antithetic=False))
    iid_paths = SimpleNamespace(method=MonteCarloMethod.PSEUDO, params=SimpleNamespace(use_antithetic=True),
                                _antithetic_pairs_in_paths=lambda: False)
    assert _estimator(anti, None) == "antithetic_pair_means"
    assert _estimator(plain, None) == "pathwise_iid"
    assert _estimator(iid_paths, None) == "pathwise_iid"

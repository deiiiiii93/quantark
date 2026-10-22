"""Tests for credit VaR engines (parametric / historical / Monte Carlo)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment
from quantark.var import (
    CreditHistoricalVaREngine,
    CreditMonteCarloVaREngine,
    CreditParametricVaREngine,
)
from quantark.var.config import VaRConfig


def _portfolio():
    env = CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curve=FlatHazardCurve(hazard_rate=0.02),
    )
    pf = CreditPortfolio(portfolio_name="cr", pricing_environments={"ACME": env})
    pf.add_position(
        product=CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4, coupon_spread=0.01),
        quantity=1.0, entry_price=0.0, reference_entity="ACME",
        engine=CDSReducedFormEngine(),
    )
    return pf


def _history(days=300, seed=1):
    rng = np.random.default_rng(seed)
    hazard = 0.02 + np.cumsum(rng.normal(0, 0.0005, days))
    rate = 0.03 + np.cumsum(rng.normal(0, 0.0003, days))
    return pd.DataFrame({"ACME_hazard": np.abs(hazard), "ACME_rate": rate})


def _config():
    return VaRConfig(confidence_level=0.99, lookback_days=250,
                     calculate_factor_var=True, calculate_component_var=True)


def test_parametric_var_positive_with_attribution():
    res = CreditParametricVaREngine(_config()).calculate_var(_portfolio(), _history())
    assert res.var > 0
    assert res.factor_var and "ACME_hazard_change" in res.factor_var
    assert res.component_var


def test_historical_var_positive():
    res = CreditHistoricalVaREngine(_config()).calculate_var(_portfolio(), _history())
    assert res.var > 0
    assert res.cvar >= res.var


def test_monte_carlo_var_positive():
    cfg = _config()
    cfg.mc_num_simulations = 5000
    cfg.mc_seed = 42
    res = CreditMonteCarloVaREngine(cfg).calculate_var(_portfolio(), _history())
    assert res.var > 0


def test_engine_rejects_non_credit_portfolio():
    with pytest.raises(Exception):
        CreditParametricVaREngine(_config()).calculate_var(object(), _history())


def test_include_spread_is_deprecated_alias_for_include_hazard():
    from quantark.var.credit.config import CreditRiskFactorConfig

    # Construction via the legacy kwarg still disables the credit factor.
    with pytest.warns(DeprecationWarning):
        cfg = CreditRiskFactorConfig(include_spread=False)
    assert cfg.include_hazard is False

    # Read and assignment through the alias both warn and map to include_hazard.
    cfg2 = CreditRiskFactorConfig()
    with pytest.warns(DeprecationWarning):
        assert cfg2.include_spread is True
    with pytest.warns(DeprecationWarning):
        cfg2.include_spread = False
    assert cfg2.include_hazard is False


def test_bump_env_spread_change_is_deprecated_alias():
    from quantark.var.credit.revaluation import bump_env

    env = CreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curve=FlatHazardCurve(hazard_rate=0.02),
    )
    with pytest.warns(DeprecationWarning):
        bumped = bump_env(env, spread_change=1e-4)
    ref = bump_env(env, hazard_change=1e-4)
    # The legacy alias applies exactly the hazard shift (no recovery conversion).
    assert bumped.get_hazard_rate(1.0) == pytest.approx(ref.get_hazard_rate(1.0))

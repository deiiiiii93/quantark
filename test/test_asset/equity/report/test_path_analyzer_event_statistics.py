"""The path analyzer reports the unified ``EventStatistics`` (unified event statistics T8).

``AutocallablePathAnalyzer.analyze_snowball_risk_neutral`` used to return its own record type with a
KI-ever probability and no knock-in split. It now fills ``EventStatistics`` from its own simulated paths:

* ``pv`` is the MC engine's ``price()`` (spec D23); the analyzer's own cash sum shows in the type's
  ``reconciliation_error``;
* every path ends in one outcome (KO row, maturity never KO never KI, KI settlement), so the partition holds
  to round-off;
* knock-in is counted only before a knock-out (DESIGN §4.5): a barrier crossed after the note knocked out is
  not a knock-in;
* discrete KI: the per-date first-KI table and the expected KI time come from the same paths; continuous KI:
  both are unavailable with a reason (the analyzer knows a crossing only to its monitoring step);
* one path set is one batch: no standard errors (``"undefined: single batch"``).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.analysis.autocallable_path_analyzer import AutocallablePathAnalyzer
from quantark.asset.equity.engine.event_stats import EventStatistics, StatisticUnavailable
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.numerical import Tolerance

PARAMS = MCParams(num_paths=4000, time_steps=50, seed=7)


def _env(vol: float = 0.25) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=0.02), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _discrete_ki_snowball(*, ko_barrier: float = 103.0, ki_barrier: float = 85.0, ki_dates=None):
    ki_dates = [0.25, 0.5, 0.75, 1.0] if ki_dates is None else ki_dates
    return create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=ko_barrier, ko_rate=0.15,
        ki_barrier=ki_barrier, num_observations=4, include_principal=True,
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False, ki_observation_dates=ki_dates,
    )


def _continuous_ki_snowball():
    return create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=103.0, ko_rate=0.15, ki_barrier=85.0,
        num_observations=4, include_principal=True,
    )


def _analyze(product, env=None) -> EventStatistics:
    return AutocallablePathAnalyzer(mc_params=PARAMS).analyze_snowball_risk_neutral(product, env or _env())


def test_the_analyzer_returns_event_statistics_with_pv_from_price():
    product = _discrete_ki_snowball()
    stats = _analyze(product)
    assert isinstance(stats, EventStatistics)
    assert stats.product_family == "snowball" and stats.coupons is None
    assert stats.pv == SnowballMCEngine(params=PARAMS).price(product, _env())       # spec D23, bitwise
    assert stats.reconciliation_error == stats.pv - stats.total_discounted_cash
    assert stats.provenance.statistics_model == "path_analyzer_mc"


def test_every_path_ends_in_one_outcome():
    stats = _analyze(_discrete_ki_snowball())
    assert abs(stats.probability_residual) <= Tolerance.PROBABILITY
    obs = stats.observations
    # survival is P(not knocked out after the row), computed from the paths
    assert np.all(np.diff(obs.survival) <= 0.0)
    assert obs.survival[-1] == pytest.approx(1.0 - float(np.sum(obs.probability)), abs=Tolerance.PROBABILITY)
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability
    assert stats.knock_in.probability >= stats.ki_settlement.probability
    assert stats.ledger.total_discounted_cash == pytest.approx(stats.total_discounted_cash, rel=Tolerance.RELATIVE)


def test_deterministic_payment_rows_pay_discounted_over_the_payment_df():
    env = _env()
    stats = _analyze(_discrete_ki_snowball(), env)
    obs = stats.observations
    for i in range(obs.size):
        df = env.get_discount_factor(float(obs.payment_times[i]))
        assert obs.discounted_cash[i] == pytest.approx(obs.cash_to_pay[i] * df, rel=Tolerance.RELATIVE)
    df_t = env.get_discount_factor(stats.maturity.payment_time)
    assert stats.maturity.discounted_cash == pytest.approx(stats.maturity.cash_to_pay * df_t, rel=Tolerance.RELATIVE)
    assert stats.ki_settlement.discounted_cash == pytest.approx(
        stats.ki_settlement.cash_to_pay * df_t, rel=Tolerance.RELATIVE)


def test_discrete_ki_table_sums_to_the_knock_in_probability_and_its_mean_time():
    stats = _analyze(_discrete_ki_snowball())
    table = stats.require("ki_table")
    p_ki = stats.require("knock_in.probability")
    assert p_ki > 0.0
    assert np.array_equal(table.times, np.array([0.25, 0.5, 0.75, 1.0]))
    assert float(np.sum(table.probability)) == pytest.approx(p_ki, abs=Tolerance.PROBABILITY)
    expected_time = stats.require("knock_in.expected_time")
    assert float(np.sum(table.times * table.probability)) / p_ki == pytest.approx(expected_time, rel=Tolerance.RELATIVE)
    assert np.all(np.diff(table.survival) <= 0.0)
    assert not np.any(table.interpolated) and table.interpolation_scheme is None
    loss = stats.require("knock_in.expected_loss")
    assert loss > 0.0


def test_a_barrier_crossed_after_a_certain_knock_out_is_not_a_knock_in():
    """KO certain at the first observation (barrier far below spot); the KI dates lie after it."""
    product = _discrete_ki_snowball(ko_barrier=1.0, ki_barrier=99.0, ki_dates=[0.5, 0.75, 1.0])
    stats = _analyze(product)
    assert stats.observations.probability[0] == 1.0
    assert stats.require("knock_in.probability") == 0.0
    assert stats.ki_settlement.probability == 0.0
    assert stats.require("knock_in.expected_loss") == 0.0
    assert stats.knock_in.expected_time is None
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert float(np.sum(stats.require("ki_table").probability)) == 0.0


def test_continuous_ki_reports_the_probability_and_flags_the_table_and_the_time():
    stats = _analyze(_continuous_ki_snowball())
    assert stats.require("knock_in.probability") >= stats.ki_settlement.probability
    for path in ("ki_table", "knock_in.expected_time"):
        with pytest.raises(StatisticUnavailable, match="path analyzer"):
            stats.require(path)
    assert stats.require("knock_in.expected_loss") >= 0.0


def test_one_path_set_is_one_batch_without_standard_errors():
    stats = _analyze(_discrete_ki_snowball())
    assert stats.standard_errors is None
    assert stats.unavailable["standard_errors"] == "undefined: single batch"

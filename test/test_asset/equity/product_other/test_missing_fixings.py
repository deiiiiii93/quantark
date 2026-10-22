"""Past fixings without a recorded outcome are decided at the current spot, with one warning per pricing call.

User decision 2026-10-09: range accruals and Asian options fill a past observation that lacks its fixing with the
environment's CURRENT spot, warn, and do not raise. The product is never changed by pricing; a future fixing that
carries an outcome still raises.
"""

import copy
import warnings
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import (
    AsianOptionAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import AsianOptionMCEngine, RangeAccrualMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.asset.fx.engine.analytical import (
    FxForeignRangeAccrualAnalyticalEngine,
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.engine.mc import FxRangeAccrualMCEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxForeignRangeAccrualOption,
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
    FxRangeAccrualOption,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.enum import AsianStrikeType, AveragingType, OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning

VAL = datetime(2025, 1, 1)


def _missing_fixing_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [w for w in caught if issubclass(w.category, MissingFixingWarning)]


# =============================================================================
# Equity range accrual
# =============================================================================


def _eq_env(spot=100.0):
    return PricingEnvironment(
        valuation_date=VAL,
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
    )


def _eq_option(records, *, is_reverse=False, maturity=1.0):
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(
            upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05, is_reverse=is_reverse
        ),
        maturity=maturity,
        tenor=1.25,
        observation_records=records,
    )


def _eq_records(first_outcome=None, **first_kwargs):
    return [
        RangeAccrualObservationRecord(observation_time=-0.2, weight=1.0, observed_in_range=True),
        RangeAccrualObservationRecord(
            observation_time=-0.1, weight=2.0, observed_in_range=first_outcome, **first_kwargs
        ),
        RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
        RangeAccrualObservationRecord(observation_time=1.0, weight=1.0),
    ]


class TestEquityRangeAccrualFill:
    def test_filled_in_range(self):
        (past, future, total), caught = _missing_fixing_warnings(
            _eq_option(_eq_records()).resolve_observations, _eq_env(100.0)
        )
        assert past == [(1.0, True), (2.0, True)]
        assert len(future) == 2 and total == 5.0
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "1 past fixing" in message and "#1 t=-0.1" in message and "current spot" in message

    def test_filled_out_of_range(self):
        (past, _, _), caught = _missing_fixing_warnings(
            _eq_option(_eq_records()).resolve_observations, _eq_env(120.0)
        )
        assert past == [(1.0, True), (2.0, False)]
        assert len(caught) == 1

    def test_filled_reverse_mode(self):
        (past, _, _), _ = _missing_fixing_warnings(
            _eq_option(_eq_records(), is_reverse=True).resolve_observations, _eq_env(100.0)
        )
        assert past[1] == (2.0, False)  # inside the band pays nothing in reverse mode

    def test_filled_with_per_record_barrier(self):
        records = _eq_records(lower_barrier=101.0, upper_barrier=120.0)
        (past, _, _), _ = _missing_fixing_warnings(_eq_option(records).resolve_observations, _eq_env(100.0))
        assert past[1] == (2.0, False)  # the record's own band [101, 120] excludes 100

    def test_dated_fixing_named_by_date(self):
        records = [
            RangeAccrualObservationRecord(observation_date=datetime(2024, 12, 20)),
            RangeAccrualObservationRecord(observation_date=datetime(2025, 1, 1)),
            RangeAccrualObservationRecord(observation_date=datetime(2025, 6, 1)),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05),
            exercise_date=datetime(2025, 6, 1),
            initial_date=datetime(2024, 12, 1),
            observation_records=records,
        )
        (past, _, _), caught = _missing_fixing_warnings(option.resolve_observations, _eq_env(100.0))
        assert past == [(1.0, True), (1.0, True)]
        assert len(caught) == 1
        message = str(caught[0].message)
        # #1 is dated on the valuation date: t = 0 is live, not history (run fixings §3.2), so only #0 is named
        assert "1 past fixing" in message
        assert "#0 2024-12-20" in message and "#1" not in message

    def test_get_past_accrual_fills(self):
        (accrual, caught) = _missing_fixing_warnings(_eq_option(_eq_records()).get_past_accrual, _eq_env(120.0))
        assert accrual == (1.0, 3.0)
        assert len(caught) == 1

    def test_future_fixing_with_outcome_still_raises(self):
        records = [RangeAccrualObservationRecord(observation_time=0.5, observed_in_range=True)]
        with pytest.raises(ValidationError, match="should not have observed_in_range set"):
            _eq_option(records).resolve_observations(_eq_env())

    @pytest.mark.parametrize("spot", [100.0, 120.0])
    @pytest.mark.parametrize(
        "make_engine",
        [
            RangeAccrualAnalyticalEngine,
            lambda: RangeAccrualMCEngine(params=MCParams(num_paths=4000, seed=7), method=MonteCarloMethod.PSEUDO),
        ],
        ids=["analytical", "mc"],
    )
    def test_price_equals_explicit_record_and_product_unchanged(self, make_engine, spot):
        env = _eq_env(spot)
        option = _eq_option(_eq_records())
        before = copy.deepcopy(option.observation_records)
        price, caught = _missing_fixing_warnings(make_engine().price, option, env)
        assert len(caught) == 1  # once per pricing call
        assert option.observation_records == before  # pricing does not record the fixing
        explicit = _eq_option(_eq_records(first_outcome=spot <= 110.0))
        explicit_price, explicit_caught = _missing_fixing_warnings(make_engine().price, explicit, env)
        assert explicit_caught == []
        assert price == explicit_price

    def test_analytical_greeks_warn_once(self):
        greeks, caught = _missing_fixing_warnings(
            RangeAccrualAnalyticalEngine().calculate_greeks, _eq_option(_eq_records()), _eq_env()
        )
        assert len(caught) == 1
        explicit = RangeAccrualAnalyticalEngine().calculate_greeks(
            _eq_option(_eq_records(first_outcome=True)), _eq_env()
        )
        assert greeks == explicit

    @pytest.mark.parametrize(
        "make_engine",
        [
            RangeAccrualAnalyticalEngine,
            lambda: RangeAccrualMCEngine(params=MCParams(num_paths=1000, seed=7), method=MonteCarloMethod.PSEUDO),
        ],
        ids=["analytical", "mc"],
    )
    def test_near_expiry_fills(self, make_engine):
        records = [
            RangeAccrualObservationRecord(observation_time=0.0, observed_in_range=True),
            RangeAccrualObservationRecord(observation_time=0.0),
        ]
        option = _eq_option(records, maturity=1e-12)
        price, caught = _missing_fixing_warnings(make_engine().price, option, _eq_env(120.0))
        assert len(caught) == 0  # both records are at t = 0: live, not history (run fixings §3.2)
        explicit = _eq_option(
            [
                RangeAccrualObservationRecord(observation_time=0.0, observed_in_range=True),
                RangeAccrualObservationRecord(observation_time=0.0, observed_in_range=False),
            ],
            maturity=1e-12,
        )
        assert price == make_engine().price(explicit, _eq_env(120.0))


# =============================================================================
# FX range accrual
# =============================================================================

FX_SPOT = 1.20


def _fx_env(spot=FX_SPOT):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15),
        spot_quote=SpotQuote(spot=spot),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


def _fx_records(first_outcome=None, **first_kwargs):
    return [
        FxRangeAccrualObservationRecord(observation_time=-0.25, observed_in_range=True),
        FxRangeAccrualObservationRecord(
            observation_time=-0.1, weight=2.0, observed_in_range=first_outcome, **first_kwargs
        ),
        FxRangeAccrualObservationRecord(observation_time=0.5),
        FxRangeAccrualObservationRecord(observation_time=1.0),
    ]


def _fx_option(records, *, is_reverse=False, cls=FxRangeAccrualOption):
    return cls(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(
            upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04, is_reverse=is_reverse
        ),
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=1.0,
        observation_records=records,
    )


class TestFxRangeAccrualFill:
    def test_filled_in_range(self):
        (past, future, total), caught = _missing_fixing_warnings(
            _fx_option(_fx_records()).resolve_observations, _fx_env()
        )
        assert past == [(1.0, True), (2.0, True)]
        assert len(future) == 2 and total == 5.0
        assert len(caught) == 1
        assert "1 past fixing" in str(caught[0].message) and "#1 t=-0.1" in str(caught[0].message)

    def test_filled_out_of_range(self):
        (past, _, _), _ = _missing_fixing_warnings(_fx_option(_fx_records()).resolve_observations, _fx_env(1.35))
        assert past[1] == (2.0, False)

    def test_filled_reverse_mode(self):
        (past, _, _), _ = _missing_fixing_warnings(
            _fx_option(_fx_records(), is_reverse=True).resolve_observations, _fx_env()
        )
        assert past[1] == (2.0, False)

    def test_filled_with_per_record_barrier(self):
        records = _fx_records(lower_barrier=1.21, upper_barrier=1.40)
        (past, _, _), _ = _missing_fixing_warnings(_fx_option(records).resolve_observations, _fx_env())
        assert past[1] == (2.0, False)

    def test_future_fixing_with_outcome_still_raises(self):
        records = [FxRangeAccrualObservationRecord(observation_time=0.5, observed_in_range=True)]
        with pytest.raises(ValidationError, match="must not set observed_in_range"):
            _fx_option(records).resolve_observations(_fx_env())

    @pytest.mark.parametrize("spot", [FX_SPOT, 1.35])
    @pytest.mark.parametrize(
        "cls, make_engine",
        [
            (FxRangeAccrualOption, FxRangeAccrualAnalyticalEngine),
            (FxForeignRangeAccrualOption, FxForeignRangeAccrualAnalyticalEngine),
            (FxRangeAccrualOption, lambda: FxRangeAccrualMCEngine(num_paths=4000, seed=7)),
        ],
        ids=["analytical", "foreign", "mc"],
    )
    def test_price_equals_explicit_record_and_product_unchanged(self, cls, make_engine, spot):
        env = _fx_env(spot)
        option = _fx_option(_fx_records(), cls=cls)
        before = copy.deepcopy(option.observation_records)
        price, caught = _missing_fixing_warnings(make_engine().price, option, env)
        assert len(caught) == 1
        assert option.observation_records == before
        explicit = _fx_option(_fx_records(first_outcome=spot <= 1.30), cls=cls)
        explicit_price, explicit_caught = _missing_fixing_warnings(make_engine().price, explicit, env)
        assert explicit_caught == []
        assert price == explicit_price


# =============================================================================
# Equity Asian
# =============================================================================


def _asian_records(first_price=None):
    return [
        AsianObservationRecord(observation_time=-0.2, observed_price=95.0),
        AsianObservationRecord(observation_time=-0.1, observed_price=first_price),
        AsianObservationRecord(observation_time=0.5),
        AsianObservationRecord(observation_time=1.0),
    ]


def _asian(records):
    return AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=1.0,
        observation_records=records,
    )


class TestAsianFill:
    def test_resolve_fills_current_spot(self):
        (past_prices, past_weights, future_times, _, n), caught = _missing_fixing_warnings(
            _asian(_asian_records()).resolve_observations, _eq_env(103.0)
        )
        assert past_prices == [95.0, 103.0]
        assert past_weights == [0.25, 0.25] and future_times == [0.5, 1.0] and n == 4
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "1 past fixing" in message and "observed_price" in message and "#1 t=-0.1" in message

    def test_future_fixing_with_price_still_raises(self):
        records = [AsianObservationRecord(observation_time=0.5, observed_price=100.0)]
        with pytest.raises(ValidationError, match="should not have observed_price set"):
            _asian(records).resolve_observations(_eq_env())

    @pytest.mark.parametrize(
        "make_engine",
        [
            AsianOptionAnalyticalEngine,
            lambda: AsianOptionMCEngine(params=MCParams(num_paths=4000, seed=7), method=MonteCarloMethod.PSEUDO),
        ],
        ids=["analytical", "mc"],
    )
    def test_price_equals_explicit_record_and_product_unchanged(self, make_engine):
        env = _eq_env(103.0)
        option = _asian(_asian_records())
        before = copy.deepcopy(option.observation_records)
        price, caught = _missing_fixing_warnings(make_engine().price, option, env)
        assert len(caught) == 1
        assert option.observation_records == before
        explicit_price, explicit_caught = _missing_fixing_warnings(
            make_engine().price, _asian(_asian_records(first_price=103.0)), env
        )
        assert explicit_caught == []
        assert price == explicit_price

"""Focused public-API and valuation-date guards for the DCN extension."""

from datetime import datetime

import pytest

from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
from quantark.util.exceptions import ValidationError

from dcn_fixtures import DCN_A, schedule_kwargs


def test_pde_package_exports_dcn_engine_and_result():
    from quantark.asset.equity.engine.pde import DCNPDEEngine, DCNPDEResult
    from quantark.asset.equity.engine.pde.dcn_pde_solver import (
        DCNPDEEngine as DirectEngine,
        DCNPDEResult as DirectResult,
    )

    assert DCNPDEEngine is DirectEngine
    assert DCNPDEResult is DirectResult


def test_schedule_accepts_non_trading_valuation_date():
    # A7' (T12): a closed-day valuation builds; its first row / KI date is the
    # next trading day (was: ValidationError "must be a trading day").
    kwargs = schedule_kwargs(DCN_A)
    kwargs["valuation_date"] = datetime(2023, 1, 7)  # Saturday

    schedule = build_dcn_schedule(**kwargs)
    assert schedule.valuation_date == datetime(2023, 1, 7)
    assert schedule.daily_ki_dates[0] == datetime(2023, 1, 9)


def test_schedule_rejects_valuation_date_after_maturity():
    kwargs = schedule_kwargs(DCN_A)
    kwargs["valuation_date"] = datetime(2025, 1, 6)

    with pytest.raises(ValidationError, match="after maturity_date"):
        build_dcn_schedule(**kwargs)

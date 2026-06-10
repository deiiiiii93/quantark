"""
Unit tests for risk factor extraction.
"""

import pytest
import pandas as pd
import numpy as np

from quantark.var.risk_factors import (
    SpotReturnFactor,
    VolChangeFactor,
    RateShiftFactor,
    DivYieldShiftFactor,
    ParallelShiftFactor,
    KeyRateShiftFactor,
)


def test_spot_return_factor_from_dataframe():
    """Test SpotReturnFactor extraction from DataFrame."""
    df = pd.DataFrame({
        'spot': [100, 102, 101, 103, 104]
    })
    
    factor = SpotReturnFactor()
    returns = factor.extract_from_dataframe(df)
    
    assert len(returns) == 4
    assert np.isclose(returns.iloc[0], 0.02)
    assert np.isclose(returns.iloc[1], -0.0098, atol=1e-4)


def test_spot_return_factor_from_returns_column():
    """Test SpotReturnFactor with pre-computed returns."""
    df = pd.DataFrame({
        'spot_return': [0.02, -0.01, 0.015, 0.01]
    })
    
    factor = SpotReturnFactor()
    returns = factor.extract_from_dataframe(df)
    
    assert len(returns) == 4
    assert returns.iloc[0] == 0.02


def test_vol_change_factor_from_dataframe():
    """Test VolChangeFactor extraction from DataFrame."""
    df = pd.DataFrame({
        'vol': [0.20, 0.22, 0.21, 0.23, 0.22]
    })
    
    factor = VolChangeFactor()
    changes = factor.extract_from_dataframe(df)
    
    assert len(changes) == 4
    assert np.isclose(changes.iloc[0], 0.02)
    assert np.isclose(changes.iloc[1], -0.01)


def test_rate_shift_factor_from_dataframe():
    """Test RateShiftFactor extraction from DataFrame."""
    df = pd.DataFrame({
        'rate': [0.05, 0.051, 0.052, 0.050, 0.049]
    })
    
    factor = RateShiftFactor()
    shifts = factor.extract_from_dataframe(df)
    
    assert len(shifts) == 4
    assert np.isclose(shifts.iloc[0], 0.001)
    assert np.isclose(shifts.iloc[2], -0.002, atol=1e-6)


def test_div_yield_shift_factor_from_dataframe():
    """Test DivYieldShiftFactor extraction from DataFrame."""
    df = pd.DataFrame({
        'div_yield': [0.02, 0.021, 0.020, 0.022, 0.021]
    })
    
    factor = DivYieldShiftFactor()
    shifts = factor.extract_from_dataframe(df)
    
    assert len(shifts) == 4
    assert np.isclose(shifts.iloc[0], 0.001)


def test_parallel_shift_factor_from_dataframe():
    """Test ParallelShiftFactor extraction from DataFrame."""
    df = pd.DataFrame({
        'rate': [0.03, 0.032, 0.031, 0.033, 0.032]
    })
    
    factor = ParallelShiftFactor()
    shifts = factor.extract_from_dataframe(df)
    
    assert len(shifts) == 4
    assert np.isclose(shifts.iloc[0], 0.002)


def test_key_rate_shift_factor_from_dataframe():
    """Test KeyRateShiftFactor extraction from DataFrame."""
    df = pd.DataFrame({
        'rate_2y': [0.02, 0.021, 0.022, 0.021, 0.023],
        'rate_5y': [0.03, 0.031, 0.030, 0.032, 0.031],
        'rate_10y': [0.04, 0.041, 0.042, 0.041, 0.043]
    })
    
    factor = KeyRateShiftFactor(tenors=[2.0, 5.0, 10.0])
    shifts = factor.extract_from_dataframe(df)
    
    assert shifts.shape == (4, 3)
    assert 'shift_2y' in shifts.columns
    assert 'shift_5y' in shifts.columns
    assert 'shift_10y' in shifts.columns
    assert np.isclose(shifts['shift_2y'].iloc[0], 0.001)


def test_risk_factor_missing_column():
    """Test error handling for missing columns."""
    df = pd.DataFrame({
        'price': [100, 102, 101]
    })
    
    factor = SpotReturnFactor()
    
    with pytest.raises(ValueError, match="must contain"):
        factor.extract_from_dataframe(df)


def test_key_rate_shift_missing_tenor():
    """Test KeyRateShiftFactor with missing tenor."""
    df = pd.DataFrame({
        'rate_2y': [0.02, 0.021, 0.022],
        'rate_5y': [0.03, 0.031, 0.030]
    })
    
    factor = KeyRateShiftFactor(tenors=[2.0, 5.0, 10.0])
    
    with pytest.raises(ValueError, match="must contain 'rate_10y'"):
        factor.extract_from_dataframe(df)

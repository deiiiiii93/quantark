# Testing Guidelines for QuantArk

## Testing Framework
**pytest** is the primary testing framework for QuantArk.

## Test File Organization

### Location
All tests are in the `/test/` directory (parallel to source code)

### Naming Convention
- **Format**: `test_<module_name>.py`
- **Examples**:
  - `test_european_option.py`
  - `test_parametric_var.py`
  - `test_fixed_bond.py`

### Test Class Organization
```python
import pytest
from module import ClassUnderTest

class TestClassUnderTest:
    """Test suite for ClassUnderTest."""

    @pytest.fixture
    def sample_object(self):
        """Fixture creating a sample object for testing."""
        return ClassUnderTest(parameter=value)

    def test_method_basic(self, sample_object):
        """Test basic functionality."""
        result = sample_object.method()
        assert result == expected_value

    def test_method_with_parameters(self):
        """Test method with different parameters."""
        obj = ClassUnderTest(param1=1, param2=2)
        assert obj.method() == expected
```

## Test Structure and Naming

### Test Method Naming
Use **descriptive** test names that explain what is being tested:
```python
def test_european_call_option_price_at_maturity():
    """Test call option price at maturity equals intrinsic value."""

def test_var_calculation_with_empty_portfolio_raises_error():
    """Test that empty portfolio raises ValidationError."""

def test_parametric_var_faster_than_historical_for_large_portfolios():
    """Test performance comparison between engines."""
```

### Bad Examples (Avoid)
```python
def test1():  # ❌ Undescriptive
    pass

def test_option():  # ❌ Too vague
    pass

def test_var():  # ❌ Not specific
    pass
```

## Fixtures

### Purpose
Fixtures provide **reusable** test data and setup.

### Built-in Fixtures
```python
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'spot_return': np.random.normal(0, 0.02, 100),
        'vol_change': np.random.normal(0, 0.01, 100)
    }, index=dates)
    return data

@pytest.fixture
def sample_portfolio():
    """Create sample portfolio for testing."""
    # Create mock portfolio
    return Portfolio(positions={'AAPL': 100, 'MSFT': 50})

@pytest.fixture
def at_the_money_option():
    """Create ATM European call option."""
    return EuropeanVanillaOption(
        strike=100.0,
        maturity=1.0,
        option_type=OptionType.CALL
    )
```

### Fixture Scope
```python
@pytest.fixture(scope='function')  # Default - new for each test
def sample_data():
    return create_data()

@pytest.fixture(scope='class')  # Once per test class
def shared_setup():
    return setup_class()

@pytest.fixture(scope='module')  # Once per test module
def database():
    return setup_test_db()

@pytest.fixture(scope='session')  # Once per test session
def global_config():
    return load_config()
```

### Parametrized Fixtures
```python
@pytest.fixture(params=[0.95, 0.99, 0.999])
def confidence_level(request):
    """Test with multiple confidence levels."""
    return request.param
```

## Testing Strategies

### 1. Unit Tests
Test **individual components** in isolation:
```python
def test_black_scholes_call_price():
    """Test Black-Scholes formula for call option."""
    # Setup
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    vol = 0.20
    div_yield = 0.02

    # Execute
    price = black_scholes_call(spot, strike, maturity, rate, vol, div_yield)

    # Assert
    assert price > 0
    assert price < spot  # Call can't be worth more than spot
```

### 2. Integration Tests
Test **component interactions**:
```python
def test_portfolio_var_integration():
    """Test full VaR calculation workflow."""
    # Create portfolio
    portfolio = EquityPortfolio({'AAPL': 100})

    # Create market data
    data = create_sample_market_data()

    # Configure VaR
    config = VaRConfig(confidence_level=0.99)

    # Calculate VaR
    engine = ParametricVaREngine(config)
    result = engine.calculate_var(portfolio, data)

    # Assert results
    assert result.var > 0
    assert result.cvar > result.var
```

### 3. Regression Tests
Ensure **existing functionality** doesn't break:
```python
def test_put_call_parity_regression():
    """Test that put-call parity holds (regression test)."""
    option = EuropeanVanillaOption(strike=100, maturity=1.0)
    env = create_pricing_environment()

    call_price = price_option(option, env, OptionType.CALL)
    put_price = price_option(option, env, OptionType.PUT)

    # Put-Call Parity: C - P = S - K*e^(-rT)
    parity_diff = call_price - put_price - (
        env.spot - strike * np.exp(-env.rate * option.maturity)
    )

    assert abs(parity_diff) < 1e-10
```

### 4. Edge Case Tests
Test **boundary conditions**:
```python
def test_option_near_expiry():
    """Test option pricing near expiry."""
    option = EuropeanVanillaOption(strike=100, maturity=1e-8)
    price = price_option(option, create_env())

    # Near expiry, price should approximate intrinsic value
    assert abs(price - option.strike) < 0.01

def test_deep_itm_option():
    """Test deep in-the-money option."""
    option = EuropeanVanillaOption(strike=50, maturity=1.0)  # Spot=100
    price = price_option(option, create_env())

    # Deep ITM call should be worth spot - strike (minus PV)
    intrinsic = option.spot - option.strike * np.exp(-env.rate)
    assert abs(price - intrinsic) < 0.01

def test_var_with_single_position():
    """Test VaR with single position."""
    portfolio = create_portfolio_with_single_position()
    result = engine.calculate_var(portfolio, data)

    # Single position VaR should equal position VaR
    assert result.var > 0
```

### 5. Performance Tests
Test **performance characteristics**:
```python
def test_parametric_var_scales_linearly():
    """Test that parametric VaR scales linearly with portfolio size."""
    engine = ParametricVaREngine()

    # Small portfolio
    small_portfolio = create_portfolio(10)
    start = time.time()
    engine.calculate_var(small_portfolio, data)
    small_time = time.time() - start

    # Large portfolio
    large_portfolio = create_portfolio(100)
    start = time.time()
    engine.calculate_var(large_portfolio, data)
    large_time = time.time() - start

    # Should scale roughly linearly (10x size ~ 10x time)
    ratio = large_time / small_time
    assert 5 < ratio < 15  # Allow for noise
```

## Assertion Patterns

### Numeric Comparisons
```python
# Absolute tolerance
assert abs(result - expected) < 1e-10

# Relative tolerance
assert abs(result - expected) / expected < 1e-6

# For arrays
assert np.allclose(result_array, expected_array, rtol=1e-10)

# For NaN checks
assert not np.isnan(result)
assert np.isnan(expected_nan_result)
```

### Exception Testing
```python
# Test that exception is raised
with pytest.raises(ValidationError, match="confidence level"):
    VaRConfig(confidence_level=1.5)

# Test exception message
with pytest.raises(ValueError) as exc_info:
    risky_function()
assert "expected message" in str(exc_info.value)

# Test exception type
with pytest.raises(QuantArkException):
    problematic_operation()
```

### Collection Testing
```python
# Test list contents
assert len(results) == 10
assert all(r > 0 for r in results)

# Test set contents
assert set(positions) == {'AAPL', 'MSFT', 'GOOGL'}

# Test dictionary
assert 'var' in result
assert result['var'] > 0
```

## Mocking and Patching

### Mock External Dependencies
```python
from unittest.mock import Mock, patch

@patch('external_service.get_rate')
def test_with_external_call(mock_get_rate):
    """Test function that calls external service."""
    mock_get_rate.return_value = 0.05

    result = calculate_with_rate()

    assert result == expected_value
    mock_get_rate.assert_called_once()

def test_with_database_mock():
    """Test with mocked database."""
    mock_db = Mock()
    mock_db.query.return_value = [{'price': 100}]

    result = get_price_from_db(mock_db)

    assert result == 100
    mock_db.query.assert_called_once()
```

### Mock Pricing Engines
```python
def test_portfolio_with_mocked_engine():
    """Test portfolio with mocked pricing engine."""
    mock_engine = Mock()
    mock_engine.price.return_value = 10.5

    portfolio = EquityPortfolio({'AAPL': 100})
    portfolio.engine = mock_engine

    price = portfolio.calculate_value()

    assert price == 1050.0
    mock_engine.price.assert_called()
```

## Test Data Management

### Synthetic Data
```python
@pytest.fixture
def synthetic_option_chain():
    """Generate synthetic option chain data."""
    return {
        f'call_{strike}': create_option(strike)
        for strike in [90, 95, 100, 105, 110]
    }

@pytest.fixture
def random_market_data():
    """Generate random market data with reproducible seed."""
    np.random.seed(42)
    return generate_market_data(1000)
```

### Fixtures in conftest.py
```python
# test/conftest.py - Global fixtures available to all tests

import pytest
import numpy as np

@pytest.fixture(scope='session')
def test_config():
    """Global test configuration."""
    return {
        'confidence_level': 0.99,
        'num_simulations': 1000
    }

@pytest.fixture
def clean_environment():
    """Ensure clean test environment."""
    # Setup
    yield
    # Teardown
    cleanup_test_data()
```

## Running Tests

### Command Line Options
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test/test_european_option.py

# Run with verbose output
python -m pytest -v

# Run with quiet output
python -m pytest -q

# Run specific test method
python -m pytest test/test_var.py::TestVaREngine::test_basic_calculation

# Run tests matching pattern
python -m pytest -k "test_parametric"

# Run tests marked with marker
python -m pytest -m "not slow"

# Stop on first failure
python -m pytest -x

# Run last failed tests only
python -m pytest --lf

# Show local variables on failure
python -m pytest -l

# Capture output
python -m pytest -s  # Show print statements
python -m pytest --capture=no  # Don't capture output
```

### Test Markers
```python
# Define custom markers
@pytest.mark.slow
def test_monte_carlo_convergence():
    """Test that may take several seconds."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Integration test."""
    pass

# Run only fast tests
python -m pytest -m "not slow"

# Run only integration tests
python -m pytest -m "integration"
```

## Coverage

### Install Coverage Plugin
```bash
pip install pytest-cov
```

### Generate Coverage Report
```bash
# Run with coverage
python -m pytest --cov=.

# Coverage with HTML report
python -m pytest --cov=. --cov-report=html
# Open htmlcov/index.html in browser

# Coverage with terminal report
python -m pytest --cov=. --cov-report=term-missing
```

### Coverage Targets
- **Overall**: >90% code coverage
- **Critical modules**: >95% coverage
- **New features**: 100% coverage required

## Best Practices

### DO ✅
1. Write **descriptive** test names
2. Use **fixtures** for reusable setup
3. Test **edge cases** and **boundary conditions**
4. Test **error conditions** (exception handling)
5. Use **parametrization** for multiple inputs
6. Keep tests **independent** and **isolated**
7. Use **assertions** with clear messages
8. Test **integration** of components
9. **Mock** external dependencies
10. Aim for **high coverage** (>90%)

### DON'T ❌
1. Don't test **private implementation** details
2. Don't write **brittle tests** that break on refactoring
3. Don't use **magic numbers** without explanation
4. Don't **mock** too much (lose test value)
5. Don't make tests **order-dependent**
6. Don't **ignore** test failures
7. Don't write **very long** test methods (>50 lines)
8. Don't **share state** between tests
9. Don't test **third-party libraries**
10. Don't commit **failing tests**

## Test Documentation

### Test Docstrings
```python
def test_black_scholes_call_price():
    """
    Test Black-Scholes call option pricing.

    Verifies that the analytical Black-Scholes formula correctly
    prices a European call option under standard market conditions.

    Test Case: ATM call option
    Spot Price: $100
    Strike Price: $100
    Time to Maturity: 1 year
    Risk-Free Rate: 5%
    Volatility: 20%
    Dividend Yield: 2%

    Expected Result: Price ≈ $8.15
    """
    # Test implementation
    pass
```

## Continuous Integration

### Test Automation
Tests should run automatically on:
- Every git commit
- Pull request creation
- Merge to main branch
- Nightly builds

### Performance Benchmarks
Track test execution time:
```python
def test_performance_benchmark():
    """Ensure VaR calculation completes within time limit."""
    start = time.time()
    result = engine.calculate_var(portfolio, data)
    elapsed = time.time() - start

    # Should complete within 5 seconds
    assert elapsed < 5.0, f"Calculation took {elapsed}s (too slow)"
```

## Test Data Files

### Data Directory
```
test/
├── data/                       # Test data files
│   ├── sample_market_data.csv
│   ├── option_chain.json
│   └── calibration_results.pkl
├── test_module.py
└── conftest.py
```

### Loading Test Data
```python
def test_with_data_file():
    """Test using external data file."""
    data_path = Path(__file__).parent / 'data' / 'market_data.csv'
    data = pd.read_csv(data_path)

    result = process_data(data)
    assert len(result) > 0
```

## Summary

Effective testing ensures:
- **Correctness**: Code works as expected
- **Reliability**: Consistent behavior
- **Maintainability**: Safe to refactor
- **Performance**: Meets performance requirements
- **Documentation**: Tests document expected behavior

Good tests are:
- **Fast**: Run in milliseconds
- **Independent**: Don't depend on other tests
- **Deterministic**: Same result every time
- **Clear**: Easy to understand
- **Comprehensive**: Cover all scenarios

"""
Test suite for portfolio management module.
"""
import sys
from pathlib import Path
from datetime import datetime
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.portfolio import Portfolio, Position, PortfolioSnapshot, PortfolioExporter
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError


# Test fixtures
@pytest.fixture
def pricing_env():
    """Create a test pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name="TEST"),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1)
    )


@pytest.fixture
def call_option():
    """Create a test call option."""
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0
    )


@pytest.fixture
def put_option():
    """Create a test put option."""
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0
    )


@pytest.fixture
def engine():
    """Create a test pricing engine."""
    return BlackScholesEngine()


@pytest.fixture
def portfolio(pricing_env):
    """Create a test portfolio."""
    return Portfolio(
        portfolio_name="Test Portfolio",
        pricing_environments={'TEST': pricing_env},
        creation_date=datetime(2024, 1, 1)
    )


# Position Tests
class TestPosition:
    """Tests for Position class."""
    
    def test_position_creation(self, call_option, engine):
        """Test creating a position."""
        pos = Position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine,
            entry_timestamp=datetime(2024, 1, 1)
        )
        
        assert pos.quantity == 10
        assert pos.entry_price == 5.0
        assert pos.underlying == 'TEST'
        assert pos.product == call_option
        assert pos.engine == engine
        assert pos.is_long()
        assert not pos.is_short()
    
    def test_position_short(self, put_option, engine):
        """Test creating a short position."""
        pos = Position(
            product=put_option,
            quantity=-5,
            entry_price=4.0,
            underlying='TEST',
            engine=engine,
            entry_timestamp=datetime(2024, 1, 1)
        )
        
        assert pos.quantity == -5
        assert pos.is_short()
        assert not pos.is_long()
    
    def test_position_validation_zero_quantity(self, call_option, engine):
        """Test that zero quantity raises error."""
        with pytest.raises(ValidationError, match="cannot be zero"):
            Position(
                product=call_option,
                quantity=0,
                entry_price=5.0,
                underlying='TEST',
                engine=engine,
                entry_timestamp=datetime(2024, 1, 1)
            )
    
    def test_position_validation_negative_entry_price(self, call_option, engine):
        """Test that negative entry price raises error."""
        with pytest.raises(ValidationError, match="non-negative"):
            Position(
                product=call_option,
                quantity=10,
                entry_price=-5.0,
                underlying='TEST',
                engine=engine,
                entry_timestamp=datetime(2024, 1, 1)
            )
    
    def test_position_current_price(self, call_option, engine, pricing_env):
        """Test getting current price."""
        pos = Position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine,
            entry_timestamp=datetime(2024, 1, 1)
        )
        
        current_price = pos.get_current_price(pricing_env)
        assert current_price > 0
        assert isinstance(current_price, float)
    
    def test_position_market_value(self, call_option, engine, pricing_env):
        """Test calculating market value."""
        pos = Position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine,
            entry_timestamp=datetime(2024, 1, 1)
        )
        
        market_value = pos.get_market_value(pricing_env)
        current_price = pos.get_current_price(pricing_env)
        assert abs(market_value - current_price * 10) < 1e-10
    
    def test_position_pnl(self, call_option, engine, pricing_env):
        """Test calculating P&L."""
        entry_price = 5.0
        pos = Position(
            product=call_option,
            quantity=10,
            entry_price=entry_price,
            underlying='TEST',
            engine=engine,
            entry_timestamp=datetime(2024, 1, 1)
        )
        
        pnl = pos.get_pnl(pricing_env)
        current_price = pos.get_current_price(pricing_env)
        expected_pnl = (current_price - entry_price) * 10
        assert abs(pnl - expected_pnl) < 1e-10
    
    def test_position_greeks(self, call_option, engine, pricing_env):
        """Test calculating position Greeks."""
        greeks_calc = GreeksCalculator()
        pos = Position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine,
            entry_timestamp=datetime(2024, 1, 1)
        )
        
        greeks = pos.get_greeks(pricing_env, greeks_calc)
        
        assert 'market_value' in greeks
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        assert 'rho' in greeks
    
    def test_position_to_dict(self, call_option, engine):
        """Test serializing position to dict."""
        pos = Position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine,
            entry_timestamp=datetime(2024, 1, 1)
        )
        
        pos_dict = pos.to_dict()
        
        assert pos_dict['position_id'] == pos.position_id
        assert pos_dict['underlying'] == 'TEST'
        assert pos_dict['quantity'] == 10
        assert pos_dict['entry_price'] == 5.0
        assert pos_dict['direction'] == 'LONG'


# Portfolio Tests
class TestPortfolio:
    """Tests for Portfolio class."""
    
    def test_portfolio_creation(self, pricing_env):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            portfolio_name="Test Portfolio",
            pricing_environments={'TEST': pricing_env}
        )
        
        assert portfolio.portfolio_name == "Test Portfolio"
        assert len(portfolio.positions) == 0
        assert 'TEST' in portfolio.pricing_environments
    
    def test_add_position(self, portfolio, call_option, engine):
        """Test adding a position to portfolio."""
        pos = portfolio.add_position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine
        )
        
        assert len(portfolio.positions) == 1
        assert pos.position_id in portfolio.positions
        assert portfolio.get_position(pos.position_id) == pos
    
    def test_add_position_invalid_underlying(self, portfolio, call_option, engine):
        """Test adding position with invalid underlying raises error."""
        with pytest.raises(ValidationError, match="not found"):
            portfolio.add_position(
                product=call_option,
                quantity=10,
                entry_price=5.0,
                underlying='INVALID',
                engine=engine
            )
    
    def test_remove_position(self, portfolio, call_option, engine):
        """Test removing a position."""
        pos = portfolio.add_position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine
        )
        
        removed = portfolio.remove_position(pos.position_id)
        
        assert removed == pos
        assert len(portfolio.positions) == 0
        assert portfolio.get_position(pos.position_id) is None
    
    def test_update_position_quantity(self, portfolio, call_option, engine):
        """Test updating position quantity."""
        pos = portfolio.add_position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine
        )
        
        portfolio.update_position(pos.position_id, quantity=15)
        
        assert pos.quantity == 15
    
    def test_update_position_entry_price(self, portfolio, call_option, engine):
        """Test updating position entry price."""
        pos = portfolio.add_position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine
        )
        
        portfolio.update_position(pos.position_id, entry_price=6.0)
        
        assert pos.entry_price == 6.0
    
    def test_update_position_zero_quantity_error(self, portfolio, call_option, engine):
        """Test that updating to zero quantity raises error."""
        pos = portfolio.add_position(
            product=call_option,
            quantity=10,
            entry_price=5.0,
            underlying='TEST',
            engine=engine
        )
        
        with pytest.raises(ValidationError, match="cannot be zero"):
            portfolio.update_position(pos.position_id, quantity=0)
    
    def test_get_positions_by_underlying(self, pricing_env):
        """Test filtering positions by underlying."""
        # Create portfolio with two underlyings
        env2 = PricingEnvironment(
            spot_quote=SpotQuote(spot=200.0, asset_name="TEST2"),
            vol_surface=FlatVolSurface(volatility=0.25),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=datetime(2024, 1, 1)
        )
        
        portfolio = Portfolio(
            portfolio_name="Multi-underlying Portfolio",
            pricing_environments={'TEST': pricing_env, 'TEST2': env2}
        )
        
        engine = BlackScholesEngine()
        option1 = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
        option2 = EuropeanVanillaOption(strike=200.0, option_type=OptionType.CALL, maturity=1.0)
        
        pos1 = portfolio.add_position(option1, 10, 5.0, 'TEST', engine)
        pos2 = portfolio.add_position(option2, 5, 10.0, 'TEST2', engine)
        pos3 = portfolio.add_position(option1, 3, 5.5, 'TEST', engine)
        
        test_positions = portfolio.get_positions_by_underlying('TEST')
        assert len(test_positions) == 2
        assert pos1 in test_positions
        assert pos3 in test_positions
        
        test2_positions = portfolio.get_positions_by_underlying('TEST2')
        assert len(test2_positions) == 1
        assert pos2 in test2_positions
    
    def test_portfolio_value(self, portfolio, call_option, put_option, engine):
        """Test calculating portfolio value."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        portfolio.add_position(put_option, 5, 3.0, 'TEST', engine)
        
        total_value = portfolio.get_portfolio_value()
        
        assert total_value > 0
        assert isinstance(total_value, float)
    
    def test_portfolio_pnl(self, portfolio, call_option, engine):
        """Test calculating portfolio P&L."""
        # Add position at entry price
        pos = portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        pnl = portfolio.get_portfolio_pnl()
        
        # Should have some P&L (positive or negative depending on current price)
        assert isinstance(pnl, float)
    
    def test_portfolio_greeks(self, portfolio, call_option, engine):
        """Test calculating portfolio Greeks."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        greeks_calc = GreeksCalculator()
        greeks = portfolio.get_portfolio_greeks(greeks_calc)
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        assert 'rho' in greeks
    
    def test_portfolio_to_dataframe(self, portfolio, call_option, put_option, engine):
        """Test converting portfolio to DataFrame."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        portfolio.add_position(put_option, -5, 3.0, 'TEST', engine)
        
        df = portfolio.to_dataframe()
        
        assert len(df) == 2
        assert 'position_id' in df.columns
        assert 'underlying' in df.columns
        assert 'quantity' in df.columns
        assert 'market_value' in df.columns
        assert 'pnl' in df.columns
    
    def test_portfolio_summary(self, portfolio, call_option, put_option, engine):
        """Test getting portfolio summary."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        portfolio.add_position(put_option, -5, 3.0, 'TEST', engine)
        
        summary = portfolio.get_summary()
        
        assert summary['portfolio_name'] == "Test Portfolio"
        assert summary['num_positions'] == 2
        assert summary['long_positions'] == 1
        assert summary['short_positions'] == 1
        assert 'total_value' in summary
        assert 'total_pnl' in summary


# Portfolio Snapshot Tests
class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot class."""
    
    def test_snapshot_from_portfolio(self, portfolio, call_option, engine):
        """Test creating snapshot from portfolio."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        greeks_calc = GreeksCalculator()
        snapshot = PortfolioSnapshot.from_portfolio(portfolio, greeks_calc)
        
        assert snapshot.portfolio_name == "Test Portfolio"
        assert len(snapshot.positions_data) == 1
        assert snapshot.total_value > 0
        assert 'aggregated_greeks' in snapshot.to_dict()
    
    def test_snapshot_to_dict(self, portfolio, call_option, engine):
        """Test serializing snapshot to dict."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        greeks_calc = GreeksCalculator()
        snapshot = PortfolioSnapshot.from_portfolio(portfolio, greeks_calc)
        
        snapshot_dict = snapshot.to_dict()
        
        assert 'timestamp' in snapshot_dict
        assert 'portfolio_name' in snapshot_dict
        assert 'total_value' in snapshot_dict
        assert 'positions' in snapshot_dict
    
    def test_snapshot_get_summary(self, portfolio, call_option, engine):
        """Test getting snapshot summary."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        greeks_calc = GreeksCalculator()
        snapshot = PortfolioSnapshot.from_portfolio(portfolio, greeks_calc)
        
        summary = snapshot.get_summary()
        
        assert 'timestamp' in summary
        assert 'total_value' in summary
        assert 'num_positions' in summary
        assert summary['num_positions'] == 1


# Portfolio Exporter Tests
class TestPortfolioExporter:
    """Tests for PortfolioExporter class."""
    
    def test_export_to_parquet(self, portfolio, call_option, engine, tmp_path):
        """Test exporting portfolio to Parquet."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        exporter = PortfolioExporter(base_path=str(tmp_path))
        greeks_calc = GreeksCalculator()
        
        filepath = exporter.export_to_parquet(
            portfolio,
            include_greeks=True,
            greeks_calculator=greeks_calc
        )
        
        assert filepath.exists()
        assert filepath.suffix == '.parquet'
    
    def test_export_to_excel(self, portfolio, call_option, engine, tmp_path):
        """Test exporting portfolio to Excel."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        exporter = PortfolioExporter(base_path=str(tmp_path))
        greeks_calc = GreeksCalculator()
        
        filepath = exporter.export_to_excel(
            portfolio,
            greeks_calculator=greeks_calc
        )
        
        assert filepath.exists()
        assert filepath.suffix == '.xlsx'
    
    def test_export_snapshot_to_parquet(self, portfolio, call_option, engine, tmp_path):
        """Test exporting snapshot to Parquet."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        greeks_calc = GreeksCalculator()
        snapshot = PortfolioSnapshot.from_portfolio(portfolio, greeks_calc)
        
        exporter = PortfolioExporter(base_path=str(tmp_path))
        filepath = exporter.export_snapshot_to_parquet(snapshot)
        
        assert filepath.exists()
        assert filepath.suffix == '.parquet'
    
    def test_load_from_parquet(self, portfolio, call_option, engine, tmp_path):
        """Test loading portfolio from Parquet."""
        portfolio.add_position(call_option, 10, 5.0, 'TEST', engine)
        
        exporter = PortfolioExporter(base_path=str(tmp_path))
        greeks_calc = GreeksCalculator()
        
        # Export
        filepath = exporter.export_to_parquet(
            portfolio,
            include_greeks=True,
            greeks_calculator=greeks_calc
        )
        
        # Load
        loaded_df = exporter.load_from_parquet(filepath)
        
        assert len(loaded_df) == 1
        assert 'metadata' in loaded_df.attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


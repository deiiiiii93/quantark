"""
Mock data generator for creating realistic synthetic market data.

Uses various stochastic models:
- Geometric Brownian Motion with jumps for spot prices
- Mean-reverting process for volatility (Ornstein-Uhlenbeck)
- Mean-reverting rates with bounds
- Synthetic vol surfaces with realistic skew
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from quantark.util.marketdata.models import MarketDataPoint, TimeSeriesData, OptionMarketData
from quantark.util.exceptions import ValidationError


class MockDataGenerator:
    """
    Generator for realistic synthetic market data.
    
    Generates correlated time series for spot prices, volatility,
    interest rates, and dividend yields using stochastic models.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
    
    def generate_spot_prices(self, 
                           initial_spot: float,
                           num_points: int,
                           dt: float = 1/252,  # Daily by default
                           drift: float = 0.05,
                           volatility: float = 0.20,
                           jump_intensity: float = 0.0,
                           jump_mean: float = -0.02,
                           jump_std: float = 0.03) -> np.ndarray:
        """
        Generate spot price path using GBM with optional jumps.
        
        Uses: dS/S = μ dt + σ dW + J dN
        where J is jump size and N is Poisson process
        
        Args:
            initial_spot: Starting spot price
            num_points: Number of time points
            dt: Time step (in years)
            drift: Annual drift rate
            volatility: Annual volatility
            jump_intensity: Jump intensity (jumps per year)
            jump_mean: Mean jump size (log-return)
            jump_std: Jump size standard deviation
            
        Returns:
            Array of spot prices
        """
        if initial_spot <= 0:
            raise ValidationError(f"Initial spot must be positive, got {initial_spot}")
        
        spots = np.zeros(num_points)
        spots[0] = initial_spot
        
        for i in range(1, num_points):
            # Brownian motion component
            dW = self.rng.normal(0, np.sqrt(dt))
            drift_component = (drift - 0.5 * volatility**2) * dt
            diffusion_component = volatility * dW
            
            # Jump component
            jump_component = 0
            if jump_intensity > 0:
                num_jumps = self.rng.poisson(jump_intensity * dt)
                if num_jumps > 0:
                    jumps = self.rng.normal(jump_mean, jump_std, num_jumps)
                    jump_component = np.sum(jumps)
            
            # Update spot
            log_return = drift_component + diffusion_component + jump_component
            spots[i] = spots[i-1] * np.exp(log_return)
        
        return spots
    
    def generate_stochastic_volatility(self,
                                      initial_vol: float,
                                      num_points: int,
                                      dt: float = 1/252,
                                      mean_vol: float = 0.20,
                                      mean_reversion_speed: float = 2.0,
                                      vol_of_vol: float = 0.5) -> np.ndarray:
        """
        Generate volatility path using Ornstein-Uhlenbeck process.
        
        Uses: dσ = κ(θ - σ)dt + ν dW
        where κ is mean reversion speed, θ is long-term mean, ν is vol-of-vol
        
        Args:
            initial_vol: Starting volatility
            num_points: Number of time points
            dt: Time step (in years)
            mean_vol: Long-term mean volatility
            mean_reversion_speed: Speed of mean reversion
            vol_of_vol: Volatility of volatility
            
        Returns:
            Array of volatility values
        """
        if initial_vol <= 0:
            raise ValidationError(f"Initial vol must be positive, got {initial_vol}")
        
        vols = np.zeros(num_points)
        vols[0] = initial_vol
        
        for i in range(1, num_points):
            dW = self.rng.normal(0, np.sqrt(dt))
            drift_component = mean_reversion_speed * (mean_vol - vols[i-1]) * dt
            diffusion_component = vol_of_vol * np.sqrt(vols[i-1]) * dW
            
            vols[i] = vols[i-1] + drift_component + diffusion_component
            
            # Keep volatility positive and reasonable
            vols[i] = max(0.05, min(vols[i], 2.0))
        
        return vols
    
    def generate_interest_rates(self,
                               initial_rate: float,
                               num_points: int,
                               dt: float = 1/252,
                               mean_rate: float = 0.03,
                               mean_reversion_speed: float = 0.5,
                               rate_volatility: float = 0.01,
                               min_rate: float = -0.01,
                               max_rate: float = 0.15) -> np.ndarray:
        """
        Generate interest rate path using bounded mean-reverting process.
        
        Args:
            initial_rate: Starting interest rate
            num_points: Number of time points
            dt: Time step (in years)
            mean_rate: Long-term mean rate
            mean_reversion_speed: Speed of mean reversion
            rate_volatility: Volatility of rates
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
            
        Returns:
            Array of interest rates
        """
        rates = np.zeros(num_points)
        rates[0] = initial_rate
        
        for i in range(1, num_points):
            dW = self.rng.normal(0, np.sqrt(dt))
            drift_component = mean_reversion_speed * (mean_rate - rates[i-1]) * dt
            diffusion_component = rate_volatility * dW
            
            rates[i] = rates[i-1] + drift_component + diffusion_component
            
            # Keep rates within bounds
            rates[i] = max(min_rate, min(rates[i], max_rate))
        
        return rates
    
    def generate_dividend_yields(self,
                                initial_div_yield: float,
                                num_points: int,
                                dt: float = 1/252,
                                mean_div_yield: float = 0.02,
                                mean_reversion_speed: float = 1.0,
                                div_volatility: float = 0.005) -> np.ndarray:
        """
        Generate dividend yield path using mean-reverting process.
        
        Args:
            initial_div_yield: Starting dividend yield
            num_points: Number of time points
            dt: Time step (in years)
            mean_div_yield: Long-term mean dividend yield
            mean_reversion_speed: Speed of mean reversion
            div_volatility: Volatility of dividend yields
            
        Returns:
            Array of dividend yields
        """
        div_yields = np.zeros(num_points)
        div_yields[0] = initial_div_yield
        
        for i in range(1, num_points):
            dW = self.rng.normal(0, np.sqrt(dt))
            drift_component = mean_reversion_speed * (mean_div_yield - div_yields[i-1]) * dt
            diffusion_component = div_volatility * dW
            
            div_yields[i] = div_yields[i-1] + drift_component + diffusion_component
            
            # Keep div yields non-negative and reasonable
            div_yields[i] = max(0.0, min(div_yields[i], 0.10))
        
        return div_yields
    
    def generate_market_data_series(self,
                                   start_date: datetime,
                                   end_date: datetime,
                                   asset_name: str = "MOCK",
                                   initial_spot: float = 100.0,
                                   initial_vol: float = 0.20,
                                   initial_rate: float = 0.05,
                                   initial_div_yield: float = 0.02,
                                   drift: float = 0.08,
                                   vol_of_vol: float = 0.3,
                                   jump_intensity: float = 0.0,
                                   frequency: str = 'D') -> Tuple[TimeSeriesData, TimeSeriesData, 
                                                                  TimeSeriesData, TimeSeriesData]:
        """
        Generate complete market data time series.
        
        Args:
            start_date: Start date
            end_date: End date
            asset_name: Asset identifier
            initial_spot: Starting spot price
            initial_vol: Starting volatility
            initial_rate: Starting interest rate
            initial_div_yield: Starting dividend yield
            drift: Annual drift for spot prices
            vol_of_vol: Volatility of volatility
            jump_intensity: Jump intensity for spot (jumps per year)
            frequency: Data frequency ('D' for daily, 'H' for hourly)
            
        Returns:
            Tuple of (spot_data, vol_data, rate_data, div_yield_data)
        """
        # Generate date range
        if frequency == 'D':
            dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
            dt = 1/252
        elif frequency == 'H':
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
            dt = 1/(252*24)
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
            dt = 1/252  # Default
        
        num_points = len(dates)
        
        # Generate paths
        spots = self.generate_spot_prices(
            initial_spot, num_points, dt, drift=drift, 
            volatility=initial_vol, jump_intensity=jump_intensity
        )
        
        vols = self.generate_stochastic_volatility(
            initial_vol, num_points, dt, mean_vol=initial_vol,
            vol_of_vol=vol_of_vol
        )
        
        rates = self.generate_interest_rates(
            initial_rate, num_points, dt, mean_rate=initial_rate
        )
        
        div_yields = self.generate_dividend_yields(
            initial_div_yield, num_points, dt, mean_div_yield=initial_div_yield
        )
        
        # Create DataFrames
        spot_df = pd.DataFrame({'spot': spots}, index=dates)
        vol_df = pd.DataFrame({'volatility': vols}, index=dates)
        rate_df = pd.DataFrame({'rate': rates}, index=dates)
        div_df = pd.DataFrame({'div_yield': div_yields}, index=dates)
        
        # Create TimeSeriesData objects
        metadata = {
            'generator': 'MockDataGenerator',
            'seed': self.seed,
            'initial_spot': initial_spot,
            'drift': drift,
            'vol_of_vol': vol_of_vol
        }
        
        spot_ts = TimeSeriesData(spot_df, asset_name, 'spot', metadata)
        vol_ts = TimeSeriesData(vol_df, asset_name, 'volatility', metadata)
        rate_ts = TimeSeriesData(rate_df, asset_name, 'rate', metadata)
        div_ts = TimeSeriesData(div_df, asset_name, 'div_yield', metadata)
        
        return spot_ts, vol_ts, rate_ts, div_ts
    
    def generate_vol_surface(self,
                            spot: float,
                            base_vol: float,
                            strikes: np.ndarray,
                            maturities: np.ndarray,
                            skew: float = -0.1,
                            smile: float = 0.05,
                            term_structure_slope: float = -0.02) -> np.ndarray:
        """
        Generate synthetic volatility surface with skew and smile.
        
        Args:
            spot: Current spot price
            base_vol: Base volatility level
            strikes: Array of strike prices
            maturities: Array of maturities (in years)
            skew: Skew parameter (negative for typical equity skew)
            smile: Smile parameter (controls curvature)
            term_structure_slope: Slope of term structure
            
        Returns:
            2D array of implied vols (strikes x maturities)
        """
        # Create meshgrid
        K_grid, T_grid = np.meshgrid(strikes, maturities, indexing='ij')
        
        # Moneyness
        moneyness = np.log(K_grid / spot) / np.sqrt(T_grid)
        
        # Vol surface with skew and smile
        vol_surface = base_vol * (
            1.0 + 
            skew * moneyness +  # Linear skew
            smile * moneyness**2 +  # Smile (convexity)
            term_structure_slope * (T_grid - 1.0)  # Term structure
        )
        
        # Keep vols positive and reasonable
        vol_surface = np.clip(vol_surface, 0.05, 2.0)
        
        return vol_surface
    
    def compute_option_price_bs(self,
                               spot: float,
                               strike: float,
                               maturity: float,
                               volatility: float,
                               rate: float,
                               div_yield: float,
                               option_type: str) -> float:
        """
        Compute Black-Scholes option price.
        
        Args:
            spot: Spot price
            strike: Strike price
            maturity: Time to maturity (years)
            volatility: Implied volatility
            rate: Risk-free rate
            div_yield: Dividend yield
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        from scipy.stats import norm
        
        if maturity <= 0:
            # At expiry
            if option_type.lower() == 'call':
                return max(spot - strike, 0)
            else:
                return max(strike - spot, 0)
        
        d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * volatility**2) * maturity) / (volatility * np.sqrt(maturity))
        d2 = d1 - volatility * np.sqrt(maturity)
        
        if option_type.lower() == 'call':
            price = spot * np.exp(-div_yield * maturity) * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
        else:
            price = strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * np.exp(-div_yield * maturity) * norm.cdf(-d1)
        
        return price
    
    def generate_option_greeks_bs(self,
                                 spot: float,
                                 strike: float,
                                 maturity: float,
                                 volatility: float,
                                 rate: float,
                                 div_yield: float,
                                 option_type: str) -> Dict[str, float]:
        """
        Compute Black-Scholes Greeks analytically.
        
        Args:
            spot: Spot price
            strike: Strike price
            maturity: Time to maturity (years)
            volatility: Implied volatility
            rate: Risk-free rate
            div_yield: Dividend yield
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary of Greeks
        """
        from scipy.stats import norm
        
        if maturity <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
        
        d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * volatility**2) * maturity) / (volatility * np.sqrt(maturity))
        d2 = d1 - volatility * np.sqrt(maturity)
        
        # Delta
        if option_type.lower() == 'call':
            delta = np.exp(-div_yield * maturity) * norm.cdf(d1)
        else:
            delta = np.exp(-div_yield * maturity) * (norm.cdf(d1) - 1)
        
        # Gamma (same for call and put)
        gamma = np.exp(-div_yield * maturity) * norm.pdf(d1) / (spot * volatility * np.sqrt(maturity))
        
        # Vega (same for call and put)
        vega = spot * np.exp(-div_yield * maturity) * norm.pdf(d1) * np.sqrt(maturity) / 100  # Per 1% change
        
        # Theta
        term1 = -spot * norm.pdf(d1) * volatility * np.exp(-div_yield * maturity) / (2 * np.sqrt(maturity))
        if option_type.lower() == 'call':
            term2 = -rate * strike * np.exp(-rate * maturity) * norm.cdf(d2)
            term3 = div_yield * spot * np.exp(-div_yield * maturity) * norm.cdf(d1)
            theta = (term1 + term2 + term3) / 365  # Per day
        else:
            term2 = rate * strike * np.exp(-rate * maturity) * norm.cdf(-d2)
            term3 = -div_yield * spot * np.exp(-div_yield * maturity) * norm.cdf(-d1)
            theta = (term1 + term2 + term3) / 365  # Per day
        
        # Rho
        if option_type.lower() == 'call':
            rho = strike * maturity * np.exp(-rate * maturity) * norm.cdf(d2) / 100  # Per 1% change
        else:
            rho = -strike * maturity * np.exp(-rate * maturity) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def add_bid_ask_spread(self, price: float, spread_pct: float = 0.01) -> Tuple[float, float]:
        """
        Add realistic bid-ask spread to a price.
        
        Args:
            price: Mid price
            spread_pct: Spread as percentage of price
            
        Returns:
            Tuple of (bid, ask)
        """
        half_spread = price * spread_pct / 2
        bid = price - half_spread
        ask = price + half_spread
        return bid, ask


"""
Library of predefined stress test scenarios.

This module provides common market stress scenarios that can be used
directly or customized for specific needs.
"""

from typing import Optional, Sequence
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.scenario.scenario_builder import ScenarioBuilder
from quantark.stresstest.stress.stress_types import StressType, StressLevel


class ScenarioLibrary:
    """
    Library of predefined stress test scenarios.
    
    Provides factory methods for creating common market stress scenarios
    such as market crashes, volatility spikes, rate shocks, etc.
    
    Example:
        >>> scenarios = [
        ...     ScenarioLibrary.market_crash(),
        ...     ScenarioLibrary.vol_spike(),
        ...     ScenarioLibrary.rate_hike()
        ... ]
    """
    
    @staticmethod
    def market_crash(
        spot_shock: float = -0.20,
        vol_increase: float = 0.50,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Market crash scenario: equity drop with vol spike.
        
        Args:
            spot_shock: Percentage spot price drop (default: -20%)
            vol_increase: Percentage vol increase (default: +50%)
            underlying: Optional target underlying (None = portfolio-wide)
            
        Returns:
            Market crash scenario
        """
        level, target = _get_level_target(underlying)
        
        return Scenario(
            name="Market Crash",
            description=f"Equity drops {spot_shock:.0%} with volatility spike {vol_increase:+.0%}",
            stresses=[
                Stress("spot", StressType.PERCENTAGE, spot_shock, level, target),
                Stress("volatility", StressType.PERCENTAGE, vol_increase, level, target),
            ],
            metadata={"category": "equity", "severity": "high"}
        )
    
    @staticmethod
    def market_rally(
        spot_increase: float = 0.15,
        vol_decrease: float = -0.30,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Market rally scenario: equity rise with vol compression.
        
        Args:
            spot_increase: Percentage spot price increase (default: +15%)
            vol_decrease: Percentage vol decrease (default: -30%)
            underlying: Optional target underlying
            
        Returns:
            Market rally scenario
        """
        level, target = _get_level_target(underlying)
        
        return Scenario(
            name="Market Rally",
            description=f"Equity rises {spot_increase:+.0%} with volatility compression {vol_decrease:.0%}",
            stresses=[
                Stress("spot", StressType.PERCENTAGE, spot_increase, level, target),
                Stress("volatility", StressType.PERCENTAGE, vol_decrease, level, target),
            ],
            metadata={"category": "equity", "severity": "low"}
        )
    
    @staticmethod
    def vol_spike(
        vol_increase: float = 0.80,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Volatility spike scenario: pure vol increase.
        
        Args:
            vol_increase: Percentage vol increase (default: +80%)
            underlying: Optional target underlying
            
        Returns:
            Volatility spike scenario
        """
        level, target = _get_level_target(underlying)
        
        return Scenario(
            name="Volatility Spike",
            description=f"Volatility increases {vol_increase:+.0%}",
            stresses=[
                Stress("volatility", StressType.PERCENTAGE, vol_increase, level, target),
            ],
            metadata={"category": "volatility", "severity": "medium"}
        )
    
    @staticmethod
    def vol_crush(
        vol_decrease: float = -0.50,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Volatility crush scenario: pure vol decrease.
        
        Args:
            vol_decrease: Percentage vol decrease (default: -50%)
            underlying: Optional target underlying
            
        Returns:
            Volatility crush scenario
        """
        level, target = _get_level_target(underlying)
        
        return Scenario(
            name="Volatility Crush",
            description=f"Volatility decreases {vol_decrease:.0%}",
            stresses=[
                Stress("volatility", StressType.PERCENTAGE, vol_decrease, level, target),
            ],
            metadata={"category": "volatility", "severity": "low"}
        )
    
    @staticmethod
    def rate_hike(
        rate_increase: float = 0.02,
        stress_type: StressType = StressType.ABSOLUTE,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Interest rate hike scenario.
        
        Args:
            rate_increase: Rate increase (default: +200bps absolute)
            stress_type: Type of stress (ABSOLUTE or PERCENTAGE)
            underlying: Optional target underlying
            
        Returns:
            Rate hike scenario
        """
        level, target = _get_level_target(underlying)
        
        desc = f"Interest rates increase by {rate_increase*10000:.0f}bps" if stress_type == StressType.ABSOLUTE \
               else f"Interest rates increase by {rate_increase:.0%}"
        
        return Scenario(
            name="Rate Hike",
            description=desc,
            stresses=[
                Stress("rate", stress_type, rate_increase, level, target),
            ],
            metadata={"category": "rates", "severity": "medium"}
        )
    
    @staticmethod
    def rate_cut(
        rate_decrease: float = -0.015,
        stress_type: StressType = StressType.ABSOLUTE,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Interest rate cut scenario.
        
        Args:
            rate_decrease: Rate decrease (default: -150bps absolute)
            stress_type: Type of stress (ABSOLUTE or PERCENTAGE)
            underlying: Optional target underlying
            
        Returns:
            Rate cut scenario
        """
        level, target = _get_level_target(underlying)
        
        desc = f"Interest rates decrease by {abs(rate_decrease)*10000:.0f}bps" if stress_type == StressType.ABSOLUTE \
               else f"Interest rates decrease by {abs(rate_decrease):.0%}"
        
        return Scenario(
            name="Rate Cut",
            description=desc,
            stresses=[
                Stress("rate", stress_type, rate_decrease, level, target),
            ],
            metadata={"category": "rates", "severity": "low"}
        )
    
    @staticmethod
    def severe_downturn(
        spot_shock: float = -0.35,
        vol_increase: float = 1.00,
        rate_change: float = -0.01,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Severe market downturn: major crash with flight to quality.
        
        Args:
            spot_shock: Percentage spot drop (default: -35%)
            vol_increase: Percentage vol increase (default: +100%)
            rate_change: Rate change (default: -100bps, flight to quality)
            underlying: Optional target underlying
            
        Returns:
            Severe downturn scenario
        """
        level, target = _get_level_target(underlying)
        
        return Scenario(
            name="Severe Downturn",
            description="Severe market crash with flight to quality",
            stresses=[
                Stress("spot", StressType.PERCENTAGE, spot_shock, level, target),
                Stress("volatility", StressType.PERCENTAGE, vol_increase, level, target),
                Stress("rate", StressType.ABSOLUTE, rate_change, level, target),
            ],
            metadata={"category": "combined", "severity": "extreme"}
        )
    
    @staticmethod
    def inflation_shock(
        rate_increase: float = 0.03,
        vol_increase: float = 0.30,
        spot_shock: float = -0.10,
        underlying: Optional[str] = None
    ) -> Scenario:
        """
        Inflation shock scenario: rising rates with equity pressure.
        
        Args:
            rate_increase: Rate increase (default: +300bps)
            vol_increase: Percentage vol increase (default: +30%)
            spot_shock: Percentage spot drop (default: -10%)
            underlying: Optional target underlying
            
        Returns:
            Inflation shock scenario
        """
        level, target = _get_level_target(underlying)
        
        return Scenario(
            name="Inflation Shock",
            description="Rising rates with equity market pressure",
            stresses=[
                Stress("rate", StressType.ABSOLUTE, rate_increase, level, target),
                Stress("volatility", StressType.PERCENTAGE, vol_increase, level, target),
                Stress("spot", StressType.PERCENTAGE, spot_shock, level, target),
            ],
            metadata={"category": "combined", "severity": "high"}
        )

    @staticmethod
    def fi_parallel_shift(
        shift_bps: float = 0.01,
        curve_name: str = "UST",
        buckets: Optional[Sequence[str]] = None,
    ) -> Scenario:
        """Parallel shift for a fixed income curve."""
        if buckets is None:
            buckets = ["2Y", "5Y", "10Y", "30Y"]

        builder = (
            ScenarioBuilder()
            .name("FI Parallel Shift")
            .description(f"{curve_name} curve shift of {shift_bps*10000:.0f} bps")
        )
        for bucket in buckets:
            builder.key_rate_stress(shift_bps, tenor_bucket=bucket, curve=curve_name)
        return builder.build()

    @staticmethod
    def fi_steepener(
        front_end_bps: float = 0.01,
        long_end_bps: float = -0.005,
        curve_name: str = "UST",
        front_bucket: str = "2Y",
        long_bucket: str = "30Y",
    ) -> Scenario:
        """Bear steepener scenario."""
        builder = (
            ScenarioBuilder()
            .name("FI Steepener")
            .description(
                f"{curve_name} {front_bucket} {front_end_bps:+.2%}, {long_bucket} {long_end_bps:+.2%}"
            )
        )
        builder.key_rate_stress(front_end_bps, tenor_bucket=front_bucket, curve=curve_name)
        builder.key_rate_stress(long_end_bps, tenor_bucket=long_bucket, curve=curve_name)
        return builder.build()

    @staticmethod
    def fi_spread_shock(
        spread_bps: float = 0.002,
        curve_name: str = "IG",
    ) -> Scenario:
        """Credit spread shock scenario."""
        builder = (
            ScenarioBuilder()
            .name("FI Spread Shock")
            .description(f"{curve_name} spread move of {spread_bps:+.2%}")
        )
        builder.spread_stress(spread_bps, spread_curve=curve_name)
        return builder.build()
    
    @staticmethod
    def black_monday_1987() -> Scenario:
        """
        Historical scenario: Black Monday (October 19, 1987).
        
        22.6% drop in one day with extreme vol spike.
        
        Returns:
            Black Monday scenario
        """
        return Scenario(
            name="Black Monday 1987",
            description="Historical: Oct 19, 1987 - 22.6% drop",
            stresses=[
                Stress("spot", StressType.PERCENTAGE, -0.226, StressLevel.PORTFOLIO, None),
                Stress("volatility", StressType.PERCENTAGE, 1.50, StressLevel.PORTFOLIO, None),
            ],
            metadata={
                "category": "historical",
                "date": "1987-10-19",
                "severity": "extreme"
            }
        )
    
    @staticmethod
    def financial_crisis_2008() -> Scenario:
        """
        Historical scenario: 2008 Financial Crisis peak stress.
        
        Approximates market conditions during Sept-Oct 2008.
        
        Returns:
            2008 crisis scenario
        """
        return Scenario(
            name="Financial Crisis 2008",
            description="Historical: Sep-Oct 2008 stress period",
            stresses=[
                Stress("spot", StressType.PERCENTAGE, -0.40, StressLevel.PORTFOLIO, None),
                Stress("volatility", StressType.PERCENTAGE, 1.20, StressLevel.PORTFOLIO, None),
                Stress("rate", StressType.ABSOLUTE, -0.02, StressLevel.PORTFOLIO, None),
            ],
            metadata={
                "category": "historical",
                "date": "2008-09-15",
                "severity": "extreme"
            }
        )
    
    @staticmethod
    def covid_crash_2020() -> Scenario:
        """
        Historical scenario: COVID-19 crash (March 2020).
        
        Rapid 34% decline in about a month.
        
        Returns:
            COVID crash scenario
        """
        return Scenario(
            name="COVID Crash 2020",
            description="Historical: March 2020 - rapid 34% decline",
            stresses=[
                Stress("spot", StressType.PERCENTAGE, -0.34, StressLevel.PORTFOLIO, None),
                Stress("volatility", StressType.PERCENTAGE, 2.00, StressLevel.PORTFOLIO, None),
                Stress("rate", StressType.ABSOLUTE, -0.015, StressLevel.PORTFOLIO, None),
            ],
            metadata={
                "category": "historical",
                "date": "2020-03-16",
                "severity": "extreme"
            }
        )
    
    @staticmethod
    def get_all_predefined() -> list:
        """
        Get all predefined scenarios with default parameters.
        
        Returns:
            List of all predefined scenarios
        """
        return [
            ScenarioLibrary.market_crash(),
            ScenarioLibrary.market_rally(),
            ScenarioLibrary.vol_spike(),
            ScenarioLibrary.vol_crush(),
            ScenarioLibrary.rate_hike(),
            ScenarioLibrary.rate_cut(),
            ScenarioLibrary.severe_downturn(),
            ScenarioLibrary.inflation_shock(),
            ScenarioLibrary.fi_parallel_shift(),
            ScenarioLibrary.fi_steepener(),
            ScenarioLibrary.fi_spread_shock(),
        ]
    
    @staticmethod
    def get_historical_scenarios() -> list:
        """
        Get historical stress scenarios.
        
        Returns:
            List of historical scenarios
        """
        return [
            ScenarioLibrary.black_monday_1987(),
            ScenarioLibrary.financial_crisis_2008(),
            ScenarioLibrary.covid_crash_2020(),
        ]


def _get_level_target(underlying: Optional[str]) -> tuple:
    """Helper to determine stress level and target."""
    if underlying:
        return StressLevel.UNDERLYING, underlying
    else:
        return StressLevel.PORTFOLIO, None


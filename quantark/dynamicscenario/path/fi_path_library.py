"""
Library of predefined FI-specific day path patterns.

This module provides factory methods for creating common Fixed Income
multi-day market scenarios focused on rate curve movements.
"""

from typing import Optional, List
from datetime import datetime
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.dynamicscenario.path.path_builder import PathBuilder
from quantark.stresstest.stress.stress_types import StressType, StressLevel


class FIPathLibrary:
    """
    Library of predefined FI day path patterns.
    
    Provides factory methods for creating common Fixed Income scenarios
    such as rate hike cycles, parallel shifts, curve twists, etc.
    
    Example:
        >>> # Get a parallel shift scenario
        >>> path = FIPathLibrary.parallel_shift(days=5, total_bps=50)
        
        >>> # Get a Fed tightening cycle
        >>> path = FIPathLibrary.rate_hike_cycle(days=10, total_bps=100)
    """
    
    @staticmethod
    def parallel_shift(
        days: int = 5,
        total_bps: float = 50,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a parallel rate shift scenario.
        
        Uniform rate change across all tenors over N days.
        
        Args:
            days: Number of days (default: 5)
            total_bps: Total rate shift in basis points (default: +50bps)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for parallel shift scenario
            
        Example:
            >>> path = FIPathLibrary.parallel_shift(days=5, total_bps=50)
            >>> # Creates 5 days of +10bps rate increase each day
        """
        daily_bps = total_bps / days
        
        builder = PathBuilder(
            num_days=days,
            name=f"Parallel Shift {total_bps:+.0f}bps",
            description=f"Parallel rate shift of {total_bps:+.0f}bps over {days} days",
            start_date=start_date,
        )
        
        builder.rate_parallel_shift(
            daily_bps=daily_bps,
            underlying=underlying,
        )
        
        # Also apply to base 'rate' parameter for compatibility
        builder.rate_trend(
            daily_change=daily_bps / 10000,
            stress_type=StressType.ABSOLUTE,
            underlying=underlying,
        )
        
        builder.metadata("pattern", "parallel_shift")
        builder.metadata("total_bps", total_bps)
        builder.metadata("asset_class", "fixed_income")
        
        return builder.build()
    
    @staticmethod
    def steepener(
        days: int = 5,
        short_bps: float = -25,
        long_bps: float = 25,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a curve steepener scenario.
        
        Short-end rates decrease while long-end rates increase,
        widening the yield curve spread.
        
        Args:
            days: Number of days (default: 5)
            short_bps: Total short-end change in bps (default: -25bps)
            long_bps: Total long-end change in bps (default: +25bps)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for steepener scenario
        """
        daily_short = short_bps / days
        daily_long = long_bps / days
        
        builder = PathBuilder(
            num_days=days,
            name=f"Curve Steepener ({short_bps:+.0f}bps/{long_bps:+.0f}bps)",
            description=f"Curve steepening: short {short_bps:+.0f}bps, long {long_bps:+.0f}bps over {days} days",
            start_date=start_date,
        )
        
        builder.rate_curve_twist(
            short_daily_bps=daily_short,
            long_daily_bps=daily_long,
            underlying=underlying,
        )
        
        # Labels
        for day in range(days):
            cum_short = (day + 1) * daily_short
            cum_long = (day + 1) * daily_long
            builder.set_day_label(day, f"Short {cum_short:+.0f}bps, Long {cum_long:+.0f}bps")
        
        builder.metadata("pattern", "steepener")
        builder.metadata("short_bps", short_bps)
        builder.metadata("long_bps", long_bps)
        builder.metadata("asset_class", "fixed_income")
        
        return builder.build()
    
    @staticmethod
    def flattener(
        days: int = 5,
        short_bps: float = 25,
        long_bps: float = -25,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a curve flattener scenario.
        
        Short-end rates increase while long-end rates decrease,
        narrowing the yield curve spread.
        
        Args:
            days: Number of days (default: 5)
            short_bps: Total short-end change in bps (default: +25bps)
            long_bps: Total long-end change in bps (default: -25bps)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for flattener scenario
        """
        daily_short = short_bps / days
        daily_long = long_bps / days
        
        builder = PathBuilder(
            num_days=days,
            name=f"Curve Flattener ({short_bps:+.0f}bps/{long_bps:+.0f}bps)",
            description=f"Curve flattening: short {short_bps:+.0f}bps, long {long_bps:+.0f}bps over {days} days",
            start_date=start_date,
        )
        
        builder.rate_curve_twist(
            short_daily_bps=daily_short,
            long_daily_bps=daily_long,
            underlying=underlying,
        )
        
        # Labels
        for day in range(days):
            cum_short = (day + 1) * daily_short
            cum_long = (day + 1) * daily_long
            builder.set_day_label(day, f"Short {cum_short:+.0f}bps, Long {cum_long:+.0f}bps")
        
        builder.metadata("pattern", "flattener")
        builder.metadata("short_bps", short_bps)
        builder.metadata("long_bps", long_bps)
        builder.metadata("asset_class", "fixed_income")
        
        return builder.build()
    
    @staticmethod
    def rate_hike_cycle(
        days: int = 10,
        total_bps: float = 100,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a gradual rate hike cycle scenario.
        
        Simulates a central bank tightening cycle with gradual rate increases.
        
        Args:
            days: Number of days (default: 10)
            total_bps: Total rate hike in basis points (default: +100bps)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for rate hike cycle
        """
        daily_bps = total_bps / days
        
        builder = PathBuilder(
            num_days=days,
            name=f"Rate Hike Cycle {total_bps:+.0f}bps",
            description=f"Gradual rate increase of {total_bps:+.0f}bps over {days} days",
            start_date=start_date,
        )
        
        builder.rate_parallel_shift(
            daily_bps=daily_bps,
            underlying=underlying,
        )
        
        builder.rate_trend(
            daily_change=daily_bps / 10000,
            stress_type=StressType.ABSOLUTE,
            underlying=underlying,
        )
        
        # Labels marking cumulative hikes
        for day in range(days):
            cumulative = (day + 1) * daily_bps
            builder.set_day_label(day, f"Cumulative hike: {cumulative:+.0f}bps")
        
        builder.metadata("pattern", "rate_hike_cycle")
        builder.metadata("total_bps", total_bps)
        builder.metadata("direction", "tightening")
        builder.metadata("asset_class", "fixed_income")
        
        return builder.build()
    
    @staticmethod
    def rate_cut_cycle(
        days: int = 10,
        total_bps: float = -100,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a gradual rate cut cycle scenario.
        
        Simulates a central bank easing cycle with gradual rate decreases.
        
        Args:
            days: Number of days (default: 10)
            total_bps: Total rate cut in basis points (default: -100bps)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for rate cut cycle
        """
        # Ensure negative value
        if total_bps > 0:
            total_bps = -total_bps
        
        daily_bps = total_bps / days
        
        builder = PathBuilder(
            num_days=days,
            name=f"Rate Cut Cycle {total_bps:+.0f}bps",
            description=f"Gradual rate decrease of {total_bps}bps over {days} days",
            start_date=start_date,
        )
        
        builder.rate_parallel_shift(
            daily_bps=daily_bps,
            underlying=underlying,
        )
        
        builder.rate_trend(
            daily_change=daily_bps / 10000,
            stress_type=StressType.ABSOLUTE,
            underlying=underlying,
        )
        
        # Labels marking cumulative cuts
        for day in range(days):
            cumulative = (day + 1) * daily_bps
            builder.set_day_label(day, f"Cumulative cut: {cumulative:+.0f}bps")
        
        builder.metadata("pattern", "rate_cut_cycle")
        builder.metadata("total_bps", total_bps)
        builder.metadata("direction", "easing")
        builder.metadata("asset_class", "fixed_income")
        
        return builder.build()
    
    @staticmethod
    def historical_fed_tightening_2022(
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a scenario modeled on 2022 Fed rate hike cycle.
        
        Simulates the aggressive tightening cycle of 2022 with
        multiple large rate hikes.
        
        Args:
            start_date: Optional start date
            
        Returns:
            DayPath for Fed 2022 tightening scenario
        """
        # Simplified 10-day version representing major Fed meetings
        builder = PathBuilder(
            num_days=10,
            name="Fed Tightening 2022 Style",
            description="Aggressive rate hikes modeled on 2022 Fed cycle",
            start_date=start_date,
        )
        
        # Simulate progressive Fed hikes (March-Dec 2022 compressed)
        # 25bps, 50bps, 75bps, 75bps, 75bps, 75bps, 50bps pattern
        hikes_bps = [25, 50, 75, 75, 75, 75, 50, 50, 25, 0]
        
        for day, hike in enumerate(hikes_bps):
            if hike > 0:
                builder.set_day_change(
                    day, "rate", hike / 10000, StressType.ABSOLUTE
                )
                builder.set_day_change(
                    day, "rate_parallel", hike / 10000, StressType.ABSOLUTE
                )
                builder.set_day_label(day, f"Fed hike +{hike}bps")
            else:
                builder.set_day_label(day, "No change")
        
        builder.metadata("pattern", "historical_fed_tightening_2022")
        builder.metadata("total_bps", sum(hikes_bps))
        builder.metadata("asset_class", "fixed_income")
        builder.metadata("historical_reference", "2022 Fed tightening cycle")
        
        return builder.build()
    
    @staticmethod
    def bear_steepener(
        days: int = 5,
        short_bps: float = 50,
        long_bps: float = 75,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a bear steepener scenario.
        
        Both short and long rates rise, but long rates rise more,
        causing curve to steepen. This typically happens during
        inflationary periods or rising rate expectations.
        
        Args:
            days: Number of days
            short_bps: Total short-end increase in bps
            long_bps: Total long-end increase in bps (should be > short_bps)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for bear steepener scenario
        """
        daily_short = short_bps / days
        daily_long = long_bps / days
        
        builder = PathBuilder(
            num_days=days,
            name=f"Bear Steepener ({short_bps:+.0f}bps/{long_bps:+.0f}bps)",
            description=f"Bear steepening: short {short_bps:+.0f}bps, long {long_bps:+.0f}bps",
            start_date=start_date,
        )
        
        builder.rate_curve_twist(
            short_daily_bps=daily_short,
            long_daily_bps=daily_long,
            underlying=underlying,
        )
        
        builder.metadata("pattern", "bear_steepener")
        builder.metadata("short_bps", short_bps)
        builder.metadata("long_bps", long_bps)
        builder.metadata("asset_class", "fixed_income")
        
        return builder.build()
    
    @staticmethod
    def bull_flattener(
        days: int = 5,
        short_bps: float = -25,
        long_bps: float = -50,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a bull flattener scenario.
        
        Both short and long rates fall, but long rates fall more,
        causing curve to flatten. This typically happens during
        flight-to-quality or rate cut expectations.
        
        Args:
            days: Number of days
            short_bps: Total short-end decrease in bps
            long_bps: Total long-end decrease in bps (should be < short_bps)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for bull flattener scenario
        """
        daily_short = short_bps / days
        daily_long = long_bps / days
        
        builder = PathBuilder(
            num_days=days,
            name=f"Bull Flattener ({short_bps:+.0f}bps/{long_bps:+.0f}bps)",
            description=f"Bull flattening: short {short_bps:+.0f}bps, long {long_bps:+.0f}bps",
            start_date=start_date,
        )
        
        builder.rate_curve_twist(
            short_daily_bps=daily_short,
            long_daily_bps=daily_long,
            underlying=underlying,
        )
        
        builder.metadata("pattern", "bull_flattener")
        builder.metadata("short_bps", short_bps)
        builder.metadata("long_bps", long_bps)
        builder.metadata("asset_class", "fixed_income")
        
        return builder.build()
    
    @staticmethod
    def get_all_predefined() -> List[DayPath]:
        """
        Get all predefined FI scenarios with default parameters.
        
        Returns:
            List of all predefined FI day paths
        """
        return [
            FIPathLibrary.parallel_shift(),
            FIPathLibrary.steepener(),
            FIPathLibrary.flattener(),
            FIPathLibrary.rate_hike_cycle(),
            FIPathLibrary.rate_cut_cycle(),
            FIPathLibrary.historical_fed_tightening_2022(),
            FIPathLibrary.bear_steepener(),
            FIPathLibrary.bull_flattener(),
        ]
    
    @staticmethod
    def get_tightening_scenarios() -> List[DayPath]:
        """Get rate tightening scenarios."""
        return [
            FIPathLibrary.parallel_shift(total_bps=50),
            FIPathLibrary.rate_hike_cycle(),
            FIPathLibrary.historical_fed_tightening_2022(),
            FIPathLibrary.bear_steepener(),
        ]
    
    @staticmethod
    def get_easing_scenarios() -> List[DayPath]:
        """Get rate easing scenarios."""
        return [
            FIPathLibrary.parallel_shift(total_bps=-50),
            FIPathLibrary.rate_cut_cycle(),
            FIPathLibrary.bull_flattener(),
        ]
    
    @staticmethod
    def get_curve_scenarios() -> List[DayPath]:
        """Get curve movement scenarios (steepeners/flatteners)."""
        return [
            FIPathLibrary.steepener(),
            FIPathLibrary.flattener(),
            FIPathLibrary.bear_steepener(),
            FIPathLibrary.bull_flattener(),
        ]


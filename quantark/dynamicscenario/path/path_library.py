"""
Library of predefined day path patterns.

This module provides factory methods for creating common multi-day
market scenarios easily.
"""

from typing import Optional, List
from datetime import datetime
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.dynamicscenario.path.path_builder import PathBuilder
from quantark.stresstest.stress.stress_types import StressType, StressLevel


class PathLibrary:
    """
    Library of predefined day path patterns.
    
    Provides factory methods for creating common market scenarios
    such as rallies, declines, V-shaped recoveries, volatility spikes, etc.
    
    Example:
        >>> # Get a 5-day rally scenario
        >>> path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)
        
        >>> # Get a V-shaped recovery
        >>> path = PathLibrary.v_shaped_recovery(down_days=3, up_days=5, magnitude=0.15)
    """
    
    @staticmethod
    def consecutive_rally(
        days: int = 5,
        daily_pct: float = 0.02,
        vol_change_pct: float = -0.05,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a consecutive rally scenario.
        
        N days of consistent price increase with optional volatility compression.
        
        Args:
            days: Number of days (default: 5)
            daily_pct: Daily price increase (default: +2%)
            vol_change_pct: Daily volatility change (default: -5% per day)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for rally scenario
            
        Example:
            >>> path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)
            >>> # Creates 5 days of +2% spot increase each day
        """
        builder = PathBuilder(
            num_days=days,
            name=f"{days}-Day Rally",
            description=f"{days} consecutive days of {daily_pct:.1%} price increase",
            start_date=start_date,
        )
        
        builder.spot_trend(
            daily_change=daily_pct,
            stress_type=StressType.PERCENTAGE,
            underlying=underlying,
        )
        
        if vol_change_pct != 0:
            builder.vol_trend(
                daily_change=vol_change_pct,
                stress_type=StressType.PERCENTAGE,
                underlying=underlying,
            )
        
        builder.metadata("pattern", "consecutive_rally")
        builder.metadata("severity", "bullish")
        
        return builder.build()
    
    @staticmethod
    def consecutive_decline(
        days: int = 5,
        daily_pct: float = -0.02,
        vol_change_pct: float = 0.10,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a consecutive decline scenario.
        
        N days of consistent price decline with optional volatility increase.
        
        Args:
            days: Number of days (default: 5)
            daily_pct: Daily price change (default: -2%)
            vol_change_pct: Daily volatility change (default: +10% per day)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for decline scenario
        """
        builder = PathBuilder(
            num_days=days,
            name=f"{days}-Day Decline",
            description=f"{days} consecutive days of {daily_pct:.1%} price decline",
            start_date=start_date,
        )
        
        builder.spot_trend(
            daily_change=daily_pct,
            stress_type=StressType.PERCENTAGE,
            underlying=underlying,
        )
        
        if vol_change_pct != 0:
            builder.vol_trend(
                daily_change=vol_change_pct,
                stress_type=StressType.PERCENTAGE,
                underlying=underlying,
            )
        
        builder.metadata("pattern", "consecutive_decline")
        builder.metadata("severity", "bearish")
        
        return builder.build()
    
    @staticmethod
    def v_shaped_recovery(
        down_days: int = 3,
        up_days: int = 4,
        magnitude: float = 0.15,
        vol_spike: float = 0.50,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a V-shaped recovery scenario.
        
        Sharp decline followed by recovery. Price drops over down_days,
        then recovers over up_days.
        
        Args:
            down_days: Days of decline
            up_days: Days of recovery
            magnitude: Total decline/recovery magnitude (default: 15%)
            vol_spike: Initial vol spike (default: +50%)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for V-shaped recovery
        """
        total_days = down_days + up_days
        daily_down = -magnitude / down_days
        daily_up = magnitude / up_days
        
        builder = PathBuilder(
            num_days=total_days,
            name="V-Shaped Recovery",
            description=f"{magnitude:.0%} decline over {down_days} days, then recovery over {up_days} days",
            start_date=start_date,
        )
        
        # Decline phase
        builder.spot_trend(
            daily_change=daily_down,
            stress_type=StressType.PERCENTAGE,
            underlying=underlying,
            start_day=0,
            end_day=down_days - 1,
        )
        
        # Recovery phase
        builder.spot_trend(
            daily_change=daily_up,
            stress_type=StressType.PERCENTAGE,
            underlying=underlying,
            start_day=down_days,
            end_day=total_days - 1,
        )
        
        # Volatility: spike on first day, then decay
        if vol_spike > 0:
            vol_decay_per_day = vol_spike / (total_days - 1)
            builder.set_day_change(0, "volatility", vol_spike, StressType.PERCENTAGE, underlying)
            for day in range(1, total_days):
                builder.set_day_change(day, "volatility", -vol_decay_per_day, StressType.PERCENTAGE, underlying)
        
        # Labels
        for day in range(down_days):
            builder.set_day_label(day, f"Decline Day {day + 1}")
        for day in range(down_days, total_days):
            builder.set_day_label(day, f"Recovery Day {day - down_days + 1}")
        
        builder.metadata("pattern", "v_shaped_recovery")
        builder.metadata("severity", "high")
        
        return builder.build()
    
    @staticmethod
    def volatility_spike_decay(
        spike_pct: float = 1.0,
        decay_days: int = 5,
        spot_shock: float = -0.05,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a volatility spike with gradual decay scenario.
        
        Initial volatility spike with gradual mean reversion over decay_days.
        
        Args:
            spike_pct: Initial vol spike percentage (default: +100%)
            decay_days: Days for vol to decay (default: 5)
            spot_shock: Initial spot shock (default: -5%)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for vol spike decay scenario
        """
        builder = PathBuilder(
            num_days=decay_days,
            name="Vol Spike Decay",
            description=f"{spike_pct:.0%} vol spike with {decay_days}-day decay",
            start_date=start_date,
        )
        
        # Day 0: Initial spike
        builder.set_day_change(0, "volatility", spike_pct, StressType.PERCENTAGE, underlying)
        if spot_shock != 0:
            builder.set_day_change(0, "spot", spot_shock, StressType.PERCENTAGE, underlying)
        builder.set_day_label(0, "Vol Spike")
        
        # Subsequent days: exponential decay
        remaining_spike = spike_pct
        for day in range(1, decay_days):
            # Decay half of remaining spike each day
            decay = -remaining_spike * 0.3
            builder.set_day_change(day, "volatility", decay, StressType.PERCENTAGE, underlying)
            remaining_spike += decay
            builder.set_day_label(day, f"Decay Day {day}")
        
        builder.metadata("pattern", "vol_spike_decay")
        builder.metadata("severity", "medium")
        
        return builder.build()
    
    @staticmethod
    def mean_reversion(
        days: int = 10,
        amplitude: float = 0.05,
        period: int = 4,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a mean-reverting price path.
        
        Price oscillates around baseline with given amplitude and period.
        
        Args:
            days: Total number of days
            amplitude: Oscillation amplitude (default: 5%)
            period: Days per cycle (default: 4)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for mean reversion scenario
        """
        import math
        
        builder = PathBuilder(
            num_days=days,
            name="Mean Reversion",
            description=f"Price oscillates with {amplitude:.0%} amplitude, {period}-day period",
            start_date=start_date,
        )
        
        # Generate sinusoidal pattern
        for day in range(days):
            # Sinusoidal oscillation
            angle = 2 * math.pi * day / period
            change = amplitude * math.sin(angle)
            builder.set_day_change(day, "spot", change, StressType.PERCENTAGE, underlying)
        
        builder.metadata("pattern", "mean_reversion")
        builder.metadata("severity", "low")
        
        return builder.build()
    
    @staticmethod
    def gradual_crash(
        days: int = 10,
        total_decline: float = -0.30,
        vol_spike: float = 1.0,
        rate_change: float = -0.01,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a gradual market crash scenario.
        
        Steady decline over multiple days with vol spike and rate cut.
        
        Args:
            days: Number of days for crash
            total_decline: Total price decline (default: -30%)
            vol_spike: Total vol increase (default: +100%)
            rate_change: Total rate change (default: -1%)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for gradual crash scenario
        """
        daily_decline = total_decline / days
        daily_vol_increase = vol_spike / days
        daily_rate_change = rate_change / days
        
        builder = PathBuilder(
            num_days=days,
            name="Gradual Crash",
            description=f"Gradual {total_decline:.0%} decline over {days} days",
            start_date=start_date,
        )
        
        builder.spot_trend(
            daily_change=daily_decline,
            stress_type=StressType.PERCENTAGE,
            underlying=underlying,
        )
        
        builder.vol_trend(
            daily_change=daily_vol_increase,
            stress_type=StressType.PERCENTAGE,
            underlying=underlying,
        )
        
        if rate_change != 0:
            builder.rate_trend(
                daily_change=daily_rate_change,
                stress_type=StressType.ABSOLUTE,
                underlying=underlying,
            )
        
        builder.metadata("pattern", "gradual_crash")
        builder.metadata("severity", "extreme")
        
        return builder.build()
    
    @staticmethod
    def rate_hike_cycle(
        days: int = 5,
        total_hike_bps: float = 100,
        spot_impact_pct: float = -0.05,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a rate hike cycle scenario.
        
        Gradual rate increases with associated equity pressure.
        
        Args:
            days: Number of days
            total_hike_bps: Total rate increase in basis points (default: 100bps)
            spot_impact_pct: Associated spot decline (default: -5%)
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath for rate hike scenario
        """
        daily_hike = (total_hike_bps / 10000) / days
        daily_spot_change = spot_impact_pct / days
        
        builder = PathBuilder(
            num_days=days,
            name="Rate Hike Cycle",
            description=f"{total_hike_bps:.0f}bps rate hike over {days} days",
            start_date=start_date,
        )
        
        builder.rate_trend(
            daily_change=daily_hike,
            stress_type=StressType.ABSOLUTE,
            underlying=underlying,
        )
        
        if spot_impact_pct != 0:
            builder.spot_trend(
                daily_change=daily_spot_change,
                stress_type=StressType.PERCENTAGE,
                underlying=underlying,
            )
        
        builder.metadata("pattern", "rate_hike_cycle")
        builder.metadata("severity", "medium")
        
        return builder.build()
    
    @staticmethod
    def custom_path(
        spot_values: Optional[List[float]] = None,
        vol_values: Optional[List[float]] = None,
        rate_values: Optional[List[float]] = None,
        div_yield_values: Optional[List[float]] = None,
        name: str = "Custom Path",
        description: Optional[str] = None,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a fully custom path with explicit values.
        
        Uses VALUE stress type to set exact parameter values for each day.
        
        Args:
            spot_values: List of spot values per day
            vol_values: List of volatility values per day
            rate_values: List of rate values per day
            div_yield_values: List of dividend yield values per day
            name: Path name
            description: Path description
            underlying: Optional specific underlying
            start_date: Optional start date
            
        Returns:
            DayPath with custom values
            
        Raises:
            ValidationError: If no values provided or lengths mismatch
        """
        from quantark.util.exceptions import ValidationError
        
        # Determine num_days from first non-None list
        all_values = [spot_values, vol_values, rate_values, div_yield_values]
        provided = [v for v in all_values if v is not None]
        
        if not provided:
            raise ValidationError("At least one value list must be provided")
        
        num_days = len(provided[0])
        
        # Validate all provided lists have same length
        for v in provided:
            if len(v) != num_days:
                raise ValidationError(
                    f"All value lists must have same length. Expected {num_days}, got {len(v)}"
                )
        
        builder = PathBuilder(
            num_days=num_days,
            name=name,
            description=description,
            start_date=start_date,
        )
        
        if spot_values:
            builder.spot_values(spot_values, underlying)
        
        if vol_values:
            builder.vol_values(vol_values, underlying)
        
        if rate_values:
            builder.rate_values(rate_values, underlying)
        
        if div_yield_values:
            builder.div_yield_values(div_yield_values, underlying)
        
        builder.metadata("pattern", "custom")
        
        return builder.build()
    
    @staticmethod
    def historical_black_monday(
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a Black Monday (1987) style scenario.
        
        Single day massive crash followed by volatile recovery attempt.
        
        Args:
            start_date: Optional start date
            
        Returns:
            DayPath for Black Monday scenario
        """
        builder = PathBuilder(
            num_days=5,
            name="Black Monday Style",
            description="22% crash followed by volatile recovery",
            start_date=start_date,
        )
        
        # Day 0: The crash (-22%)
        builder.set_day_change(0, "spot", -0.22, StressType.PERCENTAGE)
        builder.set_day_change(0, "volatility", 1.50, StressType.PERCENTAGE)
        builder.set_day_label(0, "Black Monday")
        
        # Day 1: Dead cat bounce
        builder.set_day_change(1, "spot", 0.05, StressType.PERCENTAGE)
        builder.set_day_change(1, "volatility", -0.10, StressType.PERCENTAGE)
        builder.set_day_label(1, "Dead Cat Bounce")
        
        # Day 2: Another decline
        builder.set_day_change(2, "spot", -0.08, StressType.PERCENTAGE)
        builder.set_day_change(2, "volatility", 0.20, StressType.PERCENTAGE)
        builder.set_day_label(2, "Continued Selling")
        
        # Day 3-4: Stabilization
        builder.set_day_change(3, "spot", 0.02, StressType.PERCENTAGE)
        builder.set_day_change(3, "volatility", -0.15, StressType.PERCENTAGE)
        builder.set_day_label(3, "Stabilization Begins")
        
        builder.set_day_change(4, "spot", 0.03, StressType.PERCENTAGE)
        builder.set_day_change(4, "volatility", -0.20, StressType.PERCENTAGE)
        builder.set_day_label(4, "Recovery Attempt")
        
        builder.metadata("pattern", "historical_black_monday")
        builder.metadata("severity", "extreme")
        builder.metadata("historical_date", "1987-10-19")
        
        return builder.build()
    
    @staticmethod
    def historical_covid_crash(
        start_date: Optional[datetime] = None
    ) -> DayPath:
        """
        Create a COVID-19 crash (March 2020) style scenario.
        
        Multi-week rapid decline with extreme volatility.
        
        Args:
            start_date: Optional start date
            
        Returns:
            DayPath for COVID crash scenario
        """
        # Simplified 10-day version of the crash
        builder = PathBuilder(
            num_days=10,
            name="COVID Crash Style",
            description="Rapid 34% decline over ~2 weeks",
            start_date=start_date,
        )
        
        # Sharp decline with high volatility
        daily_declines = [-0.03, -0.05, -0.04, -0.06, -0.03, -0.04, -0.02, -0.04, -0.02, -0.01]
        vol_changes = [0.30, 0.25, 0.15, 0.20, 0.10, 0.05, -0.05, 0.10, -0.10, -0.05]
        
        for day, (spot_chg, vol_chg) in enumerate(zip(daily_declines, vol_changes)):
            builder.set_day_change(day, "spot", spot_chg, StressType.PERCENTAGE)
            builder.set_day_change(day, "volatility", vol_chg, StressType.PERCENTAGE)
        
        # Rate cuts
        builder.set_day_change(4, "rate", -0.0075, StressType.ABSOLUTE)
        builder.set_day_change(8, "rate", -0.0075, StressType.ABSOLUTE)
        
        builder.metadata("pattern", "historical_covid_crash")
        builder.metadata("severity", "extreme")
        builder.metadata("historical_date", "2020-03")
        
        return builder.build()
    
    @staticmethod
    def get_all_predefined() -> List[DayPath]:
        """
        Get all predefined scenarios with default parameters.
        
        Returns:
            List of all predefined day paths
        """
        return [
            PathLibrary.consecutive_rally(),
            PathLibrary.consecutive_decline(),
            PathLibrary.v_shaped_recovery(),
            PathLibrary.volatility_spike_decay(),
            PathLibrary.mean_reversion(),
            PathLibrary.gradual_crash(),
            PathLibrary.rate_hike_cycle(),
            PathLibrary.historical_black_monday(),
            PathLibrary.historical_covid_crash(),
        ]
    
    @staticmethod
    def get_bullish_scenarios() -> List[DayPath]:
        """Get bullish market scenarios."""
        return [
            PathLibrary.consecutive_rally(),
            PathLibrary.consecutive_rally(days=10, daily_pct=0.01),
        ]
    
    @staticmethod
    def get_bearish_scenarios() -> List[DayPath]:
        """Get bearish market scenarios."""
        return [
            PathLibrary.consecutive_decline(),
            PathLibrary.gradual_crash(),
            PathLibrary.historical_black_monday(),
            PathLibrary.historical_covid_crash(),
        ]
    
    @staticmethod
    def get_volatile_scenarios() -> List[DayPath]:
        """Get high volatility scenarios."""
        return [
            PathLibrary.volatility_spike_decay(),
            PathLibrary.v_shaped_recovery(),
            PathLibrary.mean_reversion(),
        ]


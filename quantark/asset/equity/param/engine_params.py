"""
Engine configuration parameters.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from quantark.util.enum.engine_enums import EventProjectionMode, KnockInMonitoringMode
from quantark.util.exceptions import ValidationError


def _infer_target_scale(product: Optional[Any], pricing_env: Optional[Any]) -> float:
    """
    Infer a reasonable scale for relative RQMC target std error.
    """
    if product is not None:
        for attr in (
            "contract_multiplier",
            "notional",
            "notional_amount",
            "notional_value",
        ):
            value = getattr(product, attr, None)
            try:
                if value is not None and float(value) > 0:
                    return float(value)
            except (TypeError, ValueError):
                continue

    if pricing_env is not None:
        spot = getattr(pricing_env, "spot", None)
        try:
            if spot is not None and float(spot) > 0:
                return float(spot)
        except (TypeError, ValueError):
            pass

    return 1.0


@dataclass
class BumpConfig:
    """
    Configuration for factor-specific bump sizes in numerical Greeks calculation.

    Each risk factor can have its own bump size and type (relative or absolute).
    Defaults follow industry conventions from Tolerance constants.

    Attributes:
        spot_bump: Relative bump size for spot delta/gamma (default: 1%).
                   Applied as: S * (1 +/- spot_bump)
        vol_bump: Absolute bump size for vega (default: 1 vol point = 0.01).
                  Applied as: sigma +/- vol_bump
        time_bump_days: Absolute bump in days for theta (default: 1 day).
        time_bump_mode: Date advance mode for theta:
                        "calendar_days" advances by calendar dates;
                        "business_days" advances by valid business days;
                        "auto" uses business days when the pricing environment
                        uses BUSINESS_DAYS and has a calendar, otherwise
                        calendar days.
        rate_bump: Absolute bump size for rho (default: 1bp = 0.0001).
                   Applied as: r +/- rate_bump
        div_bump: Absolute bump size for dividend_rho (default: 1bp = 0.0001).
                  Applied as: q +/- div_bump

    Examples:
        >>> # Use all defaults
        >>> config = BumpConfig()
        >>>
        >>> # Custom spot bump for higher precision
        >>> config = BumpConfig(spot_bump=0.001)  # 0.1% instead of 1%
        >>>
        >>> # Aggressive bumps for illiquid securities
        >>> config = BumpConfig(vol_bump=0.02, rate_bump=0.001)  # 2 vol pts, 10bp
    """

    # Spot bump (relative) - for delta/gamma
    spot_bump: float = 0.01

    # Volatility bump (absolute) - for vega
    vol_bump: float = 0.01

    # Time bump (absolute in days) - for theta
    time_bump_days: int = 1
    time_bump_mode: str = "auto"

    # Rate bump (absolute) - for rho
    rate_bump: float = 0.0001

    # Dividend bump (absolute) - for dividend_rho
    div_bump: float = 0.0001

    def __post_init__(self):
        """Validate bump sizes."""
        # Spot bump: must be positive and reasonable (0.01% to 10%)
        if self.spot_bump <= 0:
            raise ValidationError(f"spot_bump must be positive, got {self.spot_bump}")
        if self.spot_bump > 0.1:  # 10% seems too large
            raise ValidationError(
                f"spot_bump seems too large: {self.spot_bump} (max 10%)"
            )

        # Vol bump: must be positive and reasonable
        if self.vol_bump <= 0:
            raise ValidationError(f"vol_bump must be positive, got {self.vol_bump}")
        if self.vol_bump > 0.1:  # 10 vol points seems too large
            raise ValidationError(
                f"vol_bump seems too large: {self.vol_bump} (max 10 vol points)"
            )

        # Time bump: must be positive
        if self.time_bump_days <= 0:
            raise ValidationError(
                f"time_bump_days must be positive, got {self.time_bump_days}"
            )
        if self.time_bump_days > 30:  # More than a month seems wrong
            raise ValidationError(
                f"time_bump_days seems too large: {self.time_bump_days} (max 30 days)"
            )
        self.time_bump_mode = self.time_bump_mode.lower()
        if self.time_bump_mode not in {"auto", "calendar_days", "business_days"}:
            raise ValidationError(
                "time_bump_mode must be one of 'auto', 'calendar_days', "
                f"or 'business_days', got {self.time_bump_mode!r}"
            )

        # Rate bump: must be positive
        if self.rate_bump <= 0:
            raise ValidationError(f"rate_bump must be positive, got {self.rate_bump}")
        if self.rate_bump > 0.01:  # 100bp seems too large
            raise ValidationError(
                f"rate_bump seems too large: {self.rate_bump} (max 100bp)"
            )

        # Div bump: must be positive
        if self.div_bump <= 0:
            raise ValidationError(f"div_bump must be positive, got {self.div_bump}")
        if self.div_bump > 0.01:  # 100bp seems too large
            raise ValidationError(
                f"div_bump seems too large: {self.div_bump} (max 100bp)"
            )

    @classmethod
    def from_tolerance(cls) -> "BumpConfig":
        """
        Create BumpConfig from Tolerance constants.

        Returns:
            BumpConfig with values from Tolerance class.
        """
        from quantark.util.numerical.constants import Tolerance

        return cls(
            spot_bump=Tolerance.BUMP_SPOT,
            vol_bump=Tolerance.BUMP_VOL,
            rate_bump=Tolerance.BUMP_RATE,
            div_bump=Tolerance.BUMP_RATE,  # Same as rate bump
        )

    def get_bump_for_factor(self, factor: str) -> float:
        """
        Get bump size for a specific risk factor.

        Args:
            factor: Risk factor name ('spot', 'vol', 'time', 'rate', 'div')

        Returns:
            Bump size for the factor

        Raises:
            ValidationError: If factor is unknown
        """
        factor_map = {
            "spot": self.spot_bump,
            "vol": self.vol_bump,
            "time": float(self.time_bump_days),
            "rate": self.rate_bump,
            "div": self.div_bump,
        }
        if factor not in factor_map:
            raise ValidationError(f"Unknown risk factor: {factor}")
        return factor_map[factor]


@dataclass
class EngineParams:
    """
    Configuration parameters for pricing engines.

    Attributes:
        bump_size: DEPRECATED - Use bump_config instead. Bump size for finite
                   difference method (default: 1e-4). Only used for spot delta/gamma.
        bump_config: Factor-specific bump configuration. If None, creates default
                     from bump_size for backward compatibility.
        bus_days_in_year: Number of business days per year (default: 252)
    """

    bump_size: float = 1e-4
    bump_config: Optional[BumpConfig] = None
    bus_days_in_year: int = 252

    def __post_init__(self):
        """Validate parameters and create bump_config if needed."""
        # Legacy bump_size validation
        if self.bump_size <= 0:
            raise ValidationError(f"Bump size must be positive, got {self.bump_size}")
        if self.bump_size > 0.01:  # 1% seems too large
            raise ValidationError(f"Bump size seems too large: {self.bump_size}")
        if self.bus_days_in_year <= 0:
            raise ValidationError(
                f"Business days must be positive, got {self.bus_days_in_year}"
            )

        # Create bump_config from bump_size for backward compatibility
        if self.bump_config is None:
            self.bump_config = BumpConfig(spot_bump=self.bump_size)

    def get_effective_bump_config(self) -> BumpConfig:
        """
        Get the effective bump configuration.

        Returns bump_config if set, otherwise creates from legacy bump_size.

        Returns:
            Effective BumpConfig
        """
        if self.bump_config is not None:
            return self.bump_config
        # Fallback for safety (should not reach here due to __post_init__)
        return BumpConfig(spot_bump=self.bump_size)


@dataclass
class MCParams(EngineParams):
    """
    Monte Carlo engine configuration.

    Attributes:
        seed: Random seed for reproducibility
        num_paths: Number of simulation paths
        time_steps: Number of time steps per path
        use_business_day_grid: Use daily business-day grid for schedule-based engines
        use_qmc: Use quasi-Monte Carlo (default: False)
        use_antithetic: Use antithetic variates (default: False)
        rqmc_target_std: RQMC target standard error (absolute or relative)
        rqmc_target_std_mode: absolute|relative_notional|relative_price
        rqmc_target_std_floor: Minimum absolute target std error
        rqmc_min_batches: Minimum RQMC batches before stopping
        rqmc_max_batches: Maximum RQMC batches to run
        rqmc_target_std_scale: Optional explicit scale for relative modes
        rqmc_paths_mode: per_batch (default) or total (interprets num_paths as total)
    """

    seed: int = 42
    num_paths: int = 10000
    time_steps: int = 100
    use_business_day_grid: bool = False
    use_qmc: bool = False
    use_antithetic: bool = False
    rqmc_target_std: float = 1e-4
    rqmc_target_std_mode: str = "absolute"
    rqmc_target_std_floor: float = 0.0
    rqmc_min_batches: int = 4
    rqmc_max_batches: int = 32
    rqmc_target_std_scale: Optional[float] = None
    rqmc_paths_mode: str = "per_batch"

    def __post_init__(self):
        """Validate MC parameters."""
        super().__post_init__()
        if self.num_paths <= 0:
            raise ValidationError(
                f"Number of paths must be positive, got {self.num_paths}"
            )
        if self.time_steps <= 0:
            raise ValidationError(f"Time steps must be positive, got {self.time_steps}")
        if self.rqmc_target_std <= 0:
            raise ValidationError(
                f"rqmc_target_std must be positive, got {self.rqmc_target_std}"
            )
        if self.rqmc_target_std_floor < 0:
            raise ValidationError(
                f"rqmc_target_std_floor must be non-negative, got {self.rqmc_target_std_floor}"
            )
        if self.rqmc_target_std_mode not in (
            "absolute",
            "relative_notional",
            "relative_price",
        ):
            raise ValidationError(
                "rqmc_target_std_mode must be one of absolute, relative_notional, "
                "relative_price"
            )
        if self.rqmc_min_batches <= 0:
            raise ValidationError(
                f"rqmc_min_batches must be positive, got {self.rqmc_min_batches}"
            )
        if self.rqmc_max_batches <= 0:
            raise ValidationError(
                f"rqmc_max_batches must be positive, got {self.rqmc_max_batches}"
            )
        if self.rqmc_min_batches > self.rqmc_max_batches:
            raise ValidationError(
                "rqmc_min_batches cannot exceed rqmc_max_batches"
            )
        if self.rqmc_target_std_scale is not None and self.rqmc_target_std_scale <= 0:
            raise ValidationError(
                "rqmc_target_std_scale must be positive when provided"
            )
        if self.rqmc_paths_mode not in ("per_batch", "total"):
            raise ValidationError(
                "rqmc_paths_mode must be one of per_batch, total"
            )

    def resolve_rqmc_target_std(
        self,
        product: Optional[Any] = None,
        pricing_env: Optional[Any] = None,
        scale: Optional[float] = None,
    ) -> float:
        """
        Resolve RQMC target std error based on absolute or relative mode.

        Args:
            product: Optional product for inferring notional scale.
            pricing_env: Optional pricing environment for scale hints.
            scale: Explicit scale override for relative modes.

        Returns:
            Target standard error in price units.
        """
        mode = self.rqmc_target_std_mode
        base = float(self.rqmc_target_std)
        floor = float(self.rqmc_target_std_floor)

        if mode == "absolute":
            return max(base, floor)

        resolved_scale = scale
        if resolved_scale is None:
            resolved_scale = self.rqmc_target_std_scale
        if resolved_scale is None:
            resolved_scale = _infer_target_scale(product, pricing_env)

        target = base * float(resolved_scale)
        return max(target, floor)

    def resolve_rqmc_paths_per_batch(self, max_batches: Optional[int] = None) -> int:
        """
        Resolve per-batch path count for RQMC.

        Args:
            max_batches: Optional override for maximum batches (defaults to rqmc_max_batches).

        Returns:
            Per-batch path count (Sobol-ideal if rqmc_paths_mode=total).
        """
        total_paths = int(self.num_paths)
        if total_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {total_paths}")

        if self.rqmc_paths_mode == "per_batch":
            return total_paths

        batches = int(max_batches if max_batches is not None else self.rqmc_max_batches)
        batches = max(batches, 1)
        per_batch_raw = int(math.ceil(total_paths / float(batches)))
        return _next_power_of_two(per_batch_raw)


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


@dataclass
class PDEParams(EngineParams):
    """
    PDE engine configuration.

    Attributes:
        grid_size: Number of spatial grid points (default: 400)
        time_steps: Number of time steps (default: 200)
        adaptive_grid: Use adaptive grid spacing with Tavella-Randall (default: False)
        cache_enabled: Enable PDE cache usage (default: True)
        cache_strategy: Cache strategy: disable, strict, standard, aggressive (default: standard)
        grid_cache_max_entries: Max cached grids for PDE solvers (default: 128)
        use_banded_solver: Use banded solve for tridiagonal systems (default: True)
        banded_cache_max_entries: Max cached banded systems per solve (default: 512)
        s_min: Lower bound for spatial grid (0 = auto-calculate)
        s_max: Upper bound for spatial grid (0 = auto-calculate)
        auto_grid: Enable feature-aware default grids (default: True)
        time_grid_type: Type of time grid - "uniform", "graded", "event_clustered", or "event_aligned"
        grade_exponent: Exponent for graded time grid (higher = more clustering near maturity)
        bus_days_in_year: Days per year for day-based grid heuristics (from EngineParams, default: 252)
        event_steps_per_day: Steps per day between observation events (default: 4)
        event_min_steps_per_interval: Minimum steps between consecutive events (default: 10)
        max_time_steps: Upper bound for auto-generated time steps (default: 5000)
        log_dx_target: Target log-price spacing near critical points for adaptive grids (default: 0.003)
        max_grid_size: Upper bound for auto-generated spatial grid points (default: 2000)
        include_spot_in_critical_points: Include spot as a critical point when auto_grid is enabled (default: True)
        event_projection: Spatial representation of discrete coupon/KO/KI event
            jumps — "nodal" (legacy Boolean masks, default) or "cell_average"
            (conservative dual-cell projection; removes the half-cell
            trigger-phase bias, see pde_auto_grid_investigation.md)
        rannacher_at_events: Apply implicit (Rannacher-style) damping steps
            after each discrete event whose time lands on a grid node,
            regardless of auto_grid (default: True)
        event_theta: Theta value applied immediately before event times (default: 1.0)
        event_rannacher_steps: Number of event-adjacent steps using event_theta (default: 1)
        barrier_refine_log_width: Log-space refinement half-width around barriers (default: 0.0 = disabled)
        barrier_refine_levels: Number of refinement layers on each side of a barrier (default: 1)
        barrier_domain_expand: Extra relative domain expansion around barriers (default: 0.0 = disabled)
        boundary_mode: Boundary condition mode (asymptotic, default)
        theta: Finite difference scheme parameter (0.5 = Crank-Nicolson, 1.0 = Backward Euler)
        use_rannacher: Apply Rannacher smoothing for first steps (default: True)
        rannacher_steps: Number of backward Euler steps for smoothing (default: 1)
    """

    grid_size: int = 400
    time_steps: int = 200
    adaptive_grid: bool = False
    cache_enabled: bool = True
    cache_strategy: str = "standard"
    grid_cache_max_entries: int = 128
    use_banded_solver: bool = True
    banded_cache_max_entries: int = 512

    # Feature-aware default grids
    auto_grid: bool = True

    # Spatial grid configuration
    s_min: float = 0.0  # Auto-calculate if 0
    s_max: float = 0.0  # Auto-calculate if 0

    # Time grid configuration
    time_grid_type: str = "uniform"  # "uniform", "graded", "event_clustered", "event_aligned"
    grade_exponent: float = 2.0

    # Auto-grid tuning parameters
    # event_steps_per_day: resolution (fill steps per business day) for the
    # decoupled event-aligned grid (TimeGrid.build_mandatory). Validated by
    # test_pde_grid_convergence_gate: at spd=4 the autocallable price is
    # grid-converged for discrete-KI rows (<=1e-4 vs spd=8) and matches the
    # pre-change production grid within 1.2e-5 (oracle b). Not lowered below 4
    # because continuously-monitored KI needs the resolution (slow O(1/N)
    # barrier convergence).
    event_steps_per_day: int = 4
    event_min_steps_per_interval: int = 10
    max_time_steps: int = 5000
    log_dx_target: float = 0.003
    max_grid_size: int = 2000
    include_spot_in_critical_points: bool = True
    # When set (by create_bump_context), the spatial grid concentrates on exactly
    # these frozen critical points instead of recomputing them per bump — so a
    # spot bump does not snap the grid to a moved spot [§11.4].
    frozen_critical_points: Optional[tuple] = None
    # Treatment of discretely-monitored knock-in schedules [§11.6]. EXACT_DISCRETE
    # (default) applies the KI regime jump on every observation date exactly, with
    # an event-aligned grid. BGK_APPROXIMATION is an opt-in performance/robustness
    # mode that replaces a dense daily KI schedule with continuous monitoring at a
    # Broadie-Glasserman-Kou (1997) shifted barrier; it engages only for discretely
    # monitored KI (European / continuous / no-KI / already-knocked-in are inert
    # with a logged note). Same enum, beta, and shift as SnowballQuadEngine so PDE
    # and quad BGK prices are directly comparable.
    ki_monitoring_mode: Union[KnockInMonitoringMode, str] = (
        KnockInMonitoringMode.EXACT_DISCRETE
    )
    # Spatial representation of discrete event operators (coupon/KO/KI jumps)
    # in the 1D BSM autocallable solvers. NODAL (default) preserves the legacy
    # Boolean-mask application; CELL_AVERAGE opts into the conservative
    # dual-cell projection that removes the half-cell trigger-phase bias
    # documented in pde_auto_grid_investigation.md. Continuously monitored
    # barriers keep nodal treatment either way; volatility-model PDE solvers
    # (Heston/SLV) do not consume this setting yet.
    event_projection: Union[EventProjectionMode, str] = EventProjectionMode.NODAL
    rannacher_at_events: bool = True
    event_theta: float = 1.0
    event_rannacher_steps: int = 1
    barrier_refine_log_width: float = 0.0
    barrier_refine_levels: int = 1
    barrier_domain_expand: float = 0.0
    boundary_mode: str = "asymptotic"

    # Numerical scheme configuration
    theta: float = 0.5  # 0.5 = Crank-Nicolson, 1.0 = Backward Euler
    use_rannacher: bool = True
    rannacher_steps: int = 1

    def __post_init__(self):
        """Validate PDE parameters."""
        super().__post_init__()
        if self.grid_size <= 0:
            raise ValidationError(f"Grid size must be positive, got {self.grid_size}")
        if self.time_steps <= 0:
            raise ValidationError(f"Time steps must be positive, got {self.time_steps}")
        if self.grid_cache_max_entries <= 0:
            raise ValidationError(
                f"grid_cache_max_entries must be positive, got {self.grid_cache_max_entries}"
            )
        if self.banded_cache_max_entries <= 0:
            raise ValidationError(
                "banded_cache_max_entries must be positive, got "
                f"{self.banded_cache_max_entries}"
            )
        if self.cache_strategy not in ("disable", "strict", "standard", "aggressive"):
            raise ValidationError(
                "cache_strategy must be one of disable, strict, standard, aggressive, got "
                f"{self.cache_strategy}"
            )
        if self.s_min < 0:
            raise ValidationError(f"s_min must be non-negative, got {self.s_min}")
        if self.s_max < 0:
            raise ValidationError(f"s_max must be non-negative, got {self.s_max}")
        if self.s_min > 0 and self.s_max > 0 and self.s_min >= self.s_max:
            raise ValidationError(
                f"s_min ({self.s_min}) must be less than s_max ({self.s_max})"
            )
        if self.time_grid_type not in ("uniform", "graded", "event_clustered", "event_aligned"):
            raise ValidationError(
                f"time_grid_type must be 'uniform', 'graded', 'event_clustered', or 'event_aligned', "
                f"got '{self.time_grid_type}'"
            )
        if self.grade_exponent <= 0:
            raise ValidationError(
                f"grade_exponent must be positive, got {self.grade_exponent}"
            )
        if self.event_steps_per_day <= 0:
            raise ValidationError(
                f"event_steps_per_day must be positive, got {self.event_steps_per_day}"
            )
        if self.event_min_steps_per_interval <= 0:
            raise ValidationError(
                f"event_min_steps_per_interval must be positive, got {self.event_min_steps_per_interval}"
            )
        if self.max_time_steps <= 0:
            raise ValidationError(
                f"max_time_steps must be positive, got {self.max_time_steps}"
            )
        if self.log_dx_target <= 0:
            raise ValidationError(
                f"log_dx_target must be positive, got {self.log_dx_target}"
            )
        if self.max_grid_size <= 0:
            raise ValidationError(
                f"max_grid_size must be positive, got {self.max_grid_size}"
            )
        if not 0.0 <= self.theta <= 1.0:
            raise ValidationError(f"theta must be in [0, 1], got {self.theta}")
        if self.rannacher_steps < 0:
            raise ValidationError(
                f"rannacher_steps must be non-negative, got {self.rannacher_steps}"
            )
        if not 0.0 < self.event_theta <= 1.0:
            raise ValidationError(
                f"event_theta must be in (0, 1], got {self.event_theta}"
            )
        if self.event_rannacher_steps < 0:
            raise ValidationError(
                f"event_rannacher_steps must be non-negative, got {self.event_rannacher_steps}"
            )
        if self.barrier_refine_log_width < 0.0:
            raise ValidationError(
                f"barrier_refine_log_width must be non-negative, got {self.barrier_refine_log_width}"
            )
        if self.barrier_refine_levels < 0:
            raise ValidationError(
                f"barrier_refine_levels must be non-negative, got {self.barrier_refine_levels}"
            )
        if self.barrier_domain_expand < 0.0:
            raise ValidationError(
                f"barrier_domain_expand must be non-negative, got {self.barrier_domain_expand}"
            )
        if self.boundary_mode not in ("default", "asymptotic"):
            raise ValidationError(
                f"boundary_mode must be one of default, asymptotic, got {self.boundary_mode}"
            )
        # Coerce ki_monitoring_mode to the enum (same contract as QuadParams).
        if isinstance(self.ki_monitoring_mode, str):
            try:
                self.ki_monitoring_mode = KnockInMonitoringMode(
                    self.ki_monitoring_mode.lower()
                )
            except ValueError:
                raise ValidationError(
                    "ki_monitoring_mode must be one of "
                    f"{[mode.value for mode in KnockInMonitoringMode]}, "
                    f"got {self.ki_monitoring_mode!r}"
                )
        elif not isinstance(self.ki_monitoring_mode, KnockInMonitoringMode):
            raise ValidationError(
                "ki_monitoring_mode must be a KnockInMonitoringMode or its "
                f"string value, got {type(self.ki_monitoring_mode).__name__}"
            )
        # Coerce event_projection to the enum (same contract as above).
        if isinstance(self.event_projection, str):
            try:
                self.event_projection = EventProjectionMode(
                    self.event_projection.lower()
                )
            except ValueError:
                raise ValidationError(
                    "event_projection must be one of "
                    f"{[mode.value for mode in EventProjectionMode]}, "
                    f"got {self.event_projection!r}"
                )
        elif not isinstance(self.event_projection, EventProjectionMode):
            raise ValidationError(
                "event_projection must be an EventProjectionMode or its "
                f"string value, got {type(self.event_projection).__name__}"
            )

    @classmethod
    def from_config(
        cls,
        config_or_path: Any,
        profile: Optional[str] = None,
        product: Optional[Any] = None,
        reverse: Optional[bool] = None,
        **overrides: Any,
    ) -> "PDEParams":
        """
        Create PDEParams from a dict/YAML/JSON config and optional overrides.

        Args:
            config_or_path: Mapping or path to YAML/JSON file.
            profile: Optional preset profile to use.
            product: Optional product object for light auto-tuning.
            reverse: Optional reverse flag (overrides config if provided).
            **overrides: Explicit parameter overrides.

        Returns:
            PDEParams instance.
        """
        from .engine_param_profiles import make_pde_params, parse_profile_config

        resolved_profile, config_overrides, reverse_flag = parse_profile_config(
            config_or_path, engine="pde", profile=profile, reverse=reverse
        )
        if overrides:
            config_overrides.update(overrides)
        return make_pde_params(
            profile=resolved_profile,
            product=product,
            reverse=reverse_flag,
            **config_overrides,
        )


@dataclass
class QuadParams(EngineParams):
    """
    Quadrature engine configuration.

    Attributes:
        grid_points: Number of integration points (default: 1000).
                     Must be odd for Simpson's rule.
        num_std_devs: Number of standard deviations for integration bounds (default: 10).
                      Larger values capture more of the distribution tail.
        fft_padding_factor: Zero-padding factor for FFT-based convolution (default: 2).
        fft_filter_alpha: Spectral filter strength (0 disables filtering).
        fft_filter_power: Spectral filter power/exponent (higher = sharper cutoff).
        stability_preset: Optional stability preset overriding FFT padding/filter defaults.
                          Supported: conservative, balanced, aggressive.
        align_priority: Barrier alignment priority (auto, ko, coupon, ki).
        event_smoothing_cells: Half-width of smoothing window in grid cells (0 disables).
        event_smoothing_mode: Smoothing mode (fixed, auto, reverse_aware).
        event_smoothing_kernel: Smoothing kernel (cosine, tanh).
        event_smoothing_log_width: Target log-space width for auto smoothing.
        min_diffusion_stddev_cells: Minimum spatial-grid cells per discrete-KI
            interval diffusion standard deviation. The engine increases its
            internal odd grid size when dense KI intervals would otherwise be
            under-resolved. The default 2.5 is accuracy-oriented (it removes
            the material deep-carry quote error measured at 1.25);
            performance-sensitive workloads such as large Greek/scenario
            reports may explicitly lower it to 1.25 or opt into
            BGK_APPROXIMATION. Set to 0 to disable adaptive refinement.
        max_adaptive_grid_points: Maximum grid size selected by adaptive
            refinement. Explicitly requested larger grids are still honored.
        ki_monitoring_mode: Treatment of discrete KI schedules
            (KnockInMonitoringMode or its string value, case-insensitive).
            EXACT_DISCRETE (default) prices every KI observation date exactly
            with adaptive grid refinement. BGK_APPROXIMATION is an opt-in
            performance mode that replaces the discrete schedule with
            continuous monitoring at a Broadie-Glasserman-Kou shifted
            barrier; the engine validates regular spacing, constant resolved
            barrier, full-horizon coverage, and stable volatility, raising
            ValidationError when the schedule does not qualify. Ignored for
            products without a discrete KI schedule.
        bgk_min_ki_observations: Minimum number of KI observation dates for
            BGK_APPROXIMATION; sparser schedules are rejected because the
            shift's first-order residual grows with observation spacing and
            the performance motivation only exists for dense schedules.
    """

    grid_points: int = 1001  # Odd number for Simpson's rule
    num_std_devs: float = 10.0
    fft_padding_factor: int = 2
    fft_filter_alpha: float = 12.0
    fft_filter_power: int = 8
    stability_preset: Optional[str] = None
    align_priority: str = "auto"
    event_smoothing_cells: int = 1
    event_smoothing_mode: str = "fixed"
    event_smoothing_kernel: str = "cosine"
    event_smoothing_log_width: float = 0.002
    min_diffusion_stddev_cells: float = 2.5
    max_adaptive_grid_points: int = 5001
    ki_monitoring_mode: Union[KnockInMonitoringMode, str] = (
        KnockInMonitoringMode.EXACT_DISCRETE
    )
    bgk_min_ki_observations: int = 100

    def __post_init__(self):
        """Validate quadrature parameters."""
        super().__post_init__()
        if self.stability_preset is not None:
            preset = str(self.stability_preset).lower()
            if preset == "conservative":
                self.fft_padding_factor = 2
                self.fft_filter_alpha = 12.0
                self.fft_filter_power = 8
            elif preset == "balanced":
                self.fft_padding_factor = 2
                self.fft_filter_alpha = 18.0
                self.fft_filter_power = 8
            elif preset == "aggressive":
                self.fft_padding_factor = 4
                self.fft_filter_alpha = 24.0
                self.fft_filter_power = 8
            else:
                raise ValidationError(
                    "stability_preset must be one of conservative, balanced, aggressive"
                )
        if self.grid_points <= 0:
            raise ValidationError(
                f"grid_points must be positive, got {self.grid_points}"
            )
        if self.grid_points < 100:
            raise ValidationError(
                f"grid_points should be at least 100 for accuracy, got {self.grid_points}"
            )
        # Ensure odd number for Simpson's rule
        if self.grid_points % 2 == 0:
            self.grid_points += 1
        if self.num_std_devs <= 0:
            raise ValidationError(
                f"num_std_devs must be positive, got {self.num_std_devs}"
            )
        if self.num_std_devs < 3:
            raise ValidationError(
                f"num_std_devs should be at least 3 for accuracy, got {self.num_std_devs}"
            )
        if self.fft_padding_factor < 1:
            raise ValidationError(
                f"fft_padding_factor must be >= 1, got {self.fft_padding_factor}"
            )
        if self.fft_filter_alpha < 0.0:
            raise ValidationError(
                f"fft_filter_alpha must be non-negative, got {self.fft_filter_alpha}"
            )
        if self.fft_filter_power < 1:
            raise ValidationError(
                f"fft_filter_power must be >= 1, got {self.fft_filter_power}"
            )
        if self.align_priority not in ("auto", "ko", "coupon", "ki"):
            raise ValidationError(
                f"align_priority must be one of auto, ko, coupon, ki, got {self.align_priority}"
            )
        if self.event_smoothing_cells < 0:
            raise ValidationError(
                f"event_smoothing_cells must be >= 0, got {self.event_smoothing_cells}"
            )
        if self.event_smoothing_mode not in ("fixed", "auto", "reverse_aware"):
            raise ValidationError(
                "event_smoothing_mode must be one of fixed, auto, reverse_aware, got "
                f"{self.event_smoothing_mode}"
            )
        if self.event_smoothing_kernel not in ("cosine", "tanh"):
            raise ValidationError(
                "event_smoothing_kernel must be one of cosine, tanh, got "
                f"{self.event_smoothing_kernel}"
            )
        if self.event_smoothing_log_width <= 0.0:
            raise ValidationError(
                f"event_smoothing_log_width must be positive, got {self.event_smoothing_log_width}"
            )
        if self.min_diffusion_stddev_cells < 0:
            raise ValidationError(
                "min_diffusion_stddev_cells must be non-negative, "
                f"got {self.min_diffusion_stddev_cells}"
            )
        if self.max_adaptive_grid_points < 101:
            raise ValidationError(
                "max_adaptive_grid_points must be at least 101, "
                f"got {self.max_adaptive_grid_points}"
            )
        if isinstance(self.ki_monitoring_mode, str):
            try:
                self.ki_monitoring_mode = KnockInMonitoringMode(
                    self.ki_monitoring_mode.lower()
                )
            except ValueError:
                raise ValidationError(
                    "ki_monitoring_mode must be one of "
                    f"{[mode.value for mode in KnockInMonitoringMode]}, "
                    f"got {self.ki_monitoring_mode!r}"
                )
        elif not isinstance(self.ki_monitoring_mode, KnockInMonitoringMode):
            raise ValidationError(
                "ki_monitoring_mode must be a KnockInMonitoringMode or its "
                f"string value, got {type(self.ki_monitoring_mode).__name__}"
            )
        if self.bgk_min_ki_observations < 2:
            raise ValidationError(
                "bgk_min_ki_observations must be at least 2, "
                f"got {self.bgk_min_ki_observations}"
            )

    @classmethod
    def from_config(
        cls,
        config_or_path: Any,
        profile: Optional[str] = None,
        product: Optional[Any] = None,
        reverse: Optional[bool] = None,
        **overrides: Any,
    ) -> "QuadParams":
        """
        Create QuadParams from a dict/YAML/JSON config and optional overrides.

        Args:
            config_or_path: Mapping or path to YAML/JSON file.
            profile: Optional preset profile to use.
            product: Optional product object for light auto-tuning.
            reverse: Optional reverse flag (overrides config if provided).
            **overrides: Explicit parameter overrides.

        Returns:
            QuadParams instance.
        """
        from .engine_param_profiles import make_quad_params, parse_profile_config

        resolved_profile, config_overrides, reverse_flag = parse_profile_config(
            config_or_path, engine="quad", profile=profile, reverse=reverse
        )
        if overrides:
            config_overrides.update(overrides)
        return make_quad_params(
            profile=resolved_profile,
            product=product,
            reverse=reverse_flag,
            **config_overrides,
        )

"""
Engine configuration parameters.
"""

import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Union
from quantark.util.enum.engine_enums import EventProjectionMode, KnockInMonitoringMode
from quantark.util.exceptions import ValidationError

from .pde_removed_names import REMOVED_PDE_NAMES, keyword_only_init


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

    # Optional separate relative bump for the gamma second difference;
    # None means gamma reuses spot_bump (the default and common case).
    gamma_spot_bump: Optional[float] = None

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

        if self.gamma_spot_bump is not None:
            if self.gamma_spot_bump <= 0:
                raise ValidationError(
                    f"gamma_spot_bump must be positive, got {self.gamma_spot_bump}"
                )
            if self.gamma_spot_bump > 0.1:
                raise ValidationError(
                    f"gamma_spot_bump seems too large: {self.gamma_spot_bump} (max 10%)"
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
        bump_size: DEPRECATED - Use bump_config instead. When explicitly set,
                   it seeds ``BumpConfig(spot_bump=bump_size)`` and emits a
                   DeprecationWarning. Left unset (None), bump sizes come from
                   ``BumpConfig``'s documented defaults (1% spot bump).
        bump_config: Factor-specific bump configuration. If None, defaults to
                     ``BumpConfig()`` (or the legacy bump_size when that is set).
        bus_days_in_year: Number of business days per year (default: 252)
    """

    bump_size: Optional[float] = None
    bump_config: Optional[BumpConfig] = None
    bus_days_in_year: int = 252

    def __post_init__(self):
        """Validate parameters and create bump_config if needed."""
        if self.bump_size is not None:
            # Legacy bump_size validation (historic bounds kept as-is)
            if self.bump_size <= 0:
                raise ValidationError(
                    f"Bump size must be positive, got {self.bump_size}"
                )
            if self.bump_size > 0.01:  # 1% seems too large
                raise ValidationError(
                    f"Bump size seems too large: {self.bump_size}"
                )
            import warnings

            warnings.warn(
                "EngineParams.bump_size is deprecated; pass "
                "bump_config=BumpConfig(spot_bump=...) instead",
                DeprecationWarning,
                stacklevel=3,
            )
        if self.bus_days_in_year <= 0:
            raise ValidationError(
                f"Business days must be positive, got {self.bus_days_in_year}"
            )

        if self.bump_config is None:
            self.bump_config = (
                BumpConfig()
                if self.bump_size is None
                else BumpConfig(spot_bump=self.bump_size)
            )

    def get_effective_bump_config(self) -> BumpConfig:
        """
        Get the effective bump configuration.

        Returns bump_config if set, otherwise the documented defaults (with the
        legacy bump_size honored when explicitly provided).

        Returns:
            Effective BumpConfig
        """
        if self.bump_config is not None:
            return self.bump_config
        # Fallback for safety (should not reach here due to __post_init__)
        if self.bump_size is not None:
            return BumpConfig(spot_bump=self.bump_size)
        return BumpConfig()


@dataclass
class MCParams(EngineParams):
    """
    Monte Carlo engine configuration.

    Attributes:
        seed: Random seed for reproducibility
        num_paths: Number of simulation paths
        time_steps: Number of time steps per path. Schedule-based engines
                    (snowball/phoenix families) IGNORE this: their grid is the
                    observation schedule (daily only under continuous KI). To
                    refine SDE stepping between observations there, use the
                    engine-level ``substeps_per_interval`` parameter.
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


@keyword_only_init(REMOVED_PDE_NAMES)
@dataclass(kw_only=True)
class PDEParams(EngineParams):
    """1D PDE configuration: the one surface every 1D PDE route reads (0.6.0).

    Keyword arguments only; a removed 0.5 name raises ``ValidationError``
    naming its replacement (``pde_removed_names.REMOVED_PDE_NAMES``).

    Field contract (when to change it / what happens):

    - ``accuracy`` -- a faster preview or a finer answer. Selects the
      ``GridConfig`` preset: fast (200 pts, 4/day), standard (400, 4/day),
      high (800, 8/day). Resolution settings, not accuracy guarantees; a
      "day" is 1/252 of a solver year in this release.
    - ``grid`` -- override one preset field. Each non-``None`` ``GridConfig``
      field replaces that preset value; the rest inherit. Exactly one time
      basis (``steps_per_day`` or ``time_steps``) is in force after
      resolution; setting one clears the other.
    - ``stepper`` -- the time stepper. ``None`` is the built-in
      ``CrankNicolson()``; its four settings are the scheme's: ``theta`` (0.5
      Crank-Nicolson, 1.0 backward Euler; change it for ringing damping does
      not cure), ``event_theta`` (the weight on the damped steps after each
      interior event), ``terminal_damping_steps`` and ``event_damping_steps``
      (``None`` = the route's defaults, 0 disables). A custom stepper runs on
      the shared-grid routes only.
    - ``operator`` -- the spatial operator. ``None`` is the built-in
      ``CentralDifference()``; a custom operator must support the route's
      samples. Constructed in code only (not from a config file).
    - ``readout`` -- how price and delta/gamma are read off the solution.
      ``None`` is the built-in ``StandardReadout()``. Constructed in code only.
    - ``event_projection`` -- validating the event treatment. ``None`` is the
      route's own treatment (CELL_AVERAGE on the shared-grid autocallable
      solvers); explicit CELL_AVERAGE splits the trigger cell by its exact
      overlap, explicit NODAL assigns whole nodes. Lower-case strings coerce.
    - ``cache_enabled`` -- memory pressure, or a fresh solve every call. Same
      numbers to round-off: a cached run shares one factorization across
      intervals whose dt differ by an ULP (Review Focus 5).
    - ``bump_config`` (inherited) -- bump sizes for numerical Greeks.

    Routes other than the shared-grid solvers (local-vol vanilla and barrier,
    DCN) honour a subset; ``route_applicability.ROUTE_RULES`` lists what each
    rejects, and the error names the field, the route, and sub-project 3.
    """

    # Grid: ``accuracy`` selects a GridConfig preset; ``grid`` is the
    # field-by-field overlay (typed loosely: param must not import engine.pde
    # at module load; validated in __post_init__).
    accuracy: str = "standard"
    grid: Optional[object] = None
    # Components (spec 2 §4.1). None = the built-in, normalised in __post_init__
    # (param must not import engine.pde at module load: typed loosely here).
    stepper: Optional[object] = None
    operator: Optional[object] = None
    readout: Optional[object] = None
    # Discrete event operators (coupon/KO/KI jumps). None = the route's own
    # treatment (decision 15): CELL_AVERAGE on the shared-grid autocallable
    # solvers (1D BSM/LV and 2D Heston/SLV slice-wise; the dual-cell
    # projection that removes the half-cell trigger-phase bias documented in
    # pde_auto_grid_investigation.md), NODAL on DCN, a mid-cell-aligned nodal
    # mask on the local-vol barrier. Continuously monitored barriers keep
    # nodal treatment either way.
    event_projection: Optional[Union[EventProjectionMode, str]] = None
    cache_enabled: bool = True

    def __post_init__(self):
        """Validate PDE parameters."""
        super().__post_init__()
        # Declarative grid layer (lazy import: param must not pull engine.pde
        # at module load — the solvers import param back).
        from quantark.asset.equity.engine.pde.grid.config import (
            ACCURACY_PROFILES,
            GridConfig,
        )

        if self.accuracy not in ACCURACY_PROFILES:
            raise ValidationError(
                f"accuracy must be one of {sorted(ACCURACY_PROFILES)}, "
                f"got {self.accuracy!r}"
            )
        if self.grid is not None and not isinstance(self.grid, GridConfig):
            raise ValidationError(
                f"grid must be a GridConfig, got {type(self.grid).__name__}"
            )

        from quantark.asset.equity.engine.pde.core.components import (
            CentralDifference,
            CrankNicolson,
            StandardReadout,
        )

        for name, builtin, methods in (
            ("stepper", CrankNicolson, ("schedule", "step", "evidence")),
            ("operator", CentralDifference, ("assemble", "evidence", "supports")),
            ("readout", StandardReadout, ("price", "delta_gamma", "evidence")),
        ):
            if getattr(self, name) is None:
                setattr(self, name, builtin())
            missing = [m for m in methods if not hasattr(getattr(self, name), m)]
            if missing:
                raise ValidationError(
                    f"PDEParams.{name} must provide {', '.join(methods)}; "
                    f"{type(getattr(self, name)).__name__} lacks {', '.join(missing)}"
                )
        # Coerce event_projection to the enum; None is the route's treatment.
        if self.event_projection is None:
            pass
        elif isinstance(self.event_projection, str):
            try:
                self.event_projection = EventProjectionMode(
                    self.event_projection.lower()
                )
            except ValueError:
                raise ValidationError(
                    "event_projection must be None or one of "
                    f"{[mode.value for mode in EventProjectionMode]}, "
                    f"got {self.event_projection!r}"
                )
        elif not isinstance(self.event_projection, EventProjectionMode):
            raise ValidationError(
                "event_projection must be None, an EventProjectionMode or its "
                f"string value, got {type(self.event_projection).__name__}"
            )

    def evidence(self) -> dict:
        """What this params object runs with, as certificate data (spec 2 §4.6).

        The built-in stepper by its settings; any other stepper, and a non-built-in
        operator or readout, by its value fingerprint (``"uncertifiable"`` when it
        has none). ``engine_config`` serializes through this.
        """
        from quantark.asset.equity.engine.pde.core.components import (
            CentralDifference,
            CrankNicolson,
            StandardReadout,
        )
        from quantark.asset.equity.engine.pde.core.identity import component_fingerprint

        out: dict = {"accuracy": self.accuracy}
        if self.grid is not None:
            out["grid"] = dict(self.grid.evidence())
        if type(self.stepper) is CrankNicolson:
            out["stepper"] = dict(self.stepper.evidence())
        else:
            out["stepper"] = component_fingerprint(self.stepper) or "uncertifiable"
        for name, builtin in (("operator", CentralDifference()), ("readout", StandardReadout())):
            component = getattr(self, name)
            if component != builtin:
                out[name] = component_fingerprint(component) or "uncertifiable"
        out["event_projection"] = getattr(self.event_projection, "value", self.event_projection)
        out["cache_enabled"] = self.cache_enabled
        for f in fields(EngineParams):
            out[f.name] = getattr(self, f.name)
        return out

    @classmethod
    def from_config(cls, config_or_path: Any, **overrides: Any) -> "PDEParams":
        """Load PDEParams from a dict, YAML, or JSON whose keys are field names.

        ``grid`` is a nested mapping of GridConfig fields (a ``bounds`` list
        becomes a tuple). ``engine``, when present, must be ``"pde"``;
        ``profile`` and ``reverse`` are rejected (PDE has no profile tier since
        0.6.0). Unknown keys go through the removal map. ``overrides`` win.
        """
        from quantark.asset.equity.engine.pde.grid.config import GridConfig

        from .engine_param_profiles import load_param_config

        data = dict(load_param_config(config_or_path))
        for key in ("profile", "reverse"):
            if key in data:
                raise ValidationError(
                    f"PDEParams.from_config: {key!r} is not accepted; PDE has no "
                    "profile tier since 0.6.0. Use accuracy= and grid= field names."
                )
        engine = data.pop("engine", None)
        if engine is not None and str(engine).lower() != "pde":
            raise ValidationError(f"config engine {engine!r} is not 'pde'")
        data.update(overrides)
        for key in ("operator", "readout"):
            if key in data:
                raise ValidationError(
                    f"PDEParams.from_config: {key!r} cannot be named in a config file "
                    "(custom components are constructed in code; the built-ins have no "
                    "settings in this release)"
                )
        stepper = data.get("stepper")
        if isinstance(stepper, Mapping):
            from quantark.asset.equity.engine.pde.core.components import CrankNicolson

            data["stepper"] = CrankNicolson(**stepper)
        grid = data.get("grid")
        if isinstance(grid, Mapping):
            grid = dict(grid)
            if isinstance(grid.get("bounds"), list):
                grid["bounds"] = tuple(grid["bounds"])
            data["grid"] = GridConfig(**grid)
        return cls(**data)

    def resolved_grid(self):
        """The REQUESTED grid: ``accuracy``'s shared preset under the ``grid`` overlay.

        ``max_steps`` is checked once a maturity is known (``build_time``): a grid
        over it raises, it is never scaled (spec 3A §5.6). The delivered count is
        on the solve's ``TimeLayout`` (``requested_steps``, ``actual_steps``). Routes with their own preset table (local-vol
        vanilla/barrier, DCN) resolve over that table instead.
        """
        from quantark.asset.equity.engine.pde.grid.config import resolve_config

        return resolve_config(self.accuracy, self.grid)


#: How the scalar price is recovered from the nodal value surface.
#:
#: ``legacy_linear`` interpolates linearly between the two nodes straddling
#: the spot.  Because barrier alignment pins the grid to the barrier rather
#: than to spot, that makes the price piecewise linear in ``log S`` and its
#: spot derivative a staircase.  ``transition`` evaluates the final backward
#: transition operator at the spot instead, which is what
#: ``QuadratureCore._calculate_final_value`` already does for the
#: product-agnostic path.  See docs/bucket-futures-hedge/quad-readout/.
QUAD_READOUT_MODES = ("legacy_linear", "transition")


@dataclass
class QuadParams(EngineParams):
    """
    Quadrature engine configuration.

    Attributes:
        grid_points: Number of integration points (default: 1001).
                     Odd counts preserve nested refinement grids.
        num_std_devs: Number of standard deviations for integration bounds (default: 10).
                      Larger values capture more of the distribution tail.
        integration_rule: Composite integration rule used by FFT convolution.
            "trapezoid" (default) is phase-stable after discontinuous
            autocallable events; "simpson" preserves the legacy alternating
            weights.
        auto_converge: Refine nested odd grids until successive prices meet
            the configured absolute-plus-relative tolerance.
        convergence_rel_tol: Relative component of the convergence tolerance.
        convergence_abs_tol: Absolute component of the convergence tolerance.
        max_convergence_grid_points: Fail-closed refinement cap.
        filter_unreachable_barriers: Treat KO levels outside the configured
            observation-time Gaussian envelope as disabled sentinels while
            retaining their event/coupon dates.
        barrier_reach_stddevs: Optional reachability envelope override. None
            uses num_std_devs.
        fft_padding_factor: Zero-padding factor for FFT-based convolution (default: 2).
        fft_filter_alpha: Spectral filter strength (0 disables filtering).
        fft_filter_power: Spectral filter power/exponent (higher = sharper cutoff).
        stability_preset: Optional stability preset overriding FFT padding/filter defaults.
                          Supported: conservative, balanced, aggressive.
        align_priority: Barrier alignment priority (auto, ko, coupon, ki).
        align_cell_stretch: Opt-in fix for the alignment discontinuity, and
            None (unset) by default because it moves barrier prices. A
            uniform lattice can pin only ONE barrier onto a node; align_priority
            picks which, and because that choice is made by distance from
            spot, it can switch mid-bump and make delta discontinuous. Setting
            this widens the cell just enough that the separations between
            barriers are whole numbers of cells, so every barrier lands on a
            node at once and the choice stops mattering. The value is the
            largest fractional widening allowed, e.g. 0.02 for 2%; no
            widening within budget means the grid is left alone. grid_points
            is never changed and the cell never shrinks, so the grid cannot
            grow -- the cost is at most this fraction of spatial resolution,
            and the truncation domain widens by the same fraction. Exact for
            two distinct levels; three or more cannot share one lattice, so
            the widening minimises the worst off-node distance instead.
        event_smoothing_cells: Half-width of smoothing window in grid cells (0 disables).
        event_smoothing_mode: Smoothing mode (fixed, auto, reverse_aware).
        event_smoothing_kernel: Smoothing kernel (cosine, tanh).
        event_smoothing_log_width: Target log-space width for auto smoothing.
        event_projection: Spatial representation of discrete KO/coupon/KI events.
            CELL_AVERAGE (default) integrates the piecewise event over the
            threshold-straddling dual cell; NODAL preserves the legacy
            smoothing/hard-mask path.
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
        event_stats_mode: Event-distribution algorithm: "stacked"
            (per-observation indicator rows, default) or "forward_density"
            (forward transition-density march; distribution values differ at
            finite grid, npv identical; see spec 2026-08-24).
        readout: How the scalar price is recovered from the nodal surface,
            one of QUAD_READOUT_MODES. "legacy_linear" (default) interpolates
            linearly between the straddling nodes; "transition" evaluates the
            final backward transition operator at the spot. The mode governs
            EVERY readout of the recursion, price and event decomposition
            alike, so their reconciliation holds under either. Changing it
            changes prices by a few basis points and rebases QUAD goldens.
    """

    grid_points: int = 1001  # Odd count keeps nested refinements node-compatible
    num_std_devs: float = 10.0
    integration_rule: str = "trapezoid"
    auto_converge: bool = False
    convergence_rel_tol: float = 1.0e-4
    convergence_abs_tol: float = 1.0e-8
    max_convergence_grid_points: int = 64001
    filter_unreachable_barriers: bool = True
    barrier_reach_stddevs: Optional[float] = None
    fft_padding_factor: int = 2
    fft_filter_alpha: float = 12.0
    fft_filter_power: int = 8
    stability_preset: Optional[str] = None
    align_priority: str = "auto"
    align_cell_stretch: Optional[float] = None
    event_smoothing_cells: int = 1
    event_smoothing_mode: str = "fixed"
    event_smoothing_kernel: str = "cosine"
    event_smoothing_log_width: float = 0.002
    event_projection: Union[EventProjectionMode, str] = (
        EventProjectionMode.CELL_AVERAGE
    )
    min_diffusion_stddev_cells: float = 2.5
    max_adaptive_grid_points: int = 5001
    ki_monitoring_mode: Union[KnockInMonitoringMode, str] = (
        KnockInMonitoringMode.EXACT_DISCRETE
    )
    bgk_min_ki_observations: int = 100
    event_stats_mode: str = "stacked"
    readout: str = "legacy_linear"

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
        self.integration_rule = str(self.integration_rule).lower()
        if self.integration_rule not in ("trapezoid", "simpson"):
            raise ValidationError(
                "integration_rule must be one of trapezoid, simpson, got "
                f"{self.integration_rule}"
            )
        if (
            not math.isfinite(self.convergence_rel_tol)
            or self.convergence_rel_tol < 0.0
        ):
            raise ValidationError(
                "convergence_rel_tol must be non-negative, got "
                f"{self.convergence_rel_tol}"
            )
        if (
            not math.isfinite(self.convergence_abs_tol)
            or self.convergence_abs_tol < 0.0
        ):
            raise ValidationError(
                "convergence_abs_tol must be non-negative, got "
                f"{self.convergence_abs_tol}"
            )
        if (
            self.convergence_rel_tol == 0.0
            and self.convergence_abs_tol == 0.0
        ):
            raise ValidationError(
                "at least one of convergence_rel_tol or convergence_abs_tol "
                "must be positive"
            )
        if (
            self.auto_converge
            and self.max_convergence_grid_points < self.grid_points
        ):
            raise ValidationError(
                "max_convergence_grid_points must be at least grid_points, got "
                f"{self.max_convergence_grid_points} < {self.grid_points}"
            )
        if self.max_convergence_grid_points % 2 == 0:
            self.max_convergence_grid_points += 1
        if (
            self.barrier_reach_stddevs is not None
            and (
                not math.isfinite(self.barrier_reach_stddevs)
                or self.barrier_reach_stddevs <= 0.0
            )
        ):
            raise ValidationError(
                "barrier_reach_stddevs must be positive when supplied, got "
                f"{self.barrier_reach_stddevs}"
            )
        if self.readout not in QUAD_READOUT_MODES:
            raise ValidationError(
                f"readout must be one of {QUAD_READOUT_MODES}, got {self.readout!r}"
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
        if self.align_cell_stretch is not None:
            stretch = float(self.align_cell_stretch)
            if not 0.0 < stretch <= 1.0:
                raise ValidationError(
                    "align_cell_stretch must be in (0, 1], got "
                    f"{self.align_cell_stretch}"
                )
            self.align_cell_stretch = stretch
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
        if self.event_stats_mode not in ("stacked", "forward_density"):
            raise ValidationError(
                "event_stats_mode must be one of stacked, forward_density, "
                f"got {self.event_stats_mode}"
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

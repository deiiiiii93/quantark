"""
Configuration objects for OTC autocallable backtests.
"""

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Literal, Optional

from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

from .market import AutocallableMarketDataSet
from .strategy_state import AutocallableDeltaHedgeStrategy


# Canonical home: quantark.backtest.futures_ledger (re-exported here).
from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: F401,E402


# VolModelCalibrationConfig lives in quantark.volcalibration.config: the
# vol-calibration engine must not depend on the backtest package config shape.
from quantark.volcalibration.config import VolModelCalibrationConfig  # noqa: E402

@dataclass
class AutocallableEngineConfig:
    """
    Engine selection for pricing, surfaces, and event stats.

    ``vol_source`` selects the daily vol channel used by
    ``ProductReplay.build_env``:

    - ``"scalar"`` (default): historical behavior — one flat vol from the
      market ``volatility`` column.
    - ``"surface"``: reprice daily against the admitted IV-surface artifact
      (requires ``market_data.surface_history``); ``surface_vol_mode`` picks
      the vol object built from the artifact:

      - ``"flat_atm_remaining"``: ATM term structure sampled at the product's
        remaining maturity, wrapped in a flat surface (refreshed daily).
      - ``"term_structure"``: ATM pillar term structure (strike-independent).
      - ``"full_grid"``: full strike x maturity smile grid.

    ``vol_model`` selects the daily pricing model (Task 2.3):

    - ``"bsm"`` (default): the classic factory engine (``pricing_engine_type``).
    - ``"localvol"`` / ``"heston"`` / ``"heston_slv"``: the day's pricing
      engine is a per-day calibrated vol-model snowball engine
      (``create_vol_model_engine``), calibrated once per surface artifact
      (sha-keyed cache) and used for both the base price and the
      bumped-greek reprices of that day.  Requires ``vol_source="surface"``
      and forces ``surface_vol_mode="full_grid"``: the LV variants consume
      the full smile grid, and the Heston variants price from their own
      model dynamics (the env vol object is unused by those engines, so
      full_grid keeps the recorded provenance consistent).
      ``pricing_engine_type`` still drives the surface/event-stats engines.

    ``method`` belongs to ``pricing_engine_type`` (a ``PDEMethod`` for PDE, a
    ``MonteCarloMethod`` for MC, ...) and is also what the surface and
    event-stats engines receive.  A vol-model MC route needs its OWN method
    (e.g. ``MonteCarloMethod.RANDOMIZED_QUASI``) which is generally NOT valid
    for ``pricing_engine_type``: a run that prices Heston on RQMC while
    keeping the deterministic PDE surface/event-stats engines cannot express
    both through one field.  ``vol_model_mc_method`` is that second slot; it
    falls back to ``method`` when unset, so existing configurations are
    unchanged.

    ``dividend_source`` selects the daily dividend/carry channel:

    - ``None`` (default): the historical behaviour — scalar vol mode prices
      off ONE flat yield implied from the active hedge contract (simple
      compounding, floored at zero); surface vol mode prices off the
      artifact's option-implied forward pillars.
    - ``"active_contract"``: the scalar-mode flat yield, explicitly.
    - ``"futures_curve"``: every listed contract on the day with at least
      ``futures_curve_min_tenor_days`` calendar days to expiry inverted
      through ``IndexFuturesCurve`` — continuous compounding, signed, linear
      in nodal q.  Beyond the last listed tenor ``futures_curve_extrapolation``
      picks ``"flat_q"`` (the endpoint zero yield is held) or
      ``"flat_forward_carry"`` (the last segment's forward carry continues,
      via ``ForwardCarryCurve``) or ``"surface_forward_carry"`` (the chain is
      held in log-forward space and continued past its last contract with
      the admitted IV-surface artifact's option-implied FORWARD carry, level
      matched at the join, then flat forward carry; needs
      ``market_data.surface_history``).  The minimum tenor (default one week,
      at least one day: a contract expiring today has no defined yield)
      drops delivery-week contracts, whose annualised basis is noise (a 1%
      basis two days out reads as a 180% yield); a day left with ONE eligible
      contract prices off that node as a flat continuous signed yield — the
      exact one-node limit of both extrapolation conventions — and a day
      with none fails closed.  The state row's ``pricing_q`` is the term
      yield at the product's remaining maturity (for a multi-product book:
      the FIRST product's; pricing itself samples each product's own).
    - ``"surface_forwards"``: the admitted IV-surface artifact's parity
      forward pillars as a ``TermStructureDividendYield``, usable with the
      scalar vol channel (requires ``market_data.surface_history``).

    The two term sources are incompatible with ``fixed_dividend_yield`` and
    with the flat-q surface grid (``calculate_surfaces``); both are rejected
    at config time rather than silently swapping the model.
    """

    pricing_engine_type: EngineType = EngineType.PDE
    method: Optional[Any] = None
    vol_model_mc_method: Optional[Any] = None
    pde_params: Optional[PDEParams] = None
    mc_params: Optional[MCParams] = None
    quad_params: Optional[QuadParams] = None
    surface_engine_type: Optional[EngineType] = None
    event_stats_engine_type: Optional[EngineType] = None
    vol_source: Literal["scalar", "surface"] = "scalar"
    surface_vol_mode: Literal[
        "flat_atm_remaining", "term_structure", "full_grid"
    ] = "flat_atm_remaining"
    vol_model: Literal["bsm", "localvol", "heston", "heston_slv"] = "bsm"
    vol_model_solver: Literal["pde", "mc"] = "pde"
    vol_model_calibration: Optional[VolModelCalibrationConfig] = None
    vol_model_engine_options: dict[str, Any] = field(default_factory=dict)
    event_stats_fallback: Literal["none", "mc"] = "none"
    dividend_source: Optional[
        Literal["active_contract", "futures_curve", "surface_forwards"]
    ] = None
    futures_curve_extrapolation: Literal[
        "flat_q", "flat_forward_carry", "surface_forward_carry"
    ] = "flat_q"
    futures_curve_min_tenor_days: int = 7

    def uses_term_dividend_source(self) -> bool:
        """True when the day's dividend object is a term structure chosen by
        ``dividend_source`` (as opposed to the legacy scalar channel)."""
        return self.dividend_source in ("futures_curve", "surface_forwards")

    def __post_init__(self) -> None:
        supported = {
            EngineType.ANALYTICAL,
            EngineType.PDE,
            EngineType.MONTE_CARLO,
            EngineType.QUADRATURE,
        }
        if self.pricing_engine_type not in supported:
            raise ValidationError(
                "pricing_engine_type must be ANALYTICAL, PDE, MONTE_CARLO, or QUADRATURE"
            )
        if self.surface_engine_type is not None and self.surface_engine_type not in supported:
            raise ValidationError(
                "surface_engine_type must be ANALYTICAL, PDE, MONTE_CARLO, or QUADRATURE"
            )
        if (
            self.event_stats_engine_type is not None
            and self.event_stats_engine_type not in supported
        ):
            raise ValidationError(
                "event_stats_engine_type must be PDE, MONTE_CARLO, or QUADRATURE"
            )
        if self.vol_source not in ("scalar", "surface"):
            raise ValidationError("vol_source must be 'scalar' or 'surface'")
        if self.surface_vol_mode not in (
            "flat_atm_remaining",
            "term_structure",
            "full_grid",
        ):
            raise ValidationError(
                "surface_vol_mode must be 'flat_atm_remaining', "
                "'term_structure', or 'full_grid'"
            )
        if self.vol_model not in ("bsm", "localvol", "heston", "heston_slv"):
            raise ValidationError(
                "vol_model must be 'bsm', 'localvol', 'heston', or 'heston_slv'"
            )
        if self.vol_model_solver not in ("pde", "mc"):
            raise ValidationError("vol_model_solver must be 'pde' or 'mc'")
        if self.event_stats_fallback not in ("none", "mc"):
            raise ValidationError("event_stats_fallback must be 'none' or 'mc'")
        if self.dividend_source not in (
            None,
            "active_contract",
            "futures_curve",
            "surface_forwards",
        ):
            raise ValidationError(
                "dividend_source must be None, 'active_contract', "
                "'futures_curve', or 'surface_forwards'"
            )
        if self.futures_curve_extrapolation not in (
            "flat_q", "flat_forward_carry", "surface_forward_carry"
        ):
            raise ValidationError(
                "futures_curve_extrapolation must be 'flat_q', 'flat_forward_carry' "
                "or 'surface_forward_carry'"
            )
        if int(self.futures_curve_min_tenor_days) < 1:
            # a contract expiring today has T = 0 and no defined yield
            raise ValidationError("futures_curve_min_tenor_days must be at least 1")
        if self.vol_model_calibration is None:
            self.vol_model_calibration = VolModelCalibrationConfig()
        elif not isinstance(self.vol_model_calibration, VolModelCalibrationConfig):
            raise ValidationError(
                "vol_model_calibration must be a VolModelCalibrationConfig"
            )
        if self.vol_model != "bsm":
            if self.vol_source != "surface":
                raise ValidationError(
                    "vol_model != 'bsm' requires vol_source='surface' "
                    "(per-day calibration is keyed by surface artifact sha)"
                )
            # Vol-model engines consume either the full smile grid (LV) or
            # their own model dynamics (Heston/SLV); forcing full_grid keeps
            # the env vol object consistent for the daily record.
            self.surface_vol_mode = "full_grid"

    def resolve_vol_model_mc_method(self) -> Optional[Any]:
        """MC method for the vol-model engines (falls back to ``method``)."""
        if self.vol_model_mc_method is not None:
            return self.vol_model_mc_method
        return self.method

    def resolve_surface_engine_type(self) -> EngineType:
        if self.surface_engine_type is not None:
            return self.surface_engine_type
        if self.pricing_engine_type == EngineType.MONTE_CARLO:
            return EngineType.QUADRATURE
        return self.pricing_engine_type

    def resolve_event_stats_engine_type(self) -> EngineType:
        if self.event_stats_engine_type is not None:
            return self.event_stats_engine_type
        return self.pricing_engine_type


def _validate_term_dividend_source(
    engine_config: AutocallableEngineConfig,
    *,
    fixed_dividend_yield: Optional[float],
    calculate_surfaces: bool,
    dividend_roll_policy: Optional[FuturesRollPolicy] = None,
) -> None:
    """Reject run options that would silently flatten a term dividend source.

    ``fixed_dividend_yield`` is a scalar override of the active-contract
    yield, and the spot x q surface grid reprices on flat ``q`` nodes around a
    scalar centre; under ``futures_curve`` / ``surface_forwards`` either one
    would swap the pricing model behind the recorded provenance.
    """
    if not engine_config.uses_term_dividend_source():
        return
    source = engine_config.dividend_source
    if fixed_dividend_yield is not None:
        raise ValidationError(
            f"fixed_dividend_yield cannot be combined with dividend_source="
            f"{source!r}; the term structure IS the dividend input"
        )
    if calculate_surfaces:
        raise ValidationError(
            f"calculate_surfaces is not supported with dividend_source={source!r}: "
            "the spot x q surface grid prices on flat q nodes"
        )
    if dividend_roll_policy is not None:
        raise ValidationError(
            f"dividend_roll_policy cannot be combined with dividend_source="
            f"{source!r}: a term source reads the WHOLE chain, so naming one "
            "contract for the carry would be silently ignored"
        )


CARRY_AUDIT_MODES = ("none", "sampled", "daily")

#: BumpConfig.spot_bump's own default: the relative spot bump the pricing
#: engines use when a run names none.
DEFAULT_SPOT_BUMP_REL = 0.01


@dataclass(frozen=True)
class CarryRecordingPlan:
    """What a run actually decided to record, after resolution.

    Recording is on when it was asked for, OR a bucket strategy is in use, OR
    an audit mode was requested.  A non-``none`` audit with the flag left off
    turns recording ON here rather than disappearing, so an artifact never
    claims an audit it could not have produced.
    """

    record: bool
    audit_mode: str
    audit_dates: tuple
    settings: Optional[Any]
    reference_notional: Optional[float]

    def audits_on(self, date) -> bool:
        if self.audit_mode == "none":
            return False
        if self.audit_mode == "daily":
            return True
        return date in self.audit_dates

    def as_metadata(self) -> dict:
        return {
            "record_carry_exposure": self.record,
            "carry_audit_mode": self.audit_mode,
            "carry_audit_dates": [str(d) for d in self.audit_dates],
            "reference_notional": self.reference_notional,
        }


def _normalise_audit_dates(dates) -> tuple:
    stamps = tuple(dates or ())
    for stamp in stamps:
        if not hasattr(stamp, "year"):
            raise ValidationError(
                f"carry_audit_dates must hold date-like values, got {stamp!r}"
            )
    return tuple(sorted(set(stamps)))


def _gross_contractual_notional(entries) -> Optional[float]:
    """``sum |Q_p| * initial_price_p * contract_multiplier_p`` for the book.

    ``None`` when a product does not document both an initial price and a
    contract multiplier: an undocumented notional must be supplied explicitly
    rather than guessed from a signed PV, the current spot or the remaining
    alive notional.
    """
    from quantark.backtest.futures_risk import gross_contractual_notional

    rows = []
    for quantity, product in entries:
        initial_price = getattr(product, "initial_price", None)
        multiplier = getattr(product, "contract_multiplier", None)
        if initial_price is None or multiplier is None:
            return None
        rows.append((float(quantity), float(initial_price), float(multiplier)))
    if not rows:
        return None
    return gross_contractual_notional(rows)


def _validate_carry_hedge(
    strategy,
    *,
    hedge_kind: str,
    engine_config: AutocallableEngineConfig,
    dividend_roll_policy,
    pnl_explain,
    record_carry_exposure: bool,
    carry_audit_mode: str,
    carry_audit_dates,
    carry_risk_settings,
    products,
    delta_bump_size=None,
) -> CarryRecordingPlan:
    """Accept a bucket hedge only where its risk coordinates actually exist.

    Every coordinate a bucket policy sizes against must be an ACTUAL tradable
    futures contract with a quote, an expiry and a multiplier.  A flat carry
    channel has no nodes at all, and an option-implied forward is a different
    instrument from the future the hedge trades, so neither can inherit these
    formulas without a separate quote Jacobian and basis reporting.
    """
    from quantark.backtest.futures_risk import CarryRiskSettings
    from quantark.backtest.strategy.futures_bucket_strategy import (
        FuturesBucketHedgeStrategy,
    )

    if carry_audit_mode not in CARRY_AUDIT_MODES:
        raise ValidationError(
            f"carry_audit_mode must be one of {CARRY_AUDIT_MODES}, "
            f"got {carry_audit_mode!r}"
        )
    audit_dates = _normalise_audit_dates(carry_audit_dates)
    if carry_risk_settings is not None and not isinstance(
        carry_risk_settings, CarryRiskSettings
    ):
        raise ValidationError(
            "carry_risk_settings must be a CarryRiskSettings instance"
        )

    is_bucket = isinstance(strategy, FuturesBucketHedgeStrategy)
    if is_bucket:
        if hedge_kind != "futures":
            raise ValidationError("bucket hedge requires futures instruments")
        if engine_config.dividend_source != "futures_curve":
            raise ValidationError(
                "bucket hedge requires actual futures_curve quotes"
            )
        if engine_config.futures_curve_extrapolation not in {
            "flat_q",
            "flat_forward_carry",
        }:
            raise ValidationError(
                "bucket hedge does not support external tail coordinates"
            )
        if dividend_roll_policy is not None or pnl_explain is not None:
            raise ValidationError(
                "bucket hedge is incompatible with dividend rolls and "
                "single-leg explain"
            )

    record = bool(record_carry_exposure) or is_bucket or carry_audit_mode != "none"
    if not record:
        # Nothing new is enabled: do not impose a notional requirement on a
        # legacy caller that never asked for a risk report.
        return CarryRecordingPlan(
            record=False,
            audit_mode="none",
            audit_dates=(),
            settings=None,
            reference_notional=None,
        )

    seen = [pid for pid in (getattr(p, "position_id", None) for p in products)]
    if len(set(seen)) != len(seen):
        raise ValidationError(
            f"an audited risk book needs unique position ids, got {seen}"
        )
    for entry in products:
        quantity = float(getattr(entry, "quantity", 0.0))
        if not math.isfinite(quantity):
            raise ValidationError(
                "an audited risk book needs finite signed quantities"
            )

    settings = carry_risk_settings or CarryRiskSettings()
    notional = settings.reference_notional
    if notional is None:
        notional = _gross_contractual_notional(
            [(getattr(p, "quantity", 0.0), getattr(p, "product", None)) for p in products]
        )
    if notional is None:
        raise ValidationError(
            "carry recording needs a reference notional: this book holds a "
            "product without a documented initial price and contract "
            "multiplier, so set carry_risk_settings.reference_notional"
        )
    # The audit's spot bump follows the run's EFFECTIVE pricing bump, so the
    # direct D_F is measured the same way the day's delta is, and the
    # resolved value is recorded rather than the request to infer one.
    spot_bump = settings.audit_spot_bump_rel
    if spot_bump is None:
        spot_bump = (
            float(delta_bump_size)
            if delta_bump_size is not None
            else DEFAULT_SPOT_BUMP_REL
        )
    settings = settings.resolved(
        reference_notional=notional, audit_spot_bump_rel=spot_bump
    )
    return CarryRecordingPlan(
        record=True,
        audit_mode=carry_audit_mode,
        audit_dates=audit_dates,
        settings=settings,
        reference_notional=float(notional),
    )


@dataclass
class SurfaceGridConfig:
    """Compact surface grid configuration."""

    spot_nodes: int = 5
    spot_width: float = 0.05
    q_nodes: int = 3
    q_width: float = 0.01

    def __post_init__(self) -> None:
        if self.spot_nodes < 3:
            raise ValidationError("spot_nodes must be at least 3")
        if self.q_nodes < 1:
            raise ValidationError("q_nodes must be positive")
        if self.spot_width <= 0:
            raise ValidationError("spot_width must be positive")
        if self.q_width < 0:
            raise ValidationError("q_width must be non-negative")


@dataclass
class AutocallableBacktestConfig:
    """
    Full configuration for an OTC autocallable historical hedge replay.
    """

    product: Any
    market_data: AutocallableMarketDataSet
    engine_config: AutocallableEngineConfig = field(
        default_factory=AutocallableEngineConfig
    )
    strategy: Optional[Any] = None
    roll_policy: FuturesRollPolicy = field(default_factory=FuturesRollPolicy)
    transaction_cost_model: TransactionCostModel = field(default_factory=ZeroCostModel)
    product_quantity: float = -1.0
    underlying: str = "equity_index"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_product_price: Optional[float] = None
    fixed_dividend_yield: Optional[float] = None
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    surface_config: SurfaceGridConfig = field(default_factory=SurfaceGridConfig)
    calculate_surfaces: bool = True
    calculate_event_probabilities: bool = True
    terminate_on_lifecycle_end: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    # Appended AFTER metadata so existing positional construction keeps its slots.
    pnl_explain: Optional[Any] = None   # PnLExplainConfig; None = no explain, no behaviour change
    # Which contract the FLAT carry channel is inverted from.  None (default)
    # = the contract the hedge holds, which is the historical behaviour: the
    # engine hands one selected row to both the hedge trade and build_env.
    # Set it to price the carry off a contract the hedge does not trade, e.g.
    # read from the longest listed contract while the delta sits in the front
    # month.  A term dividend_source reads the WHOLE chain and would ignore
    # it, so that combination is rejected rather than silently dropped.
    dividend_roll_policy: Optional[FuturesRollPolicy] = None
    # Carry risk recording, appended after dividend_roll_policy so existing
    # positional construction keeps its slots.  All default to off: a legacy
    # run prices nothing extra and resolves no notional.
    record_carry_exposure: bool = False
    carry_audit_mode: str = "none"
    carry_audit_dates: tuple = ()
    carry_risk_settings: Optional[Any] = None  # CarryRiskSettings

    def __post_init__(self) -> None:
        if self.product is None:
            raise ValidationError("product is required")
        if self.market_data is None:
            raise ValidationError("market_data is required")
        if self.product_quantity == 0:
            raise ValidationError("product_quantity must be non-zero")
        if self.fixed_dividend_yield is not None and not math.isfinite(
            float(self.fixed_dividend_yield)
        ):
            raise ValidationError("fixed_dividend_yield must be finite")
        _validate_term_dividend_source(
            self.engine_config,
            fixed_dividend_yield=self.fixed_dividend_yield,
            calculate_surfaces=self.calculate_surfaces,
            dividend_roll_policy=self.dividend_roll_policy,
        )
        self.carry_recording = _validate_carry_hedge(
            self.strategy,
            hedge_kind="futures",
            engine_config=self.engine_config,
            dividend_roll_policy=self.dividend_roll_policy,
            pnl_explain=self.pnl_explain,
            record_carry_exposure=self.record_carry_exposure,
            carry_audit_mode=self.carry_audit_mode,
            carry_audit_dates=self.carry_audit_dates,
            carry_risk_settings=self.carry_risk_settings,
            delta_bump_size=self.delta_bump_size,
            products=[
                ReplayProduct(
                    product=self.product,
                    quantity=self.product_quantity,
                    position_id=0,
                    has_lifecycle=True,
                    initial_price=self.initial_product_price,
                )
            ],
        )
        self.carry_audit_dates = self.carry_recording.audit_dates
        for field_name in ("delta_bump_size", "gamma_bump_size"):
            bump = getattr(self, field_name)
            if bump is None:
                continue
            bump_value = float(bump)
            if not math.isfinite(bump_value) or bump_value <= 0.0:
                raise ValidationError(f"{field_name} must be a positive finite value")


@dataclass
class ReplayProduct:
    product: Any
    quantity: float
    position_id: int
    has_lifecycle: bool
    initial_price: Optional[float] = None

    def __post_init__(self):
        if self.product is None:
            raise ValidationError("ReplayProduct.product is required")
        if self.quantity == 0:
            raise ValidationError("ReplayProduct.quantity must be non-zero")


@dataclass
class HedgeSpec:
    kind: str = "futures"
    multiplier: float = 1.0
    roll_policy: Optional[FuturesRollPolicy] = None

    def __post_init__(self):
        if self.kind not in ("futures", "spot"):
            raise ValidationError(f"HedgeSpec.kind must be futures|spot, got {self.kind}")
        if self.kind == "futures" and self.roll_policy is None:
            self.roll_policy = FuturesRollPolicy()


@dataclass
class ReplayBacktestConfig:
    products: list[ReplayProduct]
    market_data: AutocallableMarketDataSet
    hedge: HedgeSpec = field(default_factory=HedgeSpec)
    engine_config: AutocallableEngineConfig = field(default_factory=AutocallableEngineConfig)
    strategy: Any = None
    transaction_cost_model: TransactionCostModel = field(default_factory=ZeroCostModel)
    underlying: str = "equity_index"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    fixed_dividend_yield: Optional[float] = None
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    surface_config: SurfaceGridConfig = field(default_factory=SurfaceGridConfig)
    calculate_surfaces: bool = False
    calculate_event_probabilities: bool = True
    terminate_on_lifecycle_end: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    # Appended AFTER metadata so existing positional construction keeps its slots.
    pnl_explain: Optional[Any] = None   # PnLExplainConfig; None = no explain, no behaviour change
    # Which contract the FLAT carry channel is inverted from.  None (default)
    # = the contract the hedge holds, which is the historical behaviour: the
    # engine hands one selected row to both the hedge trade and build_env.
    # Set it to price the carry off a contract the hedge does not trade, e.g.
    # read from the longest listed contract while the delta sits in the front
    # month.  A term dividend_source reads the WHOLE chain and would ignore
    # it, so that combination is rejected rather than silently dropped.
    dividend_roll_policy: Optional[FuturesRollPolicy] = None
    # Carry risk recording, appended after dividend_roll_policy so existing
    # positional construction keeps its slots.  All default to off: a legacy
    # run prices nothing extra and resolves no notional.
    record_carry_exposure: bool = False
    carry_audit_mode: str = "none"
    carry_audit_dates: tuple = ()
    carry_risk_settings: Optional[Any] = None  # CarryRiskSettings

    def __post_init__(self):
        if not self.products:
            raise ValidationError("ReplayBacktestConfig.products must be non-empty")
        if self.market_data is None:
            raise ValidationError("market_data is required")
        if self.strategy is None:
            self.strategy = AutocallableDeltaHedgeStrategy()
        _validate_term_dividend_source(
            self.engine_config,
            fixed_dividend_yield=self.fixed_dividend_yield,
            calculate_surfaces=self.calculate_surfaces,
            dividend_roll_policy=self.dividend_roll_policy,
        )
        self.carry_recording = _validate_carry_hedge(
            self.strategy,
            hedge_kind=self.hedge.kind,
            engine_config=self.engine_config,
            dividend_roll_policy=self.dividend_roll_policy,
            pnl_explain=self.pnl_explain,
            record_carry_exposure=self.record_carry_exposure,
            carry_audit_mode=self.carry_audit_mode,
            carry_audit_dates=self.carry_audit_dates,
            carry_risk_settings=self.carry_risk_settings,
            delta_bump_size=self.delta_bump_size,
            products=self.products,
        )
        self.carry_audit_dates = self.carry_recording.audit_dates
        if self.engine_config.vol_model != "bsm":
            # create_vol_model_engine builds Snowball engines only; pricing a
            # Phoenix/European with one would be silently wrong (fail-closed).
            from quantark.asset.equity.product.option.snowball_option import (
                SnowballOption,
            )

            for bp in self.products:
                if not isinstance(bp.product, SnowballOption):
                    raise ValidationError(
                        "vol_model != 'bsm' supports SnowballOption products "
                        f"only; position_id={bp.position_id} holds "
                        f"{type(bp.product).__name__}. Wire the corresponding "
                        "vol-model engines before booking other product types."
                    )


# Compatible aliases (canonical names above).
BookProduct = ReplayProduct
BookAutocallableBacktestConfig = ReplayBacktestConfig

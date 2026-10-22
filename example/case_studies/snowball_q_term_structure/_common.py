"""Shared pieces of the snowball q-term-structure hedging study.

The study asks one question: when a desk hedges a CSI 1000 snowball with IM
index futures, does pricing off the futures chain's implied dividend/carry
TERM STRUCTURE hedge better than pricing off ONE flat yield implied from the
hedge contract (the replay engine's historical default)?

Everything the three stages share lives here:

- paths and the locked term sheet (1Y standard snowball, seller short, 50mio);
- the q-model catalogue (``Q_MODELS``) and the ``AutocallableEngineConfig``
  each one maps to (the engine's opt-in ``dividend_source`` channel);
- ``dividend_for``: the SAME dividend object the engine builds, constructed
  outside the engine for the static stage (pinned by test against
  ``ProductReplay.build_env``);
- the two hedge-contract policies (front-month roll; longest listed);
- market-data loading from the mo_volmodels history caches (fail-closed);
- trading calendar, term sheet, inception scheduler, fair-coupon solver;
- run I/O and the paired hedge measures the aggregator reports.

Data: ``example/mo_volmodels/data/history`` (CSI 1000 spot, IM chain, admitted
IV-surface artifacts).  It is a local cache written by the mo_volmodels fetch
stages and is not tracked; every loader here raises ``StudyDataError`` with
the stage to run when it is missing.
"""

from __future__ import annotations

import calendar as _calendar_mod
import json
import math
import statistics
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.backtest.futures_ledger import FuturesRollPolicy
from quantark.backtest.replay import (
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
)
from quantark.backtest.strategy import AutocallableDeltaHedgeStrategy
from quantark.backtest.replay.dividend_source import term_dividend_yield
from quantark.backtest.replay.market import (
    SignedDividendYield,
    derive_implied_dividend_yield,
)
from quantark.param import FlatRateCurve
from quantark.param.vol.surface_history import VolSurfaceHistory
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError
from quantark.util.io import atomic_write_json
from quantark.util.numerical import is_close, is_zero

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[1]
DATA_DIR = STUDY_DIR / "data"
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "example" / "mo_volmodels" / "data" / "history"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "snowball_q_term_structure"

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Locked term sheet
# ---------------------------------------------------------------------------

UNDERLYING_SYMBOL = "000852.SH"
UNDERLYING_NAME = "CSI1000"
FUTURES_PREFIX = "IM"
FUTURES_MULTIPLIER = 200.0
NOTIONAL = 50_000_000.0
FLAT_RATE = 0.02
MATURITY_MONTHS = 12
LOCKOUT_MONTHS = 3
KO_PCT = 1.03
KI_PCT = 0.75
ACT = 365.0
PRODUCT_QUANTITY = -1.0  # SELLER: short one unit sized to the notional
FIRST_INCEPTION_MONTH = (2023, 5)
ATM_VOL_TENOR_YEARS = 1.0  # scalar vol channel: ATM IV at the product tenor
DEFAULT_QUAD_GRID = 401
# The shipped engine default. "transition" evaluates the final backward
# transition at spot instead of interpolating between nodes; it removes the
# delta staircase but moves prices, so it is opt-in here as in the engine.
DEFAULT_QUAD_READOUT = "legacy_linear"
ROLL_DAYS_BEFORE_EXPIRY = 5
# Contracts inside their delivery week carry no measurable annualised carry
# (a 1% basis two days out reads as a 180% yield); the curve skips them.
# Same value as the engine default so ``dividend_for`` mirrors ``build_env``.
FUTURES_CURVE_MIN_TENOR_DAYS = 7

# Coupon solve: PV is affine in the coupon (neither trigger depends on it).
COUPON_LOWER = 0.0
COUPON_UPPER = 0.40
# A deep IM discount with 30% vol prices a 1Y snowball's fair coupon above
# 40%; the bracket expands (x1.5) up to this cap before failing closed.
COUPON_UPPER_CAP = 2.0
COUPON_PV_TOL_FRACTION = 1e-5  # |PV| <= 1e-5 * notional = 500 CNY on 50mio
COUPON_MAX_ITERATIONS = 12


class StudyDataError(RuntimeError):
    """Raised when the study cannot proceed (fail-closed, never fabricate)."""


# ---------------------------------------------------------------------------
# q-model catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QModel:
    """One way of turning the day's market into the pricer's dividend input."""

    name: str
    label: str
    dividend_source: Optional[str]  # AutocallableEngineConfig.dividend_source
    extrapolation: str = "flat_q"
    needs_surface: bool = False
    description: str = ""
    min_tenor_days: int = FUTURES_CURVE_MIN_TENOR_DAYS
    # Flat channel only: which contract the yield is inverted from, as a key
    # of HEDGE_POLICIES.  None = the contract the hedge holds (the engine's
    # own behaviour).  Naming one decouples carry from the hedge leg.
    dividend_policy: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return asdict(self)


Q_MODELS: Dict[str, QModel] = {
    "flat_from_hedge": QModel(
        "flat_from_hedge",
        "Flat q (hedged contract)",
        None,
        description=(
            "The replay engine's historical default: one flat yield implied "
            "from the hedge contract's basis, simple compounding, floored at "
            "zero. Changes every time the hedge rolls.  The ONLY model whose "
            "q depends on the hedge policy: the engine inverts whichever "
            "contract the roll policy holds, so the same name under 'front' "
            "and 'far' is a different carry, not just a different hedge leg."
        ),
    ),
    "flat_from_far": QModel(
        "flat_from_far",
        "Flat q (longest listed contract)",
        None,
        dividend_policy="far",
        description=(
            "The same flat channel as flat_from_hedge, but always inverted "
            "from the LONGEST listed contract, whatever the hedge holds.  "
            "Paired against flat_from_hedge__front under a front-month hedge "
            "it isolates the carry contract, the one comparison the "
            "hedge-following model cannot make on its own.  Hedged with the "
            "far contract it is flat_from_hedge__far by construction."
        ),
    ),
    "term_flat_q": QModel(
        "term_flat_q",
        "Term q (IM chain, flat-q tail)",
        "futures_curve",
        "flat_q",
        description=(
            "Every listed IM contract inverted to q(T_i) = r - ln(F_i/S)/T_i, "
            "linear in q between nodes; beyond the last contract the endpoint "
            "zero yield is held."
        ),
    ),
    "term_flat_fwd": QModel(
        "term_flat_fwd",
        "Term q (IM chain, flat-forward-carry tail)",
        "futures_curve",
        "flat_forward_carry",
        description=(
            "Same nodes; beyond the last contract the last segment's FORWARD "
            "carry continues (ForwardCarryCurve), so the tail forward keeps "
            "drifting at the far spread."
        ),
    ),
    "surface_fwd": QModel(
        "surface_fwd",
        "Term q (MO option forwards)",
        "surface_forwards",
        needs_surface=True,
        description=(
            "Cross-market control: the option-implied parity forwards of the "
            "admitted MO IV-surface artifact, same flat-q tail convention."
        ),
    ),
    "term_opt_tail": QModel(
        "term_opt_tail",
        "Term q (IM chain in log-forward space, option-forward tail)",
        "futures_curve",
        "surface_forward_carry",
        needs_surface=True,
        min_tenor_days=1,
        description=(
            "The literature's answer to the near-expiry annualisation "
            "problem: every listed contract enters as ln(F_i/S) (bounded, "
            "so the minimum tenor is only a one-day backstop), carry is "
            "piecewise-linear in log-forward between contracts (the "
            "calendar-spread slope), and beyond the last contract the "
            "option-implied FORWARD carry of the MO surface continues the "
            "curve, level-matched at the join, then flat."
        ),
    ),
}
Q_MODEL_ORDER: Tuple[str, ...] = tuple(Q_MODELS)
BASELINE_MODEL = "flat_from_hedge"
REFERENCE_MODEL = "term_flat_q"  # the fair coupon is solved under this model


def engine_config_for(
    model: QModel,
    *,
    quad_grid_points: int = DEFAULT_QUAD_GRID,
    quad_readout: str = DEFAULT_QUAD_READOUT,
    align_cell_stretch: Optional[float] = None,
) -> AutocallableEngineConfig:
    """QUAD-engine replay config for one q model (scalar vol channel).

    ``quad_readout`` selects how the engine recovers the price from its nodal
    surface; see docs/bucket-futures-hedge/quad-readout/. It changes prices,
    so it belongs in the run fingerprint.

    ``align_cell_stretch`` widens the cell by at most that fraction so every
    barrier lands on a node, which stops the alignment target moving with
    spot; see docs/bucket-futures-hedge/gates.md. It changes prices too, so
    it belongs in the fingerprint for the same reason.
    """
    return AutocallableEngineConfig(
        pricing_engine_type=EngineType.QUADRATURE,
        quad_params=QuadParams(
            grid_points=int(quad_grid_points), readout=str(quad_readout),
            align_cell_stretch=(
                None if align_cell_stretch is None else float(align_cell_stretch)
            ),
        ),
        dividend_source=model.dividend_source,
        futures_curve_extrapolation=model.extrapolation,
        futures_curve_min_tenor_days=int(model.min_tenor_days),
    )


# ---------------------------------------------------------------------------
# Static dividend construction (mirrors ProductReplay.build_env)
# ---------------------------------------------------------------------------


def live_chain(
    chain_slice: pd.DataFrame, valuation: pd.Timestamp, min_tenor_days: int = 1
) -> pd.DataFrame:
    """Contracts with at least ``min_tenor_days`` calendar days to expiry,
    expiry-sorted (``min_tenor_days=1``: every contract not expiring today)."""
    if int(min_tenor_days) < 1:
        raise ValidationError("min_tenor_days must be at least 1 (T = 0 has no yield)")
    valuation = pd.Timestamp(valuation).normalize()
    days = (pd.to_datetime(chain_slice["expiry_date"]) - valuation).dt.days
    rows = chain_slice[days >= int(min_tenor_days)]
    return rows.sort_values(["expiry_date", "contract"]).reset_index(drop=True)


def curve_quotes(
    chain_slice: pd.DataFrame,
    valuation: pd.Timestamp,
    min_tenor_days: int = FUTURES_CURVE_MIN_TENOR_DAYS,
) -> List[IndexFuturesQuote]:
    valuation = pd.Timestamp(valuation).normalize()
    rows = live_chain(chain_slice, valuation, min_tenor_days)
    return [
        IndexFuturesQuote(
            contract=str(row["contract"]),
            maturity=(pd.Timestamp(row["expiry_date"]) - valuation).days / ACT,
            price=float(row["futures_price"]),
            multiplier=float(row["multiplier"]),
            expiry_date=pd.Timestamp(row["expiry_date"]).to_pydatetime(),
        )
        for _, row in rows.iterrows()
    ]


def futures_curve(
    chain_slice: pd.DataFrame,
    valuation: pd.Timestamp,
    spot: float,
    min_tenor_days: int = FUTURES_CURVE_MIN_TENOR_DAYS,
) -> IndexFuturesCurve:
    """``IndexFuturesCurve`` over the eligible chain (>= 2 nodes required)."""
    valuation = pd.Timestamp(valuation).normalize()
    quotes = curve_quotes(chain_slice, valuation, min_tenor_days)
    if len(quotes) < 2:
        raise ValidationError(
            f"need at least 2 eligible contracts on {valuation.date()}, found {len(quotes)}"
        )
    return IndexFuturesCurve(underlying=UNDERLYING_NAME, spot=float(spot), quotes=quotes)


def dividend_for(
    model: QModel,
    *,
    valuation: pd.Timestamp,
    spot: float,
    rate: float,
    chain_slice: pd.DataFrame,
    active_row: Any = None,
    artifact: Any = None,
):
    """The dividend object ``model`` would hand the pricer on ``valuation``.

    Mirrors ``ProductReplay.build_env`` exactly (pinned by test).  A flat
    model needs ONE contract row, simple-compounded and floored at zero as the
    engine has always done: ``active_row`` (the hedged contract) unless the
    model names its own ``dividend_policy``, in which case that policy picks
    the row out of ``chain_slice`` and ``active_row`` is not used.  The term
    models need the whole chain slice; the surface model needs the artifact.
    """
    valuation = pd.Timestamp(valuation).normalize()
    rate_curve = FlatRateCurve(rate=float(rate))
    if model.dividend_source in (None, "active_contract"):
        policy = dividend_roll_policy_for(model)
        if policy is not None:
            active_row = policy.select_contract(chain_slice, valuation)
        if active_row is None:
            raise ValidationError(f"{model.name} needs the active contract row")
        expiry = pd.Timestamp(active_row["expiry_date"]).normalize()
        ttm = (expiry - valuation).days / ACT
        _, implied_q = derive_implied_dividend_yield(
            rate=float(rate),
            spot=float(spot),
            futures_price=float(active_row["futures_price"]),
            time_to_maturity=ttm,
        )
        return SignedDividendYield(float(implied_q))
    if model.dividend_source == "futures_curve":
        quotes = curve_quotes(chain_slice, valuation, int(model.min_tenor_days))
        if not quotes:
            raise ValidationError(
                f"no contract with at least {model.min_tenor_days} days to "
                f"expiry on {valuation.date()}"
            )
        return term_dividend_yield(
            quotes,
            spot=float(spot),
            rate_curve=rate_curve,
            extrapolation=model.extrapolation,
            underlying=UNDERLYING_NAME,
            artifact=artifact,
        )
    if model.dividend_source == "surface_forwards":
        if artifact is None:
            raise ValidationError("surface_fwd needs the day's IV-surface artifact")
        return artifact.term_structure_dividend_yield(rate=float(rate))
    raise ValidationError(f"unknown dividend_source {model.dividend_source!r}")


def forward_pricing_error_bp(
    div_yield: Any,
    *,
    valuation: pd.Timestamp,
    spot: float,
    rate: float,
    chain_slice: pd.DataFrame,
) -> Dict[str, float]:
    """Per-contract log error of the model forward vs the market mark, in bp.

    ``err_i = 1e4 * ln(F_model(T_i) / F_mkt(T_i))`` with
    ``F_model = S * exp((r - q(T_i)) * T_i)``.  A model that reprices the
    chain scores zero at every node; the flat model reprices only its own
    contract (up to compounding) and misses every other tenor.
    """
    valuation = pd.Timestamp(valuation).normalize()
    out: Dict[str, float] = {}
    for _, row in live_chain(chain_slice, valuation).iterrows():
        t = (pd.Timestamp(row["expiry_date"]) - valuation).days / ACT
        q = float(div_yield.get_yield(t))
        model_forward = float(spot) * math.exp((float(rate) - q) * t)
        out[str(row["contract"])] = 1e4 * math.log(model_forward / float(row["futures_price"]))
    return out


def rms(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# Hedge-contract policies
# ---------------------------------------------------------------------------


@dataclass
class FarContractRollPolicy(FuturesRollPolicy):
    """Hold the LONGEST listed contract; roll only inside the expiry window.

    Same stickiness as the engine's front-month policy (keep the current
    contract while it is more than ``roll_days_before_expiry`` from expiry);
    only the fresh selection differs: the latest expiry instead of the
    earliest.  With the CFFEX listing cycle (当月/次月/当季/次季) this means a
    quarterly contract held for roughly a quarter at a time.
    """

    def select_contract(
        self, futures_slice, valuation_date, current_contract: Optional[str] = None
    ):
        valuation_date = pd.Timestamp(valuation_date).normalize()
        rows = futures_slice.copy()
        rows = rows[pd.to_datetime(rows["expiry_date"]) > valuation_date]
        if rows.empty:
            raise ValidationError(
                f"No non-expired futures contract on {valuation_date.date()}"
            )
        if current_contract is not None:
            current = rows[rows["contract"] == current_contract]
            if not current.empty:
                current_row = current.sort_values("expiry_date").iloc[0]
                days_to_expiry = (
                    pd.Timestamp(current_row["expiry_date"]) - valuation_date
                ).days
                if days_to_expiry > self.roll_days_before_expiry:
                    return current_row
        min_expiry = valuation_date + timedelta(days=self.roll_days_before_expiry)
        candidates = rows[pd.to_datetime(rows["expiry_date"]) > min_expiry]
        if candidates.empty:
            candidates = rows
        return candidates.sort_values(["expiry_date", "contract"]).iloc[-1]


HEDGE_POLICIES: Dict[str, Callable[[], FuturesRollPolicy]] = {
    "front": lambda: FuturesRollPolicy(roll_days_before_expiry=ROLL_DAYS_BEFORE_EXPIRY),
    "far": lambda: FarContractRollPolicy(roll_days_before_expiry=ROLL_DAYS_BEFORE_EXPIRY),
}
HEDGE_POLICY_LABELS = {
    "front": "front-month IM (5-day roll)",
    "far": "longest listed IM (5-day roll)",
}

# ---------------------------------------------------------------------------
# Hedge STRATEGIES, separate from the roll policy above
#
# HEDGE_POLICIES answers "which contract does the reference leg follow"; the
# factory below answers "how is the hedge sized".  The two were the same
# question while every cell held one contract; a bucket cell holds several,
# and still needs a reference contract for the legacy scalar columns.
# ---------------------------------------------------------------------------

#: Study policy name -> the design's objective.  No label calls any of these
#: "bucket-neutral": each neutralises a DIFFERENT thing, and the primary one
#: is not neutral to every node.
BUCKET_OBJECTIVES: Dict[str, str] = {
    "buckets_nodes": "nodes",
    "buckets_far": "spot_far",
    "buckets_spot_parallel": "spot_parallel",
}

#: The single-contract policies that apply the S/F scaling correction.
SCALED_POLICIES: Tuple[str, ...] = ("front_scaled", "far_scaled")

#: Which contract each policy's REFERENCE leg follows.  Bucket policies use
#: the front selector for their scalar reference columns only; the hedge
#: itself spans the whole curve.
HEDGE_ROLL_SELECTOR: Dict[str, str] = {
    "front": "front",
    "far": "far",
    "front_scaled": "front",
    "far_scaled": "far",
    "buckets_nodes": "front",
    "buckets_far": "front",
    "buckets_spot_parallel": "front",
}

HEDGE_STRATEGY_LABELS: Dict[str, str] = {
    "front": "front-month, -D/m",
    "far": "longest listed, -D/m",
    "front_scaled": "front-month, -D S/(m F)",
    "far_scaled": "longest listed, -D S/(m F)",
    "buckets_nodes": "buckets: every modelled node neutral",
    "buckets_far": "buckets + far fold: spot neutral",
    "buckets_spot_parallel": "buckets + two-tenor fold: spot and parallel neutral",
}


def hedge_strategy_for(
    name: str,
    *,
    delta_threshold: float,
    round_contracts: bool,
    hedge_ratio: float = 1.0,
):
    """The sizing strategy for one study hedge policy."""
    from quantark.backtest.strategy import (
        FuturesBucketHedgeStrategy,
        ProportionalFuturesDeltaHedgeStrategy,
    )

    options = dict(
        delta_threshold=delta_threshold,
        round_contracts=round_contracts,
        hedge_ratio=hedge_ratio,
    )
    if name in BUCKET_OBJECTIVES:
        return FuturesBucketHedgeStrategy(
            objective=BUCKET_OBJECTIVES[name], **options
        )
    if name in SCALED_POLICIES:
        return ProportionalFuturesDeltaHedgeStrategy(**options)
    if name in HEDGE_POLICIES:
        return AutocallableDeltaHedgeStrategy(**options)
    raise StudyDataError(f"unknown hedge policy: {name}")


def hedge_roll_policy_for(name: str) -> FuturesRollPolicy:
    """The reference-contract roll policy for one study hedge policy."""
    selector = HEDGE_ROLL_SELECTOR.get(name)
    if selector is None:
        raise StudyDataError(f"unknown hedge policy: {name}")
    return HEDGE_POLICIES[selector]()


def uses_buckets(name: str) -> bool:
    return name in BUCKET_OBJECTIVES


#: The revised study's primary comparison: both supported term models crossed
#: with seven hedge policies.  Fourteen cells, all on ACTUAL futures quotes,
#: because every bucket coordinate must be a contract the hedge can trade.
BUCKET_TERM_MODELS: Tuple[str, ...] = ("term_flat_q", "term_flat_fwd")
BUCKET_HEDGES: Tuple[str, ...] = (
    "front",
    "far",
    "front_scaled",
    "far_scaled",
    "buckets_nodes",
    "buckets_far",
    "buckets_spot_parallel",
)
PRIMARY_BUCKET_CELLS: Tuple[Tuple[str, str], ...] = tuple(
    (model, hedge) for model in BUCKET_TERM_MODELS for hedge in BUCKET_HEDGES
)


def carry_context_for(
    model: QModel,
    *,
    valuation: pd.Timestamp,
    spot: float,
    rate: float,
    chain_slice: pd.DataFrame,
):
    """The day's ``CarryCurveContext`` for a supported term model.

    Static analysis and the replay share this one builder, so a stage-01
    bucket and a replay bucket cannot drift apart.  Flat and external-tail
    controls have no futures-node coordinates and are rejected here rather
    than served a curve they do not use.
    """
    from quantark.backtest.replay.carry_context import (
        SUPPORTED_EXTRAPOLATIONS,
        CarryCurveContext,
    )

    if model.dividend_source != "futures_curve":
        raise StudyDataError(
            f"{model.name} has no futures-node coordinates: bucket risk needs "
            "dividend_source='futures_curve'"
        )
    if model.extrapolation not in SUPPORTED_EXTRAPOLATIONS:
        raise StudyDataError(
            f"{model.name} uses {model.extrapolation!r}, which is not a "
            f"tradable futures tail; supported: {SUPPORTED_EXTRAPOLATIONS}"
        )
    valuation = pd.Timestamp(valuation).normalize()
    quotes = curve_quotes(chain_slice, valuation, int(model.min_tenor_days))
    if not quotes:
        raise StudyDataError(
            f"no contract with at least {model.min_tenor_days} days to expiry "
            f"on {valuation.date()}"
        )
    return CarryCurveContext(
        quotes=tuple(quotes),
        spot=float(spot),
        rate_curve=FlatRateCurve(rate=float(rate)),
        extrapolation=model.extrapolation,
        underlying=UNDERLYING_NAME,
        valuation_date=valuation,
    )


def dividend_roll_policy_for(model: QModel) -> Optional[FuturesRollPolicy]:
    """The roll policy the model's FLAT carry is inverted from.

    ``None`` means the carry follows the hedge contract, which is what the
    replay engine does when ``dividend_roll_policy`` is unset.
    """
    if model.dividend_policy is None:
        return None
    if model.dividend_policy not in HEDGE_POLICIES:
        raise ValidationError(
            f"{model.name} names dividend_policy={model.dividend_policy!r}, "
            f"not one of {tuple(HEDGE_POLICIES)}"
        )
    if model.dividend_source is not None:
        raise ValidationError(
            f"{model.name} sets dividend_policy with dividend_source="
            f"{model.dividend_source!r}: a term source reads the whole chain"
        )
    return HEDGE_POLICIES[model.dividend_policy]()


# ---------------------------------------------------------------------------
# Market data (mo_volmodels history caches)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryFrames:
    spot: pd.DataFrame      # columns: date, spot
    futures: pd.DataFrame   # columns: date, contract, futures_price, expiry_date, multiplier
    history_dir: Path

    @property
    def dates(self) -> List[pd.Timestamp]:
        return [pd.Timestamp(d) for d in self.spot["date"]]


def load_history(history_dir: Path | str = DEFAULT_HISTORY_DIR) -> HistoryFrames:
    """Load the CSI 1000 spot and IM chain caches (fail-closed)."""
    history_dir = Path(history_dir)
    spot_path = history_dir / "csi1000_spot.csv"
    futures_path = history_dir / "im_futures.csv"
    for path in (spot_path, futures_path):
        if not path.exists():
            raise StudyDataError(
                f"missing {path}; refresh the caches with "
                "`/opt/anaconda3/bin/python example/mo_volmodels/01_refresh_market_cache.py`"
            )
    spot = pd.read_csv(spot_path, parse_dates=["date"])
    if "spot" not in spot.columns:
        raise StudyDataError(f"{spot_path} must have a 'spot' column")
    spot = spot.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    futures = pd.read_csv(futures_path, parse_dates=["date", "expiry_date"])
    required = {"date", "contract", "futures_price", "expiry_date", "multiplier"}
    missing = required - set(futures.columns)
    if missing:
        raise StudyDataError(f"{futures_path} missing columns: {sorted(missing)}")
    futures = futures.sort_values(["date", "expiry_date"]).reset_index(drop=True)
    return HistoryFrames(spot=spot, futures=futures, history_dir=history_dir)


_HISTORY_CACHE: Dict[str, VolSurfaceHistory] = {}


def surface_history(history_dir: Path | str = DEFAULT_HISTORY_DIR) -> VolSurfaceHistory:
    """One sha-verifying ``VolSurfaceHistory`` per directory per process."""
    key = str(history_dir)
    cached = _HISTORY_CACHE.get(key)
    if cached is None:
        manifest = Path(history_dir) / "surface_manifest.json"
        if not manifest.exists():
            raise StudyDataError(
                f"missing {manifest}; build it with "
                "`.venv/bin/python example/mo_volmodels/03_build_iv_surface_history.py`"
            )
        cached = VolSurfaceHistory(key)
        _HISTORY_CACHE[key] = cached
    return cached


def atm_vol_channel(
    dates: Sequence[pd.Timestamp],
    history: VolSurfaceHistory,
    tenor_years: float = ATM_VOL_TENOR_YEARS,
) -> pd.DataFrame:
    """Daily scalar vol: ATM IV at ``tenor_years`` off that day's admitted
    surface (carry-forward per the manifest gap policy).  Identical across
    every q model, so the vol channel never confounds the comparison."""
    rows = []
    for stamp in dates:
        artifact = history.surface_for(pd.Timestamp(stamp).date())
        surface = artifact.term_structure_vol_surface()
        rows.append(
            {
                "date": pd.Timestamp(stamp),
                "volatility": float(surface.get_vol(0.0, float(tenor_years), 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_market_dataset(
    frames: HistoryFrames,
    *,
    history: VolSurfaceHistory,
    rate: float = FLAT_RATE,
    vol_tenor_years: float = ATM_VOL_TENOR_YEARS,
    attach_surface_history: bool = True,
) -> AutocallableMarketDataSet:
    dates = frames.dates
    vol = atm_vol_channel(dates, history, vol_tenor_years)
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=frames.spot,
        vol_data=vol,
        rate_data=pd.DataFrame({"date": dates, "rate": float(rate)}),
        futures_data=frames.futures,
        surface_history=history if attach_surface_history else None,
        metadata={
            "spot_symbol": UNDERLYING_SYMBOL,
            "futures_prefix": FUTURES_PREFIX,
            "vol_channel": f"atm_{vol_tenor_years:g}y_from_admitted_surface",
            "rate_source": "flat",
        },
    )


# ---------------------------------------------------------------------------
# Trading calendar and term sheet
# ---------------------------------------------------------------------------


def add_months(d: date, months: int) -> date:
    """Add calendar months, clamping the day to the target month's length."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, _calendar_mod.monthrange(y, m)[1])
    return date(y, m, day)


class TradingCalendar:
    """Trading days from the spot history; plain weekdays beyond its end."""

    def __init__(self, days: Sequence[date]) -> None:
        if not days:
            raise StudyDataError("TradingCalendar needs at least one day")
        self._days = sorted({pd.Timestamp(d).date() for d in days})
        self._set = set(self._days)

    @classmethod
    def from_frames(cls, frames: HistoryFrames) -> "TradingCalendar":
        return cls([d.date() for d in frames.dates])

    @property
    def first(self) -> date:
        return self._days[0]

    @property
    def last(self) -> date:
        return self._days[-1]

    @property
    def days(self) -> List[date]:
        return list(self._days)

    def is_trading_day(self, d: date) -> bool:
        if d in self._set:
            return True
        return d > self._days[-1] and d.weekday() < 5

    def next_trading_day(self, d: date) -> date:
        i = bisect_left(self._days, d)
        if i < len(self._days):
            return self._days[i]
        cur = d
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        return cur

    def trading_days_between(self, start: date, end: date) -> List[date]:
        """All trading days t with ``start < t <= end``."""
        out: List[date] = []
        cur = start
        while cur < end:
            cur += timedelta(days=1)
            if self.is_trading_day(cur):
                out.append(cur)
        return out


@dataclass(frozen=True)
class SnowballTerms:
    inception: date
    maturity_date: date
    maturity_years: float
    ko_times: Tuple[float, ...]
    ki_times: Tuple[float, ...]
    ko_pct: float = KO_PCT
    ki_pct: float = KI_PCT
    notional: float = NOTIONAL

    def summary(self) -> Dict[str, Any]:
        return {
            "inception": self.inception.isoformat(),
            "maturity_date": self.maturity_date.isoformat(),
            "maturity_years": self.maturity_years,
            "n_ko": len(self.ko_times),
            "n_ki": len(self.ki_times),
            "first_ko_time": self.ko_times[0] if self.ko_times else None,
            "ko_pct": self.ko_pct,
            "ki_pct": self.ki_pct,
            "notional": self.notional,
        }


def build_terms(
    inception: date,
    calendar: TradingCalendar,
    *,
    maturity_months: int = MATURITY_MONTHS,
    lockout_months: int = LOCKOUT_MONTHS,
) -> SnowballTerms:
    """KO on month anniversaries lockout..maturity (next trading day), KI on
    every trading day in (inception, maturity]; ACT/365 from inception."""
    maturity_date = calendar.next_trading_day(add_months(inception, maturity_months))
    ko_dates: List[date] = []
    for m in range(lockout_months, maturity_months + 1):
        kd = calendar.next_trading_day(add_months(inception, m))
        ko_dates.append(min(kd, maturity_date))
    ko_dates = sorted(set(ko_dates))
    ki_days = calendar.trading_days_between(inception, maturity_date)
    maturity_years = (maturity_date - inception).days / ACT
    ko_times = tuple((d - inception).days / ACT for d in ko_dates)
    ki_times = tuple((d - inception).days / ACT for d in ki_days)
    if not ko_times or not is_close(ko_times[-1], maturity_years, rel_tol=0.0, abs_tol=1e-12):
        raise StudyDataError("last KO observation must coincide with maturity")
    return SnowballTerms(
        inception=inception,
        maturity_date=maturity_date,
        maturity_years=maturity_years,
        ko_times=ko_times,
        ki_times=ki_times,
    )


def build_product(terms: SnowballTerms, s0: float, coupon: float):
    """Standard snowball, barriers off the inception spot, one unit = notional."""
    s0 = float(s0)
    if s0 <= 0.0:
        raise StudyDataError(f"s0 must be positive, got {s0}")
    product = create_standard_snowball(
        initial_price=s0,
        strike=s0,
        maturity=float(terms.maturity_years),
        contract_multiplier=float(terms.notional) / s0,
        ko_barrier=terms.ko_pct * s0,
        ko_rate=float(coupon),
        ki_barrier=terms.ki_pct * s0,
        num_observations=len(terms.ko_times),
        is_reverse=False,
        ko_observation_dates=list(terms.ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(terms.ki_times),
        rebate_rate=float(coupon),
        include_principal=False,
    )
    product.initial_date = datetime.combine(terms.inception, datetime.min.time())
    return product


# ---------------------------------------------------------------------------
# Inception schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InceptionSchedule:
    inception: date
    maturity_date: date
    trade_end: date
    censored: bool

    @property
    def tag(self) -> str:
        return self.inception.strftime("%Y%m%d")

    def summary(self) -> Dict[str, Any]:
        return {
            "inception": self.inception.isoformat(),
            "maturity_date": self.maturity_date.isoformat(),
            "trade_end": self.trade_end.isoformat(),
            "censored": self.censored,
        }


def enumerate_inceptions(
    calendar: TradingCalendar,
    *,
    first_month: Tuple[int, int] = FIRST_INCEPTION_MONTH,
    data_end: Optional[date] = None,
    maturity_months: int = MATURITY_MONTHS,
) -> List[InceptionSchedule]:
    """First trading day of every month in the window; censored when the
    maturity lies beyond the data end."""
    end = calendar.last if data_end is None else min(data_end, calendar.last)
    first_of_month: Dict[Tuple[int, int], date] = {}
    for d in calendar.days:
        if d > end:
            break
        first_of_month.setdefault((d.year, d.month), d)
    out: List[InceptionSchedule] = []
    cursor = date(first_month[0], first_month[1], 1)
    while cursor <= end:
        inception = first_of_month.get((cursor.year, cursor.month))
        if inception is not None:
            maturity_date = calendar.next_trading_day(add_months(inception, maturity_months))
            out.append(
                InceptionSchedule(
                    inception=inception,
                    maturity_date=maturity_date,
                    trade_end=min(maturity_date, end),
                    censored=maturity_date > end,
                )
            )
        cursor = add_months(cursor, 1)
    if not out:
        raise StudyDataError(f"no inceptions inside the data window ending {end}")
    return out


# ---------------------------------------------------------------------------
# Fair coupon (PV affine in the coupon)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouponSolution:
    coupon: float
    pv: float
    iterations: int
    evaluations: int
    tolerance: float
    converged: bool

    def summary(self) -> Dict[str, Any]:
        return asdict(self)


def solve_fair_coupon(
    pv_at: Callable[[float], float],
    *,
    notional: float = NOTIONAL,
    lower: float = COUPON_LOWER,
    upper: float = COUPON_UPPER,
    tol_fraction: float = COUPON_PV_TOL_FRACTION,
    max_iterations: int = COUPON_MAX_ITERATIONS,
    upper_cap: float = COUPON_UPPER_CAP,
) -> CouponSolution:
    """Illinois-damped false position for a root that should be affine.

    Neither the KO nor the KI trigger depends on the coupon, so the snowball
    PV is affine in it and false position lands on the root in one step; the
    returned coupon is always re-priced and checked against the tolerance
    (the affine structure is exploited, never trusted).  When ``upper`` does
    not bracket the root the bracket grows by 1.5x up to ``upper_cap``;
    fails closed when the root is still not bracketed or the budget runs out.
    """
    tolerance = float(tol_fraction) * float(notional)
    evaluations = 0

    def f(c: float) -> float:
        nonlocal evaluations
        evaluations += 1
        v = float(pv_at(c))
        if not math.isfinite(v):
            raise ValidationError(f"non-finite PV at coupon={c:.8f}")
        return v

    low, high = float(lower), float(upper)
    f_low, f_high = f(low), f(high)
    while f_low * f_high > 0.0 and high < float(upper_cap):
        high = min(high * 1.5, float(upper_cap))
        f_high = f(high)
    if f_low * f_high > 0.0:
        raise ValidationError(
            f"fair coupon not bracketed: f({low:.4f})={f_low:,.2f}, f({high:.4f})={f_high:,.2f}"
        )
    for root, value in ((low, f_low), (high, f_high)):
        if abs(value) <= tolerance:
            return CouponSolution(root, value, 0, evaluations, tolerance, True)
    side = 0
    for iteration in range(1, int(max_iterations) + 1):
        denom = f_high - f_low
        if is_zero(denom):
            raise ValidationError("fair coupon: degenerate bracket")
        c = high - f_high * (high - low) / denom
        fc = f(c)
        if abs(fc) <= tolerance:
            return CouponSolution(c, fc, iteration, evaluations, tolerance, True)
        if fc * f_high < 0.0:
            low, f_low = high, f_high
            if side == -1:
                f_low *= 0.5
            side = -1
        else:
            if side == +1:
                f_high *= 0.5
            side = +1
        high, f_high = c, fc
    raise ValidationError(
        f"fair coupon did not converge in {max_iterations} iterations "
        f"(last |PV|={abs(fc):,.2f} > {tolerance:,.2f})"
    )


# ---------------------------------------------------------------------------
# Run I/O
# ---------------------------------------------------------------------------

#: The five frames every run has ever written.  A historical run has only
#: these, and must stay readable.
LEGACY_RUN_FRAMES = ("states", "greeks", "trades", "rebalances", "actions")

#: The three carry frames a revised (audited) run adds.
CARRY_RUN_FRAMES = ("hedge_legs", "hedge_attribution", "hedge_stresses")

RUN_FRAMES = LEGACY_RUN_FRAMES  # the historical name, unchanged

#: Format versions.  ``legacy`` runs predate carry recording; ``carry_v2``
#: runs carry all eight frames plus the resolved configuration.
RUN_FORMAT_LEGACY = "legacy"
RUN_FORMAT_CARRY = "carry_v2"

#: The revised study writes here, so historical artifacts are never touched.
BUCKET_RUN_VERSION = "bucket_hedge_v2"


def required_frames(run_format: str) -> Tuple[str, ...]:
    if run_format == RUN_FORMAT_CARRY:
        return LEGACY_RUN_FRAMES + CARRY_RUN_FRAMES
    return LEGACY_RUN_FRAMES


def result_frame(results: Any, name: str):
    """One frame from EITHER result API.

    The single result exposes properties; the book result exposes methods.
    The attribute is inspected for callability rather than the frame being
    tested for truth: an empty DataFrame is falsy, and a truth test would
    silently turn "no rows" into "wrong API".
    """
    accessors = {
        "states": ("states_df",),
        "greeks": ("greeks_df",),
        "trades": ("trades_df",),
        "rebalances": ("rebalance_df", "rebalances_df"),
        "actions": ("actions_df",),
        "hedge_legs": ("hedge_legs_df",),
        "hedge_attribution": ("hedge_attribution_df",),
        "hedge_stresses": ("hedge_stresses_df",),
    }[name]
    for accessor in accessors:
        attribute = getattr(type(results), accessor, None)
        if attribute is None:
            continue
        value = getattr(results, accessor)
        return value() if callable(value) else value
    raise StudyDataError(f"results expose no frame named {name!r}")


def cell_name(model: str, hedge: str) -> str:
    return f"{model}__{hedge}"


def run_dir_for(out_dir: Path, inception_tag: str, model: str, hedge: str) -> Path:
    return Path(out_dir) / "runs" / inception_tag / cell_name(model, hedge)


def write_run(
    run_dir: Path,
    results: Any,
    summary: Dict[str, Any],
    *,
    run_format: str = RUN_FORMAT_LEGACY,
    run_config: Optional[Dict[str, Any]] = None,
    audit_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the run's frames, then its configuration, then its summary.

    The completed summary is published LAST and atomically, so a reader that
    finds one can rely on every frame beside it already being there.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in required_frames(run_format):
        # every replay frame is date-indexed (trades and actions included)
        result_frame(results, name).to_csv(run_dir / f"{name}.csv", index=True)
    if run_format == RUN_FORMAT_CARRY:
        atomic_write_json(run_dir / "run_config.json", _jsonable(run_config or {}))
        atomic_write_json(
            run_dir / "audit_summary.json", _jsonable(audit_summary or {})
        )
    summary = dict(summary)
    summary.setdefault("run_format", run_format)
    atomic_write_json(run_dir / "run_summary.json", _jsonable(summary))


def audit_coverage(legs: Any, attribution: Any) -> Dict[str, Any]:
    """Measured/pass/fail/inconclusive counts, not one boolean.

    ``not_available`` is reserved for a run that predates carry recording:
    it is NOT the same as a run whose audits were scheduled and failed.
    """
    if attribution is None or len(attribution) == 0:
        return {"audit_coverage": "not_available"}
    statuses = list(attribution["audit_status"])
    counts = {
        status: int(statuses.count(status))
        for status in ("pass", "fail", "not_measured", "inconclusive")
    }
    measured = len(statuses) - counts["not_measured"]
    by_scenario: Dict[str, int] = {}
    if legs is not None and len(legs):
        for family in sorted(set(legs.get("audit_status", []))):
            by_scenario[str(family)] = int(
                (legs["audit_status"] == family).sum()
            )
    return {
        "audit_coverage": "measured" if measured else "not_measured",
        "dates": len(statuses),
        "measured": measured,
        "by_status": counts,
        "leg_rows_by_status": by_scenario,
        # A completed run with a failed audit stays available for diagnosis
        # and can never be reused as a passing gate result.
        "all_measured_passed": bool(
            measured and counts["fail"] == 0 and counts["inconclusive"] == 0
        ),
    }


def load_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    out: Dict[str, Any] = {}
    summary_path = run_dir / "run_summary.json"
    run_format = RUN_FORMAT_LEGACY
    if summary_path.exists():
        run_format = json.loads(summary_path.read_text()).get(
            "run_format", RUN_FORMAT_LEGACY
        )
    out["run_format"] = run_format
    for name in required_frames(run_format):
        path = run_dir / f"{name}.csv"
        if not path.exists():
            raise StudyDataError(f"run frame missing: {path}")
        out[name] = pd.read_csv(path, index_col=0, parse_dates=True)
    if run_format == RUN_FORMAT_CARRY:
        for name in ("run_config", "audit_summary"):
            path = run_dir / f"{name}.json"
            if not path.exists():
                raise StudyDataError(f"run artifact missing: {path}")
            out[name] = json.loads(path.read_text())
    else:
        out["audit_summary"] = {"audit_coverage": "not_available"}
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise StudyDataError(f"run summary missing: {summary_path}")
    out["summary"] = json.loads(summary_path.read_text())
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _jsonable(payload))


# ---------------------------------------------------------------------------
# Paired hedge measures
# ---------------------------------------------------------------------------


from quantark.backtest.simulation.measures import bp_of as _bp  # noqa: E402  (the study's historical name)
from quantark.backtest.simulation.measures import max_drawdown, path_measures  # noqa: E402


def hedge_measures(states: pd.DataFrame, trades: pd.DataFrame, *, notional: float) -> Dict[str, Any]:
    """Hedge-quality measures of one replay run: ``quantark.backtest.simulation.measures.path_measures``.

    The definitions moved into the library so the simulated-path study
    reports the same numbers; see that module's docstring for each measure.
    """
    return path_measures(states, trades, notional=notional)


def paired_difference(rows: Sequence[Dict[str, Any]], *, key: str, base: str, variant: str, by: str = "inception") -> List[float]:
    """variant - base of ``key`` on matched ``by`` groups (both present)."""
    base_map = {r[by]: r[key] for r in rows if r.get("model") == base}
    out: List[float] = []
    for r in rows:
        if r.get("model") != variant or r[by] not in base_map:
            continue
        a, b = r[key], base_map[r[by]]
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if math.isfinite(a) and math.isfinite(b):
            out.append(a - b)
    return out


def describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None,
                "share_positive": None, "t_stat": None}
    std = statistics.stdev(vals) if len(vals) > 1 else None
    mean = statistics.fmean(vals)
    t_stat = mean / (std / math.sqrt(len(vals))) if std not in (None, 0.0) else None
    return {
        "n": len(vals),
        "mean": mean,
        "median": statistics.median(vals),
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "share_positive": sum(1 for v in vals if v > 0) / len(vals),
        "t_stat": t_stat,
    }

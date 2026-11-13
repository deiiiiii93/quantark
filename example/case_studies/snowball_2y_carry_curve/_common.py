"""Shared pieces of the 2Y snowball flat-q vs CarryCurve study.

Question: how much does a single flat dividend yield misprice and mis-hedge a
2Y CSI 1000 snowball, against the term structure the IM futures + MO options
chain actually quotes (2026-11-05)?

Everything the stages share lives here:

- paths, the valuation date and the locked term sheet;
- market-data loading: the index-basis dashboard DB (read-only) and the CFFEX
  daily statistics CSV, frozen into ``data/market_20261105.json``;
- (T4+) term schedule, product, curves per tail convention, environments,
  engines, the coupon solve, and the study-local forward-interval yield bump.

Stage scripts load this module by path under the unique name
``snowball_2y_cc_common`` (several studies own a ``_common.py``).

Spec: ``.ship/2026-11-05-snowball-2y-carry-curve/spec.md`` (run directory, untracked).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import (
    CarryCurve,
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    FuturesQuote,
    OptionPairQuote,
    SpotQuote,
    build_carry_curve,
)
from quantark.param.div.dividend_yield import DividendYield
from quantark.param.extrapolation import CarryExtrapolation
from quantark.param.vol.marketquotes import implied_vol_black
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import ObservationType
from quantark.util.exceptions import MarketDataError, ValidationError
from quantark.util.numerical import is_close, safe_exp

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

STUDY_DIR = Path(__file__).resolve().parent
DATA_DIR = STUDY_DIR / "data"
RESULTS_DIR = STUDY_DIR / "results"
SNAPSHOT_PATH = DATA_DIR / "market_20261105.json"

VALUATION_DATE = date(2026, 11, 5)
TRADING_DATE = VALUATION_DATE.isoformat()
INDEX_CODE = "000852.SH"
DEFAULT_DB = Path("/Users/fuxinyao/index-dashboard-codex/data/basis.sqlite3")
CFFEX_URL = "http://www.cffex.com.cn/sj/hqsj/rtj/{yyyymm}/{dd}/{yyyymmdd}_1.csv"

#: ACT/365F year fraction from the valuation date (the study's one clock).
ACT = 365.0
#: Flat continuously-compounded rate, the dashboard's own synthetic-forward rate.
RATE = 0.014
IM_MULTIPLIER = 200.0  # CNY per index point per IM futures lot
MO_MULTIPLIER = 100.0  # CNY per index point per MO option

LISTED_SLOTS = (1, 2, 3, 4)
SYNTHETIC_SLOT = 5

# CFFEX daily statistics CSV (GBK): code, open, high, low, volume, turnover,
# open interest, OI change, close (今收盘), settlement (今结算), …
_CSV_CODE, _CSV_CLOSE, _CSV_SETTLE = 0, 8, 9
#: Header names the parser reads by position; any layout change fails closed.
_CSV_HEADER = {_CSV_CODE: "合约代码", _CSV_CLOSE: "今收盘", _CSV_SETTLE: "今结算"}


class StudyDataError(MarketDataError):
    """A market-data row the study needs is missing or malformed (fail closed)."""


def year_fraction(d: date) -> float:
    """ACT/365F from the valuation date."""
    return (d - VALUATION_DATE).days / ACT


# ---------------------------------------------------------------------------
# Market data: dashboard DB + CFFEX CSV → frozen snapshot
# ---------------------------------------------------------------------------


def parse_cffex_csv(payload: bytes) -> Dict[str, Dict[str, float]]:
    """``{contract code: {"close", "settle"}}`` from a CFFEX daily statistics CSV.

    Rows with a non-numeric close or settlement (``null``/``--``) are skipped;
    a missing header raises ``StudyDataError``.
    """
    text = payload.decode("gbk")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or len(header) <= _CSV_SETTLE or any(
        header[i].strip() != name for i, name in _CSV_HEADER.items()
    ):
        raise StudyDataError(f"not a CFFEX daily statistics CSV in the expected layout: {_CSV_HEADER}")
    out: Dict[str, Dict[str, float]] = {}
    for row in reader:
        if len(row) <= _CSV_SETTLE:
            continue
        try:
            close, settle = float(row[_CSV_CLOSE]), float(row[_CSV_SETTLE])
        except ValueError:
            continue
        out[row[_CSV_CODE].strip()] = {"close": close, "settle": settle}
    return out


def read_dashboard_chain(db_path: Path, trading_date: str = TRADING_DATE) -> Dict[str, Any]:
    """Spot, IM T1–T4 and the MO synthetic row of one day, from the dashboard DB.

    Opened read-only (``mode=ro``). Fails closed when any slot is missing or the
    rows disagree on spot.
    """
    uri = f"file:{Path(db_path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT revision_id, slot, contract_code, futures, expiry_date, spot, content_hash, "
            "call_option_contract, put_option_contract FROM latest_daily_basis "
            "WHERE index_code = ? AND trading_date = ? ORDER BY slot",
            (INDEX_CODE, trading_date),
        ).fetchall()
    finally:
        con.close()
    by_slot = {int(r[1]): r for r in rows}
    missing = [s for s in LISTED_SLOTS + (SYNTHETIC_SLOT,) if s not in by_slot]
    if missing:
        raise StudyDataError(f"dashboard DB has no {INDEX_CODE} rows for slots {missing} on {trading_date}")
    spots = {float(r[5]) for r in by_slot.values()}
    if len(spots) != 1:
        raise StudyDataError(f"rows on {trading_date} disagree on spot: {sorted(spots)}")
    futures = []
    for s in LISTED_SLOTS:
        rid, _, code, price, expiry, _, chash, _, _ = by_slot[s]
        if not str(code).startswith("IM"):
            raise StudyDataError(f"slot {s} is {code}, expected an IM contract")
        futures.append({"contract": code, "price": float(price), "expiry": expiry,
                        "revision_id": int(rid), "content_hash": chash})
    rid, _, code, price, expiry, _, chash, call_c, put_c = by_slot[SYNTHETIC_SLOT]
    if not (call_c and put_c):
        raise StudyDataError(f"slot {SYNTHETIC_SLOT} ({code}) carries no option pair")
    return {
        "trading_date": trading_date,
        "spot": spots.pop(),
        "futures": futures,
        "synthetic": {"contract": code, "price": float(price), "expiry": expiry,
                      "call_contract": call_c, "put_contract": put_c,
                      "revision_id": int(rid), "content_hash": chash},
    }


def _cffex_option_code(dashboard_code: str) -> tuple:
    """``MO2709C6600`` → (``MO2709``, ``C``, 6600.0)."""
    code = dashboard_code.strip()
    for i in range(len(code) - 1, 0, -1):
        if code[i] in "CP" and code[:i][-1].isdigit():
            return code[:i], code[i], float(code[i + 1:])
    raise StudyDataError(f"cannot parse option code {dashboard_code!r}")


def _row(rows: Dict[str, Dict[str, float]], code: str) -> Dict[str, float]:
    if code not in rows:
        raise StudyDataError(f"CFFEX CSV has no row {code}")
    return rows[code]


def build_snapshot(chain: Dict[str, Any], csv_payload: bytes, *, source_url: str) -> Dict[str, Any]:
    """Freeze the study's market data (decisions 1, 2, 10).

    The MO pillar is the dashboard's pair (same strike, same expiry) priced at
    the CFFEX settlement; the flat vol is the Black-76 implied vol of the MO
    settlement pair on the grid strike nearest the pillar forward, with the
    pillar's own forward and ``DF = exp(−r·T)``.
    """
    rows = parse_cffex_csv(csv_payload)
    syn = chain["synthetic"]
    series, _, strike = _cffex_option_code(syn["call_contract"])
    put_series, _, put_strike = _cffex_option_code(syn["put_contract"])
    if (series, strike) != (put_series, put_strike):
        raise StudyDataError(f"dashboard pair {syn['call_contract']}/{syn['put_contract']} is not one strike")
    k_label = f"{strike:g}"
    call = _row(rows, f"{series}-C-{k_label}")
    put = _row(rows, f"{series}-P-{k_label}")
    expiry = date.fromisoformat(syn["expiry"])
    T = year_fraction(expiry)
    df = float(safe_exp(-RATE * T))
    mo_forward = strike + (call["settle"] - put["settle"]) / df

    futures = []
    for f in chain["futures"]:
        cf = _row(rows, f["contract"])
        futures.append({**f, "close_cffex": cf["close"], "settle": cf["settle"]})

    strikes = sorted({float(c.rsplit("-", 1)[1]) for c in rows if c.startswith(f"{series}-C-")})
    k_atm = min(strikes, key=lambda k: abs(k - mo_forward))
    atm_label = f"{k_atm:g}"
    c_atm = _row(rows, f"{series}-C-{atm_label}")["settle"]
    p_atm = _row(rows, f"{series}-P-{atm_label}")["settle"]
    call_iv = implied_vol_black(c_atm, mo_forward, k_atm, T, df, True)
    put_iv = implied_vol_black(p_atm, mo_forward, k_atm, T, df, False)

    return {
        "valuation_date": VALUATION_DATE.isoformat(),
        "index_code": INDEX_CODE,
        "spot": float(chain["spot"]),
        "rate": RATE,
        "futures": futures,
        "option_pair": {
            "series": series, "strike": strike, "expiry": syn["expiry"],
            "call_contract": f"{series}-C-{k_label}", "put_contract": f"{series}-P-{k_label}",
            "call_settle": call["settle"], "put_settle": put["settle"],
            "call_close": call["close"], "put_close": put["close"],
        },
        "mo_forward": mo_forward,
        "vol": {"value": 0.5 * (call_iv + put_iv), "strike": k_atm,
                "call_iv": call_iv, "put_iv": put_iv, "forward": mo_forward,
                "method": "Black-76 on MO settlement, pillar forward, DF=exp(-rT)"},
        "dashboard_syn_fut": {"price": syn["price"], "contract": syn["contract"],
                              "revision_id": syn["revision_id"], "content_hash": syn["content_hash"],
                              "note": "dashboard synthetic forward from live mids, r=1.4%; cross-check only"},
        "cffex_csv": {"url": source_url, "sha256": hashlib.sha256(csv_payload).hexdigest(),
                      "bytes": len(csv_payload)},
        "sources": {
            "spot_and_im": "index-dashboard basis.sqlite3 latest_daily_basis (closes)",
            "mo_pair": "CFFEX daily statistics CSV, settlement (今结算)",
        },
    }


def cffex_url(d: date = VALUATION_DATE) -> str:
    return CFFEX_URL.format(yyyymm=d.strftime("%Y%m"), dd=d.strftime("%d"), yyyymmdd=d.strftime("%Y%m%d"))


def write_json(path: Path, payload: Any) -> None:
    """Deterministic JSON (sorted keys, 2-space indent, no NaN)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise StudyDataError(f"{path} missing; run the stage that writes it")
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    """The frozen market snapshot (``01_snapshot.py`` writes it)."""
    return read_json(SNAPSHOT_PATH if path is None else path)


# ---------------------------------------------------------------------------
# Term sheet (decision 8)
# ---------------------------------------------------------------------------

NOTIONAL = 10_000_000.0
KO_PCT = 1.00
KI_PCT = 0.75
LOCKOUT_MONTHS = 3
MATURITY_MONTHS = 24
#: |PV| tolerance of the coupon solve, as a fraction of notional (decision 9).
COUPON_PV_TOL_FRACTION = 1e-6
COUPON_PROBES = (0.05, 0.25)

#: Tail conventions (decision 6): study key → (CarryExtrapolation, short label).
TAILS: Dict[str, Tuple[CarryExtrapolation, str]] = {
    "hold_forward": (CarryExtrapolation.FLAT_FORWARD_CARRY, "hold forward carry"),
    "hold_yield": (CarryExtrapolation.FLAT_CARRY_YIELD, "hold zero carry yield"),
    "zero_carry": (CarryExtrapolation.ZERO_FORWARD_CARRY, "zero carry past last node"),
}
COUPON_TAIL = "hold_forward"

#: Mark variants for the sensitivity rows (decision 2): key → description.
MARKS: Dict[str, str] = {
    "base": "spot + IM closes, MO settlement",
    "mo_close": "spot + IM closes, MO close",
    "all_settle": "IM + MO settlements, spot close (mark-inconsistent)",
}

#: QUAD readout everywhere (decision 11): removes the legacy_linear spot staircase.
QUAD_READOUT = "transition"


def add_months(d: date, months: int) -> date:
    """Same day-of-month ``months`` later, clipped to the month end."""
    import calendar as _cal

    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, _cal.monthrange(year, month)[1]))


@dataclass(frozen=True)
class Terms:
    """The 2Y schedule on the SSE calendar, ACT/365F from the valuation date."""

    maturity_date: date
    maturity_years: float
    ko_dates: Tuple[date, ...]
    ko_times: Tuple[float, ...]
    ki_times: Tuple[float, ...]


def build_terms() -> Terms:
    """KO on month anniversaries 3..24 rolled to the next SSE trading day (last =
    maturity); KI on every SSE trading day in (valuation date, maturity]."""
    cal = create_calendar(CalendarType.CHINA_SSE)

    def is_trading(d: date) -> bool:
        return cal.is_business_day(datetime(d.year, d.month, d.day))

    def next_trading(d: date) -> date:
        while not is_trading(d):
            d += timedelta(days=1)
        return d

    maturity = next_trading(add_months(VALUATION_DATE, MATURITY_MONTHS))
    ko_dates = sorted({min(next_trading(add_months(VALUATION_DATE, m)), maturity)
                       for m in range(LOCKOUT_MONTHS, MATURITY_MONTHS + 1)})
    if ko_dates[-1] != maturity:
        raise StudyDataError("last KO observation must be the maturity date")
    ki_dates = []
    d = VALUATION_DATE + timedelta(days=1)
    while d <= maturity:
        if is_trading(d):
            ki_dates.append(d)
        d += timedelta(days=1)
    return Terms(
        maturity_date=maturity,
        maturity_years=year_fraction(maturity),
        ko_dates=tuple(ko_dates),
        ko_times=tuple(year_fraction(x) for x in ko_dates),
        ki_times=tuple(year_fraction(x) for x in ki_dates),
    )


def build_product(coupon: float, s0: float, terms: Optional[Terms] = None):
    """Standard snowball off ``s0`` (one unit = ``NOTIONAL``), booked numeric."""
    terms = terms or build_terms()
    return create_standard_snowball(
        initial_price=s0,
        strike=s0,
        maturity=terms.maturity_years,
        contract_multiplier=NOTIONAL / s0,
        ko_barrier=KO_PCT * s0,
        ko_rate=float(coupon),
        ki_barrier=KI_PCT * s0,
        num_observations=len(terms.ko_times),
        is_reverse=False,
        ko_observation_dates=list(terms.ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(terms.ki_times),
        rebate_rate=float(coupon),
        include_principal=False,
    )


# ---------------------------------------------------------------------------
# Market objects (decisions 3, 5, 6, 7)
# ---------------------------------------------------------------------------


def rate_curve() -> FlatRateCurve:
    return FlatRateCurve(RATE)


def carry_quotes(snap: Dict[str, Any], mark: str = "base") -> List[Any]:
    """Four IM ``FuturesQuote`` + one MO ``OptionPairQuote`` under a mark variant."""
    if mark not in MARKS:
        raise ValidationError(f"unknown mark {mark!r}")
    quotes: List[Any] = []
    for f in snap["futures"]:
        price = f["settle"] if mark == "all_settle" else f["price"]
        quotes.append(FuturesQuote(price=price, tenor_years=year_fraction(date.fromisoformat(f["expiry"])),
                                   label=f["contract"]))
    pair = snap["option_pair"]
    use_close = mark == "mo_close"
    quotes.append(OptionPairQuote(
        strike=pair["strike"],
        call_price=pair["call_close"] if use_close else pair["call_settle"],
        put_price=pair["put_close"] if use_close else pair["put_settle"],
        tenor_years=year_fraction(date.fromisoformat(pair["expiry"])),
        label=f"{pair['series']} C-P",
    ))
    return quotes


def pillar_labels(snap: Dict[str, Any]) -> List[str]:
    return [f["contract"] for f in snap["futures"]] + [f"{snap['option_pair']['series']} C-P"]


def lot_multipliers(snap: Dict[str, Any]) -> List[float]:
    """IM: 200 CNY/pt per lot; MO pillar: one C−P pair = 100·DF(T5) per point of F."""
    t5 = year_fraction(date.fromisoformat(snap["option_pair"]["expiry"]))
    return [IM_MULTIPLIER] * len(snap["futures"]) + [MO_MULTIPLIER * float(safe_exp(-RATE * t5))]


def build_curve(snap: Dict[str, Any], tail: str, mark: str = "base") -> CarryCurve:
    return build_carry_curve(snap["spot"], carry_quotes(snap, mark), rate_curve(), extrapolation=TAILS[tail][0])


def flat_q(curve: CarryCurve, maturity: float) -> float:
    """The flat model's yield: the curve's own q at the trade maturity (decision 7)."""
    return float(curve.to_dividend_yield(rate_curve()).get_yield(maturity))


def make_env(snap: Dict[str, Any], div_yield: DividendYield, spot: Optional[float] = None) -> PricingEnvironment:
    return PricingEnvironment(
        rate_curve=rate_curve(),
        valuation_date=datetime(VALUATION_DATE.year, VALUATION_DATE.month, VALUATION_DATE.day),
        spot_quote=SpotQuote(spot=snap["spot"] if spot is None else spot),
        vol_surface=FlatVolSurface(volatility=snap["vol"]["value"]),
        div_yield=div_yield,
    )


def curve_env(snap: Dict[str, Any], tail: str, mark: str = "base") -> PricingEnvironment:
    return make_env(snap, build_curve(snap, tail, mark).to_dividend_yield(rate_curve()))


def flat_env(snap: Dict[str, Any], tail: str, maturity: float, mark: str = "base") -> PricingEnvironment:
    return make_env(snap, ContinuousDividendYield(flat_q(build_curve(snap, tail, mark), maturity)))


def quad_engine(grid_points: Optional[int] = None) -> SnowballQuadEngine:
    kwargs: Dict[str, Any] = {"readout": QUAD_READOUT}
    if grid_points is not None:
        kwargs["grid_points"] = int(grid_points)
    return SnowballQuadEngine(params=QuadParams(**kwargs))


def resolved_quad_grid(engine: SnowballQuadEngine, product, env: PricingEnvironment) -> int:
    """The grid the QUAD engine actually uses (``max(requested, required)``)."""
    times = sorted(set(product.barrier_config.ko_observation_dates)
                   | set(product.barrier_config.ki_observation_dates))
    vol = float(env.get_vol(product.strike, product.maturity))
    return int(engine._resolve_grid_points(product.maturity, vol, times))  # noqa: SLF001 (diagnostic)


def pde_engine() -> PDEEngine:
    return PDEEngine(params=PDEParams(accuracy="high"))


# ---------------------------------------------------------------------------
# Coupon solve (decision 9) and event statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CouponSolution:
    coupon: float
    pv: float
    annuity: float  # dPV/dcoupon (PV is affine in the coupon)
    tolerance: float


def solve_coupon(pv_at: Callable[[float], float], notional: float = NOTIONAL) -> CouponSolution:
    """PV is affine in the coupon (no trigger reads it): two probes, one root, one
    re-price check against ``COUPON_PV_TOL_FRACTION·notional`` (fails closed)."""
    c0, c1 = COUPON_PROBES
    v0, v1 = float(pv_at(c0)), float(pv_at(c1))
    annuity = (v1 - v0) / (c1 - c0)
    if not annuity > 0.0:
        raise ValidationError(f"PV not increasing in the coupon (dPV/dc = {annuity})")
    coupon = c0 - v0 / annuity
    pv = float(pv_at(coupon))
    tol = COUPON_PV_TOL_FRACTION * notional
    if abs(pv) > tol:
        raise ValidationError(f"coupon {coupon:.6f} re-prices to {pv:.4f} > tol {tol:.4f}: PV not affine?")
    return CouponSolution(coupon=coupon, pv=pv, annuity=annuity, tolerance=tol)


def event_summary(stats, maturity: float) -> Dict[str, Any]:
    """P(KO), P(KI before any KO), P(KI and no KO), expected life from QUAD ``EventStatistics``.

    ``p_ki_before_ko`` counts a path only if it knocks in before any knock-out (the old
    "P(KI ever)" also counted knock-ins after a KO; that statistic no longer exists).
    """
    obs = stats.observations
    ko_p = np.asarray(obs.probability, dtype=float)[obs.is_ko_obs]
    ko_t = np.asarray(obs.times, dtype=float)[obs.is_ko_obs]
    p_ko = float(ko_p.sum())
    return {
        "p_ko_total": p_ko,
        "p_ko_first_year": float(ko_p[ko_t <= 1.0].sum()),
        "p_ki_before_ko": float(stats.knock_in.probability),
        "p_ki_survive": float(stats.knock_in.probability_no_ko),
        "expected_life_years": float(stats.expected_lifetime),
        "ko_probability_by_obs": [float(x) for x in ko_p],
        "ko_times": [float(x) for x in ko_t],
    }


def bp(value: float, notional: float = NOTIONAL) -> float:
    """CNY → basis points of notional."""
    return 1e4 * value / notional


def tail_leverage(curve: CarryCurve, maturity: float, h: float = 1e-6) -> List[float]:
    """``∂B(T_mat)/∂B_i`` per node (central difference; exact for the linear rules)."""
    out = []
    nodes = curve.nodes
    for i, (t, b) in enumerate(nodes):
        up = list(nodes)
        dn = list(nodes)
        up[i], dn[i] = (t, b + h), (t, b - h)
        out.append((float(curve.with_nodes(up).carry(maturity)) - float(curve.with_nodes(dn).carry(maturity))) / (2 * h))
    return out


def flat_forward_matches(curve: CarryCurve, q: float, maturity: float, spot: float) -> bool:
    """AC8: the flat model and the curve agree on F(T_mat)."""
    f_curve = float(curve.forward(spot, maturity))
    f_flat = spot * float(safe_exp((RATE - q) * maturity))
    return is_close(f_curve, f_flat, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Greeks helpers (decisions 12, 13)
# ---------------------------------------------------------------------------

#: Central yield step for rho (reported per +1%) and the spot step for deltas.
Q_BUMP = 1e-3
SPOT_BUMP_REL = 0.01
FORWARD_BUMP_REL = 1e-3
SCENARIO_SHOCK = 0.01
STEEPENER_SHOCK = 0.02


@dataclass(frozen=True)
class ForwardIntervalBumpDividendYield(DividendYield):
    """``q'(T)·T = q(T)·T + bump·|(0, T] ∩ (start, end]|``: the instantaneous
    forward yield moved by ``bump`` on ``(start, end]`` only (decision 13).

    A partition of ``(0, ∞)`` into intervals gives bumps that sum exactly to
    the parallel bump. Array-safe; entries at ``T <= 0`` pass through the base.
    """

    base: DividendYield
    start: float
    end: float  # math.inf allowed
    bump: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.start < self.end):
            raise ValidationError(f"need 0 <= start < end, got ({self.start}, {self.end})")

    def get_yield(self, time_to_maturity):  # type: ignore[override]
        t = np.asarray(time_to_maturity, dtype=float)
        base = np.asarray(self.base.get_yield(time_to_maturity), dtype=float)
        overlap = np.clip(np.minimum(t, self.end) - self.start, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            extra = np.where(t > 0.0, self.bump * overlap / np.where(t > 0.0, t, 1.0), 0.0)
        out = base + extra
        return float(out) if out.ndim == 0 else out


def interval_edges(curve: CarryCurve, maturity: float) -> List[float]:
    """``0, T_1, …, T_n, T_mat``: the forward-interval rho partition (the last
    interval is the unquoted tail)."""
    return [0.0] + list(curve.tenors) + [float(maturity)]


def node_yield_bumped(curve: CarryCurve, shocks: Dict[int, float]) -> CarryCurve:
    """Nodes ``B_i − shock_i·T_i`` (q at node i up by ``shock_i``)."""
    nodes = curve.nodes
    return curve.with_nodes([(t, b - shocks.get(i, 0.0) * t) for i, (t, b) in enumerate(nodes)])

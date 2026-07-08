"""Comprehensive bucketed Greeks API demo.

Run:
    python example/bucketed_greeks_consolidated_api_demo.py

This is a teaching example, similar in spirit to the ``example/mo_volmodels``
stages. It shows why bucketed Greeks matter and how the consolidated
``GreeksCalculator.calculate_bucketed_greeks()`` facade computes them.

Why bucketed Greeks matter:

1. Scalar spot delta does not say which futures tenor should hedge the trade.
   Futures bucket delta answers ``dPV / dF_i`` for each tradable futures mark.

2. Scalar dividend rho does not say which carry date drives the risk. Carry
   rhoq buckets answer ``dPV / dq_i`` on the implied futures-carry curve or on
   generic tenor intervals.

3. Scalar vega hides expiry concentration. Vol-tenor vega answers which
   maturity bucket drives PnL under a local vol bump.

4. LocalVol/Heston/SLV market-IV vega is not a direct flat-vol bump. The facade
   delegates that calculation to ``VolModelRiskCalculator`` so model-specific
   recalibration logic stays in one place.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from html import escape
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantark.asset.equity.engine.analytical.black_scholes_engine import (  # noqa: E402
    BlackScholesEngine,
)
from quantark.asset.equity.engine.pde import LocalVolPDESolver  # noqa: E402
from quantark.asset.equity.engine.quad import SnowballQuadEngine  # noqa: E402
from quantark.asset.equity.market import (  # noqa: E402
    IndexFuturesCurve,
    IndexFuturesQuote,
    bump_term_yield_node,
)
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.option import EuropeanVanillaOption  # noqa: E402
from quantark.asset.equity.product.option.snowball_helpers import (  # noqa: E402
    create_standard_snowball,
)
from quantark.asset.equity.report.term_structure import (  # noqa: E402
    BucketedVolSurface,
    default_tenor_buckets,
)
from quantark.asset.equity.riskmeasures import (  # noqa: E402
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
    BucketedGreeksRequest,
    GreeksCalculator,
)
from quantark.param import (  # noqa: E402
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    GridVolSurface,
    SpotQuote,
    TermStructureDividendYield,
)
from quantark.param.rrf import LinearRateCurve  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import OptionType  # noqa: E402
from quantark.volmodels.risk import MarketVegaRequest, SurfaceBump  # noqa: E402


HERE = Path(__file__).resolve().parent
MO_VOLMODELS_DATA = HERE / "mo_volmodels" / "data"
DEFAULT_HTML_OUTPUT = HERE / "data" / "bucketed_greeks_lecture_latest.html"
DEFAULT_NOTIONAL = 10_000_000.0
IM_FUTURES_MULTIPLIER = 200.0


def make_env(spot: float = 100.0, vol: float = 0.22, q: float = 0.01):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 8),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(q),
    )


def make_grid_env():
    env = make_env()
    env.vol_surface = GridVolSurface(
        strikes=[80.0, 90.0, 100.0, 110.0, 120.0],
        maturities=[0.10, 0.25, 0.50, 0.75, 1.00],
        iv_grid=np.array(
            [
                [0.285, 0.255, 0.225, 0.220, 0.230],
                [0.280, 0.250, 0.220, 0.215, 0.225],
                [0.275, 0.245, 0.215, 0.212, 0.222],
                [0.270, 0.240, 0.212, 0.210, 0.220],
                [0.268, 0.238, 0.210, 0.208, 0.218],
            ]
        ),
    )
    return env


def make_curve():
    return IndexFuturesCurve(
        underlying="IC",
        spot=100.0,
        quotes=[
            IndexFuturesQuote("IC00", maturity=0.10, price=100.10, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.25, price=100.70, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.50, price=101.60, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.75, price=102.30, multiplier=200.0),
        ],
    )


def load_mo_surface(tag: str = "latest") -> dict:
    path = MO_VOLMODELS_DATA / f"mo_iv_surface_{tag}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_im_futures_snapshot(tag: str = "latest") -> dict:
    path = MO_VOLMODELS_DATA / f"im_futures_{tag}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_mo_model_surface(surface: dict) -> dict:
    mo_dir = HERE / "mo_volmodels"
    mo_dir_text = str(mo_dir)
    if mo_dir_text not in sys.path:
        sys.path.insert(0, mo_dir_text)
    import _mo_common as mo_common

    return mo_common.prepare_model_surface(surface, iv_smoothing="sabr")


def make_mo_market_env(surface: dict) -> PricingEnvironment:
    per_expiry = surface["per_expiry"]
    valuation_date = datetime.fromisoformat(surface["fetched_at"]).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return PricingEnvironment(
        rate_curve=LinearRateCurve([(p["T"], p["r"]) for p in per_expiry]),
        valuation_date=valuation_date,
        spot_quote=SpotQuote(float(surface["s0"])),
        vol_surface=GridVolSurface(
            surface["strikes"],
            surface["maturities"],
            np.array(surface["iv_grid"]),
        ),
        div_yield=TermStructureDividendYield(
            times=[p["T"] for p in per_expiry],
            yields=[p["q"] for p in per_expiry],
        ),
    )


def make_im_futures_curve(surface: dict, futures_snapshot: dict) -> IndexFuturesCurve:
    spot = float(surface["s0"])
    quotes = [
        IndexFuturesQuote(
            contract=q["contract"],
            maturity=float(q["maturity"]),
            price=float(q["close"]),
            multiplier=float(q.get("multiplier", IM_FUTURES_MULTIPLIER)),
            expiry_date=datetime.fromisoformat(q["expiry_date"]),
        )
        for q in futures_snapshot["quotes"]
    ]
    return IndexFuturesCurve(
        underlying=futures_snapshot["underlying"]["futures_prefix"],
        spot=spot,
        quotes=quotes,
    )


def make_mo_forward_proxy_curve(surface: dict) -> IndexFuturesCurve:
    """Fallback only: real MO parity forwards, not listed IM close marks."""
    spot = float(surface["s0"])
    quotes = []
    for p in surface["per_expiry"]:
        expiry = datetime.fromisoformat(p["expiry_date"])
        quotes.append(
            IndexFuturesQuote(
                contract=f"IM{expiry:%y%m}",
                maturity=float(p["T"]),
                price=float(p["forward"]),
                multiplier=IM_FUTURES_MULTIPLIER,
                expiry_date=expiry,
            )
        )
    return IndexFuturesCurve(underlying="IM-proxy", spot=spot, quotes=quotes)


def make_market_snowball(
    s0: float,
    *,
    notional: float = DEFAULT_NOTIONAL,
    maturity: float = 2.0,
):
    return create_standard_snowball(
        initial_price=float(s0),
        strike=round(float(s0), 2),
        maturity=float(maturity),
        contract_multiplier=float(notional) / float(s0),
        ko_barrier=round(1.03 * float(s0), 2),
        ko_rate=0.12,
        ki_barrier=round(0.75 * float(s0), 2),
        num_observations=24,
        is_reverse=False,
        include_principal=False,
    )


def make_vanilla():
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=0.70,
        contract_multiplier=1.0,
    )


def make_snowball():
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=6,
        is_reverse=False,
        include_principal=False,
    )


def _env_with_div(env, div_yield):
    out = deepcopy(env)
    out.div_yield = div_yield
    return out


def _env_with_vol(env, vol_surface):
    out = deepcopy(env)
    out.vol_surface = vol_surface
    return out


def _print_section(title: str, why: str) -> None:
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")
    print(why)


def _print_table(title: str, headers: list[str], rows: list[list[object]]) -> None:
    print(f"\n{title}")
    text_rows = [[_format_cell(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in text_rows))
        for i, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(row[i].rjust(widths[i]) for i in range(len(headers))))


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:+.8f}"
    return str(value)


def _html_number(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.8f}"
    return escape(str(value))


def _fmt_level(value: float, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}"


def _fmt_money(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"CNY {float(value) / 1_000_000:,.3f}m"
    if abs_value >= 1_000:
        return f"CNY {float(value):,.0f}"
    return f"CNY {float(value):,.2f}"


def _fmt_signed_money(value: float) -> str:
    sign = "+" if float(value) >= 0.0 else "-"
    return f"{sign}{_fmt_money(abs(float(value)))}"


def _fmt_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):+,.{digits}f}%"


def _fmt_plain_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):,.{digits}f}%"


def _fmt_num(value: float, digits: int = 4) -> str:
    return f"{float(value):+,.{digits}f}"


def _html_kv_table(rows: list[tuple[str, object]]) -> str:
    body = "".join(
        "<tr>"
        f"<th>{escape(str(label))}</th>"
        f"<td>{escape(str(value))}</td>"
        "</tr>"
        for label, value in rows
    )
    return f"<table class='kv'><tbody>{body}</tbody></table>"


def _html_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = []
        for value in row:
            css = " class='num'" if isinstance(value, float) else ""
            cells.append(f"<td{css}>{_html_number(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _html_point_table(points: tuple[BucketedGreekPoint, ...]) -> str:
    return _html_table(
        ["name", "coordinate", "bucket/contract", "mode", "reported", "derivative", "pnl"],
        [
            [
                point.name,
                point.coordinate.value,
                point.contract or point.bucket or "",
                point.difference_mode,
                point.reported,
                point.derivative,
                point.pnl,
            ]
            for point in points
        ],
    )


def _active_point(points: tuple[BucketedGreekPoint, ...]) -> BucketedGreekPoint:
    return max(points, key=lambda point: abs(float(point.derivative or 0.0)))


def _manual_futures_delta_central(product, env, engine, curve, contract, bump):
    up_curve = curve.bump_contract(contract, bump)
    down_curve = curve.bump_contract(contract, -bump)
    up_env = _env_with_div(env, up_curve.to_dividend_yield_curve(env.rate_curve))
    down_env = _env_with_div(env, down_curve.to_dividend_yield_curve(env.rate_curve))
    up_price = engine.price(product, up_env)
    down_price = engine.price(product, down_env)
    return (up_price - down_price) / (2.0 * bump), up_price, down_price


def _manual_implied_carry_rhoq_central(product, env, engine, curve, node_index, bump):
    base_div = curve.to_dividend_yield_curve(env.rate_curve)
    up_env = _env_with_div(env, bump_term_yield_node(base_div, node_index, bump))
    down_env = _env_with_div(env, bump_term_yield_node(base_div, node_index, -bump))
    up_price = engine.price(product, up_env)
    down_price = engine.price(product, down_env)
    derivative = (up_price - down_price) / (2.0 * bump)
    return derivative * 0.01, derivative, up_price, down_price


def _manual_vol_tenor_vega_central(product, env, engine, bucket, bump):
    up_env = _env_with_vol(
        env,
        BucketedVolSurface(
            base=env.vol_surface,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=bump,
        ),
    )
    down_env = _env_with_vol(
        env,
        BucketedVolSurface(
            base=env.vol_surface,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=-bump,
        ),
    )
    up_price = engine.price(product, up_env)
    down_price = engine.price(product, down_env)
    derivative = (up_price - down_price) / (2.0 * bump)
    return derivative * 0.01, derivative, up_price, down_price


def stage_scalar_context(calc, product, env, engine):
    _print_section(
        "Stage 1 - Scalar Greeks: useful but not enough",
        "A scalar Greek is a single number. It is fine for first-line risk, but it "
        "does not identify which futures tenor, carry node, or vol expiry drives the PnL.",
    )
    greeks = calc.calculate(
        product,
        env,
        engine,
        greeks=("price", "delta", "vega", "dividend_rho"),
    )
    _print_table(
        "Scalar output",
        ["metric", "value", "meaning"],
        [
            ["price", greeks["price"], "base PV"],
            ["spot delta", greeks["delta"], "dPV / dSpot"],
            ["flat vega", greeks["vega"], "PV per +1 vol point"],
            ["dividend rhoq", greeks["dividend_rho"], "PV per +1% flat carry"],
        ],
    )
    print(
        "\nThe next stages split those scalar-looking risks into tradable or "
        "market-observable coordinates."
    )


def stage_futures_and_carry(calc, product, env, engine, curve):
    _print_section(
        "Stage 2 - Futures bucket delta and implied-carry rhoq",
        "The option is priced from the futures marks by first converting the curve "
        "to an implied dividend/carry term structure. A futures bucket delta bumps "
        "one quoted futures mark. A carry rhoq bucket bumps one implied q(T_i) node.",
    )
    request = BucketedGreeksRequest(
        futures_curve=curve,
        futures_price_bump=0.50,
        carry_bump=0.0001,
        difference_mode_overrides={
            BucketedGreekCoordinate.FUTURES_DELTA: BucketedGreekDifferenceMode.CENTRAL,
            BucketedGreekCoordinate.CARRY_RHOQ: BucketedGreekDifferenceMode.CENTRAL,
        },
    )
    result = calc.calculate_bucketed_greeks(product, env, engine, request)
    delta_points = result.by_coordinate(BucketedGreekCoordinate.FUTURES_DELTA)
    rhoq_points = result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)

    _print_table(
        "Futures delta buckets",
        ["contract", "T", "F", "dPV/dF", "PnL@bump", "hedge hands"],
        [
            [
                p.contract,
                p.maturity,
                p.future_price,
                p.derivative,
                p.pnl,
                p.hedge_hands,
            ]
            for p in delta_points
        ],
    )
    _print_table(
        "Implied-carry rhoq buckets",
        ["contract", "T", "rhoq +1%", "raw dPV/dq", "PnL@bump"],
        [
            [p.contract, p.maturity, p.reported, p.derivative, p.pnl]
            for p in rhoq_points
        ],
    )

    active_delta = _active_point(delta_points)
    manual_delta, up_price, down_price = _manual_futures_delta_central(
        product,
        env,
        engine,
        curve,
        active_delta.contract,
        active_delta.bump_size,
    )
    active_rhoq = _active_point(rhoq_points)
    node_index = [q.contract for q in curve.quotes].index(active_rhoq.contract)
    manual_rhoq, raw_rhoq, rhoq_up, rhoq_down = _manual_implied_carry_rhoq_central(
        product,
        env,
        engine,
        curve,
        node_index,
        active_rhoq.bump_size,
    )
    _print_table(
        "Manual finite-difference checks",
        ["coordinate", "bucket", "API", "manual", "up PV", "down PV"],
        [
            [
                "dPV/dF",
                active_delta.contract,
                active_delta.derivative,
                manual_delta,
                up_price,
                down_price,
            ],
            [
                "rhoq +1%",
                active_rhoq.contract,
                active_rhoq.reported,
                manual_rhoq,
                rhoq_up,
                rhoq_down,
            ],
        ],
    )
    print(
        "\nRisk-management point: hedge hands are per contract bucket. A scalar "
        "spot delta cannot tell you whether the exposure belongs in IC00, IC01, "
        "IC02, or IC03."
    )


def stage_generic_tenor(calc, product, env, engine):
    _print_section(
        "Stage 3 - Generic vol-tenor vega and carry buckets",
        "Without a futures curve, the facade can still bucket risk by standard "
        "tenor intervals. Vol-tenor vega defaults to central differences. Carry "
        "rhoq keeps the historical one-sided-up default unless overridden.",
    )
    request = BucketedGreeksRequest(
        coordinates=(
            BucketedGreekCoordinate.VOL_TENOR_VEGA,
            BucketedGreekCoordinate.CARRY_RHOQ,
        ),
        vol_bump=0.01,
        carry_bump=0.0001,
    )
    result = calc.calculate_bucketed_greeks(product, env, engine, request)
    vega_points = result.by_coordinate(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    carry_points = result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)

    _print_table(
        "Generic tenor buckets",
        ["bucket", "coordinate", "mode", "reported", "raw derivative"],
        [
            [p.bucket, p.coordinate.value, p.difference_mode, p.reported, p.derivative]
            for p in (*vega_points, *carry_points)
        ],
    )
    active_vega = _active_point(vega_points)
    buckets = {
        f"{bucket.label} ({bucket.start:.3g}-{bucket.end:.3g}y)": bucket
        for bucket in default_tenor_buckets(product.get_maturity(env))
    }
    bucket = buckets[active_vega.bucket]
    manual_reported, raw_derivative, up_price, down_price = (
        _manual_vol_tenor_vega_central(
            product, env, engine, bucket, active_vega.bump_size
        )
    )
    _print_table(
        "Manual vol-tenor check",
        ["bucket", "API reported", "manual reported", "raw dPV/dVol", "up PV", "down PV"],
        [
            [
                active_vega.bucket,
                active_vega.reported,
                manual_reported,
                raw_derivative,
                up_price,
                down_price,
            ]
        ],
    )


def stage_localvol_market_vega(calc, product):
    _print_section(
        "Stage 4 - Market-IV vega delegates through the vol-model risk backend",
        "LocalVol market vega is a sticky-strike IV quote bump followed by a "
        "local-vol rebuild. Heston and SLV use their own calibration artifacts. "
        "That is why this coordinate delegates to VolModelRiskCalculator instead "
        "of directly bumping pricing_env.vol_surface.",
    )
    env = make_grid_env()
    engine = LocalVolPDESolver(params=PDEParams(grid_size=100, time_steps=50))
    request = BucketedGreeksRequest(
        coordinates=(BucketedGreekCoordinate.MARKET_IV_VEGA,),
        market_vega_request=MarketVegaRequest(
            surface_bumps=(
                SurfaceBump.parallel(0.01),
                SurfaceBump.maturity_row(2, 0.01),
                SurfaceBump.strike_column(2, 0.01),
            )
        ),
    )
    result = calc.calculate_bucketed_greeks(product, env, engine, request)
    _print_table(
        "LocalVol market-IV vega buckets",
        ["scenario", "model", "mode", "reported +1 vol pt", "raw derivative"],
        [
            [p.name, p.model, p.difference_mode, p.reported, p.derivative]
            for p in result.points
        ],
    )


def stage_structured_product(calc, env):
    _print_section(
        "Stage 5 - Structured payoff profile: why buckets are more actionable",
        "A path-dependent autocallable can have risk before final maturity. The "
        "same API produces a bucket profile that reports where the vega and carry "
        "risk lives across tenor intervals.",
    )
    snowball = make_snowball()
    result = calc.calculate_bucketed_greeks(
        snowball,
        env,
        SnowballQuadEngine(),
        BucketedGreeksRequest(
            coordinates=(
                BucketedGreekCoordinate.VOL_TENOR_VEGA,
                BucketedGreekCoordinate.CARRY_RHOQ,
            ),
            vol_bump=0.01,
            carry_bump=0.0001,
            difference_mode_overrides={
                BucketedGreekCoordinate.VOL_TENOR_VEGA: (
                    BucketedGreekDifferenceMode.ONE_SIDED_UP
                ),
                BucketedGreekCoordinate.CARRY_RHOQ: (
                    BucketedGreekDifferenceMode.ONE_SIDED_UP
                ),
            },
        ),
    )
    vega = {
        p.bucket: p for p in result.by_coordinate(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    }
    carry = {
        p.bucket: p for p in result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
    }
    _print_table(
        "Snowball bucket profile",
        ["bucket", "vega +1 vol pt", "vega raw", "rhoq +1%", "rhoq raw"],
        [
            [
                bucket,
                vega[bucket].reported,
                vega[bucket].derivative,
                carry[bucket].reported,
                carry[bucket].derivative,
            ]
            for bucket in vega
        ],
    )
    print(
        "\nRisk-management point: this table tells a hedger which vol/carry "
        "maturities matter. That is the missing information in one scalar vega "
        "or one scalar dividend rho."
    )


def render_market_snowball_html_report(
    calc,
    *,
    tag: str = "latest",
    include_localvol: bool = True,
) -> str:
    surface = load_mo_surface(tag)
    env = make_mo_market_env(surface)
    try:
        futures_snapshot = load_im_futures_snapshot(tag)
        curve = make_im_futures_curve(surface, futures_snapshot)
        curve_source = (
            "real listed IM futures closes from AKShare/Sina, aligned to the MO "
            "option snapshot date"
        )
        futures_date = futures_snapshot["valuation_date"]
    except FileNotFoundError:
        futures_snapshot = None
        curve = make_mo_forward_proxy_curve(surface)
        curve_source = (
            "MO put-call parity forwards used as an IM futures proxy because no "
            "listed IM snapshot file was found"
        )
        futures_date = surface["fetched_at"].split("T", 1)[0]

    pricing_env = deepcopy(env)
    pricing_env.div_yield = curve.to_dividend_yield_curve(env.rate_curve)

    s0 = float(surface["s0"])
    notional = DEFAULT_NOTIONAL
    snowball = make_market_snowball(s0, notional=notional)
    engine = SnowballQuadEngine()
    request = BucketedGreeksRequest(
        coordinates=(
            BucketedGreekCoordinate.FUTURES_DELTA,
            BucketedGreekCoordinate.CARRY_RHOQ,
            BucketedGreekCoordinate.VOL_TENOR_VEGA,
        ),
        futures_curve=curve,
        futures_price_bump=5.0,
        carry_bump=0.0001,
        vol_bump=0.01,
        difference_mode_overrides={
            BucketedGreekCoordinate.FUTURES_DELTA: (
                BucketedGreekDifferenceMode.CENTRAL
            ),
            BucketedGreekCoordinate.CARRY_RHOQ: (
                BucketedGreekDifferenceMode.CENTRAL
            ),
            BucketedGreekCoordinate.VOL_TENOR_VEGA: (
                BucketedGreekDifferenceMode.CENTRAL
            ),
        },
    )
    result = calc.calculate_bucketed_greeks(snowball, pricing_env, engine, request)
    delta_points = result.by_coordinate(BucketedGreekCoordinate.FUTURES_DELTA)
    rhoq_points = result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
    vega_points = result.by_coordinate(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    base_price = float(result.points[0].base_price)
    price_pct_notional = base_price / notional

    active_delta = _active_point(delta_points)
    manual_delta, delta_up, delta_down = _manual_futures_delta_central(
        snowball,
        pricing_env,
        engine,
        curve,
        active_delta.contract,
        active_delta.bump_size,
    )
    active_rhoq = _active_point(rhoq_points)
    node_index = [q.contract for q in curve.quotes].index(active_rhoq.contract)
    manual_rhoq, raw_rhoq, rhoq_up, rhoq_down = _manual_implied_carry_rhoq_central(
        snowball,
        pricing_env,
        engine,
        curve,
        node_index,
        active_rhoq.bump_size,
    )
    active_vega = _active_point(vega_points)
    buckets = {
        f"{bucket.label} ({bucket.start:.3g}-{bucket.end:.3g}y)": bucket
        for bucket in default_tenor_buckets(snowball.get_maturity(pricing_env))
    }
    manual_vega, raw_vega, vega_up, vega_down = _manual_vol_tenor_vega_central(
        snowball,
        pricing_env,
        engine,
        buckets[active_vega.bucket],
        active_vega.bump_size,
    )

    term_sheet = _html_kv_table(
        [
            ("Underlying", f"CSI 1000 index, 000852.SH, spot {_fmt_level(s0)}"),
            ("Hedge instrument", "CFFEX IM CSI 1000 index futures"),
            ("Notional", _fmt_money(notional)),
            ("Tenor", "2.0 years"),
            ("KO schedule", "24 monthly observations, flat 103% barrier"),
            ("KO barrier", f"{_fmt_level(snowball.barrier_config.ko_barrier)}"),
            ("Coupon on KO", _fmt_plain_pct(snowball.barrier_config.ko_rate) + " p.a."),
            ("KI monitoring", "continuous, 75% barrier"),
            ("KI barrier", f"{_fmt_level(snowball.barrier_config.ki_barrier)}"),
            ("Payoff scaling", "principal excluded; option-leg PV scaled to notional"),
            ("Pricing model", "SnowballQuadEngine with MO implied-vol surface"),
        ]
    )
    market_table = _html_table(
        ["expiry", "T", "MO forward", "implied r", "implied q", "OTM strikes"],
        [
            [
                p["expiry_date"],
                f"{p['T']:.3f}y",
                _fmt_level(p["forward"]),
                _fmt_pct(p["r"]),
                _fmt_pct(p["q"]),
                len(p["points"]),
            ]
            for p in surface["per_expiry"]
        ],
    )
    futures_table = _html_table(
        ["contract", "expiry", "T", "close", "basis vs spot", "volume", "open interest"],
        [
            [
                q.contract,
                q.expiry_date.strftime("%Y-%m-%d") if q.expiry_date else "",
                f"{q.maturity:.3f}y",
                _fmt_level(q.price),
                _fmt_pct(q.price / s0 - 1.0),
                (
                    futures_snapshot["quotes"][i].get("volume", "")
                    if futures_snapshot is not None
                    else ""
                ),
                (
                    futures_snapshot["quotes"][i].get("open_interest", "")
                    if futures_snapshot is not None
                    else ""
                ),
            ]
            for i, q in enumerate(curve.quotes)
        ],
    )
    delta_table = _html_table(
        ["IM contract", "T", "F close", "dPV/dF", "hedge hands", "PnL @ +5pt"],
        [
            [
                p.contract,
                f"{p.maturity:.3f}y",
                _fmt_level(p.future_price),
                _fmt_signed_money(p.derivative),
                _fmt_num(p.hedge_hands, 3),
                _fmt_signed_money(p.pnl),
            ]
            for p in delta_points
        ],
    )
    rhoq_table = _html_table(
        ["carry node", "T", "rhoq +1%", "raw dPV/dq", "PnL @ +1bp"],
        [
            [
                p.contract,
                f"{p.maturity:.3f}y",
                _fmt_signed_money(p.reported),
                _fmt_signed_money(p.derivative),
                _fmt_signed_money(p.pnl),
            ]
            for p in rhoq_points
        ],
    )
    vega_table = _html_table(
        ["vol bucket", "mode", "vega +1 vol pt", "raw dPV/dsigma", "PnL @ +1vol"],
        [
            [
                p.bucket,
                p.difference_mode,
                _fmt_signed_money(p.reported),
                _fmt_signed_money(p.derivative),
                _fmt_signed_money(p.pnl),
            ]
            for p in vega_points
        ],
    )
    manual_table = _html_table(
        ["coordinate", "bucket", "API", "manual", "PV up", "PV down"],
        [
            [
                "futures delta",
                active_delta.contract,
                _fmt_signed_money(active_delta.derivative),
                _fmt_signed_money(manual_delta),
                _fmt_money(delta_up),
                _fmt_money(delta_down),
            ],
            [
                "carry rhoq +1%",
                active_rhoq.contract,
                _fmt_signed_money(active_rhoq.reported),
                _fmt_signed_money(manual_rhoq),
                _fmt_money(rhoq_up),
                _fmt_money(rhoq_down),
            ],
            [
                "vol vega +1pt",
                active_vega.bucket,
                _fmt_signed_money(active_vega.reported),
                _fmt_signed_money(manual_vega),
                _fmt_money(vega_up),
                _fmt_money(vega_down),
            ],
        ],
    )

    model_section = ""
    if include_localvol:
        try:
            model_surface = prepare_mo_model_surface(surface)
            model_env = make_mo_market_env(model_surface)
            vanilla = EuropeanVanillaOption(
                strike=round(s0, 2),
                option_type=OptionType.CALL,
                maturity=float(model_surface["maturities"][-1]),
                contract_multiplier=notional / s0,
            )
            lv_engine = LocalVolPDESolver(params=PDEParams(grid_size=90, time_steps=45))
            lv_result = calc.calculate_bucketed_greeks(
                vanilla,
                model_env,
                lv_engine,
                BucketedGreeksRequest(
                    coordinates=(BucketedGreekCoordinate.MARKET_IV_VEGA,),
                    market_vega_request=MarketVegaRequest(
                        surface_bumps=(
                            SurfaceBump.parallel(0.01),
                            SurfaceBump.maturity_row(2, 0.01),
                            SurfaceBump.maturity_row(5, 0.01),
                        )
                    ),
                ),
            )
            localvol_rows = [
                [
                    p.name,
                    p.model or "LocalVol",
                    p.difference_mode,
                    p.status,
                    _fmt_signed_money(p.reported) if p.reported is not None else "",
                    _fmt_signed_money(p.derivative)
                    if p.derivative is not None
                    else escape(p.error or ""),
                ]
                for p in lv_result.points
            ]
            localvol_table = _html_table(
                [
                    "scenario",
                    "model",
                    "mode",
                    "status",
                    "reported +1vol",
                    "raw derivative",
                ],
                localvol_rows,
            )
        except Exception as exc:  # pragma: no cover - example robustness
            localvol_table = (
                "<div class='note error'>LocalVol market-IV vega demo failed: "
                f"{escape(str(exc))}</div>"
            )

        model_facts = []
        for file_name, label in (
            (f"mo_calib_heston_{tag}.json", "Heston"),
            (f"mo_reprice_slv_{tag}.json", "Heston-SLV"),
        ):
            path = MO_VOLMODELS_DATA / file_name
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rmse = payload.get("overall_rmse_iv")
            if rmse is not None:
                model_facts.append([label, f"{100.0 * rmse:.2f} vol pts"])
        model_fit_table = (
            _html_table(["model artifact", "vanilla IV RMSE"], model_facts)
            if model_facts
            else ""
        )
        model_section = f"""
<section id="model">
  <p class="eyebrow">Model-aware bucket input</p>
  <h2>Keeping LocalVol, Heston, and SLV inputs distinct</h2>
  <p>The consolidated API does not pretend that every vega is the same bump.
  <code>VOL_TENOR_VEGA</code> directly bumps the pricing environment surface;
  <code>MARKET_IV_VEGA</code> delegates to the volatility-model risk backend, so
  LocalVol can rebuild a Dupire surface and Heston/SLV can use their calibration
  artifacts.</p>
  {localvol_table}
  {model_fit_table}
</section>
"""

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_date = surface["fetched_at"].split("T", 1)[0]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bucketed Greeks for a 000852.SH Snowball</title>
<style>
:root {{
  color-scheme: light;
  --paper: #f5f7f4;
  --ink: #17211f;
  --muted: #5d6964;
  --line: #cfd8d1;
  --panel: #ffffff;
  --panel2: #eef4f0;
  --teal: #006d77;
  --red: #b23a48;
  --amber: #9b6a08;
  --blue: #3559a6;
  --shadow: 0 18px 50px rgba(23, 33, 31, .08);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  background:
    linear-gradient(90deg, rgba(0,109,119,.07) 0 1px, transparent 1px 100%),
    linear-gradient(180deg, rgba(0,109,119,.06) 0 1px, transparent 1px 100%),
    var(--paper);
  background-size: 44px 44px;
  color: var(--ink);
  font-family: "Avenir Next", "Gill Sans", "Segoe UI", sans-serif;
  line-height: 1.55;
}}
a {{ color: var(--teal); }}
.wrap {{
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 28px 70px;
}}
header.hero {{
  min-height: 78vh;
  display: grid;
  grid-template-columns: minmax(0, 1.12fr) minmax(340px, .88fr);
  gap: 34px;
  align-items: center;
  border-bottom: 1px solid var(--line);
  padding: 22px 0 34px;
}}
.eyebrow {{
  margin: 0 0 10px;
  color: var(--teal);
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .13em;
  text-transform: uppercase;
}}
h1, h2, h3 {{
  font-family: Georgia, "Times New Roman", serif;
  letter-spacing: 0;
  line-height: 1.08;
}}
h1 {{
  max-width: 900px;
  margin: 0;
  font-size: 4.25rem;
  font-weight: 700;
}}
h2 {{
  margin: 0 0 14px;
  font-size: 2.15rem;
}}
h3 {{
  margin: 22px 0 8px;
  font-size: 1.28rem;
}}
p {{ max-width: 880px; margin: 0 0 14px; }}
.lede {{
  margin-top: 20px;
  color: var(--muted);
  font-size: 1.14rem;
  max-width: 760px;
}}
.market-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 24px;
}}
.chip {{
  border: 1px solid var(--line);
  background: rgba(255,255,255,.7);
  padding: 8px 11px;
  border-radius: 999px;
  font-size: .9rem;
}}
.hero-panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
  border-radius: 8px;
  padding: 22px;
}}
.price-big {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 4px;
  padding-bottom: 18px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--line);
}}
.price-big span {{ color: var(--muted); font-size: .82rem; text-transform: uppercase; }}
.price-big strong {{
  font-size: 2.55rem;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}}
.hero-metrics {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}}
.metric {{
  border: 1px solid var(--line);
  background: var(--panel2);
  border-radius: 7px;
  padding: 12px;
  min-height: 80px;
}}
.metric small {{
  display: block;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 700;
  font-size: .68rem;
}}
.metric b {{
  display: block;
  margin-top: 5px;
  font-size: 1.22rem;
  font-variant-numeric: tabular-nums;
}}
section {{
  padding: 42px 0;
  border-bottom: 1px solid var(--line);
}}
.section-grid {{
  display: grid;
  grid-template-columns: minmax(250px, .72fr) minmax(0, 1.28fr);
  gap: 28px;
  align-items: start;
}}
.rail {{
  position: sticky;
  top: 18px;
}}
.rail p {{ color: var(--muted); }}
.panel {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 20px;
  box-shadow: var(--shadow);
}}
.timeline {{
  margin: 22px 0 8px;
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 8px;
  padding: 18px;
}}
.track {{
  position: relative;
  height: 11px;
  background: #dfe8e2;
  border-radius: 999px;
  margin: 28px 8px 22px;
}}
.track::before, .track::after {{
  position: absolute;
  top: -26px;
  color: var(--muted);
  font-size: .82rem;
}}
.track::before {{ content: "trade date"; left: 0; }}
.track::after {{ content: "2Y maturity"; right: 0; }}
.dot {{
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  background: var(--teal);
  border: 2px solid #fff;
  box-shadow: 0 0 0 1px var(--teal);
}}
.barrier-map {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-top: 14px;
}}
.barrier {{
  border-left: 4px solid var(--teal);
  background: #f8fbf9;
  padding: 10px 12px;
  min-height: 78px;
}}
.barrier.ki {{ border-color: var(--red); }}
.barrier.carry {{ border-color: var(--amber); }}
.barrier b {{ display: block; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 22px;
  font-size: .9rem;
  background: var(--panel);
}}
th {{
  text-align: left;
  background: #edf2ee;
  color: #26322e;
  border: 1px solid var(--line);
  padding: 9px 10px;
  font-weight: 800;
}}
td {{
  border: 1px solid var(--line);
  padding: 9px 10px;
  vertical-align: top;
}}
td.num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
table.kv th {{
  width: 210px;
  background: #f3f6f2;
}}
code {{
  font-family: "SFMono-Regular", Consolas, monospace;
  background: #edf2ee;
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1px 5px;
  font-size: .92em;
}}
.note {{
  border-left: 4px solid var(--teal);
  background: #edf7f5;
  padding: 13px 15px;
  margin: 14px 0 22px;
  border-radius: 0 8px 8px 0;
}}
.note.warn {{ border-color: var(--amber); background: #fff7e3; }}
.note.error {{ border-color: var(--red); background: #fff0f2; }}
.formula {{
  font-family: "SFMono-Regular", Consolas, monospace;
  background: #17211f;
  color: #edf7f5;
  border-radius: 8px;
  padding: 14px 16px;
  overflow-x: auto;
}}
.toc {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 22px;
}}
.toc a {{
  color: var(--ink);
  text-decoration: none;
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 999px;
  padding: 7px 11px;
  font-size: .88rem;
}}
footer {{
  color: var(--muted);
  font-size: .86rem;
  padding-top: 24px;
}}
@media (max-width: 900px) {{
  .wrap {{ padding: 22px 16px 46px; }}
  header.hero, .section-grid {{ grid-template-columns: 1fr; }}
  header.hero {{ min-height: auto; }}
  h1 {{ font-size: 2.75rem; }}
  .rail {{ position: static; }}
  .barrier-map, .hero-metrics {{ grid-template-columns: 1fr; }}
  table {{ display: block; overflow-x: auto; white-space: nowrap; }}
}}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <div>
    <p class="eyebrow">Bucketed Greeks lecture</p>
    <h1>2Y monthly KO Snowball on 000852.SH, hedged with IM futures</h1>
    <p class="lede">The goal is not to produce one impressive Greek. The goal is
    to locate the risk on the tradable market coordinates: which IM futures
    contract, which carry node, and which volatility tenor bucket moves the
    Snowball PnL.</p>
    <div class="market-strip">
      <span class="chip">MO snapshot {escape(surface["fetched_at"])}</span>
      <span class="chip">IM futures date {escape(futures_date)}</span>
      <span class="chip">spot {_fmt_level(s0)}</span>
      <span class="chip">notional {_fmt_money(notional)}</span>
    </div>
    <nav class="toc">
      <a href="#terms">Product terms</a>
      <a href="#market">Market data</a>
      <a href="#buckets">Bucket Greeks</a>
      <a href="#checks">Finite-difference checks</a>
      <a href="#model">Vol-model inputs</a>
    </nav>
  </div>
  <aside class="hero-panel">
    <div class="price-big">
      <span>base option-leg PV</span>
      <strong>{_fmt_signed_money(base_price)}</strong>
      <span>{_fmt_pct(price_pct_notional)} of notional, principal excluded</span>
    </div>
    <div class="hero-metrics">
      <div class="metric"><small>Largest IM hedge</small><b>{active_delta.contract}: {_fmt_num(active_delta.hedge_hands, 2)}</b></div>
      <div class="metric"><small>Largest carry node</small><b>{active_rhoq.contract}</b></div>
      <div class="metric"><small>Largest vol bucket</small><b>{escape(active_vega.bucket or "")}</b></div>
      <div class="metric"><small>Difference mode</small><b>central</b></div>
    </div>
  </aside>
</header>

<section id="terms">
  <div class="section-grid">
    <div class="rail">
      <p class="eyebrow">Step 1</p>
      <h2>Explain the product before the Greeks</h2>
      <p>A Snowball is path-dependent. KO dates, KI monitoring, barrier levels,
      coupon convention, principal treatment, and notional scaling all change
      the meaning of the risk table.</p>
    </div>
    <div>
      <div class="panel">{term_sheet}</div>
      <div class="timeline">
        <div class="track">
          <span class="dot" style="left:4.17%"></span><span class="dot" style="left:8.33%"></span>
          <span class="dot" style="left:12.5%"></span><span class="dot" style="left:16.67%"></span>
          <span class="dot" style="left:20.83%"></span><span class="dot" style="left:25%"></span>
          <span class="dot" style="left:29.17%"></span><span class="dot" style="left:33.33%"></span>
          <span class="dot" style="left:37.5%"></span><span class="dot" style="left:41.67%"></span>
          <span class="dot" style="left:45.83%"></span><span class="dot" style="left:50%"></span>
          <span class="dot" style="left:54.17%"></span><span class="dot" style="left:58.33%"></span>
          <span class="dot" style="left:62.5%"></span><span class="dot" style="left:66.67%"></span>
          <span class="dot" style="left:70.83%"></span><span class="dot" style="left:75%"></span>
          <span class="dot" style="left:79.17%"></span><span class="dot" style="left:83.33%"></span>
          <span class="dot" style="left:87.5%"></span><span class="dot" style="left:91.67%"></span>
          <span class="dot" style="left:95.83%"></span><span class="dot" style="left:100%"></span>
        </div>
        <div class="barrier-map">
          <div class="barrier"><b>Monthly KO</b>Any observation above 103% redeems and pays accrued coupon.</div>
          <div class="barrier ki"><b>Continuous KI</b>Any path touch below 75% changes terminal downside.</div>
          <div class="barrier carry"><b>Principal excluded</b>PV and Greeks shown for the option leg scaled to notional.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<section id="market">
  <div class="section-grid">
    <div class="rail">
      <p class="eyebrow">Step 2</p>
      <h2>Use real 000852.SH and IM market marks</h2>
      <p>The volatility surface comes from MO options on CSI 1000. The hedge
      curve uses listed IM futures closes on the same market date. That lets
      carry risk and futures delta be reported against the instruments a desk
      can actually trade.</p>
    </div>
    <div>
      <h3>MO parity surface inputs</h3>
      {market_table}
      <h3>IM futures hedge curve</h3>
      {futures_table}
      <div class="note warn"><b>Curve source.</b> {escape(curve_source)}. The
      Snowball is 2Y while the listed IM curve here reaches {curve.quotes[-1].maturity:.3f}y,
      so the last carry node is a tail extrapolation. The table marks this
      explicitly through the bucket metadata and makes the model assumption visible.</div>
    </div>
  </div>
</section>

<section id="buckets">
  <div class="section-grid">
    <div class="rail">
      <p class="eyebrow">Step 3</p>
      <h2>Bucketed Greeks make the hedge actionable</h2>
      <p>Scalar delta, rhoq, and vega compress away where the risk lives. The
      consolidated API returns normalized points, so each coordinate carries
      the same fields: <code>derivative</code>, <code>reported</code>,
      <code>pnl</code>, <code>bump_size</code>, and <code>difference_mode</code>.</p>
    </div>
    <div>
      <h3>Futures delta: bump one IM close</h3>
      {delta_table}
      <div class="note"><b>Why it matters.</b> The sign and size of
      <code>hedge_hands</code> tell the desk which IM contract to trade. A
      scalar spot delta cannot distinguish IM2607 from IM2612.</div>
      <h3>Carry rhoq: bump one implied q node</h3>
      {rhoq_table}
      <div class="note"><b>Why it matters.</b> CSI 1000 Snowballs are highly
      exposed to futures basis. Bucketed rhoq shows whether the risk is front
      carry, quarter-end carry, or the extrapolated tail.</div>
      <h3>Vol-tenor vega: bump one maturity interval</h3>
      {vega_table}
      <div class="note"><b>Why it matters.</b> KO and KI features concentrate
      exposure around observation and barrier-relevant horizons. A single flat
      vega hides this term profile.</div>
    </div>
  </div>
</section>

<section id="checks">
  <div class="section-grid">
    <div class="rail">
      <p class="eyebrow">Step 4</p>
      <h2>Finite differences are explicit</h2>
      <p>This example requests central mode per coordinate. That is the least
      biased finite-difference choice when both up and down bumps are valid.</p>
    </div>
    <div>
      <div class="formula">central derivative = (PV_up - PV_down) / (2 * bump)</div>
      {manual_table}
      <div class="note"><b>Per-coordinate overrides.</b> The request sets
      futures delta, carry rhoq, and vol-tenor vega to central mode through
      <code>difference_mode_overrides</code>. The API still preserves older
      coordinate defaults when callers do not override them.</div>
    </div>
  </div>
</section>

{model_section}

<footer>
Generated {escape(generated_at)} from local artifact tag <code>{escape(tag)}</code>.
MO data date {escape(snapshot_date)}; IM data date {escape(futures_date)}.
</footer>
</div>
</body>
</html>
"""


def render_html_report(
    calc,
    product,
    env,
    engine,
    curve,
    *,
    include_localvol: bool = True,
    include_structured: bool = True,
) -> str:
    scalar = calc.calculate(
        product,
        env,
        engine,
        greeks=("price", "delta", "vega", "dividend_rho"),
    )
    futures_result = calc.calculate_bucketed_greeks(
        product,
        env,
        engine,
        BucketedGreeksRequest(
            futures_curve=curve,
            futures_price_bump=0.50,
            carry_bump=0.0001,
            difference_mode_overrides={
                BucketedGreekCoordinate.FUTURES_DELTA: (
                    BucketedGreekDifferenceMode.CENTRAL
                ),
                BucketedGreekCoordinate.CARRY_RHOQ: (
                    BucketedGreekDifferenceMode.CENTRAL
                ),
            },
        ),
    )
    delta_points = futures_result.by_coordinate(BucketedGreekCoordinate.FUTURES_DELTA)
    rhoq_points = futures_result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)

    active_delta = _active_point(delta_points)
    manual_delta, delta_up, delta_down = _manual_futures_delta_central(
        product,
        env,
        engine,
        curve,
        active_delta.contract,
        active_delta.bump_size,
    )
    active_rhoq = _active_point(rhoq_points)
    node_index = [q.contract for q in curve.quotes].index(active_rhoq.contract)
    manual_rhoq, raw_rhoq, rhoq_up, rhoq_down = _manual_implied_carry_rhoq_central(
        product,
        env,
        engine,
        curve,
        node_index,
        active_rhoq.bump_size,
    )

    tenor_result = calc.calculate_bucketed_greeks(
        product,
        env,
        engine,
        BucketedGreeksRequest(
            coordinates=(
                BucketedGreekCoordinate.VOL_TENOR_VEGA,
                BucketedGreekCoordinate.CARRY_RHOQ,
            ),
            vol_bump=0.01,
            carry_bump=0.0001,
        ),
    )
    vega_points = tenor_result.by_coordinate(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    carry_points = tenor_result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
    active_vega = _active_point(vega_points)
    buckets = {
        f"{bucket.label} ({bucket.start:.3g}-{bucket.end:.3g}y)": bucket
        for bucket in default_tenor_buckets(product.get_maturity(env))
    }
    vega_bucket = buckets[active_vega.bucket]
    manual_vega, raw_vega, vega_up, vega_down = _manual_vol_tenor_vega_central(
        product,
        env,
        engine,
        vega_bucket,
        active_vega.bump_size,
    )

    localvol_section = ""
    if include_localvol:
        lv_env = make_grid_env()
        lv_engine = LocalVolPDESolver(params=PDEParams(grid_size=100, time_steps=50))
        lv_result = calc.calculate_bucketed_greeks(
            product,
            lv_env,
            lv_engine,
            BucketedGreeksRequest(
                coordinates=(BucketedGreekCoordinate.MARKET_IV_VEGA,),
                market_vega_request=MarketVegaRequest(
                    surface_bumps=(
                        SurfaceBump.parallel(0.01),
                        SurfaceBump.maturity_row(2, 0.01),
                        SurfaceBump.strike_column(2, 0.01),
                    )
                ),
            ),
        )
        localvol_section = f"""
<section id="s4">
<h2>4. Market-IV vega delegates to the vol-model risk backend</h2>
<p>LocalVol market vega is not a flat-vol bump. The IV quote surface is bumped,
then the local-vol surface is rebuilt before repricing. Heston and SLV use their
own calibration artifacts through the same delegated risk backend.</p>
{_html_point_table(lv_result.points)}
</section>
"""

    structured_section = ""
    if include_structured:
        snowball = make_snowball()
        snow_result = calc.calculate_bucketed_greeks(
            snowball,
            env,
            SnowballQuadEngine(),
            BucketedGreeksRequest(
                coordinates=(
                    BucketedGreekCoordinate.VOL_TENOR_VEGA,
                    BucketedGreekCoordinate.CARRY_RHOQ,
                ),
                vol_bump=0.01,
                carry_bump=0.0001,
                difference_mode_overrides={
                    BucketedGreekCoordinate.VOL_TENOR_VEGA: (
                        BucketedGreekDifferenceMode.ONE_SIDED_UP
                    ),
                    BucketedGreekCoordinate.CARRY_RHOQ: (
                        BucketedGreekDifferenceMode.ONE_SIDED_UP
                    ),
                },
            ),
        )
        snow_vega = {
            p.bucket: p
            for p in snow_result.by_coordinate(BucketedGreekCoordinate.VOL_TENOR_VEGA)
        }
        snow_carry = {
            p.bucket: p
            for p in snow_result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
        }
        structured_section = f"""
<section id="s5">
<h2>5. Structured payoff profile</h2>
<p>A Snowball has observation dates and barrier states before final maturity.
Bucketed Greeks show which vol and carry tenors actually matter for hedging and
risk limits.</p>
{_html_table(
    ["bucket", "vega reported", "vega derivative", "rhoq reported", "rhoq derivative"],
    [
        [
            bucket,
            snow_vega[bucket].reported,
            snow_vega[bucket].derivative,
            snow_carry[bucket].reported,
            snow_carry[bucket].derivative,
        ]
        for bucket in snow_vega
    ],
)}
<div class="callout"><b>Risk-management point.</b> A scalar vega or scalar rhoq
cannot tell a hedger which maturity should be traded. This table does.</div>
</section>
"""

    scalar_table = _html_table(
        ["metric", "value", "meaning"],
        [
            ["price", scalar["price"], "base PV"],
            ["spot delta", scalar["delta"], "dPV / dSpot"],
            ["flat vega", scalar["vega"], "PV per +1 vol point"],
            ["dividend rhoq", scalar["dividend_rho"], "PV per +1% flat carry"],
        ],
    )
    delta_table = _html_table(
        ["contract", "T", "F", "dPV/dF", "pnl at bump", "hedge hands"],
        [
            [p.contract, p.maturity, p.future_price, p.derivative, p.pnl, p.hedge_hands]
            for p in delta_points
        ],
    )
    rhoq_table = _html_table(
        ["contract", "T", "rhoq reported", "raw dPV/dq", "pnl at bump"],
        [[p.contract, p.maturity, p.reported, p.derivative, p.pnl] for p in rhoq_points],
    )
    manual_table = _html_table(
        ["coordinate", "bucket", "API", "manual", "up PV", "down PV"],
        [
            [
                "dPV/dF",
                active_delta.contract,
                active_delta.derivative,
                manual_delta,
                delta_up,
                delta_down,
            ],
            [
                "rhoq +1%",
                active_rhoq.contract,
                active_rhoq.reported,
                manual_rhoq,
                rhoq_up,
                rhoq_down,
            ],
            [
                "vega +1 vol pt",
                active_vega.bucket,
                active_vega.reported,
                manual_vega,
                vega_up,
                vega_down,
            ],
        ],
    )
    tenor_table = _html_point_table((*vega_points, *carry_points))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bucketed Greeks Lecture</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172033;
  --muted: #5f6b7a;
  --line: #d9e1ea;
  --accent: #1f5d8f;
  --accent-soft: #e8f2fb;
  --panel: #f7f9fc;
}}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #ffffff;
  line-height: 1.55;
}}
main {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 36px 28px 64px;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 34px;
  letter-spacing: 0;
}}
h2 {{
  margin: 34px 0 12px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
  font-size: 24px;
  letter-spacing: 0;
}}
p {{ max-width: 860px; }}
code {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1px 5px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 14px 0 22px;
  font-size: 14px;
}}
th {{
  text-align: left;
  background: var(--panel);
  color: #263449;
  border: 1px solid var(--line);
  padding: 8px 10px;
}}
td {{
  border: 1px solid var(--line);
  padding: 7px 10px;
  vertical-align: top;
}}
td.num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.lead {{
  font-size: 17px;
  color: var(--muted);
}}
.callout {{
  border-left: 4px solid var(--accent);
  background: var(--accent-soft);
  padding: 12px 14px;
  margin: 14px 0 24px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin: 18px 0;
}}
.tile {{
  border: 1px solid var(--line);
  background: var(--panel);
  padding: 14px;
  border-radius: 6px;
}}
.tile b {{ display: block; margin-bottom: 4px; }}
</style>
</head>
<body>
<main>
<h1>Bucketed Greeks Lecture</h1>
<p class="lead">A self-contained walkthrough of
<code>GreeksCalculator.calculate_bucketed_greeks()</code>: what each coordinate
means, how the finite differences are calculated, and why bucketed risk is more
actionable than scalar Greeks.</p>

<div class="grid">
  <div class="tile"><b>Futures delta</b>dPV/dF_i by tradable futures tenor.</div>
  <div class="tile"><b>Carry rhoq</b>dPV/dq_i by implied-carry or tenor node.</div>
  <div class="tile"><b>Vol-tenor vega</b>dPV/dsigma_i by volatility maturity bucket.</div>
  <div class="tile"><b>Market-IV vega</b>Delegated model-aware IV quote risk.</div>
</div>

<section id="s1">
<h2>1. Scalar Greeks are useful but incomplete</h2>
<p>Scalar Greeks summarize the trade into one number per risk type. That is not
enough when the hedge instruments and market data are bucketed by tenor.</p>
{scalar_table}
</section>

<section id="s2">
<h2>2. Futures bucket delta and implied-carry rhoq</h2>
<p>The futures curve is first inverted into an implied carry curve. Then the API
bumps one futures mark or one implied carry node at a time. Per-coordinate
overrides request central differences for both coordinates in this section.</p>
{delta_table}
{rhoq_table}
<div class="callout"><b>Why this matters.</b> A scalar spot delta cannot say how
many hands belong in IC00 versus IC03. Futures bucket delta can.</div>
</section>

<section id="s3">
<h2>3. Manual finite-difference checks</h2>
<p>The API output below is checked against direct bump-and-reprice formulas:
<code>(PV_up - PV_down) / (2 * bump)</code> for raw derivatives and
<code>reported = derivative * convention_scale</code>.</p>
{manual_table}
<h2>4. Generic tenor buckets without a futures curve</h2>
<p>When no futures curve is supplied, the facade can still split vol and carry
risk into standard tenor buckets. Vol-tenor vega defaults to central mode.
Carry rhoq defaults to the historical one-sided-up desk convention.</p>
{tenor_table}
</section>

{localvol_section}
{structured_section}

<section id="schema">
<h2>Result schema</h2>
<p>Every point exposes <code>derivative</code>, <code>reported</code>,
<code>pnl</code>, <code>bump_size</code>, <code>convention_scale</code>, and
the resolved <code>difference_mode</code>. Coordinate-specific fields such as
<code>contract</code>, <code>bucket</code>, <code>hedge_hands</code>, and
<code>model</code> make the result directly usable in reports and hedge tools.</p>
</section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-structured", action="store_true")
    parser.add_argument("--skip-localvol", action="store_true")
    parser.add_argument(
        "--html-output",
        default=str(DEFAULT_HTML_OUTPUT),
        help="write a self-contained HTML lecture page to this path",
    )
    parser.add_argument(
        "--market-tag",
        default="latest",
        help="MO/IM market artifact tag used by the HTML lecture",
    )
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    calc = GreeksCalculator()
    env = make_env()
    curve = make_curve()
    product = make_vanilla()
    engine = BlackScholesEngine()

    print(
        "Bucketed Greeks demo: compact console stages use synthetic fixtures for "
        "speed; the HTML lecture uses real 000852.SH/MO and IM market artifacts."
    )
    print(
        "Normalized point fields: reported = derivative * convention_scale; "
        "pnl = finite-difference PnL for the configured bump."
    )

    stage_scalar_context(calc, product, env, engine)
    stage_futures_and_carry(calc, product, env, engine, curve)
    stage_generic_tenor(calc, product, env, engine)
    if not args.skip_localvol:
        stage_localvol_market_vega(calc, product)
    if not args.skip_structured:
        stage_structured_product(calc, env)
    if not args.no_html:
        html_path = Path(args.html_output)
        if not html_path.is_absolute():
            html_path = Path.cwd() / html_path
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            render_market_snowball_html_report(
                calc,
                tag=args.market_tag,
                include_localvol=not args.skip_localvol,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {html_path}")


if __name__ == "__main__":
    main()

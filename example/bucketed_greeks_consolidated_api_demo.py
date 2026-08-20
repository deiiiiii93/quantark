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


def make_spot_shifted_market_state(
    env: PricingEnvironment,
    curve: IndexFuturesCurve,
    *,
    base_spot: float,
    scenario_spot: float,
) -> tuple[PricingEnvironment, IndexFuturesCurve]:
    """Shift spot and futures marks together, preserving observed basis ratios."""
    ratio = float(scenario_spot) / float(base_spot)
    scenario_env = deepcopy(env)
    scenario_env.spot_quote.spot = float(scenario_spot)
    scenario_curve = IndexFuturesCurve(
        underlying=curve.underlying,
        spot=float(scenario_spot),
        quotes=[
            IndexFuturesQuote(
                quote.contract,
                maturity=quote.maturity,
                price=quote.price * ratio,
                multiplier=quote.multiplier,
                beta=quote.beta,
                expiry_date=quote.expiry_date,
            )
            for quote in curve.quotes
        ],
        mode=curve.mode,
        interpolation=curve.interpolation,
    )
    scenario_env.div_yield = scenario_curve.to_dividend_yield_curve(
        scenario_env.rate_curve
    )
    return scenario_env, scenario_curve


def calculate_futures_delta_points(
    calc: GreeksCalculator,
    product,
    env: PricingEnvironment,
    engine,
    curve: IndexFuturesCurve,
    *,
    price_bump: float,
) -> tuple[BucketedGreekPoint, ...]:
    result = calc.calculate_bucketed_greeks(
        product,
        env,
        engine,
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.FUTURES_DELTA,),
            futures_curve=curve,
            futures_price_bump=price_bump,
            difference_mode_overrides={
                BucketedGreekCoordinate.FUTURES_DELTA: (
                    BucketedGreekDifferenceMode.CENTRAL
                )
            },
        ),
    )
    return result.by_coordinate(BucketedGreekCoordinate.FUTURES_DELTA)


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


def _fmt_addend(value: float, digits: int = 2) -> str:
    sign = "+" if float(value) >= 0.0 else "-"
    return f"{sign} {abs(float(value)):,.{digits}f}"


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

    above_ko_spot = 1.05 * s0
    deep_above_ko_spot = 1.20 * s0
    scenario_rows = []
    above_ko_delta_points = None
    for label, scenario_spot, points in (
        ("market snapshot", s0, delta_points),
        ("spot above KO", above_ko_spot, None),
        ("spot deep above KO", deep_above_ko_spot, None),
    ):
        if points is None:
            scenario_env, scenario_curve = make_spot_shifted_market_state(
                env,
                curve,
                base_spot=s0,
                scenario_spot=scenario_spot,
            )
            points = calculate_futures_delta_points(
                calc,
                snowball,
                scenario_env,
                engine,
                scenario_curve,
                price_bump=request.futures_price_bump,
            )
            if label == "spot above KO":
                above_ko_delta_points = points
        total_abs_delta = sum(abs(float(p.derivative or 0.0)) for p in points)
        front_point = points[0]
        tail_point = points[-1]
        top_point = _active_point(points)
        scenario_rows.append(
            [
                label,
                _fmt_plain_pct(scenario_spot / s0),
                _fmt_pct(scenario_spot / snowball.barrier_config.ko_barrier - 1.0),
                top_point.contract,
                _fmt_plain_pct(abs(front_point.derivative) / total_abs_delta),
                _fmt_plain_pct(abs(tail_point.derivative) / total_abs_delta),
                _fmt_num(front_point.hedge_hands, 3),
                _fmt_num(tail_point.hedge_hands, 3),
            ]
        )
    if above_ko_delta_points is None:
        above_ko_delta_points = ()
    barrier_scenario_summary_table = _html_table(
        [
            "case",
            "spot / initial",
            "spot vs KO",
            "largest bucket",
            "front share",
            "tail share",
            "front hedge",
            "tail hedge",
        ],
        scenario_rows,
    )
    above_ko_delta_table = _html_table(
        [
            "IM contract",
            "T",
            "shifted F",
            "dPV/dF",
            "hedge hands",
            "PnL @ +5pt",
        ],
        [
            [
                p.contract,
                f"{p.maturity:.3f}y",
                _fmt_level(p.future_price),
                _fmt_signed_money(p.derivative),
                _fmt_num(p.hedge_hands, 3),
                _fmt_signed_money(p.pnl),
            ]
            for p in above_ko_delta_points
        ],
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
    delta_by_contract = {p.contract: p for p in delta_points}
    scalar_greeks = calc.calculate(
        snowball,
        pricing_env,
        engine,
        greeks=("delta", "dividend_rho"),
    )
    scalar_delta = float(scalar_greeks["delta"])
    scalar_rhoq = float(scalar_greeks["dividend_rho"])
    if futures_snapshot is not None:
        scalar_hedge_contract = max(
            futures_snapshot["quotes"],
            key=lambda row: float(row.get("volume", 0.0)),
        )["contract"]
    else:
        scalar_hedge_contract = curve.quotes[0].contract
    scalar_hedge_quote = curve.get_quote(scalar_hedge_contract)
    scalar_delta_cash_1pct = scalar_delta * s0 * 0.01
    scalar_total_hands = -scalar_delta / scalar_hedge_quote.multiplier
    bucket_total_hands = sum(float(p.hedge_hands or 0.0) for p in delta_points)
    bucket_distribution = "; ".join(
        f"{p.contract} {_fmt_num(p.hedge_hands, 3)}" for p in delta_points
    )
    bucket_delta_sum = sum(float(p.derivative or 0.0) for p in delta_points)
    bucket_delta_cash_1pct = sum(
        float(p.derivative or 0.0) * float(p.future_price or 0.0) * 0.01
        for p in delta_points
    )
    bucket_rhoq_sum = sum(float(p.reported or 0.0) for p in rhoq_points)
    scalar_multiplier_text = f"{float(scalar_hedge_quote.multiplier):,.0f}"
    scalar_hands_formula = (
        f"-({_fmt_num(scalar_delta, 2)} CNY/spot pt) / "
        f"{scalar_multiplier_text}"
    )
    bucket_hands_detail_table = _html_table(
        [
            "IM contract",
            "delta input",
            "F close",
            "delta cash @ +1% F",
            "delta per hand",
            "hands formula",
            "hedge hands",
        ],
        [
            [
                p.contract,
                f"{_fmt_num(p.derivative, 2)} CNY/F pt",
                _fmt_level(p.future_price),
                _fmt_signed_money(float(p.derivative) * float(p.future_price) * 0.01),
                f"{float(p.delta_per_hand):,.0f}",
                f"-({_fmt_num(p.derivative, 2)}) / {float(p.delta_per_hand):,.0f}",
                _fmt_num(p.hedge_hands, 3),
            ]
            for p in delta_points
        ],
    )
    delta_cash_math_table = _html_table(
        [
            "view",
            "delta input",
            "reference move",
            "delta cash",
            "hands formula",
            "hands",
        ],
        [
            [
                f"scalar delta IM hedge ({scalar_hedge_contract})",
                f"{_fmt_num(scalar_delta, 2)} CNY/spot pt",
                f"1% spot = {_fmt_level(s0 * 0.01)} index pts",
                _fmt_signed_money(scalar_delta_cash_1pct),
                scalar_hands_formula,
                _fmt_num(scalar_total_hands, 3),
            ],
            [
                "bucketed IM futures delta",
                f"sum dPV/dF_i = {_fmt_num(bucket_delta_sum, 2)}",
                "1% move in each listed IM futures mark",
                _fmt_signed_money(bucket_delta_cash_1pct),
                "sum_i [-(dPV/dF_i) / 200]",
                _fmt_num(bucket_total_hands, 3),
            ],
        ],
    )
    scalar_bucket_relationship_table = _html_table(
        ["risk", "scalar story", "bucket-sum story", "comparison rule"],
        [
            [
                "delta",
                (
                    f"dPV/dS = {_fmt_num(scalar_delta, 2)}; "
                    f"1% spot delta cash = {_fmt_signed_money(scalar_delta_cash_1pct)}"
                ),
                (
                    f"sum dPV/dF_i = {_fmt_num(bucket_delta_sum, 2)}; "
                    f"all-F +1% delta cash = {_fmt_signed_money(bucket_delta_cash_1pct)}"
                ),
                (
                    "Do not compare as identities: spot and listed-futures "
                    "curve shocks are different scenarios."
                ),
            ],
            [
                "rhoq",
                f"parallel scalar carry rhoq = {_fmt_signed_money(scalar_rhoq)}",
                f"sum implied-carry node rhoq = {_fmt_signed_money(bucket_rhoq_sum)}",
                (
                    "Compare only after defining the carry scenario and node "
                    "weights; bucket rhoq is mainly a hedge-placement map."
                ),
            ],
        ],
    )
    desk_scalar_practice_table = _html_table(
        [
            "reason",
            "what scalar delta cash gives",
            "what bucketed futures delta adds / caveat",
        ],
        [
            [
                "common risk language",
                (
                    "one CNY number for book limits, trader handover, PnL "
                    "explain, and stress dashboards"
                ),
                (
                    "a vector needs tenor labels, curve construction, and an "
                    "aggregation rule before comparison"
                ),
            ],
            [
                "fast intraday control",
                "updates cheaply from spot moves and portfolio aggregation",
                (
                    "requires current futures marks, implied-carry rebuild, "
                    "and product repricing per node"
                ),
            ],
            [
                "execution liquidity",
                (
                    "converts cleanly into IM hands through a house allocation "
                    "rule focused on liquid contracts"
                ),
                (
                    "may ask for far-tenor or less-liquid contracts where "
                    "rounding, margin, bid/ask, and roll calendars matter"
                ),
            ],
            [
                "model governance",
                "more stable across engines and easier for risk managers to audit",
                (
                    "depends on futures basis, interpolation, tail extrapolation, "
                    "KO/KI model details, and finite-difference settings"
                ),
            ],
            [
                "hedge objective",
                "good first-order control for broad spot/futures co-move",
                (
                    "best used as a second-layer overlay when basis, carry, "
                    "rhoq, or roll placement matters"
                ),
            ],
        ],
    )
    rhoq_hedge_rows = []
    scalar_im_hedge_by_contract: dict[str, float] = {}
    scalar_im_residual_by_contract: dict[str, float] = {}
    scalar_im_abs_residual = 0.0
    bucketed_im_abs_residual = 0.0
    option_abs_rhoq = 0.0
    for point in rhoq_points:
        delta_point = delta_by_contract[point.contract]
        scalar_im_hedge_rhoq = (
            scalar_total_hands
            * scalar_hedge_quote.multiplier
            * (-point.maturity * point.future_price)
            * 0.01
            if point.contract == scalar_hedge_contract
            else 0.0
        )
        bucketed_im_hedge_rhoq = (
            delta_point.hedge_hands
            * delta_point.delta_per_hand
            * (-point.maturity * point.future_price)
            * 0.01
        )
        scalar_im_residual = point.reported + scalar_im_hedge_rhoq
        bucketed_im_residual = point.reported + bucketed_im_hedge_rhoq
        option_abs_rhoq += abs(point.reported)
        scalar_im_abs_residual += abs(scalar_im_residual)
        bucketed_im_abs_residual += abs(bucketed_im_residual)
        scalar_im_hedge_by_contract[point.contract] = scalar_im_hedge_rhoq
        scalar_im_residual_by_contract[point.contract] = scalar_im_residual
        rhoq_hedge_rows.append(
            [
                point.contract,
                _fmt_signed_money(point.reported),
                _fmt_signed_money(scalar_im_hedge_rhoq),
                _fmt_signed_money(scalar_im_residual),
                _fmt_signed_money(bucketed_im_hedge_rhoq),
                _fmt_signed_money(bucketed_im_residual),
            ]
        )
    scalar_im_rhoq_reduction = (
        1.0 - scalar_im_abs_residual / option_abs_rhoq
        if option_abs_rhoq > 0.0
        else 0.0
    )
    bucketed_im_rhoq_reduction = (
        1.0 - bucketed_im_abs_residual / option_abs_rhoq
        if option_abs_rhoq > 0.0
        else 0.0
    )
    rhoq_hedge_table = _html_table(
        [
            "carry node",
            "option rhoq",
            "scalar IM hedge rhoq",
            "residual: scalar IM",
            "bucketed IM hedge rhoq",
            "residual: bucketed IM",
        ],
        rhoq_hedge_rows,
    )
    tail_rhoq_point = rhoq_points[-1]
    scalar_node_added_rhoq = scalar_im_hedge_by_contract.get(
        scalar_hedge_contract, 0.0
    )
    scalar_tail_residual_rhoq = scalar_im_residual_by_contract.get(
        tail_rhoq_point.contract, 0.0
    )
    rhoq_hands_table = _html_table(
        ["method", "total hands", "allocation"],
        [
            [
                f"scalar delta IM hedge ({scalar_hedge_contract})",
                _fmt_num(scalar_total_hands, 3),
                (
                    "convert scalar delta to IM hands; allocate all hands to "
                    f"{scalar_hedge_contract} (highest-volume contract)"
                ),
            ],
            [
                "bucketed IM futures delta",
                _fmt_num(bucket_total_hands, 3),
                bucket_distribution,
            ],
        ],
    )
    rhoq_hedge_summary_table = _html_table(
        ["hedge method", "absolute residual rhoq", "reduction vs unhedged"],
        [
            ["unhedged option", _fmt_money(option_abs_rhoq), "0.00%"],
            [
                f"scalar delta IM hedge ({scalar_hedge_contract})",
                _fmt_money(scalar_im_abs_residual),
                _fmt_pct(scalar_im_rhoq_reduction),
            ],
            [
                "bucketed IM futures delta hedge",
                _fmt_money(bucketed_im_abs_residual),
                _fmt_pct(bucketed_im_rhoq_reduction),
            ],
        ],
    )
    bucket_scalar_equivalent_delta = sum(
        float(p.hedge_hands or 0.0) * float(p.delta_per_hand or 0.0)
        for p in delta_points
    )
    scalar_delta_after_scalar_hedge = (
        scalar_delta + scalar_total_hands * scalar_hedge_quote.multiplier
    )
    scalar_delta_cash_after_scalar_hedge = (
        scalar_delta_after_scalar_hedge * s0 * 0.01
    )
    scalar_delta_after_bucket_hedge = scalar_delta + bucket_scalar_equivalent_delta
    scalar_delta_cash_after_bucket_hedge = (
        scalar_delta_after_bucket_hedge * s0 * 0.01
    )
    scalar_hedge_delta_contribution = (
        scalar_total_hands * scalar_hedge_quote.multiplier
    )
    bucket_hedge_delta_contribution = bucket_scalar_equivalent_delta
    scalar_bucket_delta_residual_sum = 0.0
    scalar_bucket_delta_residual_cash = 0.0
    scalar_bucket_delta_abs_residual = 0.0
    scalar_bucket_delta_abs_cash = 0.0
    bucket_bucket_delta_residual_sum = 0.0
    bucket_bucket_delta_residual_cash = 0.0
    bucket_bucket_delta_abs_residual = 0.0
    bucket_bucket_delta_abs_cash = 0.0
    delta_reconciliation_rows = []
    for point in delta_points:
        scalar_hands = (
            scalar_total_hands if point.contract == scalar_hedge_contract else 0.0
        )
        bucket_hands = float(point.hedge_hands or 0.0)
        per_hand = float(point.delta_per_hand or 0.0)
        future_price = float(point.future_price or 0.0)
        scalar_residual = float(point.derivative or 0.0) + scalar_hands * per_hand
        bucket_residual = float(point.derivative or 0.0) + bucket_hands * per_hand
        scalar_residual_cash = scalar_residual * future_price * 0.01
        bucket_residual_cash = bucket_residual * future_price * 0.01
        scalar_bucket_delta_residual_sum += scalar_residual
        scalar_bucket_delta_residual_cash += scalar_residual_cash
        scalar_bucket_delta_abs_residual += abs(scalar_residual)
        scalar_bucket_delta_abs_cash += abs(scalar_residual_cash)
        bucket_bucket_delta_residual_sum += bucket_residual
        bucket_bucket_delta_residual_cash += bucket_residual_cash
        bucket_bucket_delta_abs_residual += abs(bucket_residual)
        bucket_bucket_delta_abs_cash += abs(bucket_residual_cash)
        delta_reconciliation_rows.append(
            [
                point.contract,
                _fmt_signed_money(point.derivative),
                _fmt_num(scalar_hands, 3),
                _fmt_signed_money(scalar_residual),
                _fmt_signed_money(scalar_residual_cash),
                _fmt_num(bucket_hands, 3),
                _fmt_signed_money(bucket_residual),
                _fmt_signed_money(bucket_residual_cash),
            ]
        )
    numerical_reconciliation_table = _html_table(
        [
            "hedge",
            "scalar delta equation",
            "scalar 1% cash residual",
            "bucket-delta equation",
            "bucket 1% cash residual",
            "rhoq residual",
        ],
        [
            [
                f"scalar IM hedge ({scalar_hedge_contract})",
                (
                    f"{_fmt_num(scalar_delta, 2)} "
                    f"{_fmt_addend(scalar_hedge_delta_contribution)} = "
                    f"{_fmt_num(scalar_delta_after_scalar_hedge, 2)}"
                ),
                _fmt_signed_money(scalar_delta_cash_after_scalar_hedge),
                (
                    f"{_fmt_num(bucket_delta_sum, 2)} "
                    f"{_fmt_addend(scalar_hedge_delta_contribution)} = "
                    f"{_fmt_num(scalar_bucket_delta_residual_sum, 2)}"
                ),
                (
                    f"{_fmt_signed_money(scalar_bucket_delta_residual_cash)} "
                    f"(abs {_fmt_money(scalar_bucket_delta_abs_cash)})"
                ),
                _fmt_money(scalar_im_abs_residual),
            ],
            [
                "bucketed IM futures hedge",
                (
                    f"{_fmt_num(scalar_delta, 2)} "
                    f"{_fmt_addend(bucket_hedge_delta_contribution)} = "
                    f"{_fmt_num(scalar_delta_after_bucket_hedge, 2)}"
                ),
                _fmt_signed_money(scalar_delta_cash_after_bucket_hedge),
                (
                    f"{_fmt_num(bucket_delta_sum, 2)} "
                    f"{_fmt_addend(bucket_hedge_delta_contribution)} = "
                    f"{_fmt_num(bucket_bucket_delta_residual_sum, 2)}"
                ),
                (
                    f"{_fmt_signed_money(bucket_bucket_delta_residual_cash)} "
                    f"(abs {_fmt_money(bucket_bucket_delta_abs_cash)})"
                ),
                _fmt_money(bucketed_im_abs_residual),
            ],
        ],
    )
    delta_reconciliation_table = _html_table(
        [
            "IM contract",
            "option dPV/dF",
            "scalar hedge hands",
            "residual dPV/dF: scalar hedge",
            "residual cash: scalar hedge",
            "bucket hedge hands",
            "residual dPV/dF: bucket hedge",
            "residual cash: bucket hedge",
        ],
        delta_reconciliation_rows,
    )
    front_risk_alignment_table = _html_table(
        [
            "control question",
            "what happens in this example",
            "risk-control fix",
        ],
        [
            [
                "day-end scalar delta limit",
                (
                    "front office uses bucketed hedge hands "
                    f"{_fmt_num(bucket_total_hands, 3)}. If risk maps those "
                    "hands back through the scalar IM multiplier, the book "
                    f"looks {_fmt_num(scalar_delta_after_bucket_hedge, 2)} "
                    "CNY/spot pt, or "
                    f"{_fmt_signed_money(scalar_delta_cash_after_bucket_hedge)} "
                    "for a 1% spot move."
                ),
                (
                    "Keep the scalar delta-cash limit, but require a bridge "
                    "showing scalar before/after hedge and bucket allocation."
                ),
            ],
            [
                "day-end rhoq limit",
                (
                    "bucketed hedge residual is "
                    f"{_fmt_money(bucketed_im_abs_residual)} by node, while "
                    "the scalar report only sees a parallel carry story."
                ),
                (
                    "Report both scalar parallel rhoq and node residual rhoq; "
                    "do not let one replace the other silently."
                ),
            ],
            [
                "PnL explain",
                (
                    "trader explains futures PnL by IM contract and carry node; "
                    "risk explains by spot delta cash and parallel carry."
                ),
                (
                    "Add a daily reconciliation from scalar delta cash to "
                    "futures buckets, basis, roll, and residual rhoq."
                ),
            ],
            [
                "governance",
                (
                    "neither desk is necessarily wrong; they are using "
                    "different risk-factor bases."
                ),
                (
                    "Define the official hierarchy: scalar limits for top-down "
                    "control, bucket limits for hedge placement and basis/rhoq "
                    "risk."
                ),
            ],
        ],
    )
    reconciliation_snapshot_table = _html_table(
        ["bridge item", "scalar / risk lens", "bucket / front-office lens"],
        [
            [
                "pre-hedge delta",
                (
                    f"{_fmt_num(scalar_delta, 2)} CNY/spot pt; "
                    f"{_fmt_signed_money(scalar_delta_cash_1pct)} for 1% spot"
                ),
                (
                    f"sum dPV/dF_i = {_fmt_num(bucket_delta_sum, 2)}; "
                    f"{_fmt_signed_money(bucket_delta_cash_1pct)} for all-F +1%"
                ),
            ],
            [
                "hedge instruction",
                (
                    f"{_fmt_num(scalar_total_hands, 3)} IM hands by desk "
                    f"allocation rule into {scalar_hedge_contract}"
                ),
                (
                    f"{_fmt_num(bucket_total_hands, 3)} total IM hands split as "
                    f"{bucket_distribution}"
                ),
            ],
            [
                "post-hedge scalar delta check",
                (
                    "scalar hedge target is approximately zero before rounding "
                    "and execution slippage"
                ),
                (
                    "bucket hedge maps to "
                    f"{_fmt_num(scalar_delta_after_bucket_hedge, 2)} "
                    "CNY/spot pt, or "
                    f"{_fmt_signed_money(scalar_delta_cash_after_bucket_hedge)} "
                    "for 1% spot"
                ),
            ],
            [
                "post-hedge rhoq check",
                (
                    f"scalar-allocation hedge residual abs rhoq = "
                    f"{_fmt_money(scalar_im_abs_residual)}"
                ),
                (
                    f"bucket hedge residual abs node rhoq = "
                    f"{_fmt_money(bucketed_im_abs_residual)}"
                ),
            ],
        ],
    )
    reconciliation_framework_table = _html_table(
        ["step", "daily control action", "required output"],
        [
            [
                "1. Freeze inputs",
                (
                    "lock product terms, valuation date, spot, IM quotes, "
                    "rates, vol model, engine, bump sizes, and curve mode"
                ),
                "one reproducible market/model snapshot shared by desk and risk",
            ],
            [
                "2. Produce scalar risk view",
                (
                    "calculate scalar delta cash, scalar parallel rhoq, and "
                    "limit usage under the official risk methodology"
                ),
                "day-end hard-limit view and breach status",
            ],
            [
                "3. Produce bucket hedge view",
                (
                    "calculate dPV/dF_i, hands_i = -dPV/dF_i / multiplier, "
                    "bucket rhoq, and residual node rhoq after hedge"
                ),
                "contract allocation, node residuals, and roll/basis exposure",
            ],
            [
                "4. Translate across lenses",
                (
                    "map bucket hands back to scalar delta cash; map scalar "
                    "hands back to node rhoq residuals"
                ),
                "bridge table showing why scalar and bucket reports differ",
            ],
            [
                "5. Attribute the gap",
                (
                    "split the difference into tenor basis, tail extrapolation, "
                    "rounding/liquidity, curve interpolation, and model/bump settings"
                ),
                "explainable residual rather than an unexplained limit dispute",
            ],
            [
                "6. Govern exceptions",
                (
                    "pre-define tolerances and approvals when bucket hedging "
                    "improves rhoq but worsens scalar delta cash, or vice versa"
                ),
                "signed exception, hedge rationale, and next rebalancing trigger",
            ],
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
      <a href="#barrier-proximity">Barrier proximity</a>
      <a href="#rhoq-hedge">Rhoq hedge</a>
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

<section id="barrier-proximity">
  <div class="section-grid">
    <div class="rail">
      <p class="eyebrow">Step 4</p>
      <h2>When spot is already above the KO barrier</h2>
      <p>Barrier proximity changes where futures delta appears. To isolate that
      effect, this scenario shifts spot and all IM futures marks by the same
      level ratio, preserving the observed basis percentages from the real
      snapshot.</p>
    </div>
    <div>
      <h3>Does delta migrate to the near contracts?</h3>
      {barrier_scenario_summary_table}
      <div class="note"><b>Reading the scenario.</b> Yes, the front contract
      gets more important once spot is above KO, because the first monthly
      observation can terminate the product. But the last listed IM bucket still
      absorbs all post-December tail exposure in this four-contract curve. With
      a fuller 2Y IM strip, that tail would be split across later contracts
      instead of being concentrated in IM2612.</div>
      <h3>Spot above KO: full futures-delta buckets</h3>
      {above_ko_delta_table}
    </div>
  </div>
</section>

<section id="rhoq-hedge">
  <div class="section-grid">
    <div class="rail">
      <p class="eyebrow">Step 5</p>
      <h2>Does bucketed IM hedging mitigate rhoq?</h2>
      <p>Both methods still trade IM futures. The scalar approach computes one
      spot-delta number, converts it into total IM hands, and then needs an
      allocation rule. Here the scalar hedge is placed in
      <code>{scalar_hedge_contract}</code>, the highest-volume contract in the
      snapshot. The bucketed approach prices <code>dPV/dF_i</code> directly and
      gives the hands contract by contract. The difference is not
      index-versus-futures trading; it is the total number of IM hands and their
      tenor distribution.</p>
    </div>
    <div>
      <h3>Total hands and allocation</h3>
      {rhoq_hands_table}
      <h3>Delta cash and hands math</h3>
      <p>Delta cash below is the first-order PnL for a 1% move in the relevant
      risk variable. It is useful for reading scale. Futures hands are still
      calculated from raw per-point delta because each IM futures hand has
      <code>CNY 200</code> PnL per index point.</p>
      {delta_cash_math_table}
      <h3>Bucket hands detail</h3>
      {bucket_hands_detail_table}
      <h3>Scalar Greeks versus bucket sums</h3>
      {scalar_bucket_relationship_table}
      <div class="note"><b>Comparison rule.</b> Scalar Greeks and bucket sums
      can be compared only when they describe the same market story. A scalar
      delta is a spot shock; the futures bucket sum is a listed-futures curve
      shock. A scalar rhoq is a parallel carry shock; the bucket rhoq vector is
      a node-by-node implied-carry map. For hedging, the right comparison is
      after translating the chosen story into IM hands and measuring residual
      node exposure. In this snapshot the scalar rhoq and summed bucket rhoq are
      close because both approximate a parallel carry move on the same implied
      carry curve; the delta row is not close because the shocks are different.</div>
      <h3>Why desks still start with scalar delta cash</h3>
      {desk_scalar_practice_table}
      <div class="note"><b>Practical takeaway.</b> Scalar delta cash is a
      production control number, not a claim that traders can trade the index
      directly. It gives one robust exposure number that can be converted into
      IM futures through the desk's execution rule. Bucketed futures delta is
      more informative, but it is model- and curve-dependent, so it is usually
      introduced as an allocation, basis, rhoq, and roll-risk overlay rather
      than a full replacement for the scalar delta-cash limit.</div>
      <h3>When front office and risk use different bases</h3>
      {front_risk_alignment_table}
      <div class="note warn"><b>Control issue.</b> This is a measurement-basis
      mismatch. The front office can be right that bucketed futures delta gives
      a better contract allocation and lower node rhoq, while risk can also be
      right that the same hedge breaches a scalar delta-cash limit. The fix is
      not to choose one number blindly; the day-end package needs a governed
      bridge from scalar delta cash to futures-bucket hands and residual rhoq.</div>
      <h3>Reconciliation framework</h3>
      <p>The reconciliation should be a daily control package, not an informal
      spreadsheet note. It must preserve both views and show how the hedge looks
      when translated through the other view's risk factors.</p>
      {reconciliation_snapshot_table}
      <h3>Real-case numerical bridge</h3>
      <p>These equations use the same 000852.SH Snowball and listed IM futures
      snapshot. They show why a hedge can be correct under one lens and fail
      under the other.</p>
      {numerical_reconciliation_table}
      <h3>Delta residual by contract</h3>
      {delta_reconciliation_table}
      {reconciliation_framework_table}
      <div class="note"><b>Policy recommendation.</b> Keep scalar delta cash as
      the hard top-down limit, add bucketed futures delta and node rhoq as
      controlled sub-limits, and require an exception workflow whenever an
      optimizer improves one view while worsening the other. The bridge table is
      the evidence for approving, resizing, or rejecting that hedge.</div>
      <h3>Residual carry risk after each IM hedge</h3>
      {rhoq_hedge_table}
      {rhoq_hedge_summary_table}
      <div class="note"><b>Why total hands are so different.</b> The scalar
      hedge starts from <code>dPV/dS</code>: one spot shock, then one execution
      rule for where to trade the IM futures. The bucket hedge starts from
      <code>dPV/dF_i</code>: each listed futures mark is shocked separately and
      the implied carry curve is rebuilt. Those bucket shocks include tenor and
      basis/carry information that a spot shock does not contain. With this
      four-contract strip, <code>{tail_rhoq_point.contract}</code> also carries
      the post-December tail of the 2Y Snowball. That is why the scalar hedge is
      {_fmt_num(scalar_total_hands, 3)} IM hands while the bucket hedge totals
      {_fmt_num(bucket_total_hands, 3)} hands across contracts; the bucket
      vector is not required to add back to the scalar spot delta.</div>
      <div class="note"><b>Why the scalar IM hedge can increase rhoq.</b> The
      scalar rule puts all hands into <code>{scalar_hedge_contract}</code>, so
      it adds {_fmt_signed_money(scalar_node_added_rhoq)} of carry rhoq on that
      node. But the largest option carry exposure is the tail node
      <code>{tail_rhoq_point.contract}</code>, where the scalar hedge leaves
      {_fmt_signed_money(scalar_tail_residual_rhoq)} residual rhoq. The scalar
      hedge therefore over-hedges one node and leaves the main tail node open,
      increasing absolute residual rhoq from {_fmt_money(option_abs_rhoq)} to
      {_fmt_money(scalar_im_abs_residual)}.</div>
      <div class="note"><b>Trading interpretation.</b> The scalar IM hedge is
      still a futures hedge, but its single total-hand number does not know
      which carry node created the risk. In this snapshot it over-hedges
      <code>{scalar_hedge_contract}</code> carry and leaves the far IM tail
      mostly open. Bucketed futures delta mitigates measured rhoq because each
      IM hedge is placed on the same node that defines the implied carry curve:
      <code>dPV/dq_i = dPV/dF_i * dF_i/dq_i</code>.</div>
      <div class="note warn"><b>Limitations.</b> This is not a universal
      basis hedge. It works for the listed IM carry nodes represented in the
      curve. It does not eliminate model risk, volatility risk, realised
      rebalancing error, non-parallel basis moves, or unlisted tail-tenor risk
      that the four-contract curve approximates with its last node.</div>
    </div>
  </div>
</section>

<section id="checks">
  <div class="section-grid">
    <div class="rail">
      <p class="eyebrow">Step 6</p>
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

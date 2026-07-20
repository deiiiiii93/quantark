# Greek-Bump Scenarios (quantark core + adapter v031) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make greek bumping a framework citizen (`quantark.execution.greeks` + `PricingSession.run_scenario_plans`) and ship `otc_quantark_pricer_v031` pricing the whole book at bump-cell granularity with bitwise v023 parity.

**Architecture:** Per-row base (`BaseInputsRef` → `TradeState`) + same-type bump transformer (`greek-bump/v1`) + float runner (`greek-value/v1`); a new multi-plan session API packs all rows' cells through one processes pool; parent-side assemblers mirror `get_trade_risk` / `calculate_numerical_greeks` arithmetic verbatim. Two-phase execution (base cells → bump cells of healthy rows) mirrors v023 failure semantics.

**Tech Stack:** Python 3.11, quantark.execution scenario layer, pytest; adapter repo `/Users/fuxinyao/otc-price-adapter` (branch `greek-bump-v031`), quant-ark worktree.

## Global Constraints

- Exact parity vs v023: floats `==`, strings `==`, all 24 output columns; any mismatch is root-caused (never tolerated, never fallen back from silently).
- `GreeksCalculator`, `EquityPosition`, v023, v030 are NOT modified.
- No cross-cell artifact/draw reuse claims in this phase.
- quantark version `0.3.0rc1 → 0.3.0rc2` only in the final wheel task.
- quant-ark tests: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest` from the worktree. Adapter tests: `.venv/bin/python -m pytest` from `/Users/fuxinyao/otc-price-adapter` with `PYTHONPATH=/path/to/quantark-worktree` prepended so source shadows the vendored wheel during dev.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- New quant-ark docs files need `git add -f`.

---

### Task 1: `quantark/execution/greeks.py` — TradeState, cells, bumps, runner, assemblers

**Files:**
- Create: `quantark/execution/greeks.py`
- Test: `test/execution/test_greek_bump_cells.py`
- Modify: `quantark/execution/__init__.py` (export `TradeState`, `greek_bump_cells`, `apply_greek_bump`, `assemble_trade_greeks`, `assemble_product_greeks`, `GREEK_BUMP_TRANSFORMER_ID`, `GREEK_VALUE_RUNNER_ID`)

**Interfaces (produced):**
- `TradeState(product, pricing_env, cash_legs: tuple, engine, streams, quantity: float, greeks_params, warnings: tuple[str, ...] = (), theta_degenerate: bool = False)` — plain (non-frozen) dataclass.
- `GreekBumpCell(bump_id: str, greeks_served: frozenset[str], mutation_tags: frozenset[str])`.
- `greek_bump_cells(greeks: Sequence[str]) -> tuple[GreekBumpCell, ...]` — base first, then `spot_up, spot_down, vol_up, rate_up, div_up, theta` filtered by requested greeks (`delta`/`gamma` ⇒ both spot cells; `vega` ⇒ `vol_up`; `rho` ⇒ `rate_up`; `dividend_rho` ⇒ `div_up`; `theta` ⇒ `theta`).
- `apply_greek_bump(bump_id, state, gc) -> TradeState` — deepcopy-based, same-type.
- Registered: transformer `greek-bump/v1` (fn `greek_bump_transform(base, parameters)`, `parameters={"bump_id": ...}`), runner `greek-value/v1` (value_kind `"float"`).
- `assemble_trade_greeks(base_values, bump_values, quantity, spot, bump_config, requested) -> dict` with keys `product`, `total`, `leg_pvs` — verbatim `get_trade_risk` arithmetic.
- `assemble_product_greeks(base_price, bump_values, bump_config, requested) -> dict` — verbatim `calculate_numerical_greeks` arithmetic (BUMP mode).
- Economics keys emitted by the runner: `npv`, `leg_sum`, `degenerate` (0.0/1.0), and for base cells `leg_pv::<leg_id>`, `leg_name::<leg_id>` (name as string), `warning::<i>`.

- [ ] **Step 1: Write the failing test** (`test/execution/test_greek_bump_cells.py`):

```python
"""Core bitwise gate: greeks assembled from bump cells == the originals."""
import dataclasses

import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import SnowballOption, BarrierOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import AutocallableCashLeg  # match adapter leg type
from quantark.portfolio.equity.position import EquityPosition
from quantark.execution.greeks import (
    TradeState,
    apply_greek_bump,
    assemble_product_greeks,
    assemble_trade_greeks,
    greek_bump_cells,
    greek_bump_transform,
    run_greek_bump,
)

GREEKS7 = ["delta", "gamma", "vega", "theta", "rho", "dividend_rho"]


def _env():  # flat env with calendar-less calendar_days theta
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from datetime import datetime

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(0.22),
        rate_curve=FlatRateCurve(0.025),
        valuation_date=datetime(2026, 6, 26),
    )


def _snowball_position():
    product = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        contract_multiplier=10_000.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.12,
            ko_observation_dates=[i / 12 for i in range(1, 13)],
            ki_barrier=80.0, ki_continuous=True,
        ),
    )
    engine = SnowballPDESolver(params=PDEParams(grid_size=120, time_steps=120))
    legs = ()  # exercise the leg path too if AutocallableCashLeg fixture is cheap; else ()
    return product, engine, legs


def test_trade_greeks_bitwise_vs_get_trade_risk():
    product, engine, legs = _snowball_position()
    env = _env()
    gc = GreeksCalculator()
    pos = EquityPosition(
        product=product, quantity=1.0, entry_price=100.0,
        underlying="TEST", engine=engine,
        entry_timestamp=env.valuation_date, cash_legs=list(legs),
    )
    expected = pos.get_trade_risk(env, gc, GREEKS7)

    state = TradeState(
        product=product, pricing_env=env, cash_legs=tuple(legs),
        engine=engine, streams=pos._required_streams(), quantity=1.0,
        greeks_params=gc.params,
    )
    values = {}
    for cell in greek_bump_cells(GREEKS7):
        bumped = greek_bump_transform(state, {"bump_id": cell.bump_id})
        values[cell.bump_id] = run_greek_bump(cell.bump_id, state, bumped, gc)
    got = assemble_trade_greeks(
        values["base"], values, quantity=1.0, spot=env.spot,
        bump_config=gc._bump_config, requested=set(GREEKS7),
    )
    for key in ["price"] + GREEKS7:
        assert got["product"][key] == expected.product[key]
        assert got["total"][key] == expected.total[key]


def test_product_greeks_bitwise_vs_calculate_numerical_greeks():
    from quantark.util.enum import BarrierType, OptionType

    product = BarrierOption(
        strike=100.0, maturity=0.75, option_type=OptionType.CALL,
        barrier_type=BarrierType.DOWN_AND_OUT, barrier=85.0,
    )
    engine = BarrierAnalyticalEngine()
    env = _env()
    gc = GreeksCalculator()
    raw = float(engine.price(product, env))
    expected = gc.calculate_numerical_greeks(
        product, env, engine, base_price=raw,
        greeks=["price"] + GREEKS7,
    )
    state = TradeState(
        product=product, pricing_env=env, cash_legs=(), engine=engine,
        streams=None, quantity=1.0, greeks_params=gc.params,
    )
    values = {}
    for cell in greek_bump_cells(GREEKS7):
        bumped = greek_bump_transform(state, {"bump_id": cell.bump_id})
        values[cell.bump_id] = run_greek_bump(cell.bump_id, state, bumped, gc)
    got = assemble_product_greeks(
        base_price=raw, bump_values=values,
        bump_config=gc._bump_config, requested=set(["price"] + GREEKS7),
    )
    for key in ["price"] + GREEKS7:
        assert got[key] == expected[key]


def test_transformer_registers_and_validates_footprints():
    """Planner-level: same-type transform with real component attribution."""
    import quantark.execution.greeks  # noqa: F401 registers ids
    from quantark.execution.scenario import registries

    reg = registries.get_transformer("greek-bump/v1")
    assert reg.allowed_tags == frozenset({"spot", "vol", "rate", "div", "time"})
    runner = registries.get_runner("greek-value/v1")
    assert runner.value_kind == "float"


@pytest.mark.parametrize("bump_id,tag", [
    ("spot_up", "spot"), ("spot_down", "spot"), ("vol_up", "vol"),
    ("rate_up", "rate"), ("div_up", "div"), ("theta", "time"),
])
def test_planner_attributes_each_bump_to_its_tag(bump_id, tag):
    """POSITIVE: plan_scenarios accepts each bump with its declared tag on a
    flat-market TradeState (proves the component extractors read the real
    class fields — plan-gate finding 4). NEGATIVE: the same bump with
    mutation_tags=frozenset() must raise ValidationGateError
    (under-declared), proving attribution actually detected the change."""
    from quantark.execution.errors import ValidationGateError
    from quantark.execution.scenario.planner import plan_scenarios
    # build a legless TradeState base (barrier fixture from the test above),
    # pass resolved=state so no factory is needed; spec with declared tag
    # passes, spec with empty tags raises ValidationGateError.
```

Key intent: the two bitwise tests are the CORE GATE. Adjust constructor kwargs to the real product/engine signatures while implementing (read the fixture products' actual `__init__`), but the assertions stay `==`.

- [ ] **Step 2: Run to verify failure** — `PYTHONPATH=$PWD .venv-or-main-venv python -m pytest -n0 test/execution/test_greek_bump_cells.py -x` → ImportError.

- [ ] **Step 3: Implement `quantark/execution/greeks.py`:**

```python
"""Greek bumps as framework scenario cells (spec 2026-07-20).

Mirrors EquityPosition.get_trade_risk and
GreeksCalculator.calculate_numerical_greeks OPERATION-FOR-OPERATION so
assembled greeks are bitwise-equal to the originals. Any numerical change
here is a parity bug by definition.
"""
import dataclasses
from copy import deepcopy
from typing import Optional, Sequence

from quantark.execution.scenario import registries

GREEK_BUMP_TRANSFORMER_ID = "greek-bump/v1"
GREEK_VALUE_RUNNER_ID = "greek-value/v1"

_BUMP_IDS = ("base", "spot_up", "spot_down", "vol_up", "rate_up", "div_up", "theta")


@dataclasses.dataclass
class TradeState:
    product: object
    pricing_env: object
    cash_legs: tuple
    engine: object
    streams: object
    quantity: float
    greeks_params: object
    warnings: tuple = ()
    theta_degenerate: bool = False


@dataclasses.dataclass(frozen=True)
class GreekBumpCell:
    bump_id: str
    greeks_served: frozenset
    mutation_tags: frozenset


_CELL_TABLE = {
    "spot_up": (frozenset({"delta", "gamma"}), frozenset({"spot"})),
    "spot_down": (frozenset({"delta", "gamma"}), frozenset({"spot"})),
    "vol_up": (frozenset({"vega"}), frozenset({"vol"})),
    "rate_up": (frozenset({"rho"}), frozenset({"rate"})),
    "div_up": (frozenset({"dividend_rho"}), frozenset({"div"})),
    "theta": (frozenset({"theta"}), frozenset({"time"})),
}


def greek_bump_cells(greeks: Sequence[str]) -> tuple:
    requested = set(greeks)
    cells = [GreekBumpCell("base", frozenset({"price"}), frozenset())]
    for bump_id in _BUMP_IDS[1:]:
        served, tags = _CELL_TABLE[bump_id]
        if served & requested:
            cells.append(GreekBumpCell(bump_id, served, tags))
    return tuple(cells)


def _greeks_calculator(state: TradeState):
    from quantark.asset.equity.riskmeasures import GreeksCalculator

    return GreeksCalculator(params=state.greeks_params)


def apply_greek_bump(bump_id: str, state: TradeState, gc) -> TradeState:
    """Same-type bump application, verbatim v023 conventions."""
    bc = gc._bump_config
    env = state.pricing_env
    product = state.product
    if bump_id == "base":
        return state
    if bump_id in ("spot_up", "spot_down"):
        direction = 1.0 if bump_id == "spot_up" else -1.0
        env2 = deepcopy(env)
        env2.spot_quote.spot *= 1.0 + direction * bc.spot_bump
        return dataclasses.replace(state, pricing_env=env2)
    T = product.get_maturity(env)
    if bump_id == "vol_up":
        strike = getattr(product, "strike", env.spot)
        cur_vol = env.get_vol(strike, T)
        env2 = gc._build_vol_bumped_env(env, product, cur_vol, bc.vol_bump, direction=1.0)
        return dataclasses.replace(state, pricing_env=env2)
    if bump_id == "rate_up":
        from quantark.param.rrf import FlatRateCurve

        cur_rate = env.get_rate(T)
        env2 = deepcopy(env)
        env2.rate_curve = FlatRateCurve(cur_rate + bc.rate_bump)
        return dataclasses.replace(state, pricing_env=env2)
    if bump_id == "div_up":
        cur_div = env.get_div_yield(T)
        env2 = gc._build_div_bumped_env(env, product, cur_div, bc.div_bump, direction=1.0)
        return dataclasses.replace(state, pricing_env=env2)
    if bump_id == "theta":
        bumped_date, time_bump, resolved_mode = gc._advance_theta_bump(
            env, bc.time_bump_days, getattr(bc, "time_bump_mode", "auto")
        )
        current_maturity = product.get_maturity(env)
        if time_bump <= 0.0:
            if current_maturity <= 0.0:
                return dataclasses.replace(state, theta_degenerate=True)
            if resolved_mode == "business_days":
                from quantark.util.exceptions import ValidationError

                raise ValidationError(
                    "Business-day theta bump did not advance time: "
                    f"valuation_date={env.valuation_date}, "
                    f"bumped_date={bumped_date}, "
                    f"time_bump_days={bc.time_bump_days}"
                )
            return dataclasses.replace(state, theta_degenerate=True)
        if current_maturity <= time_bump:
            return dataclasses.replace(state, theta_degenerate=True)
        prod2 = deepcopy(product)
        env2 = deepcopy(env)
        env2.valuation_date = bumped_date
        dropped_all = prod2.time_shift(time_bump, bumped_date, env2)
        if dropped_all:
            return dataclasses.replace(state, theta_degenerate=True)
        legs2 = []
        for leg in state.cash_legs:
            shift = getattr(leg, "time_shift", None)
            shifted = shift(time_bump) if callable(shift) else leg
            if shifted is not None:
                legs2.append(shifted)
        return dataclasses.replace(
            state, product=prod2, pricing_env=env2, cash_legs=tuple(legs2)
        )
    raise ValueError(f"unknown bump_id {bump_id!r}")


def greek_bump_transform(base: TradeState, parameters: dict) -> TradeState:
    gc = _greeks_calculator(base)
    return apply_greek_bump(parameters["bump_id"], base, gc)


def _vol_summary(env):
    """Fingerprint the CONCRETE field names of every surface type the
    adapter/quantark construct (plan-gate finding 4): FlatVolSurface stores
    `volatility`, TermStructureVolSurface stores `vols`. VERIFY the real
    attribute names against the classes at implementation time and cover
    each; return None only for genuinely unknown types (conservative
    invalidation)."""
    surface = env.vol_surface
    for attr in ("volatility", "vol"):
        value = getattr(surface, attr, None)
        if value is not None:
            return float(value)
    vols = getattr(surface, "vols", None)
    if vols is not None:
        return tuple(float(v) for v in vols)
    return None


def _rate_summary(env):
    curve = env.rate_curve
    for attr in ("rate", "flat_rate"):
        value = getattr(curve, attr, None)
        if value is not None:
            return float(value)
    rates = getattr(curve, "rates", None)
    if rates is not None:
        return tuple(float(r) for r in rates)
    return None


def _div_summary(env):
    dy = env.div_yield
    if dy is None:
        return 0.0
    for attr in ("div_yield", "yield_value", "q"):
        value = getattr(dy, attr, None)
        if value is not None:
            return float(value)
    yields = getattr(dy, "yields", None)
    if yields is not None:
        return tuple(float(y) for y in yields)
    return None


_COMPONENTS = (
    ("spot", lambda s: float(s.pricing_env.spot)),
    ("vol", lambda s: _vol_summary(s.pricing_env)),
    ("rate", lambda s: _rate_summary(s.pricing_env)),
    ("div", lambda s: _div_summary(s.pricing_env)),
    ("time", lambda s: str(s.pricing_env.valuation_date)),
)


def run_greek_bump(bump_id: str, base: TradeState, bumped: TradeState, gc) -> dict:
    """One solve; returns plain floats/strings mirroring reprice()/price."""
    from quantark.portfolio.equity.position import value_leg

    if bumped.theta_degenerate:
        return {"degenerate": 1.0, "npv": 0.0, "leg_sum": 0.0}
    if base.cash_legs:
        bump_engine = gc._resolve_bump_engine(
            base.product, base.pricing_env, base.engine
        )
        result = bump_engine.price_with_events(
            bumped.product, bumped.pricing_env, streams=base.streams
        )
        # unit notional from the BUMPED env exactly as get_trade_risk's
        # reprice() computes it per call
        unit = _unit_notional(base, bumped.pricing_env)
        out = {"degenerate": 0.0, "npv": float(result.npv)}
        leg_sum = 0.0
        legs = bumped.cash_legs if bump_id == "theta" else base.cash_legs
        per_leg = {}
        for leg in legs:
            pv = float(
                value_leg(leg, result.event_distribution, bumped.pricing_env, unit)
            )
            leg_sum += pv
            per_leg[leg.leg_id] = (leg.name, leg.direction, pv)
        out["leg_sum"] = leg_sum
        if bump_id == "base":
            out["per_leg"] = per_leg
        return out
    # legless: base cell prices on the ORIGINAL engine (v023 raw_price),
    # bump cells on the resolved bump context (calculate_numerical_greeks)
    if bump_id == "base":
        price = float(base.engine.price(base.product, base.pricing_env))
    else:
        bump_engine = gc._resolve_bump_engine(
            base.product, base.pricing_env, base.engine
        )
        price = float(bump_engine.price(bumped.product, bumped.pricing_env))
    return {"degenerate": 0.0, "npv": price, "leg_sum": 0.0}


def _unit_notional(state: TradeState, env) -> float:
    """Mirror EquityPosition._get_unit_notional for the adapter trades."""
    from quantark.portfolio.equity.position import EquityPosition

    pos = EquityPosition(
        product=state.product, quantity=state.quantity,
        entry_price=float(getattr(state.product, "initial_price", 0.0) or 0.0),
        underlying="OTC", engine=state.engine,
        entry_timestamp=env.valuation_date, cash_legs=list(state.cash_legs),
    )
    return pos._get_unit_notional(env)


def assemble_trade_greeks(base_values, bump_values, *, quantity, spot,
                          bump_config, requested) -> dict:
    bc = bump_config
    q = float(quantity)
    base_npv = base_values["npv"]
    base_legs = base_values["leg_sum"]
    product_g = {"price": q * base_npv}
    total_g = {"price": q * base_npv + base_legs}

    def record(name, d_npv, d_legs):
        product_g[name] = q * d_npv
        total_g[name] = q * d_npv + d_legs

    if {"delta", "gamma"} & requested:
        h = spot * bc.spot_bump
        nu, lu = bump_values["spot_up"]["npv"], bump_values["spot_up"]["leg_sum"]
        nd, ld = bump_values["spot_down"]["npv"], bump_values["spot_down"]["leg_sum"]
        if "delta" in requested:
            record("delta", (nu - nd) / (2.0 * h), (lu - ld) / (2.0 * h))
        if "gamma" in requested:
            record("gamma", (nu - 2.0 * base_npv + nd) / h**2,
                   (lu - 2.0 * base_legs + ld) / h**2)
    if "vega" in requested:
        nv, lv = bump_values["vol_up"]["npv"], bump_values["vol_up"]["leg_sum"]
        record("vega", nv - base_npv, lv - base_legs)
    if "rho" in requested:
        nr, lr = bump_values["rate_up"]["npv"], bump_values["rate_up"]["leg_sum"]
        scale = 0.01 / bc.rate_bump
        record("rho", (nr - base_npv) * scale, (lr - base_legs) * scale)
    if "dividend_rho" in requested:
        nd_, ld_ = bump_values["div_up"]["npv"], bump_values["div_up"]["leg_sum"]
        scale = 0.01 / bc.div_bump
        record("dividend_rho", (nd_ - base_npv) * scale, (ld_ - base_legs) * scale)
    if "theta" in requested:
        tv = bump_values["theta"]
        if tv["degenerate"]:
            product_g["theta"], total_g["theta"] = 0.0, 0.0
        else:
            npv_t, legs_t = tv["npv"], tv["leg_sum"]
            product_g["theta"] = q * (npv_t - base_npv)
            total_g["theta"] = q * (npv_t - base_npv) + (legs_t - base_legs)
    leg_pvs = dict(base_values.get("per_leg", {}))
    return {"product": product_g, "total": total_g, "leg_pvs": leg_pvs}


def assemble_product_greeks(*, base_price, bump_values, bump_config,
                            requested) -> dict:
    bc = bump_config
    out = {}
    if "price" in requested:
        out["price"] = base_price
    if {"delta", "gamma"} & requested:
        up = bump_values["spot_up"]["npv"]
        down = bump_values["spot_down"]["npv"]
        # scale=spot enters via _calculate_sensitivity(bump=spot_bump, scale=spot)
        # -> denominators (2*scale*bump) and (scale*bump)**2 — but the CALLER
        # (adapter) multiplies by spot/spot^2 again, so keep raw per-spot form:
        spot = bump_values["__spot__"]
        if "delta" in requested:
            out["delta"] = (up - down) / (2.0 * spot * bc.spot_bump)
        if "gamma" in requested:
            out["gamma"] = (up - 2.0 * base_price + down) / (spot * bc.spot_bump) ** 2
    if "vega" in requested:
        out["vega"] = bump_values["vol_up"]["npv"] - base_price
    if "theta" in requested:
        tv = bump_values["theta"]
        out["theta"] = 0.0 if tv["degenerate"] else tv["npv"] - base_price
    if "rho" in requested:
        raw = bump_values["rate_up"]["npv"] - base_price
        out["rho"] = raw * (0.01 / bc.rate_bump)
    if "dividend_rho" in requested:
        raw = bump_values["div_up"]["npv"] - base_price
        out["dividend_rho"] = raw * (0.01 / bc.div_bump)
    return out


def greek_value_runner(cell, resolved, child_context):
    parameters = dict(cell.parameters)
    base = resolved.base_inputs
    bumped = resolved.transformed
    gc = _greeks_calculator(base)
    values = run_greek_bump(parameters["bump_id"], base, bumped, gc)
    economics = [("npv", values["npv"]), ("leg_sum", values["leg_sum"]),
                 ("degenerate", values["degenerate"])]
    for leg_id, (name, direction, pv) in values.get("per_leg", {}).items():
        economics.append((f"leg_pv::{leg_id}", pv))
        economics.append((f"leg_name::{leg_id}", str(name)))
        economics.append((f"leg_dir::{leg_id}", float(direction)))
    for i, warning in enumerate(base.warnings):
        economics.append((f"warning::{i}", str(warning)))
    economics.append(("spot", float(base.pricing_env.spot)))
    # Deterministic manifest (plan-gate finding 3): non-null, identical
    # serial vs processes, and version-bearing so source/wheel skew is
    # detectable at the runtime-duality gate.
    from quantark.execution.cache.fingerprint import try_fingerprint
    from quantark.execution.manifest import build_versions, platform_tag

    manifest_fp = try_fingerprint(
        ("greek-value/v1", parameters["bump_id"], cell.cell_fingerprint,
         build_versions(), platform_tag())
    )
    return values["npv"], tuple(economics), manifest_fp


if __name__ == "quantark.execution.greeks":
    registries.register_transformer(
        GREEK_BUMP_TRANSFORMER_ID, greek_bump_transform,
        allowed_tags=frozenset({"spot", "vol", "rate", "div", "time"}),
        components=_COMPONENTS,
        covered_fields=None,
    )
    registries.register_runner(
        GREEK_VALUE_RUNNER_ID, greek_value_runner, value_kind="float"
    )
```

Implementation notes to honor while making tests pass:
- `assemble_product_greeks` needs the base spot: pass `bump_values["__spot__"] = env.spot` from the caller (test and adapter both set it). If ugly in practice, change the signature to accept `spot=` explicitly — keep test and adapter consistent.
- `leg.direction` may be an enum — store `str(leg.direction)` if `float()` fails; the adapter only needs `name` and `pv`.
- Verify `value_leg` and `EquityPosition` import paths (`quantark.portfolio.equity.position`); fix imports to the real module layout.
- Theta legless path: `calculate_numerical_theta` resolves the bump engine BEFORE base-price ensure; but with `base_price` supplied by the caller the only engine use is the theta solve — mirror `run_greek_bump`'s bump-engine choice (resolved bump context), which matches `calculate_numerical_theta`'s `engine.price(product_theta, env_theta)` on the resolved engine.
- Registration must be under the module's canonical name (shown) so spawn children re-register by import.

- [ ] **Step 4: Run the core gate** — both bitwise tests pass: `pytest -n0 test/execution/test_greek_bump_cells.py -v`. If ANY assertion is `!=`: STOP and root-cause (dump both operand hexes via `float.hex()`); do not loosen.

- [ ] **Step 5: Commit** `feat(execution): greek-bump scenario cells with bitwise assembler gate`.

---

### Task 2: `PricingSession.run_scenario_plans` (multi-base packing)

**Files:**
- Modify: `quantark/execution/scenario/worker.py` (`run_plan_processes` → per-cell spec payloads; new `run_plans_processes`)
- Modify: `quantark/execution/backends/processes.py` (`iter_ordered`: `spec_payload` may be a list parallel to `cells`)
- Modify: `quantark/execution/api.py` (new method)
- Modify: `quantark/execution/scenario/runner.py` (serial multi-plan helper `run_plans_serial`)
- Test: `test/execution/test_run_scenario_plans.py`

**Interfaces (produced):**
- `PricingSession.run_scenario_plans(plan_inputs, engine_factory=None, *, collect_errors=False) -> list[list[ScenarioOutcome|PricingFailure]]` where `plan_inputs = [(base_ref_or_request, [ScenarioSpec, ...]), ...]`; outcome lists align with input order. Serial and processes backends only; threads/dask raise `CapabilityError`.
- **Per-plan error boundary (plan-gate finding 2):** with `collect_errors=True`, a BASE-RESOLUTION failure (factory raise inside `resolve_base`/`plan_scenarios`) or a PLANNING failure (transformer/validation raise) in ONE plan yields `[PricingFailure(item_id=f"plan:{index}", ...)] * len(specs)` for that plan (aligned, typed, carrying the original exception type+message) while every other plan resolves, plans, and executes normally. With `collect_errors=False`, first failure raises (existing semantics). Tests MUST cover: one factory-raise plan + one transformer-raise plan + one healthy plan, serial AND processes, asserting the healthy plan completes and failure messages carry the original exception text.
- `iter_ordered(cells, spec_payload, workers, window, ...)` — `spec_payload` is a dict (broadcast, existing behavior) OR a list of dicts (`len == len(cells)`, per-cell submission `pool.submit(run_worker_cell, spec_payloads[i], cells[i], engine_factory_id)`).

- [ ] **Step 1: Failing tests** (toy fixtures, existing `execution.scenario_process_helpers`):

```python
def test_multi_plan_serial_matches_single_plan_runs():
    plans = [(_toy_base(), [_toy_spec(f"a{i}", float(i)) for i in range(3)]),
             (_toy_base(), [_toy_spec(f"b{i}", -float(i)) for i in range(2)])]
    with PricingSession() as session:
        grouped = session.run_scenario_plans(plans, "toy-engine/v1")
    with PricingSession() as session:
        first = session.run_scenarios(plans[0][0], plans[0][1], "toy-engine/v1")
    assert [o.value for o in grouped[0]] == [o.value for o in first]
    assert len(grouped[1]) == 2

def test_multi_plan_processes_bitwise_and_ordered():
    plans = [...same...]
    with PricingSession(_process_context()) as session:
        grouped = session.run_scenario_plans(plans, "toy-engine/v1")
    # bitwise equal serial, per-plan caller order preserved

def test_multi_plan_duplicate_ids_allowed_across_plans_not_within():
    # same scenario_id in two different plans is fine; duplicate within one raises

def test_multi_plan_collect_errors_isolates_per_cell():
    # failing runner cell in plan 0 -> PricingFailure in grouped[0], plan 1 clean

def test_multi_plan_threads_backend_rejected():
    # CapabilityError
```

- [ ] **Step 2: verify failures.**
- [ ] **Step 3: Implement.** Key mechanics:
  - `run_plans_processes(plans_and_bases, engine_factory, context, collect_errors)`: per plan run the EXISTING validations from `run_plan_processes` (base-ref requirement, engine-factory string requirement, float value_kind); `build_worker_spec(plan_i, base_i, context, workers)` per plan; flatten `cell_payloads` and parallel `spec_payloads`; positions = global ordinals; single retry-driver loop copied from `run_plan_processes` (positions map back to `(plan_index, cell_index)` via a prefix-offset table); `_consume` writes into `results[plan_index][cell_index]`.
  - `run_plan_processes` becomes a thin wrapper: `run_plans_processes([(plan, base)], ...)[0]` — zero behavior change for v030 (its gates prove it).
  - Window/workers resolution unchanged (one shared window across all cells — this IS the packing win).
  - Serial: loop plans → existing `_run_serial` per plan with its own resolved base (factory runs once per plan).
  - `api.run_scenarios` keeps its exact current signature/behavior; `run_scenario_plans` resolves each base via the session (mirror the resolved-base handling `run_scenarios` uses).
- [ ] **Step 4: Full execution suite** `pytest test/execution -q` — all green including v030-era processes gates (wrapper equivalence).
- [ ] **Step 5: Commit** `feat(execution): run_scenario_plans multi-base packing API`.

---

### Task 3: Adapter branch + `otc_trade_state.py` factory

**Files (adapter repo, branch `greek-bump-v031` off main):**
- Create: `/Users/fuxinyao/otc-price-adapter/otc_trade_state.py`
- Test: `/Users/fuxinyao/otc-price-adapter/tests/test_otc_trade_state.py`

**Interfaces (produced):**
- Factory id `otc-trade-state/v1`, registered AT MODULE LEVEL in `otc_trade_state.py` (defining-module rule). Payload pairs: everything `settings_to_payload(settings)` emits PLUS `("row", row_to_parameters(row))`, `("trade_id", str)`, `("ordinal", int)`.
- `build_trade_state(payload: dict) -> TradeState`: `settings = build_settings(<settings sub-payload>)`; `row = parameters_to_row(payload["row"]); row.name = payload["trade_id"]`; then EXACTLY v023's build sequence: `env = legacy.build_pricing_env(row, settings)`; `bundle = v023.native_feature_bundle(legacy.build_product(row, settings), row)`; `sign = legacy.pricing_direction_sign(row)`; `engine_warning = legacy.apply_phoenix_mc_no_ki_lifecycle_workaround(bundle.product, bundle.model)`; `cash_legs = v023.native_autocall_cash_legs(row, settings)`; streams via a throwaway `EquityPosition._required_streams()` when legs exist else None; returns `TradeState(product=bundle.product, pricing_env=env, cash_legs=tuple(cash_legs), engine=bundle.engine, streams=streams, quantity=sign, greeks_params=legacy.engine_params(settings), warnings=(bundle.warning, engine_warning, model_tag...))` — carry `bundle.model` and warnings so the parent can byte-match v023's `model`/`pricing_warning` strings (store as `warnings=(f"model::{bundle.model}", f"bundle_warning::{bundle.warning or ''}", f"engine_warning::{engine_warning or ''}")`).
- Also `trade_base_ref(settings, row, ordinal) -> BaseInputsRef` helper.

- [ ] Steps: failing test (factory rebuild == direct v023 build: price the base cell through `run_greek_bump` vs `bundle.engine.price`/`price_with_events` directly — exact equality; UnsupportedStructureError propagates with v023's message), implement, pass, commit.

---

### Task 4: `otc_quantark_pricer_v031.py` — two-phase driver + assembly + CLI

**Files:**
- Create: `/Users/fuxinyao/otc-price-adapter/otc_quantark_pricer_v031.py`
- Test: `/Users/fuxinyao/otc-price-adapter/tests/test_otc_quantark_pricer_v031.py`

**Interfaces:** `NATIVE_VERSION_TAG = "qa031_cells"` is used ONLY for output
file suffixing and logging — **the row-level `model` and `pricing_warning`
strings must be byte-identical to v023's** (plan-gate finding 1): reuse
`pricer_v023.NATIVE_VERSION_TAG` (`qa023_native`) and v023's literal warning
strings in assembly, exactly as v030 does by re-executing `price_row`. The
24-column parity surface includes `model` and `pricing_warning` unchanged.
`price_frame(frame, settings, retries=1, window=None) -> DataFrame` (same 24
`OUTPUT_COLUMNS` as v030, import from v030); CLI mirrors v030
(`--workers/--retries/--max-in-flight/--limit/--trade-id`), canonical-import
`__main__` bootstrap; output suffix `_qa031_cells`.

Driver algorithm (complete, mirror precisely):
1. For each row ordinal `i`: `base_ref_i = trade_base_ref(settings, row, i)`; specs from `greek_bump_cells(GREEKS7)`: `ScenarioSpec(scenario_id=f"r{i:04d}::{bump_id}", transformer_id="greek-bump/v1", parameters=(("bump_id", bump_id),), mutation_tags=cell.mutation_tags, required_capabilities=frozenset({"runner:greek-value/v1"}))`.
2. **Phase A**: `run_scenario_plans([(base_ref_i, [base_spec_i]) for all i], None, collect_errors=True)`. A `PricingFailure` (factory raise = build error, or base-solve error) ⇒ row `i` becomes a v023-identical error row: `pricing_status="error"`, `unsupported_reason=str(exc message)` — assert byte-equality in tests (the failure's `error.message` must equal `str(exc)` v023 produces; both call the same code).
3. **Phase B**: healthy rows only: `run_scenario_plans([(base_ref_i, bump_specs_i) ...], None, collect_errors=True)`. Any failure ⇒ that row error with that message.
4. Assembly per healthy row, mirroring `price_row` verbatim:
   - Parse base economics: `npv`, `leg_sum`, `leg_pv::*`, `warning::*` (model tag, bundle warning, engine warning).
   - Cash-leg rows: `assemble_trade_greeks(...quantity=sign, spot=env spot...)` — spot comes from the base env; emit it from the base cell as economics `("spot", env.spot)` (add in Task 1 runner: `economics.append(("spot", float(base.pricing_env.spot)))`). Then EXACT v023 lines: `delta*spot`, `gamma*spot*spot*0.01`, leg-PV mapping through `legacy.PV_ADJUSTMENT_COLUMN_MAP` with the `UnsupportedStructureError` on unknown leg names, `adjustment_warning = "autocallable cash legs valued with QuantArk 0.2.3 native cashleg/event distribution"`, `model=f"{model}:{NATIVE_VERSION_TAG}"`, warning join `"; ".join(filter(None, [bundle_warning, engine_warning, adjustment_warning]))`.
   - Legless rows: `assemble_product_greeks` + parent-side `legacy.deterministic_adjustments(row, settings)` + `legacy.cash_leg_greeks(row, settings)` + the sign-scaled arithmetic copied line-for-line from v023 lines 239–275.
5. `_outcomes_to_frame`-style construction with `OUTPUT_COLUMNS`, empty-book schema handling as v030.

Release-gate tests (all in the new test file; subset via `diverse_subset`, full book `@pytest.mark.slow`):
- `test_v031_serial_matches_v023_exactly` (frame `check_exact=True`)
- `test_v031_processes_match_serial_completely` (complete-payload validator + exact frame + non-null manifest-fingerprint equality per cell across serial/processes — plan-gate finding 3)
- `test_error_row_byte_parity` (unsupported structure)
- `test_base_failure_suppresses_bump_cells` (monkeypatch a counting observer around `run_scenario_plans` phase B input: the broken row contributes NO bump specs)
- `test_duplicate_trade_id_book_prices_row_for_row` (duplicate index frame of 2 rows)
- `test_empty_book_keeps_output_schema`
- `test_retries_and_window_plumbing` (session context fields)
- `test_v031_cli_entry_point[1,2]` (subprocess, `--limit 6`, timeout 3600)
- `@slow test_v031_full_book_exact`

- [ ] Steps: write tests → fail → implement → subset gates green → commit `feat: v031 framework-native greek-bump pricer`.

---

### Task 5: Benchmark + README

**Files:** Create `/Users/fuxinyao/otc-price-adapter/benchmark_v031.py` (guarded main; NEVER heredoc — stdin parents cannot spawn workers); Modify `README.md`.

- [ ] Benchmark: 16-row subset, exact-parity asserted per rep; table: v030 processes x4 (window 8) vs v031 processes x4 (window 8) vs v031 x8/x14; report cell-count, packing effect, per-cell rebuild overhead (serial v031 vs serial v023 wall-clock). Honest framing per spec D4 (no reuse claims). README v031 section mirrors v030's (exactness guarantees, gates, benchmark, CLI).
- [ ] Commit `docs: v031 benchmark + README`.

---

### Task 6: quantark 0.3.0rc2 wheel + adapter final gate

**Files:** quant-ark `pyproject.toml` + `quantark/__init__.py` (`0.3.0rc2`), `CHANGELOG.md` (greeks layer + run_scenario_plans + spawn-main preflight already listed); adapter `pyproject.toml` (`quantark==0.3.0rc2`, uv source → new wheel path), `vendor/quantark-0.3.0rc2-py3-none-any.whl`, `uv.lock`.

- [ ] quant-ark full suite green (known quad golden excepted) → bump version → build wheel (`python -m build --wheel`) at the merge commit → record sha256.
- [ ] Vendor wheel on adapter MAIN as an artifact commit (codex ENOBUFS rule; `.gitattributes` already covers `vendor/*.whl`), rebase branch, relock (`uv sync --locked` from clean checkout), rerun FULL adapter default suite against the wheel (no PYTHONPATH override) — 100% green required. The wheel-run gate additionally asserts `importlib.metadata.version("quantark") == "0.3.0rc2"` and records the wheel sha256 in the README (plan-gate finding 3: runtime-duality must detect version skew, and cell manifests embed `build_versions()`).
- [ ] Commit(s): quant-ark `chore: 0.3.0rc2`; adapter `chore: pin quantark 0.3.0rc2 wheel`.

---

## Self-review notes
- Spec coverage: D1 (Task 4 whole book), D2 (Tasks 1/4 bitwise gates), D3 (Tasks 1–2), D4 (Task 5 framing), D5 (Task 4 file layout), D6 (Task 6); spec-gate findings: 1 (Task 1 transformer + Task 3 per-row base), 2 (Task 4 two-phase), 3 (Task 4 release gates + Task 6 wheel duality), 4 (ordinal ids Task 4 + duplicate test).
- Type consistency: `TradeState` fields used by Tasks 3–4 match Task 1; `run_scenario_plans` signature consistent Tasks 2/4.
- Known simplifications to validate during implementation (diagnose-don't-tolerate): exact fixture constructor signatures in Task 1; `streams`/`value_leg` import paths; `leg_dir` float coercion; `assemble_product_greeks` spot plumbing.

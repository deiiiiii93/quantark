"""DCN two-surface PDE solver tests: solver unit gates, event-operator
ordering, and the MC<->PDE cross-validation gate (acceptance criterion #3)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.dcn_pde_solver import (
    DCNPDEEngine,
    apply_dcn_events,
)
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams
from quantark.util.exceptions import ValidationError

from dcn_fixtures import (
    DCN_A,
    FLAT,
    SAMPLE_DCN_A,
    SAMPLE_DCN_B,
    flat_env,
    make_dcn,
)

# Decided validation setting: >= 2^17 Sobol paths, and stderr must be
# < 2bp*N (raise paths, never widen the gate). DCN-B's loss leg carries more
# variance, so it needs one more doubling than DCN-A.
GATE_SETTINGS = {"DCN-A": (2 ** 20, 16), "DCN-B": (2 ** 21, 32)}


def test_core_engine_prices_against_the_exact_recursion():
    # spec 3C §5.4: the hand-rolled stencil and theta step are gone; the core engine is checked against the
    # deterministic DCN recursion (the full cell sweep is test/pde_routes/test_dcn.py)
    from pde_routes.cases import dcn_case
    from pde_routes.dcn_reference import dcn_exact
    product, env = dcn_case("A")
    want = dcn_exact(product, env, spots=[float(env.spot)])["pv"][0]
    assert 1e4 * abs(DCNPDEEngine().price(product, env) - want) / product.notional <= 1.0   # bp


def test_event_operator_ordering():
    # backward order: coupon injection -> KO overwrite -> KI projection
    s_grid = np.array([4000.0, 4600.0, 5000.0, 6100.0])
    ki_mask = s_grid <= 4500.0
    cpn_mask = s_grid >= 4800.0
    ko_mask = s_grid >= 6000.0
    v0 = np.zeros(4)
    v1 = np.full(4, -100.0)
    v0_out, v1_out = apply_dcn_events(
        v0, v1, cpn_mask, ko_mask, ki_mask, coupon_amount=5.0, ko_amount=7.0
    )
    assert v0_out[3] == 7.0 and v1_out[3] == 7.0   # KO wins over coupon
    assert v0_out[2] == 5.0                        # coupon region, no KO
    assert v0_out[0] == -100.0                     # KI projection onto v1
    assert v0_out[1] == 0.0                        # no event region
    assert v1_out[0] == -100.0 and v1_out[2] == -100.0  # v1 never gets coupons


@pytest.mark.parametrize(
    "contract,ident,direction",
    [
        (SAMPLE_DCN_A, "DCN-A", DCNDirection.BUYER),
        (SAMPLE_DCN_B, "DCN-B", DCNDirection.SELLER),
    ],
    ids=["DCN-A", "DCN-B"],
)
def test_mc_pde_cross_validation_gate(contract, ident, direction):
    p = make_dcn(contract, direction=direction)
    env = flat_env(**FLAT, spot=contract["initial_price"])
    paths, batches = GATE_SETTINGS[ident]
    mc = DCNMCEngine(
        num_paths=paths, num_batches=batches, seed=42
    ).price_detailed(p, env)
    # an explicit 2401 points: the shared presets cap at 2000 (spec 3C §5.4)
    pde = DCNPDEEngine(params=PDEParams(grid=GridConfig(points=2401, max_points=2401))).price(p, env)
    notional = contract["notional"]
    assert mc.std_error < 2e-4 * notional, "stderr gate: raise paths, not tolerance"
    gate = max(3.0 * mc.std_error, 5e-4 * notional)
    assert abs(mc.pv - pde) < gate, (
        f"|MC-PDE| = {abs(mc.pv - pde):.2f} exceeds gate {gate:.2f} "
        f"(MC={mc.pv:.2f} +/- {mc.std_error:.2f}, PDE={pde:.2f})"
    )


def test_pde_buyer_seller_negation_exact():
    env = flat_env(**FLAT)
    b = DCNPDEEngine().price(make_dcn(DCN_A), env)
    s = DCNPDEEngine().price(make_dcn(DCN_A, direction=DCNDirection.SELLER), env)
    assert s == -b


def test_pde_knocked_in_seed_prices_v1():
    env = flat_env(**FLAT)
    base = DCNPDEEngine().price(make_dcn(DCN_A), env)
    ki = DCNPDEEngine().price(make_dcn(DCN_A, knocked_in_at_valuation=True), env)
    assert ki < base  # V1 <= V0 pointwise: no coupons + short put


def test_pde_valuation_date_breach_selects_knocked_in_surface():
    env = flat_env(**FLAT, spot=4000.0)  # below the 4500 KI barrier
    observed_today = DCNPDEEngine().price(make_dcn(DCN_A), env)
    seeded = DCNPDEEngine().price(
        make_dcn(DCN_A, knocked_in_at_valuation=True), env
    )
    assert observed_today == seeded


def test_pde_deterministic():
    env = flat_env(**FLAT)
    p = make_dcn(DCN_A)
    assert DCNPDEEngine().price(p, env) == DCNPDEEngine().price(p, env)


def _cfg(engine):
    return engine.grid_binder.config


def test_dcn_takes_the_shared_presets():
    # spec 3C §5.4 (decision 20): 200 / 400 / 800 points, 4 / 4 / 8 steps per day
    got = {a: _cfg(DCNPDEEngine(params=PDEParams(accuracy=a))) for a in ("fast", "standard", "high")}
    assert {a: (c.points, c.steps_per_day) for a, c in got.items()} == {
        "fast": (200, 4.0), "standard": (400, 4.0), "high": (800, 8.0)}


def test_dcn_explicit_points_above_2000_need_max_points():
    with pytest.raises(ValidationError, match="max_points"):
        DCNPDEEngine(params=PDEParams(grid=GridConfig(points=2401))).price(make_dcn(DCN_A), flat_env(**FLAT))
    assert _cfg(DCNPDEEngine(params=PDEParams(grid=GridConfig(points=2401, max_points=2401)))).points == 2401


@pytest.mark.parametrize("params,moves", [
    (PDEParams(grid=GridConfig(steps_per_day=1.0)), True),                    # +1.6e2
    (PDEParams(grid=GridConfig(time_steps=100)), True),                       # +3.6e2
    (PDEParams(event_projection="cell_average"), False),                      # the core default
    (PDEParams(event_projection="nodal"), True),                              # +8.5e2: the legacy masks (decision 23)
    (PDEParams(stepper=CrankNicolson(terminal_damping_steps=1, event_damping_steps=1)), False),   # the defaults
    (PDEParams(stepper=CrankNicolson(terminal_damping_steps=2, event_damping_steps=2)), True),    # +1.1e1
    (PDEParams(stepper=CrankNicolson(theta=1.0)), True),                      # +9.6e1
])
def test_dcn_honours_the_fields_the_route_refused(params, moves):
    # spec 3C §5.4: the DCN is a core product - its route table and refusals are gone
    env = flat_env(**FLAT, spot=DCN_A["initial_price"])
    p = make_dcn(DCN_A)
    base = DCNPDEEngine().price(p, env)
    assert (DCNPDEEngine(params=params).price(p, env) != base) is moves


def test_dcn_removed_names():
    # the route's 201-point floor is gone with its table: the shared fast preset is 200 points
    with pytest.raises(ValidationError, match="GridConfig\\(points"):
        DCNPDEEngine(num_space_nodes=801)


def test_price_detailed_layout_is_a_removed_name():
    # spec 3C §5.4: REMOVED_DCN_NAMES is extended with what 0.10.0 removes - a ValidationError naming the
    # replacement, never a TypeError
    env = flat_env(**FLAT, spot=DCN_A["initial_price"])
    with pytest.raises(ValidationError, match="GridConfig"):
        DCNPDEEngine().price_detailed(make_dcn(DCN_A), env, layout=object())

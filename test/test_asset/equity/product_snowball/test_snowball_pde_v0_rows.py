"""The snowball PDE's not-knocked-in (V0) Dirichlet rows, by edge (accrual follow-up item 7, decision D7).

Each V0 edge is either the knock-out side (upper on a standard product, lower on a reverse one) or the knock-in side.

* The knock-out edge carries the knock-out payoff once it lies beyond every knock-out level; otherwise it carries the
  contract's no-event value there.
* The knock-in edge copies the knocked-in row under continuous knock-in once it lies at or beyond the knock-in level
  (discrete knock-in: ``_knock_in_side_row``); otherwise it carries the contract's no-event value there.
* The no-event value is the discounted V0 maturity payoff's asymptote: principal + fixed rebate, or, on a call-rebate
  contract (which pays NO fixed rebate), principal + the participation's forward intrinsic value at that edge.

A contract that cannot knock out or knock in on the grid is certain to mature in V0, so its price is a closed form:
(principal + rebate x tenor^a) x DF(T), or, on a call-rebate contract, principal x DF(T) + participation x tenor^a x
the Black-Scholes call. These controls price at the 1e-6 level in both orientations once the rows are right.
"""

import math
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.product.option.accrual import leg_annualized
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig, CouponPayType
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import ObservationType
from quantark.volmodels.heston import HestonParams

S0, R, VOL, T = 100.0, 0.03, 0.2, 2.0
CALL_STRIKE, CALL_PARTICIPATION = 100.0, 0.2
# the certain-V0 price gap of the fixed-rebate controls is ~7e-7 to ~2.5e-6 in both orientations. The call rebate's
# kink at the spot leaves 3e-6 to 1.3e-5 (the reverse grid's upper edge, ~316, is the nearer one): the PDE's own call
# discretization, which neither accuracy="high" nor 800 / 1600 points removes, in both orientations (the European PDE
# misprices the same Black-Scholes call by 1.7e-4 on its default grid)
CLOSED_FORM_ABS = 1e-5
CALL_CLOSED_FORM_ABS = 2e-5
# The 2D Heston reverse control knocks out at 2 (S down 98%), not 20: under W3's Heston (v0 = theta = 0.04, kappa 1.5,
# sigma 0.3, rho -0.5) the knock-out at 20 carries a real premium, +4.90e-3 +- 2.0e-4 by independent MC (b05), so the
# closed form is not its price. At 2 the same MC sees no knock-out on 1e8 paths (sub 10 and 40): |premium| <= 18.84 x 3
# / 1e8 = 5.7e-7 at 95% (c01), below the control's 1e-6. The 1D BSM controls keep 20 (premium ~1e-8).
HESTON_REVERSE_CONTROL_KO = 2.0


def _env(grid_vol: bool = False) -> PricingEnvironment:
    vol = (GridVolSurface(list(S0 * np.exp(np.linspace(-1.5, 1.5, 13))), [0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
                          np.full((6, 13), VOL)) if grid_vol else FlatVolSurface(volatility=VOL))
    return PricingEnvironment(spot_quote=SpotQuote(spot=S0, asset_name="X"), vol_surface=vol,
                              rate_curve=FlatRateCurve(rate=R), valuation_date=datetime(2024, 1, 2))


def _barrier(ko: float, ki: float, continuous: bool) -> BarrierConfig:
    obs = [(i + 1) / 24 * T for i in range(24)]
    return BarrierConfig(
        ko_barrier=ko, ko_rate=0.15,
        ko_observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ko) for t in obs]),
        ki_barrier=ki,
        ki_observation_type=ObservationType.CONTINUOUS if continuous else ObservationType.DISCRETE,
        ki_continuous=continuous,
        ki_observation_schedule=None if continuous else ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ki) for t in obs]))


def _certain_v0(rebate_rate: float, annualized: bool, reverse: bool = False, call: bool = False,
                continuous: bool = False, ko: float = None) -> SnowballOption:
    """Knock-out 5x beyond the spot and knock-in 100x beyond it, each on its own side: neither is reachable.
    ``ko`` overrides the knock-out level (the 2D Heston reverse control: ``HESTON_REVERSE_CONTROL_KO``)."""
    ko_default, ki = (20.0, 10000.0) if reverse else (500.0, 1.0)
    ko = ko_default if ko is None else ko
    extra = (dict(call_rebate_enabled=True, call_strike=CALL_STRIKE, call_participation_rate=CALL_PARTICIPATION)
             if call else {})
    return SnowballOption(
        initial_price=S0, strike=S0, barrier_config=_barrier(ko, ki, continuous),
        payoff_config=PayoffConfig(rebate_rate=rebate_rate, include_principal=True, **extra),
        accrual_config=AccrualConfig(is_annualized_rebate=annualized), maturity=T, tenor=T, is_reverse=reverse)


def _closed_form(product, env, rebate_rate: float, annualized: bool, call: bool) -> float:
    tenor = product.get_contract_tenor(env) if annualized else 1.0
    df = math.exp(-R * T)
    if call:
        d1 = (math.log(S0 / CALL_STRIKE) + (R + 0.5 * VOL * VOL) * T) / (VOL * math.sqrt(T))
        d2 = d1 - VOL * math.sqrt(T)
        n = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))  # noqa: E731
        bs_call = S0 * n(d1) - CALL_STRIKE * df * n(d2)
        return S0 * df + CALL_PARTICIPATION * tenor * bs_call
    return (S0 + rebate_rate * S0 * tenor) * df


def _t0_rows(solver):
    """The V0 rows the march set last: the valuation step's (lower, upper)."""
    return solver._rows_v0_stash[4], solver._rows_v0_stash[5]


ORIENTATION = pytest.mark.parametrize("reverse", [False, True], ids=["standard", "reverse"])
ANNUALIZED = pytest.mark.parametrize("annualized", [False, True], ids=["flat", "annualized"])


class TestCertainV0ClosedForm:
    """The price of a contract certain to mature in V0 is its closed form, whichever side the knock-out sits on."""

    @ORIENTATION
    @ANNUALIZED
    @pytest.mark.parametrize("continuous", [False, True], ids=["discrete_ki", "continuous_ki"])
    def test_fixed_rebate(self, reverse, annualized, continuous):
        env = _env()
        product = _certain_v0(0.10, annualized, reverse=reverse, continuous=continuous)
        price = SnowballPDESolver().price(product, env)
        assert price == pytest.approx(_closed_form(product, env, 0.10, annualized, False), abs=CLOSED_FORM_ABS)

    @ORIENTATION
    @ANNUALIZED
    @pytest.mark.parametrize("rebate_rate", [0.0, 0.10])
    def test_call_rebate(self, reverse, annualized, rebate_rate):
        """A call-rebate contract pays no fixed rebate: ``rebate_rate`` (the factory sets it to the KO rate) must not
        reach the price through any row."""
        env = _env()
        product = _certain_v0(rebate_rate, annualized, reverse=reverse, call=True)
        price = SnowballPDESolver().price(product, env)
        assert price == pytest.approx(_closed_form(product, env, rebate_rate, annualized, True),
                                      abs=CALL_CLOSED_FORM_ABS)

    @ORIENTATION
    @ANNUALIZED
    def test_local_vol_1d(self, reverse, annualized):
        """The local-vol 1D solver marches the same adapter and rows (a flat surface: the BSM closed form)."""
        env = _env(grid_vol=True)
        product = _certain_v0(0.10, annualized, reverse=reverse)
        price = LocalVolSnowballPDESolver().price(product, env)
        assert price == pytest.approx(_closed_form(product, env, 0.10, annualized, False), abs=CLOSED_FORM_ABS)


class TestRowsByEdge:
    """The valuation step's V0 rows, branch and value."""

    @ANNUALIZED
    @pytest.mark.parametrize("rebate_rate", [0.0, 0.10])
    def test_call_rebate_lower_row_pays_no_fixed_rebate(self, annualized, rebate_rate):
        env = _env()
        solver = SnowballPDESolver()
        solver.price(_certain_v0(rebate_rate, annualized, call=True), env)
        lower, _ = _t0_rows(solver)
        assert lower.branch == "v0_lower_call_rebate"
        assert lower.value == pytest.approx(S0 * math.exp(-R * T), rel=1e-14)

    @ANNUALIZED
    def test_reverse_knock_out_edge_is_the_lower_row(self, annualized):
        env = _env()
        standard, reverse = SnowballPDESolver(), SnowballPDESolver()
        standard.price(_certain_v0(0.10, annualized), env)
        product = _certain_v0(0.10, annualized, reverse=True)
        reverse.price(product, env)
        lower, upper = _t0_rows(reverse)
        _, standard_upper = _t0_rows(standard)
        assert standard_upper.branch == "v0_upper_ko"
        # the same knock-out record pays on the reverse product's lower edge
        assert lower.branch == "v0_lower_ko"
        assert lower.value == standard_upper.value
        assert lower.terms == standard_upper.terms
        # the knock-in side (S -> inf) is beyond no knock-in level: the no-event value
        assert upper.branch == "v0_upper_principal_rebate"
        assert upper.value == pytest.approx(_closed_form(product, env, 0.10, annualized, False), rel=1e-14)

    def test_reverse_continuous_knock_in_edge_copies_the_knocked_in_row(self):
        """KO 97 / continuous KI 115: the upper edge lies beyond the knock-in level."""
        env = _env()
        product = SnowballOption(
            initial_price=S0, strike=S0, barrier_config=_barrier(97.0, 115.0, continuous=True),
            payoff_config=PayoffConfig(rebate_rate=0.10, include_principal=True), maturity=T, tenor=T,
            is_reverse=True)
        solver = SnowballPDESolver()
        solver.price(product, env)
        lower, upper = _t0_rows(solver)
        assert lower.branch == "v0_lower_ko"
        assert upper.branch == "v0_upper_copy_ki" and upper.copy_from_ki and not upper.terms
        assert upper.value == solver._grid_v1[-1, 0]

    @ORIENTATION
    def test_continuous_knock_in_off_the_grid_is_not_copied(self, reverse):
        """KI 100x beyond the spot never reaches the grid: the knock-in edge is not knocked in."""
        env = _env()
        product = _certain_v0(0.10, False, reverse=reverse, continuous=True)
        solver = SnowballPDESolver()
        solver.price(product, env)
        lower, upper = _t0_rows(solver)
        ki_edge = upper if reverse else lower
        assert ki_edge.branch == ("v0_upper_principal_rebate" if reverse else "v0_lower_principal_rebate")
        assert not ki_edge.copy_from_ki
        assert ki_edge.value == pytest.approx(_closed_form(product, env, 0.10, False, False), rel=1e-14)

    def test_reverse_phoenix_call_rebate_is_a_put_at_the_lower_edge(self):
        """A reverse Phoenix's call rebate pays max(K - S, 0) (PhoenixOption.get_maturity_payoff_v0): its no-event
        lower row is principal + the put's forward intrinsic value, its upper row principal alone."""
        env = _env()
        product = PhoenixOption(
            initial_price=S0, strike=S0, maturity=T, contract_multiplier=1.0, is_reverse=True,
            barrier_config=_barrier(1.0, 10000.0, continuous=False),   # neither barrier reaches the grid
            coupon_config=CouponBarrierConfig(coupon_barrier=1.0, coupon_rate=0.02,
                                              coupon_pay_type=CouponPayType.INSTANT,
                                              day_count_convention=DayCountConvention.ACT_365),
            payoff_config=PayoffConfig(rebate_rate=0.10, include_principal=True, call_rebate_enabled=True,
                                       call_strike=CALL_STRIKE, call_participation_rate=CALL_PARTICIPATION))
        solver = PhoenixPDESolver()
        solver.price(product, env)                   # the march's state (total tau, delay discount)
        s_vec = np.array([10.0, 100.0, 300.0])
        lower, upper = solver._boundary_rows_v0(0, T, product, env, s_vec)
        df = math.exp(-R * T)                        # no dividend yield: the carry discount is 1
        tenor = product.get_contract_tenor(env) if leg_annualized(product, "rebate") else 1.0
        assert lower.branch == "v0_lower_call_rebate"
        assert upper.branch == "v0_upper_call_rebate"
        assert upper.value == pytest.approx(S0 * df, rel=1e-14)
        assert lower.value == pytest.approx(
            S0 * df + CALL_PARTICIPATION * tenor * (CALL_STRIKE * df - s_vec[0]), rel=1e-14)


class TestHeston2DRows:
    """The 2D Heston/SLV solvers set the same V0 rows through their boundary hook."""

    HP = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)

    @ANNUALIZED
    def test_reverse_rows_mirror_the_standard(self, annualized):
        env = _env()
        product = _certain_v0(0.10, annualized, reverse=True, ko=HESTON_REVERSE_CONTROL_KO)
        solver = HestonSnowballPDESolver(model_params=self.HP, n_x=96, n_v=24, n_t=48)
        solver.price(product, env)
        lower, upper = _t0_rows(solver)
        closed_form = _closed_form(product, env, 0.10, annualized, False)
        assert upper.branch == "v0_upper_principal_rebate"
        assert upper.value == pytest.approx(closed_form, rel=1e-14)
        # the 2D grid reaches down past the knock-out at 2 (S_min 1.97): the lower edge is the knock-out edge
        assert lower.branch == "v0_lower_ko"

    @ANNUALIZED
    def test_call_rebate_lower_row(self, annualized):
        env = _env()
        solver = HestonSnowballPDESolver(model_params=self.HP, n_x=96, n_v=24, n_t=48)
        solver.price(_certain_v0(0.10, annualized, call=True), env)
        lower, _ = _t0_rows(solver)
        assert lower.branch == "v0_lower_call_rebate"
        assert lower.value == pytest.approx(S0 * math.exp(-R * T), rel=1e-14)


class TestHeston2DCertainV0Control:
    """The certain-V0 controls under Heston at (96, 24, 48) price at their closed form: standard KO 500 / discrete KI
    1, reverse KO 2 / discrete KI 10000 (``HESTON_REVERSE_CONTROL_KO``; the reverse KO 20 control is not certain under
    Heston, W7 Q5).

    The ADI core's explicit S-operator ``A1`` must carry the degenerate v = 0 column: the implicit S-solve carries the
    S-drift and the -rU reaction on every variance column, so a predictor that leaves that column at 0 evolves it
    under theta x A1 -- consistent only on a fully implicit step (W7 defect B). Unpatched the gap is +2.48e-3 in
    both orientations.

    Every damping step is switched off (event AND terminal): a theta = 1 step discounts the constant by 1 / (1 + r dt)
    instead of exp(-r dt), a second-order-in-dt term by design (the Rannacher start alone is +4.0e-5 here). With
    both off the residual is the Crank-Nicolson / Craig-Sneyd step's own: -9.3e-7 standard, -7.3e-7 reverse.
    """

    HP = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)

    @ORIENTATION
    def test_control_is_the_closed_form_without_damping(self, reverse):
        from quantark.asset.equity.engine.pde.core.components import CrankNicolson
        from quantark.asset.equity.param.engine_params import PDEParams

        env = _env()
        product = _certain_v0(0.10, False, reverse=reverse, ko=HESTON_REVERSE_CONTROL_KO if reverse else None)
        undamped = PDEParams(stepper=CrankNicolson(event_damping_steps=0, terminal_damping_steps=0))
        solver = HestonSnowballPDESolver(model_params=self.HP, n_x=96, n_v=24, n_t=48, params=undamped)
        price = solver.price(product, env)
        assert price == pytest.approx(_closed_form(product, env, 0.10, False, False), abs=1e-6)


class TestHeston2DContinuousKICopyRow:
    """Under continuous KI the V0 knock-in-side edge copies the knocked-in row (``v0_{edge}_copy_ki``).

    The 1D ``SnowballAdapter`` sets V1's rows of the step first and exposes its grid as ``_grid_v1``. The 2D
    Heston/SLV hooks march V1 to completion before V0 and kept no ``_grid_v1``, so the row read 0.0 (W7 defect A).
    The edge row must be the knocked-in surface's own row at the same time, on every variance column.
    """

    HP = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)

    @staticmethod
    def _snowball(reverse):
        ko, ki = (97.0, 125.0) if reverse else (103.0, 75.0)
        return SnowballOption(
            initial_price=S0, strike=S0, barrier_config=_barrier(ko, ki, continuous=True),
            payoff_config=PayoffConfig(rebate_rate=0.15, include_principal=True), maturity=T, tenor=T,
            is_reverse=reverse)

    @ORIENTATION
    def test_snowball_edge_row_is_the_knocked_in_surface_row(self, reverse):
        env = _env()
        product = self._snowball(reverse)
        solver = HestonSnowballPDESolver(model_params=self.HP, n_x=96, n_v=24, n_t=48)
        solver.price(product, env)
        lower, upper = _t0_rows(solver)
        ki_edge, row = (upper, -1) if reverse else (lower, 0)
        assert ki_edge.branch == ("v0_upper_copy_ki" if reverse else "v0_lower_copy_ki")
        assert ki_edge.copy_from_ki
        core, knocked_in = solver._solve_live_surface(product, env, T, knocked_in=True)
        np.testing.assert_array_equal(knocked_in[row, :], np.full(core.V_grid.size, ki_edge.value))
        # and the hook writes it across the variance axis
        hooked = solver._boundary_hook(core, product, env, knocked_in=False)(
            np.zeros((core.S_grid.size, core.V_grid.size)), T)
        np.testing.assert_array_equal(hooked[row, :], knocked_in[row, :])

    def test_phoenix_edge_row_is_the_knocked_in_row(self):
        from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver

        env = _env()
        product = PhoenixOption(
            initial_price=S0, strike=S0, maturity=T, contract_multiplier=1.0,
            barrier_config=_barrier(103.0, 75.0, continuous=True),
            coupon_config=CouponBarrierConfig(coupon_barrier=75.0, coupon_rate=0.01, memory_coupon=False,
                                              coupon_pay_type=CouponPayType.INSTANT,
                                              day_count_convention=DayCountConvention.ACT_365),
            payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))
        solver = HestonPhoenixPDESolver(model_params=self.HP, n_x=96, n_v=24, n_t=48)
        solver.price(product, env)
        lower, _ = _t0_rows(solver)
        assert lower.branch == "v0_lower_copy_ki"
        core = solver._make_core(product, env, T)
        knocked_in_lower, _ = solver._boundary_rows_v1(0, T, product, env, core.S_grid)
        assert lower.value == knocked_in_lower.value
        assert lower.value != 0.0


class TestPhoenix1DContinuousKICopyRow:
    """The 1D Phoenix (``PhoenixAdapter``) marches each V0 memory state before the knocked-in surfaces and kept no
    ``_grid_v1``: under continuous KI the V0 knock-in-side row read 0.0 (W7 defect A, 1D Phoenix twin). V1's rows of
    the step are set first, as ``SnowballAdapter.boundaries`` does, and the V0 row copies the knocked-in row."""

    @staticmethod
    def _phoenix(reverse, memory, maturity=T, ki=None):
        ko, ki0 = (97.0, 125.0) if reverse else (103.0, 75.0)
        ki = ki0 if ki is None else ki
        n = max(int(round(12 * maturity)), 1)
        obs = [(i + 1) / n * maturity for i in range(n)]
        barrier = BarrierConfig(
            ko_barrier=ko, ko_rate=0.0,
            ko_observation_schedule=ObservationSchedule(
                records=[ObservationRecord(observation_time=t, barrier=ko) for t in obs]),
            ki_barrier=ki, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
        return PhoenixOption(
            initial_price=S0, strike=S0, maturity=maturity, contract_multiplier=1.0, is_reverse=reverse,
            barrier_config=barrier,
            coupon_config=CouponBarrierConfig(coupon_barrier=ki, coupon_rate=0.01, memory_coupon=memory,
                                              coupon_pay_type=CouponPayType.INSTANT,
                                              day_count_convention=DayCountConvention.ACT_365),
            payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))

    @ORIENTATION
    @pytest.mark.parametrize("memory", [False, True], ids=["plain", "memory"])
    def test_edge_row_is_the_knocked_in_row(self, reverse, memory):
        solver = PhoenixPDESolver()
        solver.price(self._phoenix(reverse, memory), _env())
        # the valuation step's rows, set last: V0's (every memory state writes the same) and V1's
        v0_lower, v0_upper = _t0_rows(solver)
        v1_lower, v1_upper = solver._rows_v1_stash[4], solver._rows_v1_stash[5]
        ki_edge, knocked_in = (v0_upper, v1_upper) if reverse else (v0_lower, v1_lower)
        assert ki_edge.branch == ("v0_upper_copy_ki" if reverse else "v0_lower_copy_ki")
        assert ki_edge.copy_from_ki
        assert ki_edge.value == knocked_in.value
        assert ki_edge.value != 0.0

    @pytest.mark.parametrize("reverse,ki", [(False, 60.0), (True, 150.0)], ids=["standard", "reverse"])
    def test_statistics_reconcile_where_the_edge_row_matters(self, reverse, ki):
        """Three months, the knock-in edge nearest the knock-in level: the row moves the price (up to 1.7e-5), and
        the statistics copy the same knocked-in row (``_alive_ki_edge_copies_ki``) or the ledger misses it."""
        stats = PhoenixPDESolver().calculate_event_stats(self._phoenix(reverse, False, 0.25, ki), _env())
        assert abs(stats.reconciliation_error) <= 1e-10, stats.reconciliation_error

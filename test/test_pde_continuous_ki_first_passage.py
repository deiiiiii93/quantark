"""First-passage correction for continuously monitored KI in the PDE solvers.

Per-step application of ``V0 <- V1`` is discrete monitoring at the time-step
width: barrier crossings inside a step are never detected, the engine knocks
in too rarely, and the PV carries a positive O(sqrt(dt)) bias (root-caused
2026-08-18 against the banked RQMC certification references). The fix weights
V0 -> V1 at every interior step by the exact probability that the path touches
the barrier during the step but ends on the live side -- closed form under the
per-step-constant GBM coefficients the operator itself uses, so it removes the
temporal bias without a fitted constant.

Products and environments come from the modelvalidation builders so the cells
here are exactly the certified ones; reference values are the banked Monte
Carlo benchmarks under docs/modelvalidation/certificates/ (paired RQMC,
4 x 65536 paths). They were read from the 2026-08-18 certification and are
unchanged in the re-banked evidence: the fix moved the candidates, not the
benchmark, so the reference identity never moved.
"""

import pytest

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
    KOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_snowball import (
    make_environment,
    make_snowball,
)
from quantark.util.exceptions import ValidationError

ENV = dict(spot=100.0, vol=0.22, rate=0.025, div_yield=0.03)
SNOWBALL = dict(
    initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0,
    ko_rate=0.15, rebate_rate=0.15, months=12, maturity=1.0,
    contract_multiplier=1.0,
)
PHOENIX = dict(
    initial_price=100.0, strike=100.0, ko_barrier=103.0, ko_rate=0.0,
    ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.02, memory_coupon=False,
    num_observations=12, maturity=1.0, contract_multiplier=1.0,
)
KO_RESET = dict(
    initial_price=100.0, strike=100.0, maturity_pre=0.25, maturity_post=0.5,
    pre_ko_barrier=103.0, pre_ko_rate=0.15, post_ko_barrier=95.0,
    post_ko_rate=0.03, ki_barrier=80.0, ki_continuous=True,
    contract_multiplier=1.0,
)


def pde_pv(solver_cls, product, steps_per_day=4.0, **params):
    solver = solver_cls(
        params=PDEParams(
            accuracy="standard",
            grid=GridConfig(points=400, steps_per_day=steps_per_day, max_steps=80000),
            **params,
        )
    )
    return float(
        solver.calculate_greeks(product, make_environment(ENV))["price"]
    )


class TestTimeStepInvariance:
    """With the correction, PV must not depend materially on steps/day.

    The stock per-step scheme moves by O(sigma*sqrt(dt)); at 4 vs 32
    steps/day the certified products drift 0.017-0.032 -- far above the
    thresholds here, which is the point.
    """

    def test_snowball_dt_invariance(self):
        product = make_snowball(SNOWBALL)
        coarse = pde_pv(SnowballPDESolver, product, steps_per_day=4.0)
        fine = pde_pv(SnowballPDESolver, product, steps_per_day=32.0)
        assert abs(coarse - fine) < 0.005

    def test_phoenix_dt_invariance(self):
        product = make_phoenix(PHOENIX)
        coarse = pde_pv(PhoenixPDESolver, product, steps_per_day=4.0)
        fine = pde_pv(PhoenixPDESolver, product, steps_per_day=32.0)
        assert abs(coarse - fine) < 0.006

    def test_ko_reset_dt_invariance(self):
        product = make_ko_reset(KO_RESET)
        coarse = pde_pv(KOResetSnowballPDESolver, product, steps_per_day=4.0)
        fine = pde_pv(KOResetSnowballPDESolver, product, steps_per_day=32.0)
        assert abs(coarse - fine) < 0.008


class TestAgainstBankedBenchmark:
    """Absolute accuracy at the certified base grid vs the banked RQMC PVs."""

    def test_snowball_ordinary(self):
        # Banked reference 96.4619429590 (se 0.0086); stock PDE err +0.0342.
        # Post-fix residual is the ~+0.013 spatial floor, not the temporal bias.
        pv = pde_pv(SnowballPDESolver, make_snowball(SNOWBALL))
        assert pv == pytest.approx(96.4619429590, abs=0.025)

    def test_snowball_near_expiry(self):
        # Banked reference 103.2614422102 (se 0.0078); stock PDE err +0.1518.
        spec = dict(SNOWBALL, months=3, maturity=0.25)
        pv = pde_pv(SnowballPDESolver, make_snowball(spec))
        assert pv == pytest.approx(103.2614422102, abs=0.04)

    def test_phoenix_ordinary(self):
        # Banked reference -3.4261393035 (se 0.0042); stock PDE err +0.0491.
        pv = pde_pv(PhoenixPDESolver, make_phoenix(PHOENIX))
        assert pv == pytest.approx(-3.4261393035, abs=0.03)


class TestSchemeControl:
    def test_legacy_none_scheme_reproduces_step_nodal_pricing(self):
        # The opt-out must preserve the pinned characterization discretization
        # bit-for-bit (value captured on the stock engine, this machine).
        pv = pde_pv(
            SnowballPDESolver,
            make_snowball(SNOWBALL),
            continuous_ki_correction="none",
        )
        assert pv == pytest.approx(96.496112523484, abs=1e-9)

    def test_correction_is_inert_for_discrete_ki(self):
        # A discrete-KI product monitors only at observation dates and has no
        # intra-step crossing to restore: both schemes must agree bitwise.
        # (The ko-reset builder is the one that exposes the monitoring flag;
        # this is the certified discrete_ki construction.)
        product = make_ko_reset(dict(KO_RESET, ki_continuous=False))
        on = pde_pv(KOResetSnowballPDESolver, product)
        off = pde_pv(
            KOResetSnowballPDESolver, product, continuous_ki_correction="none"
        )
        assert on == off

    def test_invalid_scheme_rejected(self):
        with pytest.raises(ValidationError):
            PDEParams(accuracy="standard", continuous_ki_correction="sideways")

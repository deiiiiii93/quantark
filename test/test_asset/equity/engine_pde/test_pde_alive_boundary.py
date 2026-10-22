"""The alive column's lower row under continuous KI reads the knocked-in row it documents (spec 3A §5.2)."""
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_environment, make_snowball

SNOWBALL = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
                rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)
ENV = dict(spot=100.0, vol=0.22, rate=0.025, div_yield=0.03)


def test_alive_lower_row_is_written_from_the_knocked_in_row(monkeypatch):
    seen = []
    orig = SnowballPDESolver._set_boundary_conditions_v0

    def spy(self, grid, x, s, t_idx, tau, product, env):
        orig(self, grid, x, s, t_idx, tau, product, env)
        seen.append((float(grid[0, t_idx]), float(self._grid_v1[0, t_idx])))

    monkeypatch.setattr(SnowballPDESolver, "_set_boundary_conditions_v0", spy)
    SnowballPDESolver(PDEParams(accuracy="fast")).price(make_snowball(SNOWBALL), make_environment(ENV))
    assert seen
    assert all(alive == ki for alive, ki in seen)
    # 0.7.0 read the knocked-in row before it was written: zero on every step
    assert any(ki != 0.0 for _, ki in seen)

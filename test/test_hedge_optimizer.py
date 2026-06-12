"""
Unit tests for the multi-target, multi-instrument hedge optimizer.
"""

import pytest

from quantark.backtest.strategy.hedge_optimizer import HedgeOptimizer, HedgeTarget
from quantark.util.exceptions import NumericalError, ValidationError


class TestHedgeTarget:
    def test_valid_target(self):
        target = HedgeTarget("delta", target=5.0, threshold=10.0, weight=2.0)
        assert target.greek == "delta"
        assert target.target == 5.0

    def test_invalid_parameters(self):
        with pytest.raises(ValidationError):
            HedgeTarget("")
        with pytest.raises(ValidationError):
            HedgeTarget("delta", threshold=-1.0)
        with pytest.raises(ValidationError):
            HedgeTarget("delta", weight=0.0)


class TestSquareSolve:
    def test_delta_gamma_2x2(self):
        # Short option book: delta=-40, gamma=-3.
        # gamma: 0.1 * x_opt = 3        -> x_opt = 30
        # delta: 0.5 * 30 + x_spot = 40 -> x_spot = 25
        optimizer = HedgeOptimizer()
        quantities = optimizer.solve(
            portfolio_greeks={"delta": -40.0, "gamma": -3.0},
            instrument_greeks={
                "gamma_option": {"delta": 0.5, "gamma": 0.1},
                "spot": {"delta": 1.0, "gamma": 0.0},
            },
            targets=[HedgeTarget("delta"), HedgeTarget("gamma")],
        )
        assert quantities["gamma_option"] == pytest.approx(30.0)
        assert quantities["spot"] == pytest.approx(25.0)

    def test_3x3_neutralizes_all_targets(self):
        portfolio = {"delta": -120.0, "gamma": -8.0, "vega": -45.0}
        instruments = {
            "short_opt": {"delta": 0.52, "gamma": 0.09, "vega": 0.19},
            "long_opt": {"delta": 0.55, "gamma": 0.02, "vega": 0.39},
            "spot": {"delta": 1.0, "gamma": 0.0, "vega": 0.0},
        }
        targets = [HedgeTarget("delta"), HedgeTarget("gamma"), HedgeTarget("vega")]
        quantities = HedgeOptimizer().solve(portfolio, instruments, targets)

        # Verify post-hedge Greeks are zero: g + A @ x == 0
        for greek in ("delta", "gamma", "vega"):
            post = portfolio[greek] + sum(
                instruments[name][greek] * quantities[name] for name in instruments
            )
            assert post == pytest.approx(0.0, abs=1e-9)

    def test_nonzero_targets(self):
        quantities = HedgeOptimizer().solve(
            portfolio_greeks={"delta": 100.0},
            instrument_greeks={"spot": {"delta": 1.0}},
            targets=[HedgeTarget("delta", target=20.0)],
        )
        assert quantities["spot"] == pytest.approx(-80.0)

    def test_singular_matrix_raises(self):
        with pytest.raises(NumericalError):
            HedgeOptimizer().solve(
                portfolio_greeks={"delta": 10.0, "gamma": 1.0},
                instrument_greeks={
                    "a": {"delta": 0.5, "gamma": 0.1},
                    "b": {"delta": 1.0, "gamma": 0.2},  # 2x column a
                },
                targets=[HedgeTarget("delta"), HedgeTarget("gamma")],
            )


class TestUnderdetermined:
    def test_minimum_norm_solution(self):
        # 1 target, 2 instruments: A=[1, 0.5], b=10
        # min ||x||^2 s.t. Ax=b -> x = A^T (A A^T)^-1 b = [8, 4]
        quantities = HedgeOptimizer().solve(
            portfolio_greeks={"delta": -10.0},
            instrument_greeks={"spot": {"delta": 1.0}, "opt": {"delta": 0.5}},
            targets=[HedgeTarget("delta")],
        )
        assert quantities["spot"] == pytest.approx(8.0)
        assert quantities["opt"] == pytest.approx(4.0)

    def test_costs_shift_solution_to_cheaper_instrument(self):
        quantities = HedgeOptimizer().solve(
            portfolio_greeks={"delta": -10.0},
            instrument_greeks={"spot": {"delta": 1.0}, "opt": {"delta": 0.5}},
            targets=[HedgeTarget("delta")],
            instrument_costs={"spot": 1.0, "opt": 10.0},
        )
        # Constraint still met exactly
        assert quantities["spot"] + 0.5 * quantities["opt"] == pytest.approx(10.0)
        # And the expensive instrument is barely used
        assert abs(quantities["opt"]) < abs(quantities["spot"]) / 10.0

    def test_invalid_cost_raises(self):
        with pytest.raises(ValidationError):
            HedgeOptimizer().solve(
                portfolio_greeks={"delta": -10.0},
                instrument_greeks={"spot": {"delta": 1.0}, "opt": {"delta": 0.5}},
                targets=[HedgeTarget("delta")],
                instrument_costs={"spot": -1.0},
            )


class TestOverdetermined:
    def test_weighted_least_squares(self):
        # 2 targets, 1 instrument: A=[[1],[1]], b=[10, 0]
        # Equal weights -> x = 5 (split the error)
        quantities = HedgeOptimizer().solve(
            portfolio_greeks={"delta": -10.0, "gamma": 0.0},
            instrument_greeks={"x": {"delta": 1.0, "gamma": 1.0}},
            targets=[HedgeTarget("delta"), HedgeTarget("gamma")],
        )
        assert quantities["x"] == pytest.approx(5.0)

    def test_weight_prioritizes_target(self):
        quantities = HedgeOptimizer().solve(
            portfolio_greeks={"delta": -10.0, "gamma": 0.0},
            instrument_greeks={"x": {"delta": 1.0, "gamma": 1.0}},
            targets=[
                HedgeTarget("delta", weight=100.0),
                HedgeTarget("gamma", weight=1.0),
            ],
        )
        # Heavily weighted delta target pulls x toward 10
        assert quantities["x"] == pytest.approx(10.0, abs=0.01)


class TestEdgeCases:
    def test_already_at_target_returns_zeros(self):
        quantities = HedgeOptimizer().solve(
            portfolio_greeks={"delta": 0.0, "gamma": 0.0},
            instrument_greeks={
                "opt": {"delta": 0.5, "gamma": 0.1},
                "spot": {"delta": 1.0, "gamma": 0.0},
            },
            targets=[HedgeTarget("delta"), HedgeTarget("gamma")],
        )
        assert quantities == {"opt": 0.0, "spot": 0.0}

    def test_missing_greek_in_instrument_raises(self):
        with pytest.raises(ValidationError):
            HedgeOptimizer().solve(
                portfolio_greeks={"delta": 10.0},
                instrument_greeks={"spot": {"gamma": 0.0}},
                targets=[HedgeTarget("delta")],
            )

    def test_duplicate_target_greeks_raise(self):
        with pytest.raises(ValidationError):
            HedgeOptimizer().solve(
                portfolio_greeks={"delta": 10.0},
                instrument_greeks={"spot": {"delta": 1.0}},
                targets=[HedgeTarget("delta"), HedgeTarget("delta")],
            )

    def test_empty_inputs_raise(self):
        with pytest.raises(ValidationError):
            HedgeOptimizer().solve({}, {"spot": {"delta": 1.0}}, [])
        with pytest.raises(ValidationError):
            HedgeOptimizer().solve({}, {}, [HedgeTarget("delta")])

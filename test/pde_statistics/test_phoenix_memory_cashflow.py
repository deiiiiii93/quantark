"""Memory-coupon cash against QUAD V2 (spec 3B §5.3, §7.2)."""
import dataclasses

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2
from quantark.asset.equity.param import PDEParams
from pde_statistics.cases import CASES

PER_DATE, TOTAL = 2e-4, 5e-4


@pytest.mark.slow
def test_study_product_memory_coupon_cash_against_v2(value_mode):
    value_mode("phoenix")
    product, env = CASES["ph_memory"][0](), CASES["ph_memory"][1]()
    pde = PhoenixPDESolver(PDEParams()).calculate_event_stats(product, env)
    v2 = PhoenixQuadEngineV2().calculate_event_stats(product, env)
    assert np.max(np.abs(pde.expected_discounted_coupon_cashflow - v2.expected_discounted_coupon_cashflow)) <= PER_DATE
    assert abs(pde.expected_discounted_coupon_cashflow.sum() - v2.expected_discounted_coupon_cashflow.sum()) <= TOTAL
    assert abs(pde.reconciliation_error) <= 1e-10


def test_trigger_probability_is_memory_free(value_mode):
    value_mode("phoenix")
    mem, env = CASES["ph_memory"][0](), CASES["ph_memory"][1]()
    plain = CASES["ph_cont"][0]()
    a = PhoenixPDESolver(PDEParams()).calculate_event_stats(mem, env).coupon_probability
    b = PhoenixPDESolver(PDEParams()).calculate_event_stats(plain, env).coupon_probability
    assert np.max(np.abs(a - b)) <= 1e-12


def test_carried_arrears_add_less_than_their_discounted_face(value_mode):
    """Review Focus 1: arrears carried in are paid at most once, with the first coupon that triggers."""
    value_mode("phoenix")
    env = CASES["ph_memory"][1]()
    base = CASES["ph_memory"][0]()
    carried = CASES["ph_memory"][0]()
    carried.coupon_config = dataclasses.replace(carried.coupon_config, initial_coupon_arrears=2.0)
    a = PhoenixPDESolver(PDEParams()).calculate_event_stats(base, env)
    b = PhoenixPDESolver(PDEParams()).calculate_event_stats(carried, env)
    extra = float(b.expected_discounted_coupon_cashflow.sum() - a.expected_discounted_coupon_cashflow.sum())
    first_pay = float(env.get_discount_factor(float(a.ko_times[0])))
    assert 0.0 < extra < 2.0 * first_pay
    assert abs(b.reconciliation_error) <= 1e-10

"""The DCN event route and its fold structure (ship spec 2026-10-21 D9, D10, AC2)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.cashleg import EventDistribution, EventType, fold_cash_legs
from quantark.cashleg.folding import _event_structure

from dcn_fixtures import DCN_A, DCN_B, FLAT, flat_env, make_dcn
from test_cashleg.test_folding import _leg_set, _reassemble


def _engine():
    return DCNMCEngine(num_paths=8192, seed=42)


@pytest.mark.parametrize("contract", [DCN_A, DCN_B])
def test_price_with_events_converts_the_engines_own_statistics(contract):
    product, env = make_dcn(contract), flat_env(**FLAT)
    result = _engine().price_with_events(product, env)
    detailed = _engine().price_detailed(product, env)
    stats, dist = detailed.event_stats, result.event_distribution
    assert result.npv == detailed.pv
    ctx = build_dcn_grid_context(product)
    np.testing.assert_array_equal(dist.stream_times[EventType.KO], ctx.times[ctx.obs_cols])
    np.testing.assert_array_equal(dist.probabilities[EventType.KO], np.asarray(stats.ko_timing_distribution))
    assert dist.probabilities[EventType.MATURITY_WITH_KI] == stats.prob_survive_ki
    assert dist.probabilities[EventType.MATURITY_NO_KO] == pytest.approx(stats.prob_survive_no_ki, abs=1e-12)
    np.testing.assert_array_equal(dist.payment_times[EventType.KO], ctx.ko_pay_times)
    assert dist.payment_times_for(EventType.MATURITY_NO_KO) == ctx.loss_pay_time
    assert dist.payment_times_for(EventType.MATURITY_WITH_KI) == ctx.loss_pay_time
    pay = product.payment_dates()
    assert dist.stream_dates[EventType.KO] == tuple(m.observation_date for m in product.schedule.monthly)
    assert dist.payment_dates[EventType.KO] == tuple(pay.ko_payment_dates)
    assert dist.payment_dates[EventType.MATURITY_WITH_KI] == pay.loss_payment_date


def test_the_final_observation_date_converts_its_settled_row():
    base = make_dcn(DCN_A)
    last = base.schedule.monthly[-1].observation_date
    product = make_dcn(DCN_A, valuation_date=last)
    result = _engine().price_with_events(product, flat_env(**FLAT))
    dist = result.event_distribution
    total = sum(float(np.sum(dist.probabilities[e]))
                for e in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI))
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("direction", [DCNDirection.BUYER, DCNDirection.SELLER])
@pytest.mark.parametrize("quantity", [1.0, -2.0])
def test_fold_reproduces_the_dcn_route(direction, quantity):
    product, env = make_dcn(DCN_A, direction=direction), flat_env(**FLAT)
    dist = _engine().price_with_events(product, env).event_distribution
    structure = _event_structure(product, env)
    basis = structure.distribution(np.zeros(len(structure.ko)), 1.0, 0.0)
    np.testing.assert_array_equal(basis.stream_times[EventType.KO], dist.stream_times[EventType.KO])
    assert basis.stream_dates[EventType.KO] == dist.stream_dates[EventType.KO]
    assert basis.payment_dates == {k: v for k, v in dist.payment_dates.items() if k in basis.payment_dates}
    for et in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI):
        np.testing.assert_array_equal(np.asarray(basis.payment_times[et]), np.asarray(dist.payment_times[et]))

    legs = _leg_set(dist.stream_times[EventType.KO])
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    assert overlay.maturity_no_ki.amount == 0.0
    # the engine applies the DCN direction to every cash it adds, so the overlay is buyer-view (D9)
    sign = product.direction_sign
    folded = _reassemble(overlay, dist, env, quantity * sign) - overlay.constant_pv
    expected = sum(leg.value(dist, env, 100.0) for leg in legs) - overlay.constant_pv
    assert folded == pytest.approx(expected, rel=1e-12, abs=1e-6)

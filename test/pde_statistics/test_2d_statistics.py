"""2D Heston/SLV statistics (spec 3B §5.6, decision 8)."""
import pytest

from quantark.asset.equity.engine.event_stats import StatisticsProvenance, StatisticsRefused
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from pde_bitwise.capture_book import env_flat, phoenix_plain, snowball

HP = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
SMALL = dict(n_x=48, n_v=18, n_t=16)


def test_surrogate_carries_provenance_and_the_route_check():
    s = HestonSnowballPDESolver(HP, statistics="flat_vol_surrogate", **SMALL)
    stats = s.calculate_event_stats(snowball(), env_flat())
    assert stats.provenance == StatisticsProvenance("flat_vol_1d", "heston_adi_2d") and stats.provenance.surrogate
    half = HestonSnowballPDESolver(HP, params=PDEParams(stepper=CrankNicolson(damping="half_step")),
                                   statistics="flat_vol_surrogate", **SMALL)
    with pytest.raises(ValidationError, match="2D"):
        half.calculate_event_stats(snowball(), env_flat(), npv=1.0)


def test_unknown_option_value_is_refused():
    with pytest.raises(ValidationError, match="statistics"):
        HestonPhoenixPDESolver(HP, statistics="native", **SMALL)


def test_native_1d_provenance_is_not_a_surrogate():
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(snowball(), env_flat())
    assert stats.provenance == StatisticsProvenance("flat_vol_1d", "flat_vol_1d")


def test_refuse_mode_refuses_from_both_entry_points():
    s = HestonSnowballPDESolver(HP, statistics="refuse", **SMALL)
    with pytest.raises(StatisticsRefused, match="flat_vol_surrogate"):
        s.calculate_event_stats(snowball(), env_flat())
    with pytest.raises(StatisticsRefused):
        s.price_with_events(snowball(), env_flat())


def test_default_refuses_but_a_plain_price_with_events_does_not():
    from quantark.cashleg.event_distribution import EventType
    s = HestonSnowballPDESolver(HP, **SMALL)
    with pytest.raises(StatisticsRefused, match="flat_vol_surrogate"):
        s.calculate_event_stats(snowball(), env_flat())
    res = s.price_with_events(snowball(), env_flat(), emit_distribution=False)     # Review Focus 4
    assert res.npv == s.price(snowball(), env_flat())
    assert res.event_distribution.probabilities.keys() == {EventType.MATURITY_NO_KO}

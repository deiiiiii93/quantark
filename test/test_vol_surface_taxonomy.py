from quantark.param.vol import BlackImpliedVolSurface, FlatVolSurface
from quantark.param import BlackImpliedVolSurface as BISFromParam


def test_black_implied_vol_surface_is_exported():
    assert BlackImpliedVolSurface is BISFromParam


def test_flat_surface_is_not_smile():
    assert FlatVolSurface(volatility=0.2).is_smile is False


def test_legacy_alias_still_works_and_is_same_object():
    from quantark.param.vol import VolatilitySurface
    assert VolatilitySurface is BlackImpliedVolSurface


def test_flat_surface_subclasses_renamed_base():
    assert issubclass(FlatVolSurface, BlackImpliedVolSurface)


def test_sabr_surface_is_smile():
    from quantark.param.vol import SABRVolSurface
    s = SABRVolSurface.from_params(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4, maturity=1.0)
    assert s.is_smile is True


def test_grid_surface_is_smile():
    import numpy as np
    from quantark.param.vol import GridVolSurface
    g = GridVolSurface(strikes=[90.0, 100.0, 110.0], maturities=[0.5, 1.0],
                       iv_grid=np.array([[0.2, 0.19, 0.21], [0.22, 0.20, 0.23]]))
    assert g.is_smile is True


def test_vannavolga_surface_is_smile():
    from quantark.param.vol import VannaVolgaVolSurface
    from quantark.param.vol.vannavolga import FXEnv, SmileQuotes
    s = VannaVolgaVolSurface(
        env=FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=1.0),
        quotes=SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.003),
    )
    assert s.is_smile is True

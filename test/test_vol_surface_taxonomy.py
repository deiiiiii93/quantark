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

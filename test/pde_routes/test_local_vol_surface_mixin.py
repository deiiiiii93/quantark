"""The local-vol solvers share one surface mixin (spec 3C §5.1)."""
from quantark.asset.equity.engine.pde.local_vol_surface import LocalVolSurfaceMixin
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver


def test_the_local_vol_autocallables_use_the_mixin():
    for cls in (LocalVolSnowballPDESolver, LocalVolPhoenixPDESolver):
        assert issubclass(cls, LocalVolSurfaceMixin)
        for name in ("_build_surface", "_surface_scope", "_with_surface", "_build_step_coefficients",
                     "_flat_exact_step_coefficients", "_constructor_kwargs"):
            assert name not in cls.__dict__, (cls.__name__, name)

"""Half-step damping reaches the full-step limit from closer on the discretely monitored routes that
no certificate covers under daily knock-in (spec 3A decision 16, §4.2.3)."""
import pytest

from quantark.asset.equity.engine import PDEEngine
from quantark.asset.equity.engine.pde import LocalVolPhoenixPDESolver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams

from pde_3a.products import PHOENIX_VARIANTS, env, ko_reset_discrete, localvol_phoenix_case, phoenix

FULL = CrankNicolson(damping="full_step", terminal_damping_steps=1, event_damping_steps=2)
#: The default's pv time error at standard against the limit, measured in e18 (spec §7.2), per cell
E18 = {"flat_phoenix": 7.1e-4, "flat_ko_reset": 3.7e-4, "crash_phoenix": 9.4e-4, "calm_phoenix": 5.1e-4}


def _case(name):
    if name == "flat_phoenix":
        return PDEEngine, phoenix(100.0, *PHOENIX_VARIANTS["daily_ki"]), env(), None
    if name == "flat_ko_reset":
        return PDEEngine, ko_reset_discrete(), env(), None
    product, e, lv = localvol_phoenix_case(name.split("_")[0])
    return LocalVolPhoenixPDESolver, product, e, lv


def _pv(name, stepper, steps_per_day):
    cls, product, e, lv = _case(name)
    params = PDEParams(stepper=stepper, grid=GridConfig(steps_per_day=float(steps_per_day), max_steps=400000))
    engine = cls(params=params) if lv is None else cls(params=params, local_vol_surface=lv)
    unit = 100.0 / (product.initial_price * product.contract_multiplier)
    return float(engine.price(product, e)) * unit


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(E18))
def test_the_default_is_closer_to_the_time_limit(name):
    default = CrankNicolson()
    lim_d, lim_f = _pv(name, default, 32), _pv(name, FULL, 32)
    assert abs(lim_d - lim_f) < 2e-4                             # measured 1.1e-4 at most
    limit = 0.5 * (lim_d + lim_f)
    err_default = _pv(name, default, 4) - limit
    err_full = _pv(name, FULL, 4) - limit
    assert abs(err_default) < 2.0 * E18[name]
    assert abs(err_full) >= 3.0 * abs(err_default)              # measured ratios 4.6 to 5.3

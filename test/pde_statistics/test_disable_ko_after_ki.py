"""disable_ko_after_ki Snowball statistics against QUAD V2 (fixed in spec 3B stage 2 (a)).

0.8.0's statistics ignored the setting at the final knock-out date: knocked-in paths still knocked out
there, so the last knock-out probability read 4.06e-2 instead of 2.26e-3 (continuous knock-in) and 2.42e-2
instead of 4.69e-3 (discrete), with ki_survive short by the same mass; the price was right. Measured on
this branch at standard (2026-10-04): max |ko - V2| 2.1e-5 / 2.6e-6, |ki_survive - V2| 1.9e-4 / 2.0e-5.
Bounds: twice the measurement.
"""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.v2.engine import SnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from pde_3a.products import SNOWBALL, env

KO_BOUND, KI_SURVIVE_BOUND = 5e-5, 4e-4


@pytest.mark.parametrize("extra", [{}, {"ki_monitoring": "discrete"}], ids=["continuous_ki", "discrete_ki"])
def test_knocked_in_paths_do_not_knock_out_at_maturity(extra):
    product = make_snowball(dict(SNOWBALL, disable_ko_after_ki=True, **extra))
    pde = SnowballPDESolver(PDEParams()).calculate_event_stats(product, env())
    v2 = SnowballQuadEngineV2().calculate_event_stats(product, env())
    assert np.max(np.abs(pde.ko_probability - v2.ko_probability)) <= KO_BOUND
    assert abs(pde.ki_survive_knocked_in_probability - v2.ki_survive_knocked_in_probability) <= KI_SURVIVE_BOUND
    assert abs(pde.reconciliation_error) <= 1e-10

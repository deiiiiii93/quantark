"""Open question Q1 (QUAD V2 at the maturity instant): what V2 prices if its product check accepted a numeric
maturity of exactly 0.0, and what the dated twin priced before this branch refused it.

Runtime patch, not a code change: ``validate()`` of the snowball and Phoenix products is wrapped so a numeric
maturity of exactly 0.0 is checked as if it were 1.0 (the construction rule refuses 0.0 twice, in
``_validate_maturity_dates`` and ``_validate_maturity_parameters``; these contracts' cash is flat, so the terminal
period ``validate()`` rebuilds does not reach the price), and the branch's dated refusal in ``compile_contract`` is
bypassed (``adapters.is_zero`` patched to False).

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/proposal_quad_v2.py
"""
from __future__ import annotations

import math
import warnings

from test_dated_expiry_settles import PHOENIX, PHOENIX_PLAIN, SNOWBALL, env

from quantark.asset.equity.engine.quad import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.engine.quad.v2 import adapters
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption

warnings.simplefilter("ignore")


def accept_zero(cls):
    original = cls.validate

    def patched(self):
        if self.maturity == 0.0:
            self.maturity = 1.0
            try:
                original(self)
            finally:
                self.maturity = 0.0
        else:
            original(self)
    cls.validate = patched


accept_zero(SnowballOption)
accept_zero(PhoenixOption)
adapters.is_zero = lambda value: False          # the dated twin as it priced before this branch's refusal

print("family | spot | numeric V2 (patched) | dated V2 (refusal bypassed) | settled | both exact")
for family, engine in ((SNOWBALL, SnowballQuadEngineV2), (PHOENIX, PhoenixQuadEngineV2),
                       (PHOENIX_PLAIN, PhoenixQuadEngineV2)):
    for spot in family.spots:
        numeric = float(engine().price(family.build("numeric"), env(spot)))
        dated = float(engine().price(family.build("dated"), env(spot)))
        settled = family.settled(spot)
        exact = all(math.isclose(v, settled, rel_tol=1e-12, abs_tol=1e-12) for v in (numeric, dated))
        print(f"{family.name} | {spot:g} | {numeric:.12g} | {dated:.12g} | {settled:.12g} | {'yes' if exact else 'NO'}")

"""Regenerate ``v2_reference/<key>.json`` (see ``v2_frozen.py``). Run by hand: a provenance act.

    PYTHONPATH=$PWD:$PWD/test python test/pde_statistics/build_v2_reference.py [key ...]

Each key is an independent 90-800 s V2 solve; start several processes to parallelise. With no key, all run.
Refresh only when V2's numerics changed on purpose and were re-certified.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pde_statistics.v2_frozen import freeze  # noqa: E402


def _ph_memory():
    from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2
    from pde_statistics.cases import CASES
    return PhoenixQuadEngineV2().calculate_event_stats(CASES["ph_memory"][0](), CASES["ph_memory"][1]())


def _expiry(memory):
    def build():
        from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2
        from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
        from pde_3a.products import PHOENIX, env
        product = make_phoenix(dict(PHOENIX, memory_coupon=memory, coupon_pay_type="expiry"))
        return PhoenixQuadEngineV2().calculate_event_stats(product, env())
    return build


def _release_rule():
    from datetime import datetime
    from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2
    from quantark.util.calendar import CalendarType, create_calendar
    from intraday.conftest import dated_phoenix, flat_env
    t0 = datetime(2026, 3, 16)
    product = dated_phoenix(create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028)), t0, coupon_barrier=105.0)
    return PhoenixQuadEngineV2().calculate_event_stats(product, flat_env(t0))


def _ko_reset(spot):
    def build():
        from quantark.asset.equity.engine.quad.v2.engine import KOResetSnowballQuadEngineV2
        from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
        from pde_3a.products import KO_RESET, env
        return KOResetSnowballQuadEngineV2().calculate_event_stats(make_ko_reset(KO_RESET), env(spot=spot))
    return build


# Keys match the tests' lookups; ``ko_reset_kr_cont`` also serves test_cashleg/test_ko_reset_distribution.py.
BUILDERS = {
    "phoenix_ph_memory": _ph_memory,
    "phoenix_expiry_memory_false": _expiry(False),
    "phoenix_expiry_memory_true": _expiry(True),
    "phoenix_release_rule": _release_rule,
    "ko_reset_kr_cont": _ko_reset(100.0),
    "ko_reset_kr_cont_spot90": _ko_reset(90.0),
    "ko_reset_kr_cont_spot101": _ko_reset(101.0),
}


def main(keys):
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          cwd=HERE).stdout.strip()
    for key in keys or BUILDERS:
        start = time.time()
        stats = BUILDERS[key]()
        path = freeze(key, stats, {"git_head": head, "machine": platform.machine(),
                                   "seconds": round(time.time() - start, 1)})
        print(f"{key}: {time.time() - start:.0f}s -> {path.name}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

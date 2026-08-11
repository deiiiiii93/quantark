"""Profile one trade-book row to see where autocallable MC time actually goes."""

import cProfile
import io
import pstats
import sys
import dataclasses
from pathlib import Path

ADAPTER = Path("/Users/fuxinyao/otc-price-adapter")
sys.path.insert(0, str(ADAPTER))
sys.path.insert(0, str(ADAPTER / "tests"))

TRADE_ID = sys.argv[1] if len(sys.argv) > 1 else "GJZQ-DCRX7-20260601-OPTION-01"

from freeze_v023_baseline import build_frame_and_settings
import otc_quantark_pricer_v023 as pricer_v023

frame, settings = build_frame_and_settings()
settings = dataclasses.replace(settings, autocallable_model="mc", workers=1)
row = frame.loc[TRADE_ID]
print(f"profiling {TRADE_ID}: {row.get('结构类型', '')}")

prof = cProfile.Profile()
prof.enable()
pricer_v023.price_row(row, settings)
prof.disable()

s = io.StringIO()
pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(18)
text = s.getvalue()
lines = text.splitlines()
start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("ncalls"))
print("\n".join(lines[start : start + 20]))

"""W10 evidence: a coupon decided before the valuation date but paid after it (pending cash) is not in the DCN schedule.

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-10-dcn-final-observation-evidence/measure_pending.py
"""
from __future__ import annotations

import warnings
from datetime import datetime

from dcn_fixtures import DCN_A, make_dcn

warnings.simplefilter("ignore")

# 25 SSE trading days of coupon lag: the month-23 coupon (observed 2024-12-03) pays after 2025-01-03.
contract = dict(DCN_A, coupon_settlement_offset=25, ko_settlement_offset=25, settlement_date=datetime(2025, 2, 28))
for valuation in (datetime(2024, 12, 3), datetime(2024, 12, 4), datetime(2025, 1, 3)):
    p = make_dcn(contract, valuation_date=valuation)
    rows = [(m.month_indices, f"{m.observation_date:%Y-%m-%d}", f"{m.coupon_payment_date:%Y-%m-%d}")
            for m in p.schedule.monthly]
    print(f"valuation {valuation:%Y-%m-%d}: schedule rows (months, observed, coupon paid) = {rows}")

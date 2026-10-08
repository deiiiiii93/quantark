"""Runtime patches of the tracker clock (findings 2026-10-09) -- a pytest plugin, NOT library code.

Load with ``-p tracker_clock_options`` (this directory on PYTHONPATH) and pick the arm with
``TRACKER_CLOCK_OPTION``:

* ``NONE`` -- no behaviour change; only logs every tracker call that meets the condition.
* ``A``    -- numeric record times run from ``initial_date``: every clock the tracker reads from ``start_date``
              (pricing-copy ageing, the valuation point, the terminal's due day, a REBASED schedule 2's base and
              knock-in time) runs from ``initial_date`` instead, for the duration of each public call.
* ``B``    -- numeric record times run from ``start_date``: a numeric contract's records, settlement dates and
              terminal payment date are dated from ``start_date`` (``_record_dicts``, ``_terminal_payment_date``,
              ``_schedule_resolution_env`` -- the three places that read ``initial_date or start_date``).
* ``C``    -- refuse: ``ValidationError`` on any public tracker call for such a contract.

The condition (the only case any arm touches): the contract has numeric timing (``not _uses_date_timing``), both
``initial_date`` and ``start_date`` are set and they are different days. Every call outside the condition runs the
original code, so a test that never meets it cannot change under any arm.

Hits are appended as JSON lines to ``$TRACKER_CLOCK_LOG/<pid>.jsonl``.
"""
from __future__ import annotations

import functools
import json
import os
from copy import deepcopy

import pandas as pd

from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker as T
from quantark.asset.equity.lifecycle.ko_reset import KOResetLifecycleTracker as K
from quantark.util.exceptions import ValidationError

MODE = os.environ.get("TRACKER_CLOCK_OPTION", "NONE").upper()
LOG_DIR = os.environ.get("TRACKER_CLOCK_LOG")
_seen: set = set()


def gap_days(tracker) -> int:
    """``(start_date - initial_date).days`` when the condition holds, else 0."""
    product = tracker.product
    initial = getattr(product, "initial_date", None)
    start = tracker.start_date
    if initial is None or start is None or T._uses_date_timing(product):
        return 0
    return int((pd.Timestamp(start).normalize() - pd.Timestamp(initial).normalize()).days)


def _log(tracker, method, gap):
    if not LOG_DIR:
        return
    test = os.environ.get("PYTEST_CURRENT_TEST", "<no test>").rsplit(" (", 1)[0]
    key = (test, type(tracker).__name__, type(tracker.product).__name__, gap, method)
    if key in _seen:
        return
    _seen.add(key)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, f"{os.getpid()}.jsonl"), "a") as fh:
        fh.write(json.dumps({"test": test, "tracker": key[1], "product": key[2], "gap_days": gap,
                             "method": method, "mode": MODE}) + "\n")


def _wrap_public(cls, name):
    original = cls.__dict__[name]

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        gap = gap_days(self)
        if gap == 0:
            return original(self, *args, **kwargs)
        _log(self, name, gap)
        if MODE == "C":
            raise ValidationError(
                f"tracker clock option C: a numeric-schedule contract whose initial_date "
                f"({self.product.initial_date}) is not the run start ({self.start_date})")
        if MODE == "A":
            saved = self.start_date
            self.start_date = pd.Timestamp(self.product.initial_date).normalize()
            try:
                return original(self, *args, **kwargs)
            finally:
                self.start_date = saved
        return original(self, *args, **kwargs)

    setattr(cls, name, wrapper)


PUBLIC = ("observe", "settle_maturity_if_due", "product_for_pricing", "resolve_calendar_schedule",
          "knock_in_anchor_on", "pre_contract_tenor", "final_contract_tenor", "knock_in_time")
for _cls in (T, K):
    for _name in PUBLIC:
        if _name in _cls.__dict__:
            _wrap_public(_cls, _name)


if MODE == "B":
    _record_dicts = T._record_dicts
    _terminal_payment_date = T._terminal_payment_date
    _schedule_resolution_env = T._schedule_resolution_env

    def record_dicts(self, product, resolved, payoffs=None, base=None):
        if base is None and gap_days(self):
            base = pd.Timestamp(self.start_date).normalize()
        return _record_dicts(self, product, resolved, payoffs, base)

    def terminal_payment_date(self, product, timing, base=None):
        if base is None and gap_days(self):
            base = pd.Timestamp(self.start_date).normalize()
        return _terminal_payment_date(self, product, timing, base)

    def schedule_resolution_env(self, product, env):
        if not gap_days(self):
            return _schedule_resolution_env(self, product, env)
        schedule_env = deepcopy(env)
        schedule_env.valuation_date = pd.Timestamp(self.start_date).to_pydatetime()
        return schedule_env

    T._record_dicts = record_dicts
    T._terminal_payment_date = terminal_payment_date
    T._schedule_resolution_env = schedule_resolution_env


def pytest_report_header(config):
    return f"tracker_clock_options: MODE={MODE} LOG={LOG_DIR}"

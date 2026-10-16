"""Frozen QUAD V2 event statistics for the cross-engine tests.

Four tests compare a PDE event-statistics solve against ``*QuadEngineV2.calculate_event_stats`` on a dense
reference grid. That V2 call costs 90-800 s a case and is deterministic, so the tests read it from
``v2_reference/<key>.json`` instead of re-marching it on every run. The comparison tolerances are unchanged;
only where V2's side comes from moved.

V2 itself is not left unwatched: its numbers are pinned per release by the banked quad certificates
(``docs/modelvalidation/certificates``, checked by ``test_banked_certificates.py``). If V2's numerics change
on purpose, refresh with ``build_v2_reference.py``. ``QUANTARK_LIVE_V2=1`` makes these helpers compute live,
which is also how to confirm a frozen file still matches the engine.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

DIRECTORY = Path(__file__).with_name("v2_reference")


def _encode(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=value["dtype"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def freeze(key: str, stats, provenance: dict) -> Path:
    """Write ``stats`` (a dataclass) as ``v2_reference/<key>.json``."""
    cls = type(stats)
    payload = {
        "class": f"{cls.__module__}:{cls.__qualname__}",
        "provenance": provenance,
        "fields": {f.name: _encode(getattr(stats, f.name)) for f in dataclasses.fields(stats)},
    }
    DIRECTORY.mkdir(exist_ok=True)
    path = DIRECTORY / f"{key}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def v2_event_stats(key: str, compute: Callable[[], object]):
    """V2's event statistics for ``key``: the frozen file, or ``compute()`` under ``QUANTARK_LIVE_V2``."""
    if os.environ.get("QUANTARK_LIVE_V2"):
        return compute()
    payload = json.loads((DIRECTORY / f"{key}.json").read_text())
    module, name = payload["class"].split(":")
    cls = getattr(importlib.import_module(module), name)
    return cls(**{k: _decode(v) for k, v in payload["fields"].items()})

"""Bitwise goldens for the legacy autocallable Dask batch path (Phase 6).

Frozen BEFORE the shared-reducer extraction (spec §17.3): the consolidation in
``autocallable_dask_batch`` must reproduce every recorded field exactly.
Same-machine references (version-stamped), not cross-platform bit claims.
"""
import json
import pathlib

import pytest

pytest.importorskip("dask")

GOLDEN_PATH = (
    pathlib.Path(__file__).parent / "goldens" / "legacy_dask_phase6_goldens.json"
)


def _golden_cases() -> dict:
    return json.loads(GOLDEN_PATH.read_text())["cases"]


def test_phoenix_dask_available_with_dask_installed():
    """Regression: ``from dask.compute import compute`` broke on modern dask,
    silently disabling the Phoenix parallel path while dask was installed."""
    from quantark.asset.equity.engine.mc import phoenix_mc_engine

    assert phoenix_mc_engine.DASK_AVAILABLE is True


@pytest.mark.parametrize("name", sorted(_golden_cases()))
def test_legacy_dask_path_matches_frozen_golden(name):
    from execution.freeze_goldens import (
        _phase6_result_payload,
        build_phase6_dask_cases,
    )

    engine, product, env = build_phase6_dask_cases()[name]
    observed = _phase6_result_payload(engine, product, env)
    golden = _golden_cases()[name]

    assert observed["num_paths"] == golden["num_paths"]
    assert observed["batches_used"] == golden["batches_used"]
    for field in (
        "price",
        "std_error",
        "ko_probability",
        "v0_probability",
        "v1_probability",
    ):
        assert float(observed[field]) == float(golden[field]), field
    if golden["avg_ko_time"] is None:
        assert observed["avg_ko_time"] is None
    else:
        assert float(observed["avg_ko_time"]) == float(golden["avg_ko_time"])

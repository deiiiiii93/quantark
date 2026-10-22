"""The q-study's own audit-coverage helper (split from test_futures_carry_recorder)."""
import sys

import pandas as pd


def test_an_inconclusive_day_keeps_the_run_out_of_a_validity_pass():
    import importlib.util
    from pathlib import Path as _Path

    study = _Path(__file__).resolve().parents[4] / "example" / "case_studies" / "snowball_q_term_structure"
    if str(study) not in sys.path:
        sys.path.insert(0, str(study))
    if "_common" in sys.modules:
        common = sys.modules["_common"]
    else:
        spec = importlib.util.spec_from_file_location("_common", study / "_common.py")
        common = importlib.util.module_from_spec(spec)
        # Register BEFORE exec: dataclasses resolves annotations through
        # sys.modules[cls.__module__], which is None for an unregistered name.
        sys.modules["_common"] = common
        spec.loader.exec_module(common)

    attribution = pd.DataFrame(
        {"audit_status": ["pass", "inconclusive", "pass"]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    coverage = common.audit_coverage(None, attribution)
    assert coverage["by_status"]["inconclusive"] == 1
    assert coverage["all_measured_passed"] is False

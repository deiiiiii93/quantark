"""Golden pin: the +tail carry stress reproduces the design document's -49.8752 bp.

Fails when quantark's futures carry-stress arithmetic moves. The stress is produced by the
study's offline validation stage (synthetic run, no vendor history), loaded by path on purpose.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
STUDY = REPO / "example" / "case_studies" / "snowball_q_term_structure"


def _load(name: str, filename: str):
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V = _load("validation_stage_golden", "05_bucket_hedge_validation.py")


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("validation_golden")
    code = V.main(["--synthetic", "--out-dir", str(out)])
    return code, out


def test_the_tail_stress_reproduces_the_designs_49_8752_bp(synthetic_run):
    _, out = synthetic_run
    stresses = pd.read_csv(out / "stress_results.csv")
    tail_up = stresses[stresses["scenario_id"].str.startswith("tail+")].iloc[0]
    assert tail_up["product_pnl_bp_of_base"] == pytest.approx(-49.8752, abs=1e-3)
    # No futures hedge can respond to it, however the book is positioned.
    assert (stresses["hedge_pnl"] == 0.0).all()
    assert (stresses["book_pnl"] == stresses["product_pnl"]).all()

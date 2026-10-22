"""Tests for quantark.util.io atomic file helpers."""
from __future__ import annotations

import json

import pytest

from quantark.util.io import atomic_write_json


def test_writes_exact_json(tmp_path):
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"a": 1, "b": [1.5, "x"]})
    assert json.loads(path.read_text()) == {"a": 1, "b": [1.5, "x"]}


def test_no_tmp_residue(tmp_path):
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"a": 1})
    residue = [p for p in tmp_path.iterdir() if p.name != "payload.json"]
    assert residue == []


def test_overwrite_last_wins(tmp_path):
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"v": 1})
    atomic_write_json(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 2}


def test_nan_rejected_and_target_untouched(tmp_path):
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"v": 1})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"v": float("nan")})
    assert json.loads(path.read_text()) == {"v": 1}
    residue = [p for p in tmp_path.iterdir() if p.name != "payload.json"]
    assert residue == []

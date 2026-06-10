"""Tests for the legacy flat-import compatibility shim (quantark._compat).

Most scenarios need a pristine interpreter (import order, warn-once,
finder precedence), so they shell out to fresh subprocesses. Inside the
subprocess, ``import quantark._compat`` registers the finder explicitly —
except in the .pth test, which verifies registration happens at interpreter
startup without any explicit import (requires quantark to be pip-installed,
editable or not, so the .pth is present in site-packages).
"""

import importlib.metadata
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_py(code: str, *python_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *python_args, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def quantark_is_installed() -> bool:
    try:
        importlib.metadata.distribution("quantark")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def test_deep_legacy_submodule_import():
    result = run_py(
        """
        import quantark._compat
        import asset.equity.engine.mc
        from util.enum import OptionType
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_module_identity_legacy_first():
    result = run_py(
        """
        import quantark._compat
        import util.enum as old
        import quantark.util.enum as new
        assert old is new, "module identity split"
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_module_identity_canonical_first():
    result = run_py(
        """
        import quantark._compat
        import quantark.util.enum as new
        import util.enum as old
        assert old is new, "module identity split"
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_enum_identity_across_spellings():
    result = run_py(
        """
        import quantark._compat
        from util.enum import OptionType as Old
        from quantark.util.enum import OptionType as New
        assert Old is New
        assert Old.CALL is New.CALL
        assert isinstance(Old.CALL, New)
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_deprecation_warning_once_per_root():
    result = run_py(
        """
        import warnings
        import quantark._compat
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import util
            import util.enum
            import util.calendar
        messages = [str(w.message) for w in caught
                    if issubclass(w.category, DeprecationWarning)
                    and "quantark.util" in str(w.message)]
        assert len(messages) == 1, messages
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_canonical_spec_survives_legacy_import():
    # The alias must not clobber the canonical module's __spec__/__loader__,
    # or importlib.resources lookups break.
    result = run_py(
        """
        import quantark._compat
        import util.calendar
        import importlib.resources
        ref = (importlib.resources.files("quantark.util")
               / "calendar" / "holidayfile" / "china_sse.csv")
        assert ref.is_file(), "resource lookup broken after legacy import"
        import quantark.util
        assert quantark.util.__spec__.name == "quantark.util"
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_installed_distribution_wins_over_alias(tmp_path):
    # A real package named like a legacy root must shadow the shim alias:
    # the finder is appended to sys.meta_path, so PathFinder runs first.
    fake = tmp_path / "param"
    fake.mkdir()
    (fake / "__init__.py").write_text("SENTINEL = 'real-third-party'\n")
    result = run_py(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import quantark._compat
        import param
        assert getattr(param, "SENTINEL", None) == "real-third-party", (
            "shim alias shadowed a genuinely installed package"
        )
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_unknown_submodule_still_fails():
    result = run_py(
        """
        import quantark._compat
        try:
            import util.does_not_exist
        except ModuleNotFoundError:
            print("ok")
        else:
            raise SystemExit("import of missing submodule should fail")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_pth_registers_shim_at_startup():
    if not quantark_is_installed():
        pytest.skip(
            "quantark not pip-installed; quantark_compat.pth is only "
            "deployed by an install"
        )
    # No explicit quantark import: the .pth must have registered the finder.
    result = run_py(
        """
        import asset
        print(asset.__name__)
        """
    )
    assert result.returncode == 0, result.stderr
    assert "quantark.asset" in result.stdout

"""Legacy flat-import compatibility for the pre-namespace package layout.

quantark historically installed twelve top-level packages (``asset``,
``util``, ``param``, ...). They now live under the single ``quantark``
package. This module keeps the old spellings importable by aliasing them to
the canonical modules, so ``import util.enum`` and
``import quantark.util.enum`` yield the *same* module object — classes,
enums, and module state keep a single identity regardless of spelling.

Registration happens at interpreter startup through ``quantark_compat.pth``
(installed alongside the package), which executes ``import quantark._compat``
via the ``site`` machinery. The finder is appended to ``sys.meta_path``, so
Python's standard finders run first: a genuinely installed distribution named
like a legacy root (e.g. HoloViz ``param``) wins over the alias, and the shim
only resolves names that would otherwise fail.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

LEGACY_ROOTS = frozenset({
    "asset",
    "backtest",
    "cashleg",
    "dynamicscenario",
    "param",
    "portfolio",
    "priceenv",
    "rfq",
    "simm",
    "stresstest",
    "util",
    "var",
})

_warned_roots: set[str] = set()


class _AliasLoader(importlib.abc.Loader):
    """Hands the canonical module object to the import machinery.

    ``create_module`` returns the already-imported ``quantark.*`` module, so
    the legacy name binds to the identical object. The import machinery then
    stamps the alias spec onto that module (``__spec__``/``__loader__``),
    which would corrupt the canonical module's metadata — and with it
    ``importlib.resources`` lookups — so the originals are snapshotted here
    and restored in ``exec_module``.
    """

    def __init__(self, canonical_name: str) -> None:
        self._canonical_name = canonical_name
        self._saved_attrs: dict[str, object] = {}

    def create_module(self, spec):
        module = importlib.import_module(self._canonical_name)
        for attr in ("__spec__", "__loader__"):
            if hasattr(module, attr):
                self._saved_attrs[attr] = getattr(module, attr)
        return module

    def exec_module(self, module) -> None:
        for attr, value in self._saved_attrs.items():
            setattr(module, attr, value)


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root not in LEGACY_ROOTS:
            return None
        canonical_name = f"quantark.{fullname}"
        try:
            importlib.import_module(canonical_name)
        except ImportError:
            return None
        if root not in _warned_roots:
            _warned_roots.add(root)
            warnings.warn(
                f"Top-level import '{root}' is deprecated; "
                f"import 'quantark.{root}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(canonical_name)
        )


def install() -> None:
    """Append the legacy-alias finder to ``sys.meta_path`` (idempotent)."""
    if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
        sys.meta_path.append(_LegacyAliasFinder())


install()

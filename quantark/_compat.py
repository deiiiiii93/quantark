"""Legacy flat-import compatibility for the pre-namespace package layout.

quantark historically installed twelve top-level packages (``asset``,
``util``, ``param``, ...). They now live under the single ``quantark``
package. This module keeps the old spellings importable by aliasing them to
the canonical modules, so ``import util.enum`` and
``import quantark.util.enum`` yield the *same* module object — classes,
enums, and module state keep a single identity regardless of spelling.

Registration happens at interpreter startup through ``quantark_compat.pth``
(installed alongside the package), which executes ``import quantark._compat``
via the ``site`` machinery.

The finder must be *prepended* to ``sys.meta_path``: once a legacy root is
aliased, its ``__path__`` points into ``quantark/``, and ``PathFinder`` would
otherwise resolve submodule imports (``import util.enum``) through that path
into fresh, duplicate module objects — the exact identity split this shim
exists to prevent. Precedence for real packages is enforced explicitly
instead: root imports defer to any genuinely installed distribution (e.g.
HoloViz ``param``), and submodules are only aliased when the root itself is
our alias.
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


def _is_own_package_leak(spec) -> bool:
    """True if *spec* points inside the installed quantark package itself."""
    import os

    import quantark

    package_file = getattr(quantark, "__file__", None)
    if not package_file:
        return False
    package_dir = os.path.realpath(os.path.dirname(package_file))
    candidates = []
    if spec.origin:
        candidates.append(spec.origin)
    candidates.extend(spec.submodule_search_locations or [])
    return any(
        os.path.realpath(c).startswith(package_dir + os.sep)
        for c in candidates
        if c
    )


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
        root, dot, _ = fullname.partition(".")
        if root not in LEGACY_ROOTS:
            return None
        if dot:
            # Alias submodules only beneath a root we aliased ourselves —
            # never reach inside a genuinely installed package that happens
            # to share a legacy name.
            root_module = sys.modules.get(root)
            if root_module is None or root_module is not sys.modules.get(
                f"quantark.{root}"
            ):
                return None
        else:
            # A real installed distribution wins over the alias. Namespace
            # placeholders (spec without a loader, e.g. a stray directory on
            # sys.path) do not count as a real package — and neither does a
            # subpackage of quantark itself becoming findable because the
            # quantark/ directory leaked onto sys.path: deferring there would
            # load duplicate module objects and split class identities.
            real_spec = importlib.machinery.PathFinder.find_spec(fullname)
            if (
                real_spec is not None
                and real_spec.loader is not None
                and not _is_own_package_leak(real_spec)
            ):
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
    """Prepend the legacy-alias finder to ``sys.meta_path`` (idempotent)."""
    if not any(isinstance(f, _LegacyAliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LegacyAliasFinder())


install()

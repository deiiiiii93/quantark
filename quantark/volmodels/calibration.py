"""DEPRECATED: moved to :mod:`quantark.volcalibration.calibrate` in 0.4.0.

``quantark.volmodels`` is asset-neutral and imports no pricing environments or
market-data containers; per-day calibration needs both, so it moved one layer
up into ``quantark.volcalibration``.  This shim is kept until 0.5.0.

The disk cache key is ``sha256(surface_sha|variant|config_fingerprint)`` and
contains no module paths, so entries written through either import path
interoperate -- a cache warmed before the move is still a hit after it.

**This is an alias, not a re-export.** ``sys.modules`` is pointed at the real
module, so ``import quantark.volmodels.calibration`` yields the very object
that now lives at ``quantark.volcalibration.calibrate``.

That distinction is load-bearing.  This module *was* the implementation, so
callers patch its globals -- ``monkeypatch.setattr(vol_calibrators,
"build_dupire_local_vol", ...)`` is how the OTC tests substitute kernels.  A
re-export shim keeps ``import`` working while silently breaking every one of
those patches: the name would be rebound here, and the calibrator would go on
resolving the original from its own module globals.  Aliasing keeps the two
names one object, so patching either reaches the code that runs.
"""

import sys

from quantark.volcalibration import calibrate as _calibrate

# HESTON_PRESETS reaches callers through this alias too: calibrate.py imports
# it from volcalibration.config, so it is an attribute of that module.
sys.modules[__name__] = _calibrate

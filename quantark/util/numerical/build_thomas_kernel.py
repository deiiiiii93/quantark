"""Build the optional compiled Thomas kernel (spec WS-A2).

    python -m quantark.util.numerical.build_thomas_kernel

QuantArk ships as a pure-Python wheel, so this accelerator is built locally on
demand rather than distributed as a binary (decision D-1). Without it the
package uses the NumPy sweep and behaves identically -- this only buys speed.

The flags are load-bearing, not cosmetic: ``-ffp-contract=off`` prevents the
compiler from fusing a multiply and an add into an FMA, which would change the
rounding sequence and break bit-identity with the NumPy sweep. Do not add
``-ffast-math``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "thomas_kernel.c"
BASE_FLAGS = ("-O3", "-ffp-contract=off", "-fPIC", "-shared", "-std=c99")


def library_name() -> str:
    if sys.platform == "darwin":
        return "libthomas_kernel.dylib"
    if sys.platform.startswith("win"):
        return "thomas_kernel.dll"
    return "libthomas_kernel.so"


def find_compiler() -> "str | None":
    configured = sysconfig.get_config_var("CC")
    if configured:
        candidate = configured.split()[0]
        if shutil.which(candidate):
            return candidate
    for name in ("cc", "clang", "gcc"):
        if shutil.which(name):
            return name
    return None


def build(verbose: bool = True) -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError(f"kernel source missing: {SOURCE}")
    compiler = find_compiler()
    if compiler is None:
        raise RuntimeError(
            "no C compiler found (tried the configured CC, then cc/clang/gcc); "
            "the pure-NumPy solver remains in use"
        )
    output = HERE / library_name()
    command = [compiler, *BASE_FLAGS, str(SOURCE), "-o", str(output)]
    if verbose:
        print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    return output


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet", action="store_true", help="do not echo the compiler command"
    )
    args = parser.parse_args(argv)
    try:
        output = build(verbose=not args.quiet)
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"kernel build failed: {exc}", file=sys.stderr)
        return 1
    # Import after building so the report reflects reality rather than intent.
    from quantark.util.numerical import tridiag

    print(f"built {output}")
    print(f"backend after rebuild (fresh interpreter required): {tridiag.tridiag_backend()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

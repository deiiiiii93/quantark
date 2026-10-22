"""Run the offline CFETS USD/CNY calibration pipeline from a frozen snapshot.

Stage 01 stays a deliberate network boundary and is not invoked here.  This
runner executes stages 02–07 with one tag and fails immediately if any stage
fails.  Historical tags must already have comparable stage-01 and stage-04
artifacts before they are supplied to ``--history-tags``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(HERE / script), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--history-tags",
        nargs="*",
        default=(),
        help="comparable historical snapshot/Heston tags to include before the current tag",
    )
    parser.add_argument("--fast", action="store_true", help="smoke resolutions, not publication")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _run(
        "02_build_fx_surface.py",
        "--tag",
        args.tag,
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir),
        "--tenor-set",
        "core",
    )
    source_snapshot = args.data_dir / f"cfets_usdcny_snapshot_{args.tag}.json"
    output_snapshot = output_dir / source_snapshot.name
    if source_snapshot.resolve() != output_snapshot.resolve():
        if not source_snapshot.is_file():
            raise FileNotFoundError(source_snapshot)
        shutil.copy2(source_snapshot, output_snapshot)
    common = ["--tag", args.tag, "--data-dir", str(output_dir), "--output-dir", str(output_dir)]
    _run("03_dupire_localvol.py", *common, *(("--fast",) if args.fast else ()))
    _run(
        "04_heston_calibration.py",
        *common,
        "--universe",
        "all",
        *(("--starts", "2", "--max-nfev", "150") if args.fast else ()),
    )
    _run("05_slv_calibration.py", *common, *(("--fast",) if args.fast else ()))

    diagnostic_tags = [*args.history_tags, args.tag]
    _run(
        "06_calibration_diagnostics.py",
        "--tags",
        *diagnostic_tags,
        "--output-tag",
        args.tag,
        "--universe",
        "core",
        "--data-dir",
        str(output_dir),
        "--output-dir",
        str(output_dir),
    )
    _run(
        "07_explainer.py",
        "--tag",
        args.tag,
        "--data-dir",
        str(output_dir),
    )
    print(output_dir / f"fx_calibration_explainer_{args.tag}.html")


if __name__ == "__main__":
    main()

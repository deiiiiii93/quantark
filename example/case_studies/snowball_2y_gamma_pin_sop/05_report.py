"""Stage 5: the report. One self-contained HTML page in the Bloomberg x Swiss UI language (``docs/UI_LANGUAGE.md``),
told from the snowball seller's side (spec C10, AC14, D13).

It reads the training selection, the stored ledger runs (``OUTPUT_DIR/runs/{train,test}``), the backtest payload,
the recorded states and the committed research files, and runs nothing. Test numbers are printed only when the
training selection and the test payload carry the four hashes of the current files (``gpin.measures.check_selection``
and ``gpin.measures.check_payload``); otherwise the page says which guard failed and shows the research, the sample and
the current training figures only. Any missing input degrades its own section to a stated "not available".

Run (from a script file), with ``R`` = ``example/case_studies/snowball_2y_gamma_pin_sop/05_report.py``:
    python R                                   # writes results/report.html
    python R --out <scratch>/report_train.html # anywhere else; nothing tracked is written
    python R --partial --runs-dir <scratch>/runs --selection <scratch>/selection.json \
             --states-dir <OUTPUT_DIR>/states_quick --out <scratch>/report.html      # a smoke run
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantark.util.exceptions import QuantArkException  # noqa: E402

from gpin import BOARD_PATH, CHAIN_PATH, OUTPUT_DIR, RESULTS_DIR, STATES_DIR  # noqa: E402
from gpin.report import ReportPaths, build_model, render  # noqa: E402

REPORT_PATH = RESULTS_DIR / "report.html"

#: One run of this stage: the inputs and the destination (a namedtuple: this file is also loaded by path).
Job = namedtuple("Job", ["paths", "out"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=Path, default=OUTPUT_DIR / "runs", help="holds train/ and test/")
    ap.add_argument("--selection", type=Path, default=RESULTS_DIR / "selection.json")
    ap.add_argument("--backtest", type=Path, default=RESULTS_DIR / "backtest.json")
    ap.add_argument("--states-dir", type=Path, default=STATES_DIR)
    ap.add_argument("--events", type=Path, default=RESULTS_DIR / "events.json")
    ap.add_argument("--board", type=Path, default=BOARD_PATH)
    ap.add_argument("--chain", type=Path, default=CHAIN_PATH)
    ap.add_argument("--probe", type=Path, default=RESULTS_DIR / "ki_gap_probe.json")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--partial", action="store_true",
                    help="a smoke run: admit a partial selection and a states directory holding only some notes")
    return ap.parse_args(argv)


def resolve(args: argparse.Namespace) -> Job:
    """The inputs and the destination of this run."""
    paths = ReportPaths(runs_dir=Path(args.runs_dir), selection=Path(args.selection), backtest=Path(args.backtest),
                        states_dir=Path(args.states_dir), events=Path(args.events), board=Path(args.board),
                        chain=Path(args.chain), probe=Path(args.probe), partial=bool(args.partial))
    return Job(paths, Path(args.out))


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, 0o644)                    # mkstemp creates 0600; a report is meant to be read
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    job = resolve(parse_args(argv))
    try:
        model = build_model(job.paths)
        write_atomic(job.out, render(model))
    except QuantArkException as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 2
    g = model["guard"]
    print(f"wrote {job.out}: test numbers {'SHOWN' if g['test_ok'] else 'NOT shown'}; training figures "
          f"{'shown' if g['training_ok'] else 'not shown'}")
    for problem in g["problems"]:
        print(f"  guard: {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
# Run a test list under one tracker-clock arm (NONE / A / B / C) with the runtime plugin; junit + hit log per arm.
#   run_arm.sh <worktree> <arm> <testlist-file> <out-dir>
# Uses the main checkout's venv with PYTHONPATH pinned to the worktree (a .pth otherwise imports the main checkout).
set -u
WT="$1"; ARM="$2"; LIST="$3"; OUT="$4"
EVIDENCE="$WT/docs/superpowers/specs/2026-10-09-tracker-clock-findings-evidence"
mkdir -p "$OUT/log_$ARM"
cd "$WT" || exit 2
env PYTHONPATH="$WT:$WT/test:$EVIDENCE" TRACKER_CLOCK_OPTION="$ARM" TRACKER_CLOCK_LOG="$OUT/log_$ARM" \
  /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -p tracker_clock_options -n 2 -q -p no:cacheprovider \
  --junitxml="$OUT/junit_$ARM.xml" $(cat "$LIST") > "$OUT/pytest_$ARM.txt" 2>&1
echo "exit $?" >> "$OUT/pytest_$ARM.txt"

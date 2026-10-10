#!/bin/sh
# Run the autocallable-MC test files under the w17_grid_probe plugin (mode: detect | patch), 2 workers.
# Usage: sh run_probe.sh detect|patch <file-list>
WT=/Users/fuxinyao/quant-ark/.claude/worktrees/agent-ab1b5cb9e2b9eb8b4
D=$WT/docs/superpowers/specs/2026-10-10-snowball-mc-twin-diagnosis
MODE=$1
FILES=$(cat "$2")
rm -f "$D"/probe_"$MODE".log.*
cd "$WT" || exit 1
env W17_PROBE_MODE="$MODE" W17_PROBE_LOG="$D/probe_$MODE.log" PYTHONPATH="$WT:$WT/test:$D" \
  /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n 2 -p w17_grid_probe -q -p no:cacheprovider \
  -o addopts="" $FILES > "$D/probe_$MODE.pytest.txt" 2>&1
echo "exit $?" >> "$D/probe_$MODE.pytest.txt"

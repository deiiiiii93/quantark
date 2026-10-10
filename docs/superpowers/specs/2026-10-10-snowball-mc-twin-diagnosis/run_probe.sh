#!/bin/sh
# Run test files under the w17_grid_probe plugin (mode: detect | patch), 2 xdist workers. probe_site/ on PYTHONPATH
# installs the probe via sitecustomize in every interpreter, so children spawned by a certification pipeline are
# probed too. Usage: sh run_probe.sh <mode> <file-list> [<tag>]   (tag names the outputs; default = mode)
WT=/Users/fuxinyao/quant-ark/.claude/worktrees/agent-ab1b5cb9e2b9eb8b4
D=$WT/docs/superpowers/specs/2026-10-10-snowball-mc-twin-diagnosis
MODE=$1
FILES=$(cat "$2")
TAG=${3:-$MODE}
rm -f "$D"/probe_"$TAG".log.*
cd "$WT" || exit 1
env W17_PROBE_MODE="$MODE" W17_PROBE_LOG="$D/probe_$TAG.log" PYTHONPATH="$WT:$WT/test:$D:$D/probe_site" \
  caffeinate -i /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n 2 -p w17_grid_probe -q -p no:cacheprovider \
  -o addopts="" $FILES > "$D/probe_$TAG.pytest.txt" 2>&1
echo "exit $?" >> "$D/probe_$TAG.pytest.txt"

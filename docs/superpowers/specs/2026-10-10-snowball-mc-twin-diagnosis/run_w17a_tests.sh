#!/bin/sh
# W17a run set (2 xdist workers): every file importing an autocallable MC engine (probe_files.txt: snowball /
# Phoenix / KO-reset MC, LV / Heston / SLV autocallable MC, carried-KI, dated-expiry, test_ko_reset_past_records,
# ...), the new test, theta_boundary, the replay golden, test/pde_grid, modelvalidation anchors + banked certificates.
WT=/Users/fuxinyao/quant-ark/.claude/worktrees/agent-ab1b5cb9e2b9eb8b4
D=$WT/docs/superpowers/specs/2026-10-10-snowball-mc-twin-diagnosis
cd "$WT" || exit 1
FILES="$(cat "$D/probe_files.txt") test/test_mc_time_node_merge.py test/theta_boundary test/test_replay_goldens.py \
test/pde_grid test/modelvalidation/test_anchors.py test/modelvalidation/test_banked_certificates.py"
env PYTHONPATH="$WT:$WT/test" caffeinate -i /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n 2 -q \
  -p no:cacheprovider -rfE $FILES > "$D/w17a_tests.txt" 2>&1
echo "exit $?" >> "$D/w17a_tests.txt"

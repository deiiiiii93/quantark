#!/bin/sh
# The W17a pre-fix probe over the certification directories, one directory at a time (2 xdist workers each).
D=/Users/fuxinyao/quant-ark/.claude/worktrees/agent-ab1b5cb9e2b9eb8b4/docs/superpowers/specs/2026-10-10-snowball-mc-twin-diagnosis
for dir in pde_grid mo_volmodels modelvalidation; do
  echo "test/$dir" > "$D/probe_cert_$dir.files"
  sh "$D/run_probe.sh" detect "$D/probe_cert_$dir.files" "cert_$dir"
  rm -f "$D/probe_cert_$dir.files"
done
echo done > "$D/probe_cert.done"

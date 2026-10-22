# Ledger — plan: docs/superpowers/plans/2026-10-22-branch-cleanup.md

Task 1: complete — 12 archive/<branch> tags created (14 total with 2 pre-existing).
Task 2: complete — deleted accrual-time, fix-cb-default-drift, ko-reset-lifecycle-v3, range-accrual-tenor, worktree-ship-tf-tree (git branch -d, zero unique patches each).
Task 3: finding — hazard-from-spread's two bond commits are already in main as a0c5d364/90cf13fe (git cherry '-'); its ONLY unique patch is cc975cac, the same patch afv-default-payoff carries. Awaiting user: land cc975cac on main.
Task 3: complete — user approved. cc975cac rebased to 0e3b20c7 and ff-merged to main (main checkout). afv-default-payoff deleted (-d); hazard-from-spread deleted (-D, git cherry showed 0 unique patches: its bond commits were already in main as a0c5d364/90cf13fe, its untrack patch landed as 0e3b20c7).
Task 3: Ruling: full test suite skipped before landing — the commit changes only the git index and a .gitignore (no code) — cost if wrong: none observable; the smoke test regenerates both files itself.
Task 3: note — main checkout had the two sample files dirty (HTML diff not pure float churn: "bound hits" 1/5 -> 2/5); saved to scratchpad main-checkout-sample-churn/ before discarding. The ff also removed both files from main's disk.
Note for project 2: main checkout has UNTRACKED example/intraday_ko_pin_risk/, intraday_pin_risk/, intraday_snowball_near_ki_demo.py, intraday_snowball_near_ko_demo.py, snowball_ftp_study/, standalone_rho/, test/intraday/test_ko_pin_risk_study.py — untracked work the reorg must not lose.
Task 4: complete — user said "abandon them". Evidence shown: 70–100% of each branch's added lines already present in main's files; each unique patch has a same-subject commit on main. Deleted (-D, archive/<name> tags kept): worktree-agent-af075ed0b71bc6f1f, worktree-agent-a1ccca52a141596d1, ko-reset-lifecycle-v2, handoff-small-fixes, codex/equity-option-settlement-date. Restore: git branch <name> archive/<name>.
Task 5: remote refs — reported to user, no remote action taken (awaiting per-ref word).
Project 1: complete on reorg-p1-modelvalidation (32f3f624, 6a898fac, + golden split). Subset verification 823 passed/1091 skipped vs baseline 821/1091 (+2 new alias tests). Full-suite run pending.
Project 1: full suite 18813 passed / 4 failed -> the 4 were mine (pde_3a/products.py joined root onto the legacy id, bypassing the alias); fixed, the 4 pass (6/6). Lesson: grep consumers of the data files, not only the folder string.

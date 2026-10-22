# Branch Cleanup (project 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every local and remote branch so the `example/` and `test/` reorganisation (projects 1–4) starts from a repository with no unmerged work that touches those folders.

**Architecture:** Inventory every branch against `main` by patch content (`git cherry`), not just ancestry, because many branches were rebased or cherry-picked. Safe deletions (fully contained in `main`) go first; each branch carrying unique patches is shown to the user and gets exactly one user-named outcome: land, keep (rebase over the reorg later), or abandon. Every tip is tagged before anything is deleted, so nothing is unrecoverable.

**Tech Stack:** git only. Full pytest run (py3.11 banking venv) before any land.

**Spec:** `docs/superpowers/specs/2026-10-22-example-and-test-reorg-design.md` (project 0)

## Global Constraints

- Nothing unmerged is deleted until the user names that branch. "Unmerged" means `git cherry main <branch>` shows at least one `+` line.
- Never push, never delete a remote branch, and never force-push without the user's explicit word for that specific action. `origin` is GitHub (public), `gitee` is a mirror.
- Land means `git merge --ff-only` onto `main` after a rebase, with the **full** test suite green (no sampling). The 4 known mo_volmodels/otc failures on main are expected, not new.
- Never use bare `git stash`. Stash stack is currently empty and must stay untouched.
- The main checkout (`/Users/fuxinyao/quant-ark`) is on `main`; this worktree is on `hazard-from-spread`. Check the branch before every merge (other sessions move branches).
- Tag naming follows the existing precedent `archive/<name>`.
- Commits made by this plan end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

- A branch whose commits are all patch-equivalent in `main` but whose ancestry says "unmerged": delete is safe, and the plan must say so from `git cherry`, not from `--is-ancestor`.
- A branch checked out in a worktree (`hazard-from-spread` here): `git branch -d` refuses; do not switch the worktree out from under another session.
- `afv-default-payoff` holds commit `cc975cac` (the sample-file untrack). It looks like a leftover but its single patch is not in `main`; it must land or be re-created, not deleted with the "stale" ones.
- Branches 200–1100 commits behind `main` (`codex/equity-option-settlement-date`, `handoff-small-fixes`, `ko-reset-lifecycle-v2`): a rebase may conflict heavily; the outcome offered must include "abandon, keeping an archive tag".
- Remote-tracking refs that look stale (`gitee/public-main-snapshot`) may be deliberate publication snapshots; deleting them is outward-facing.

## Inventory (measured 2026-10-22, main = `90cf13fe`)

| Branch | vs main | Unique patches (`git cherry +`) | Touches example/test |
| --- | --- | --- | --- |
| `accrual-time` | 0 ahead | 0 | 0 |
| `fix-cb-default-drift` | 0 ahead | 0 | 0 |
| `ko-reset-lifecycle-v3` | 0 ahead | 0 | 0 |
| `range-accrual-tenor` | 0 ahead | 0 | 0 |
| `worktree-ship-tf-tree` | 0 ahead | 0 | 0 |
| `afv-default-payoff` | 1 ahead, 2 behind | 1 (`cc975cac` untrack sample outputs) | 3 |
| `hazard-from-spread` (checked out here) | 3 ahead, 2 behind | 1 | 6 |
| `worktree-agent-af075ed0b71bc6f1f` | 1 ahead, 184 behind | 1 (Asian theta fixing) | 1 |
| `worktree-agent-a1ccca52a141596d1` | 15 ahead, 206 behind | 1 (numeric FX theta) | 10 |
| `ko-reset-lifecycle-v2` | 40 ahead, 277 behind | 3 | 37 |
| `handoff-small-fixes` | 32 ahead, 295 behind | 7 | 34 |
| `codex/equity-option-settlement-date` | 16 ahead, 1144 behind | 6 | 29 |
| remote `gitee/codex/equity-cash-legs`, `gitee/codex/ko-reset-fixes`, `gitee/public-main-snapshot`, `origin/feat/pde-event-distribution-redesign` | 0 ahead | 0 | 0 |

`main` is 35 commits ahead of `origin/main` and 1202 ahead of `gitee/main` (local, unpushed; not this plan's concern).

---

### Task 1: Archive tags and a recorded baseline

**Files:**
- Create: `docs/superpowers/plans/2026-10-22-branch-cleanup-ledger.md` (the inventory table above plus each decision as it is made)

**Interfaces:**
- Produces: tag `archive/<branch-name>` for every non-`main` local branch tip; the ledger file later tasks append to.

- [ ] **Step 1: Tag every local non-`main` branch tip** as `archive/<branch-name>` (slashes in branch names preserved), skipping any tag that already exists. Verify with `git tag -l 'archive/*'`; expected: 12 tags, one per branch in the inventory table (excluding the remote refs).
- [ ] **Step 2: Write the ledger** with the inventory table and a "Decision" column left empty.
- [ ] **Step 3: Commit** the ledger on a new branch `chore/branch-cleanup` created from `main` in a fresh worktree via the native EnterWorktree tool (never `git worktree add`); verify HEAD with `git rev-parse main HEAD`. Docs are excluded by `.git/info/exclude`, so use `git add -f`.

### Task 2: Delete branches already contained in `main`

**Interfaces:**
- Consumes: Task 1 tags.

- [ ] **Step 1: For each of** `accrual-time`, `fix-cb-default-drift`, `ko-reset-lifecycle-v3`, `range-accrual-tenor`, `worktree-ship-tf-tree`, re-run `git cherry main <branch>` and confirm zero `+` lines. Expected: all zero.
- [ ] **Step 2: Delete with `git branch -d`** (lower-case: git refuses if unmerged, which is the safety net). Expected: five "Deleted branch" lines, no errors.
- [ ] **Step 3: Record** each deletion and its archive tag in the ledger.

### Task 3: Land or resolve the two small current-work branches

**Files:** none new; moves `afv-default-payoff` (`cc975cac`) and `hazard-from-spread` (3 commits, 1 unique patch) toward `main`.

- [ ] **Step 1: Show the user** `git log --oneline main..<branch>` and `git cherry -v main <branch>` for each, and ask per branch: land now, keep, or abandon. Wait for the answer.
- [ ] **Step 2: For each branch the user says land:** rebase onto `main`, run the **full** suite, and on green `git merge --ff-only` in the main checkout (verify branch first). Expected: suite shows only the 4 known failures.
- [ ] **Step 3: Update the ledger.** Delete the branch only after it has landed (`git branch -d` must succeed) and, for `hazard-from-spread`, only after this worktree is no longer on it.

### Task 4: Decide the five old branches, one at a time

**Files:** none new.

Branches: `worktree-agent-af075ed0b71bc6f1f`, `worktree-agent-a1ccca52a141596d1`, `ko-reset-lifecycle-v2`, `handoff-small-fixes`, `codex/equity-option-settlement-date`.

- [ ] **Step 1: For one branch**, print `git cherry -v main <branch>` (the unique patches) and the `example/`+`test/` files they touch. Present it to the user in plain language: what each unique patch does, whether `main` already has an equivalent change (by searching `main` for the same function/test names), and the rebase difficulty (try `git rebase main` on a throwaway worktree branch and report the count of conflicted files, then abort).
- [ ] **Step 2: Wait for the user's outcome for that branch:** land, keep (rebase over the reorg later), or abandon. Abandon keeps its `archive/` tag.
- [ ] **Step 3: Execute the outcome** (land per Task 3 step 2; keep = no action; abandon = `git branch -D <branch>` only after the user named it) and record it in the ledger. Repeat for the next branch.

### Task 5: Remote-tracking refs and final check

- [ ] **Step 1: Report to the user** the four remote refs with zero unique patches and their dates; ask whether each should be deleted on the remote. Do nothing remote without a per-ref yes. `gitee/public-main-snapshot` is flagged as a possible deliberate snapshot.
- [ ] **Step 2: Final state check:** `git branch -a`, `git worktree list`, `git stash list`. Expected: `main`, `chore/branch-cleanup` (if kept), any branches the user chose to keep, no stray worktrees, empty stash.
- [ ] **Step 3: Complete the ledger** with the final branch list, and tell the user which kept branches must be rebased over the reorg (their file lists from the inventory).

# Model Validation Workflow

Detailed workflow diagram and decision points for the model orchestrator.

## Workflow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         MODEL VALIDATION WORKFLOW                             │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ 1. Initialize       │
│ - Create output dir │
│ - Create tasks.md   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────┐
│ New model/breaking  │─YES─►│ 2. OpenSpec │
│ change?             │     │   Proposal  │
└──────────┬──────────┘     └──────┬──────┘
           │NO                     │
           │◄──────────────────────┘
           ▼
┌─────────────────────┐     ┌─────────────┐
│ Skip research?      │─NO──►│ 3. Research │
│ (User option)       │     │   Phase     │
└──────────┬──────────┘     └──────┬──────┘
           │YES                    │
           │◄──────────────────────┘
           ▼
┌─────────────────────┐
│ 4. Development      │
│ (Developer A)       │
│ - model-developer   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. Logic Validation │
│ (Developer B)       │
│-model-logic-validat │
└──────────┬──────────┘
           │
           ▼
      ┌────────────┐
      │ Gate Pass? │
      └─────┬──────┘
            │
    ┌───────┴───────┐
    │YES            │NO
    ▼               ▼
┌───────┐     ┌─────────────┐
│Continue│     │ Rollback to │
└───┬───┘     │ Development │
    │         └──────┬──────┘
    │                │
    │                └──────► (back to step 4)
    ▼
┌─────────────────────┐
│ 6. Reviews          │
│ (Run in Parallel)   │
├─────────────────────┤
│ 6a. Performance     │
│ 6b. Security        │
│ 6c. Code Quality    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────┐
│ Non-MC engine?      │─YES─►│ 7. MC Cross-│
│                     │     │  Validation │
└──────────┬──────────┘     └──────┬──────┘
           │NO                     │
           │◄──────────────────────┘
           ▼
┌─────────────────────┐
│ 8. Package Reports  │
│ VALIDATION-PACKAGE  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────┐
│ OpenSpec used?      │─YES─►│ 9. OpenSpec │
│                     │     │   Archive   │
└──────────┬──────────┘     └──────┬──────┘
           │NO                     │
           │◄──────────────────────┘
           ▼
┌─────────────┐
│    END      │
└─────────────┘
```

## Decision Points

### Decision 1: OpenSpec Required?

**Question**: Is this a new model or breaking change?

| Scenario | Decision |
|----------|----------|
| New pricing model | YES - Create proposal |
| Breaking API change | YES - Create proposal |
| Bug fix | NO - Skip |
| Minor enhancement | NO - Skip |
| User requests skip | NO - Skip |

### Decision 2: Skip Research?

**Question**: Is research phase needed?

| Scenario | Decision |
|----------|----------|
| Well-known model (BS, etc.) | Skip |
| Comprehensive docs exist | Skip |
| User requests skip | Skip |
| Complex/novel model | Research |
| Missing reference materials | Research |

### Decision 3: Gate Pass?

**Question**: Did Developer B validation pass?

| Gate Result | Action |
|-------------|--------|
| PASS (< 0.1% error) | Continue |
| PASS_WITH_NOTES | Continue, document |
| FAIL | Rollback |

**Rollback Process**:
1. Document issues from gate report
2. Developer A addresses issues
3. Re-run Developer B validation
4. Max 3 rollbacks before escalation

### Decision 4: Run MC Cross-Validation?

**Question**: Should MC comparison be run?

| Scenario | Decision |
|----------|----------|
| Target is MC engine | SKIP |
| No MC engine exists | Ask user |
| Analytical/PDE/Quad engine | RUN |
| User requests skip | SKIP |

## State Machine

```
States: INIT → PROPOSAL → RESEARCH → DEV → VALIDATION → REVIEWS → CROSS → PACKAGE → ARCHIVE → DONE

Transitions:
  INIT → PROPOSAL (if new model)
  INIT → RESEARCH (if not new and research needed)
  INIT → DEV (if skip research)

  PROPOSAL → RESEARCH (after approval)
  PROPOSAL → DEV (if skip research)

  RESEARCH → DEV (always)

  DEV → VALIDATION (always)

  VALIDATION → DEV (on gate fail - rollback)
  VALIDATION → REVIEWS (on gate pass)

  REVIEWS → CROSS (if non-MC)
  REVIEWS → PACKAGE (if MC or skip)

  CROSS → PACKAGE (always)

  PACKAGE → ARCHIVE (if OpenSpec used)
  PACKAGE → DONE (if no OpenSpec)

  ARCHIVE → DONE (always)
```

## Parallel Execution Points

### Step 6: Reviews (Parallel)

All three reviews can run concurrently:
- code-performance-reviewer
- code-security-checker
- code-simplifier

**Coordination**:
- Start all three simultaneously
- Wait for all to complete
- Aggregate results
- Continue to next step

## Error Recovery

### Subagent Failure

```
if subagent_fails:
    log_error(task_file)
    options = [RETRY, SKIP, ABORT]
    user_choice = ask_user(options)

    if user_choice == RETRY:
        retry_subagent()
    elif user_choice == SKIP:
        mark_skipped(step)
        continue_workflow()
    else:  # ABORT
        cleanup()
        end_workflow()
```

### Gate Failure Loop

```
gate_failures = 0
MAX_GATE_FAILURES = 3

while gate_fails and gate_failures < MAX_GATE_FAILURES:
    gate_failures += 1
    document_issues()
    rollback_to_dev()

if gate_failures >= MAX_GATE_FAILURES:
    escalate_to_user("Gate has failed 3 times")
    options = [CONTINUE_TRYING, ABORT, SKIP_WITH_RISK]
    handle_user_decision()
```

## Output Artifacts

### Per-Step Outputs

| Step | Output Location |
|------|-----------------|
| 1. Init | `tasks.md` |
| 2. OpenSpec | `changes/<id>/proposal.md` |
| 3. Research | `research/research-report.md` |
| 4. Dev | `development/dev-report.md` |
| 5. Validation | `validation/gate-report.md` |
| 6a. Perf | `reviews/performance-report.md` |
| 6b. Security | `reviews/security-report.md` |
| 6c. Quality | `reviews/code-quality-report.md` |
| 7. Cross-Val | `cross-validation/mc-comparison-report.md` |
| 8. Package | `VALIDATION-PACKAGE.md` |

### Final Package Contents

1. All individual reports consolidated
2. Executive summary with pass/fail
3. Links to detailed reports
4. Sign-off section

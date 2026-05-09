---
name: model-orchestrator
description: |
  Master coordinator for model development and validation workflow following SR 11-7 standards.
  Manages the complete lifecycle: research, development, validation, reviews, and packaging.
  Use when the user asks to:
  - Start a new model development project with formal validation
  - Run the complete model validation workflow
  - Coordinate model development subagents
  - Generate a model validation package
  Triggers: "model validation", "validate model", "model development workflow", "SR 11-7", "model orchestrator"
---

# Model Orchestrator Skill

Master coordinator for the complete model development and validation workflow, following Federal Reserve SR 11-7 model risk management guidelines.

## When This Skill Activates

Codex should use this skill when:
- User wants to develop a new pricing model with formal validation
- User explicitly requests model validation workflow
- User mentions SR 11-7 or regulatory model validation
- User wants to coordinate multiple validation steps

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODEL ORCHESTRATOR                                    │
│  Coordinates workflow, manages file-based task tracking, packages reports    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ model-researcher │          │ model-developer │          │model-logic-     │
│  (Optional)      │    ───►  │  (Developer A)  │    ───►  │  validator      │
│  Research phase  │          │  Production code │          │  (Developer B)  │
└─────────────────┘          └─────────────────┘          └─────────────────┘
                                      │                            │
                                      │                     Gate Report
                                      │                     Pass/Fail
                                      │                            │
         ┌────────────────────────────┼────────────────────────────┤
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│code-performance-│          │code-security-   │          │  code-simplifier│
│    reviewer     │          │    checker      │          │                 │
│  Speed optimize │          │  OWASP/vulns    │          │  Clean/sanity   │
└─────────────────┘          └─────────────────┘          └─────────────────┘
                                      │
                                      ▼
                            ┌─────────────────┐
                            │model-cross-     │
                            │  validator      │
                            │  MC comparison  │
                            └─────────────────┘
                                      │
                                      ▼
                            ┌─────────────────┐
                            │ FINAL PACKAGE   │
                            │ Validation pkg  │
                            └─────────────────┘
```

---

## Complete Workflow

```
1. Initialize        → Create output directory, task file
2. OpenSpec (if new) → Trigger proposal for new model (if applicable)
3. Research          → model-researcher (optional, can skip)
4. Development       → model-developer (Developer A)
5. Logic Validation  → model-logic-validator (Developer B)
   └── Gate Check    → Pass: continue, Fail: rollback to step 4
6. Reviews (parallel)→ Performance, Security, Code Quality
7. Cross-Validation  → model-cross-validator (if non-MC)
8. Package           → Consolidate all reports
9. OpenSpec Archive  → Archive change with artifacts (if applicable)
```

---

## Step 1: Initialize

### Create Output Directory Structure

```
model-validation-output/<model-name>/
├── tasks.md                    # Task tracking (file-based)
├── research/
│   └── research-report.md
├── development/
│   ├── code/                   # Developer A code location reference
│   └── dev-report.md
├── validation/
│   ├── independent-impl/       # Developer B code
│   └── gate-report.md
├── reviews/
│   ├── performance-report.md
│   ├── security-report.md
│   └── code-quality-report.md
├── cross-validation/
│   └── mc-comparison-report.md
└── VALIDATION-PACKAGE.md       # Final consolidated report
```

### Initialize Task Tracking File

```markdown
# Model Validation Tasks / 模型验证任务

**Model**: [Model Name]
**Started**: YYYY-MM-DD HH:MM
**Status**: IN_PROGRESS / COMPLETED / BLOCKED

---

## Task Progress / 任务进度

| # | Task | Status | Started | Completed | Notes |
|---|------|--------|---------|-----------|-------|
| 1 | Initialize | DONE | ... | ... | |
| 2 | OpenSpec Proposal | PENDING/SKIP | | | |
| 3 | Research | PENDING/SKIP | | | |
| 4 | Development (A) | PENDING | | | |
| 5 | Validation (B) | PENDING | | | |
| 6a | Performance Review | PENDING | | | |
| 6b | Security Review | PENDING | | | |
| 6c | Code Quality | PENDING | | | |
| 7 | MC Cross-Validation | PENDING | | | |
| 8 | Package | PENDING | | | |
| 9 | OpenSpec Archive | PENDING/SKIP | | | |

---

## Current Status / 当前状态

**Active Task**: [Task Name]
**Blockers**: [Any blockers]
**Next Steps**: [What happens next]

---

## History / 历史记录

### YYYY-MM-DD HH:MM - [Event]
[Description]

### ...
```

---

## Step 2: OpenSpec Proposal (Conditional)

### When to Create Proposal

Create OpenSpec proposal if:
- New model being added to QuantArk
- Breaking changes to existing model
- Architecture changes

Skip if:
- Bug fix to existing model
- Minor enhancements
- User explicitly requests skip

### Proposal Integration

```markdown
**OpenSpec Change**: changes/YYYY-MM-DD-<model-name>/

Proposal triggers:
- `openspec-propose` skill invocation
- Wait for approval before proceeding to development
```

---

## Step 3: Research (Optional)

### Invoke model-researcher

**Input to researcher:**
- Model name and type
- User-provided reference materials
- Specific focus areas

**Output from researcher:**
- `research/research-report.md`
- Confidence levels
- Benchmark values
- Edge case scenarios

### Skip Conditions

Skip research if:
- User explicitly requests skip
- Model is well-known (e.g., Black-Scholes for vanilla)
- Comprehensive reference documentation already exists

---

## Step 4: Development (Developer A)

### Invoke model-developer

**Input to developer:**
- Model specification
- Research report (if available)
- Reference materials

**Output from developer:**
- Engine implementation in `asset/.../engine/`
- Reference documentation
- `development/dev-report.md`

### Update Task Status

After development:
```markdown
| 4 | Development (A) | DONE | ... | ... | Files: [list] |
```

---

## Step 5: Logic Validation (Developer B)

### Invoke model-logic-validator

**CRITICAL**: Developer B does NOT read Developer A's code.

**Input to validator:**
- Reference documentation path
- Research report path
- Benchmark values

**Output from validator:**
- Independent implementation in `validation/independent-impl/`
- `validation/gate-report.md`
- Pass/Fail decision

### Gate Decision

| Gate Result | Action |
|-------------|--------|
| **PASS** | Continue to reviews |
| **PASS_WITH_NOTES** | Continue, document notes |
| **FAIL** | Rollback to Development |

### Rollback on Failure

If gate FAILS:
1. Document failure reason
2. Notify user
3. Return to Step 4 with specific issues
4. Developer A addresses issues
5. Re-run Step 5

```markdown
### YYYY-MM-DD HH:MM - Gate FAILED
**Reason**: [Description from gate report]
**Action**: Rolling back to Development
**Issues for Developer A**:
1. [Issue 1]
2. [Issue 2]
```

---

## Step 6: Reviews (Parallel)

### Run Three Reviews in Parallel

These can be executed concurrently:

#### 6a. Performance Review
- Invoke: `code-performance-reviewer`
- Input: Developer A implementation files
- Output: `reviews/performance-report.md`

#### 6b. Security Review
- Invoke: `code-security-checker`
- Input: Developer A implementation files
- Output: `reviews/security-report.md`

#### 6c. Code Quality Review
- Invoke: `code-simplifier`
- Input: Developer A implementation files
- Output: `reviews/code-quality-report.md`

### Review Results

| Review | Pass? | Issues |
|--------|-------|--------|
| Performance | YES/NO | X critical, Y medium |
| Security | YES/NO | X critical, Y medium |
| Code Quality | YES/NO | X issues |

---

## Step 7: MC Cross-Validation (Conditional)

### Skip Conditions

Skip if:
- Model IS a Monte Carlo engine
- No corresponding MC engine exists and user accepts
- User explicitly requests skip

### Invoke model-cross-validator

**Input:**
- Target engine path
- Product type

**Output:**
- `cross-validation/mc-comparison-report.md`
- Pass/Fail with convergence analysis

---

## Step 8: Package All Reports

### Consolidate into Validation Package

Generate `VALIDATION-PACKAGE.md` with all results.

See [validation-package-template.md](validation-package-template.md) for format.

### Package Contents

1. Executive Summary
2. Model Specification
3. Research Summary (if applicable)
4. Development Summary
5. Validation Results (Gate Report)
6. Review Results
7. Cross-Validation Results
8. Final Recommendation
9. Appendices (links to detailed reports)

---

## Step 9: OpenSpec Archive (Conditional)

### Archive if OpenSpec was Used

If OpenSpec proposal was created:
- Invoke `openspec-archive-change`
- Include validation package as artifact
- Update specs if needed

---

## User Interaction Points

### Required User Decisions

| Point | Question | Options |
|-------|----------|---------|
| Start | Skip research? | Yes / No |
| Gate Fail | Rollback or abort? | Rollback / Abort |
| Review Issues | Address or defer? | Fix / Defer |
| MC Missing | Skip or create? | Skip / Create |
| Package | Approve? | Approve / Revise |

### Progress Updates

Provide status updates after each step:

```markdown
## Status Update / 状态更新

**Completed**: Research, Development
**Current**: Logic Validation (Developer B)
**Remaining**: Reviews, Cross-Validation, Package

**Notes**: Development complete. Developer B is independently verifying...
```

---

## Error Handling

### Subagent Failure

If any subagent fails:
1. Log error in task file
2. Notify user with details
3. Offer options: Retry / Skip / Abort

### Gate Failure Limit

After 3 gate failures:
```markdown
**Gate has failed 3 times.**
Options:
1. Continue trying (may indicate fundamental issue)
2. Abort validation and investigate
3. Skip gate with documented risk
```

---

## Bilingual Output

All reports use bilingual headers:

```markdown
## 1. 执行摘要 / Executive Summary
## 2. 模型规范 / Model Specification
## 3. 研究总结 / Research Summary
## 4. 开发总结 / Development Summary
## 5. 验证结果 / Validation Results
## 6. 审查结果 / Review Results
## 7. 交叉验证 / Cross-Validation
## 8. 最终建议 / Final Recommendation
```

---

## Integration with Existing Skills

### Skills Invoked by Orchestrator

| Step | Skill | Type |
|------|-------|------|
| 3 | model-researcher | Project-scoped |
| 4 | model-developer | Project-scoped |
| 5 | model-logic-validator | Project-scoped |
| 6a | code-performance-reviewer | Project-scoped |
| 6b | code-security-checker | Project-scoped |
| 6c | code-simplifier | Project-scoped |
| 7 | model-cross-validator | Project-scoped |

### OpenSpec Integration

| Step | OpenSpec Skill |
|------|----------------|
| 2 | openspec-propose |
| 9 | openspec-archive-change |

---

## Quick Start Command

To start a model validation workflow:

```
/model-orchestrator <model-name> [options]

Options:
  --skip-research    Skip the research phase
  --skip-openspec    Skip OpenSpec proposal/archive
  --skip-mc          Skip MC cross-validation
  --output-dir       Custom output directory
```

---

## Principles

1. **Independence**: Developer A and B work independently
2. **Traceability**: All decisions documented in task file
3. **Gates Matter**: Gate failures require resolution
4. **Parallel When Possible**: Reviews run concurrently
5. **User in Control**: Key decisions require user input
6. **Complete Package**: Final output is comprehensive validation record

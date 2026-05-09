---
name: code-simplifier
description: |
  Simplifies and refines code for clarity, consistency, and maintainability while preserving all functionality.
  Focuses on recently modified code unless instructed otherwise.
  Use when the user asks to:
  - Simplify or clean up code
  - Review code quality
  - Remove dead code or redundancy
  - Improve code readability
  - Perform sanity checks on implementation
  Triggers: "simplify code", "clean up", "code quality", "sanity check", "refactor for clarity"
---

# Code Simplifier Skill

Simplify and refine code for clarity, consistency, and maintainability while preserving all functionality.

## When This Skill Activates

Codex should use this skill when:
- User asks to simplify, clean up, or refactor code
- User requests a code quality review
- User wants to remove dead code or redundancy
- User asks for a sanity check on recent changes
- Part of a model validation workflow (invoked by orchestrator)

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODE SIMPLIFICATION WORKFLOW                  │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Identify Target Files   → Recent changes or specified   │
│ Step 2: Static Analysis         → Detect issues                 │
│ Step 3: Simplification Pass     → Apply improvements            │
│ Step 4: Sanity Checks           → Verify no regressions         │
│ Step 5: Generate Report         → Document changes              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Identify Target Files

### Default Behavior (No Files Specified)

Find recently modified files:
```bash
git diff --name-only HEAD~5 -- "*.py" "*.ts" "*.js"
```

### User-Specified Files

Accept explicit file paths from user input.

### Model Validation Mode

When invoked by model-orchestrator, receive file list from orchestrator:
- Developer A's implementation files
- Test files created during development

---

## Step 2: Static Analysis

### 2.1 Dead Code Detection

| Check | Pattern | Action |
|-------|---------|--------|
| Unused imports | Import not referenced | Remove |
| Unused variables | Assignment never read | Remove or prefix with `_` |
| Unreachable code | Code after `return`/`raise` | Remove |
| Empty branches | `if`/`else` with `pass` only | Remove or simplify |
| Commented-out code | Large comment blocks | Remove (use version control) |

### 2.2 Redundancy Detection

| Check | Pattern | Action |
|-------|---------|--------|
| Duplicate logic | Same code in multiple places | Extract to function (only if 3+ occurrences) |
| Unnecessary wrappers | Function that just calls another | Inline or document why needed |
| Redundant conditions | `if x: return True else: return False` | Simplify to `return x` |
| Double negation | `not not x` or `!(!x)` | Simplify to `x` |
| Redundant casts | `str(str(x))` | Remove outer cast |

### 2.3 Complexity Analysis

| Metric | Threshold | Action |
|--------|-----------|--------|
| Function length | > 50 lines | Consider splitting |
| Cyclomatic complexity | > 10 | Simplify branching |
| Nesting depth | > 4 levels | Extract inner logic |
| Parameter count | > 5 | Consider parameter object |

---

## Step 3: Simplification Pass

### 3.1 Naming Consistency

| Check | Correction |
|-------|------------|
| Mixed case styles | Standardize to project convention |
| Unclear abbreviations | Expand to meaningful names |
| Single-letter variables | Expand (except `i`, `j`, `k` in loops) |
| Inconsistent prefixes | Standardize (`is_`, `has_`, `get_`, `set_`) |

### 3.2 Expression Simplification

**Before → After:**
```python
# Conditional to boolean
if condition:
    result = True
else:
    result = False
# → result = condition

# Ternary simplification
value = x if x else default
# → value = x or default

# List comprehension (only if clearer)
result = []
for item in items:
    if condition(item):
        result.append(transform(item))
# → result = [transform(item) for item in items if condition(item)]
```

### 3.3 Control Flow Simplification

**Guard Clauses:**
```python
# Before: nested if
def process(x):
    if x is not None:
        if x > 0:
            # main logic
            pass

# After: guard clauses
def process(x):
    if x is None:
        return
    if x <= 0:
        return
    # main logic
```

**Early Returns:**
```python
# Before: else after return
if condition:
    return result_a
else:
    return result_b

# After: remove unnecessary else
if condition:
    return result_a
return result_b
```

### 3.4 Magic Number/String Elimination

```python
# Before: magic numbers
if age > 18:
    ...
if status == "ACTIVE":
    ...

# After: named constants
ADULT_AGE = 18
STATUS_ACTIVE = "ACTIVE"
if age > ADULT_AGE:
    ...
if status == STATUS_ACTIVE:
    ...
```

---

## Step 4: Sanity Checks

### 4.1 Functionality Preservation

**CRITICAL**: All simplifications must preserve functionality.

| Check | Method |
|-------|--------|
| Test suite passes | Run existing tests after changes |
| Edge cases handled | Verify edge case behavior unchanged |
| Error handling intact | Verify exceptions still raised appropriately |
| API contracts | Public interfaces unchanged |

### 4.2 Quality Verification Checklist

- [ ] No duplicate code (except intentional 2-3 similar lines)
- [ ] No magic numbers (except 0, 1, -1, 100, constants)
- [ ] Consistent error handling pattern
- [ ] Appropriate abstraction level (not over-engineered)
- [ ] No unnecessary complexity
- [ ] Type hints present (if project uses them)
- [ ] Docstrings complete (only for public methods)

### 4.3 Over-Engineering Detection

**Warning Signs to Check:**
- Helper functions used only once
- Abstractions with single implementation
- Feature flags for non-existent features
- Backward-compatibility code that's not needed
- Comments explaining "why not" for removed code
- Unused `_var` renames instead of deletion

---

## Step 5: Generate Report

### Report Structure

```markdown
# Code Simplification Report / 代码简化报告

**Date**: <date>
**Files Reviewed**: <count>
**Changes Made**: <count>

---

## Summary / 概要

| Category | Before | After | Improvement |
|----------|--------|-------|-------------|
| Lines of Code | XXX | XXX | -XX% |
| Cyclomatic Complexity | X.X | X.X | -XX% |
| Dead Code Lines | XX | 0 | Removed |
| Magic Numbers | XX | 0 | Named |

---

## Changes by File / 文件修改

### File: `path/to/file.py`

#### Dead Code Removed / 移除的死代码
- Line XX: Unused import `xyz`
- Lines XX-XX: Unreachable code after return

#### Simplifications / 简化
- Line XX: Simplified conditional to single expression
- Line XX: Replaced magic number with constant

#### Naming Improvements / 命名改进
- `x` → `calculation_result`
- `tmp` → `intermediate_value`

---

## Sanity Check Results / 完整性检查结果

| Check | Status | Notes |
|-------|--------|-------|
| Tests Pass | ✅/❌ | <details> |
| No Regressions | ✅/❌ | <details> |
| API Unchanged | ✅/❌ | <details> |

---

## Recommendations / 建议

1. <Recommendation 1>
2. <Recommendation 2>
```

---

## Language Support

### Python-Specific Patterns

```python
# Use list comprehensions (when clearer)
# Use f-strings over .format()
# Use pathlib over os.path
# Use dataclasses for data containers
# Use typing for type hints
```

### TypeScript/JavaScript-Specific Patterns

```typescript
// Use optional chaining (?.)
// Use nullish coalescing (??)
// Use destructuring
// Use template literals
// Use const over let where possible
```

---

## Bilingual Output Mode

When invoked by model-orchestrator, output bilingual section headers:

```markdown
## 1. 简化摘要 / Simplification Summary
## 2. 文件修改 / File Changes
## 3. 完整性检查 / Sanity Checks
## 4. 建议 / Recommendations
```

---

## Integration with Model Validation

When used as part of model validation workflow:

### Input from Orchestrator
- File paths of Developer A's implementation
- Test file paths
- Specific focus areas (if any)

### Output to Orchestrator
- `code-quality-report.md` in designated output directory
- Pass/Fail status based on quality metrics
- List of issues found (categorized by severity)

### Quality Gate Criteria

| Metric | Pass Threshold |
|--------|----------------|
| Dead code | 0 lines |
| Magic numbers | 0 (except standard) |
| Complexity per function | ≤ 10 |
| Test coverage | ≥ 80% (if measurable) |

---

## Principles

1. **Preserve Functionality**: Never change behavior, only presentation
2. **Minimal Changes**: Don't refactor beyond what's needed
3. **No Over-Engineering**: Three similar lines is better than a premature abstraction
4. **Respect Existing Style**: Match project conventions
5. **Document Decisions**: Explain why changes improve the code

# Task Tracking Template

This template is used to create the `tasks.md` file for tracking model validation progress.

---

# Model Validation Tasks / 模型验证任务

**Model 模型**: [MODEL_NAME]
**Type 类型**: [Analytical/MC/PDE/Quadrature/Tree]
**Product 产品**: [Product class name]
**Started 开始时间**: YYYY-MM-DD HH:MM
**Status 状态**: IN_PROGRESS

---

## Configuration / 配置

| Setting | Value |
|---------|-------|
| Output Directory | `model-validation-output/[model-name]/` |
| OpenSpec Required | YES/NO |
| Research Phase | INCLUDE/SKIP |
| MC Cross-Validation | INCLUDE/SKIP |

---

## Task Progress / 任务进度

| # | Task 任务 | Status 状态 | Started 开始 | Completed 完成 | Owner 负责人 | Notes 备注 |
|---|-----------|-------------|--------------|----------------|--------------|------------|
| 1 | Initialize 初始化 | PENDING | | | Orchestrator | |
| 2 | OpenSpec Proposal 提案 | PENDING/SKIP | | | User/Orchestrator | |
| 3 | Research 研究 | PENDING/SKIP | | | model-researcher | |
| 4 | Development 开发 | PENDING | | | model-developer | Developer A |
| 5 | Logic Validation 逻辑验证 | PENDING | | | model-logic-validator | Developer B |
| 6a | Performance Review 性能 | PENDING | | | code-performance-reviewer | |
| 6b | Security Review 安全 | PENDING | | | code-security-checker | |
| 6c | Code Quality 质量 | PENDING | | | code-simplifier | |
| 7 | MC Cross-Validation MC验证 | PENDING/SKIP | | | model-cross-validator | |
| 8 | Package 打包 | PENDING | | | Orchestrator | |
| 9 | OpenSpec Archive 归档 | PENDING/SKIP | | | Orchestrator | |

**Status Legend**: PENDING | IN_PROGRESS | DONE | BLOCKED | SKIPPED | FAILED

---

## Gate Status / 门禁状态

| Gate | Result | Attempt | Notes |
|------|--------|---------|-------|
| Logic Validation | PENDING | 0 | |

---

## Current Status / 当前状态

**Active Task 当前任务**: [Task Name]
**Progress 进度**: X/9 tasks completed
**Blockers 阻塞**: [None / Description]
**Next Steps 下一步**: [Description]

---

## Files Created / 创建的文件

| File | Step | Status |
|------|------|--------|
| `tasks.md` | 1 | Created |
| `research/research-report.md` | 3 | Pending |
| `development/dev-report.md` | 4 | Pending |
| `validation/gate-report.md` | 5 | Pending |
| `validation/independent-impl/` | 5 | Pending |
| `reviews/performance-report.md` | 6a | Pending |
| `reviews/security-report.md` | 6b | Pending |
| `reviews/code-quality-report.md` | 6c | Pending |
| `cross-validation/mc-comparison-report.md` | 7 | Pending |
| `VALIDATION-PACKAGE.md` | 8 | Pending |

---

## History / 历史记录

### YYYY-MM-DD HH:MM - Initialized
- Created output directory structure
- Created task tracking file
- Configuration set: [details]

### YYYY-MM-DD HH:MM - [Event Title]
[Description of what happened]

---

## Issues & Resolutions / 问题与解决

### Issue #1: [Title]
**Date**: YYYY-MM-DD
**Description**: [What went wrong]
**Resolution**: [How it was resolved]
**Status**: OPEN/RESOLVED

---

## Notes / 备注

[Any additional notes, decisions, or context]

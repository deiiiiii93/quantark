# Validation Package Template

This template is used to create the final `VALIDATION-PACKAGE.md` consolidating all validation results.

---

# Model Validation Package / 模型验证包

## [MODEL_NAME]

**Version 版本**: 1.0
**Date 日期**: YYYY-MM-DD
**Status 状态**: VALIDATED / NOT_VALIDATED / VALIDATED_WITH_NOTES

---

## 1. 执行摘要 / Executive Summary

### 1.1 Model Overview / 模型概述

| Attribute 属性 | Value 值 |
|---------------|----------|
| Model Name 模型名称 | [Name] |
| Model Type 模型类型 | Analytical/MC/PDE/Quadrature |
| Product 产品 | [Product class] |
| Engine Location 引擎位置 | `asset/.../engine/...` |
| Reference Doc 参考文档 | `asset/.../engine/docs/...` |

### 1.2 Validation Summary / 验证摘要

| Phase 阶段 | Status 状态 | Notes 备注 |
|------------|-------------|------------|
| Research 研究 | PASS/SKIP/N/A | |
| Development 开发 | COMPLETE | |
| Logic Validation 逻辑验证 | PASS/FAIL | Gate result |
| Performance Review 性能 | PASS/WARN/FAIL | |
| Security Review 安全 | PASS/WARN/FAIL | |
| Code Quality 质量 | PASS/WARN/FAIL | |
| MC Cross-Validation MC验证 | PASS/SKIP/N/A | |

### 1.3 Final Recommendation / 最终建议

**Decision 决定**: APPROVED FOR PRODUCTION / APPROVED WITH CONDITIONS / NOT APPROVED

**Rationale 理由**:
[Brief explanation of the decision]

**Conditions (if any) 条件**:
1. [Condition 1]
2. [Condition 2]

---

## 2. 模型规范 / Model Specification

### 2.1 Mathematical Description / 数学描述

**Pricing Formula 定价公式**:

$$
V = ...
$$

**Key Assumptions 关键假设**:
1. [Assumption 1]
2. [Assumption 2]

### 2.2 Parameters / 参数

| Parameter 参数 | Type 类型 | Range 范围 | Description 描述 |
|---------------|----------|------------|------------------|
| S | float | > 0 | Spot price |
| K | float | > 0 | Strike price |
| T | float | >= 0 | Time to maturity |
| r | float | any | Risk-free rate |
| sigma | float | >= 0 | Volatility |
| q | float | >= 0 | Dividend yield |

### 2.3 Edge Cases / 边界情况

| Condition 条件 | Expected Behavior 预期行为 |
|---------------|---------------------------|
| T = 0 | Intrinsic value |
| sigma = 0 | Deterministic price |
| S >> K | Deep ITM behavior |
| S << K | Deep OTM behavior |

---

## 3. 研究总结 / Research Summary

*[Skip this section if research was not performed]*

### 3.1 Sources Consulted / 参考来源

1. [Source 1]
2. [Source 2]

### 3.2 Key Findings / 关键发现

- [Finding 1]
- [Finding 2]

### 3.3 Confidence Level / 信心水平

**Overall 总体**: HIGH / MEDIUM / LOW

| Aspect 方面 | Confidence 信心 |
|------------|-----------------|
| Core Formula 核心公式 | HIGH/MED/LOW |
| Greeks 希腊字母 | HIGH/MED/LOW |
| Edge Cases 边界 | HIGH/MED/LOW |

**Details 详情**: See `research/research-report.md`

---

## 4. 开发总结 / Development Summary

### 4.1 Implementation Details / 实现详情

| Item 项目 | Value 值 |
|----------|----------|
| Engine Class 引擎类 | [ClassName] |
| Base Class 基类 | [BaseClassName] |
| Lines of Code 代码行数 | XXX |
| Methods 方法数 | X public, Y private |

### 4.2 Files Created / 创建的文件

| File 文件 | Description 描述 |
|----------|-----------------|
| `path/to/engine.py` | Main engine |
| `path/to/docs/engine.md` | Reference doc |

### 4.3 Input Validation / 输入验证

- [x] All parameters validated
- [x] Type checks implemented
- [x] Range checks implemented
- [x] NaN/Inf checks implemented
- [x] Sanity bounds applied

### 4.4 Edge Case Handling / 边界处理

- [x] T = 0 handled
- [x] sigma = 0 handled
- [x] Deep ITM/OTM handled

**Details 详情**: See `development/dev-report.md`

---

## 5. 验证结果 / Validation Results (Logic Gate)

### 5.1 Gate Decision / 门禁决定

**Result 结果**: PASS / FAIL / PASS_WITH_NOTES
**Attempts 尝试次数**: X

### 5.2 Comparison Summary / 比较摘要

| Metric 指标 | Value 值 |
|------------|---------|
| Test Cases 测试用例 | XX |
| Passed 通过 | XX |
| Failed 失败 | XX |
| Pass Rate 通过率 | XX.X% |
| Max Error 最大误差 | X.XX% |

### 5.3 Discrepancies / 差异

*[List any discrepancies and their explanations]*

### 5.4 Developer B Assessment / 开发B评估

[Brief assessment from Developer B]

**Details 详情**: See `validation/gate-report.md`

---

## 6. 审查结果 / Review Results

### 6.1 Performance Review / 性能审查

| Metric 指标 | Result 结果 | Target 目标 | Status 状态 |
|------------|------------|-------------|-------------|
| Single Price 单价 | X.XX ms | < X ms | PASS/FAIL |
| Full Greeks 全Greeks | X.XX ms | < X ms | PASS/FAIL |
| Memory 内存 | X MB | < X MB | PASS/FAIL |

**Issues Found 发现的问题**: X
**Recommendations 建议**: [Summary]

**Details 详情**: See `reviews/performance-report.md`

### 6.2 Security Review / 安全审查

| Category 类别 | Issues 问题数 | Severity 严重性 |
|--------------|--------------|-----------------|
| Dangerous Functions | X | CRITICAL/HIGH/MED |
| Input Validation | X | HIGH/MED/LOW |
| Safe Math | X | MED/LOW |
| Data Handling | X | HIGH/MED/LOW |

**Overall 总体**: PASS / FAIL

**Details 详情**: See `reviews/security-report.md`

### 6.3 Code Quality Review / 代码质量审查

| Aspect 方面 | Before 之前 | After 之后 | Status 状态 |
|------------|------------|-----------|-------------|
| Dead Code | X lines | 0 lines | PASS |
| Magic Numbers | X | 0 | PASS |
| Complexity | X | X | PASS/WARN |

**Details 详情**: See `reviews/code-quality-report.md`

---

## 7. 交叉验证 / Cross-Validation

*[Skip this section if MC cross-validation was not performed]*

### 7.1 MC Benchmark / MC基准

| Item 项目 | Value 值 |
|----------|----------|
| MC Engine MC引擎 | [EngineName] |
| Paths 路径数 | XXX,XXX |
| Tolerance 容差 | X% |

### 7.2 Results Summary / 结果摘要

| Metric 指标 | Value 值 |
|------------|---------|
| Test Cases 测试用例 | XX |
| Within Tolerance 容差内 | XX |
| Mean Error 平均误差 | X.XX% |
| Max Error 最大误差 | X.XX% |
| Systematic Bias 系统偏差 | YES/NO |

### 7.3 Convergence / 收敛性

[Brief convergence assessment]

**Details 详情**: See `cross-validation/mc-comparison-report.md`

---

## 8. 最终建议 / Final Recommendation

### 8.1 Approval Status / 批准状态

**Decision 决定**:
- [ ] APPROVED FOR PRODUCTION 批准生产
- [ ] APPROVED WITH CONDITIONS 有条件批准
- [ ] NOT APPROVED 不批准

### 8.2 Conditions / 条件

*[If approved with conditions]*

1. [Condition 1]
2. [Condition 2]

### 8.3 Outstanding Items / 待处理项

| Item 项目 | Priority 优先级 | Owner 负责人 | Due 截止 |
|----------|-----------------|--------------|----------|
| [Item 1] | HIGH/MED/LOW | [Name] | [Date] |

### 8.4 Risk Assessment / 风险评估

| Risk 风险 | Likelihood 可能性 | Impact 影响 | Mitigation 缓解 |
|----------|-------------------|-------------|-----------------|
| [Risk 1] | LOW/MED/HIGH | LOW/MED/HIGH | [Strategy] |

---

## 9. 签署 / Sign-Off

### 9.1 Validation Team / 验证团队

| Role 角色 | Completed By 完成者 | Date 日期 |
|----------|---------------------|-----------|
| Developer A 开发A | Codex | YYYY-MM-DD |
| Developer B 开发B | Codex | YYYY-MM-DD |
| Orchestrator 协调者 | Codex | YYYY-MM-DD |

### 9.2 Approval / 批准

| Role 角色 | Name 姓名 | Date 日期 | Signature 签名 |
|----------|----------|-----------|----------------|
| Model Owner 模型负责人 | [Name] | | |
| Risk Manager 风险经理 | [Name] | | |

---

## Appendix A: File Locations / 附录A：文件位置

| Report 报告 | Location 位置 |
|------------|---------------|
| Task Tracking | `tasks.md` |
| Research Report | `research/research-report.md` |
| Development Report | `development/dev-report.md` |
| Gate Report | `validation/gate-report.md` |
| Independent Implementation | `validation/independent-impl/` |
| Performance Report | `reviews/performance-report.md` |
| Security Report | `reviews/security-report.md` |
| Code Quality Report | `reviews/code-quality-report.md` |
| MC Comparison Report | `cross-validation/mc-comparison-report.md` |

---

## Appendix B: Test Commands / 附录B：测试命令

```bash
# Run unit tests
python -m pytest test/test_<engine_name>.py -v

# Run boundary checks
python asset/<type>/engine/validation/script/boundary_check_<engine>.py

# Run MC comparison
python asset/<type>/engine/validation/script/benchmark_check_<engine>.py
```

---

## Appendix C: Change History / 附录C：变更历史

| Version 版本 | Date 日期 | Author 作者 | Changes 变更 |
|-------------|----------|-------------|--------------|
| 1.0 | YYYY-MM-DD | Codex | Initial validation |

---

*Generated by Model Orchestrator Skill*
*Based on Federal Reserve SR 11-7 Model Risk Management Guidelines*

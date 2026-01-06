---
name: Update CLAUDE.md
description: Review and update CLAUDE.md developer guide files to ensure accuracy and conciseness.
category: Documentation
tags: [claude, docs, update]
---

**Input Handling**

Parse the argument provided after `/update-claudemd`:
- If argument is "all", "ALL", or "All": Update ALL CLAUDE.md files listed below
- If argument is a valid path (e.g., "asset/equity/CLAUDE.md" or "var"): Update that specific file
- If no argument provided: Use the AskUserQuestion tool to ask which file(s) to update

**Available CLAUDE.md Files**:
| Path | Module |
|------|--------|
| `CLAUDE.md` | Root project guide |
| `asset/equity/CLAUDE.md` | Equity derivatives module |
| `var/CLAUDE.md` | Value-at-Risk module |
| `backtest/CLAUDE.md` | Backtesting framework |
| `stresstest/CLAUDE.md` | Stress testing framework |
| `dynamicscenario/CLAUDE.md` | Dynamic scenarios |
| `simm/CLAUDE.md` | SIMM margin calculation |

**Update Process**

For each CLAUDE.md file to update, follow these steps:

1. **Read the current CLAUDE.md file** to understand its structure and content

2. **Explore the module directory** to verify accuracy:
   - List all Python files in the module
   - Check that documented classes/functions still exist
   - Identify any new public APIs not yet documented
   - Verify architecture diagrams match actual file structure

3. **Update the file** with these goals:
   - **Accuracy**: Remove references to deleted features; add new features
   - **Conciseness**: Remove redundant sections, overly verbose explanations
   - **Structure**: Follow existing section pattern (Overview, Architecture, Usage, etc.)
   - **Examples**: Ensure code examples are correct and runnable
   - **Length**: Aim for under 500 lines; if longer, ensure every section is essential

4. **Report changes** made to each file

**Quality Checklist**
Before finishing each file update, verify:
- [ ] All code examples match current API signatures
- [ ] No references to non-existent files, classes, or methods
- [ ] Module structure diagram matches actual directory layout
- [ ] No duplicate content between sections
- [ ] Descriptions are concise without losing essential information
- [ ] File is well-organized and easy to navigate

**Guardrails**
- Do NOT add new sections unless they provide clear value
- Do NOT expand content unnecessarily - favor conciseness
- Do NOT include implementation details that change frequently
- Focus on stable public APIs and usage patterns

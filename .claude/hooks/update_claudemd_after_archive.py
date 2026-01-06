#!/usr/bin/env python3
"""
PostToolUse hook for updating CLAUDE.md files after openspec:archive.

This hook:
1. Filters for only the openspec:archive skill
2. Extracts the archived change ID
3. Identifies affected CLAUDE.md files via:
   - Delta specs directory names -> CLAUDE.md mapping
   - tasks.md references to CLAUDE.md
4. Returns JSON instructing Claude to update the CLAUDE.md files
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Mapping from delta spec capability names to CLAUDE.md files
# Format: { "capability-name": "relative/path/to/CLAUDE.md" }
CAPABILITY_TO_CLAUDEMD = {
    # Equity module capabilities
    "equity-pde-engine": "asset/equity/CLAUDE.md",
    "equity-analytical-engine": "asset/equity/CLAUDE.md",
    "equity-mc-pricing": "asset/equity/CLAUDE.md",
    "equity-barrier-products": "asset/equity/CLAUDE.md",
    "equity-digital-products": "asset/equity/CLAUDE.md",
    "equity-greeks": "asset/equity/CLAUDE.md",
    "asian-option": "asset/equity/CLAUDE.md",
    "asian-option-analytical-engine": "asset/equity/CLAUDE.md",
    "snowball-option-helpers": "asset/equity/CLAUDE.md",
    "snowball-pde-engine": "asset/equity/CLAUDE.md",
    "base-engine": "asset/equity/CLAUDE.md",
    "engine-enums": "asset/equity/CLAUDE.md",
    "greeks-calculator": "asset/equity/CLAUDE.md",

    # Convertible bond capabilities (part of equity for now)
    "convertible-bond-product": "asset/equity/CLAUDE.md",
    "convertible-bond-pde-engine": "asset/equity/CLAUDE.md",
    "convertible-bond-tree-engine": "asset/equity/CLAUDE.md",
    "convertible-bond-facade-engine": "asset/equity/CLAUDE.md",

    # VaR module capabilities
    "portfolio-var": "var/CLAUDE.md",

    # Backtest module capabilities
    "backtest-protocols": "backtest/CLAUDE.md",
    "fi-backtest": "backtest/CLAUDE.md",
    "fi-portfolio": "backtest/CLAUDE.md",

    # Dynamic scenario capabilities
    "dynamicscenario-protocols": "dynamicscenario/CLAUDE.md",
    "fi-dynamicscenario": "dynamicscenario/CLAUDE.md",

    # Stress test capabilities
    "stresstest-protocols": "stresstest/CLAUDE.md",
    "fi-stresstest": "stresstest/CLAUDE.md",

    # SIMM capabilities
    "simm-crif-format": "simm/CLAUDE.md",
    "simm-risk-taxonomy": "simm/CLAUDE.md",
    "simm-calibration-data": "simm/CLAUDE.md",
    "simm-margin-calculator": "simm/CLAUDE.md",
    "simm-attribution": "simm/CLAUDE.md",
    "simm-results": "simm/CLAUDE.md",
}


def get_project_dir() -> Path:
    """Get project directory from environment or cwd."""
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))


def find_archived_change(project_dir: Path, change_id: str) -> Optional[Path]:
    """Find the archived change directory by change_id."""
    archive_dir = project_dir / "openspec" / "changes" / "archive"
    if not archive_dir.exists():
        return None

    # Look for directories ending with the change_id
    for entry in archive_dir.iterdir():
        if entry.is_dir() and entry.name.endswith(change_id):
            return entry

    return None


def extract_change_id_from_response(tool_response: dict) -> Optional[str]:
    """Extract the change ID from the archive command output."""
    response_text = str(tool_response)

    # Look for patterns like "archived add-pde-greeks-mode" or similar
    patterns = [
        r"archived\s+(\S+)",
        r"archive.*?([a-z]+-[a-z0-9-]+)",
        r"changes/archive/\d{4}-\d{2}-\d{2}-([a-z0-9-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def get_delta_specs(change_dir: Path) -> list:
    """Get list of delta spec capability names from the change directory."""
    specs_dir = change_dir / "specs"
    if not specs_dir.exists():
        return []

    return [entry.name for entry in specs_dir.iterdir() if entry.is_dir()]


def parse_tasks_for_claudemd(change_dir: Path) -> list:
    """Parse tasks.md for explicit CLAUDE.md references."""
    tasks_file = change_dir / "tasks.md"
    if not tasks_file.exists():
        return []

    content = tasks_file.read_text()

    # Look for references like "CLAUDE.md", "asset/equity/CLAUDE.md", etc.
    pattern = r"([a-zA-Z0-9_/]*CLAUDE\.md)"
    matches = re.findall(pattern, content)

    return list(set(matches))


def map_capabilities_to_claudemd(capabilities: list) -> set:
    """Map capability names to CLAUDE.md file paths."""
    claudemd_files = set()

    for cap in capabilities:
        if cap in CAPABILITY_TO_CLAUDEMD:
            claudemd_files.add(CAPABILITY_TO_CLAUDEMD[cap])
        else:
            # Try to infer from capability name prefix
            if cap.startswith("equity-") or cap.startswith("asian-") or cap.startswith("snowball-"):
                claudemd_files.add("asset/equity/CLAUDE.md")
            elif cap.startswith("var-") or cap.startswith("portfolio-var"):
                claudemd_files.add("var/CLAUDE.md")
            elif cap.startswith("backtest-") or cap.startswith("fi-backtest"):
                claudemd_files.add("backtest/CLAUDE.md")
            elif cap.startswith("simm-"):
                claudemd_files.add("simm/CLAUDE.md")
            elif cap.startswith("stresstest-") or cap.startswith("fi-stresstest"):
                claudemd_files.add("stresstest/CLAUDE.md")
            elif cap.startswith("dynamicscenario-") or cap.startswith("fi-dynamicscenario"):
                claudemd_files.add("dynamicscenario/CLAUDE.md")
            elif cap.startswith("convertible-"):
                claudemd_files.add("asset/equity/CLAUDE.md")

    return claudemd_files


def read_proposal_summary(change_dir: Path) -> str:
    """Read the proposal.md and extract a summary."""
    proposal_file = change_dir / "proposal.md"
    if not proposal_file.exists():
        return ""

    content = proposal_file.read_text()

    # Extract the first section (usually the title and Why section)
    lines = content.split('\n')
    summary_lines = []
    for line in lines[:30]:  # First 30 lines should contain key info
        summary_lines.append(line)

    return '\n'.join(summary_lines)


def main():
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    # Only process Skill tool calls
    tool_name = input_data.get("tool_name", "")
    if tool_name != "Skill":
        sys.exit(0)

    # Only process openspec:archive skill
    tool_input = input_data.get("tool_input", {})
    skill_name = tool_input.get("skill", "")

    if skill_name not in ["openspec:archive", "archive"]:
        sys.exit(0)

    # Get the tool response to extract the change ID
    tool_response = input_data.get("tool_response", {})

    # Try to extract the change ID from the response
    change_id = extract_change_id_from_response(tool_response)

    # Also check if args contains the change ID
    args = tool_input.get("args", "")
    if not change_id and args:
        # Args might contain the change ID directly
        change_id = args.strip().split()[0] if args.strip() else None

    if not change_id:
        # Cannot determine change ID, exit silently
        sys.exit(0)

    project_dir = get_project_dir()

    # Find the archived change directory
    change_dir = find_archived_change(project_dir, change_id)
    if not change_dir:
        # Change directory not found, exit silently
        sys.exit(0)

    # Get delta specs and map to CLAUDE.md files
    delta_specs = get_delta_specs(change_dir)
    claudemd_files = map_capabilities_to_claudemd(delta_specs)

    # Also parse tasks.md for explicit references
    tasks_refs = parse_tasks_for_claudemd(change_dir)
    for ref in tasks_refs:
        if ref and not ref.startswith('/'):
            claudemd_files.add(ref)

    if not claudemd_files:
        # No CLAUDE.md files to update
        sys.exit(0)

    # Read proposal summary for context
    proposal_summary = read_proposal_summary(change_dir)

    # Build the instruction message for Claude
    claudemd_list = '\n'.join(f"- {f}" for f in sorted(claudemd_files))
    delta_spec_list = '\n'.join(f"- {s}" for s in sorted(delta_specs))

    instruction = f"""The OpenSpec change '{change_id}' has been archived successfully.

**Archived Change Location**: {change_dir}

**Delta Specs (capabilities affected)**:
{delta_spec_list}

**CLAUDE.md Files That May Need Updates**:
{claudemd_list}

**Proposal Summary**:
{proposal_summary[:1500]}

---

**ACTION REQUIRED**: Please review and update the CLAUDE.md file(s) listed above to reflect the new capabilities added by this change. For each CLAUDE.md file:

1. Read the current CLAUDE.md file
2. Read the archived change's proposal.md and specs/*.md files for context
3. Add or update sections to document the new features/changes
4. Focus on:
   - New classes, methods, or APIs introduced
   - Usage examples showing how to use the new functionality
   - Any breaking changes or migration notes
   - Updated module structure if files were added

Please proceed with the CLAUDE.md updates now."""

    # Return JSON output with decision to block and provide feedback
    output = {
        "decision": "block",
        "reason": instruction,
    }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()

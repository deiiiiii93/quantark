"""Item 6: rewrite every raw reader of is_annualized_rebate / is_annualized_ki to the effective flag, IN THE WORKING TREE.

This is a measurement harness, not a change: ``apply`` rewrites the readers so a test run shows what would move if
``None`` followed ``is_annualized`` (the way knock-out accrual does through ``_effective_annualized_flag``);
``revert`` restores the files with ``git checkout --``. Never commit the applied state.

The construction-time validation (snowball_option.py ``_validate_maturity_parameters``: "initial_date or
settlement_date required for annualized accruals when tenor is not provided") is LEFT RAW, so the run measures the
cash consequence alone. The findings document counts separately what that validation would refuse.

Usage from the repo root:  <venv>/bin/python <this file> apply|revert|status
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OPT = "quantark/asset/equity/product/option/"
EFF_SELF = "self._effective_annualized_flag(self.accrual_config.{f})"
EFF_PROD = "product._effective_annualized_flag(product.accrual_config.{f})"

# (file, old, new, expected count)
EDITS = [
    (OPT + "snowball_option.py", "if self.accrual_config.is_annualized_rebate:",
     "if " + EFF_SELF.format(f="is_annualized_rebate") + ":", 2),
    (OPT + "snowball_option.py", "if self.accrual_config.is_annualized_ki:",
     "if " + EFF_SELF.format(f="is_annualized_ki") + ":", 1),
    (OPT + "ko_reset_snowball_option.py", "if self.accrual_config.is_annualized_rebate:",
     "if " + EFF_SELF.format(f="is_annualized_rebate") + ":", 2),
    (OPT + "ko_reset_snowball_option.py", "if self.accrual_config.is_annualized_ki:",
     "if " + EFF_SELF.format(f="is_annualized_ki") + ":", 1),
    (OPT + "phoenix_option.py", "if self.accrual_config.is_annualized_rebate:",
     "if " + EFF_SELF.format(f="is_annualized_rebate") + ":", 2),
    (OPT + "accrual_build.py", "((v0, bool(ac.is_annualized_rebate)), (v1, bool(ac.is_annualized_ki)))",
     "((v0, bool(product._effective_annualized_flag(ac.is_annualized_rebate))), "
     "(v1, bool(product._effective_annualized_flag(ac.is_annualized_ki))))", 1),
    ("quantark/asset/equity/engine/quad/v2/adapters.py", "and product.accrual_config.is_annualized_ki",
     "and " + EFF_PROD.format(f="is_annualized_ki"), 1),
    ("quantark/asset/equity/engine/mc/snowball_mc_engine.py",
     "post_mode == PostKOScheduleMode.REBASED and product.accrual_config.is_annualized_ki",
     "post_mode == PostKOScheduleMode.REBASED and " + EFF_PROD.format(f="is_annualized_ki"), 1),
    ("quantark/asset/equity/engine/pde/snowball_pde_solver.py",
     "if product.accrual_config.is_annualized_rebate and not call_rebate:",
     "if " + EFF_PROD.format(f="is_annualized_rebate") + " and not call_rebate:", 1),
    ("quantark/asset/equity/engine/pde/snowball_pde_solver.py",
     "if product.accrual_config.is_annualized_rebate\n",
     "if " + EFF_PROD.format(f="is_annualized_rebate") + "\n", 1),
]


def apply():
    for path, old, new, n in EDITS:
        p = ROOT / path
        text = p.read_text()
        found = text.count(old)
        if found != n:
            sys.exit(f"{path}: expected {n} x {old!r}, found {found}; nothing more applied")
        p.write_text(text.replace(old, new))
        print(f"applied {n} x in {path}")


def revert():
    files = sorted({path for path, *_ in EDITS})
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--", *files], check=True)
    print("reverted", *files, sep="\n  ")


def status():
    files = sorted({path for path, *_ in EDITS})
    subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", "--", *files], check=True)


if __name__ == "__main__":
    {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()

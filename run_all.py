"""
Instacart Pipeline — Full Automation Script
Run this once and walk away.
It runs every pipeline block in order, prints output live,
and stops only if something actually breaks.
"""

import subprocess
import sys
import os
from datetime import datetime

PYTHON = "D:/Users/Sakthi/anaconda3/python.exe"
PROJECT = r"C:\Users\Hp\Desktop\instacart-analytics1"

PIPELINE = [
    ("BLOCK 2  — EDA",                  r"pipeline\02_eda.py"),
    ("BLOCK 3  — Cleaning",             r"pipeline\03_clean.py"),
    ("BLOCK 4  — Feature Engineering",  r"pipeline\04_feature_engineering.py"),
    ("BLOCK 5  — RFM",                  r"pipeline\04_rfm.py"),
    ("BLOCK 7  — Trajectory A/B/C/D",   r"pipeline\05b_trajectory.py"),
    ("BLOCK 9  — Churn Model",          r"pipeline\08_churn_model.py"),
    ("BLOCK 13 — Category Exposure",    r"pipeline\07_category_exposure.py"),
    ("BLOCK 14 — Executive Summary",    r"pipeline\09_executive_summary.py"),
]

LOG_FILE = os.path.join(PROJECT, "outputs", "reports", "run_log.txt")

def separator(title):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{line}")

def run_script(label, script_path):
    separator(label)
    full_path = os.path.join(PROJECT, script_path)

    if not os.path.exists(full_path):
        print(f"  SKIPPED — file not found: {full_path}")
        print(f"  Claude Code needs to write this file first.")
        return True  # skip missing files, don't stop

    print(f"  Running: {script_path}\n")

    result = subprocess.run(
        [PYTHON, full_path],
        cwd=PROJECT,
        capture_output=False,   # prints live to console
        text=True,
    )

    if result.returncode != 0:
        print(f"\n  ERROR in {label}")
        print(f"  Return code: {result.returncode}")
        print(f"  Fix the error above then rerun this script.")
        return False

    print(f"\n  {label} — DONE")
    return True


def main():
    print("\n" + "=" * 60)
    print("  INSTACART PIPELINE — FULL AUTOMATION")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {PROJECT}")
    print(f"  Python:  {PYTHON}")
    print("=" * 60)

    # confirm project folder exists
    if not os.path.exists(PROJECT):
        print(f"\nERROR: Project folder not found: {PROJECT}")
        print("Check the path and try again.")
        sys.exit(1)

    # confirm python exists
    if not os.path.exists(PYTHON):
        print(f"\nERROR: Python not found: {PYTHON}")
        print("Check your Anaconda path.")
        sys.exit(1)

    completed = []
    failed = None

    for label, script in PIPELINE:
        success = run_script(label, script)
        if success:
            completed.append(label)
        else:
            failed = label
            break

    # final summary
    separator("RUN COMPLETE" if not failed else "RUN STOPPED")

    print(f"\n  Completed blocks ({len(completed)}):")
    for c in completed:
        print(f"    OK  {c}")

    if failed:
        print(f"\n  Stopped at: {failed}")
        print(f"  Fix the error above and rerun.")
    else:
        print(f"\n  ALL BLOCKS COMPLETE")
        print(f"  Check outputs/reports/ for all results")
        print(f"  Ready for GitHub and Streamlit")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

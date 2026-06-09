"""
Phase 3 Runner — Generates the SR 11-7 Validation Report.

Run: python run_phase3.py

Prerequisites: run_phase1.py and run_phase2.py must have been run first.
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "=" * 60)
print("  MODEL RISK DASHBOARD — Phase 3: SR 11-7 Report")
print("=" * 60 + "\n")

from report.generate_report import generate
report_path = generate()

print("\n" + "=" * 60)
print("  Phase 3 Complete.")
print(f"  Report → {report_path}")
print("  Open the HTML file in any browser to view the full report.")
print("=" * 60 + "\n")

# Auto-open in browser on Mac
try:
    subprocess.run(["open", report_path], check=True)
    print("  Opened report in default browser.")
except Exception:
    print("  Could not auto-open. Open manually:")
    print(f"  open \"{report_path}\"")
